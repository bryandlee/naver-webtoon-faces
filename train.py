import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from model import Encoder, Generator, Discriminator, CooccurDiscriminator
from stylegan2.dataset import MultiResolutionDataset
from stylegan2.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    (grad_real,) = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def patchify_image(img, n_crop, min_size=1 / 8, max_size=1 / 4):
    crop_size = torch.rand(n_crop) * (max_size - min_size) + min_size
    batch, channel, height, width = img.shape
    target_h = int(height * max_size)
    target_w = int(width * max_size)
    crop_h = (crop_size * height).type(torch.int64).tolist()
    crop_w = (crop_size * width).type(torch.int64).tolist()

    patches = []
    for c_h, c_w in zip(crop_h, crop_w):
        c_y = random.randrange(0, height - c_h)
        c_x = random.randrange(0, width - c_w)

        cropped = img[:, :, c_y : c_y + c_h, c_x : c_x + c_w]
        cropped = F.interpolate(
            cropped, size=(target_h, target_w), mode="bilinear", align_corners=False
        )

        patches.append(cropped)

    patches = torch.stack(patches, 1).view(-1, channel, target_h, target_w)

    return patches


def train(
    args,
    loader,
    encoder,
    generator,
    discriminator,
    cooccur,
    g_optim,
    d_optim,
    e_ema,
    g_ema,
    device,
    writer,
):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    loss_dict = {}

    if args.distributed:
        e_module = encoder.module
        g_module = generator.module
        d_module = discriminator.module
        c_module = cooccur.module

    else:
        e_module = encoder
        g_module = generator
        d_module = discriminator
        c_module = cooccur

    accum = 0.5 ** (32 / (10 * 1000))

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(encoder, False)
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        requires_grad(cooccur, True)

        real_img1, real_img2 = real_img.chunk(2, dim=0)

        structure1, texture1 = encoder(real_img1)
        _, texture2 = encoder(real_img2)

        fake_img1 = generator(structure1, texture1)
        fake_img2 = generator(structure1, texture2)

        fake_pred = discriminator(torch.cat((fake_img1, fake_img2), 0))
        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        fake_patch = patchify_image(fake_img2, args.n_crop)
        real_patch = patchify_image(real_img2, args.n_crop)
        ref_patch = patchify_image(real_img2, args.ref_crop * args.n_crop)
        fake_patch_pred, ref_input = cooccur(
            fake_patch, ref_patch, ref_batch=args.ref_crop
        )
        real_patch_pred, _ = cooccur(real_patch, ref_input=ref_input)
        cooccur_loss = d_logistic_loss(real_patch_pred, fake_patch_pred)

        loss_dict["d"] = d_loss
        loss_dict["cooccur"] = cooccur_loss
        loss_dict["real_score"] = real_pred.mean()
        fake_pred1, fake_pred2 = fake_pred.chunk(2, dim=0)
        loss_dict["fake_score"] = fake_pred1.mean()
        loss_dict["hybrid_score"] = fake_pred2.mean()

        d_optim.zero_grad()
        (d_loss + cooccur_loss).backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            real_patch.requires_grad = True
            real_patch_pred, _ = cooccur(real_patch, ref_patch, ref_batch=args.ref_crop)
            cooccur_r1_loss = d_r1_loss(real_patch_pred, real_patch)

            d_optim.zero_grad()

            r1_loss_sum = args.r1 / 2 * r1_loss * args.d_reg_every
            r1_loss_sum += args.cooccur_r1 / 2 * cooccur_r1_loss * args.d_reg_every
            r1_loss_sum += 0 * real_pred[0, 0] + 0 * real_patch_pred[0, 0]
            r1_loss_sum.backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss
        loss_dict["cooccur_r1"] = cooccur_r1_loss

        requires_grad(encoder, True)
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(cooccur, False)
        
        real_img.requires_grad = False

        structure1, texture1 = encoder(real_img1)
        _, texture2 = encoder(real_img2)

        fake_img1 = generator(structure1, texture1)
        fake_img2 = generator(structure1, texture2)

        recon_loss = F.l1_loss(fake_img1, real_img1)

        fake_pred = discriminator(torch.cat((fake_img1, fake_img2), 0))
        g_loss = g_nonsaturating_loss(fake_pred)

        fake_patch = patchify_image(fake_img2, args.n_crop)
        ref_patch = patchify_image(real_img2, args.ref_crop * args.n_crop)
        fake_patch_pred, _ = cooccur(fake_patch, ref_patch, ref_batch=args.ref_crop)
        g_cooccur_loss = g_nonsaturating_loss(fake_patch_pred)

        loss_dict["recon"] = recon_loss
        loss_dict["g"] = g_loss
        loss_dict["g_cooccur"] = g_cooccur_loss

        g_optim.zero_grad()
        (recon_loss + g_loss + g_cooccur_loss).backward()
        g_optim.step()

        accumulate(e_ema, e_module, accum)
        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        cooccur_val = loss_reduced["cooccur"].mean().item()
        recon_val = loss_reduced["recon"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        g_cooccur_val = loss_reduced["g_cooccur"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        cooccur_r1_val = loss_reduced["cooccur_r1"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        hybrid_score_val = loss_reduced["hybrid_score"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; c: {cooccur_val:.4f} g: {g_loss_val:.4f}; "
                    f"g_cooccur: {g_cooccur_val:.4f}; recon: {recon_val:.4f}; r1: {r1_val:.4f}; "
                    f"r1_cooccur: {cooccur_r1_val:.4f}"
                )
            )

            if writer is not None and i % 10 == 0:
                writer.add_scalar("Loss/Generator", g_loss_val, i)
                writer.add_scalar("Loss/Discriminator", d_loss_val, i)
                writer.add_scalar("Loss/Cooccur", cooccur_val, i)
                writer.add_scalar("Loss/Recon", recon_val, i)
                writer.add_scalar("Loss/Generator Cooccur", g_cooccur_val, i)
                writer.add_scalar("Loss/R1", r1_val, i)
                writer.add_scalar("Loss/Cooccur R1", cooccur_r1_val, i)
                writer.add_scalar("Loss/Real Score", real_score_val, i)
                writer.add_scalar("Loss/Fake Score", fake_score_val, i)
                writer.add_scalar("Loss/Hybrid Score", hybrid_score_val, i)

            if i % 100 == 0:
                with torch.no_grad():
                    e_ema.eval()
                    g_ema.eval()

                    structure1, texture1 = e_ema(real_img1)
                    _, texture2 = e_ema(real_img2)

                    fake_img1 = g_ema(structure1, texture1)
                    fake_img2 = g_ema(structure1, texture2)

                    sample = torch.cat((fake_img1, fake_img2), 0)

                    utils.save_image(
                        sample,
                        f"sample/{str(i).zfill(6)}.png",
                        nrow=int(sample.shape[0] ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 10000 == 0:
                torch.save(
                    {
                        "e": e_module.state_dict(),
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "cooccur": c_module.state_dict(),
                        "e_ema": e_ema.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                    },
                    f"checkpoint/{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=str, nargs="+")
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--cooccur_r1", type=float, default=1)
    parser.add_argument("--ref_crop", type=int, default=4)
    parser.add_argument("--n_crop", type=int, default=8)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel", type=int, default=32)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    encoder = Encoder(args.channel).to(device)
    generator = Generator(args.channel).to(device)

    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    cooccur = CooccurDiscriminator(args.channel).to(device)

    e_ema = Encoder(args.channel).to(device)
    g_ema = Generator(args.channel).to(device)
    e_ema.eval()
    g_ema.eval()
    accumulate(e_ema, encoder, 0)
    accumulate(g_ema, generator, 0)

    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        list(encoder.parameters()) + list(generator.parameters()),
        lr=args.lr,
        betas=(0, 0.99),
    )
    d_optim = optim.Adam(
        list(discriminator.parameters()) + list(cooccur.parameters()),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        encoder.load_state_dict(ckpt["e"])
        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        cooccur.load_state_dict(ckpt["cooccur"])
        e_ema.load_state_dict(ckpt["e_ema"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        encoder = nn.parallel.DistributedDataParallel(
            encoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        cooccur = nn.parallel.DistributedDataParallel(
            cooccur,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    datasets = []

    for path in args.path:
        dataset = MultiResolutionDataset(path, transform, args.size)
        datasets.append(dataset)

    loader = data.DataLoader(
        data.ConcatDataset(datasets),
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and args.tensorboard:
        writer = SummaryWriter("log")
    else:
        writer = None

    train(
        args,
        loader,
        encoder,
        generator,
        discriminator,
        cooccur,
        g_optim,
        d_optim,
        e_ema,
        g_ema,
        device,
        writer,
    )

