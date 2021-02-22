import argparse
import torch
from torch import nn, autograd, optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

import torchvision
from torchvision import datasets, transforms

from model import Encoder, Generator, Discriminator, CooccurDiscriminator
from animegan2.model import Generator as Generator_lite


import os
import cv2


def tensor2image(tensor):
    tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
    return tensor*0.5 + 0.5


def vertical_concat(imgs):
    return torch.cat([img.unsqueeze(0) for img in imgs], 2) 


def requires_grad(model, flag=True, target_layer=None):
    for name, param in model.named_parameters():
        if target_layer is None:  # every layer
            param.requires_grad = flag
        elif target_layer in name:  # target layer
            param.requires_grad = flag
        

class VGGLoss(nn.Module):
    def __init__(self, n_layers=5):
        super().__init__()
        
        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)  

        vgg = torchvision.models.vgg19(pretrained=True).features
        
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers)
            prev_layer = next_layer
        
        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()
        
    def to(self, device):
        self.layers.to(device)
        self.criterion.to(device)
        return self
        
    def forward(self, source, target):
        loss = 0 
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight*self.criterion(source, target)
            
        return loss 


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()
       
def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def train_lite(args):
    
    device = args.device

    # original model & optimizer
    encoder = Encoder(32).to(device)
    generator = Generator(32).to(device)

    ckpt = torch.load(args.swap_ae_model, map_location=device)
    encoder.load_state_dict(ckpt["e_ema"])
    generator.load_state_dict(ckpt["g_ema"])
    print(f"swap ae model loaded: {args.swap_ae_model}")
    
    encoder.eval()
    generator.eval()

    requires_grad(encoder, False)
    requires_grad(generator, False)

    if args.use_D:
        discriminator = Discriminator(256, 1).to(device)
        cooccur = CooccurDiscriminator(32).to(device)
        discriminator.load_state_dict(ckpt["d"])
        requires_grad(discriminator, False)


        d_optim = optim.Adam(
            list(discriminator.parameters()) + list(cooccur.parameters()),
            lr=args.lr,
            betas=(0, 0.99),
        )
        d_optim.load_state_dict(ckpt["d_optim"])
    
    mean_tx = torch.load(args.mean_texture).to(device).repeat(args.batch, 1)
    
    
    # lite model & optimizer
    g_lite = Generator_lite().to(device)
    if args.ckpt is not None:
        g_lite.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"checkpoint loaded: {args.ckpt}")
    
    optimizer = optim.Adam(
        g_lite.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )
    
    l2_loss_fn = nn.MSELoss().to(device)
    vgg_loss_fn = VGGLoss().to(device)


    # data
    transform = transforms.Compose([
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(root=args.data, transform=transform)
    print(f"num train images: {len(dataset)}")
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch,
        num_workers=8,
        shuffle=True,
        drop_last=True,
    )
    
    os.makedirs(f"{args.result_path}/checkpoint", exist_ok=True)
    os.makedirs(f"{args.result_path}/samples", exist_ok=True)
    writer = SummaryWriter(f"{args.result_path}/log")
        
    iters = 53000 
    while True:
        
        try:
            image, _ = next(iterator)
        except:
            iterator = iter(data_loader)
            image, _ = next(iterator)
    
        image = image.to(device)
        
        with torch.no_grad():
            st, tx = encoder(image)
            tx_clip = tx * 0.5 + mean_tx * 0.5

            textures = [tx for _ in range(args.swap_loc)] + [tx_clip for _ in range(len(generator.layers) - args.swap_loc)]
            target = generator(st, textures, noises=0).clone().detach()

        losses = {}
        g_lite.train()        

        # D
        if args.use_D:
            discriminator.train()
        
            if args.freeze_D:
                for loc in range(args.freeze_D_loc):
                        requires_grad(discriminator, True, target_layer=f'convs.{6 - loc}')
                    requires_grad(discriminator, True, target_layer=f'final_conv')
                    requires_grad(discriminator, True, target_layer=f'final_linear')
            else:
                requires_grad(discriminator, True)
            requires_grad(g_lite, False)

            fake = g_lite(image)

            fake_pred = discriminator(fake)
            real_pred = discriminator(target)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()
            
            losses["D"] = d_loss.item()
        
        # G
        fake = g_lite(image)
        
        loss_l2 = l2_loss_fn(fake, target)
        loss_vgg = vgg_loss_fn(fake, target) * 0.2
        
        loss = loss_l2 + loss_vgg

        losses["L2"] = loss_l2.item()
        losses["VGG"] = loss_vgg.item()
        
        if args.use_D:
            requires_grad(discriminator, False)            
            fake_pred = discriminator(fake)
            loss_adv = g_nonsaturating_loss(fake_pred) * 0.4
            loss += loss_adv
            losses["ADV"] = loss_adv.item()

        losses["Total"] = loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        iters += 1

        if iters == 1 or iters % 10 == 0:
            log_line = f"{str(iters).zfill(6)| }"
            for k, v in losses.items():
                log_line += f"loss {k}: {v:4.2f} "
                writer.add_scalar(f"Loss/{k}", v, iters)
        
        if iters== 1 or iters % 100 == 0:
            with torch.no_grad():
                g_lite.eval()
                fake = g_lite(image)
            save_images = torch.cat([vertical_concat(image), vertical_concat(target), vertical_concat(fake.detach())], 3)
            cv2.imwrite(f'{args.result_path}/samples/{str(iters).zfill(6)}.jpg', cv2.cvtColor(255*tensor2image(save_images), cv2.COLOR_BGR2RGB))
            
        if iters % 1000 == 0:
            save_model = {
                "iters": iters,
                "args": args,
                "state_dict": g_lite.state_dict(),
                "optim": optimizer.state_dict(),
            }
            torch.save(save_model, f"{args.result_path}/checkpoint/{str(iters).zfill(6)}.pt")
            
        if iters > args.iter:
            break
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--iter", type=int, default=200000)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--swap_ae_model", type=str, default="./checkpoint/002000.pt")
    parser.add_argument("--mean_texture", type=str, default="./mean_texture.pt")
    parser.add_argument("--swap_loc", type=int, default=5)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--use_D", type=bool, default=False)    
    parser.add_argument("--freeze_D", type=bool, default=False)
    parser.add_argument("--freeze_D_loc", type=int, default=3)
    parser.add_argument("--result_path", type=str, default="result_lite")

    args = parser.parse_args()
    
    train_lite(args)
    