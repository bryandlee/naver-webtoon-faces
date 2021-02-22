1. prepare 256 sized face images
2. clone liteweight model
  > git clone https://github.com/bryandlee/animegan2-pytorch.git animegan2
3. Initial training 
  > python train_lite.py --iter 50000
4. Finetuning with GAN loss
  > python train_lite.py --iter 2000 --ckpt result_lite/checkpoint/050000.pt --use_D
