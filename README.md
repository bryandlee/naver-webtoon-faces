# **Naver Webtoon Faces**

![./imgs/face2webtoon/dh.gif](./imgs/face2webtoon/dh.gif)\

Face to cartoon conversion code is added.

![./imgs/face2webtoon/grb.jpg](./imgs/face2webtoon/grb.jpg)\
![./imgs/face2webtoon/je.jpg](./imgs/face2webtoon/je.jpg)\
![./imgs/face2webtoon/jyp.jpg](./imgs/face2webtoon/jyp.jpg)

This repo heavily uses codes from [[this repo]](https://github.com/rosinality/swapping-autoencoder-pytorch) 

## Dataset


256*256 cartoon face images collected from on-going [NAVER Webtoon](https://comic.naver.com/index.nhn) series.

```
num titles 58
total images: 17662
```

![./imgs/dataset.png](./imgs/datasets.png)


## StyleGAN2


[[paper]](https://arxiv.org/abs/1912.04958) [[model]](https://drive.google.com/file/d/14JLmYRtUFDbTC4IVzt14Os-tf23BXMLd/view?usp=sharing)

Training detail: batch 12, transfer from FFHQ, non-leaking augmentation

### FID (10k samples)

| Iteration            | FID               | 
| :------------------- | :----------------:|
| FFHQ                 | 256.54            |
| 50k                  | 11.73             |
| 100k                 | 9.29              |
| 150k                 | 8.87              |
| 200k                 | 8.11              |
| 250k                 | 7.41              |


### Samples (FID 8.87)

![./imgs/stylegan2/samples.png](./imgs/stylegan2/samples.png)


### 4-Way Linear interpolation in w-space

![./imgs/stylegan2/1.png](./imgs/stylegan2/1.png)

![./imgs/stylegan2/2.png](./imgs/stylegan2/2.png)


### Swapping codes at different layers

![./imgs/stylegan2/3.png](./imgs/stylegan2/3.png)



## Swapping Autoencoder for Deep Image Manipulation


[[paper]](https://arxiv.org/abs/2007.00653) [model]

SwapAE is a fully unsupervised generative model that embeds images into structure and style codes (similar to MUNIT). In SwapAE, the style encoder is forced to extract the global texture of the image by explicitly matching the patch statistics of the original image and swap-generated image (patch co-occurrence loss).

Training detail: batch 10, iteration 500k, single co-occurrence patch per sample

### Samples

![./imgs/swapae/1.png](./imgs/swapae/1.png)


### Style code interpolation

![./imgs/swapae/2.png](./imgs/swapae/2.png)

![./imgs/swapae/3.png](./imgs/swapae/3.png)

![./imgs/swapae/4.png](./imgs/swapae/4.png)


### Structure code interpolation

![./imgs/swapae/5.png](./imgs/swapae/5.png)

![./imgs/swapae/6.png](./imgs/swapae/6.png)

![./imgs/swapae/7.png](./imgs/swapae/7.png)

Simply interpolating the structure code didn't work well. The regional structure code editing method introduced in the original paper might work.


### Swapping codes at different layers

![./imgs/swapae/8.png](./imgs/swapae/8.png)

Injecting target style codes from the very first layer of the decoder often changes the whole identity of the original character. Detailed structures can be better preserved by applying the source style in the first few layers.

swap location = 2

![./imgs/swapae/9.png](./imgs/swapae/9.png)

![./imgs/swapae/10.png](./imgs/swapae/10.png)

![./imgs/swapae/11.png](./imgs/swapae/11.png)

swap location = 3

![./imgs/swapae/12.png](./imgs/swapae/12.png)

![./imgs/swapae/13.png](./imgs/swapae/13.png)

![./imgs/swapae/14.png](./imgs/swapae/14.png)

For the webcomics data domain, injecting target style codes from the 2nd~3rd layer gives pleasing style-transfer results.

swap location = 7

![./imgs/swapae/15.png](./imgs/swapae/15.png)

![./imgs/swapae/16.png](./imgs/swapae/16.png)

![./imgs/swapae/17.png](./imgs/swapae/17.png)

The last few layers of the generator control the overall color.


### Test image
From [**Slam Dunk**](https://en.wikipedia.org/wiki/Slam_Dunk_(manga))

![./imgs/swapae/18.png](./imgs/swapae/18.png)



## Face2Webtoon


### Finetuning SwapAE
Finetuning the webtoon-trained SwapAE model with real human faces for small amount of steps improves the robustness. Using the StyleGAN as a style image generator, exploration of styles without real webtoon data is also possible. See `train_aux.py` and `face2webtoon.ipynb`.

![./imgs/face2webtoon/dh.jpg](./imgs/face2webtoon/dh.jpg)



### Facial cartoonizer + Style Swapping
Without finetuning, EdgeExtraction [[code]](https://github.com/xavysp/DexiNed/tree/master/DexiNed-Pytorch)  + FacialCartoonization [[code]](https://github.com/SystemErrorWang/FacialCartoonization) → SwapAE also produces nice results.

![./imgs/swapae/19.png](./imgs/swapae/19.png)

![./imgs/swapae/20.png](./imgs/swapae/20.png)

![./imgs/swapae/21.png](./imgs/swapae/21.png)

Failures

![./imgs/swapae/22.png](./imgs/swapae/22.png)


### JTBC Samples

![./imgs/swapae/jtbc.png](./imgs/swapae/jtbc.png)


