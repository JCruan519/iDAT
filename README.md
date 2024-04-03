# iDAT: inverse Distillation Adapter-Tuning

This is the offical repository of iDAT: inverse Distillation Adapter-Tuning, which is accepted by ICME 2024. {[Arxiv Paper](https://arxiv.org/abs/2403.15750)}

## Abstract

Adapter-Tuning (AT) method involves freezing a pre-trained model and introducing trainable adapter modules to acquire downstream knowledge, thereby calibrating the model for better adaptation to downstream tasks. This paper proposes a distillation framework for the AT method instead of crafting a carefully designed adapter module, which aims to improve fine-tuning performance. For the first time, we explore the possibility of combining the AT method with knowledge distillation. Via statistical analysis, we observe significant differences in the knowledge acquisition between adapter modules of different models. Leveraging these differences, we propose a simple yet effective framework called inverse Distillation Adapter-Tuning (iDAT). Specifically, we designate the smaller model as the teacher and the larger model as the student. The two are jointly trained, and online knowledge distillation is applied to inject knowledge of different perspective to student model, and significantly enhance the fine-tuning performance on downstream tasks. Extensive experiments on the VTAB-1K benchmark with 19 image classification tasks demonstrate the effectiveness of iDAT. The results show that using existing AT method within our iDAT framework can further yield a 2.66\% performance gain, with only an additional 0.07M trainable parameters. Our approach compares favorably with state-of-the-arts without bells and whistles.


## 0. Main Environments

- Create a conda virtual environment and activate it:

```bash
conda create -n iDAT python=3.8 -y
conda activate iDAT
```

- Install requirements:

```bash
pip install -r requirements.txt
```


## 1. Prepare the dataset

**VTAB-1K**: You can follow [SSF](https://github.com/dongzelian/SSF?tab=readme-ov-file#data-preparation) to download them, or download directly through this link ([Baidu Netdisk](https://pan.baidu.com/s/1C5HfQa45ttgk3nBVcG7YsA?pwd=4r6f)).

## 2. Prepare the pre_trained weights

For pre-trained ViT models on ImageNet-21K, the weights will be automatically downloaded. You can also manually download them from [ViT](https://github.com/google-research/vision_transformer).



## 3. Fine-tuning within our iDAT framework

To fine-tune a pre-trained ViT model via `Adapter` within our iDAT framework on VTAB-1K, run:

```bash
bash train_scripts/vit/train_vtab_onlinekd.sh
```



### 4. Acknowledgement
Thanks for the open-source code from [SSF](https://github.com/dongzelian/SSF).
