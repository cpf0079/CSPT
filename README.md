# CSPT
Official source codes for our TIP 2022 paper "[Contrastive Self-Supervised Pre-Training for Video Quality Assessment](https://ieeexplore.ieee.org/abstract/document/9640574)".

![image](https://github.com/cpf0079/CSPT/blob/main/framework.png)

## Self-supervised training
**Step 1.** Prepare the training/validation data and txt files.

* each item in .txt files should be like "vid.mp4" (vid denotes the video name)

**Step 2.** Uncertainty-based ranking to split target domain into subdomains by running:
```
$ python ./source/main.py
```

## CSPT-pretrained weights
We only provide pre-trained model with resnet-50 backbone here. You can use this model to finetune on your own data.

Download link: [CSPT-resnet50](https://pan.baidu.com/s/19LDkykn2rE4xivtoNwvyiA)

Password: 6xwg

## Environment
* Python 3.9.7
* Pytorch 1.11.0 Torchvision 0.12.0
* Cuda 11.3 Cudnn 8.2.1 
* cv2

## Citation
If you find this work useful for your research, please cite our paper:
```
@article{chen2021contrastive,
  title={Contrastive Self-Supervised Pre-Training for Video Quality Assessment},
  author={Chen, Pengfei and Li, Leida and Wu, Jinjian and Dong, Weisheng and Shi, Guangming},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={458--471},
  year={2021},
  publisher={IEEE}
}
```
