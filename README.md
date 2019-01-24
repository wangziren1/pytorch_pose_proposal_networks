# pytorch_pose_proposal_network
Pytorch implementation of pose proposal network
## Introduction
We train the network on MPII human pose dataset but don't evaluate the performance. I don't think it can match the performance mentioned in paper due to some problems(see below).
## Requirements
1. pytorch 0.4.1
2. opencv
3. numpy
4. tensorboardX
## demo
[demo1](https://youtu.be/l_zgAg_loFk)
[demo2](https://youtu.be/FLmLAwOvqOA)
[demo3](https://youtu.be/IU2r3-T3LGs)
## speed
The speed depend on numbers of person. More persons there are, a little slower it runs.

 example       |resnet18                   |  resnet50
:---------:|:-------------------------:|:-------------------------:
A | <img src="https://github.com/wangziren1/pytorch_pose_proposal_network/blob/master/images/res18-1.png" width="200">  |  <img src="https://github.com/wangziren1/pytorch_pose_proposal_network/blob/master/images/res50-1.png" width="200"> |
B | <img src="https://github.com/wangziren1/pytorch_pose_proposal_network/blob/master/images/res18-3.png" width="200"> | <img src="https://github.com/wangziren1/pytorch_pose_proposal_network/blob/master/images/res50-3.png" width="200"> |

## problems
<p align="center">
<img src="https://github.com/wangziren1/pytorch_pose_proposal_network/blob/master/images/loss_iou.png" width="600">
</p>

The above figure is iou loss. you can see it that decrease not very better than other losses(under **images** folder). In fact, when I print it and ground truth, I find relative large error between them, especially when ground truth iou is low. For example while ground truth iou is 0.2, prediction iou is 0.5 or 0.6. So at parse stage, I just use resp instead of resp * iou.

Another problem is that I can't figure out one formula: S * E * D in the paper. I just use D * E * D. 

## train
First you need to put MPII dataset images in **data/images**. Then You can train from scratch or download our well trained weights.
* train from scratch: download pretrained weights [resnet18 weights](https://download.pytorch.org/models/resnet18-5c106cde.pth) and [resnet50 weights](https://download.pytorch.org/models/resnet50-19c8e357.pth), then put them in **src/model/pretrain_weight** directory.
* well trained weights: download [resnet18 weights](https://drive.google.com/open?id=1NUTmRxsWuEqB7uAkBK2JEP1ZdZE3FXsn) and
[resnet50 weights](https://drive.google.com/open?id=1IHBv8SCUrO0OJiaxMwrp8Ok1nKxVyzjG)
and put them in **src/checkpoint** directory.

## Reference
when I do this project, I mainly refer to this repo: https://github.com/hizhangp/yolo_tensorflow.
