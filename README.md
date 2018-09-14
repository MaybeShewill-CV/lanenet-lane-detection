# LaneNet-Lane-Detection
Use tensorflow to implement a Deep Neural Network for real time lane detection mainly based on the IEEE IV conference 
paper "Towards End-to-End Lane Detection: an Instance Segmentation Approach".You can refer to their paper for details 
https://arxiv.org/abs/1802.05591. This model consists of a encoder-decoder stage, binary semantic segmentation stage 
and instance semantic segmentation using discriminative loss function for real time lane detection task.

The main network architecture is as follows:

`Network Architecture`
![NetWork_Architecture](https://github.com/TJCVRS/lanenet-lane-detection/blob/master/data/source_image/network_architecture.png)

## Installation
This software has only been tested on ubuntu 16.04(x64), python3.5, cuda-8.0, cudnn-6.0 with a GTX-1070 GPU. 
To install this software you need tensorflow 1.3.0 and other version of tensorflow has not been tested but I think 
it will be able to work properly in tensorflow above version 1.0. Other required package you may install them by

```
pip3 install -r requirements.txt
```

## Test model
In this repo I uploaded a model trained on tusimple lane dataset [Tusimple_Lane_Detection](http://benchmark.tusimple.ai/#/).
The deep neural network inference part can achieve around a 50fps which is similar to the description in the paper. But
the input pipeline I implemented now need to be improved to achieve a real time lane detection system.

The trained lanenet model weights files are stored in 
[dropbox_lanenet_model_file](https://www.dropbox.com/sh/2ptlvq4u1qzs46n/AAAwKhXEHkxew8HJMGXWciyda?dl=0). You can 
download the model and put them in folder model/tusimple_lanenet/

You can test a single image on the trained model as follows

```
python tools/test_lanenet.py --is_batch False --batch_size 1 
--weights_path model/tusimple_lanenet/tusimple_lanenet_vgg_2018-05-21-11-11-03.ckpt-94000 
--image_path data/tusimple_test_image/0.jpg
```
The results are as follows:

`Test Input Image`

![Test Input](https://github.com/TJCVRS/lanenet-lane-detection/blob/master/data/tusimple_test_image/0.jpg)

`Test Lane Mask Image`

![Test Lane_Mask](https://github.com/TJCVRS/lanenet-lane-detection/blob/master/data/source_image/lanenet_mask_result.png)

`Test Lane Binary Segmentation Image`

![Test Lane_Binary_Seg](https://github.com/TJCVRS/lanenet-lane-detection/blob/master/data/source_image/lanenet_binary_seg.png)

`Test Lane Instance Segmentation Image`

![Test Lane_Instance_Seg](https://github.com/TJCVRS/lanenet-lane-detection/blob/master/data/source_image/lanenet_instance_seg.png)

`Test Lane Instance Embedding Image`

![Test Lane_Embedding](https://github.com/TJCVRS/lanenet-lane-detection/blob/master/data/source_image/lanenet_embedding.png)

If you want to test the model on a whole dataset you may call
```
python tools/test_lanenet.py --is_batch True --batch_size 2 --save_dir data/tusimple_test_image/ret 
--weights_path model/tusimple_lanenet/tusimple_lanenet_vgg_2018-05-21-11-11-03.ckpt-94000 
--image_path data/tusimple_test_image/
```
If you set the save_dir argument the result will be saved in that folder or the result will not be saved but be 
displayed during the inference process holding on 3 seconds per image. I test the model on the whole tusimple lane 
detection dataset and make it a video. You may catch a glimpse of it bellow.
`Tusimple test dataset gif`
![tusimple_batch_test_gif](https://github.com/TJCVRS/lanenet-lane-detection/blob/master/data/source_image/lanenet_batch_test.gif)

## Train your own model
#### Data Preparation
Firstly you need to organize your training data refer to the data/training_data_example folder structure. And you need 
to generate a train.txt and a val.txt to record the data used for training the model. 

The training samples are consist of three components. A binary segmentation label file and a instance segmentation label
file and the original image. The binary segmentation use 255 to represent the lane field and 0 for the rest. The 
instance use different pixel value to represent different lane field and 0 for the rest.

All your training image will be scaled into the same scale according to the config file.

#### Train model
In my experiment the training epochs are 94000, batch size is 4, initialized learning rate is 0.0001 and decrease by 
multiply 0.96 every 5000 epochs. About training parameters you can check the global_configuration/config.py for details. 
You can switch --net argument to change the base encoder stage. If you choose --net vgg then the vgg16 will be used as 
the base encoder stage and a pretrained parameters will be loaded and if you choose --net dense then the dense net will 
be used as the base encoder stage instead and no pretrained parameters will be loaded. And you can modified the training 
script to load your own pretrained parameters or you can implement your own base encoder stage. 
You may call the following script to train your own model

```
python tools/train_lanenet.py --net vgg --dataset_dir data/training_data_example/
```
You can also continue the training process from the snapshot by
```
python tools/train_lanenet.py --net vgg --dataset_dir data/training_data_example/ --weights_path path/to/your/last/checkpoint
```

You may monitor the training process using tensorboard tools

During my experiment the `Total loss` drops as follows:  
![Training loss](https://github.com/TJCVRS/lanenet-lane-detection/blob/master/data/source_image/total_loss.png)

The `Binary Segmentation loss` drops as follows:  
![Training binary_seg_loss](https://github.com/TJCVRS/lanenet-lane-detection/blob/master/data/source_image/binary_seg_loss.png)

The `Instance Segmentation loss` drops as follows:  
![Training instance_seg_loss](https://github.com/TJCVRS/lanenet-lane-detection/blob/master/data/source_image/instance_seg_loss.png)

## Experiment
The accuracy during training process rises as follows: 
![Training accuracy](https://github.com/TJCVRS/lanenet-lane-detection/blob/master/data/source_image/accuracy.png)

Please cite my repo [lanenet-lane-detection](https://github.com/MaybeShewill-CV/lanenet-lane-detection) if you find it helps you.

## TODO
- [ ] Add a embedding visualization tools to visualize the embedding feature map
- [ ] Add detailed explanation of training the components of lanenet separately.
- [ ] Training the model on different dataset
- [ ] Adjust the lanenet hnet model and merge the hnet model to the main lanenet model
