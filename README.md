# LaneNet-Lane-Detection
Use tensorflow to implement a Deep Neural Network for real time lane detection mainly based on the IEEE IV conference 
paper "Towards End-to-End Lane Detection: an Instance Segmentation Approach".You can refer to their paper for details 
https://arxiv.org/abs/1802.05591. This model consists of a encoder-decoder stage, binary semantic segmentation stage 
and instance semantic segmentation using discriminative loss function for real time lane detection task.

The main network architecture is as follows:

`Network Architecture`
![NetWork_Architecture](./data/source_image/network_architecture.png)

## Installation
This software has only been tested on ubuntu 16.04(x64), python3.5, cuda-9.0, cudnn-7.0 with a GTX-1070 GPU. 
To install this software you need tensorflow 1.10.0 and other version of tensorflow has not been tested but I think 
it will be able to work properly in tensorflow above version 1.10. Other required package you may install them by

```
pip3 install -r requirements.txt
```

## Test model
In this repo I uploaded a model trained on tusimple lane dataset [Tusimple_Lane_Detection](http://benchmark.tusimple.ai/#/).
The deep neural network inference part can achieve around a 50fps which is similar to the description in the paper. But
the input pipeline I implemented now need to be improved to achieve a real time lane detection system.

The trained lanenet model weights files are stored in 
[new_lanenet_model_file](https://www.dropbox.com/sh/tnsf0lw6psszvy4/AAA81r53jpUI3wLsRW6TiPCya?dl=0). You can 
download the model and put them in folder model/tusimple_lanenet/

You can test a single image on the trained model as follows

```
python tools/test_lanenet.py --weights_path ./model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt 
--image_path ./data/tusimple_test_image/0.jpg
```
The results are as follows:

`Test Input Image`

![Test Input](./data/tusimple_test_image/0.jpg)

`Test Lane Mask Image`

![Test Lane_Mask](./data/source_image/lanenet_mask_result.png)

`Test Lane Binary Segmentation Image`

![Test Lane_Binary_Seg](./data/source_image/lanenet_binary_seg.png)

`Test Lane Instance Segmentation Image`

![Test Lane_Instance_Seg](./data/source_image/lanenet_instance_seg.png)

If you want to evaluate the model on the whole tusimple test dataset you may call
```
python tools/evaluate_lanenet_on_tusimple.py 
--image_dir ROOT_DIR/TUSIMPLE_DATASET/test_set/clips 
--weights_path ./model/tusimple_lanenet_vgg/tusimple_lanenet.ckpt 
--save_dir ROOT_DIR/TUSIMPLE_DATASET/test_set/test_output
```
If you set the save_dir argument the result will be saved in that folder 
or the result will not be saved but be 
displayed during the inference process holding on 3 seconds per image. 
I test the model on the whole tusimple lane 
detection dataset and make it a video. You may catch a glimpse of it bellow.

`Tusimple test dataset gif`
![tusimple_batch_test_gif](./data/source_image/lanenet_batch_test.gif)

## Train your own model
#### Data Preparation
Firstly you need to organize your training data refer to the data/training_data_example folder structure. And you need 
to generate a train.txt and a val.txt to record the data used for training the model. 

The training samples are consist of three components. A binary segmentation label file and a instance segmentation label
file and the original image. The binary segmentation use 255 to represent the lane field and 0 for the rest. The 
instance use different pixel value to represent different lane field and 0 for the rest.

All your training image will be scaled into the same scale according to the config file.

Use the script here to generate the tensorflow records file

```
python data_provider/lanenet_data_feed_pipline.py 
--dataset_dir ./data/training_data_example
--tfrecords_dir ./data/training_data_example/tfrecords
```

#### Train model
In my experiment the training epochs are 80010, batch size is 4, initialized learning rate is 0.001 and use polynomial 
decay with power 0.9. About training parameters you can check the global_configuration/config.py for details. 
You can switch --net argument to change the base encoder stage. If you choose --net vgg then the vgg16 will be used as 
the base encoder stage and a pretrained parameters will be loaded. And you can modified the training 
script to load your own pretrained parameters or you can implement your own base encoder stage. 
You may call the following script to train your own model

```
python tools/train_lanenet.py 
--net vgg 
--dataset_dir ./data/training_data_example
-m 0
```
You can also continue the training process from the snapshot by
```
python tools/train_lanenet.py 
--net vgg 
--dataset_dir data/training_data_example/ 
--weights_path path/to/your/last/checkpoint
-m 0
```

You may monitor the training process using tensorboard tools

During my experiment the `Total loss` drops as follows:  
![Training loss](./data/source_image/total_loss.png)

The `Binary Segmentation loss` drops as follows:  
![Training binary_seg_loss](./data/source_image/binary_seg_loss.png)

The `Instance Segmentation loss` drops as follows:  
![Training instance_seg_loss](./data/source_image/instance_seg_loss.png)

## Experiment
The accuracy during training process rises as follows: 
![Training accuracy](./data/source_image/accuracy.png)

Please cite my repo [lanenet-lane-detection](https://github.com/MaybeShewill-CV/lanenet-lane-detection) if you use it.

## Recently updates 2018.11.10
Adjust some basic cnn op according to the new tensorflow api. Use the 
traditional SGD optimizer to optimize the whole model instead of the
origin Adam optimizer used in the origin paper. I have found that the
SGD optimizer will lead to more stable training process and will not 
easily stuck into nan loss which may often happen when using the origin
code.

I have uploaded a new lanenet model trained on tusimple dataset using the
new code here [new_lanenet_model_file](https://www.dropbox.com/sh/tnsf0lw6psszvy4/AAA81r53jpUI3wLsRW6TiPCya?dl=0).
You may download the new model weights and update the new code. To update
the new code you just need to

```
git pull origin master
```
The rest are just the same as which mentioned above. And recently I will 
release a new model trained on culane dataset.

## Recently updates 2018.12.13
Since a lot of user want a automatic tools to generate the training samples
from the Tusimple Dataset. I upload the tools I use to generate the training
samples. You need to firstly download the Tusimple dataset and unzip the 
file to your local disk. Then run the following command to generate the 
training samples and the train.txt file.

```angular2html
python tools/generate_tusimple_dataset.py --src_dir path/to/your/unzipped/file
```

The script will make the train folder and the test folder. The training 
samples of origin rgb image, binary label image, instance label image will
be automatically generated in the training/gt_image, training/gt_binary_image,
training/gt_instance_image folder.You may check it yourself before start
the training process.

Pay attention that the script only process the training samples and you 
need to select several lines from the train.txt to generate your own 
val.txt file. In order to obtain the test images you can modify the 
script on your own.

## Recently updates 2019.05.16

New model weights can be found [here](https://www.dropbox.com/sh/tnsf0lw6psszvy4/AAA81r53jpUI3wLsRW6TiPCya?dl=0)


## TODO
- [x] Add a embedding visualization tools to visualize the embedding feature map
- [x] Add detailed explanation of training the components of lanenet separately.
- [x] Training the model on different dataset
- ~~[ ] Adjust the lanenet hnet model and merge the hnet model to the main lanenet model~~
- ~~[ ] Change the normalization function from BN to GN~~
