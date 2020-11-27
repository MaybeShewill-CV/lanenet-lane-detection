import os
import sys
import numpy as np
import shutil

def genList():
    gt_image_path = "./data/training_data_example/training/gt_image/"
    gt_binary_image_path = "./data/training_data_example/training/gt_binary_image/"
    gt_instance_image_path = "./data/training_data_example/training/gt_instance_image/"
    train_path = './data/training_data_example/training/train.txt'
    os.remove(train_path)
    train_txt = open(train_path, 'a')
    imgs = os.listdir(gt_image_path)
    for name in imgs:
        suffix = name[-3:]
        if suffix == "png" or suffix == "PNG":
            n1 = gt_image_path + name
            n2 = gt_binary_image_path + name
            n3 = gt_instance_image_path + name
            train_txt.write(n1 + ' ' + n2 + ' ' + n3 + "\n")
    train_txt.close()
    return

if __name__ == '__main__':
    #genVisualLabelImg()
    #checkMaskImg()
    #mergeMaskAndOrg()
    #chooseImgs()
    #genLabelImg()
    #mvImgs()
    genList()
    print("======finished.======")
