import os
import argparse
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import shutil
import random
import json
import math
import sys
import codecs

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

semantic_label_dict = {
    #'单虚线': 255,
    #'单实线': 255,
    #'双实线': 255,
    #'双虚线': 255,
    '可行驶区域': 0,
    '直行或左转': 0,
    '左转或直行': 0,
    '直行或右转': 0,
    '左弯或向左合流': 0,
    '右弯或向右合流': 0,
    '右转或向右合流': 0,
    '左右转弯': 0,
    '左转或掉头': 0,
    '直行': 0,
    '左转': 0,
    '右转': 0,
    '掉头': 0,
    '箭头': 0,

    '停止线': 0,
    '减速带': 0,
    '减速让行': 0,
    '斑马线': 0,
    '车距确认线': 0,
    '导流带': 0,
    '菱形减速标': 0,

    '限速': 0,
    '文字': 0,
    '其他': 0,
    '其它': 0,
    'TrafficSign': 0,
}
instance_label_dict = {
    '左三': 40,
    '左二': 80,
    '左一': 120,
    '右一': 160,
    '右二': 200,
    '右三': 240,
    '左四': 0,
    '右四': 0,
    '左五': 0,
    '右五': 0,
    '左六': 0,
    '右六': 0,
}

#semantic_new = semantic_image[min_h:max_h, :]
#CityTunnel: 20032514*; 505, 825
#Highway:    14*;       415, 735
#sanhuan:    2002*;     480, 800
#shunyi:     frame*;    281, 505

def deleteInvalidXML(root_path):
    xmls = os.listdir(xml_dir_path)
    for idx, name in enumerate(xmls):
        name_path = os.path.join(xml_dir_path, name)
        if not (name.endswith('.xml')):
            os.remove(name_path)
            print('Error postfix: ', name_path)
            continue
        tree = ET.parse(name_path)
        root = tree.getroot()
        if root is None:
            print('Error name: ', name_path)
            os.remove(name_path)
            continue
        label = root.find('labeled')
        if label.text == 'false':
            print("get null label: ", name_path)
            os.remove(name_path)
            continue

        for obt in root.iter('object'):
            item = obt.find('item')
            if item is None:
                print('Error name: ', name_path)
                os.remove(name_path)
    return

def compute_polygon_area(points):
    point_num = len(points)
    if (point_num < 3): return 0.0
    s = points[0][1] * (points[point_num - 1][0] - points[1][0])
    # for i in range(point_num): # (int i = 1 i < point_num ++i):
    for i in range(1, point_num):  # 有小伙伴发现一个bug，这里做了修改，但是没有测试，需要使用的亲请测试下，以免结果不正确。
        s += points[i][1] * (points[i - 1][0] - points[(i + 1) % point_num][0])
    return abs(s / 2.0)


def processXml(root_path):
    label_dir = os.path.join(root_path, 'gt_xml')
    gt_image_dir = os.path.join(root_path, 'gt_image')
    gt_semantic_dir = os.path.join(root_path, 'gt_binary_image')
    gt_instance_dir = os.path.join(root_path, 'gt_instance_image')
    semantic_keys = semantic_label_dict.keys()
    instance_keys = instance_label_dict.keys()

    files = os.listdir(label_dir)
    for idx, name in enumerate(files):
        if not name.endswith('.xml'):
            continue
        name = name[:-3]+'xml'
        png_name = name[:-3] + 'png'
        jpg_name = name[:-3] + 'jpg'
        src_path = os.path.join(gt_image_dir, jpg_name)
        semantic_path = os.path.join(gt_semantic_dir, png_name)
        instance_path = os.path.join(gt_instance_dir, png_name)
        _path = os.path.join(gt_image_dir, jpg_name)
        src_image = cv2.imread(src_path, cv2.IMREAD_COLOR)
        semantic_image = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)
        instance_image = cv2.imread(instance_path, cv2.IMREAD_GRAYSCALE)

        xml_path = os.path.join(label_dir, name)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for item in root.iter('item'):
            instance = item.find('name')
            semantic_v = -1
            instance_v = -1
            for semantic_key in semantic_keys:
                if instance.text.find(semantic_key) >= 0:
                    semantic_v = semantic_label_dict[semantic_key]
                    break
            for instance_key in instance_keys:
                if instance.text.find(instance_key) >= 0:
                    semantic_v = 1
                    instance_v = instance_label_dict[instance_key]
                    break
            if semantic_v == -1 and instance_v == -1:
                print(name, instance.text)
                continue
            if semantic_v == 0 or instance_v == 0:
                continue

            arr = []
            polygon = item.find('polygon')
            cubic_bezier = item.find('cubic_bezier')
            bndbox = item.find('bndbox')
            if polygon is not None:
                for xy in polygon:
                    arr.append(int(xy.text))
            elif cubic_bezier is not None:
                for xy in cubic_bezier:
                    if len(xy.tag) <= 3:
                        arr.append(int(xy.text))
            elif bndbox is not None:
                for xy in bndbox:
                    arr.append(int(xy.text))
            else:
                print('Error boundingbox: %s, %s ' % (name, instance.text))
                # shutil.copy(src_path, os.path.join(root_path, 'errors'))
                # shutil.copy(xml_path, os.path.join(root_path, 'errors'))
                continue

            pt = []
            for i in range(0, len(arr) - 1, 2):
                pt.append([arr[i], arr[i + 1]])
            b = np.array([pt], dtype=np.int32)
            s = compute_polygon_area(pt)
            # print('v: %d, s: %d' % (instance_v, s))
            if s > 3 and semantic_v > 0:
                cv2.fillPoly(semantic_image, b, semantic_v)
            if s > 3 and instance_v > 0:
                cv2.fillPoly(instance_image, b, instance_v)
            if s <= 3:
                print('Warning: Too little target: %s, %s, %d' % (name, instance.text, s))
        if semantic_image.max() > 0:
            cv2.imwrite(semantic_path, semantic_image)
            cv2.imwrite(instance_path, instance_image)
        else:
            # print('Error null name: ', name)
            os.remove(xml_path) 

def processJson(root_path):
    gt_json_dir = os.path.join(root_path, 'gt_json')
    gt_image_dir = os.path.join(root_path, 'gt_image')
    gt_semantic_dir = os.path.join(root_path, 'gt_binary_image')
    if os.path.exists(gt_semantic_dir):
        shutil.rmtree(gt_semantic_dir)
    os.mkdir(gt_semantic_dir)
    gt_instance_dir = os.path.join(root_path, 'gt_instance_image')
    if os.path.exists(gt_instance_dir):
        shutil.rmtree(gt_instance_dir)
    os.mkdir(gt_instance_dir)
    semantic_keys = semantic_label_dict.keys()
    instance_keys = instance_label_dict.keys()

    jsons = os.listdir(gt_json_dir)
    for name in jsons:
        json_path = os.path.join(gt_json_dir, name)
        org_im_name = name[:-4] + 'jpg'
        semantic_name = name[:-4] + 'png'
        instance_name = semantic_name
        im = cv2.imread(os.path.join(gt_image_dir, org_im_name), cv2.IMREAD_COLOR)
        ss = (im.shape[0], im.shape[1])
        semantic_img_path = os.path.join(gt_semantic_dir, semantic_name)
        instance_img_path = os.path.join(gt_instance_dir, instance_name)
        semantic_im = np.zeros(ss, np.uint8)
        instance_im = np.zeros(ss, np.uint8)
        #print(json_path)
        with open(json_path, encoding='utf-8-sig', errors='ignore') as f:
            info_dict = json.loads(f.read(), strict=False)
            info_shapes = info_dict['shapes']
            for info in info_shapes:
                semantic_v = -1
                instance_v = -1
                for semantic_key in semantic_keys:
                    if info['label'].find(semantic_key) >= 0:
                        semantic_v = semantic_label_dict[semantic_key]
                        break
                for instance_key in instance_keys:
                    if info['label'].find(instance_key) >= 0:
                        semantic_v = 1
                        instance_v = instance_label_dict[instance_key]
                        break
                if semantic_v == -1 and instance_v == -1:
                    print(json_path, info['label'])
                    continue

                info_pts = info['points']
                # print(info_pts)
                pts_nums = len(info_pts)
                edge_lines = []
                for pti in range(pts_nums):
                    x0 = info_pts[pti][0]
                    y0 = info_pts[pti][1]
                    edge_lines.append([round(x0),round(y0)]) 
                b = np.array([edge_lines], dtype=np.int32)
                s = compute_polygon_area(edge_lines)
                # print('v: %d, s: %d' % (instance_v, s))
                if semantic_v > -1:
                    cv2.fillPoly(semantic_im, b, semantic_v)
                if instance_v > -1:
                    cv2.fillPoly(instance_im, b, instance_v)
                cv2.imwrite(semantic_img_path, semantic_im)
                cv2.imwrite(instance_img_path, instance_im)
    return

def resizeAll(root_path):
    gt_image_dir = os.path.join(root_path, 'gt_image')
    gt_semantic_dir = os.path.join(root_path, 'gt_binary_image')
    gt_instance_dir = os.path.join(root_path, 'gt_instance_image')
    pngs = os.listdir(gt_semantic_dir)
    for png in pngs:
        #semantic_new = semantic_image[min_h:max_h, :]
        #CityTunnel: 20032514*; 505, 825
        #Highway:    14*;       415, 735
        #sanhuan:    2002*;     480, 800
        #shunyi:     frame*;    281, 505

        jpg = png[:-3] + 'jpg'
        org_im_path = os.path.join(gt_image_dir, jpg)
        semantic_im_path = os.path.join(gt_semantic_dir, png)
        instance_im_path = os.path.join(gt_instance_dir, png)
        org_im = cv2.imread(org_im_path, cv2.IMREAD_COLOR)
        semantic_im = cv2.imread(semantic_im_path, cv2.IMREAD_GRAYSCALE)
        instance_im = cv2.imread(instance_im_path, cv2.IMREAD_GRAYSCALE)
        '''
        if jpg[:8] == '20032514': #CityTunnel
            #print(jpg[:8])
            org_im = org_im[505:825,:]
            semantic_im = semantic_im[505:825,:]
            instance_im = instance_im[505:825,:]
        elif jpg[:2] == '14': #Highway
            #print(jpg[:2])
            org_im = org_im[415:735,:]
            semantic_im = semantic_im[415:735,:]
            instance_im = instance_im[415:735,:]
        elif jpg[:4] == '2002': #sanhuan
            #print(jpg[:4])
            org_im = org_im[480:800,:]
            semantic_im = semantic_im[480:800,:]
            instance_im = instance_im[480:800,:]
        elif jpg[:5] == 'frame': #shunyi
            #print(jpg[:5])
            org_im = org_im[281:505,:]
            semantic_im = semantic_im[281:505,:]
            instance_im = instance_im[281:505,:]
        '''
        if org_im.shape[1] > 1280:
            #dim = (1280, 224)
            dim = (1280, 720)
            org_im = cv2.resize(org_im, dim, interpolation = cv2.INTER_CUBIC)
            semantic_im = cv2.resize(semantic_im, dim, interpolation = cv2.INTER_NEAREST)
            instance_im = cv2.resize(instance_im, dim, interpolation = cv2.INTER_NEAREST)
            cv2.imwrite(org_im_path, org_im)
            cv2.imwrite(semantic_im_path, semantic_im)
            cv2.imwrite(instance_im_path, instance_im)

def gen_index(root_path):
    semantic_path = os.path.join(root_path, 'gt_binary_image')
    index_path = os.path.join(root_path, 'index')
    if not os.path.exists(index_path):
        os.mkdir(index_path)
    else:
        shutil.rmtree(index_path)
        os.mkdir(index_path)
    all_txt_path = os.path.join(index_path, 'all.txt')
    train_txt_path = os.path.join(index_path, 'train.txt')
    trainval_txt_path = os.path.join(index_path, 'test.txt')
    val_txt_path = os.path.join(index_path, 'val.txt')    

    semantic_list = os.listdir(semantic_path)
    for each in semantic_list:
        if not each.endswith('.png'):
            continue
        jpg_each = each[:-3] + 'jpg'
        im_path = os.path.join(root_path, 'gt_image', jpg_each)
        semantic_path = os.path.join(root_path, 'gt_binary_image', each)
        instance_path = os.path.join(root_path, 'gt_instance_image', each)
        line = im_path + ' ' + semantic_path + ' ' + instance_path
        with open(all_txt_path, 'a') as f:
            f.write(line + '\n')

    with open(all_txt_path, 'r') as f:
        lines = f.readlines()
        g = [i for i in range(len(lines))]
        random.shuffle(g)
        train = g[:int(len(lines) * 19 / 20)]
        #trainval = g[int(len(lines) * 18 / 20):int(len(lines) * 19 / 20)]
        val = g[int(len(lines) * 19 / 20):]

        for n, line in enumerate(lines):
            if n in train:
                with open(train_txt_path, 'a') as trainf:
                    trainf.write(line)
            #elif n in trainval:
                #with open(trainval_txt_path, 'a') as trainvalf:
                    #trainvalf.write(line)
            elif n in val:
                with open(val_txt_path, 'a') as valf:
                    valf.write(line)
    shutil.copyfile(val_txt_path, trainval_txt_path)

if __name__ == '__main__':
    root_path = '/workspace/mogo_data'
    processJson(root_path)
    print('processJson done')
    processXml(root_path)
    print('processxml done')
    resizeAll(root_path)
    print('resizeAll done')
    gen_index(root_path)
    print('======finished!======')

