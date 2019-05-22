import argparse

import math
import os
import os.path as ops
import time

import cv2

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import global_config

from lanenet_model import (lanenet_cluster, lanenet_merge_model,
                           lanenet_postprocess)



CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]

# path = ''
# save_path = ''
# path_to_folder = ''
#no_of_frames = 0
#OPTIONAL : logging.basicConfig('log_file_params')
"""
Execute Program in batch:
python tools/test_lanenet.py --is_batch True --batch_size 2 --save_dir data/tusimple_test_image/ret
--weights_path C:\\lanenet-lane-detection\\tools\\model\\tusimple_lanenet\\tusimple_lanenet_vgg_2018-10-19-13-33-56.ckpt-200000
--image_path data\\tusimple_test_image\\


"""

class LanenetLaneDetection():
    """
    Main class to execute complete Lane Detection program
    """

    def init_args(self):
        """

        :return:
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--image_path', type=str,
                            help='The image path or the src image save dir')
        parser.add_argument('--weights_path', type=str,
                            help='The model weights path')
        parser.add_argument('--is_batch', type=str,
                            help='If test a batch of images', default='false')
        parser.add_argument('--batch_size', type=int,
                            help='The batch size of the test images', default=2)  # ORIGINALLY 32
        parser.add_argument('--save_dir', type=str,
                            help='Test result image save dir', default=None)
        parser.add_argument('--use_gpu', type=int,
                            help='If use gpu set 1 or 0 instead', default=1)
        dirpath = os.getcwd()
        #print("current directory is : " + dirpath)
        #foldername = os.path.basename(dirpath)
        #print("Directory name is : " + foldername)
        self.path = 'BASE_PATH'
        if not os.path.exists(self.path):
            os.mkdir(dirpath + '\\video_to_frames')
        self.save_path = '[PATH_TO_SAVE_PROCESSED_FRAMES]'
        if not os.path.exists(self.save_path):
            os.mkdir(dirpath + '\\processed_frames')
        self.weights_path = '[PATH_TO_WEIGHTS_FILE]'
        if not os.path.exists(self.save_path):
            log.info('Path to weights file is incorrect or does not exist.')
        
        self.save_processed_video_path = '[PATH_TO_SAVE_PROCESSED_VIDEO]'
        self.orig_video_path = r'[PATH_TO_SAVE_ORIGINAL_VIDEO]'
        self.no_of_frames = 0
        

        return parser.parse_args()


    def minmax_scale(self, input_arr):
        """

        :param input_arr:
        :return:
        """
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

        return output_arr

 
    def test_lanenet_frames_to_video(self):

        image_folder = self.save_path
       
        video_name = self.save_processed_video_path

        # height, width, layers = 256, 512, 3
        height, width, layers = 352, 640, 3

        video = cv2.VideoWriter(
            video_name, cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height))

        # print(images)
        counter = 0
        stitch_start = time.time()
        for f in sorted(glob.glob(os.path.join(image_folder, "*.jpg"))):

            fname = os.path.dirname(f) + '/' + str(counter)+".jpg"
            if os.path.exists(fname):
                print(fname)
                counter += 1
                #print(counter)
                if counter == self.no_of_frames:
                    break
                print(fname)
                im = cv2.imread(fname)
                
                # cv2.imshow('abc', im)
                video.write(im)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        stitch_end = time.time() - stitch_start
        log.info(
            'Time to stitch video: {:.5f}s'.format(stitch_end))
        cv2.destroyAllWindows()
        video.release()


    def test_lanenet_video_to_frames(self):

        clip = cv2.VideoCapture(
            self.orig_video_path)

        log.info('Getting Video Clip Location...')
        log.info('Reading Video file data...')
        #  If using MoviePy instead of OpenCV methods : 
        #      print("Duration of video : ", clip.duration)
        #      print("FPS : ", clip.fps)
        #      no_of_frames = clip.fps * clip.duration
        #      print('Number of frames : ', no_of_frames)
        rc, send_frames_to_pipeline = clip.read()
        time_start = time.time()
        counter = 0
        while rc:
            #resize to fit the lanenet algorithm
            resize_send_frames_to_pipeline = cv2.resize(
                send_frames_to_pipeline, (640, 352))
            # save frame as JPEG image file
            cv2.imwrite(os.path.join(self.path, "%d.jpg" % counter),
                        resize_send_frames_to_pipeline)     
            rc, send_frames_to_pipeline = clip.read()
            
            counter += 1
        
        # If using Moviepy instead of opencv to pass frames to processing functions : 
            # send_frames_to_pipeline = clip.fl_image(test_lanenet)
            # send_frames_to_pipeline = clip.fl_image(test_lanenet_batch)
        log.info('Video converted to frames successfully')
       

        return send_frames_to_pipeline, self.no_of_frames


    def test_lanenet(self, image_path, weights_path, use_gpu):
        """

        :param image_path:
        :param weights_path:
        :param use_gpu:
        :return:
        """
        assert ops.exists(image_path), '{:s} not exist'.format(image_path)

        log.info('Start reading image data and pre-processing')
        t_start = time.time()
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_vis = image
        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image - VGG_MEAN
        log.info('Image is read, time taken {:.5f}s'.format(time.time() - t_start))

        input_tensor = tf.placeholder(dtype=tf.float32, shape=[
                                    1, 256, 512, 3], name='input_tensor')
        phase_tensor = tf.constant('test', tf.string)

        net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
        binary_seg_ret, instance_seg_ret = net.inference(
            input_tensor=input_tensor, name='lanenet_model')

        cluster = lanenet_cluster.LaneNetCluster()
        postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

        saver = tf.train.Saver()

        # Set sess configuration
        if use_gpu:
            sess_config = tf.ConfigProto(device_count={'GPU': 1})
            log.info('GPU detected, processing on GPU now..')
        else:
            sess_config = tf.ConfigProto(device_count={'CPU': 0})
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'

        sess = tf.Session(config=sess_config)

        with sess.as_default():

            saver.restore(sess=sess, save_path=self.weights_path)
 
            t_start = time.time()
           
            binary_seg_image, instance_seg_image = sess.run([binary_seg_ret, instance_seg_ret],
                                                            feed_dict={input_tensor: [image]})
            t_cost = time.time() - t_start
            log.info('Single image lane line prediction time consuming: {:.5f}s'.format(t_cost))

            binary_seg_image[0] = postprocessor.postprocess(binary_seg_image[0])
            mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                            instance_seg_ret=instance_seg_image[0])

            for i in range(4):
                instance_seg_image[0][:, :, i] = self.minmax_scale(
                    instance_seg_image[0][:, :, i])
            embedding_image = np.array(instance_seg_image[0], np.uint8)

            plt.figure('mask_image')
            plt.imshow(mask_image[:, :, (2, 1, 0)])
            plt.figure('src_image')
            plt.imshow(image_vis[:, :, (2, 1, 0)])
            plt.figure('instance_image')
            plt.imshow(embedding_image[:, :, (2, 1, 0)])
            plt.figure('binary_image')
            plt.imshow(binary_seg_image[0] * 255, cmap='gray')
            plt.show()

        sess.close()

        return


    def test_lanenet_batch(self, batch_size=2, use_gpu=1):
        """

        :param image_dir:
        :param weights_path:
        :param batch_size:
        :param use_gpu:
        :param save_dir:
        :return:
        """
        assert ops.exists(self.path), '{:s} not exist'.format(self.path)
        assert ops.exists(self.save_path), '{:s} not exist'.format(self.save_path)
        assert ops.exists(self.save_processed_video_path), '{:s} not exist'.format(self.save_processed_video_path)

        log.info('Start getting the image file path...')
        image_path_list = sorted(glob.glob('{:s}/**/*.jpg'.format(self.path), recursive=True) +
                                glob.glob('{:s}/**/*.png'.format(self.path), recursive=True) +
                                glob.glob('{:s}/**/*.jpeg'.format(self.path), recursive=True))
        input_tensor = tf.placeholder(dtype=tf.float32, shape=[
                                    2, 352, 640, 3], name='input_tensor')  # 2, 640, 352, 3  #None, 256, 512, 3
        phase_tensor = tf.constant('test', tf.string)

        net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
        binary_seg_ret, instance_seg_ret = net.inference(
            input_tensor=input_tensor, name='lanenet_model')
        cluster = lanenet_cluster.LaneNetCluster()
        postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

        saver = tf.train.Saver()

        # Set sess configuration
        if use_gpu:
            sess_config = tf.ConfigProto(device_count={'GPU': 1})
            log.info('GPU detected, processing on GPU now..')
        else:
            sess_config = tf.ConfigProto(device_count={'GPU': 0})
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'

        sess = tf.Session(config=sess_config)

        with sess.as_default():

            saver.restore(sess=sess, save_path=self.weights_path)

            epoch_nums = int(math.ceil(len(image_path_list) / batch_size))
            
            for epoch in range(epoch_nums):
                log.info(
                    '[Epoch:{:d}] Start image reading and preprocessing...'.format(epoch))
                t_start = time.time()
                image_path_epoch = image_path_list[epoch *
                                                batch_size:(epoch + 1) * batch_size]
                image_list_epoch = [cv2.imread(
                    tmp, cv2.IMREAD_COLOR) for tmp in image_path_epoch]
                image_vis_list = image_list_epoch
                image_list_epoch = [cv2.resize(tmp, (640, 352), interpolation=cv2.INTER_LINEAR)
                                    for tmp in image_list_epoch]
                image_list_epoch = [tmp - VGG_MEAN for tmp in image_list_epoch]
                t_cost = time.time() - t_start
                log.info('[Epoch:{:d}] Pretreatment{:d}Image, total time consuming: {:.5f}s, Average Time per Sheet: {:.5f}'.format(
                    epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

                t_start = time.time()
                binary_seg_images, instance_seg_images = sess.run(
                    [binary_seg_ret, instance_seg_ret], feed_dict={input_tensor: image_list_epoch})
                t_cost = time.time() - t_start
                log.info('[Epoch:{:d}] prediction{:d}Image lane line, total time consuming: {:.5f}s, Average Time per Sheet: {:.5f}s'.format(
                    epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

                cluster_time = []
                for index, binary_seg_image in enumerate(binary_seg_images):
                    t_start = time.time()
                    binary_seg_image = postprocessor.postprocess(binary_seg_image)
                    mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image,
                                                    instance_seg_ret=instance_seg_images[index])
                    cluster_time.append(time.time() - t_start)
                    mask_image = cv2.resize(mask_image, (image_vis_list[index].shape[1],
                                                        image_vis_list[index].shape[0]),
                                            interpolation=cv2.INTER_LINEAR)

                    if self.save_path is None:
                        plt.ion()
                        plt.figure('mask_image')
                        plt.imshow(mask_image[:, :, (2, 1, 0)])
                        plt.figure('src_image')
                        plt.imshow(image_vis_list[index][:, :, (2, 1, 0)])
                        plt.pause(3.0)
                        plt.show()
                        plt.ioff()

                    if self.save_path is not None:
                        mask_image = cv2.addWeighted(
                            image_vis_list[index], 1.0, mask_image, 1.0, 0)
                        image_name = ops.split(image_path_epoch[index])[1]
                        image_save_path = ops.join(self.save_path, image_name)
                        cv2.imwrite(image_save_path, mask_image)

                log.info('[Epoch:{:d}] Get on {:d}Image lane line clustering, total time consuming: {:.5f}s, Average Time per Sheet: {:.5f}'.format(
                    epoch, len(image_path_epoch), np.sum(cluster_time), np.mean(cluster_time)))
           
        
        sess.close()

        return


    def main(self):
        try:
            start_processing_time = time.time()
            
            # init args
            self.init_args()
            
            self.test_lanenet_video_to_frames()
           
            self.test_lanenet_batch()
         
            self.test_lanenet_frames_to_video()
            log.info('Done PREPARING video')
            total_processing_time = time.time() - start_processing_time
           
        #    if args.save_dir is not None and not ops.exists(args.save_dir):
        #        log.error('{:s} not exist and has been made'.format(args.save_dir))
        #        os.makedirs(args.save_dir)

        #    if args.is_batch.lower() == 'false':
        #        test hnet model on single image
        #        test_lanenet(args.image_path, args.weights_path, args.use_gpu)
        #    else:
        #        test hnet model on a batch of image
        #        test_lanenet_batch(image_dir=args.image_path, weights_path=args.weights_path,
        #                           save_dir=args.save_dir, use_gpu=args.use_gpu, batch_size=args.batch_size)
            SystemExit()
        except KeyboardInterrupt:
            print('KEYBOARD INTERRUPT')



if __name__ == '__main__':
    program = LanenetLaneDetection()
    program.main()

   
