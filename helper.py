import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import cv2

import labels


def gen_batch_function(data_folder='/home/ubuntu/CarND-Semantic-Segmentation/data/cityscapes',
                       image_shape=(256,512), mask_shape=(256,512), num_classes=30):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size, mode='train'):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
       
        image_paths = glob(os.path.join(data_folder, 'leftImg8bit_trainvaltest/leftImg8bit/'+mode+'/*/*'))
        random.shuffle(image_paths)
        
        label_paths = {re.sub(r'_gtFine_labelIds.png', '', os.path.basename(path)): path 
                       for path in glob(os.path.join(data_folder,
                                                     'gtFine_trainvaltest/gtFine/'+mode+'/*/*_*_*_*_labelIds.png'))}
            
        n_images = len(image_paths)
        steps_per_epoch = n_images//batch_size + min(n_images%batch_size, 1)
        
        for batch_i in range(0, n_images, batch_size):
            images = []
            gt_images = []
            fns= []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                image = scipy.misc.imread(image_file)[:,:,:3]
                orig_shape = image.shape
                if mode != 'test':
                    image_base = re.sub(r'_leftImg8bit.png', '', os.path.basename(image_file))
                    gt_image_file = label_paths[image_base]
                    temp_y = scipy.misc.imread(gt_image_file)
                    gt_image = np.zeros((mask_shape[0], mask_shape[1], num_classes))

                    flip = np.random.randint(0,2)
                    if (flip > 0) and (mode=='train'):
                        image = cv2.flip(image, 1)
                        temp_y = cv2.flip(temp_y, 1)

                    crop = np.random.randint(0,2)
                    if (crop > 0) and (mode=='train'):
                        image = cv2.flip(image, 0)
                        temp_y = cv2.flip(temp_y, 0)
                        h = np.random.randint(image_shape[0], orig_shape[0])
                        image = image[-h:,-2*h:, :]
                        temp_y = temp_y[-h:,-2*h:]
                        image = cv2.flip(image, 0)
                        temp_y = cv2.flip(temp_y, 0)

                    temp_y = scipy.misc.imresize(temp_y, mask_shape)


                    for i in range(len(labels.labels)):
                        gt_image[:, :, labels.labels[i].trainId][temp_y == labels.labels[i].id] = 1
                    gt_images.append(gt_image)
                
                image = scipy.misc.imresize(image, image_shape)
                images.append(image)
                fns.append(image_file)
               
            if mode == 'test':
                yield np.array(images) / 255., fns, steps_per_epoch
            else: 
                yield np.array(images) / 255., np.array(gt_images), steps_per_epoch
    return get_batches_fn

def gen_batch_function_KITTI(data_folder='/home/ubuntu/CarND-Semantic-Segmentation/data/training', 
                       image_shape=(256,856), mask_shape=(256,856), num_classes=2, mode='train'):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    if mode != 'test':
        label_paths = {
                    re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
                    for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        
        random.shuffle(image_paths)
        mode_len = int(0.9*len(image_paths))
        train_paths = image_paths[:mode_len]
        val_paths = image_paths[mode_len:]
       
    else:
        val_paths = image_paths
        label_paths = {
                    re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
                    for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
    
        
    def get_batches_fn(batch_size, mode='train'):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        
        if mode == 'train':
            new_image_paths = train_paths
        
        elif mode == 'val':
            new_image_paths = val_paths
            
        else:
            new_image_paths = image_paths
            
        n_images = len(new_image_paths)
        steps_per_epoch = n_images//batch_size + min(n_images%batch_size, 1)
        
        for batch_i in range(0, n_images, batch_size):
            images = []
            gt_images = []
            fns= []
            for image_file in new_image_paths[batch_i:batch_i+batch_size]:
                #image = scipy.misc.imresize(scipy.misc.imread(image_file).astype(float), image_shape)
                image = scipy.misc.imread(image_file).astype(float)
                orig_shape = image.shape
                hw_mul = -1*orig_shape[1] / orig_shape[2]
                
                if mode != 'test':
                    gt_image_file = label_paths[os.path.basename(image_file)]
                    #gt_image_green = scipy.misc.imresize(scipy.misc.imread(gt_image_file).astype(float), image_shape)[:, :, 2]
                    gt_image_green = scipy.misc.imread(gt_image_file).astype(float)[:, :, 2]
                    
                    flip = np.random.randint(0,2)
                    if (flip > 0) and (mode=='train'):
                        image = cv2.flip(image, 1)
                        gt_image_green = cv2.flip(gt_image_green, 1)

                    crop = np.random.randint(0,2)
                    if (crop > 0) and (mode=='train'):
                        image = cv2.flip(image, 0)
                        gt_image_green = cv2.flip(gt_image_green, 0)
                        h = np.random.randint(image_shape[0]//2, orig_shape[0])
                        image = image[-h:, int(hw_mul*h):, :]
                        gt_image_green = gt_image_green[-h:, int(hw_mul*h):]
                        image = cv2.flip(image, 0)
                        gt_image_green = cv2.flip(gt_image_green, 0)
                    
                    
                    gt_image_green = scipy.misc.imresize(gt_image_green, image_shape)
                    
                    gt_image_green[gt_image_green>0] = 255
                    gt_zeros = np.zeros_like(gt_image_green).reshape(image_shape[0], image_shape[1], 1)
                    gt_ones = 255*np.ones_like(gt_image_green)
                    gt_image_blue =  gt_ones - gt_image_green
                    gt_image_blue = gt_image_blue.reshape(image_shape[0], image_shape[1], 1)
                    gt_image_green = gt_image_green.reshape(image_shape[0], image_shape[1], 1)
                    gt_image = np.concatenate((gt_image_green, gt_image_blue), axis=2)
                    gt_images.append(gt_image)
                
                image = scipy.misc.imresize(image, image_shape)
                images.append(image)
                fns.append(image_file)
            if mode == 'test':
                yield np.array(images) / 255., fns, steps_per_epoch
            else:
                yield np.array(images) / 255., np.array(gt_images) / 255., steps_per_epoch
    return get_batches_fn


def load_graph(frozen_graph_filename='checks/frozen_model.pb'):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def view_val_egs(x, y, pred, num_classes, orig_shape=(1024,2048)):
    #temp_y = np.zeros_like(x)
    #temp_pred = np.zeros_like(x)
    temp_y = np.zeros((y.shape[0], y.shape[1], 3))
    temp_pred = np.zeros((y.shape[0], y.shape[1], 3))
    
    for i in range(num_classes):
        #ID = labels.labels[i].trainId
        color = labels.id2color[i]
        temp_y[:,:,0][y[:,:,i] == 1] = color[0] / 255.
        temp_y[:,:,1][y[:,:,i] == 1] = color[1] / 255.
        temp_y[:,:,2][y[:,:,i] == 1] = color[2] / 255.
        
        pred_arg = np.argmax(pred, axis=-1)
        temp_pred[:,:,0][pred_arg == i] = color[0] / 255.
        temp_pred[:,:,1][pred_arg == i] = color[1] / 255.
        temp_pred[:,:,2][pred_arg == i] = color[2] / 255.
        
    #plt.figure(figsize=(15,5))
    plt.figure(figsize=(15,6))
    plt.subplot(2,2,1)
    img = scipy.misc.imresize(x, orig_shape)
    plt.imshow(img)
    plt.title("Image")
    
    plt.subplot(2,2,2)
    gt_img = scipy.misc.imresize(temp_y, orig_shape)
    #gt_img[:,:,2] = 0
    plt.imshow(gt_img)
    plt.title("Ground Truth")
    
    plt.subplot(2,2,3)
    pred_img = scipy.misc.imresize(temp_pred, orig_shape)
    plt.imshow(pred_img)
    plt.title("Prediction")
    
    plt.subplot(2,2,4)
    new_img = cv2.addWeighted(img, 1, pred_img, 0.3, 0)
    plt.imshow(new_img)
    plt.title('Prediction Overlay')
    
    plt.show()
    #plt.imshow(img)
   # plt.imshow(temp_pred)
    
    
    return 0