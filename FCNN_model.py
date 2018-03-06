import tensorflow as tf

import numpy as np
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.5, image_shape=(224, 224), num_classes=35):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout
        self.image_shape = image_shape
        self.num_classes = num_classes

    def build(self, rgb, train_mode):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        #assert red.get_shape().as_list()[1:] == [self.image_shape[0], self.image_shape[1], 1]
        #assert green.get_shape().as_list()[1:] == [self.image_shape[0], self.image_shape[1], 1]
        #assert blue.get_shape().as_list()[1:] == [self.image_shape[0], self.image_shape[1], 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        #assert bgr.get_shape().as_list()[1:] == [self.image_shape[0], self.image_shape[1], 3]
        
        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')
        
        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        #self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        #self.pool4 = self.max_pool(self.conv4_4, 'pool4')
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4", atrous=True, atrous_rate=2)
        
        self.conv5_1 = self.conv_layer(self.conv4_4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        #self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        #self.pool5 = self.max_pool(self.conv5_4, 'pool5')
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4", atrous=True, atrous_rate=4)
        
        self.spp6 = self.spatial_pyramid_pool(self.conv5_4)
        self.spp6_1x1_1 = self.conv1x1_layer(self.spp6[0], 64, "spp6_1x1_1", train_mode, kernel=1, 
                                             relu=True, drop=True, alpha=0.2, batch=False)
        self.spp6_1x1_2 = self.conv1x1_layer(self.spp6[1], 64, "spp6_1x1_2", train_mode, kernel=1, 
                                             relu=True, drop=True, alpha=0.2, batch=True)
        self.spp6_1x1_3 = self.conv1x1_layer(self.spp6[2], 64, "spp6_1x1_3", train_mode, kernel=1, 
                                             relu=True, drop=True, alpha=0.2, batch=False)
        self.spp6_1x1_4 = self.conv1x1_layer(self.spp6[3], 64, "spp6_1x1_4", train_mode, kernel=1, 
                                             relu=True, drop=True, alpha=0.2, batch=True)
        
        self.conv5_shape = self.conv5_4.get_shape().as_list()
        self.upsample7_1 = tf.image.resize_bilinear(self.spp6_1x1_1, [self.conv5_shape[1], self.conv5_shape[2]])
        self.upsample7_2 = tf.image.resize_bilinear(self.spp6_1x1_2, [self.conv5_shape[1], self.conv5_shape[2]])
        self.upsample7_3 = tf.image.resize_bilinear(self.spp6_1x1_3, [self.conv5_shape[1], self.conv5_shape[2]])
        self.upsample7_4 = tf.image.resize_bilinear(self.spp6_1x1_4, [self.conv5_shape[1], self.conv5_shape[2]])
        
        self.conv1x1_8_1 = self.conv1x1_layer(self.conv5_4, 256, "conv1x1_8_1", train_mode, kernel=1, 
                                              relu=True, drop=True, alpha=0.2, batch=True)
        #self.conv1x1_8_2 = self.conv1x1_layer(self.conv4_4, 256, "conv1x1_8_2", train_mode, kernel=1, 
        #                                      relu=True, drop=True, alpha=0.2, batch=True)
        self.fuse8 = tf.concat([self.upsample7_1, self.upsample7_2, self.upsample7_3, 
                                self.upsample7_4, self.conv1x1_8_1], 
                               axis=-1, name="fuse8")
        
        
        self.conv1x1_9 = self.conv1x1_layer(self.fuse8, 512, "conv1x1_9", train_mode, kernel=3, 
                                            relu=True, drop=True, alpha=0.2, batch=True)
        
        self.convT_10 = self.convT_layer(self.conv1x1_9, 128, "convT_10", train_mode, relu=True, alpha=0.2)
        
        self.convT_11 = self.convT_layer(self.convT_10, 128, "convT_11", train_mode, relu=True, alpha=0.2)

        self.convT_12 = self.convT_layer(self.convT_11, 64, "convT_12", train_mode, relu=True, alpha=0.2)
        #self.conv1x1_12 = self.conv1x1_layer(self.convT_11, 64, "conv1x1_12", train_mode, relu=True, drop=False)
        
        self.convT_13 = self.conv1x1_layer(self.convT_12, 64, "convT_13", train_mode, kernel=3, 
                                           relu=True, drop=False, alpha=0.2, batch=False)
        
        self.logits = self.conv1x1_layer(self.convT_13, self.num_classes, "logits", train_mode, kernel=3,
                                         relu=False, drop=False, batch=False)
        
        self.preds = tf.nn.softmax(self.logits, name='preds')
        
        #self.data_dict = None

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name, atrous=False, atrous_rate=2):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)
            if(atrous):
                conv = tf.nn.atrous_conv2d(bottom, filt, atrous_rate, 'SAME')
            else:
                conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu
       
    def conv1x1_layer(self, bottom, out_channels, name, train_mode, kernel=3, relu=True, drop=False, alpha=0.2, batch=True):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(bottom, out_channels, kernel, padding='same',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
            if batch:
                conv = tf.layers.batch_normalization(conv, training=train_mode)
                
            if relu:
                relu = tf.maximum(alpha * conv, conv)
            else:
                relu = conv
               
            if drop:
                drop = tf.layers.dropout(relu, rate=0.5, training=train_mode)
            else:
                drop = relu
                
            return drop
    
    def convT_layer(self, bottom, out_channels, name, train_mode, relu=True, alpha=0.2, init=None):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d_transpose(bottom, out_channels, 3, 2, padding='same',
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
            batch = tf.layers.batch_normalization(conv, training=train_mode)
            if relu:
                relu = tf.maximum(alpha * batch, batch)
            else:
                relu = batch
                
            return relu
    
    def spatial_pyramid_pool(self, inputs, dimensions=[6,3,2,1]):
        pool_list = []
        shape = inputs.get_shape().as_list()
        for d in dimensions:
            h = shape[1]
            w = shape[2]
            ph = np.ceil(h * 1.0 / d).astype(np.int32)
            pw = np.ceil(w * 1.0 / d).astype(np.int32)
            sh = np.floor(h * 1.0 / d + 1).astype(np.int32)
            sw = np.floor(w * 1.0 / d + 1).astype(np.int32)
            pool_result = tf.nn.max_pool(inputs,
                                         ksize=[1, ph, pw, 1], 
                                         strides=[1, sh, sw, 1],
                                         padding='SAME')
            pool_list.append(pool_result)
        return pool_list

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value
            print("Name <{}> not found".format(name))
            
        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./fcc-vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
