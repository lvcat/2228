#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os
import keras


from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *




BATCH_SIZE = 64

class Model(ModelDesc):
    def __init__(self, depth):
        super(Model, self).__init__()
        self.N = int((depth - 4)  / 3)
        self.growthRate =12

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')
               ]

    def _build_graph(self, input_vars):
        image, label = input_vars
        image = image / 128.0 - 1

        def conv(name, l, channel, stride):
            return Conv2D(name, l, channel, 3, stride=stride,
                          nl=tf.identity, use_bias=False,
                          W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/channel)))
        def add_layer(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                c = BatchNorm('bn1', l)
                c = tf.nn.relu(c)
                c = conv('conv1', c, self.growthRate, 1)
                l = tf.concat([c, l], 3)
            return l

        def add_transition(name, l):
            shape = l.get_shape().as_list()
            print(shape)
            in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                l = BatchNorm('bn1', l)
                l = tf.nn.relu(l)
                l = Conv2D('conv1', l, 256, 1, stride=1, use_bias=False, nl=tf.nn.relu)
                l = AvgPooling('pool', l, 2)
            return l

        def deconv(l):
            filters=shape = l.get_shape().as_list()
            filters = shape[3]
            data = tf.layers.conv2d_transpose(l,filters, kernel_size=(2, 2), strides=2,
                                             kernel_regularizer=keras.regularizers.l2(1e-4))
            data = tf.layers.average_pooling2d(data,[2, 2], strides=2)
            l = tf.concat([l, data], axis=-1)
            l=tf.reduce_sum(l,axis=-1)
            l=l[:,:,:,np.newaxis]
    
            return l

        def dense_net(name):
            # fil = tf.Variable(tf.truncated_normal([3,3,3,3]))
            l = tf.layers.conv2d_transpose(image, 3 , 2, strides=(2,2),name='deconv0')
            l = MaxPooling('pool0', l, 2)
            #print(l.shape)
            l = conv('conv0', l, 16, 1)
            with tf.variable_scope('block1') as scope:

                for i in range(self.N):
                    l = add_layer('dense_layer.{}'.format(i), l)
                
                l = add_transition('transition1', l)

            with tf.variable_scope('block2') as scope:

                for i in range(self.N):
                    l = add_layer('dense_layer.{}'.format(i), l)
                l = add_transition('transition2', l)

            with tf.variable_scope('block3') as scope:

                for i in range(self.N):
                    l = add_layer('dense_layer.{}'.format(i), l)
            l = BatchNorm('bnlast', l)
            l = tf.nn.relu(l)
            l = GlobalAvgPooling('gap', l)
            logits = FullyConnected('linear', l, out_dim=100, nl=tf.identity)

            return logits
        def prediction_incorrect(logits,lable,topk=1,name='incorrect_vector'):    #new
            with tf.name_scope('prediction_incorrect'):
                 x=tf.logical_not(tf.nn.in_top_k(logits,label,topk))
            return tf.cast(x,tf.float32,name=name)
        logits = dense_net("dense_net")

        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label)
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W
        wd_cost = tf.multiply(1e-4, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(['.*/W', ['histogram']])   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar100(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),  #将图像粘贴到背景画布的中心�?            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            #imgaug.Brightness(20),
            #imgaug.Contrast((0.6,1.4)),#Apply ``x = (x - mean) * contrast_factor + mean`` to each channel.
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)  #增强图像组件：创建虚拟类（“增强图像组件”，“cv2”）�?    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)  #MultiProcessPrefetchData 使用python多处理实用程序从数据流预取数据�?    return ds

def get_config():
    #log_dir = '/dataset/cifar-10-batches-py' % (str(args.drop_1), str(args.drop_2), str(args.max_epoch))
    log_dir = './tmp/cifar100-single-fisrt%s-second%s-max%s' % (str(args.drop_1), str(args.drop_2), str(args.max_epoch))
    logger.set_logger_dir(log_dir, action='n')

    # prepare dataset
    dataset_train = get_data('train')
    steps_per_epoch = dataset_train.size()
    dataset_test = get_data('test')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                [ScalarStats('cost'), ClassificationError()]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001)])
        ],
        model=Model(depth=args.depth),
        steps_per_epoch=steps_per_epoch,
        max_epoch=args.max_epoch,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',  default='0',help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    # parser.add_argument('--load',help='load model')
    parser.add_argument('--load', default='./checkpoint',help='load model')
    parser.add_argument('--drop_1',default=150, help='Epoch to drop learning rate to 0.01.') # nargs='*' in multi mode ----150 225 300
    parser.add_argument('--drop_2',default=250,help='Epoch to drop learning rate to 0.001')
    parser.add_argument('--depth',default=40, help='The depth of densenet')
    parser.add_argument('--max_epoch',default=320,help='max epoch')
    args = parser.parse_args()

    # args.gpu = '1'
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)  # Restore a tensorflow checkpoint saved by :class:`tf.train.Saver` or :class:`ModelSaver`.
    
    nr_tower = 0

    if args.gpu:
        nr_tower = len(args.gpu.split(','))

    # print(args.gpu)
    # print(nr_tower)

    # SyncMultiGPUTrainer(config).train()
    launch_train_with_config(config, SyncMultiGPUTrainer([nr_tower]))
