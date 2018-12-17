#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, sequence_length, num_classes, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0):
        """
        初始化模型图
        :param sequence_length: 句子长度，预处理阶段我们已经处理过，长度不足的补0
        :param num_classes: 输出层的类别数，这里是两类（好/坏）
        :param embedding_size: 嵌入层词向量的大小或维度
        :param filter_sizes: 过滤器/卷积核大小，覆盖的词汇个数，每种尺寸的数量由num_filters决定
        :param num_filters: 同上
        :param l2_reg_lambda: L2约束
        """

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        # 在dropout层保留一个神经元的概率也是网络的输入之一，训练中会开启dropout，在评估和测试是这一项会停用
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        # 不对权重向量做L2约束
        l2_loss = tf.constant(0.0)

        # 1. 嵌入层：将词汇映射到低围向量表征，就是从数据中学习到一张速查表
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            # tf.nn.embedding_lookup创建了实际的嵌入操作，输出结果是3D张量，形如[None, sequence_length, embedding_size]
            self.embedded_chars = self.input_x
            # 为啥要增加一维，下面tf.nn.conv2d输入需要
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)  # -1表示最后一维

            # 注： TensorFlow中，想要维度增加一维，可以使用tf.expand_dims(input, dim, name=None)函数。
            # 当然，我们常用tf.reshape(input, shape=[])也可以达到相同效果，
            # 但是有些时候在构建图的过程中，placeholder没有被feed具体的值，
            # 这时就会包下面的错误：TypeError: Expected binary or unicode string, got 1

        # 2. 卷积和最大池化层
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 卷积层
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # input 具有[batch, in_height, in_width, in_channels]这样的shape
                # filter 形如 [filter_height,filter_width,in_channels,out_channels]
                # strides 步长信息
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # 添加非线性因子 激活函数
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # 池化
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool")
                pooled_outputs.append(pooled)
        #  合并所有池化的特征
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Dropout 随机“抛弃”一部分神经元，以此防止他们共同适应(co-adapting)，并强制他们独立学习有用而特征。
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # 分数与预测
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            # 相当于matmul(x, weights) + biases.
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # 返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # 损失与精度
        # 有了上一步的分数就可以定义损失函数了，损失函数是衡量网络预测误差的指标。我们的目标是最小化。
        #  使用交叉熵计算损失
        with tf.name_scope("loss"):
            # 1. 第一步是先对网络最后一层的输出做一个softmax
            # 2. 第二步是softmax的输出向量[Y1，Y2,Y3…]和样本的实际标签做一个交叉熵
            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            # 如果不指定第二个参数，那么就在所有的元素中取平均值
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            # 这里不用求和，是为了不同batch size的损失做比较。

        # 这个量在追踪训练和测试过程中很有用。
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
