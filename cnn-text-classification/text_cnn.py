#! /usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        """
        为了便于调教超参数，我们把代码放进名为TextCNN的类里，用init函数生成模型图。
        :param sequence_length: 句子长度，在预处理环节我们已经填充句子，以保持相同长度(59)；
        :param num_classes: 输出层的类别数，这里是2（好评/差评）；
        :param vocab_size: 词空间大小，用于定义嵌入层维度：[vocabulary_size, embedding_size]；
        :param embedding_size: 嵌入维度；
        :param filter_sizes: 卷积核覆盖的词汇数，每种尺寸的数量由num_filters定义，比如[3,4,5]表示我们有3 * num_filters个卷积核，分别每次滑过3、4、5个词。
        :param num_filters: num_filters：如上。
        :param l2_reg_lambda:
        """

        # Placeholders for input, output and dropout
        # tf.placeholder创建占位符，也就是在训练或测试时将要输入给神经网络的变量。第二个设置项是输入张量的形状。None表示这一维度可以是任意值，使网络可以处理任意数量的批数据。
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        # 在dropout层保留一个神经元的概率也是网络的输入之一，因为我们在训练过程中开启了dropout，在评估和与测试这一项会停用。
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        # 不对权重向量做L2约束
        l2_loss = tf.constant(0.0)

        # Embedding layer
        # 嵌入层（网络的第一层是嵌入层，将词汇映射到低维向量表征，就像是从数据中学习到一张速查表。）
        # tf.device("/cpu:0")：强制使用CPU。默认情况下TensorFlow会尝试调用GPU，但是词嵌入操作们目前还没有GPU支持，所以有可能会报错。
        # tf.name_scope：新建了一个命名域“embedding”，这样在TensorBoard可视化网络的时候，操作会具有更好的继承性。
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # W就是我们的嵌入矩阵，也是训练中学习的目标，用随机均匀分布初始化。
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            # tf.nn.embedding_lookup创建了实际的嵌入操作，输出结果是3D张量，形如[None, sequence_length, embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        #注：TensorFlow的卷积操作conv2d接收4维张量[batch, width, height, channel]，而我们的嵌入结果没有通道维，所以手动加一个变成[None, sequence_length, embedding, 1]。

        # Create a convolution + maxpool layer for each filter size
        # 卷积与最大池化层（现在我们来搭建卷积层和紧随其后的池化层，注意我们的卷积核有多种不同的尺寸。因为每个卷积产生的张量形状不一，我们需要迭代对每一个创建一层，然后再把结果融合到一个大特征向量里。）
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # W是卷积矩阵
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                # h是经过非线性激活函数的输出
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # 每个卷积核都在整个词嵌入空间中扫过，但是每次扫过的个数不一。"VALID" padding 表示卷积核不在边缘做填补，也就是“窄卷积”，输出形状是[1, sequence_length - filter_size + 1， 1， 1]。
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # 最大池化让我们的张量形状变成了[batch_size, 1, 1, num_filters]，最后一位对应特征。把所有经过池化的输出张量组合成一个很长的特征向量，形如[batch_size, num_filter_total]。
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        # 在TensorFlow里，如果需要将高维向量展平，可以在tf.reshape中设置-1。
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        # Dropout大概是正则化卷积神经网络最流行的方法。其背后的原理很简单，就是随机“抛弃”一部分神经元，以此防止他们共同适应(co-adapting)，并强制他们独立学习有用而特征。不被抛弃的比例我们通过dropout_keep_prob这个变量来控制，训练过程中设为0.5，评估过程中设为1.
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        # 分数与预测（借助最大池化+dropout所得的特征向量，我们可以做个矩阵乘法并选择分数最高的类，来做个预测。当然也可以用softmax函数来把生数据转化成正规化的概率，但这并不会改变最终的预测结果。）
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # 这里tf.nn.xw_plus_b是Wx+b的封装。
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # 返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # 损失与精度
        # Calculate mean cross-entropy loss
        # 损失函数(loss)是衡量网络预测误差的指标。我们的目标便是最小化。分类问题中标准的损失函数是交叉熵, cross-entropy loss。
        with tf.name_scope("loss"):
            # tf.nn.softmax_cross_entropy_with_logits是给定分数和正确输入标签之后，计算交叉熵的函数。
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            # 对损失取平均值。我们也可以求和，但那样的话不同batch size的损失就很难比较了。
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        # 定义了精度的表达，这个量在追踪训练和测试过程中很有用
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
