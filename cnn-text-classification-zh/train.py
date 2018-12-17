#!/usr/bin/env python
# encoding: utf-8
"""
训练文件

"""
import os
import time, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from cnn_params_flags import FLAGS
from process_data import load_data_and_labels, padding_sentences, batch_iter
from word2vec_helpers import embedding_sentences
from text_cnn import TextCNN

from _compat import *


def preprocess():
    """
    数据准备阶段

    :return:
    """
    # 1. 加载数据文件
    x_text, y = load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    #  文本进行向量化
    sentences, max_document_length = padding_sentences(x_text, '<PAD>')
    x = np.array(embedding_sentences(sentences, FLAGS.word2vec_fname))
    y = np.array(list(y))
    print("x.shape = {}".format(x.shape))
    print("y.shape = {}".format(y.shape))

    # shuffle 数据
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # 分拆数据，训练和测试
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, x_dev, y_dev


def train(x_train, y_train, x_dev, y_dev):
    # 训练
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # 实例化TextCNN模型，所有定义的变量和操作就会被放进默认的计算图和会话。
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
            # 模型参数
            params = {
                'sequence_length': x_train.shape[1],
                'num_classes': y_train.shape[1],
                'embedding_size': FLAGS.embedding_dim,
                'filter_sizes': FLAGS.filter_sizes,
                'num_filters': FLAGS.num_filters,
            }

            # 定义训练过程
            # 优化网络损失函数。这里使用Adam优化
            global_step = tf.Variable(0, trainable=False, name="global_step")
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            # train是新建的操作，用来对参数做梯度更新，每一次运行train_op就是一次训练。 tf会自动识别出那些参数是“可计算的”，然后计算他们的梯度。
            # 定义了global_step变量并传入优化器，就可以让TF来完成计数，每运行一次train_op，global_step就+1

            # 定义汇总信息
            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # loss 和 accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # 训练汇总
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # 检查点
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model.ckpt")

            # 检查目录是否存在，TF会假定存在
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            projector_writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
            config = projector.ProjectorConfig()
            embed = config.embeddings.add()
            embed.tensor_name = 'input_x'
            embed.metadata_path = os.path.join(checkpoint_dir, 'metadata.tsv')
            projector.visualize_embeddings(projector_writer, config)

            # 保存模型参数，评估时会使用
            with open(os.path.join(out_dir, 'params'), 'w', encoding='utf-8') as f:
                f.write(str(params))

            # 变量初始化
            sess.run(tf.global_variables_initializer())

            # 定义单步训练
            def train_step(x_batch, y_batch):
                #  喂数据
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }

                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            # 评估模型过程
            def dev_step(x_batch, y_batch, writer=None):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }

                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

        # 生成批处理
        batches = batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")

            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    x_train, y_train, x_dev, y_dev = preprocess()
    train(x_train, y_train, x_dev, y_dev)


if __name__ == "__main__":
    tf.app.run()
