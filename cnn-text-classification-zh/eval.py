#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

import csv

from process_data import load_data_and_labels, padding_sentences, batch_iter
from word2vec_helpers import embedding_sentences

# 1. 定义参数 模型路径等
# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/eval_taptap_data/taptap_test.pos",
                       "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/eval_taptap_data/taptap_test.neg",
                       "Data source for the negative data.")
word2vec_path = 'E:\BaiduYunDownload\Chinese-Word-Vectors\sgns.weibo.bigram-char.bz2'
# word2vec_path = '/DataScience/Taptap数据集/part_fucun2vec.model'
tf.flags.DEFINE_string("word2vec_fname", word2vec_path,
                       "Word2vec model vectors data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1545015720/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS.flag_values_dict()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

# 2. 加载数据、词典及模型
# 2.1 加载数据


if FLAGS.eval_train:
    x_raw, y_test = load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [0, 1]

#  文本进行向量化
sentences, max_document_length = padding_sentences(x_raw, '<PAD>', 112)
x_test = np.array(embedding_sentences(sentences, word2vec_path))
y_test = np.array(list(y_test))

print("x.shape = {}".format(x_test.shape))
print("y.shape = {}".format(y_test.shape))

# 3 加载模型及预测
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print("latest checkpoint: %s" % checkpoint_file)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )

    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # a. 从模型文件恢复模型
        new_saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        new_saver.restore(sess, checkpoint_file)

        # b. 获得模型输入占位符
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # c. 取得模型输出
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # d. 生成批处理
        batchs = batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False);

        # 进行预测并合并预测结果
        all_predictions = []

        for batch in batchs:
            feed_dict = {
                input_x: batch,
                dropout_keep_prob: 1.0
            }
            _prediction = sess.run(predictions, feed_dict=feed_dict)
            all_predictions = np.concatenate([all_predictions, _prediction])

# 4. 准确度
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w', encoding='utf-8') as f:
    csv.writer(f).writerows(predictions_human_readable)
