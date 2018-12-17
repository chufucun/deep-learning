#!/usr/bin/env python
# -*- coding:utf-8 -*
"""
 数据处理，可以参考《自然语言处理时，通常的文本清理流程是什么？》
 以英文文本处理为例。大致分为以下几个步骤：
    1. Normalization
        标准化：字母小写转换、标点符号处理等，英文里通常只要A-Za-z0-9，根据实际情况确定处理
    2. Tokenization
        Token 是“符号”的高级表达。一般指具有某种意义，无法再分拆的符号。就是将每个句子分拆成一系列词，英文里词之间天然有空格。
    3. Stop words
        Stop Word 是无含义的词，例如'is'/'our'/'the'/'in'/'at'等。它们不会给句子增加太多含义，单停止词是频率非常多的词。 为了减少我们要处理的词汇量，从而降低后续程序的复杂度，需要清除停止词。
    4. Part-of-Speech Tagging   词性标注
    5. Named Entity Recognition 命名实体
    6. Stemming and Lemmatization  将词的不同变化和变形标准化
  中文来说在进行分词的时候就处理了。
"""
import os, sys, re
import time
import logging

import numpy as np

from smart_open import smart_open
import pickle

import jieba

from _compat import *

logger = logging.getLogger(__name__)

# 加载词典
jieba.load_userdict(get_module_res("data/userdict.txt"))


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def is_contain_chinese(check_str):
    """
    判断是都包含中文
    :param check_str:
    :return:
    """
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return True


def preprocess_text(concent_lines, sentences, cut_sentence=False):
    '''
    预处理文本：标准化，分句，分词，停用词处理
    :param concent_lines:
    :param sentences:
    :return:
    '''
    if isinstance(concent_lines, str):
        concent_lines = [concent_lines]
    for line in concent_lines:
        try:
            # 1. Normalization   标准化
            line = line.lower()
            # 2. Tokenization
            segs = jieba.lcut(line)
            segs = [v for v in segs if not str(v).isdigit()]  # 去数字
            segs = list(filter(lambda x: x.strip(), segs))  # 去空格
            segs = list(filter(lambda x: len(x) > 1, segs))  # 去长度为1的字符
            # 3. Stop words
            segs = list(filter(lambda x: x not in stopwords, segs))  # 去停用词
            sentences.append(' '.join(segs))
        except Exception:
            print(line)


def load_data_file(source):
    """
    Loads data files
    :param data_file: 目录或文件
    :return:
    """
    if os.path.isfile(source):
        logger.debug('single file given as source, rather than a directory of files')
        logger.debug('consider using models.word2vec.LineSentence for a single file')
        input_files = [source]  # force code compatibility with list of files
    elif os.path.isdir(source):
        source = os.path.join(source, '')  # ensures os-specific slash at end of path
        logger.info('reading directory %s', source)
        input_files = os.listdir(source)
        input_files = [source + filename for filename in input_files]  # make full paths
        input_files.sort()  # makes sure it happens in filename order
    else:  # not a file or a directory, then we can't do anything with it
        raise ValueError('input is neither a file nor a path')
    logger.info('files read into PathLineSentences:%s', '\n'.join(input_files))

    for file_name in input_files:
        logger.info('reading file %s', file_name)
        with smart_open(file_name, encoding='utf-8') as fin:
            for line in fin:
                if line:
                    yield line


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    :param positive_data_file:
    :param negative_data_file:
    :return:
    """
    # 1. 加载数据文件
    positive_data = load_data_file(positive_data_file)
    # 处理行前后空字符
    positive_data = [line.strip() for line in positive_data]
    negative_data = load_data_file(negative_data_file)
    negative_data = [line.strip() for line in negative_data]

    # 2. Split by words
    x_text = positive_data + negative_data
    # x_text = [clean_str(sent) for sent in x_text]

    # 3. Generate labels
    y_positive = [[1, 0] for _ in positive_data]
    y_negative = [[0, 1] for _ in negative_data]
    # 数组拼接，这里使用tf.concat()怎么做?
    y = np.concatenate([y_positive, y_negative], 0)
    return [x_text, y]


def padding_sentences(input_sentences, padding_token, padding_sentence_length=None):
    sentences = [sentence.split(' ') for sentence in input_sentences]
    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max(
        [len(sentence) for sentence in sentences])
    for sentence in sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
    return (sentences, max_sentence_length)


def saveDict(input_dict, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(input_dict, f)


def loadDict(dict_file):
    output_dict = None
    with open(dict_file, 'rb') as f:
        output_dict = pickle.load(f)
    return output_dict


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
