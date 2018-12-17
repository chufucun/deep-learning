#!/usr/bin/env python
# -*- coding:utf-8 -*
"""
    数据预处理单元测试
"""
'''
'''
import unittest
from train import preprocess

import tests


class TestPreprocess(unittest.TestCase):

    def test_preprocess(self):
        x_train, y_train, x_dev, y_dev = preprocess()
        print("x_train, y_train, x_dev, y_dev: {0},{1}".format(len(x_train), len(x_dev)))


if __name__ == '__main__':
    unittest.main()
