#!/usr/bin/env python
# encoding: utf-8

import unittest

from . import *

logger = logging.getLogger(__name__)


class MyTestCase(unittest.TestCase):
    def test_something(self):
        setup_logging()
        logger.info('It works!')


if __name__ == '__main__':
    unittest.main()
