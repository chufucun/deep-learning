#!/usr/bin/env python
# encoding: utf-8
"""
日志辅助模块
"""
import logging
import logging.config
import os
import yaml

import _compat

log_path = 'log'
if not os.path.exists(log_path):
    os.mkdir(log_path)

cfg_file_name = "config/logging.yaml"
cfg_file_name = _compat.resolve_filename(_compat.get_module_res(cfg_file_name))


def setup_logging(
        default_path=cfg_file_name,
        default_level=logging.INFO,
        env_key='LOG_CFG'
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


logger = logging.getLogger(__name__)
