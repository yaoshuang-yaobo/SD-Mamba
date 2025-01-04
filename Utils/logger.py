# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys

__all__ = ['setup_logger']


# reference from: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/logger.py
def setup_logger(name, save_dir, distributed_rank, filename="log.txt", mode='w'):
    # 建一个logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 不要记录非主进程的结果
    if distributed_rank > 0:
        return logger

    # logging.StreamHandler: 再创建一个handler，用于输出到控制台，错误输出流sys.stderr、标准输出流sys.stdout或者文件
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式（formatter）
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    ch.setFormatter(formatter)  # 给handler添加formatter
    logger.addHandler(ch)       # 给logger添加handler

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)  # 利用FileHandler将log写入文件
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
