import os
from easydict import EasyDict as edict

__C = edict()
cfg = __C
__C.DATA = edict()
__C.TRAIN = edict()
__C.LOSS = edict()
# 程序根目录
__C.DATA.BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# 数据集根目录
__C.DATA.IMAGE_PATH = os.path.join(__C.DATA.BASE_PATH, 'data/CAU_Flood/')
# tensorboard保存路径
__C.DATA.TENSERBOARD_PATH = os.path.join(__C.DATA.BASE_PATH, 'visualization/tensorboard')
# 权重存放目录
__C.DATA.WEIGHTS_PATH = os.path.join(__C.DATA.BASE_PATH, 'weights/train_weights')
# 存放预测结果地址
__C.DATA.PRE_PATH = os.path.join(__C.DATA.BASE_PATH, 'Results/')

__C.TRAIN.WEIGHT = 256
__C.TRAIN.WIDTH = 256
__C.TRAIN.CLASSES = 2
__C.TRAIN.IN_CHANNELS_PRE = 3
__C.TRAIN.IN_CHANNELS_POST = 1
# batch_size
__C.TRAIN.BATCH_SIZE = 2
__C.TRAIN.EPOCHS = 400
# 设置线程数
__C.TRAIN.WORKERS = 0
# 是否接着训练
__C.TRAIN.RESUME = False
# 设置学习率
# __C.TRAIN.LR = 1e-4
__C.TRAIN.LR = 0.001800
# 设置 MOMENTUM
__C.TRAIN.MOMENTUM = 0.9
# 设置weight-decay   default: 5e-4
__C.TRAIN.WEIGHT_DECAY = 1e-4
__C.TRAIN.WARMUP_FACTOR = 1.0 / 3
__C.TRAIN.WARMUP_ITERS = 0
__C.TRAIN.WARMUP_METHOD = 'linear'
__C.TRAIN.SKIP_VAL = False

__C.TRAIN.VAL_EPOCH = 1
__C.TRAIN.SAVE_EPOCH = 10
__C.TRAIN.LOG_ITER = 10
__C.TRAIN.LOG_DIR = 'log/'

# 设置dc损失所占的权重
__C.LOSS.DC_WEIGHT = 0.03
# 设置dice损失所占的权重
__C.LOSS.DICE_WEIGHT = 0.4


# 是否使用ohem的损失函数
__C.LOSS.USE_OHEM = False
# 是否使用辅助损失函数
__C.LOSS.AUX = False
# 设置辅助损失所占的权重
__C.LOSS.AUX_WEIGHT = 0.4

