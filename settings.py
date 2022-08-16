import logging
import os.path as osp
import time
import argparse

# EVAL = True: just test, EVAL = False: train and eval
EVAL = False

BATCH_SIZE = 32
CODE_LEN = 16
MOMENTUM = 0.7
WEIGHT_DECAY = 5e-4

GPU_ID = 1
NUM_WORKERS = 8
EPOCH_INTERVAL = 2
WIDTH = 3

parser = argparse.ArgumentParser(description='Ours')
parser.add_argument('--Bit', default=16, help='hash bit', type=int)
parser.add_argument('--GID', default=1, help='hash bit', type=int)
parser.add_argument('--DS', default=0, help='0 MIR, 1 NUS', type=int)
parser.add_argument('--Alpha', default=1, help='alpha', type=float)
parser.add_argument('--Beta', default=1, help='beta', type=float)
args = parser.parse_args()
CODE_LEN = args.Bit
GPU_ID = args.GID
ALPHA = args.Alpha
BETA = args.Beta

if args.DS == 0:
    DATASET = 'MIRFlickr'
    LABEL_DIR = '../../0_data/MIR/mirflickr25k-lall.mat'
    TXT_DIR = '../../0_data/MIR/mirflickr25k-yall.mat'
    IMG_DIR = '../../0_data/MIR/mirflickr25k-iall.mat'
    NUM_EPOCH = 200
    LR_IMG = 0.005
    LR_TXT = 0.005
    EVAL_INTERVAL = 40

if args.DS == 1:
    DATASET = 'NUSWIDE'
    LABEL_DIR = '../../0_data/NUS-WIDE/nus-wide-tc10-lall.mat'
    TXT_DIR = '../../0_data/NUS-WIDE/nus-wide-tc10-yall.mat'
    IMG_DIR = '../../0_data/NUS-WIDE/nus-wide-tc10-iall.mat'
    NUM_EPOCH = 200
    LR_IMG = 0.005
    LR_TXT = 0.005
    EVAL_INTERVAL = 40

MODEL_DIR = './checkpoint'

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
log_name = now + '_log.txt'
log_dir = './log'
txt_log = logging.FileHandler(osp.join(log_dir, log_name))
txt_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
txt_log.setFormatter(formatter)
logger.addHandler(txt_log)

stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)

logger.info('--------------------------Current Settings--------------------------')
logger.info('EVAL = %s' % EVAL)
logger.info('DATASET = %s' % DATASET)
logger.info('CODE_LEN = %d' % CODE_LEN)
logger.info('GPU_ID =  %d' % GPU_ID)
logger.info('WIDTH =  %d' % WIDTH)
logger.info('ALPHA = %.4f' % ALPHA)
logger.info('BETA = %.4f' % BETA)


logger.info('NUM_EPOCH = %d' % NUM_EPOCH)
logger.info('BATCH_SIZE = %d' % BATCH_SIZE)
logger.info('NUM_WORKERS = %d' % NUM_WORKERS)
logger.info('EPOCH_INTERVAL = %d' % EPOCH_INTERVAL)
logger.info('EVAL_INTERVAL = %d' % EVAL_INTERVAL)


logger.info('LR_IMG = %.4f' % LR_IMG)
logger.info('LR_TXT = %.4f' % LR_TXT)

logger.info('MOMENTUM = %.4f' % MOMENTUM)
logger.info('WEIGHT_DECAY = %.4f' % WEIGHT_DECAY)

logger.info('--------------------------------------------------------------------')
