# Scale
SCALE_MIN = 0.7
SCALE_MAX = 1.1
SCALE_PROB = 1
TARGET_DIST = 1

# Rotate
MAX_ROTATE_DEGREE = 40

# Crop
CENTER_PERTURB_MAX = 40

# Flip
FLIP_PROB = 0.5

# Path
IMG_DIR = '../data/images'
ANNOTATION_PATH = '../data/mpi_annotations.json'
CHECKPOINT_PATH = 'checkpoint/checkpoint_resnet18.pth'
SUMMARY_PATH = 'runs/resnet18'
IS_SHALLOW = True

BATCH_SIZE = 1
NUM_WORKS = 0
MAX_EPOCH = 422
LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

IMG_SIZE = 384
CELL_SIZE = 32
CELL_NUM = 12

# Loss weights
scale_resp = 0.25
scale_iou = 1
scale_coor = 5
scale_size = 5
scale_limb = 0.5

# nms
thres1 = 0.1
thres2 = 0.3


