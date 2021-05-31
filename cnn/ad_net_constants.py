COVER_FOLDER = '/cover'
STEGO_FOLDER = '/stego'

COVER_SIGNAL = 0
STEGO_SIGNAL = 1

EPOCH = 300

# batch size for data loader
# 8 cover/stego pairs
DATA_LOADER_BATCH_SIZE = 16

# optimizer hyper parameters
MOMENTUM = 0.9
# with LEARNING_RATE_GAMMA it will be
# lr = 0.005    if epoch < 50
# lr = 0.0001   if 50 <= epoch < 250
# lr = 0.0002   if epoch >= 250
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.0005
LR_GAMMA = 0.2

# a number of epoch when the learning rate it decrease
DECAY_EPOCH = [50, 150, 250]
