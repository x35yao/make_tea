import deeplabcut
from glob import glob
from numba import cuda

# objs to train 'teabag', 'tap', 'cup'
objs = ['teabag', 'tap', 'cup']

for obj in objs:
    config_path = glob(f'/home/luke/Desktop/project/make_tea/dlc/make_tea_{obj}*/config.yaml')[0]
    deeplabcut.create_multianimaltraining_dataset(config_path)
    try:
        deeplabcut.train_network(config_path, gputouse = 1, saveiters = 10000, maxiters=50000)
    except ResourceExhaustedError:
        print('Clean the memory.')
        device = cuda.get_current_device()
        device.reset()
        print('Start training.')
        deeplabcut.train_network(config_path, gputouse = 1, saveiters = 10000, maxiters=50000)
