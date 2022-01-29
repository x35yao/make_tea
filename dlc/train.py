import deeplabcut
from glob import glob
# objs to train 'teabag', 'tap', 'cup'
objs = ['teabag', 'tap', 'cup']

for obj in objs:
    config_path = glob(f'/home/luke/Desktop/project/make_tea/dlc/make_tea_{obj}*/config.yaml')[0]
    deeplabcut.create_multianimaltraining_dataset(config_path)
    deeplabcut.train_network(config_path, gputouse = 1, saveiters = 10000, maxiters=50000)
    
