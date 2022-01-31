import argparse

def obtain_predict_args():

    parser = argparse.ArgumentParser(description='LEStereo Prediction')
    parser.add_argument('--crop_height', type=int, default=576, help="crop height")
    parser.add_argument('--crop_width', type=int, default=960, help="crop width")
    parser.add_argument('--maxdisp', type=int, default=192, help="max disp")
    parser.add_argument('--resume', type=str, default='./run/sceneflow/best/checkpoint/best.pth', help="resume from saved model")
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
    parser.add_argument('--sceneflow', type=int, default=1, help='sceneflow dataset? Default=False')
    parser.add_argument('--kitti2012', type=int, default=0, help='kitti 2012? Default=False')
    parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
    parser.add_argument('--middlebury', type=int, default=0, help='Middlebury? Default=False')
    parser.add_argument('--data_path', type=str, default='./dataset/', help="data root")
    parser.add_argument('--test_list', type=str, help="training list")
    parser.add_argument('--save_path', type=str, default='./dataset/', help="location to save result")
    ######### LEStereo params####################
    parser.add_argument('--fea_num_layers', type=int, default=6)
    parser.add_argument('--mat_num_layers', type=int, default=12)
    parser.add_argument('--fea_filter_multiplier', type=int, default=8)
    parser.add_argument('--mat_filter_multiplier', type=int, default=8)
    parser.add_argument('--fea_block_multiplier', type=int, default=4)
    parser.add_argument('--mat_block_multiplier', type=int, default=4)
    parser.add_argument('--fea_step', type=int, default=3)
    parser.add_argument('--mat_step', type=int, default=3)
    parser.add_argument('--net_arch_fea', default='run/sceneflow/best/architecture/feature_network_path.npy', type=str)
    parser.add_argument('--cell_arch_fea', default='run/sceneflow/best/architecture/feature_genotype.npy', type=str)
    parser.add_argument('--net_arch_mat', default='run/sceneflow/best/architecture/matching_network_path.npy', type=str)
    parser.add_argument('--cell_arch_mat', default='run/sceneflow/best/architecture/matching_genotype.npy', type=str)

    args = parser.parse_args()
    return args

class defaultConfig():
    def __init__(self):
        self.crop_height = 576
        self.crop_width=960
        self.maxdisp=192
        self.resume='./run/sceneflow/best/checkpoint/best.pth'
        self.cuda=True
        self.sceneflow=1

        self.data_path='./dataset/'
        self.save_path='./dataset/'
        ######### LEStereo params####################
        self.fea_num_layers=6
        self.mat_num_layers=12
        self.fea_filter_multiplier=8
        self.mat_filter_multiplier=8
        self.fea_block_multiplier=4
        self.mat_block_multiplier=4
        self.fea_step=3
        self.mat_step=3
        self.net_arch_fea='run/sceneflow/best/architecture/feature_network_path.npy'
        self.cell_arch_fea='run/sceneflow/best/architecture/feature_genotype.npy'
        self.net_arch_mat='run/sceneflow/best/architecture/matching_network_path.npy'
        self.cell_arch_mat='run/sceneflow/best/architecture/matching_genotype.npy'
