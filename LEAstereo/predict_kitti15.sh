CUDA_VISIBLE_DEVICES=0 python predictDepth.py \
                --kitti2015=1    --maxdisp=192 \
                --crop_height=384  --crop_width=1248  \
                --data_path='./dataset/testing/' \
                --test_list='./dataloaders/lists/kitti2015_test.list' \
                --save_path='./results/' \
                --fea_num_layer 6 --mat_num_layers 12\
                --fea_filter_multiplier 8 --fea_block_multiplier 4 --fea_step 3  \
                --mat_filter_multiplier 8 --mat_block_multiplier 4 --mat_step 3  \
                --net_arch_fea='run/sceneflow/best/architecture/feature_network_path.npy' \
                --cell_arch_fea='run/sceneflow/best/architecture/feature_genotype.npy' \
                --net_arch_mat='run/sceneflow/best/architecture/matching_network_path.npy' \
                --cell_arch_mat='run/sceneflow/best/architecture/matching_genotype.npy' \
                --resume './run/Kitti15/best/best.pth' 

