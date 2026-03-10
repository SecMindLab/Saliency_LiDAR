'''
environment commands: 

. /vol/cuda/11.7.1/setup.sh
export PATH=/vol/bitbucket/${USER}/Anaconda3/bin/:$PATH
export CUDA_HOME=/vol/cuda/11.7.1
export C_INCLUDE_PATH=/vol/cuda/11.7.1/include:/vol/cuda/11.7.1/targets/x86_64-linux/include
export CPATH=/vol/cuda/11.7.1/include:/vol/cuda/11.7.1/targets/x86_64-linux/include
export PYTHONPATH=/vol/bitbucket/cy19/saliency_map_v2:$PYTHONPATH
source activate
conda activate env_saliency_map_v3
cd /vol/bitbucket/cy19/saliency_map_v2/second

python ./pytorch/saliency_map_addition.py --dataset1 'nusc' --detector1 'pp' --target_object1 'car' --dataset2 'kitti' --detector2 'pp' --target_object2 'Car'
python ./pytorch/saliency_map_addition.py --dataset1 'kitti' --detector1 'pp' --target_object1 'Car' --dataset2 'kitti' --detector2 'second' --target_object2 'Car'

python ./pytorch/saliency_map_addition.py --dataset1 'nusc' --detector1 'pp' --target_object1 'car' --dataset2 'nusc' --detector2 'second' --target_object2 'car'
python ./pytorch/saliency_map_addition.py --dataset1 'nusc' --detector1 'pp_second' --target_object1 'car' --dataset2 'kitti' --detector2 'pp_second' --target_object2 'Car'

'''
import os
from PIL import Image
import numpy as np
import argparse


def scale_map(raw_map):
    # scale a map to standard value between[0, 255]
    min_pix = np.min(raw_map)
    max_pix = np.max(raw_map)
    scaled_map = (raw_map - min_pix)/(max_pix - min_pix) * 255
    return np.uint8(scaled_map)

def car_check(target_object): 
    res = target_object == 'car' or target_object == 'Car'
    return res
def ped_check(target_object):
    res = target_object == 'pedestrian' or target_object == 'Pedestrian'
    return res
def cyc_check(target_object):
    res = target_object == 'bicycle' or target_object == 'Cyclist'
    return res

def split_channels(saliency_map):
    '''
    split a map into 3 channels:
    positive values take the first channel
    fill in the second channel with zeros
    negative values take the third channel
    '''
    map_shape = saliency_map.shape
    positive_mask = saliency_map > 0
    negative_mask = np.logical_not(positive_mask)
    positive_mask = positive_mask.astype(np.float32)
    negative_mask = negative_mask.astype(np.float32)

    # positive_channel is set to the first channel
    positive_channel = np.multiply(positive_mask, saliency_map).reshape(map_shape[0], map_shape[1], 1)
    positive_channel = scale_map(positive_channel)
    # print('positive_channel: ', positive_channel)

    # negative channel is set to the third channel
    negative_channel = -np.multiply(negative_mask, saliency_map).reshape(map_shape[0], map_shape[1], 1)
    negative_channel = scale_map(negative_channel)
    # print('negative_channel: ', negative_channel)

    # add an extra zero channel
    zero_channel = np.zeros_like(positive_channel)

    saliency_map_3channels = np.concatenate([positive_channel,zero_channel, negative_channel], axis = 2)

    return saliency_map_3channels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector1", type=str, default='pp')
    parser.add_argument("--detector2", type=str, default='pp')
    parser.add_argument("--target_object1", type=str, default='Car')
    parser.add_argument("--target_object2", type=str, default='Car')
    parser.add_argument("--dataset1", type=str, default='kitti')
    parser.add_argument("--dataset2", type=str, default='kitti')
    parser.add_argument("--basemap", type=str, default='nearest_corner')
    parser.add_argument("--box_size_factor", type=float, default=1.5)
    args = parser.parse_args()

    if car_check(args.target_object1):  
        # Average car bounding box size (2.6m*4.6m*1.7m)
        patch_size_x = 64
        patch_size_y = 32
    if ped_check(args.target_object1):
        # Average ped bounding box size (0.65m*0.8m*1.7m) 
        patch_size_x = 8
        patch_size_y = 8
    if cyc_check(args.target_object1):
        # Average cyl bounding box size (0.65m*1.6m*1.7m)
        patch_size_x = 16
        patch_size_y = 8
    
    patch_size_x = np.round(patch_size_x * args.box_size_factor).astype(int)
    patch_size_y = np.round(patch_size_y * args.box_size_factor).astype(int)

    all_standard_saliency_map = []
    # restore summarized_saliency_map1
    saliency_map_path1 = "./results/v5.1/{}/basemap_{}/map_{}_{}".format(args.dataset1, args.basemap, args.detector1, args.target_object1) 
    map_name1 = '/summarized_saliency_map_3channels_final'
    summarized_saliency_map_path1 = saliency_map_path1 + map_name1 + '.bin'
    print('restoring summarized_saliency_map from: ', summarized_saliency_map_path1)
    restored_summarized_saliency_map1 = np.fromfile(summarized_saliency_map_path1, dtype = np.float32).reshape(patch_size_x, patch_size_y)
    all_standard_saliency_map.append(restored_summarized_saliency_map1)
    print('restored_summarized_saliency_map: ', restored_summarized_saliency_map1)
    print('max: ', np.max(restored_summarized_saliency_map1))
    print('min: ', np.min(restored_summarized_saliency_map1))

    # restore summarized_saliency_map2
    saliency_map_path2 = "./results/v5.1/{}/basemap_{}/map_{}_{}".format(args.dataset2, args.basemap, args.detector2, args.target_object2) 
    map_name2 = '/summarized_saliency_map_3channels_final'
    summarized_saliency_map_path2 = saliency_map_path2 + map_name2 + '.bin'
    print('restoring summarized_saliency_map from: ', summarized_saliency_map_path2)
    restored_summarized_saliency_map2 = np.fromfile(summarized_saliency_map_path2, dtype = np.float32).reshape(patch_size_x, patch_size_y)
    all_standard_saliency_map.append(restored_summarized_saliency_map2)
    print('restored_summarized_saliency_map: ', restored_summarized_saliency_map2)
    print('max: ', np.max(restored_summarized_saliency_map2))
    print('min: ', np.min(restored_summarized_saliency_map2))
    
    # generate new saliency map
    new_dataset = args.dataset1
    if new_dataset != args.dataset2:
        new_dataset = args.dataset1 + '_' + args.dataset2
    new_detector = args.detector1
    if new_detector != args.detector2:
        new_detector = args.detector1 + '_' + args.detector2
    new_target_object = 'Car' # for now just consider Car object
    if args.target_object1 == args.target_object2:
        new_target_object = args.target_object1
    saliency_map_path = "./results/v5.1/{}/basemap_{}/map_{}_{}".format(new_dataset, args.basemap, new_detector, new_target_object) 
    os.makedirs(saliency_map_path, exist_ok=True)
    summarized_saliency_map = np.sum(all_standard_saliency_map, axis = 0)
    print('summarized_saliency_map: ', summarized_saliency_map)
    print('max: ', np.max(summarized_saliency_map))
    print('min: ', np.min(summarized_saliency_map))
    
    # save raw data of summarized saliency map
    map_name = '/summarized_saliency_map_3channels_final'
    summarized_saliency_map_path = saliency_map_path + map_name + '.bin'
    summarized_saliency_map.tofile(summarized_saliency_map_path)
    print('summarized_saliency_map_path: ', summarized_saliency_map_path)
    # summarized_saliency_map = np.fromfile(summarized_saliency_map_path)
    # save as a PNG file
    summarized_saliency_map_3channels  = split_channels(summarized_saliency_map)
    print('summarized_saliency_map_3channels_final: ', summarized_saliency_map_3channels)
    data = Image.fromarray(summarized_saliency_map_3channels, mode='RGB')
    data.save(saliency_map_path + map_name + '.png')

