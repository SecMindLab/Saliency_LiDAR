#    Copyright 2026 Y. All Rights Reserved.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ==============================================================================

'''
cd /vol/bitbucket/cy19/saliency_map_v2/second

python ./pytorch/saliency_map.py --box_size_factor 1.5 --dataset 'nusc' --basemap 'nearest_corner' --detector 'second' --target_object 'car'
python ./pytorch/saliency_map.py --box_size_factor 1.5 --dataset 'kitti' --basemap 'nearest_corner' --detector 'pp' --target_object 'Car' 

python ./pytorch/saliency_map.py --box_size_factor 1.5 --dataset 'kitti' --basemap 'nearest_corner' --detector 'pp' --target_object 'Car' --start_scene 1136 --restore True --save_step 25
python ./pytorch/saliency_map.py --box_size_factor 1.5 --dataset 'kitti' --basemap 'nearest_corner' --detector 'second' --target_object 'Car' --start_scene 3690 --restore True --save_step 50

'''
import os
import torch
import torchplus
import math
from PIL import Image
import numpy as np
from pathlib import Path
from google.protobuf import text_format
import argparse
# import second.data.kitti_common as kitti
import act_max as act_max
from second.builder import target_assigner_builder, voxel_builder
from second.core import box_np_ops
from second.data.preprocess import merge_second_batch
from second.pytorch.builder import (box_coder_builder, input_reader_builder, second_builder)
from second.protos import pipeline_pb2
from second.utils.progress_bar import ProgressBar

def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "importance"
    ]
    for k, v in example.items():
        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = torch.tensor(
                v, dtype=dtype, device=device)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(
                    v1, dtype=dtype, device=device)
            example_torch[k] = calib
        elif k == "num_voxels":
            example_torch[k] = torch.tensor(v)
        else:
            example_torch[k] = v
    return example_torch

def build_network(model_cfg, measure_time=False):
    # voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator, device = torch.device("cuda:0"), requires_grad = True)
    bv_range = torch.tensor(model_cfg.voxel_generator.point_cloud_range)[[0, 1, 3, 4]]
    # bv_range = voxel_generator.coors_range_xyz[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    box_coder.custom_ndim = target_assigner._anchor_generators[0].custom_ndim
    net = second_builder.build(
        model_cfg, voxel_generator, target_assigner, measure_time=measure_time)
    return net

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
    # # print('positive_channel: ', positive_channel)

    # negative channel is set to the third channel
    negative_channel = -np.multiply(negative_mask, saliency_map).reshape(map_shape[0], map_shape[1], 1)
    negative_channel = scale_map(negative_channel)
    # # print('negative_channel: ', negative_channel)

    # add an extra zero channel
    zero_channel = np.zeros_like(positive_channel)

    saliency_map_3channels = np.concatenate([positive_channel,zero_channel, negative_channel], axis = 2)

    return saliency_map_3channels

def adaptive_voxelization(target_cloud, bbox_origin, dimension_lidar, yaw_lidar, patch_size_x, patch_size_y):
    '''
    project point cloud to the same size of voxelized bbox. voxelize at raw point cloud format.
    target_cloud: [n, 4] one point cloud, e.g. car cloud
    yaw_lidar: rotation around positive x axis(forward) in lidar format
    return: indeces. xy index of each point
    '''

    # shift coordinate origin from lidar origin to bbox origin
    target_points = target_cloud[:, 0:2] # only focus on x(forward), y(left) values, shape [n,2]
    shifted_points = target_points - bbox_origin[0:2]

    # define rotation matrix for all points
    # rotated the coordinate system to the right(clockwise) <=> rotate target points to the left(counter clockwise)
    rot_mat = [
        [math.cos(yaw_lidar), math.sin(yaw_lidar)],
        [-math.sin(yaw_lidar), math.cos(yaw_lidar)],
    ]

    # rotate the coordinate system to align with bbox edges
    rotated_points = np.dot(shifted_points, rot_mat)

    # voxelization based on bbox coordinates
    bbox_w, bbox_l, bbox_h = dimension_lidar[0], dimension_lidar[1], dimension_lidar[2] # check in box_np_ops.box_camera_to_lidar
    # # print('dimension_lidar: ', dimension_lidar)
    voxel_x_step = bbox_l/patch_size_x
    voxel_y_step = bbox_w/patch_size_y
    step_kernel = [voxel_x_step, voxel_y_step]
    indices = np.array(rotated_points/step_kernel, dtype = np.int32)

    # # print('=============== rotated voxelization ===================')
    # # print('yaw_lidar: ', yaw_lidar)
    # # print('target_points: ', target_points)
    # # print('shifted_points: ', shifted_points)
    # # print('rotated_points: ', rotated_points)
    # # print('voxel_x_step: ', voxel_x_step, ', voxel_y_step: ', voxel_y_step)
    # # print('indices: ', indices)

    # ensure there is no negative indices
    negative_check = indices < 0
    assert len(np.where(negative_check == True)[0]) == 0

    return indices

def generate_saliency_map(args,
                        config_path,
                        model_dir=None,
                        ckpt_path=None,
                        saliency_map_path = None,
                        measure_time=False,
                        batch_size=None,
                        **kwargs):

    assert len(kwargs) == 0
    model_dir = str(Path(model_dir).resolve())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to eval with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    if car_check(args.target_object):  
        # Average car bounding box size (2.6m*4.6m*1.7m)
        patch_size_x = 64
        patch_size_y = 32
    if ped_check(args.target_object):
        # Average ped bounding box size (0.65m*0.8m*1.7m) 
        patch_size_x = 8
        patch_size_y = 8
    if cyc_check(args.target_object):
        # Average cyl bounding box size (0.65m*1.6m*1.7m)
        patch_size_x = 16
        patch_size_y = 8
    
    patch_size_x = np.round(patch_size_x * args.box_size_factor).astype(int)
    patch_size_y = np.round(patch_size_y * args.box_size_factor).astype(int)

    # train_input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    generator_config = model_cfg.voxel_generator

    net = build_network(model_cfg, measure_time=measure_time).to(device)
    target_assigner = net.target_assigner
    # voxel_generator = net.voxel_generator
    voxel_generator_in_dataloader = voxel_builder.build(model_cfg.voxel_generator)
    
    if ckpt_path is None:
        assert model_dir is not None
        # print('try_restore_latest_checkpoints')
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)
    batch_size = batch_size or eval_input_cfg.batch_size
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        # train_input_cfg,
        model_cfg,
        # training=True,
        training=False,
        voxel_generator=voxel_generator_in_dataloader,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    os.makedirs(saliency_map_path, exist_ok=True)
    # print("Generate output labels...")
    bar = ProgressBar()
    bar.start((len(eval_dataset) + batch_size - 1) // batch_size)

    all_standard_saliency_map = []
    all_runtime = {}
    all_runtime['preprocessing'] = []
    all_runtime['IG_computation'] = []
    all_runtime['indexing'] = []
    all_runtime['aggregation'] = []
    start_scene = args.start_scene
    end_scene = args.end_scene
    if args.end_scene == -1:
        end_scene = len(eval_dataset)
    save_step = args.save_step
    # print('start_scene: ', start_scene)
    # print('end_scene: ', end_scene)
    # print('save_step: ', save_step)
    valid_idx = 0

    # restore lastest saliency map
    if args.restore:
        map_name = '/summarized_saliency_map_3channels_scenestep_{}'.format(start_scene)
        # map_name = '/summarized_saliency_map_3channels_final'
        summarized_saliency_map_path = saliency_map_path + map_name + '.bin'
        # print('restoring summarized_saliency_map from: ', summarized_saliency_map_path)
        restored_summarized_saliency_map = np.fromfile(summarized_saliency_map_path, dtype = np.float32).reshape(patch_size_x, patch_size_y)
        all_standard_saliency_map.append(restored_summarized_saliency_map)
        # print('restored_summarized_saliency_map: ', restored_summarized_saliency_map)
        # print('max: ', np.max(restored_summarized_saliency_map))
        # print('min: ', np.min(restored_summarized_saliency_map))

    dataloader = iter(eval_dataloader)
    # skip scenes before start_scene
    for idx in range(start_scene):
        example = next(dataloader)
        # print('skipping example {} : {}'.format(idx, example))
        print('skipping example {} '.format(idx))
    # contiue saliency map aggregation
    for idx in range(start_scene, end_scene):
        example = next(dataloader)
        # print('example {}: {}'.format(idx, example))
        print('example {} '.format(idx))

        # select bboxes in the first batch
        gt_boxes_names = example["gt_boxes_names"][0]
        gt_boxes_lidar = example["gt_boxes_lidar"][0]
        # # print('gt_boxes_lidar: ', gt_boxes_lidar)
        # # print(gt_boxes_lidar.shape)

        # go through all gt boxes in the target scene
        valid_gt_boxes_lidar = [] 
        expand_valid_gt_boxes_lidar = []

        front_near = False
        for j in range(len(gt_boxes_names)):
            type_check = gt_boxes_names[j] == args.target_object

            # define region of interest
            if args.dataset == 'kitti':
                if args.target_object == 'Car':
                    roi = [5, 8, 0, 8]
                if args.target_object == 'Pedestrian':
                    roi = [0, 20, 0, 20]
                if args.target_object == 'Cyclist':
                    roi = [0, 20, 0, 20]
            if args.dataset == 'nusc':
                if args.target_object == 'car':
                    roi = [-8, 0, 5, 8]
                if args.target_object == 'pedestrian':
                    roi = [-20, 0, 0, 20]
                if args.target_object == 'bicycle':
                    roi = [-20, 0, 0, 20]
            
            gt_box_lidar = gt_boxes_lidar[j] 
            region_check = box_np_ops.region_check(gt_box_lidar[0], gt_box_lidar[1], roi)

            front_near = type_check and region_check 
            if front_near:
                valid_gt_boxes_lidar.append(gt_box_lidar) # for now only select one valid gt bbox instead all of them in the scene
                
                # expand the bbox by size
                expand_gt_box_lidar = np.zeros_like(gt_box_lidar)
                expand_gt_box_lidar[0]= gt_box_lidar[0]
                expand_gt_box_lidar[1]= gt_box_lidar[1]
                expand_gt_box_lidar[2]= gt_box_lidar[2]
                expand_gt_box_lidar[3]= gt_box_lidar[3] * args.box_size_factor
                expand_gt_box_lidar[4]= gt_box_lidar[4] * args.box_size_factor
                expand_gt_box_lidar[5]= gt_box_lidar[5] * args.box_size_factor
                expand_gt_box_lidar[6]= gt_box_lidar[6]
                expand_valid_gt_boxes_lidar.append(expand_gt_box_lidar)
                break
        
        valid_gt_boxes_lidar = np.array(valid_gt_boxes_lidar, dtype = np.float64)
        expand_valid_gt_boxes_lidar = np.array(expand_valid_gt_boxes_lidar, dtype = np.float64)
        # print('valid_gt_boxes_lidar: ', valid_gt_boxes_lidar)
        # print('expand_valid_gt_boxes_lidar: ', expand_valid_gt_boxes_lidar)
        ''' 
        compute saliency map 
        '''
        # sum up voxel contributions from point contribution
        standard_saliency_map = np.zeros((patch_size_x, patch_size_y), dtype=np.float32)

        if front_near:
            print('computing saliency map ...') 
            # find the nearest corner of the target bbox
            # since other format was converted to KITTI format before training and inference, all bbox operations are in kitti fasion.
            expand_gt_corners_lidar = box_np_ops.rbbox3d_to_corners(expand_valid_gt_boxes_lidar, origin=[0.5, 0.5, 0.0], axis=2)
            # print('gt corners lidar: ', expand_gt_corners_lidar)
            expand_nearest_corner = expand_gt_corners_lidar[0, 4, 0:3] # first bbox, fifth corner (x, y, z)
            # print('expand_nearest_corner: ', expand_nearest_corner)

            # selected_layer = 'rpn.conv_cls' 
            # selected_layer = 'rpn.conv_box'
            selected_layer = 'rpn.conv_dir_cls'
            activation_dictionary = {}
            net.rpn.conv_dir_cls.register_forward_hook(act_max.layer_hook(activation_dictionary, selected_layer))
            # activation_dictionary is actually not used. final logits is used.

            # basemap = 'zeros'
            example = example_convert_to_torch(example, torch.float32, device)

            contri_points, points_in_box, preprocessing_time, IG_computation_time  = act_max.act_max(network=net,
                            generator_config = generator_config,
                            input=example,
                            roi = roi,
                            expand_valid_gt_boxes_lidar = expand_valid_gt_boxes_lidar,
                            valid_gt_boxes_lidar = valid_gt_boxes_lidar,
                            nearest_corner = expand_nearest_corner,
                            target_object = args.target_object,
                            IG_steps=args.IG_steps,
                            basemap=args.basemap
                            )
            all_runtime['preprocessing'].append(preprocessing_time)
            all_runtime['IG_computation'].append(IG_computation_time)

            '''
            Rotated voxelization on extracted points_in_box
            '''
            import time
            indexing_start_time = time.time()
            # change the start axis of bbox rotation from  positive y axis to positive x axis(forward)
            r_positive_y = expand_valid_gt_boxes_lidar[0][6] - np.pi/2
            while r_positive_y < -np.pi:
                r_positive_y += (np.pi * 2)
            while r_positive_y > np.pi:
                r_positive_y -= (np.pi * 2)

            # Adaptive voxelization to project all bbox(also points in the bbox) to the same size
            indices = adaptive_voxelization(points_in_box, expand_nearest_corner, expand_valid_gt_boxes_lidar[0][3:6], r_positive_y, patch_size_x, patch_size_y)
            # # print('indices shape: ', indices.shape)
            
            # # concate indices with contri_points
            # contri_indices = np.concatenate([indices, contri_points.reshape(-1,1)], axis = 1)
            # # print('contri_indices: ', contri_indices, contri_indices.shape)

            for point_id in range(len(contri_points)):
                # contri_points: contribution of each point
                # indices: xy index of each point
                map_idx, map_idy = indices[point_id][0], indices[point_id][1]
                standard_saliency_map[map_idx][map_idy] += contri_points[point_id]

            indexing_time = time.time() - indexing_start_time
            all_runtime['indexing'].append(indexing_time)

            # # print('standard_saliency_map: ', standard_saliency_map, standard_saliency_map.shape)

            # # saving as a PNG file
            # standard_saliency_map_3channels  = split_channels(standard_saliency_map)
            # # print('standard_saliency_map_3channels: ', standard_saliency_map_3channels)
            # data = Image.fromarray(standard_saliency_map_3channels, mode='RGB')
            # data.save(saliency_map_path + '/standard_saliency_map_3channels_{}.png'.format(idx))
            
            aggregation_start_time = time.time()
            all_standard_saliency_map.append(standard_saliency_map)
            summarized_saliency_map = np.sum(all_standard_saliency_map, axis = 0)
            aggregation_time = time.time() - aggregation_start_time
            all_runtime['aggregation'].append(aggregation_time)

            # # print(all_standard_saliency_map)
            # print('summarized_saliency_map: ', summarized_saliency_map, summarized_saliency_map.shape)
            # print('max: ', np.max(summarized_saliency_map))
            # print('min: ', np.min(summarized_saliency_map))

            valid_id_check = valid_idx > 0 and valid_idx % save_step == 0
            if valid_id_check:
                # save aggregated results every save step
                # print('saving summarized_saliency_map of {} scene steps...'.format(idx))         

                map_name = '/summarized_saliency_map_3channels_scenestep_{}'.format(idx)
                # save raw data of summarized saliency map
                summarized_saliency_map.tofile(saliency_map_path + map_name + '.bin')
                # summarized_saliency_map = np.fromfile(summarized_saliency_map_path)
                # save as a PNG file
                summarized_saliency_map_3channels  = split_channels(summarized_saliency_map)
                # print('summarized_saliency_map_3channels: ', summarized_saliency_map_3channels)
                data = Image.fromarray(summarized_saliency_map_3channels, mode='RGB')
                data.save(saliency_map_path + map_name + '.png')

            valid_idx += 1

    print(all_runtime)

    for phase, time in all_runtime.items():
        print(phase)
        print('mean: ', np.mean(time), 'std: ', np.std(time))


    final_id_check = idx == end_scene - 1
    if final_id_check:
        map_name = '/summarized_saliency_map_3channels_final'
        # save raw data of summarized saliency map
        summarized_saliency_map.tofile(saliency_map_path + map_name + '.bin')
        # summarized_saliency_map = np.fromfile(summarized_saliency_map_path)
        # save as a PNG file
        summarized_saliency_map_3channels  = split_channels(summarized_saliency_map)
        # print('summarized_saliency_map_3channels_final: ', summarized_saliency_map_3channels)
        data = Image.fromarray(summarized_saliency_map_3channels, mode='RGB')
        data.save(saliency_map_path + map_name + '.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", "--d", type=str, default='pp')
    parser.add_argument("--target_object", "--t", type=str, default='car')
    parser.add_argument("--dataset", type=str, default='nusc')
    parser.add_argument("--basemap", type=str, default='nearest_corner')
    parser.add_argument("--start_scene", type=int, default=0)
    parser.add_argument("--end_scene", type=int, default=-1)
    parser.add_argument("--save_step", type=int, default=20)
    parser.add_argument("--IG_steps", type=int, default=25)
    parser.add_argument("--box_size_factor", type=float, default=1.0)
    parser.add_argument("--restore", action="store_true",  help = 'whether restore aggregated saliency map of start_scene')
    args = parser.parse_args()

    if args.dataset == 'nusc':
        if args.detector == 'pp' : 
            config_path = "./configs/nuscenes/all.pp.mida.trainval.config" # pp_model_for_nuscenes_pretrain, only single class such as car
            ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/nuscenes/all_pp_mida_trainval/voxelnet-58650.tckpt" # pretrained model
            model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/nuscenes/all_pp_mida_trainval" # for restore the latest checkpoint(.json)
        if args.detector == 'second':
            config_path = "./configs/nuscenes/all.second.fhd.config"
            ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/nuscenes/all_second_fhd_trainval/voxelnet-58650.tckpt" # pretrained model
            model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/nuscenes/all_second_fhd_trainval" # for restore the latest checkpoint(.json)

    elif args.dataset == 'kitti':
        # link dataset_path to raw dataset, in case pointing to the poisoned dataset 
        dataset_path = Path('/vol/bitbucket/cy19/saliency_map_v2/second/data/kitti/training/velodyne')
        if dataset_path.is_symlink():
            dataset_path.unlink()
        raw_dataset_path = '/vol/bitbucket/cy19/saliency_map_v2/second/data/kitti/training/velodyne_raw'
        dataset_path.symlink_to(raw_dataset_path)

        if args.detector == 'pp':
            if args.target_object == 'Car':
                config_path = "./configs/pointpillars/car/xyres_16.config" 
                ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/pp_car/voxelnet-296960.tckpt" # pretrained model
                model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/pp_car" # for restore the latest checkpoint(.json)
            if args.target_object == 'Pedestrian' or args.target_object == 'Cyclist':
                config_path = "./configs/pointpillars/ped_cycle/xyres_16.config" 
                ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/pp_ped_cycle/voxelnet-296960.tckpt" # pretrained model
                model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/pp_ped_cycle" # for restore the latest checkpoint(.json)
        if args.detector == 'second':
            # # 'all.fhd.config'
            # config_path = "./configs/all.fhd.config" 
            # ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_all/voxelnet-99040.tckpt" # pretrained model
            # model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_all" # for restore the latest checkpoint(.json)
            if args.target_object == 'Car':
                # 'car_fhd'
                config_path = "./configs/car.fhd.config" 
                ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_car_fhd/voxelnet-74280.tckpt" # pretrained model
                model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_car_fhd" # for restore the latest checkpoint(.json)
                # # 'car_onestage'
                # config_path = "./configs/car.fhd.onestage.config" 
                # ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_car_onestage/voxelnet-27855.tckpt" # pretrained model
                # model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_car_onestage" # for restore the latest checkpoint(.json)
                # # 'car_lite'
                # config_path = "./configs/car.lite.config" 
                # ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_car_lite/voxelnet-15500.tckpt" # pretrained model
                # model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_car_lite" # for restore the latest checkpoint(.json)
            if args.target_object == 'Pedestrian' or args.target_object == 'Cyclist':
                config_path = "./configs/people.fhd.config" 
                ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_people/voxelnet-30950.tckpt" # pretrained model
                model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_people" # for restore the latest checkpoint(.json)

    saliency_map_path = "./results/v5.2/{}/basemap_{}/map_{}_{}/IGstep{}".format(args.dataset, args.basemap, args.detector, args.target_object, args.IG_steps) 
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    generate_saliency_map(
        args,
        config,
        model_dir=str(model_dir),
        ckpt_path=str(ckpt_name),
        saliency_map_path = saliency_map_path,
        batch_size=1,
        measure_time=True)
