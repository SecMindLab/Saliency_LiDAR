# use dataloader of second
# support patch: mannual patch, saliency maps, pretrianed SUP matrix
# use expanded saliency maps and bboxes to include surrunding points
# attack 3 types of objects: car, pedestrian, bicycle
# expand bboxes
# perturbation shape: saliency map top n voxels and other manual shapes such as X shape and half edges
# selection strategy: critical_first, random
# perturbation strategy: empty, ray
# save perturbed lidar

# several commands can be run simultaneously

'''
combined codes:
./data/kitti_dataset2.py
./data/nuscenes_dataset3.py

environment commands: 

. /vol/cuda/11.7.1/setup.sh
export PATH=/vol/bitbucket/${USER}/Anaconda3/bin/:$PATH
export CUDA_HOME=/vol/cuda/11.7.1
export C_INCLUDE_PATH=/vol/cuda/11.7.1/include:/vol/cuda/11.7.1/targets/x86_64-linux/include
export CPATH=/vol/cuda/11.7.1/include:/vol/cuda/11.7.1/targets/x86_64-linux/include
export PYTHONPATH=/vol/bitbucket/cy19/saliency_map_v2:$PYTHONPATH
source activate
conda activate env_saliency_map_v3
cd /vol/bitbucket/cy19/saliency_map_v2/second/

python ./attack/saliency_hiding.py --pert_shape 'saliency_map' --map_source_dataset 'kitti' --map_source_detector 'pp' --map_source_object 'Car' --top_n 1.0 --selection_level 'frustum_level' --selection_strategy 'random' --perturbation_strategy 'ray' --frustum_budget 10  --target_dataset 'kitti' --target_detector 'pp'  --target_object 'Car' --iou_thres 0.7 --conf_thres 0.7
python ./attack/saliency_hiding.py --pert_shape 'saliency_map' --map_source_dataset 'kitti' --map_source_detector 'pp' --map_source_object 'Car' --top_n 1.0 --selection_level 'frustum_level' --selection_strategy 'critical_first' --perturbation_strategy 'ray' --frustum_budget 10  --target_dataset 'kitti' --target_detector 'pp'  --target_object 'Car' --iou_thres 0.7 --conf_thres 0.7

'''
import os
import torch
import torchplus
import numpy as np
from pathlib import Path
from google.protobuf import text_format
import argparse
import pickle
import math
import random
from PIL import Image
from second.core import box_np_ops
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.pytorch.builder import (box_coder_builder, input_reader_builder, second_builder)
from second.protos import pipeline_pb2
from second.utils.progress_bar import ProgressBar

from nuscenes.utils.data_classes import Box # nuscenes format box

class VeloBox: # kitti format box
    def __init__(self, scene: int, classification: np.ndarray, boxes: np.ndarray, logits: np.array):
        self.scene = scene
        self.classification = classification
        self.boxes = boxes # [x, y, z, w, l, h, r]
        self.logits = logits

    def __str__(self):
        return " scene: {}, cls: {}, boxes: {}, logits: {}".format(
            self.scene, self.classification, self.boxes, self.logits)

def box_riou(target_gt_boxes, target_pred_boxes):
    # filter predicted bboxes by gt boxes based on max IOU
    gt_boxes_2d = np.delete(target_gt_boxes, [2,5], 1)
    # print('gt_boxes_2d: ', gt_boxes_2d)
    pred_boxes_2d = np.delete(target_pred_boxes, [2,5], 1)
    # print('pred_boxes_2d: ', pred_boxes_2d)
    IOUs = box_np_ops.riou_cc(gt_boxes_2d.astype(np.float64), pred_boxes_2d.astype(np.float64))
    # print('IOUs: ', IOUs)
    # # find the max_id among all elements
    # max_gt_pred_id = np.where(IOUs == np.amax(IOUs))
    # print('max_gt_pred_id: ', max_gt_pred_id)
    # max_gt_id = max_gt_pred_id[0]
    # max_pred_id = max_gt_pred_id[1]
    return IOUs

def check_cat_id(cls):
    if cls == 'car' or cls == 'Car':
        cat_id = 0
    if cls == 'pedestrian' or cls == 'Pedestrian':
        cat_id = 1
    if cls == 'bicycle' or cls == 'Cyclist':
        cat_id = 2
    return cat_id

def cart2sph(x, y, z):
    # Transform Cartesian to spherical coordinates
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def sph2cart(az, el, r):
    # Transform spherical to Cartesian coordinates
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def sort_by_column(inputArray, n):
    '''
    Sorting a Numpy Array by the nth column
    n: n-th column
    '''
    sortedArray = inputArray[inputArray[:,n].argsort()]
    return sortedArray

def random_bool(num_of_perturbations, num_of_empty_voxels):
    # randomly generate some empty voxels 
    empa = np.full(num_of_perturbations, 1, dtype=int)
    # randomly assign a number of '0' value to empa. '0' means unoccupied/removal
    random_index = random.sample(range(0, num_of_perturbations), num_of_empty_voxels)
    for i in random_index:
        empa[random_index] = 0

    return empa

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

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
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    # voxel_generator = voxel_builder.build(model_cfg.voxel_generator, device = torch.device("cuda:0"))
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

def select_top(saliency_path, top_percent = 1.0, map_size = [64, 32], extract_negative = False):
    # load the Image
    # print('loading saliency map...')
    img = Image.open(saliency_path)
    summarized_saliency_map = np.asarray(img)
    # print('summarized_saliency_map: ', summarized_saliency_map)
    
    if extract_negative:
        # extract negative channel
        extrated_channel = np.array(summarized_saliency_map[:,:,2]).reshape(map_size[0], map_size[1])
    else:
        # extract positive channel
        # extrated_channel = np.array(summarized_saliency_map[:,:,0]).reshape(map_size[0], map_size[1], 1)
        extrated_channel = np.array(summarized_saliency_map[:,:,0]).reshape(map_size[0], map_size[1])
    
    # sort all values
    extrated_channel_1d = extrated_channel.reshape(map_size[0] * map_size[1])
    # print('extrated_channel_1d: ', extrated_channel_1d)
    sorted_indices = extrated_channel_1d.argsort() # sort in ascending order
    # print('sorted_indices: ', sorted_indices)

    # find threshold based on top_percent
    top_num = int( (1-top_percent) * map_size[0] * map_size[1] )
    thres_index = sorted_indices[top_num]
    thres = extrated_channel_1d[thres_index]
    # print('top_num: ', top_num)
    # print('thres_index: ', thres_index)
    # print('thres: ', thres)

    # filter out smaller values
    mask = extrated_channel > thres
    filtered_extrated_channel = extrated_channel * mask

    return filtered_extrated_channel

def adaptive_voxelization(target_cloud, bbox_origin, dimension_lidar, yaw_lidar, map_size):
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
    print('dimension_lidar: ', dimension_lidar)
    patch_size_x, patch_size_y = map_size[0], map_size[1]
    voxel_x_step = bbox_l/patch_size_x
    voxel_y_step = bbox_w/patch_size_y
    step_kernel = [voxel_x_step, voxel_y_step]
    indices = np.array(rotated_points/step_kernel, dtype = np.int32)

    # print('=============== rotated voxelization ===================')
    # print('yaw_lidar: ', yaw_lidar)
    # print('target_points: ', target_points)
    # print('shifted_points: ', shifted_points)
    # print('rotated_points: ', rotated_points)
    # print('voxel_x_step: ', voxel_x_step, ', voxel_y_step: ', voxel_y_step)
    # print('indices: ', indices)

    # ensure there is no negative indices
    negative_check = indices < 0
    print('indices[negative_check]: ', indices[negative_check])

    assert len(np.where(negative_check == True)[0]) == 0

    return indices

def ray_shifting(point_to_be_shifted, shifting_distance):
    # point_to_be_shifted: (x,y,z,intensity)
    # shifting_distance: float, shifting distance in the ray direction.

    origin = (0,0,0)
    # Calculate delt x,y,z
    delt_x = point_to_be_shifted[0] - origin[0]
    delt_y = point_to_be_shifted[1] - origin[1]
    delt_z = point_to_be_shifted[2] - origin[2]

    az, el, r = cart2sph(delt_x, delt_y, delt_z)
    shifted_r = r + shifting_distance
    shifted_delt_x, shifted_delt_y, shifted_delt_z = sph2cart(az, el, shifted_r)

    shifted_x = shifted_delt_x + origin[0]
    shifted_y = shifted_delt_y + origin[1]
    shifted_z = shifted_delt_z + origin[2]
    shifted_intensity = point_to_be_shifted[3]

    shifted_point = (shifted_x, shifted_y, shifted_z, shifted_intensity)
    return shifted_point

def random_point_filter(cloud, perturbation_rate):
    '''
    randomly filter cloud points leaving some points not perturbing
    '''

    point_budget = np.int32(len(cloud) * perturbation_rate)
    is_target = random_bool(len(cloud), point_budget) == 0
    target_cloud = cloud[is_target]
    nontarget_cloud = cloud[np.logical_not(is_target)]

    return target_cloud, nontarget_cloud

def critical_point_selection(perturbation_strategy, target_cloud, points_contri, point_budget):
    # shift certain points with the guidance of saliency_map
    shifted_cloud = []
    not_shifted_cloud = []
    
    points_contri = np.reshape(points_contri,(-1, 1))
    print('points_contri: ', points_contri)
    cloud_with_contri = np.concatenate([target_cloud, points_contri], axis = 1)
    # sort new cloud by points_contri in ascending order
    sorted_cloud_with_contri = sort_by_column(cloud_with_contri, 4)
    print('sorted_cloud_with_contri: ', sorted_cloud_with_contri)
    sorted_target_cloud = sorted_cloud_with_contri[:, :4].tolist()
    # print('sorted_target_cloud: ', sorted_target_cloud)

    for point_index in range(point_budget):
        # print('shifting the {}-th critical point'.format(point_index))
        point = sorted_target_cloud.pop()
        if perturbation_strategy == 'ray':
            shifting_distance = random.uniform(-2.0, 2.0)
            shifted_point = ray_shifting(point, shifting_distance)
            shifted_cloud.append(shifted_point)
        elif perturbation_strategy == 'remove':
            # don't append the point in shifted_cloud
            pass
        
    not_shifted_cloud = sorted_target_cloud

    return shifted_cloud, not_shifted_cloud

def random_point_selection(perturbation_strategy, target_cloud, point_budget):
    # randomly shift certain points under point budget
    shifted_cloud = []
    not_shifted_cloud = []
    
    is_shifting = random_bool(len(target_cloud), point_budget)
    for point_index in range(len(target_cloud)):
        point = target_cloud[point_index]
        if is_shifting[point_index] == 0:
            if perturbation_strategy == 'ray':
                shifting_distance = random.uniform(-2.0, 2.0)
                shifted_point = ray_shifting(point, shifting_distance)
                shifted_cloud.append(shifted_point)
            elif perturbation_strategy == 'remove':
                # don't append the point in shifted_cloud
                pass
        else:
            not_shifted_cloud.append(point)

    return shifted_cloud, not_shifted_cloud

def critical_frustum_selection(perturbation_strategy, target_cloud, frustum_indices, points_contri, frustum_budget, frustum_drop_budget = 0):
    # shift certain points with the guidance of saliency_map
    shifted_cloud = []
    not_shifted_cloud = []
    
    points_contri = np.reshape(points_contri,(-1, 1))
    print('points_contri: ', points_contri)
    cloud_with_contri = np.concatenate([target_cloud, points_contri], axis = 1)
    print('cloud_with_contri: ', cloud_with_contri)

    frustums = []
    num_frustums = np.max(frustum_indices) + 1
    
    print('extracting frustums...')
    for fid in range(num_frustums):
        frustum = {}
        # print('===== extracting frustum {} ====='.format(fid))
        fid_mask = frustum_indices == fid
        frustum_cloud_with_contri = cloud_with_contri[fid_mask]
        # print('frustum_cloud_with_contri: ', frustum_cloud_with_contri)

        frustum['fid'] = fid
        frustum['points'] = frustum_cloud_with_contri[:,:4].tolist()
        frustum['contribution'] = np.sum(frustum_cloud_with_contri[:,4])
        frustums.append(frustum)

    # sort frustums by contribution in ascending order
    sorted_frustums = sorted(frustums, key=lambda x: x['contribution'])
    # print('sorted_frustums: ', sorted_frustums)

    # generate a list of frustum ids to drop (for ablation study)
    frustum_drop_ids = random.sample(range(0, num_frustums), frustum_drop_budget)
    
    print('num_frustums: ', num_frustums)
    for fid in range(frustum_budget):
        frustum = sorted_frustums.pop()
        # print('frustum points:', frustum['points'])
        print('perturbing critical frustum id: ', frustum['fid'])

        if fid in set(frustum_drop_ids):
            # if drop the frustum, then don't perturb
            not_shifted_cloud.extend(frustum['points'])
        else:
            # perturb each target frustum
            if perturbation_strategy == 'ray':
                # shift all points in frustum
                for point in frustum['points']:
                    shifting_distance = random.uniform(0, 40.0)
                    shifted_point = ray_shifting(point, shifting_distance)
                    shifted_cloud.append(shifted_point)
            elif perturbation_strategy == 'remove':
                # don't append the point in shifted_cloud
                pass
            
    # left frustums are not perturbed 
    for frustum in sorted_frustums:
        not_shifted_cloud.extend(frustum['points'])

    return shifted_cloud, not_shifted_cloud

def random_frustum_selection(perturbation_strategy, target_cloud, frustum_indices, frustum_budget):
    # randomly shift certain frustums under frustum budget
    shifted_cloud = []
    not_shifted_cloud = []
    
    frustums = []
    num_frustums = np.max(frustum_indices) + 1
    is_shifting = random_bool(num_frustums, frustum_budget)
    print('extracting frustums...')
    for fid in range(num_frustums):
        frustum = {}
        # print('===== extracting frustum {} ====='.format(fid))
        fid_mask = frustum_indices == fid
        frustum_cloud = target_cloud[fid_mask]
        # print('frustum_cloud: ', frustum_cloud)

        frustum['fid'] = fid
        frustum['points'] = frustum_cloud.tolist()
        frustums.append(frustum)

    for fid in range(num_frustums):
        frustum = frustums[fid]
        if is_shifting[fid] == 0:
            if perturbation_strategy == 'ray':
                # shift all points in frustum
                for point in frustum['points']:
                    shifting_distance = random.uniform(0, 40.0)
                    shifted_point = ray_shifting(point, shifting_distance)
                    shifted_cloud.append(shifted_point)
            elif perturbation_strategy == 'remove':
                # don't append the point in shifted_cloud
                pass
        else:
            not_shifted_cloud.extend(frustum['points'])

    return shifted_cloud, not_shifted_cloud

def fixed_frustum_selection(perturbation_strategy, target_cloud, frustum_indices, target_frustum_ids = []):
    # randomly shift certain frustums under frustum budget
    shifted_cloud = []
    not_shifted_cloud = []
    
    frustums = []
    num_frustums = np.max(frustum_indices) + 1
    print('extracting frustums...')
    for fid in range(num_frustums):
        frustum = {}
        # print('===== extracting frustum {} ====='.format(fid))
        fid_mask = frustum_indices == fid
        frustum_cloud = target_cloud[fid_mask]
        # print('frustum_cloud: ', frustum_cloud)

        frustum['fid'] = fid
        frustum['points'] = frustum_cloud.tolist()
        frustums.append(frustum)
    
    print('num_frustums: ', num_frustums)
    for fid in target_frustum_ids:
        # print('target frustum id: ', fid)
        frustum = frustums.pop(fid)
        print('perturbing fixed frustum id: ', frustum['fid'])
        # print(frustum)
        if perturbation_strategy == 'ray':
            # shift all points in frustum
            for point in frustum['points']:
                shifting_distance = random.uniform(0, 40.0)
                shifted_point = ray_shifting(point, shifting_distance)
                shifted_cloud.append(shifted_point)
        elif perturbation_strategy == 'remove':
            # don't append the point in shifted_cloud
            pass
    
    # left frustums are not perturbed 
    for frustum in frustums:
        not_shifted_cloud.extend(frustum['points'])

    return shifted_cloud, not_shifted_cloud

def check_saliency_map(voxel_coords, top_n_voxels, voxel_offset = [0, 0], map_size = [64, 32]):
    '''
    offset: xy coordinate offset. This can be used for ablation study.
    '''
    # check if the voxel falls in saliency_map
    id_x = voxel_coords[0] + voxel_offset[0]
    id_y = voxel_coords[1] + voxel_offset[1]

    if (0 <= id_x < map_size[0]) and (0 <= id_y < map_size[1]):
        contri = top_n_voxels[id_x][id_y] 
    else: 
        contri = -1000
    # check_res = contri > 0

    
    return contri

def check_shape(self, voxel_coords, map_size = [64, 32]):
    # check if the voxel falls in perturbed shape
    # pert_shape: string, shape of perturbation
    pert_shape = self.pert_shape
    patch_size_x, patch_size_y = map_size[0], map_size[1]

    # no voxels are perturbed
    if pert_shape == 'none':
        return False

    # perturbe the whole region
    if pert_shape == 'whole_box':
       return True

    # edge check
    if pert_shape == 'edges':
        ### edge thickness = 10 voxels ###
        edge_check_y = voxel_coords[1]<10 or voxel_coords[1]>patch_size_y - 11 # check voxel y index
        edge_check_x = voxel_coords[0]<10 or voxel_coords[0]>patch_size_x - 11
        check_res = edge_check_x or edge_check_y

    if pert_shape == 'corners':
        # perturb 10*10 voxels in each corner
        edge_check_y = voxel_coords[1]<10 or voxel_coords[1]>patch_size_y - 11 # check voxel y index
        edge_check_x = voxel_coords[0]<10 or voxel_coords[0]>patch_size_x - 11
        check_res = edge_check_x and edge_check_y

    if pert_shape == 'nearest_corner':
        # assume index = (0,0) is the nearest corner
        # perturb 10*10 voxels in the nearest corner
        nearest_corner_check = (voxel_coords[0] < 10 ) and (voxel_coords[1] < 10) 
        check_res = nearest_corner_check

    if pert_shape == 'X_shape':
        # thickness = 6 voxels

        p1 = np.array([0,0])
        p2 = np.array([0,patch_size_y])
        p3 = np.array([patch_size_x,patch_size_y])
        p4 = np.array([patch_size_x,0])
        
        d1 = np.linalg.norm(np.cross(p3-p1, p1-voxel_coords))/np.linalg.norm(p3-p1)
        d2 = np.linalg.norm(np.cross(p4-p2, p2-voxel_coords))/np.linalg.norm(p4-p2)

        x_check = (d1 <= 3) or (d2 <= 3)
        check_res = x_check

    if pert_shape == 'center':
        # remove 3/4 center voxels
        center_check_x = (voxel_coords[0] > patch_size_x/8 and voxel_coords[0] < 7 * patch_size_x/8)
        center_check_y = (voxel_coords[1] > patch_size_y/8 and voxel_coords[1] < 7 * patch_size_y/8)
        check_res = center_check_x and center_check_y

    if pert_shape == 'half_edges':
        region1 = voxel_coords[0]>=32 and voxel_coords[1]<10
        region2 = voxel_coords[0]<32 and voxel_coords[1]>patch_size_y - 11
        check_res = region1 or region2

    if pert_shape == 'L_shape':
        # perturb 10 voxels in L shape on the nearest corner
        edge_check_y = voxel_coords[1]<10
        edge_check_x = voxel_coords[0]<10
        check_res = edge_check_x or edge_check_y

    return check_res
def center_2d_distance(gt_box, pred_box) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box[:2]) - np.array(gt_box[:2]))

def center_3d_distance(gt_box, pred_box) -> float:
    """
    L2 distance between the box centers (xyz only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box[:3]) - np.array(gt_box[:3]))

def relative_2d_degree(points, origin): 
    '''
    given an origin, compuate relative 2D degree.
    points: (N, 3), target points, xyz
    origin: (,3), origin used to compute degree, xyz
    '''
    delta_xyz = points - origin 
    gradients = delta_xyz[:, 1] / delta_xyz[:, 0]
    degrees = np.rad2deg(np.arctan(gradients))
    return degrees


def check_attack(args, benign_res, poisoned_res):
    '''
    Check if there are unmatched benign res. 
    benign_res: detected results without attacks
    poisoned_res: detected results with attacks
    '''

    poisoned_benign_boxes_mask = []
    matched_poisoned_res = []
    for benign_scene in benign_res:
        for poisoned_scene in poisoned_res:
            # find scene match
            if benign_scene.scene == poisoned_scene.scene:
                # print('========')
                # print('benign_scene: ', benign_scene)
                # print('poisoned_scene: ', poisoned_scene)
                matched_poisoned_scene = VeloBox(scene=poisoned_scene.scene, classification=[], boxes=[], logits=[])
                for b_idx, benign_box in enumerate(benign_scene.boxes):
                    # only focus on non-empty bboxes
                    if len(benign_box) > 0:
                        is_poisoned = True
                        # Find match for each benign box
                        for p_idx, poisoned_box in enumerate(poisoned_scene.boxes):
                            compare_cat = poisoned_scene.classification[p_idx] == benign_scene.classification[b_idx]
                            compare_conf = poisoned_scene.logits[p_idx] > args.conf_thres
                            compare_iou = box_riou([benign_box], [poisoned_box])[0][0] > args.iou_thres
                            # if args.target_dataset == 'nusc':
                            #     compare_iou = center_2d_distance(benign_box, poisoned_box) < args.dist_thres
                            # check if the box and the check box overlap, two parameters can be used: center_distance and scale_iou
                            # compare_size = scale_iou(benign_box,poisoned_box)> scale_thres # normally IOU > 0.7
                            
                            if compare_cat and compare_conf and compare_iou:
                                # if the benign box find a match, then this benign box is not poisoned
                                is_poisoned = False
                                matched_poisoned_scene.classification.append(poisoned_scene.classification[p_idx])
                                matched_poisoned_scene.logits.append(poisoned_scene.logits[p_idx])
                                matched_poisoned_scene.boxes.append(poisoned_box)
                                break
                        # collect poison infomation of each benign box
                        poisoned_benign_boxes_mask.append(is_poisoned)
                # collect each valid poisoned scene
                # print('matched_poisoned_scene: ', matched_poisoned_scene)
                matched_poisoned_res.append(matched_poisoned_scene)
                # print('matched_poisoned_scene: ', matched_poisoned_scene)
    return poisoned_benign_boxes_mask, matched_poisoned_res

def perturbation(args, target_cloud, pillar_indices, expand_gt_corners_lidar, map_size = [64, 32]):
    # perturb certain points under point budget
    points_in_shape = []
    points_contri = []
    shifted_cloud = []
    not_shifted_cloud = []

    if args.pert_shape == 'saliency_map':
        top_n_voxels = select_top(saliency_path=args.saliency_path, top_percent=args.top_n, map_size=map_size, extract_negative=args.extract_negative)
        # print('top_n_voxels: ', top_n_voxels)

    for point, voxel_coords in zip(target_cloud, pillar_indices):
        # only consider voxels fall in perturbation shapes
        if args.pert_shape == 'saliency_map':
            contri = check_saliency_map(voxel_coords, top_n_voxels, args.voxel_offset, map_size = map_size)
            # inshape = contri > 0 # only consider positive contributions of saliency map
            inshape = True # consider all points in the bbox
            if inshape:
                points_contri.append(contri) 
        else:
            inshape = check_shape(args, voxel_coords, map_size)
        # print('voxels in shape: ', voxel_coords)

        if inshape:
            points_in_shape.append(point)            
        else:
            not_shifted_cloud.append(point)

    # print('points_in_shape: ', points_in_shape)
    if len(points_in_shape) > 0:

        if args.selection_level == 'point_level':
            # compare point_budget with actual point candidates
            num_perturbed_points = args.point_budget
            if len(points_in_shape) < num_perturbed_points:
                num_perturbed_points = len(points_in_shape) 

            if args.selection_strategy == 'critical_first':
                shifted_cloud, not_shifted_cloud_in_shape = critical_point_selection(args.perturbation_strategy, points_in_shape, points_contri, num_perturbed_points)
            if args.selection_strategy == 'random':
                shifted_cloud, not_shifted_cloud_in_shape = random_point_selection(args.perturbation_strategy, points_in_shape, num_perturbed_points)

        if args.selection_level == 'frustum_level':
            # frustum creation: divide the whole bbox (points) to small frustums
            lidar_origin = [0.5, 0.5, 0.0]

            # get boundary lines of the bbox: max and min gradient corners/lines
            # for eight corners, compute 2D gradients and get the max and min gradient corners
            corners_degrees = relative_2d_degree(expand_gt_corners_lidar[0], lidar_origin) # we use batch_size = 1, so chose the first bbox
            # max_corner_degrees = np.max(corners_degrees)
            # print('corners_degrees: ', corners_degrees) 
            points_in_shape = np.array(points_in_shape)
            points_degrees = relative_2d_degree(points_in_shape[:,:3], lidar_origin)
            # print('points_degrees: ', points_degrees)
            min_degree = np.min([np.min(corners_degrees), np.min(points_degrees)]) # ensure no negative frustum indices
            
            frustum_indices = np.array((points_degrees - min_degree)/args.frustum_degree, dtype = np.int32)
            print('points_in_shape: ', points_in_shape[:,:3])
            print('frustum_indices: ', frustum_indices) 

            # compare frustum_budget with actual frustum candidates
            frustum_budget = args.frustum_budget
            num_frustums = np.max(frustum_indices) + 1
            if num_frustums < frustum_budget:
                frustum_budget = num_frustums
            if args.selection_strategy == 'critical_first':
                shifted_cloud, not_shifted_cloud_in_shape = critical_frustum_selection(args.perturbation_strategy, points_in_shape, frustum_indices, points_contri, frustum_budget, args.frustum_drop_budget)
            elif args.selection_strategy == 'random':
                shifted_cloud, not_shifted_cloud_in_shape = random_frustum_selection(args.perturbation_strategy, points_in_shape, frustum_indices, frustum_budget)
            elif args.selection_strategy == 'fixed':
                target_frustum_ids = [np.int32((num_frustums - 1)/2)] # select the middle frustum
                shifted_cloud, not_shifted_cloud_in_shape = fixed_frustum_selection(args.perturbation_strategy, points_in_shape, frustum_indices, target_frustum_ids)

        not_shifted_cloud += not_shifted_cloud_in_shape

    # print('num perturbed points: ', len(target_cloud) - len(not_shifted_cloud))
    print('num perturbed points: ', len(shifted_cloud))
    print('num not perturbed points: ', len(not_shifted_cloud))
    # print('num not perturbed points: ', len(not_shifted_cloud))
    if args.perturbation_strategy == 'ray':
        assert len(shifted_cloud) + len(not_shifted_cloud) == len(target_cloud)

    return shifted_cloud, not_shifted_cloud

def hiding_attack(args,
                        config_path,
                        model_dir=None,
                        ckpt_path=None,
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

    # train_input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    net = build_network(model_cfg, measure_time=measure_time).to(device)
    target_assigner = net.target_assigner
    # voxel_generator = net.voxel_generator
    voxel_generator_in_dataloader = voxel_builder.build(model_cfg.voxel_generator)
    
    if ckpt_path is None:
        assert model_dir is not None
        print('try_restore_latest_checkpoints')
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)
    batch_size = batch_size or eval_input_cfg.batch_size
    dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator_in_dataloader,
        target_assigner=target_assigner)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    # path of saving perturbed pc
    attack_path = '/vol/bitbucket/cy19/saliency_map_v2/second/attack/{}_hiding_{}_{}'.format(args.pert_shape, args.target_dataset, args.target_object)
    data_save_dir = os.path.join(attack_path, 'map_source_{}_{}_top{}'.format(args.map_source_dataset, args.map_source_detector, args.top_n))
    if args.selection_level == 'point_level':
        perturbed_lidar_directory = os.path.join(data_save_dir, '{}_{}_selection_{}_perturbation_budget{}_rate{}'.format(args.selection_level, args.selection_strategy, args.perturbation_strategy, args.point_budget, args.perturbation_rate), 'velodyne')
    else:
        perturbed_lidar_directory = os.path.join(data_save_dir, '{}_{}_selection_{}_perturbation_budget{}_rate{}'.format(args.selection_level, args.selection_strategy, args.perturbation_strategy, args.frustum_budget, args.perturbation_rate), 'velodyne')
    
    perturbed_lidar_directory = check_folder(perturbed_lidar_directory)

    start_scene = args.start_scene
    end_scene = args.end_scene
    if args.end_scene == -1:
        end_scene = len(dataset)
    attack_obj_limit = 1
    print('saliency_path: ', args.saliency_path)
    print('target dataset: ', args.target_dataset)
    print('target detector: ', args.target_detector)
    print('target object: ', args.target_object)
    print('attack num: ', attack_obj_limit)
    print('start_scene: ', start_scene)
    print('end_scene: ', end_scene)
    
    # define region of interest
    if args.target_dataset == 'kitti':
        if args.target_object == 'Car':
            # roi = [5, 8, 0, 8]
            roi = args.roi
            # Average car bounding box size (2.6m*4.6m*1.7m)
            map_size = np.array([64, 32]) # [96, 48]
        if args.target_object == 'Pedestrian':
            roi = [0, 20, 0, 20]
            # Average ped bounding box size (0.65m*0.8m*1.7m) 
            map_size = np.array([8, 8])
        if args.target_object == 'Cyclist':
            roi = [0, 20, 0, 20]
            # Average cyl bounding box size (0.65m*1.6m*1.7m)
            map_size = np.array([16, 8])
    if args.target_dataset == 'nusc':
        if args.target_object == 'car':
            roi = [-8, 0, 5, 8]
            # Average car bounding box size (2.6m*4.6m*1.7m)
            map_size = np.array([64, 32]) # [96, 48]
        if args.target_object == 'pedestrian':
            roi = [-20, 0, 0, 20]
            # Average ped bounding box size (0.65m*0.8m*1.7m) 
            map_size = np.array([8, 8])
        if args.target_object == 'bicycle':
            roi = [-20, 0, 0, 20]
            # Average cyl bounding box size (0.65m*1.6m*1.7m)
            map_size = np.array([16, 8])

    map_size = np.round(map_size * args.box_size_factor).astype(int)
    print('roi: ', roi)
    print('map_size: ', map_size)
    # load top n% voxels of saliency_map.
    # top_n_voxels = select_top(saliency_path=args.saliency_path, top_percent=args.top_n, map_size=map_size)

    if args.restore_point_budgets:
        # load num_perturbed_points in perturbed_infos
        perturbed_infos_path = '/vol/bitbucket/cy19/saliency_map_v2/second/attack/saliency_map_hiding_kitti_Car/map_source_kitti_pp_top1.0/frustum_level_critical_first_selection_ray_perturbation_budget{}_rate1.0/velodyne/perturbed_infos.pkl'.format(args.frustum_budget)
        print('perturbed_infos_path: ', perturbed_infos_path)
        with open(perturbed_infos_path, 'rb') as f:
            all_perturbed_infos = pickle.load(f)
    else:
        all_perturbed_infos = []

    dataloader = iter(dataloader)
    all_target_gt_boxes = []
    all_pred_boxes = []
    perturbed_infos = []
    # skip scenes before start_scene
    for idx in range(start_scene):
        example = next(dataloader)
        print('skipping example{} : {}'.format(idx, example))
    # contiue saliency map aggregation
    for idx in range(start_scene, end_scene):
        example = next(dataloader)
        print('---------- attack lidar_points in example {} ----------'.format(idx))   
        print('example: ', example)
        # velo_path = example['metadata'][0]['velo_path'] # velo_path of kitti dataset
        lidar_points = example['lidar_points'][0]
        # print('lidar_point.shape: ', lidar_point.shape)
        gt_boxes_lidar = example['gt_boxes_lidar'][0]
        gt_boxes_names = example['gt_boxes_names'][0]

        # go through all gt boxes in the current scene
        # find valid boxes
        valid_gt_boxes_lidar = [] 
        valid_classification = []
        expand_valid_gt_boxes_lidar = []
        peruturbed_infos = []
        num_target_object = 0
        for j in range(len(gt_boxes_names)):
            # check by label
            type_check = gt_boxes_names[j] == args.target_object
            # check by region
            gt_box_lidar = gt_boxes_lidar[j] 
            region_check = box_np_ops.region_check(gt_box_lidar[0], gt_box_lidar[1], roi)
            # check by num of target object
            num_check = num_target_object < attack_obj_limit # only perturb one object each sweep

            if type_check and region_check and num_check:
                valid_gt_boxes_lidar.append(gt_boxes_lidar[j])
                valid_classification.append(gt_boxes_names[j])
                
                # expand the bbox by size
                expand_gt_box_lidar = np.zeros_like(gt_box_lidar)
                expand_gt_box_lidar[0]= gt_box_lidar[0] + args.box_position_offset[0]
                expand_gt_box_lidar[1]= gt_box_lidar[1] + args.box_position_offset[1]
                expand_gt_box_lidar[2]= gt_box_lidar[2] + args.box_position_offset[2]
                expand_gt_box_lidar[3]= gt_box_lidar[3] * args.box_size_factor
                expand_gt_box_lidar[4]= gt_box_lidar[4] * args.box_size_factor
                expand_gt_box_lidar[5]= gt_box_lidar[5] * args.box_size_factor
                expand_gt_box_lidar[6]= gt_box_lidar[6] + np.deg2rad(args.box_rotation_offset)
                expand_valid_gt_boxes_lidar.append(expand_gt_box_lidar)

                num_target_object += 1
        
        valid_gt_boxes_lidar = np.array(valid_gt_boxes_lidar, dtype = np.float64)
        expand_valid_gt_boxes_lidar = np.array(expand_valid_gt_boxes_lidar, dtype = np.float64)
        print('valid_gt_boxes_lidar: ', valid_gt_boxes_lidar)
        print('expand_valid_gt_boxes_lidar: ', expand_valid_gt_boxes_lidar)
        all_target_gt_boxes.append( VeloBox(idx, valid_classification, valid_gt_boxes_lidar, logits=[]) )

        if len(expand_valid_gt_boxes_lidar) > 0:
            # extract points in bbox
            gt_point_mask = box_np_ops.points_in_rbbox(lidar_points, expand_valid_gt_boxes_lidar)
            print('gt_point_mask: ', gt_point_mask)
            target_mask = gt_point_mask.any(-1)
            bg_mask = np.logical_not(target_mask)
            print('target_mask: ', target_mask)
            print('bg_mask: ', bg_mask)
            # divide the whole scene into 3 groups     
            bg_cloud = lidar_points[bg_mask] 
            points_in_box = lidar_points[target_mask]
            print('num of points in raw lidar: ', len(lidar_points))
            print('num of points in background: ', len(bg_cloud))  
            print('num of points in box: ', len(points_in_box))  
            target_points_in_box, nontarget_points_in_box = random_point_filter(points_in_box, args.perturbation_rate)
            target_points_in_box = np.array(target_points_in_box)
            print('num of target points in box: ', len(target_points_in_box))  
            print('num of nontarget points in box: ', len(nontarget_points_in_box))  

            if len(target_points_in_box) > 0:
                # find the nearest corner of the target bbox
                # since other format was converted to KITTI format before training and inference, all bbox operations are in kitti fasion.
                expand_gt_corners_lidar = box_np_ops.rbbox3d_to_corners(expand_valid_gt_boxes_lidar, origin=[0.5, 0.5, 0.0], axis=2)
                print('gt corners lidar: ', expand_gt_corners_lidar)
                expand_nearest_corner = expand_gt_corners_lidar[0, 4, 0:3] # we use batch_size = 1, so chose the first bbox, fifth corner (x, y, z)
                print('expand_nearest_corner: ', expand_nearest_corner)

                # change the start axis of bbox rotation from  positive y axis to positive x axis(forward)
                r_positive_y = expand_valid_gt_boxes_lidar[0][6] - np.pi/2
                while r_positive_y < -np.pi:
                    r_positive_y += (np.pi * 2)
                while r_positive_y > np.pi:
                    r_positive_y -= (np.pi * 2)

                # Adaptive voxelization: project all bbox(also points in the bbox) to the same size and rotation. This operation is used for aligning bboxes with standard saliency map
                pillar_indices = adaptive_voxelization(target_points_in_box, expand_nearest_corner, expand_valid_gt_boxes_lidar[0][3:6], r_positive_y, map_size)
                print('indices shape: ', pillar_indices.shape)

                if args.restore_point_budgets:
                    # print('all_perturbed_infos: ', all_perturbed_infos)
                    for perturbed_info in all_perturbed_infos:
                        if perturbed_info['scene'] == idx:
                            print('perturbed_info: ', perturbed_info)
                            args.point_budget = perturbed_info['point_budget']

                shifted_cloud, not_shifted_cloud = perturbation(args, target_points_in_box, pillar_indices, expand_gt_corners_lidar, map_size)

                perturbed_info = {}
                perturbed_info['scene'] = idx
                perturbed_info['point_budget'] = len(shifted_cloud)
                perturbed_infos.append(perturbed_info)
                # print(perturbed_infos)
                print('target_points_in_box: ', target_points_in_box)

                perturbed_lidar_points = shifted_cloud + not_shifted_cloud + nontarget_points_in_box.tolist()  + bg_cloud.tolist() 
                print('num of points in perturbed lidar:', len(perturbed_lidar_points))
            else:
                # when there is no points inside the bbox
                perturbed_lidar_points = target_points_in_box.tolist() + nontarget_points_in_box.tolist()  + bg_cloud.tolist()

            if args.perturbation_strategy == 'ray':
                assert len(perturbed_lidar_points) == len(lidar_points), 'points missing in perturbation'

            # save perturbed lidars
            if args.save_lidar:
                perturbed_lidar_save = perturbed_lidar_directory + '/' + velo_path.name
                np.array(perturbed_lidar_points, dtype=np.float32).tofile(perturbed_lidar_save)
                print('perturbed_lidar saved to: ', perturbed_lidar_save)
                assert False

            # recreate voxels based on perturbed lidar
            perturbed_example = example
            res = voxel_generator_in_dataloader.generate(np.array(perturbed_lidar_points)) 
            voxels = res["voxels"]
            coordinates = res["coordinates"]
            num_points = res["num_points_per_voxel"]
            num_voxels = np.array([[voxels.shape[0]]], dtype=np.int64)

            new_coordinates = torch.cat([torch.zeros(size = (len(coordinates),1), dtype = torch.int32), coordinates], axis = 1)
            # print('voxelize perturbed points: ')
            # print('voxels: ', voxels)
            # print('coordinates: ', new_coordinates)
            # print('num_points: ', num_points)
            # print('num_voxels: ', num_voxels)

            # create new example with perturbations
            perturbed_example['voxels'] = voxels
            perturbed_example['coordinates'] = new_coordinates
            perturbed_example['num_points'] = num_points
            perturbed_example['num_voxels'] = num_voxels

            print('---------- get predictions ----------')    
            perturbed_example = example_convert_to_torch(perturbed_example, torch.float32, device)
            pred_dict= net(perturbed_example)
            print('pred_dict: ', pred_dict)

            # retrieve bboxes
            pred_boxes_lidar = pred_dict[0]['box3d_lidar'].detach().cpu().numpy()
            label_preds = pred_dict[0]['label_preds'].detach().cpu().numpy()
            logits = pred_dict[0]['scores'].detach().cpu().numpy()
            # velo_path = pred_dict[0]['metadata']['velo_path']
            # print('velo_path: ', velo_path)
            # print('pred_boxes_lidar: ', pred_boxes_lidar)
            unique_pred_boxes_lidar, unique_indices = np.unique(pred_boxes_lidar, return_index=True, axis=0)
            # print('unique_indices: ', unique_indices)
            # print('unique_pred_boxes_lidar: ', unique_pred_boxes_lidar)
            # print('pred_boxes_lidar[unique_indices]: ', pred_boxes_lidar[unique_indices])

            assert len(label_preds) == len(pred_boxes_lidar)
            assert len(label_preds) == len(logits)
            # pred_boxes = VeloBox(idx, label_preds, pred_boxes_lidar, logits)
            pred_boxes = VeloBox(idx, label_preds[unique_indices], pred_boxes_lidar[unique_indices], logits[unique_indices])
            print('pred_boxes: ', pred_boxes)
            all_pred_boxes.append(pred_boxes)

    # save attacked gt bboxes
    res_path = os.path.join(attack_path, 'attacked_gt_bboxes.pkl')
    with open(res_path, 'wb') as f:
        pickle.dump(all_target_gt_boxes, f)    

    # save predictions
    res_path = os.path.join(perturbed_lidar_directory, '{}_pred_bboxes.pkl'.format(args.target_detector))
    with open(res_path, 'wb') as f:
        pickle.dump(all_pred_boxes, f)    

    # save perturbed infos
    perturbed_infos_path = os.path.join(perturbed_lidar_directory, 'perturbed_infos.pkl')
    with open(perturbed_infos_path, 'wb') as f:
        pickle.dump(perturbed_infos, f)    

    print('---------- load bboxes ----------')
    # load gt boxes
    gt_boxes_path = os.path.join(attack_path, 'attacked_gt_bboxes.pkl')
    print('gt_boxes_path: ', gt_boxes_path)
    with open(gt_boxes_path, 'rb') as f:
        all_target_gt_boxes = pickle.load(f)
    # select gt boxes
    selected_gt_boxes = []
    for scene_gt_boxes in all_target_gt_boxes:
        if len(scene_gt_boxes.boxes) > 0:
            # print('scene_gt_boxes:')
            # print(scene_gt_boxes.scene)
            # print(scene_gt_boxes.classification)
            # print(scene_gt_boxes.boxes)
            gt_cat_ids = []
            for gt_cls in scene_gt_boxes.classification:
                gt_cat_ids.append(check_cat_id(gt_cls))
            scene_gt_boxes.classification = gt_cat_ids
            selected_gt_boxes.append(scene_gt_boxes)

    # load benign boxes
    if args.target_dataset == 'kitti':
        benign_boxes_path = '/vol/bitbucket/cy19/saliency_map_v2/second/data/kitti/training/velodyne_raw'
    if args.target_dataset == 'nusc':
        benign_boxes_path = '/vol/bitbucket/cy19/saliency_map_v2/second/data/v1.0-mini/velodyne_raw/LIDAR_TOP'
    benign_boxes_path = os.path.join(benign_boxes_path, '{}_pred_bboxes.pkl'.format(args.target_detector))
    print('benign_boxes_path: ', benign_boxes_path)
    with open(benign_boxes_path, 'rb') as f:
        all_benign_boxes = pickle.load(f)

    print('---------- Calculate ASR ----------')    
    print('----- Benign calculation -----')
    # calculate ASR under benign scenario  
    gt_benign_boxes_mask, matched_benign_boxes = check_attack(args, selected_gt_boxes, all_benign_boxes)
    success = np.sum(gt_benign_boxes_mask)
    all_target = len(gt_benign_boxes_mask)
    # print('ASR(benign) = {}/{}({:.2f}%)'.format(success, all_target, success/all_target*100))

    print('----- True calculation -----')
    # compare benign bboxes with poisoned bboxes
    benign_poisoned_boxes_mask, matched_pred_boxes = check_attack(args, matched_benign_boxes, all_pred_boxes)
    success = np.sum(benign_poisoned_boxes_mask)
    all_target = len(benign_poisoned_boxes_mask)
    print('ASR(true) = {}/{}({:.2f}%)'.format(success, all_target, success/all_target*100))

    print('----- End-to-end calculation -----')
    # compare gt bboxes with poisoned bboxes
    gt_poisoned_boxes_mask, matched_pred_boxes = check_attack(args, selected_gt_boxes, all_pred_boxes)
    success = np.sum(gt_poisoned_boxes_mask)
    all_target = len(gt_poisoned_boxes_mask)
    # print('ASR(end) = {}/{}({:.2f}%)'.format(success, all_target, success/all_target*100))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_scene", "--s", type=int, default=0)
    parser.add_argument("--end_scene", "--n", type=int, default=-1)

    parser.add_argument("--pert_shape", default="saliency_map", type=str, help="the shape of perturbation")
    parser.add_argument("--top_n", default=1.0, type=float, help="top n percent of saliency map")
    parser.add_argument("--selection_level", default="frustum_level", type=str, help="point or frustum level")
    parser.add_argument("--selection_strategy", default="random", type=str, help="random or critical_first")
    parser.add_argument("--perturbation_strategy", default="ray", type=str, help="ray or empty")
    parser.add_argument("--perturbation_rate", default=1.0, type=float, help="point perturbation rates")
    parser.add_argument("--point_budget", "--p", type=int, default=10, help="point budget of pertubation")
    parser.add_argument("--frustum_budget", type=int, default=10, help = "num of frustums to perturb")
    parser.add_argument("--frustum_degree", type=float, default=1.0, help="degree step to create frustums")
    
    parser.add_argument("--map_source_dataset", default="kitti", type=str, help="source dataset of saliency map")
    parser.add_argument("--map_source_detector", default="second", type=str, help="source detector of saliency map")
    parser.add_argument("--map_source_object", default="Car", type=str, help="source object of saliency map")
    parser.add_argument("--map_source_basemap", type=str, default='nearest_corner')
    parser.add_argument("--map_source_igsteps", default=25, type=int, help="source IG steps of saliency map")
    parser.add_argument("--target_dataset", type=str, default="kitti")
    parser.add_argument("--target_detector", type=str, default="second")
    parser.add_argument("--target_object", type=str, default="Car", help = "Car, Pedestrian, Cyclist")
    parser.add_argument("--conf_thres", type=float, default=0.7, help="confidence threshold to measure bbox")
    parser.add_argument("--iou_thres", type=float, default=0.7, help="iou threshold to measure two bboxes in kitti")
    parser.add_argument("--dist_thres", type=float, default=0.5, help="iou threshold to measure two bboxes in nusc")
    parser.add_argument("--saliency_path", default="", type=str, help="path of saliency map")

    parser.add_argument("--box_size_factor", type=float, default=1.5, help = "expand bbox size to include surrounding points")
    parser.add_argument("--box_rotation_offset", type=float, default=0.0, help="bbox rotation degree offset for ablation study")
    parser.add_argument("--box_position_offset", type=float, nargs="+", default=[0.0,0.0,0.0], help="bbox position offset to perform ablation study")
    parser.add_argument("--voxel_offset", type=int, nargs="+", default=[0,0], help="voxelized coord offset to perform ablation study")
    parser.add_argument("--frustum_drop_budget", type=int, default=0, help="number of frustums to drop to perform ablation study")
    parser.add_argument("--roi", type=int, nargs="+", default=[5,8,0,8], help="region of interest")
    
    parser.add_argument("--save_lidar", action="store_true", help="whether save the perturbed lidar scene") # store_true: True is set only when triggered
    parser.add_argument("--restore_point_budgets", action = "store_true", help="whether restore point budgets to perturb")
    parser.add_argument("--extract_negative", action = "store_true", help="whether use the positive channel of saliency maps for ablation study")
    
    args = parser.parse_args()

    # print(args.voxel_offset)
    # print(args.box_position_offset)
    # assert False
    # set model and configs
    if args.target_dataset == 'nusc':
        # # check sample_path 
        # sample_path = Path('/vol/bitbucket/cy19/saliency_map_v2/second/data/v1.0-mini/samples/LIDAR_TOP')
        # if sample_path.is_symlink():
        #     sample_path.unlink()
        # # check sweep_path
        # sweep_path = Path('/vol/bitbucket/cy19/saliency_map_v2/second/data/v1.0-mini/sweeps/LIDAR_TOP')
        # if sweep_path.is_symlink():
        #     sweep_path.unlink()
        # # link sample_path and sweep_path to raw dataset
        # data_root = '/vol/bitbucket/cy19/saliency_map_v2/second'
        # raw_dataset_path = data_root + 'data/v1.0-mini/velodyne_raw/LIDAR_TOP' # raw nuscense LIDAR_TOP(samples + sweeps)
        # sample_path.symlink_to(raw_dataset_path)
        # sweep_path.symlink_to(raw_dataset_path)

        # set model and configs
        if args.target_detector == 'pp' : 
            config_path = "./configs/nuscenes/all.pp.mida.trainval.config" # pp_model_for_nuscenes_pretrain, only single class such as car
            ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/nuscenes/all_pp_mida_trainval/voxelnet-58650.tckpt" # pretrained model
            model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/nuscenes/all_pp_mida_trainval" # for restore the latest checkpoint(.json)
        if args.target_detector == 'second':
            config_path = "./configs/nuscenes/all.second.fhd.config"
            ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/nuscenes/all_second_fhd_trainval/voxelnet-58650.tckpt" # pretrained model
            model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/nuscenes/all_second_fhd_trainval" # for restore the latest checkpoint(.json)

    elif args.target_dataset == 'kitti':
        # # check dataset_path 
        # dataset_path = Path('/vol/bitbucket/cy19/saliency_map_v2/second/data/kitti/training/velodyne')
        # if dataset_path.is_symlink():
        #     dataset_path.unlink()
        # # link dataset_path to raw dataset 
        # data_root = '/vol/bitbucket/cy19/saliency_map_v2/second'
        # raw_dataset_path = data_root + '/data/kitti/training/velodyne_raw'
        # dataset_path.symlink_to(raw_dataset_path)

        # set model and configs
        if args.target_detector == 'pp':
            if args.target_object == 'Car':
                config_path = "./configs/pointpillars/car/xyres_16.config" 
                ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/pp_car/voxelnet-296960.tckpt" # pretrained model
                model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/pp_car" # for restore the latest checkpoint(.json)
                args.target_detector = 'pp_car'
            if args.target_object == 'Pedestrian' or args.target_object == 'Cyclist':
                config_path = "./configs/pointpillars/ped_cycle/xyres_16.config" 
                ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/pp_ped_cycle/voxelnet-296960.tckpt" # pretrained model
                model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/pp_ped_cycle" # for restore the latest checkpoint(.json)
                args.target_detector = 'pp_ped_cycle'
        if args.target_detector == 'second':
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
                args.target_detector = 'second_car'
            if args.target_object == 'Pedestrian' or args.target_object == 'Cyclist':
                config_path = "./configs/people.fhd.config" 
                ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_people/voxelnet-30950.tckpt" # pretrained model
                model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_people" # for restore the latest checkpoint(.json)
                args.target_detector = 'second_ped_cycle'
    # configure path of saliency_map
    if args.map_source_igsteps == 25:
        # old path of saliency maps (IGsteps = 25)
        args.saliency_path = '/vol/bitbucket/cy19/saliency_map_v2/second/results/v5.1/{}/basemap_{}/map_{}_{}/summarized_saliency_map_3channels_final.png'.format(args.map_source_dataset, args.map_source_basemap, args.map_source_detector, args.map_source_object)
        # args.saliency_path = '/vol/bitbucket/cy19/saliency_map_v2/second/results/v5.1/{}/basemap_nearest_corner/second(all.fhd.config)/map_{}_{}/summarized_saliency_map_3channels_final.png'.format(args.map_source_dataset, args.map_source_detector, args.map_source_object)
    if args.map_source_basemap == 'zeros' or args.map_source_igsteps != 25:
        args.saliency_path = '/vol/bitbucket/cy19/saliency_map_v2/second/results/v5.1/{}/basemap_{}/map_{}_{}/IGstep{}/summarized_saliency_map_3channels_final.png'.format(args.map_source_dataset, args.map_source_basemap, args.map_source_detector, args.map_source_object, args.map_source_igsteps)
    
    # print('extract_negative: ', args.extract_negative)
    # print('save_lidar: ', args.save_lidar)

    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    hiding_attack(
        args,
        config,
        model_dir=str(model_dir),
        ckpt_path=str(ckpt_name),
        batch_size=1,
        measure_time=True
        )

