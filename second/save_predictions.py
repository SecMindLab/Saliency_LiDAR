# map source datasest: nuscenes and kitti
# map source detector: second and pointpillars
# target datasest: nuscenes and kitti
# target detector: second and pointpillars
# target object: car, pedestrian, cyclist
# replace adverarial dataset with benign dataset
# save detection results
# calculate ASR

# wait for saliency map and perturbed pcs
# caution: changing symbolic link would influence the generation of saliency maps. Run this code after all saliency maps and poisoned datasets are generated.
# run each command one by one in case of interference


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
# cd /vol/bitbucket/cy19/saliency_map_v2/second/attack/saliency_map_hiding_kitti_Car

### kitti ###
## benign outputs
python save_predictions.py --target_dataset 'kitti' --target_detector 'pp'  --detector_object 'Car' -p '/data/kitti/training/velodyne_raw' 
python save_predictions.py --target_dataset 'kitti' --target_detector 'pp'  --detector_object 'Pedestrian' -p '/data/kitti/training/velodyne_raw' # also predict Cyclist
python save_predictions.py --target_dataset 'kitti' --target_detector 'second' -p '/data/kitti/training/velodyne_raw' 

## attack second
# Car
python save_predictions.py --target_dataset 'kitti' --target_detector 'second'  -p '/attack/saliency_map_hiding_kitti_Car/map_source_kitti_second_top0.3/critical_first_selection_random_shifting_pb200/velodyne'
# Cyclist
python save_predictions.py --target_dataset 'kitti' --target_detector 'second'  -p '/attack/saliency_map_hiding_kitti_Cyclist/map_source_kitti_second_top0.3/critical_first_selection_random_shifting_pb100/velodyne/'
# Pedestrian
python save_predictions.py --target_dataset 'kitti' --target_detector 'second'  -p '/attack/saliency_map_hiding_kitti_Pedestrian/map_source_kitti_second_top0.3/critical_first_selection_random_shifting_pb50/velodyne/'
## attack pp
# Car
python save_predictions.py --target_dataset 'kitti' --target_detector 'pp' --detector_object 'Car' -p '/attack/saliency_map_hiding_kitti_Car/map_source_kitti_second_top0.3/critical_first_selection_random_shifting_pb200/velodyne'
# Cyclist
python save_predictions.py --target_dataset 'kitti' --target_detector 'pp' --detector_object 'Cyclist' -p '/attack/saliency_map_hiding_kitti_Cyclist/map_source_kitti_second_top0.3/critical_first_selection_random_shifting_pb100/velodyne/'
# Pedestrian
python save_predictions.py --target_dataset 'kitti' --target_detector 'pp' --detector_object 'Pedestrian' -p '/attack/saliency_map_hiding_kitti_Pedestrian/map_source_kitti_second_top0.3/critical_first_selection_random_shifting_pb50/velodyne/'

### nusc ###
## benign outputs
python save_predictions.py --target_dataset 'nusc' --target_detector 'pp' -p '/data/v1.0-mini/velodyne_raw/LIDAR_TOP'
python save_predictions.py --target_dataset 'nusc' --target_detector 'second' -p '/data/v1.0-mini/velodyne_raw/LIDAR_TOP' 

## attack second
# car
python save_predictions.py --target_dataset 'nusc' --target_detector 'second' -p '/attack/saliency_map_hiding_nusc_car/map_source_nusc_second_top0.3/critical_first_selection_random_shifting_pb200/velodyne/LIDAR_TOP/'
# bicycle
python save_predictions.py --target_dataset 'nusc' --target_detector 'second' -p '/attack/saliency_map_hiding_nusc_bicycle/map_source_nusc_second_top0.3/critical_first_selection_random_shifting_pb100/velodyne/LIDAR_TOP/'
# pedestrian
python save_predictions.py --target_dataset 'nusc' --target_detector 'second' -p '/attack/saliency_map_hiding_nusc_pedestrian/map_source_nusc_second_top0.3/critical_first_selection_random_shifting_pb50/velodyne/LIDAR_TOP/'
## attack pp
# car
python save_predictions.py --target_dataset 'nusc' --target_detector 'pp' -p '/attack/saliency_map_hiding_kitti_car/map_source_nusc_second_top0.3/critical_first_selection_random_shifting_pb200/velodyne/LIDAR_TOP/'
# bicycle
python save_predictions.py --target_dataset 'nusc' --target_detector 'pp' -p '/attack/saliency_map_hiding_kitti_bicycle/map_source_nusc_second_top0.3/critical_first_selection_random_shifting_pb100/velodyne/LIDAR_TOP/'
# pedestrian
python save_predictions.py --target_dataset 'nusc' --target_detector 'pp' -p '/attack/saliency_map_hiding_kitti_pedestrian/map_source_nusc_second_top0.3/critical_first_selection_random_shifting_pb50/velodyne/LIDAR_TOP/'

'''
import os
import torch
import torchplus
import numpy as np
from pathlib import Path
from google.protobuf import text_format
import argparse
import pickle

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

def check_cat_id(cls):
    if cls == 'car' or cls == 'Car':
        cat_id = 0
    if cls == 'pedestrian' or cls == 'Pedestrian':
        cat_id = 1
    if cls == 'bicycle' or cls == 'Cyclist':
        cat_id = 2
    return cat_id

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

def check_attack(benign_res, poisoned_res, dist_thres, conf_thres):
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
                print('========')
                print('benign_scene: ', benign_scene)
                print('poisoned_scene: ', poisoned_scene)
                matched_poisoned_scene = VeloBox(scene=poisoned_scene.scene, classification=[], boxes=[], logits=[])
                for b_idx, benign_box in enumerate(benign_scene.boxes):
                    # only focus on non-empty bboxes
                    if len(benign_box) > 0:
                        print('benign_box: ', benign_box)
                        is_poisoned = True
                        # Find match for each benign box
                        for p_idx, poisoned_box in enumerate(poisoned_scene.boxes):
                            compare_cat = poisoned_scene.classification[p_idx] == benign_scene.classification[b_idx]
                            compare_conf = poisoned_scene.logits[p_idx] > conf_thres
                            # check if the box and the check box overlap, two parameters can be used: center_distance and scale_iou
                            # compare_size = scale_iou(benign_box,poisoned_box)>0.5 # normally IOU > 0.7
                            compare_center = center_2d_distance(benign_box, poisoned_box)< dist_thres
                            if compare_cat and compare_conf and compare_center:
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
    return poisoned_benign_boxes_mask, matched_poisoned_res

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

def prediction(args,
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
    target_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator_in_dataloader,
        target_assigner=target_assigner)
    target_dataloader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    print("Generate output labels...")
    bar = ProgressBar()
    bar.start((len(target_dataset) + batch_size - 1) // batch_size)

    start_scene = args.start_scene
    end_scene = args.end_scene
    if args.end_scene == -1:
        end_scene = len(target_dataset)
    print('start_scene: ', start_scene)
    print('end_scene: ', end_scene)

    dataloader = iter(target_dataloader)
    all_pred_boxes = []
    # skip scenes before start_scene
    for idx in range(start_scene):
        example = next(dataloader)
        print('skipping example{} : {}'.format(idx, example))
    # contiue saliency map aggregation
    for idx in range(start_scene, end_scene):
        example = next(dataloader)
        print('example {}: {}'.format(idx, example))
        # velo_path = example['metadata'][0]['velo_path']
        # print('velo_path: ', velo_path)

        example = example_convert_to_torch(example, torch.float32, device)
        pred_dict= net(example)
        print('pred_dict: ', pred_dict)
        
        # retrieve bboxes
        pred_boxes_lidar = pred_dict[0]['box3d_lidar'].detach().cpu().numpy()
        label_preds = pred_dict[0]['label_preds'].detach().cpu().numpy()
        logits = pred_dict[0]['scores'].detach().cpu().numpy()
        # velo_path = pred_dict[0]['metadata']['velo_path']
        # print('velo_path: ', velo_path)
        print('pred_boxes_lidar: ', pred_boxes_lidar)
        # unique_pred_boxes_lidar, unique_indices = np.unique(pred_boxes_lidar, return_index=True, axis=0)
        # print('unique_indices: ', unique_indices)
        # print('unique_pred_boxes_lidar: ', unique_pred_boxes_lidar)
        # print('pred_boxes_lidar[unique_indices]: ', pred_boxes_lidar[unique_indices])


        assert len(label_preds) == len(pred_boxes_lidar)
        assert len(label_preds) == len(logits)
        pred_boxes = VeloBox(idx, label_preds, pred_boxes_lidar, logits)
        # pred_boxes = VeloBox(idx, label_preds[unique_indices], pred_boxes_lidar[unique_indices], logits[unique_indices])
        # print(pred_boxes)
        all_pred_boxes.append(pred_boxes)
    
    # save predictions
    res_path = os.path.join(poisoned_dataset_path, '{}_pred_bboxes.pkl'.format(args.target_detector))
    with open(res_path, 'wb') as f:
        pickle.dump(all_pred_boxes, f)    
    
    print('---------- load bboxes ----------')
    # load gt boxes
    gt_boxes_path = os.path.join('/', *poisoned_dataset_path.split('/')[:8], 'attacked_gt_bboxes.pkl')
    print('gt_boxes_path: ', gt_boxes_path)
    with open(gt_boxes_path, 'rb') as f:
        all_gt_boxes = pickle.load(f)
    # select gt boxes
    selected_gt_boxes = []
    for scene_gt_boxes in all_gt_boxes:
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

    # load prediction results
    res_path = os.path.join(poisoned_dataset_path, '{}_pred_bboxes.pkl'.format(args.target_detector))
    with open(res_path, 'rb') as f:
        all_pred_boxes = pickle.load(f)
    print('poisoned_dataset_path: ', res_path)
    # print('all_pred_boxes: ',all_pred_boxes)

    print('---------- Calculate ASR ----------')    
    print('----- Benign calculation -----')
    # calculate ASR under benign scenario  
    gt_benign_boxes_mask, matched_benign_boxes = check_attack(selected_gt_boxes, all_benign_boxes, args.dist_thres, args.conf_thres)
    success = np.sum(gt_benign_boxes_mask)
    all_target = len(gt_benign_boxes_mask)
    print('ASR(benign) = {}/{}({:.2f}%)'.format(success, all_target, success/all_target*100))

    print('----- True calculation -----')
    # compare benign bboxes with poisoned bboxes
    benign_poisoned_boxes_mask, matched_pred_boxes = check_attack(matched_benign_boxes, all_pred_boxes, args.dist_thres, args.conf_thres)
    success = np.sum(benign_poisoned_boxes_mask)
    all_target = len(benign_poisoned_boxes_mask)
    print('ASR(true) = {}/{}({:.2f}%)'.format(success, all_target, success/all_target*100))

    print('----- End-to-end calculation -----')
    # compare gt bboxes with poisoned bboxes
    gt_poisoned_boxes_mask, matched_pred_boxes = check_attack(selected_gt_boxes, all_pred_boxes, args.dist_thres, args.conf_thres)
    success = np.sum(gt_poisoned_boxes_mask)
    all_target = len(gt_poisoned_boxes_mask)
    print('ASR(end) = {}/{}({:.2f}%)'.format(success, all_target, success/all_target*100))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pert_shape", type=str, default='saliency_map')
    parser.add_argument("--map_source_detector", type=str, default='pp')
    parser.add_argument("--map_source_dataset", type=str, default='kitti')
    parser.add_argument("--target_detector", type=str, default='pp')
    parser.add_argument("--target_dataset", type=str, default='kitti')
    parser.add_argument("--detector_object", type=str, default='Car', help='help select specific detector for pointpillar')
    parser.add_argument("--poisoned_dataset_path", '-p', type=str, default='/data/kitti/training/velodyne_raw')
    parser.add_argument("--start_scene", type=int, default=0)
    parser.add_argument("--end_scene", type=int, default=-1)
    parser.add_argument("--dist_thres", type=float, default=0.5, help='distance threshold to measure two bboxes')
    parser.add_argument("--conf_thres", type=float, default=0.2, help='confidence threshold to measure two bboxes')
    args = parser.parse_args()

    if args.target_dataset == 'nusc':
        # check sample_path 
        sample_path = Path('/vol/bitbucket/cy19/saliency_map_v2/second/data/v1.0-mini/samples/LIDAR_TOP')
        if sample_path.is_symlink():
            sample_path.unlink()
        # check sweep_path
        sweep_path = Path('/vol/bitbucket/cy19/saliency_map_v2/second/data/v1.0-mini/sweeps/LIDAR_TOP')
        if sweep_path.is_symlink():
            sweep_path.unlink()
        # link sample_path and sweep_path to poisoned dataset
        data_root = '/vol/bitbucket/cy19/saliency_map_v2/second'
        # 'data/v1.0-mini/velodyne_raw/LIDAR_TOP' is raw nuscense LIDAR_TOP(samples + sweeps)
        poisoned_dataset_path = data_root + args.poisoned_dataset_path
        sample_path.symlink_to(poisoned_dataset_path)
        sweep_path.symlink_to(poisoned_dataset_path)

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
        # check dataset_path 
        dataset_path = Path('/vol/bitbucket/cy19/saliency_map_v2/second/data/kitti/training/velodyne')
        if dataset_path.is_symlink():
            dataset_path.unlink()
        # link dataset_path to poisoned dataset 
        data_root = '/vol/bitbucket/cy19/saliency_map_v2/second'
        # '/data/kitti/training/velodyne_raw' is raw kitti LIDAR scenes
        poisoned_dataset_path = data_root + args.poisoned_dataset_path
        dataset_path.symlink_to(poisoned_dataset_path)

        # set model and configs
        if args.target_detector == 'pp':
            if args.detector_object == 'Car':
                config_path = "./configs/pointpillars/car/xyres_16.config" 
                ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/pp_car/voxelnet-296960.tckpt" # pretrained model
                model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/pp_car" # for restore the latest checkpoint(.json)
                args.target_detector = 'pp_car'
            if args.detector_object == 'Pedestrian' or args.detector_object == 'Cyclist':
                config_path = "./configs/pointpillars/ped_cycle/xyres_16.config" 
                ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/pp_ped_cycle/voxelnet-296960.tckpt" # pretrained model
                model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/pp_ped_cycle" # for restore the latest checkpoint(.json)
                args.target_detector = 'pp_ped_cycle'
        if args.target_detector == 'second':
            # 'all.fhd.config'
            config_path = "./configs/all.fhd.config" 
            ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_all/voxelnet-99040.tckpt" # pretrained model
            model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_all" # for restore the latest checkpoint(.json)
            # if args.detector_object == 'Car':
            #     # 'car_fhd'
            #     config_path = "./configs/car.fhd.config" 
            #     ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_car_fhd/voxelnet-74280.tckpt" # pretrained model
            #     model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_car_fhd" # for restore the latest checkpoint(.json)
            #     # # 'car_onestage'
            #     # config_path = "./configs/car.fhd.onestage.config" 
            #     # ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_car_onestage/voxelnet-27855.tckpt" # pretrained model
            #     # model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_car_onestage" # for restore the latest checkpoint(.json)
            #     # # 'car_lite'
            #     # config_path = "./configs/car.lite.config" 
            #     # ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_car_lite/voxelnet-15500.tckpt" # pretrained model
            #     # model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_car_lite" # for restore the latest checkpoint(.json)
            # if args.detector_object == 'Pedestrian' or args.detector_object == 'Cyclist':
            #     config_path = "./configs/people.fhd.config" 
            #     ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_people/voxelnet-30950.tckpt" # pretrained model
            #     model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_people" # for restore the latest checkpoint(.json)

    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    prediction(
        args,
        config,
        model_dir=str(model_dir),
        ckpt_path=str(ckpt_name),
        batch_size=1,
        measure_time=True)
