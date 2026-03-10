# calculate ASRs based on GT and benign bboxes
# ref to plots_motion_attack11.py
# TBC...





'''
combined codes:
./data/kitti_dataset2.py

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

mv /vol/bitbucket/cy19/saliency_map_v2/second/data/kitti/training/velodyne /vol/bitbucket/cy19/saliency_map_v2/second/data/kitti/training/velodyne_raw
ln -sfn /vol/bitbucket/cy19/saliency_map_v2/second/data/kitti/training/velodyne_raw  /vol/bitbucket/cy19/saliency_map_v2/second/data/kitti/training/velodyne 

python calculate_ASR.py --end_scene 5

'''
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

class VeloBox:
    def __init__(self, scene: int, classification: str, centroid: np.ndarray, dimension: np.ndarray, yaw: float):
        self.scene = scene
        self.classification = classification
        self.centroid = centroid
        self.dimension = dimension
        self.yaw = yaw

    def __str__(self):
        return " scene: %d, cls: %s, x: %f, y: %f, l: %f, w: %f, yaw: %f" % (
            self.scene, self.classification, self.centroid[0], self.centroid[1], self.dimension[0], self.dimension[1], self.yaw)

def build_network(model_cfg, measure_time=False):
    # voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator, device = torch.device("cuda:0"))
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

def car_check(target_object): 
    res = target_object == 'car' or target_object == 'Car'
    return res
def ped_check(target_object):
    res = target_object == 'pedestrian' or target_object == 'Pedestrian'
    return res
def cyl_check(target_object):
    res = target_object == 'bicycle' or target_object == 'Cyclist'
    return res

def generate_saliency_map(args,
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
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
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

    print("Generate output labels...")
    bar = ProgressBar()
    bar.start((len(eval_dataset) + batch_size - 1) // batch_size)

    start_scene = args.start_scene
    end_scene = args.end_scene
    if args.end_scene == -1:
        end_scene = len(eval_dataset)
    print('start_scene: ', start_scene)
    print('end_scene: ', end_scene)

    dataloader = iter(eval_dataloader)
    # skip scenes before start_scene
    for idx in range(start_scene):
        example = next(dataloader)
        print('skipping example{} : {}'.format(idx, example))
    # contiue saliency map aggregation
    for idx in range(start_scene, end_scene):
        example = next(dataloader)
        print('example {}: {}'.format(idx, example))

        # select bboxes in the first batch
        gt_boxes_names = example["gt_boxes_names"][0]
        gt_boxes_lidar = example["gt_boxes_lidar"][0]
        # print('gt_boxes_lidar: ', gt_boxes_lidar)
        # print(gt_boxes_lidar.shape)

        example = example_convert_to_torch(example, torch.float32, device)
        pred_dict= net(example)
        print('pred_dict: ', pred_dict)
        # logits = pred_dict[0]['scores']
        pred_boxes_lidar = pred_dict[0]['box3d_lidar']
        label_preds = pred_dict[0]['label_preds']
        # print('logits: ', logits)
        print('gt_boxes_lidar: ', gt_boxes_lidar)
        print('pred_boxes_lidar: ', pred_boxes_lidar)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", "--d", type=str, default='pp')
    parser.add_argument("--target_object", "--t", type=str, default='Car')
    parser.add_argument("--dataset", type=str, default='kitti')
    parser.add_argument("--start_scene", type=int, default=0)
    parser.add_argument("--end_scene", type=int, default=-1)
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
            # 'all.fhd.config'
            config_path = "./configs/all.fhd.config" 
            ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_all/voxelnet-99040.tckpt" # pretrained model
            model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_all" # for restore the latest checkpoint(.json)
            # if args.target_object == 'Car':
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
            # if args.target_object == 'Pedestrian' or args.target_object == 'Cyclist':
            #     config_path = "./configs/people.fhd.config" 
            #     ckpt_name = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_people/voxelnet-30950.tckpt" # pretrained model
            #     model_dir = "/vol/bitbucket/cy19/saliency_map_v2/pretrained_models_v1.6/kitti/second_people" # for restore the latest checkpoint(.json)

    with open('/vol/bitbucket/cy19/saliency_map_v2/second/attack/{}_saliency_hiding_1_object/{}/hiding_type_{}/attacked_gt_bboxes.pkl'.format(args.dataset, args.detector, args.target_object), 'rb') as f:
        all_target_gt_boxes = pickle.load(f)
    print('all_target_gt_boxes: ', all_target_gt_boxes)
    for batch_gt_boxes in all_target_gt_boxes:
        for scene_gt_boxes in batch_gt_boxes:
            for gt_box in scene_gt_boxes:
                print('Box | scene: {}, centroid: {}, dimension: {}, yaw: {}'.format(gt_box.scene, gt_box.centroid, gt_box.dimension, gt_box.yaw))

    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    generate_saliency_map(
        args,
        config,
        model_dir=str(model_dir),
        ckpt_path=str(ckpt_name),
        batch_size=1,
        measure_time=True)
