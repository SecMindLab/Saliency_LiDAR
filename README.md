# Saliency-LiDAR

This is the official implementation of Saliency-LiDAR (Explainability-aware Frustum Attack: Exposing Structural Vulnerabilities in LiDAR-Based 3D Object Detectors). 

We introduce the Saliency-LiDAR (SALL) method, which aggregates Integrated Gradient attributions across scenes to produce universal saliency maps for LiDAR-based 3D object detectors. 

<img width="4142" height="1668" alt="saliency_lidar" src="https://github.com/user-attachments/assets/29c2d837-801b-4cda-adbe-50b256cfb0f1" />


Guided by these maps, we design the Explainability-aware Frustum Attack (EFA), which selectively perturbs only the most influential frustums rather than uniformly attacking entire object regions. 

<img width="4145" height="1188" alt="efa" src="https://github.com/user-attachments/assets/8dd35cfe-d858-4a36-8738-7fd06d932888" />


Our experiments show that EFA reduces detection recall by more than 15 percentage points while requiring 25–50\% fewer perturbed frustums than the SOTA non–saliency-aware baseline. These findings reveal that modern 3D detectors concentrate discriminative evidence in a small subset of spatial regions, exposing a structural robustness vulnerability in current LiDAR perception systems. 

Paper link:

## Requirements
 - CUDA 11
 - Python 3
 - pytorch, spconv, PIL, numpy and other relevant packages

## Usage
#### 1. Get pretrained models
Add your pretrained models to folder './pretrained_models_v1.6/'.
This project uses [Second](https://github.com/traveller59/second.pytorch.git) as the target model to generate and apply our saliency maps.  


#### 2. Dataset
Download [Nuscenes dataset](https://www.nuscenes.org/) and [KITTI dataset](https://www.cvlibs.net/datasets/kitti/) to folder: `.second/data/`. For example, to download Nuscenes mini split:
```
wget https://www.nuscenes.org/data/v1.0-mini.tgz
```

#### 3. Generate saliency maps
For example, generate a saliency map for car objects predicted by PointPillars in KITTI dataset:
```
cd ./second/
python ./pytorch/saliency_map.py --box_size_factor 1.5 --dataset 'kitti' --basemap 'zeros' --detector 'pp' --target_object 'Car' 
```
Our pretrained saliency maps are saved in folder './results/v5.1/'.

#### 4. Saliency-guided attacks
Custumize below parameters to attack your target model based on the generated universal saliency map:
```
python ./attack/saliency_hiding.py --pert_shape 'saliency_map' --map_source_dataset 'kitti' --map_source_detector 'pp' --map_source_object 'Car' --top_n 1.0 --selection_level 'frustum_level' --frustum_degree 1.0 --selection_strategy 'critical_first' --perturbation_strategy 'ray' --frustum_budget 20  --target_dataset 'kitti' --target_detector 'pp'  --target_object 'Car' --iou_thres 0.7 --conf_thres 0.7 
```



## Reference
If you find this project useful in your research, please cite:
```
   
```
