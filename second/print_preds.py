import os
import pickle
import numpy as np
class VeloBox:
    def __init__(self, scene: int, classification: np.ndarray, boxes: np.ndarray, logits: np.array):
        self.scene = scene
        self.classification = classification
        self.boxes = boxes
        self.logits = logits

    def __str__(self):
        return " scene: {}, cls: {}, boxes: {}, logits: {}".format(
            self.scene, self.classification, self.boxes, self.logits)

def check_cat_id(target_object):
    if target_object == 'car' or target_object == 'Car':
        cat_id = 0
    if target_object == 'pedestrian' or target_object == 'Pedestrian':
        cat_id = 1
    if target_object == 'bicycle' or target_object == 'Cyclist':
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

# load prediction results
# poisoned_dataset_path = '/vol/bitbucket/cy19/saliency_map_v2/second/data/v1.0-mini/velodyne_raw/LIDAR_TOP'
# res_path = os.path.join(poisoned_dataset_path, 'second_pred_bboxes.pkl')
# with open(res_path, 'rb') as f:
#     all_pred_boxes = pickle.load(f)
# # print(all_pred_boxes[-1])
# for pred_boxes in all_pred_boxes:
#     print('==========')
#     # print('===== scene {} ====='.format(pred_boxes.scene))
#     print(pred_boxes)

# load gt boxes
data_root = '/vol/bitbucket/cy19/saliency_map_v2/second'
poisoned_dataset_path = data_root + '/attack/saliency_map_hiding_kitti_Car/map_source_kitti_second_top0.3/critical_first_selection_random_shifting_pb200/velodyne'
gt_boxes_path = os.path.join('/', *poisoned_dataset_path.split('/')[:8], 'attacked_gt_bboxes.pkl')
print('gt_boxes_path: ', gt_boxes_path)
with open(gt_boxes_path, 'rb') as f:
    all_gt_boxes = pickle.load(f)

# load benign boxes
benign_boxes_path = '/vol/bitbucket/cy19/saliency_map_v2/second/data/kitti/training/velodyne_raw'
benign_boxes_path = os.path.join(benign_boxes_path, 'pp_car_pred_bboxes.pkl')
with open(benign_boxes_path, 'rb') as f:
    all_benign_boxes = pickle.load(f)

# select gt boxes
selected_gt_boxes = []
for batch_gt_boxes in all_gt_boxes:
    for scene_gt_boxes in batch_gt_boxes:
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

# # select benign boxes
# selected_benign_boxes = []
dist_thres = 0.5
conf_thres = 0.7
# for scene_gt_box in selected_gt_boxes:
#     # use attacked gt bboxes to confirm target benign bboxes
#     for scene_benign_boxes in all_benign_boxes:
#         if scene_gt_box.scene == scene_benign_boxes.scene:
#             print(scene_gt_box)
#             print(scene_benign_boxes) 
#             scene_valid_benign_boxes = VeloBox(scene=scene_gt_box.scene, classification=[], boxes=[], logits=[])
#             for gt_cat_id, gt_box in zip(scene_gt_box.classification, scene_gt_box.boxes):
#                 # mask out other categories in benign bboxes
#                 cls_mask = scene_benign_boxes.classification == gt_cat_id
#                 logit_mask = scene_benign_boxes.logits > conf_thres
#                 union_mask = np.logical_and(cls_mask, logit_mask)
#                 # print('cls_mask: ', cls_mask)
#                 mask_scene_benign_boxes = scene_benign_boxes.boxes[union_mask]
#                 mask_scene_benign_box_logits = scene_benign_boxes.logits[union_mask]
#                 # print('mask_scene_benign_boxes: ', mask_scene_benign_boxes)
#                 # print('mask_scene_benign_box_logits: ', mask_scene_benign_box_logits)

#                 # find the most valid bbox for each gt box
#                 most_valid_benign_box = []
#                 most_valid_benign_box_logit = None
#                 for idx, benign_box in enumerate(mask_scene_benign_boxes):
#                     min_dist = dist_thres
#                     dist = center_2d_distance(benign_box, gt_box)
#                     if dist < min_dist:
#                         # if the benign box find a matched gt box
#                         most_valid_benign_box = benign_box
#                         most_valid_benign_box_logit = mask_scene_benign_box_logits[idx]
#                         min_dist = dist
#                 # print('most_valid_benign_box: ', most_valid_benign_box)
#                 scene_valid_benign_boxes.boxes.append(most_valid_benign_box)
#                 scene_valid_benign_boxes.logits.append(most_valid_benign_box_logit)
#             print('========')
#             print('scene_gt_box: ', scene_gt_box)
#             print('scene_valid_benign_boxes: ', scene_valid_benign_boxes)
#             selected_benign_boxes.append(scene_valid_benign_boxes)

# compare with check_attack
# make sure this part can be deleted
# TBC...

# load prediction results
res_path = os.path.join(poisoned_dataset_path, 'pp_car_pred_bboxes.pkl')
with open(res_path, 'rb') as f:
    all_pred_boxes = pickle.load(f)
print('poisoned_dataset_path: ', res_path)
# print('all_pred_boxes: ',all_pred_boxes)
# for pred_boxes in all_pred_boxes:
#     # print('==========')
#     # print('===== scene {} ====='.format(pred_boxes.scene))
#     print(pred_boxes)

def check_attack(benign_res, poisoned_res, dist_thres):
    '''
    Check if there are unmatched benign res. 
    benign_res: detected results without attacks
    poisoned_res: detected results with attacks
    '''

    poisoned_benign_boxes_mask = []
    selected_poisoned_res = []
    for benign_scene in benign_res:
        for poisoned_scene in poisoned_res:
            # find scene match
            if benign_scene.scene == poisoned_scene.scene:
                print('========')
                print('benign_scene: ', benign_scene)
                # print('poisoned_scene: ', poisoned_scene)
                valid_poisoned_scene = VeloBox(scene=poisoned_scene.scene, classification=[], boxes=[], logits=[])
                for benign_cat_id, benign_box in zip(benign_scene.classification, benign_scene.boxes):
                    # only focus on non-empty bboxes
                    if len(benign_box) > 0:
                        is_poisoned = True
                        # Find match for each benign box
                        for p_idx, poisoned_box in enumerate(poisoned_scene.boxes):
                            compare_cat = poisoned_scene.classification[p_idx] == benign_cat_id
                            compare_conf = poisoned_scene.logits[p_idx] > conf_thres
                            # check if the box and the check box overlap, two parameters can be used: center_distance and scale_iou
                            # compare_size = scale_iou(benign_box,poisoned_box)>0.5 # normally IOU > 0.7
                            compare_center = center_2d_distance(benign_box, poisoned_box)< dist_thres
                            if compare_cat and compare_conf and compare_center:
                                # if the benign box find a match, then this benign box is not poisoned
                                is_poisoned = False
                                valid_poisoned_scene.classification.append(poisoned_scene.classification[p_idx])
                                valid_poisoned_scene.logits.append(poisoned_scene.logits[p_idx])
                                valid_poisoned_scene.boxes.append(poisoned_box)
                                break
                        # collect poison infomation of each benign box
                        poisoned_benign_boxes_mask.append(is_poisoned)
                # collect each valid poisoned scene
                print('valid_poisoned_scene: ', valid_poisoned_scene)
                selected_poisoned_res.append(valid_poisoned_scene)
    return poisoned_benign_boxes_mask, selected_poisoned_res


print('---------- Calculate ASR ----------')    
print('----- Benign calculation -----')
# calculate ASR under benign scenario  
gt_benign_boxes_mask, selected_benign_boxes = check_attack(selected_gt_boxes, all_benign_boxes, dist_thres)
success = np.sum(gt_benign_boxes_mask)
all_target = len(gt_benign_boxes_mask)
print('ASR(benign) = {}/{}({:.2f}%)'.format(success, all_target, success/all_target*100))

print('----- True calculation -----')
# compare benign bboxes with poisoned bboxes
benign_poisoned_boxes_mask, selected_pred_boxes = check_attack(selected_benign_boxes, all_pred_boxes, dist_thres)
success = np.sum(benign_poisoned_boxes_mask)
all_target = len(benign_poisoned_boxes_mask)
print('ASR(true) = {}/{}({:.2f}%)'.format(success, all_target, success/all_target*100))

print('----- End-to-end calculation -----')
# compare gt bboxes with poisoned bboxes
gt_poisoned_boxes_mask, selected_pred_boxes = check_attack(selected_gt_boxes, all_pred_boxes, dist_thres)
success = np.sum(gt_poisoned_boxes_mask)
all_target = len(gt_poisoned_boxes_mask)
print('ASR(end) = {}/{}({:.2f}%)'.format(success, all_target, success/all_target*100))

# go through each part and log 
# double check
# TBC...