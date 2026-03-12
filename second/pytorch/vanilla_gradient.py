#    Copyright 2026 Chengzeng You. All Rights Reserved.

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


import torch
import numpy as np
from second.core import box_np_ops
from second.builder import voxel_builder
from saliency_mask import SaliencyMask
from scipy.spatial import ConvexHull

def polygon_clip(subjectPolygon, clipPolygon): 
   """ Clip a polygon with another polygon.
   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    # # print('iou_2d = %f / %f (%f)'%(inter_area, (area1+area2-inter_area), iou_2d))

    # ymax = min(corners1[0,1], corners2[0,1])
    # ymin = max(corners1[4,1], corners2[4,1])
    # inter_vol = inter_area * max(0.0, ymax-ymin)
    # vol1 = box3d_vol(corners1)
    # vol2 = box3d_vol(corners2) 
    # iou_3d = inter_vol / (vol1 + vol2 - inter_vol)
    # # print('iou_3d = %f / %f (%f)'%(inter_vol, (vol1+vol2-inter_vol), iou_3d))

    return iou_2d

def get_3d_box_corners(center, box_size, heading_angle):
    ''' Calculate 3D bounding box corners from its parameterization.
    similar implementation of processors/get_cam2_corners, except performing transpose in the end.
    reference: https://github.com/charlesq34/frustum-pointnets/blob/2ffdd345e1fce4775ecb508d207e0ad465bcca80/train/provider.py#L264
    
    Input:
        center: tuple of (x,y,z)
        box_size: tuple of (h, w, l)
        heading_angle: rad scalar, clockwise from pos x axis
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])
    R = roty(heading_angle)

    h,w,l = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];

    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def max_IOU_v1(target_gt_boxes, target_pred_boxes):
    # filter predicted bboxes by gt boxes based on max IOU
    # not reliable. Need to furter check
    predictions_for_gt = []
    if len(target_gt_boxes)!=0 and len(target_pred_boxes)!=0:
        for gt_id, gt_box in enumerate(target_gt_boxes):
            iou_thres = 0.0
            max_id = 0
            # print('===== gt_box %d =====' %(gt_id) )
            gt_corners = get_3d_box_corners(gt_box[0:3],gt_box[3:6],gt_box[6])
            for index, pred_box in enumerate(target_pred_boxes):
                pred_corners = get_3d_box_corners(pred_box[0:3],pred_box[3:6],pred_box[6])
                iou_2d= box_iou(gt_corners,pred_corners)
                if iou_2d>iou_thres:
                    iou_thres = iou_2d
                    max_id = index
            predictions_for_gt.append(target_pred_boxes[max_id])

    # print('predictions_for_gt: ', predictions_for_gt,' iou = ', iou_thres)
    return max_id

def max_IOU_v2(target_gt_boxes, target_pred_boxes):
    # filter predicted bboxes by gt boxes based on max IOU
    gt_boxes_2d = np.delete(target_gt_boxes, [2,5], 1)
    # print('gt_boxes_2d: ', gt_boxes_2d)
    pred_boxes_2d = np.delete(target_pred_boxes, [2,5], 1)
    # print('pred_boxes_2d: ', pred_boxes_2d)
    IOUs = box_np_ops.riou_cc(gt_boxes_2d, pred_boxes_2d)
    # print('IOUs: ', IOUs)
    # find the max_id among all elements
    max_gt_pred_id = np.where(IOUs == np.amax(IOUs))
    # print('max_gt_pred_id: ', max_gt_pred_id)
    max_gt_id = max_gt_pred_id[0]
    max_pred_id = max_gt_pred_id[1]
    return max_pred_id, IOUs

def region_check_torch(x, y, region, buffer = 0.0):
    region_x_min, region_x_max, region_y_min, region_y_max = region[0], region[1], region[2], region[3]
    x_check = torch.logical_and((region_x_min - buffer < x), (x < region_x_max + buffer)) 
    y_check = torch.logical_and((region_y_min - buffer < y), (y < region_y_max + buffer))
    return torch.logical_and(x_check, y_check)

def mask_boxes(boxes_lidar, label_preds, target_object, roi):
    # mask unvalid bboxes to zeros
    if target_object == 'car' or target_object == 'Car':
        type_mask = label_preds == 0
        region_mask = region_check_torch(boxes_lidar[:, 0], boxes_lidar[:, 1], roi, buffer=1.0)
    if target_object == 'pedestrian' or target_object == 'Pedestrian':
        type_mask = label_preds == 1
        region_mask = region_check_torch(boxes_lidar[:, 0], boxes_lidar[:, 1], roi, buffer=5.0)
    if target_object == 'bicycle' or target_object == 'Cyclist':
        type_mask = label_preds == 2
        region_mask = region_check_torch(boxes_lidar[:, 0], boxes_lidar[:, 1], roi, buffer=5.0)
    
    # print('type_mask: ', type_mask)
    # print('region_mask: ', region_mask)

    boxes_mask = torch.logical_and(type_mask, region_mask).reshape(-1, 1)
    # # print('boxes_mask: ', boxes_mask)

    return boxes_mask

class VanillaGradient(SaliencyMask):
    def __init__(self, model, generator_config):
        super(VanillaGradient, self).__init__(model)
        voxel_generator_grad = voxel_builder.build(generator_config, device = torch.device("cuda:0"), requires_grad = True) 
        self.generate_voxel_torch = voxel_generator_grad.generate_torch
        self.generate_voxel_fast = voxel_generator_grad.generate_fast

    def get_mask(self, input_tensor, roi, valid_gt_boxes_lidar, target_object, num_points_in_box):
        # create new example with perturbations
        new_example = input_tensor
        perturbed_points_torch = torch.as_tensor(input_tensor['lidar_points'], dtype=torch.float32, device=torch.device("cuda:0"))
        perturbed_points_torch.requires_grad = True
        perturbed_points_torch.retain_grad()
        # # print('perturbed_points_torch: ', perturbed_points_torch)

        # recreate voxels based on perturbed lidar
        voxels, coordinates, num_points = self.generate_voxel_torch(perturbed_points_torch) 
        # voxels_fast, coordinates_fast, num_points_fast = self.generate_voxel_fast(perturbed_points_torch) 

        # # compare fast version with genuine version
        # voxels, _ = torch.sort(voxels, 0)
        # coordinates, _  = torch.sort(coordinates, 0)
        # num_points, _  = torch.sort(num_points, 0)
        # voxels_fast, _  = torch.sort(voxels_fast, 0)
        # coordinates_fast, _  = torch.sort(coordinates_fast, 0)
        # num_points_fast, _  = torch.sort(num_points_fast, 0)
        

        # # print('-'*20)
        # # print('voxels: ', voxels, voxels.shape)
        # # print('voxels_fast: ', voxels_fast, voxels_fast.shape)
        # voxel_diff = torch.sum(voxels - voxels_fast)
        # # print('voxel_diff: ', voxel_diff)

        # # print('-'*20)
        # # print('coordinates: ', coordinates, coordinates.shape)
        # # print('coordinates_fast: ', coordinates_fast, coordinates_fast.shape)
        # coordinates_diff = coordinates - coordinates_fast
        # # print('coordinates_diff: ', coordinates_diff)

        # # print('-'*20)
        # # print('num_points: ', num_points, num_points.shape)
        # # print('num_points_fast: ', num_points_fast, num_points_fast.shape)
        # num_points_diff = num_points - num_points_fast
        # # print('num_points_diff: ', num_points_diff)

        new_coordinates = torch.cat([torch.zeros(size = (len(coordinates),1), dtype = torch.int32, device = torch.device("cuda:0")), coordinates], axis = 1)
        # # print('voxelize perturbed points: ')
        # # print('voxels: ', voxels)
        # # print('coordinates: ', new_coordinates)
        # # print('num_points: ', num_points)

        # perturb new example
        new_example['voxels'] = voxels
        new_example['coordinates'] = new_coordinates
        new_example['num_points'] = num_points
        new_example['num_voxels'] = torch.tensor([voxels.shape[0]], dtype=torch.int64)

        # new_example = example_convert_to_torch(example, float_dtype, device)
        # delete unnecessary infomation
        # new_example.pop("lidar_points") 
        # new_example.pop("gt_boxes_lidar")
        # new_example.pop("gt_boxes_names")
        # print('new_example: ', new_example)

        pred_dict= self.model(new_example)
        # print('pred_dict: ', pred_dict)
        logits = pred_dict[0]['scores']
        pred_boxes_lidar = pred_dict[0]['box3d_lidar']
        label_preds = pred_dict[0]['label_preds']
        # print('logits: ', logits)
        # print('valid_gt_boxes_lidar: ', valid_gt_boxes_lidar)
        # print('pred_boxes_lidar: ', pred_boxes_lidar)

        # # test gradients
        # target_logits = torch.ones_like(logits) 
        # logits.backward(target_logits)
        # # print('perturbed_points_torch.grad: ', perturbed_points_torch.grad)
        # # print('sum of grads: ', torch.sum(perturbed_points_torch.grad))
        # assert 0 > 1 

        # mask pred bboxes based on focus region and labels
        boxes_mask = mask_boxes(pred_boxes_lidar, label_preds, target_object, roi)
        valid_pred_boxes_lidar = pred_boxes_lidar * boxes_mask

        # Filter logits to only focus on the target prediction
        target_logits = torch.zeros_like(logits)
        if torch.sum(boxes_mask) > 0:
            # calculate IOU between target GT bbox and predicted bboxes(3d lidar) 
            # Based on IOU, find the best prediction in each IG step
            # print('=== calculate IOU based on box3d_lidar ===')
            max_pred_id, IOUs = max_IOU_v2(valid_gt_boxes_lidar, valid_pred_boxes_lidar.detach().cpu().numpy().astype(np.float64))
            # print('pred box id: ', max_pred_id)
            # print('target pred box: ', valid_pred_boxes_lidar[max_pred_id])
            # print('target logits: ', logits[max_pred_id])
            if np.max(IOUs) > 0:
                target_logits[max_pred_id] = 1
            # # print('target_logits: ', target_logits)
        
        # Filter gradients to only focus on target logits
        if len(logits) > 0:
            self.model.zero_grad()
            logits.backward(target_logits) 
            # backward is slow(more than 10s). lidar pc too large? 
            # normally among 100,000 points, we only need gradients of 1000 points.

            # print('perturbed_points_torch.grad: ', perturbed_points_torch.grad)
            # print('sum of grads: ', torch.sum(perturbed_points_torch.grad))
            grad_all_points = perturbed_points_torch.grad.detach().cpu().numpy() # gradients of all voxels in the scene
            # print('grad_all_points: ', grad_all_points, 'shape: ', grad_all_points.shape)
            grad_target_points = grad_all_points[:num_points_in_box, :3]
        else:
            grad_target_points = np.zeros_like(perturbed_points_torch.detach().cpu().numpy())[:num_points_in_box, :3]
        # print('grad_target_points: ', grad_target_points, 'shape: ', grad_target_points.shape)
        return grad_target_points

    def get_smoothed_mask(self, image_tensor, target_class=None, samples=25, std=0.15, process=lambda x: x**2):
        std = std * (torch.max(image_tensor) - torch.min(image_tensor)).detach().cpu().numpy()

        batch, channels, width, height = image_tensor.size()
        grad_sum = np.zeros((width, height, channels))
        for sample in range(samples):
            noise = torch.empty(image_tensor.size()).normal_(0, std).to(image_tensor.device)
            noise_image = image_tensor + noise
            grad_sum += process(self.get_mask(noise_image, target_class))
        return grad_sum / samples

    @staticmethod
    def apply_region(mask, region):
        return mask * region[..., np.newaxis]
