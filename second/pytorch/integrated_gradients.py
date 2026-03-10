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


import numpy as np
import time
from second.core import box_np_ops
from vanilla_gradient import VanillaGradient

class IntegratedGradients(VanillaGradient): 

    def get_mask(self, input_tensor, roi, expand_valid_gt_boxes_lidar, valid_gt_boxes_lidar, nearest_corner, target_object, basemap='zeros', steps=25, process=lambda x: x):
        raw_points = input_tensor['lidar_points'][0]
        # extract points in bbox first
        pre_start_time = time.time()

        gt_point_mask = box_np_ops.points_in_rbbox(raw_points, expand_valid_gt_boxes_lidar)
        # print('gt_point_mask: ', gt_point_mask)
        target_mask = gt_point_mask.any(-1)
        bg_mask = np.logical_not(target_mask)
        # print('target_mask: ', target_mask)
        # print('bg_mask: ', bg_mask)
        # divide the whole scene into 2 groups     
        bg_lidar_pc = raw_points[bg_mask]  
        target_lidar_pc = raw_points[target_mask]
        points_in_box = target_lidar_pc[:, :3]
        density = target_lidar_pc[:, 3:]

        preprocessing_time = time.time() - pre_start_time

        if basemap == 'zeros':
            basemap = np.zeros_like(points_in_box)
        elif basemap == 'nearest_corner':
            basemap = nearest_corner
        image_diff = points_in_box - basemap
        # print('points_in_box: ', points_in_box)
        # print('basemap: ', basemap)
        # print('image_diff: ', image_diff)
        grad_sum = np.zeros_like(points_in_box)

        # process IG for 25 times
        IG_start_time = time.time()
        for step, alpha in enumerate(np.linspace(0, 1, steps)):
            # print('Processing Integrated Gradients at iteration: ', step)
            # perturb points_in_box
            pc_step = basemap + alpha * image_diff
            # print('pc_step: ', pc_step)
            # combine with density
            lidar_step = np.concatenate([pc_step, density], axis = 1)
            # merge with bg_lidar_pc
            perturbed_lidar = np.concatenate([lidar_step, bg_lidar_pc], axis = 0)
            # print('target_lidar_pc: ', target_lidar_pc)
            # print('bg_lidar_pc: ', bg_lidar_pc)
            # print('perturbed_lidar: ', perturbed_lidar)

            # print('num raw points: ', len(raw_points)) 
            # print('num points in box: ', len(points_in_box))
            # print('num bg points: ', len(bg_lidar_pc))
            # print('num perturbed points: ', len(perturbed_lidar))
            assert len(raw_points) == len(perturbed_lidar)

            input_tensor['lidar_points'] = perturbed_lidar
            
            grad_sum = grad_sum + process(super(IntegratedGradients, self).get_mask(input_tensor, roi, valid_gt_boxes_lidar, target_object, len(points_in_box)))
            mask = grad_sum * image_diff / steps

        IG_computation_time = (time.time() - IG_start_time)/steps
        return mask, points_in_box, preprocessing_time, IG_computation_time