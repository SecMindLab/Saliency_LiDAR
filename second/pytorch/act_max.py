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


import numpy as np
from integrated_gradients import IntegratedGradients 

"""
Create a hook into target layer
    Example to hook into classifier 6 of Alexnet:
        alexnet.classifier[6].register_forward_hook(layer_hook('classifier_6'))
"""

def layer_hook(act_dict, layer_name): 
    def hook(module, input, output):
        act_dict[layer_name] = output
    return hook

"""
Optimizing Loop
    Dev: maximize layer vs neuron
"""
def act_max(network, 
    generator_config,
    input, 
    roi,
    expand_valid_gt_boxes_lidar,
    valid_gt_boxes_lidar,
    nearest_corner,
    target_object,
    IG_steps=25,
    basemap='zeros'
    ):

    # IG = IntegratedGradients(network.train(), generator_config)
    IG = IntegratedGradients(network, generator_config)
    mask, points_in_box, preprocessing_time, IG_computation_time = IG.get_mask(input, roi, expand_valid_gt_boxes_lidar, valid_gt_boxes_lidar, nearest_corner, target_object, basemap, IG_steps)
    # print('mask: ', mask)
    contri_points = np.sum(mask,axis=1)
    # print('points_in_box: ', points_in_box)
    # print('contri_pionts: ', contri_points)
    # print('shape of contri_points: ', contri_points.shape)
    
    # num_positive_voxels = np.sum(contri_points>=0)
    # num_negative_voxels = np.sum(contri_points<0)

    return contri_points, points_in_box, preprocessing_time, IG_computation_time 