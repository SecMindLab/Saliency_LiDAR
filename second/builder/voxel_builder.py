
# Torch version of voxel builder by Chengzeng You
# ==============================================================================
import numpy as np
import torch
from spconv.pytorch.utils import PointToVoxel
from second.protos import voxel_generator_pb2

@torch.jit.script
def _points_to_voxel_reverse_kernel_torch(points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points=torch.tensor(35),
                                    max_voxels=torch.tensor(20000)):
    # put all computations to one loop. 
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    # print('coors_range: ', coors_range)
    # print('voxel_size: ', voxel_size)
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = torch.round(grid_size).to(torch.int32)
    # print('grid_size: ', grid_size)
    coor = torch.zeros(size=(3, ), dtype=torch.int32, device = torch.device("cuda:0"))

    voxel_num = torch.tensor(0, dtype=torch.int64, device = torch.device("cuda:0"))
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            raw_c = (points[i, j] - coors_range[j]) / voxel_size[j]
            c = torch.floor(raw_c) - raw_c + raw_c # remain gradients of floor operation. But since coors variable is int type, c has to be converted to int finally without gradients.
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        # print('points: ', points[i])
        # print('coor: ', coor)
        # print('coor_to_voxelidx shape: ', coor_to_voxelidx.shape)
        voxelidx = coor_to_voxelidx[coor[0].long(), coor[1].long(), coor[2].long()]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num = voxel_num + 1
            coor_to_voxelidx[coor[0].long(), coor[1].long(), coor[2].long()] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num.long()] = points[i]
            num_points_per_voxel[voxelidx] = num_points_per_voxel[voxelidx].clone() + 1
    return voxel_num

def _points_to_voxel_kernel_torch(points,
                            voxel_size,
                            coors_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points=35,
                            max_voxels=20000):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    lower_bound = coors_range[:3]
    upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = torch.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num

def points_to_voxel_torch(points,
                     voxel_size,
                     coors_range,
                     max_points=35,
                     reverse_index=True,
                     max_voxels=20000,
                     device = torch.device("cuda:0")):
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud) 
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output 
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """

    voxel_size = torch.tensor(voxel_size)
    coors_range = torch.tensor(coors_range)
    max_points = torch.tensor(max_points)
    max_voxels = torch.tensor(max_voxels)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = torch.round(voxelmap_shape).to(torch.int32)
    if reverse_index:
        voxelmap_shape = torch.index_select(voxelmap_shape, 0, torch.LongTensor([2,1,0]))

    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = torch.zeros(size=(max_voxels, ), dtype=torch.int32, device = device)
    coor_to_voxelidx = -torch.ones(size=voxelmap_shape.numpy().tolist(), dtype=torch.int64, device = device)
    voxels = torch.zeros(size=(max_voxels, max_points, points.shape[-1]), dtype=torch.float32, device=device)
    coors = torch.zeros(size=(max_voxels, 3), dtype=torch.int32, device = device)

    print('points in points_to_voxel_torch: ', points)
    if reverse_index:
        voxel_num = _points_to_voxel_reverse_kernel_torch(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    else:
        voxel_num = _points_to_voxel_kernel_torch(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    # voxels[:, :, -3:] = voxels[:, :, :3] - \
    #     voxels[:, :, :3].sum(axis=1, keepdims=True)/num_points_per_voxel.reshape(-1, 1, 1)
    return voxels, coors, num_points_per_voxel


class VoxelGeneratorV2():
    def __init__(self,
                 vsize_xyz,
                 coors_range_xyz,
                 num_point_features,
                 max_num_voxels,
                 max_num_points_per_voxel,
                 device = torch.device("cpu:0"),
                 requires_grad = False,
                 full_mean=False,
                 block_filtering=False,
                 block_factor=8,
                 block_size=3,
                 height_threshold=0.1,
                 height_high_threshold=2.0
                 ):
        PointToVoxelGen = PointToVoxel(
            vsize_xyz = vsize_xyz,
            coors_range_xyz = coors_range_xyz,
            num_point_features = num_point_features,
            max_num_voxels = max_num_voxels,
            max_num_points_per_voxel = max_num_points_per_voxel,
            device = device,
            requires_grad=requires_grad)

        assert full_mean is False, "don't use this."
        point_cloud_range = np.array(coors_range_xyz, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(vsize_xyz, dtype=np.float32)
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        if block_filtering:
            assert block_size > 0
            assert grid_size[0] % block_factor == 0
            assert grid_size[1] % block_factor == 0

        voxelmap_shape = tuple(np.round(grid_size).astype(np.int32).tolist())
        voxelmap_shape = voxelmap_shape[::-1]
        self._coor_to_voxelidx = np.full(voxelmap_shape, -1, dtype=np.int32)
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points_per_voxel
        self._max_voxels = max_num_voxels
        self._grid_size = grid_size
        self._full_mean = full_mean 
        self._block_filtering = block_filtering
        self._block_factor = block_factor
        self._height_threshold = height_threshold
        self._block_size = block_size
        self._height_high_threshold = height_high_threshold
        self.PointToVoxelGen = PointToVoxelGen
        self.voxel_point_mask = np.zeros(shape=(max_num_voxels, max_num_points_per_voxel))

    def generate(self, points, max_voxels=None):
        res = {}
        points = torch.from_numpy(points)
        voxels, coors, num_points_per_voxel = self.PointToVoxelGen(points)
        res["voxels"] = voxels
        res["coordinates"] = coors
        res["num_points_per_voxel"] = num_points_per_voxel

        return res

    def generate_torch_built_in(self, points):
        # operate based on pytorch tensors on the device, but would cause gradients missing. Need to fix this.
        voxels, coors, num_points_per_voxel = self.PointToVoxelGen(points)
        return voxels, coors, num_points_per_voxel
    
    def generate_torch(self, points, device = torch.device("cuda:0")):
        return points_to_voxel_torch(
            points, self._voxel_size, self._point_cloud_range,
            self._max_num_points, True, self._max_voxels, device = device) 


    def generate_multi_gpu(self, points, max_voxels=None):
        res = {}
        points = torch.from_numpy(points)
        voxels, coors, num_points_per_voxel = self.PointToVoxelGen(points)

        res["voxels"] = voxels
        res["coordinates"] = coors
        res["num_points_per_voxel"] = num_points_per_voxel

        return res

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size

def build(voxel_config, device = torch.device("cpu:0"), requires_grad = False):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(voxel_config, (voxel_generator_pb2.VoxelGenerator)):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')
    voxel_generator = VoxelGeneratorV2(
        vsize_xyz=list(voxel_config.voxel_size),
        coors_range_xyz=list(voxel_config.point_cloud_range),
        num_point_features=4,
        max_num_points_per_voxel=voxel_config.max_number_of_points_per_voxel,
        max_num_voxels=30000,
        device = device,
        requires_grad = requires_grad)
    # num_point_features: ref to config file
    # max_num_voxels: ref to preprocess in config file
    return voxel_generator

