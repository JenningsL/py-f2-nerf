import torch
import py_f2nerf

conf = 'exp/ngp_fox/test/record/runtime_config.yaml'
data_pool = py_f2nerf.GlobalDataPool(conf)
data_pool.base_exp_dir = 'exp/ngp_fox/test'
dataset = py_f2nerf.Dataset(data_pool)

# c2w_train = torch.zeros(32, 3, 4)
# w2c_train = torch.zeros(32, 3, 4)
# dist_params = torch.zeros(32, 4)
# intri_train = torch.zeros(32, 3, 3)
# bounds_train = torch.zeros(32, 2)

data_pool = py_f2nerf.CreateGlobalDataPool(
    conf, 'exp/ngp_fox/test', dataset.c2w_train, dataset.w2c_train, dataset.intri_train, dataset.bounds_train)
# data_pool.base_exp_dir = 'exp/ngp_fox/test'
# print(data_pool.base_exp_dir)

# print(py_f2nerf.PoseInterpolate(torch.eye(4), torch.eye(4), 0.5))

# train_rays, gt_colors, emb_idx
train_rays, gt_colors, emb_idx = dataset.RandRaysData(128, 1)
sampler = py_f2nerf.PersSampler(data_pool)
samples = sampler.GetSamples(train_rays.origins, train_rays.dirs, train_rays.bounds)
print(samples.pts.shape, samples.anchors.shape) # perspective warpped points(-1,1), leaf node index 
# from IPython import embed;embed()
field = py_f2nerf.Hash3DAnchoredField(data_pool)
feats = field.AnchoredQuery(samples.pts, samples.anchors[:,0])
print(feats.shape)
# # renderer = py_f2nerf.Renderer(data_pool, 10)

# from torch.utils.cpp_extension import load
# py_f2nerf = load(name='py_f2nerf', 
#     sources=['src/py_f2nerf.cpp', 'src/Utils/GlobalDataPool.cpp']
# )