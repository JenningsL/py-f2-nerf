import torch
import py_f2nerf

conf = 'exp/ngp_fox/test/record/runtime_config.yaml'
data_pool = py_f2nerf.GlobalDataPool(conf)
data_pool.base_exp_dir = 'exp/ngp_fox/test'
print(data_pool.base_exp_dir)

# print(py_f2nerf.PoseInterpolate(torch.eye(4), torch.eye(4), 0.5))

dataset = py_f2nerf.Dataset(data_pool)

# train_rays, gt_colors, emb_idx
train_rays, gt_colors, emb_idx = dataset.RandRaysData(128, 1)
sampler = py_f2nerf.PersSampler(data_pool)
samples = sampler.GetSamples(train_rays.origins, train_rays.dirs, train_rays.bounds)
print(samples.pts)

# # renderer = py_f2nerf.Renderer(data_pool, 10)

# from torch.utils.cpp_extension import load
# py_f2nerf = load(name='py_f2nerf', 
#     sources=['src/py_f2nerf.cpp', 'src/Utils/GlobalDataPool.cpp']
# )