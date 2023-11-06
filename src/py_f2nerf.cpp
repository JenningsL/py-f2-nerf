#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "Utils/GlobalDataPool.h"
#include "Utils/CameraUtils.h"
#include "PtsSampler/PtsSamplerFactory.h"
#include "Dataset/Dataset.h"
#include "PtsSampler/PersSampler.h"
#include "Field/Hash3DAnchored.h"
#include "Renderer/Renderer.h"
#include "Utils/CustomOps/CustomOps.h"
#include "Utils/CustomOps/Scatter.h"
#include "Utils/CustomOps/FlexOps.h"

namespace py = pybind11;

PYBIND11_MODULE(py_f2nerf, m) {
    py::enum_<RunningMode>(m, "RunningMode")
        .value("TRAIN", RunningMode::TRAIN)
        .value("VALIDATE", RunningMode::VALIDATE)
        .export_values();

    // py::class_<BoundedRays>(m, "BoundedRays")
    //     .def_readwrite("origins", &BoundedRays::origins)
    //     .def_readwrite("dirs", &BoundedRays::dirs)
    //     .def_readwrite("bounds", &BoundedRays::bounds);

    py::class_<SampleResultFlex>(m, "SampleResultFlex")
        .def_readwrite("pts_warp", &SampleResultFlex::pts)
        .def_readwrite("pts", &SampleResultFlex::pts_real)
        .def_readwrite("dt_warp", &SampleResultFlex::dt)
        .def_readwrite("dt", &SampleResultFlex::dt_real)
        .def_readwrite("t", &SampleResultFlex::t)
        .def_readwrite("anchors", &SampleResultFlex::anchors)
        .def_readwrite("pts_idx_bounds", &SampleResultFlex::pts_idx_bounds)
        .def_readwrite("first_oct_dis", &SampleResultFlex::first_oct_dis)
        .def_readwrite("dirs", &SampleResultFlex::dirs);
    
    py::class_<GlobalDataPool>(m, "GlobalDataPool")
        .def(py::init<const std::string&>())
        .def_readwrite("mode", &GlobalDataPool::mode_) // 0 - TRAIN, 1 - VALIDATE
        .def_readwrite("n_volumes", &GlobalDataPool::n_volumes_)
        .def_readwrite("backward_nan", &GlobalDataPool::backward_nan_)
        .def_readwrite("sampled_oct_per_ray", &GlobalDataPool::sampled_oct_per_ray_)
        .def_readwrite("ray_march_init_fineness", &GlobalDataPool::ray_march_init_fineness_)
        .def_readwrite("ray_march_fineness_decay_end_iter", &GlobalDataPool::ray_march_fineness_decay_end_iter_)
        .def_readwrite("ray_march_fineness", &GlobalDataPool::ray_march_fineness_)
        .def_readwrite("iter_step", &GlobalDataPool::iter_step_)
        .def_readwrite("c2w_train", &GlobalDataPool::c2w_train_)
        .def_readwrite("w2c_train", &GlobalDataPool::w2c_train_)
        .def_readwrite("intri_train", &GlobalDataPool::intri_train_)
        .def_readwrite("bounds_train", &GlobalDataPool::bounds_train_)
        .def_readwrite("base_exp_dir", &GlobalDataPool::base_exp_dir_);

    // py::class_<Dataset>(m, "Dataset")
    //     .def(py::init<GlobalDataPool*>())
    //     .def_readwrite("c2w_train", &Dataset::c2w_train_)
    //     .def_readwrite("w2c_train", &Dataset::w2c_train_)
    //     .def_readwrite("intri_train", &Dataset::intri_train_)
    //     .def_readwrite("bounds_train", &Dataset::bounds_train_)
    //     .def("RandRaysData", &Dataset::RandRaysData)
    //     .def("RaysOfCamera", &Dataset::RaysOfCamera);

    // py::class_<PersOctree>(m, "PersOctree")
    //     .def(py::init<int , float , float ,
    //          const torch::Tensor& , const torch::Tensor& , const torch::Tensor& , const torch::Tensor& >());

    py::class_<PersSampler>(m, "PersSampler")
        .def(py::init<GlobalDataPool*>())
        .def("States", &PersSampler::States)
        .def("LoadStates", &PersSampler::LoadStates)
        .def("GetSamples", &PersSampler::GetSamples)
        .def("GetEdgeSamples", &PersSampler::GetEdgeSamples)
        .def("UpdateOctNodes", &PersSampler::UpdateOctNodes);

    torch::python::bind_module<Hash3DAnchored>(m, "Hash3DAnchoredField")
        .def(py::init<GlobalDataPool*>())
        .def_readwrite("out_dim", &Hash3DAnchored::mlp_out_dim_)
        .def_readwrite("feat_dim", &Hash3DAnchored::feat_dim_)
        .def("AnchoredQuery", &Hash3DAnchored::AnchoredQuery)
        .def("QueryFeature", &Hash3DAnchored::QueryFeature)
        .def("States", &Hash3DAnchored::States)
        .def("LoadStates", &Hash3DAnchored::LoadStates)
        .def("OptimParamGroups", &Hash3DAnchored::OptimParamGroups);

    // torch::python::bind_module<TCNNWP>(m, "TCNNWP")
    //     .def(py::init<GlobalDataPool*,int,int,int,int>());

    // m.def("create_sampler", &create_sampler);
    m.def("CreateGlobalDataPool", &CreateGlobalDataPool);
    m.def("ScatterIdx", &CustomOps::ScatterIdx);
    m.def("ScatterMask", &CustomOps::ScatterMask);
    m.def("WeightVar", &CustomOps::WeightVar);
    m.def("GradientScaling", &CustomOps::GradientScaling);
    m.def("AccumulateSum", &FlexOps::AccumulateSum);
    m.def("FilterIdxBounds", &FilterIdxBounds);
}