#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "Utils/GlobalDataPool.h"
#include "Utils/CameraUtils.h"
#include "PtsSampler/PtsSamplerFactory.h"
#include "Dataset/Dataset.h"
#include "PtsSampler/PersSampler.h"
// #include "Renderer/Renderer.h"

namespace py = pybind11;

PYBIND11_MODULE(py_f2nerf, m) {
    py::class_<BoundedRays>(m, "BoundedRays")
        .def_readwrite("origins", &BoundedRays::origins)
        .def_readwrite("dirs", &BoundedRays::dirs)
        .def_readwrite("bounds", &BoundedRays::bounds);

    py::class_<SampleResultFlex>(m, "SampleResultFlex")
        .def_readwrite("pts", &SampleResultFlex::pts)
        .def_readwrite("dt", &SampleResultFlex::dt)
        .def_readwrite("t", &SampleResultFlex::t)
        .def_readwrite("anchors", &SampleResultFlex::anchors)
        .def_readwrite("pts_idx_bounds", &SampleResultFlex::pts_idx_bounds)
        .def_readwrite("first_oct_dis", &SampleResultFlex::first_oct_dis)
        .def_readwrite("dirs", &SampleResultFlex::dirs);
    
    py::class_<GlobalDataPool>(m, "GlobalDataPool")
        .def(py::init<const std::string&>())
        .def_readwrite("base_exp_dir", &GlobalDataPool::base_exp_dir_);

    py::class_<Dataset>(m, "Dataset")
        .def(py::init<GlobalDataPool*>())
        .def("RandRaysData", &Dataset::RandRaysData)
        .def("RaysOfCamera", &Dataset::RaysOfCamera);

    // py::class_<PersOctree>(m, "PersOctree")
    //     .def(py::init<int , float , float ,
    //          const torch::Tensor& , const torch::Tensor& , const torch::Tensor& , const torch::Tensor& >());

    py::class_<PersSampler>(m, "PersSampler")
        .def(py::init<GlobalDataPool*>())
        .def("GetSamples", &PersSampler::GetSamples)
        .def("UpdateOctNodes", &PersSampler::UpdateOctNodes);

    // py::class_<Renderer>(m, "Renderer")
    //     .def(py::init<GlobalDataPool*, int>());

    // m.def("create_sampler", &create_sampler);
    m.def("PoseInterpolate", &PoseInterpolate);
    m.def("ConstructPtsSampler", &ConstructPtsSampler);
}