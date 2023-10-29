//
// Created by ppwang on 2022/9/16.
//

#include "GlobalDataPool.h"

GlobalDataPool::GlobalDataPool(const std::string &config_path) {
  config_ = YAML::LoadFile(config_path);
  learning_rate_ = config_["train"]["learning_rate"].as<float>();
}

std::unique_ptr<GlobalDataPool> CreateGlobalDataPool(const std::string &config_path, 
                                                      std::string &base_exp_dir, 
                                                      torch::Tensor &c2w_train, 
                                                      torch::Tensor &w2c_train, 
                                                      torch::Tensor &intri_train, 
                                                      torch::Tensor &bounds_train) {
  auto data_pool = std::make_unique<GlobalDataPool>(config_path);
  data_pool->base_exp_dir_ = base_exp_dir;
  data_pool->c2w_train_ = c2w_train;
  data_pool->w2c_train_ = w2c_train;
  data_pool->intri_train_ = intri_train;
  data_pool->bounds_train_ = bounds_train;
  return data_pool;
}