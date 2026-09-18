#pragma once
#include "models/features.h"
#include <string>
#include <vector>
#include <memory>

class DecisionTreeModel {
public:
    virtual ~DecisionTreeModel() = default;

    virtual std::string predict_name(const FeatureVector& feat) const = 0;
};

std::unique_ptr<DecisionTreeModel> create_model1_platform();
std::unique_ptr<DecisionTreeModel> create_model2_cpu();
std::unique_ptr<DecisionTreeModel> create_model3_gpu();
std::unique_ptr<DecisionTreeModel> create_model4_universal();
