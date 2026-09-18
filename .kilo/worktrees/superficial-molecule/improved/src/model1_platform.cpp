#include "models/models.h"

class Model1Platform : public DecisionTreeModel {
public:
    std::string predict_name(const FeatureVector& feat) const override {
        if (feat[8] > 0.5f) {
            if (feat[10] > 0.05f) return "GPU";
            return "GPU";
        } else {
            return "CPU";
        }
    }
};

std::unique_ptr<DecisionTreeModel> create_model1_platform() {
    return std::make_unique<Model1Platform>();
}
