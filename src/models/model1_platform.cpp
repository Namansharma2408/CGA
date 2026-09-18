#include "models/models.h"

class Model1Platform : public DecisionTreeModel {
public:
    std::string predict_name(const FeatureVector& feat) const override {
        // P0-6: both arms previously returned GPU. Scale-free + dense
        // frontier -> GPU, otherwise CPU (branch-covered, see test_models).
        if (feat[F_IS_SCALE_FREE] > 0.5f) {
            if (feat[F_X_DENSITY] > 0.05f) return "GPU";
            return "CPU";
        } else {
            return "CPU";
        }
    }
};

std::unique_ptr<DecisionTreeModel> create_model1_platform() {
    return std::make_unique<Model1Platform>();
}
