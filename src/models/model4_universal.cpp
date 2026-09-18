#include "models/models.h"

class Model4Universal : public DecisionTreeModel {
public:
    std::string predict_name(const FeatureVector& feat) const override {
        float xd = feat[F_X_DENSITY];

        if (xd > 0.04f || feat[F_IS_SCALE_FREE] < 0.5f) {
            return "SpMV";
        }
        return "LB-MSPA";
    }
};

std::unique_ptr<DecisionTreeModel> create_model4_universal() {
    return std::make_unique<Model4Universal>();
}
