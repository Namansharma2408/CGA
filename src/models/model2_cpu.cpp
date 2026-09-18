#include "models/models.h"

class Model2CPU : public DecisionTreeModel {
public:
    std::string predict_name(const FeatureVector& feat) const override {
        float xd = feat[F_X_DENSITY];
        float md = feat[F_M_DENSITY];
        float dgx = feat[F_DEGREE_X];

        if (xd > 0.05f) {
            return "LB-MSPA";
        }
        if (md > 0.50f) {
            if (dgx > 2.0f) return "LB-PB-MSPA";
            else return "PB-MSPA";
        }
        if (dgx > 2.0f) return "LB-PM-BHash";
        return "PM-BHash";
    }
};

std::unique_ptr<DecisionTreeModel> create_model2_cpu() {
    return std::make_unique<Model2CPU>();
}
