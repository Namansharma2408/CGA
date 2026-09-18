#include "models/models.h"

class Model3GPU : public DecisionTreeModel {
public:
    std::string predict_name(const FeatureVector& feat) const override {
        float xd = feat[10];

        if (xd < 0.01f) {
            return "Sort-Based SpMSpV";
        } else {
            float row_cv = feat[3];
            if (row_cv > 1.5f) {
                return "Merge-Based SpMV";
            }
            return "SpMV";
        }
    }
};

std::unique_ptr<DecisionTreeModel> create_model3_gpu() {
    return std::make_unique<Model3GPU>();
}
