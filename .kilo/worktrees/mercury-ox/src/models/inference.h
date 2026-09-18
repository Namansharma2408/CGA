#pragma once
#include "models/features.h"
#include "models/models.h"
#include <map>
#include <vector>
#include <memory>

struct KernelStats {
    int n_obs{0};
    double mean_ms{0.0};
    double M2{0.0};
    double ema_ms{0.0};
    double ema_alpha{0.3};

    void update(double t_ms);
    double std_ms() const;
};

struct AdaptiveThreshold {
    double value{0.0};
    double lo{0.0}, hi{0.0};
    double lr{0.05}, momentum{0.0};

    void init(double val, double min_v, double max_v, double learn_rate);
    void update(double gradient);
};

class DTA {
public:
    DTA(const MatrixStats& ms);

    std::string select_kernel(const FeatureVector& feat);
    void update(const std::string& kernel, double time_ms, const FeatureVector& feat);

private:
    int iteration{0};
    int explore_left{3};
    std::map<std::string, KernelStats> stats;
    AdaptiveThreshold t_x;
    AdaptiveThreshold t_m;
    AdaptiveThreshold t_lb;

    double ucb_score(const std::string& kernel) const;
};

class InferenceEngine {
    std::unique_ptr<DecisionTreeModel> m1;
    std::unique_ptr<DecisionTreeModel> m2;
    std::unique_ptr<DecisionTreeModel> m3;
    std::unique_ptr<DecisionTreeModel> m4;
    DTA* dta{nullptr};

public:
    enum Mode { BASELINE, STATIC, DTA_MODE };
    Mode mode{STATIC};

    InferenceEngine() {
        m1 = create_model1_platform();
        m2 = create_model2_cpu();
        m3 = create_model3_gpu();
        m4 = create_model4_universal();
    }

    void init_dta(const MatrixStats& ms) { if(mode==DTA_MODE) dta = new DTA(ms); }
    ~InferenceEngine() { delete dta; }

    std::pair<std::string, std::string> select_kernel(
        const FeatureVector& feat, bool x_decreasing);

    void update_dta(const std::string& kernel, double time_ms, const FeatureVector& feat) {
        if (dta) dta->update(kernel, time_ms, feat);
    }
};
