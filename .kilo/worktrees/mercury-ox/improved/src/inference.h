#pragma once
#include "features.h"
#include "models.h"
#include <map>
#include <vector>
#include <memory>
#include <deque>
#include <future>
#include <string>
#include <utility>

struct KernelStats {
    int n_obs{0};
    double mean_ms{0.0};
    double M2{0.0};
    double ema_ms{0.0};
    double ema_alpha{0.3};
    std::deque<double> window;
    int window_size{10};

    void update(double t_ms);
    double std_ms() const;
    double sw_ucb_penalty(int total_iters) const;
};

struct AdaptiveThreshold {
    double value{0.0};
    double lo{0.0}, hi{0.0};
    double lr{0.05}, momentum{0.0};
    double decay{0.99};

    void init(double val, double min_v, double max_v, double learn_rate, double decay_factor = 0.99);
    void update(double gradient);
};

class DTA {
public:
    virtual ~DTA() = default;
    DTA(const MatrixStats& ms);

    virtual std::string select_kernel(const FeatureVector& feat);
    virtual void update(const std::string& kernel, double time_ms, const FeatureVector& feat);

protected:
    int iteration{0};
    int explore_left{3};
    std::map<std::string, KernelStats> stats;
    AdaptiveThreshold t_x;
    AdaptiveThreshold t_m;
    AdaptiveThreshold t_lb;
    AdaptiveThreshold t_var;

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
    enum DTA_TYPE { BASE_DTA, UCB_DTA, GRADIENT_DTA };

    InferenceEngine() {
        m1 = create_model1_platform();
        m2 = create_model2_cpu();
        m3 = create_model3_gpu();
        m4 = create_model4_universal();
    }

    void init_dta(const MatrixStats& ms, DTA_TYPE type = BASE_DTA);
    ~InferenceEngine() { delete dta; }

    std::pair<std::string, std::string> select_kernel(
        const FeatureVector& feat, bool x_decreasing);

    void update_dta(const std::string& kernel, double time_ms, const FeatureVector& feat) {
        if (dta) dta->update(kernel, time_ms, feat);
    }
};
