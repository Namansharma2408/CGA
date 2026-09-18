#pragma once
#include "models/features.h"
#include "models/models.h"
#include "models/stats_utils.h"
#include "core/kernel_registry.h"
#include <map>
#include <vector>
#include <memory>

// P1-6: thin wrappers over separated utils (kept for compat; new code
// should use RunningStats/EmaTracker/SlidingWindow/ThresholdAdaptor).
struct KernelStats {
    RunningStats run;
    EmaTracker ema;
    int n_obs{0};
    double mean_ms{0.0};
    double M2{0.0};
    double ema_ms{0.0};
    double ema_alpha{0.3};

    void update(double t_ms);
    double std_ms() const;
};

struct AdaptiveThreshold {
    ThresholdAdaptor impl;
    double value{0.0};
    double lo{0.0}, hi{0.0};
    double lr{0.05}, momentum{0.0};

    void init(double val, double min_v, double max_v, double learn_rate);
    void update(double gradient);
};

// P1-5/P1-9: explicit state + virtual interface (matches improved DTA).
struct DTAState {
    int iteration{0};
    int explore_left{3};
};

class DTAInterface {
public:
    virtual ~DTAInterface() = default;
    virtual std::string select_kernel(const FeatureVector& feat) = 0;
    virtual void update(const std::string& kernel, double time_ms, const FeatureVector& feat) = 0;
    virtual DTAState state() const = 0;
};

class DTA : public DTAInterface {
public:
    explicit DTA(const MatrixStats& ms);
    ~DTA() override = default;

    std::string select_kernel(const FeatureVector& feat) override;
    void update(const std::string& kernel, double time_ms, const FeatureVector& feat) override;
    DTAState state() const override { return dta_state; }
    int get_iteration() const { return dta_state.iteration; }
    void reset() { dta_state = DTAState{}; }

private:
    DTAState dta_state;
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
    std::unique_ptr<DTA> dta;

public:
    // P2-9: STATIC is the paper's "static ML" mode; STATIC_ML is the
    // unambiguous alias (avoids confusion with the `static` keyword).
    enum Mode { BASELINE, STATIC, STATIC_ML_ALIAS_UNUSED = STATIC, DTA_MODE };
    static constexpr Mode STATIC_ML = STATIC;
    Mode mode{STATIC};

    InferenceEngine() {
        m1 = create_model1_platform();
        m2 = create_model2_cpu();
        m3 = create_model3_gpu();
        m4 = create_model4_universal();
    }
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&) = default;
    InferenceEngine& operator=(InferenceEngine&&) = default;

    // P1-1: RAII ownership, no new/delete at call sites.
    void init_dta(const MatrixStats& ms) {
        if (mode == DTA_MODE) dta = std::make_unique<DTA>(ms);
    }
    // Test hook.
    bool has_dta() const { return (bool)dta; }

    std::pair<std::string, std::string> select_kernel(
        const FeatureVector& feat, bool x_decreasing);

    void update_dta(const std::string& kernel, double time_ms, const FeatureVector& feat) {
        if (dta) dta->update(kernel, time_ms, feat);
    }
};
