#include "models/inference.h"
#include "core/kernel_registry.h"
#include <cmath>
#include <algorithm>
#include <iostream>

void KernelStats::update(double t_ms) {
    run.update(t_ms);
    n_obs = run.n_obs;
    mean_ms = run.mean;
    M2 = run.M2;
    ema.alpha = ema_alpha;
    ema.update(t_ms);
    ema_ms = ema.ema;
    ema_alpha = ema.alpha;
}

double KernelStats::std_ms() const {
    return run.std();
}

void AdaptiveThreshold::init(double val, double min_v, double max_v, double lr_init) {
    impl.init(val, min_v, max_v, lr_init, 0.99);
    value = impl.value; lo = impl.lo; hi = impl.hi; lr = impl.lr; momentum = impl.momentum;
}

void AdaptiveThreshold::update(double gradient) {
    impl.update(gradient);
    value = impl.value; lr = impl.lr; momentum = impl.momentum;
}

DTA::DTA(const MatrixStats& ms) {
    double x_start = 0.07, m_start = 0.60;
    if (ms.gini_row > 0.5 && ms.avg_degree > 10) {
        x_start = 0.05; m_start = 0.50;
    } else if (ms.gini_row < 0.3 && ms.avg_degree < 6) {
        x_start = 0.10; m_start = 0.70;
    }
    t_x.init(x_start, 0.001, 0.40, 0.05);
    t_m.init(m_start, 0.05, 0.99, 0.025);
    t_lb.init(1.0, 0.1, 10.0, 0.015);

    // P0-5: all 8 dispatchable kernels are candidates (sort/merge reachable
    // via UCB exploration even though greedy only emits SpMV/CPU).
    for (const auto& n : all_kernel_names()) stats[n] = KernelStats();
}

double DTA::ucb_score(const std::string& kernel) const {
    const auto& s = stats.at(kernel);
    if (s.n_obs == 0) return -1e9;
    // P1-5: cap window to UCB_WINDOW=20 so bonus does not decay to 0 forever.
    double eff_n = std::min<double>(s.n_obs, 20);
    double bonus = 1.0 * std::sqrt(std::log(dta_state.iteration + 1.0) / eff_n);
    return s.ema_ms - bonus;
}

std::string DTA::select_kernel(const FeatureVector& feat) {
    dta_state.iteration++;
    float xd = feat[F_X_DENSITY], md = feat[F_M_DENSITY], dgx = feat[F_DEGREE_X];

    std::string greedy;
    if (xd >= t_x.value) greedy = "SpMV";
    else if (md >= t_m.value) greedy = (dgx >= t_lb.value) ? "LB-PB-MSPA" : "PB-MSPA";
    else greedy = (dgx >= t_lb.value) ? "LB-PM-BHash" : "PM-BHash";

    // P1-5: decrement explore budget whenever we explore, not only on override.
    if (dta_state.explore_left > 0 && dta_state.iteration <= 6) {
        std::string best_ucb = greedy;
        double min_sc = 1e9;
        for (const auto& [k, s] : stats) {
            double sc = ucb_score(k);
            if (sc < min_sc) { min_sc = sc; best_ucb = k; }
        }
        dta_state.explore_left--;
        if (best_ucb != greedy) return best_ucb;
    }
    return greedy;
}

void DTA::update(const std::string& kernel, double time_ms, const FeatureVector& feat) {
    auto& s = stats.at(kernel);
    s.update(time_ms);

    if (s.n_obs > 5 && s.std_ms() > 0) {
        double z = (time_ms - s.mean_ms) / s.std_ms();
        if (z > 2.5) s.ema_alpha = std::min(0.75, s.ema_alpha * 1.6);
        else s.ema_alpha = std::max(0.20, s.ema_alpha * 0.98);
    }

    auto rival_ema = [&](const std::vector<std::string>& rivals) -> double {
        double mn = 1e9;
        for (const auto& r : rivals)
            if (stats.at(r).n_obs > 0) mn = std::min(mn, stats.at(r).ema_ms);
        return mn == 1e9 ? -1.0 : mn;
    };

    if (kernel == "SpMV") {
        double r = rival_ema({"PM-BHash", "LB-PM-BHash", "PB-MSPA", "LB-PB-MSPA", "LB-MSPA"});
        if (r > 0) t_x.update(-(r / std::max(time_ms, 1e-9) - 1.0) * feat[F_X_DENSITY]);
    } else if (kernel == "PM-BHash" || kernel == "LB-PM-BHash") {
        double r = rival_ema({"PB-MSPA", "LB-PB-MSPA"});
        if (r > 0) t_m.update((r / std::max(time_ms, 1e-9) - 1.0) * feat[F_M_DENSITY]);
    }

    if (kernel == "LB-PM-BHash" || kernel == "LB-PB-MSPA") {
        std::string non_lb = (kernel == "LB-PM-BHash") ? "PM-BHash" : "PB-MSPA";
        double r = rival_ema({non_lb});
        if (r > 0) t_lb.update(-(r / std::max(time_ms, 1e-9) - 1.0) * feat[F_DEGREE_X] * 0.4);
    }
}

std::pair<std::string, std::string> InferenceEngine::select_kernel(
    const FeatureVector& feat, bool x_decreasing)
{
    if (mode == BASELINE) return {"CPU", "LB-MSPA"};
    if (mode == DTA_MODE && dta) {
        std::string k = dta->select_kernel(feat);
        return {k == "SpMV" ? "GPU" : "CPU", k};
    }

    std::string platform = m1->predict_name(feat);

    if (platform == "CPU") {
        return {"CPU", m2->predict_name(feat)};
    } else {
        if (x_decreasing) {
            std::string k4 = m4->predict_name(feat);
            if (k4 != "SpMV") return {"CPU", k4};
            else return {"GPU", k4};
        } else {
            return {"GPU", m3->predict_name(feat)};
        }
    }
}
