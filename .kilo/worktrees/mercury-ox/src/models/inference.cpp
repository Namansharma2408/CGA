#include "models/inference.h"
#include <cmath>
#include <algorithm>
#include <iostream>

void KernelStats::update(double t_ms) {
    n_obs++;
    double delta = t_ms - mean_ms;
    mean_ms += delta / n_obs;
    M2 += delta * (t_ms - mean_ms);
    if (n_obs == 1) ema_ms = t_ms;
    else ema_ms = ema_alpha * t_ms + (1 - ema_alpha) * ema_ms;
}

double KernelStats::std_ms() const {
    if (n_obs < 2) return 0.0;
    return std::sqrt(std::max(0.0, M2 / (n_obs - 1)));
}

void AdaptiveThreshold::init(double val, double min_v, double max_v, double lr_init) {
    value = val; lo = min_v; hi = max_v; lr = lr_init; momentum = 0.0;
}

void AdaptiveThreshold::update(double gradient) {
    momentum = 0.9 * momentum + 0.1 * gradient;
    double new_val = value + lr * momentum;
    value = std::max(lo, std::min(hi, new_val));
    lr = std::max(0.001, lr * 0.99);
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

    auto names = {"PM-BHash", "LB-PM-BHash", "PB-MSPA", "LB-PB-MSPA", "LB-MSPA", "SpMV"};
    for (auto n : names) stats[n] = KernelStats();
}

double DTA::ucb_score(const std::string& kernel) const {
    const auto& s = stats.at(kernel);
    if (s.n_obs == 0) return -1e9;
    double bonus = 1.0 * std::sqrt(std::log(iteration + 1.0) / s.n_obs);
    return s.ema_ms - bonus;
}

std::string DTA::select_kernel(const FeatureVector& feat) {
    iteration++;
    float xd = feat[10], md = feat[12], dgx = feat[16];

    std::string greedy;
    if (xd >= t_x.value) greedy = "SpMV";
    else if (md >= t_m.value) greedy = (dgx >= t_lb.value) ? "LB-PB-MSPA" : "PB-MSPA";
    else greedy = (dgx >= t_lb.value) ? "LB-PM-BHash" : "PM-BHash";

    if (explore_left > 0 && iteration <= 6) {
        std::string best_ucb = greedy;
        double min_sc = 1e9;
        for (const auto& [k, s] : stats) {
            double sc = ucb_score(k);
            if (sc < min_sc) { min_sc = sc; best_ucb = k; }
        }
        if (best_ucb != greedy) {
            explore_left--;
            return best_ucb;
        }
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
        if (r > 0) t_x.update(-(r / std::max(time_ms, 1e-9) - 1.0) * feat[10]);
    } else if (kernel == "PM-BHash" || kernel == "LB-PM-BHash") {
        double r = rival_ema({"PB-MSPA", "LB-PB-MSPA"});
        if (r > 0) t_m.update((r / std::max(time_ms, 1e-9) - 1.0) * feat[12]);
    }

    if (kernel == "LB-PM-BHash" || kernel == "LB-PB-MSPA") {
        std::string non_lb = (kernel == "LB-PM-BHash") ? "PM-BHash" : "PB-MSPA";
        double r = rival_ema({non_lb});
        if (r > 0) t_lb.update(-(r / std::max(time_ms, 1e-9) - 1.0) * feat[16] * 0.4);
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
