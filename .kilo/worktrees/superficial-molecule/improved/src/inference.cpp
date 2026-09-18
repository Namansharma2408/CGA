#include "inference.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <future>
#include <utility>
#include <string>
#include <map>
#include <vector>
#include <deque>

void KernelStats::update(double t_ms) {
    n_obs++;
    double delta = t_ms - mean_ms;
    mean_ms += delta / n_obs;
    M2 += delta * (t_ms - mean_ms);
    
    if (n_obs == 1) ema_ms = t_ms;
    else ema_ms = ema_alpha * t_ms + (1 - ema_alpha) * ema_ms;

    window.push_back(t_ms);
    if ((int)window.size() > window_size) {
        window.pop_front();
    }
}

double KernelStats::std_ms() const {
    if (n_obs < 2) return 0.0;
    return std::sqrt(std::max(0.0, M2 / (n_obs - 1)));
}

double KernelStats::sw_ucb_penalty(int total_iters) const {
    int effective_n = std::min(n_obs, (int)window.size());
    if (effective_n == 0) return 1e6; 
    return 1.0 * std::sqrt(std::log(total_iters + 1.0) / effective_n);
}

void AdaptiveThreshold::init(double val, double min_v, double max_v, double lr_init, double decay_factor) {
    value = val; lo = min_v; hi = max_v; lr = lr_init; momentum = 0.0; decay = decay_factor;
}

void AdaptiveThreshold::update(double gradient) {
    double beta = 0.9 * std::exp(-std::abs(gradient) * 0.1);
    beta = std::max(0.3, std::min(0.9, beta));
    
    momentum = beta * momentum + (1.0 - beta) * gradient;
    double new_val = value + lr * momentum;
    value = std::max(lo, std::min(hi, new_val));
    lr = std::max(0.001, lr * decay);
}

DTA::DTA(const MatrixStats& ms) {
    double x_start = 0.07, m_start = 0.60, var_start = 100.0;
    if (ms.gini_row > 0.5 && ms.avg_degree > 10) {
        x_start = 0.05; m_start = 0.50; var_start = 50.0;
    } else if (ms.gini_row < 0.3 && ms.avg_degree < 6) {
        x_start = 0.10; m_start = 0.70; var_start = 200.0;
    }
    
    t_x.init(x_start, 0.001, 0.40, 0.05, 0.995);
    t_m.init(m_start, 0.05, 0.99, 0.025, 0.99);
    t_lb.init(1.0, 0.1, 10.0, 0.015, 0.98);
    t_var.init(var_start, 10.0, 1000.0, 0.05, 0.995);

    auto names = {"PM-BHash", "LB-PM-BHash", "PB-MSPA", "LB-PB-MSPA", "LB-MSPA", "SpMV"};
    for (auto n : names) {
        stats[n] = KernelStats();
        stats[n].window_size = 12; 
    }
}

double DTA::ucb_score(const std::string& kernel) const {
    const auto& s = stats.at(kernel);
    if (s.n_obs == 0) {
        double global_median = 0.0;
        int count = 0;
        for (const auto& pair : stats) {
            if (pair.second.n_obs > 0) {
                global_median += pair.second.ema_ms;
                count++;
            }
        }
        if (count > 0) return (global_median / count);
        return 1.0;
    }
    return s.ema_ms - s.sw_ucb_penalty(iteration);
}

std::string DTA::select_kernel(const FeatureVector& feat) {
    iteration++;
    float xd = feat[10], md = feat[12], dgx = feat[16], dvar = feat[4];

    if (iteration % 10 == 0) {
        std::string oldest = "PM-BHash";
        int min_obs = 1e9;
        for (auto const& [k, s] : stats) {
            if (s.n_obs < min_obs) { min_obs = s.n_obs; oldest = k; }
        }
        return oldest;
    }

    std::string greedy;
    if (xd >= t_x.value) greedy = "SpMV";
    else if (md >= t_m.value) {
        greedy = (dvar >= t_var.value || dgx >= t_lb.value) ? "LB-PB-MSPA" : "PB-MSPA";
    } else {
        greedy = (dgx >= t_lb.value) ? "LB-PM-BHash" : "PM-BHash";
    }

    std::string best_ucb = greedy;
    double min_sc = 1e9;
    for (const auto& [k, s] : stats) {
        double sc = ucb_score(k);
        if (sc < min_sc) { min_sc = sc; best_ucb = k; }
    }
    return (best_ucb != greedy && iteration <= 20) ? best_ucb : greedy;
}

void DTA::update(const std::string& kernel, double time_ms, const FeatureVector& feat) {
    auto& s = stats.at(kernel);
    s.update(time_ms);

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
        if (r > 0) {
            t_lb.update(-(r / std::max(time_ms, 1e-9) - 1.0) * feat[16] * 0.4);
            t_var.update(-(r / std::max(time_ms, 1e-9) - 1.0) * feat[4] * 0.1);
        }
    }
}

void InferenceEngine::init_dta(const MatrixStats& ms, DTA_TYPE type) {
    if (dta) delete dta;
    dta = new DTA(ms);
}

std::pair<std::string, std::string> InferenceEngine::select_kernel(
    const FeatureVector& feat, bool x_decreasing)
{
    if (mode == BASELINE) return {"CPU", "LB-MSPA"};

    float x_nnz = feat[9];
    float n_nodes = std::exp(feat[0]);
    if (x_nnz <= 1.0f) return {"CPU", "PM-BHash"}; 
    if (x_nnz > 0.3f * n_nodes) return {"GPU", "SpMV"}; 

    if (mode == DTA_MODE && dta) {
        std::string k = dta->select_kernel(feat);
        return {k == "SpMV" ? "GPU" : "CPU", k};
    }

    std::string platform = m1->predict_name(feat);

    if (platform == "CPU") {
        auto f_cpu = std::async(std::launch::async, [&](){ return m2->predict_name(feat); });
        return {"CPU", f_cpu.get()};
    } else {
        auto f_gpu = std::async(std::launch::async, [&](){ 
            return x_decreasing ? m4->predict_name(feat) : m3->predict_name(feat); 
        });
        std::string k = f_gpu.get();
        return {k == "SpMV" ? "GPU" : "CPU", k};
    }
}
