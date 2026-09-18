#include "inference.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <future>
#include <climits>


void KernelStats::update(double t_ms, int iteration) {
    n_obs++;
    last_used_iter = iteration;

    double delta = t_ms - mean_ms;
    mean_ms += delta / n_obs;
    M2 += delta * (t_ms - mean_ms);

    if (n_obs == 1) ema_ms = t_ms;
    else ema_ms = ema_alpha * t_ms + (1 - ema_alpha) * ema_ms;

    window.push_back(t_ms);
    if ((int)window.size() > UCB_WINDOW)
        window.pop_front();
}

double KernelStats::std_ms() const {
    if (n_obs < 2) return 0.0;
    return std::sqrt(std::max(0.0, M2 / (n_obs - 1)));
}


void AdaptiveThreshold::init(double val, double min_v, double max_v,
                              double learn_rate, double lr_decay) {
    value = val;
    lo = min_v; hi = max_v;
    lr = learn_rate;
    decay = lr_decay;
    momentum = 0.0;
}

void AdaptiveThreshold::update(double gradient) {
    double beta = 0.9 * std::exp(-std::fabs(gradient) * 0.1);
    beta = std::max(0.3, std::min(0.9, beta));
    momentum = beta * momentum + (1.0 - beta) * gradient;

    double new_val = value + lr * momentum;
    value = std::max(lo, std::min(hi, new_val));

    lr = std::max(0.001, lr * decay);
}


DTA::DTA(const MatrixStats& ms) {
    double x_start = 0.07, m_start = 0.60;
    if (ms.gini_row > 0.5 && ms.avg_degree > 10) {
        x_start = 0.05; m_start = 0.50;
    } else if (ms.gini_row < 0.3 && ms.avg_degree < 6) {
        x_start = 0.10; m_start = 0.70;
    }
    t_x.init(x_start, 0.001, 0.40, 0.05,  0.995);
    t_m.init(m_start, 0.05,  0.99, 0.025, 0.990);
    t_lb.init(1.0,    0.1,   10.0, 0.015, 0.980);
    double var_start = ms.row_std * ms.row_std * 0.01;
    t_var.init(var_start, 0.0, 1e6, 0.01, 0.975);

    auto names = {"PM-BHash", "LB-PM-BHash", "PB-MSPA",
                  "LB-PB-MSPA", "LB-MSPA", "SpMV"};
    for (auto n : names) stats[n] = KernelStats{};
}

double DTA::ucb_score(const std::string& kernel) const {
    const auto& s = stats.at(kernel);
    int wc = s.window_count();
    if (wc == 0) return -1e9;

    double t_capped = std::min(iteration, KernelStats::UCB_WINDOW);
    double bonus = 1.0 * std::sqrt(std::log(t_capped + 1.0) / wc);
    return s.ema_ms - bonus;
}

double DTA::global_median_ema() const {
    std::vector<double> emas;
    for (const auto& [k, s] : stats)
        if (s.n_obs > 0) emas.push_back(s.ema_ms);
    if (emas.empty()) return 1.0;
    std::sort(emas.begin(), emas.end());
    return emas[emas.size() / 2];
}

std::string DTA::oldest_kernel() const {
    std::string oldest;
    int min_iter = INT_MAX;
    for (const auto& [k, s] : stats) {
        if (s.last_used_iter < min_iter) {
            min_iter = s.last_used_iter;
            oldest   = k;
        }
    }
    return oldest;
}

std::string DTA::select_kernel(const FeatureVector& feat) {
    iteration++;
    float xd    = feat[10];
    float md    = feat[12];
    // [HIGHLIGHT] Fix F: Using improved feature index 4 (degree_variance) for selection
    float dv    = feat[4]; 

    if (iteration % 10 == 0) {
        return oldest_kernel();
    }

    std::string greedy;
    if (xd >= t_x.value) {
        greedy = "SpMV";
    } else if (md >= t_m.value) {
        greedy = (dv >= t_var.value) ? "LB-PB-MSPA" : "PB-MSPA";
    } else {
        greedy = (dv >= t_var.value) ? "LB-PM-BHash" : "PM-BHash";
    }

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

void DTA::update(const std::string& kernel, double time_ms,
                 const FeatureVector& feat)
{
    auto& s = stats.at(kernel);
    s.update(time_ms, iteration);

    if (s.n_obs > 5 && s.std_ms() > 0) {
        double z = (time_ms - s.mean_ms) / s.std_ms();
        if (z > 2.5) s.ema_alpha = std::min(0.75, s.ema_alpha * 1.6);
        else         s.ema_alpha = std::max(0.20, s.ema_alpha * 0.98);
    }

    auto rival_ema_or_prior = [&](const std::vector<std::string>& rivals) -> double {
        double mn = 1e9;
        for (const auto& r : rivals)
            if (stats.at(r).n_obs > 0)
                mn = std::min(mn, stats.at(r).ema_ms);
        return (mn == 1e9) ? global_median_ema() : mn;
    };

    auto confident = [](double rival, double self) -> bool {
        return self > 1e-9 && (rival / self - 1.0) > 0.15;
    };

    if (kernel == "SpMV") {
        double r = rival_ema_or_prior(
            {"PM-BHash", "LB-PM-BHash", "PB-MSPA", "LB-PB-MSPA", "LB-MSPA"});
        if (confident(r, time_ms))
            t_x.update(-(r / std::max(time_ms, 1e-9) - 1.0) * feat[10]);
    } else if (kernel == "PM-BHash" || kernel == "LB-PM-BHash") {
        double r = rival_ema_or_prior({"PB-MSPA", "LB-PB-MSPA"});
        if (confident(r, time_ms))
            t_m.update((r / std::max(time_ms, 1e-9) - 1.0) * feat[12]);
    }

    if (kernel == "LB-PM-BHash" || kernel == "LB-PB-MSPA") {
        std::string non_lb = (kernel == "LB-PM-BHash") ? "PM-BHash" : "PB-MSPA";
        double r = rival_ema_or_prior({non_lb});
        if (confident(r, time_ms)) {
            // [HIGHLIGHT] Fix F: Using improved feature index 4 (degree_variance) for t_var
            t_var.update(-(r / std::max(time_ms, 1e-9) - 1.0) * feat[4] * 0.2);
        }
        t_lb.update(-(r / std::max(time_ms, 1e-9) - 1.0) * feat[16] * 0.4);
    }
}


std::pair<std::string, std::string> InferenceEngine::select_kernel(
    const FeatureVector& feat, bool x_decreasing, int x_nnz, int n)
{
    if (x_nnz == 1)
        return {"CPU", "PM-BHash"};
    if (n > 0 && x_nnz > n * 0.30f)
        return {"GPU", "SpMV"};

    if (mode == BASELINE) return {"CPU", "LB-MSPA"};

    if (mode == DTA_MODE && dta) {
        bool on_gpu = false;
        if (dta->iteration == 0) {
            // [HIGHLIGHT] Passing 25-dim Improved Features to Platform Model (M1)
            std::string plat = m1->predict_name(feat);
            on_gpu = (plat == "GPU");
        }

        std::string k = dta->select_kernel(feat);

        if (on_gpu && x_decreasing) {
            // [HIGHLIGHT] Passing 25-dim Improved Features to Universal Model (M4)
            std::string k4 = m4->predict_name(feat);
            if (k4 != "SpMV" && k4 != "Sort-Based SpMSpV") {
                return {"CPU", k};
            }
        }
        return {k == "SpMV" ? "GPU" : "CPU", k};
    }

    // [HIGHLIGHT] Parallel Inference using 25-dim Improved Features
    auto f2 = std::async(std::launch::async,
        [&]{ return m2->predict_name(feat); });
    auto f4 = std::async(std::launch::async,
        [&]{ return m4->predict_name(feat); });

    std::string platform   = m1->predict_name(feat);
    std::string cpu_kernel = f2.get();
    std::string uni_kernel = f4.get();

    if (platform == "CPU") {
        return {"CPU", cpu_kernel};
    } else {
        if (x_decreasing) {
            if (uni_kernel != "SpMV") return {"CPU", uni_kernel};
            else                      return {"GPU", uni_kernel};
        } else {
            return {"GPU", m3->predict_name(feat)};
        }
    }
}
