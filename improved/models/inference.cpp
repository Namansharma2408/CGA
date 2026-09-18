#include "models/inference.h"
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
    // Round-robin on ties (all last_used_iter==0 at start) so the forced
    // probe does not deterministically return the first map entry.
    std::string oldest;
    int min_iter = INT_MAX;
    std::vector<std::string> tied;
    for (const auto& [k, s] : stats) {
        if (s.last_used_iter < min_iter) {
            min_iter = s.last_used_iter;
            tied.clear();
            tied.push_back(k);
        } else if (s.last_used_iter == min_iter) {
            tied.push_back(k);
        }
    }
    if (tied.empty()) return "";
    if (tied.size() == 1) return tied[0];
    return tied[(iteration / 10) % tied.size()];
}

std::string DTA::select_kernel(const FeatureVector& feat) {
    iteration++;
    float xd    = feat[F25_X_DENSITY];
    float md    = feat[F25_M_DENSITY];
    // P0-4: degree_variance now at index 19 (was 4, which is col_cv).
    float dv    = feat[F25_DEGREE_VAR];

    // P0-10 variant: forced probe respects Model-0 (dense frontiers stay SpMV).
    if (enable_forced_probe && iteration % 10 == 0) {
        std::string oldest = oldest_kernel();
        if (xd >= t_x.value) return "SpMV";  // do not violate dense->GPU rule
        if (!oldest.empty()) return oldest;
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
        explore_left--;
        if (best_ucb != greedy) return best_ucb;
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

    auto rival_ema_plain = [&](const std::vector<std::string>& rivals) -> double {
        double mn = 1e9;
        for (const auto& r : rivals)
            if (stats.at(r).n_obs > 0) mn = std::min(mn, stats.at(r).ema_ms);
        return mn == 1e9 ? -1.0 : mn;
    };
    if (kernel == "SpMV") {
        double r = enable_counterfactual ? rival_ema_or_prior(
            {"PM-BHash", "LB-PM-BHash", "PB-MSPA", "LB-PB-MSPA", "LB-MSPA"})
            : rival_ema_plain({"PM-BHash", "LB-PM-BHash", "PB-MSPA", "LB-PB-MSPA", "LB-MSPA"});
        if (confident(r, time_ms))
            t_x.update(-(r / std::max(time_ms, 1e-9) - 1.0) * feat[F25_X_DENSITY]);
    } else if (kernel == "PM-BHash" || kernel == "LB-PM-BHash") {
        double r = enable_counterfactual ? rival_ema_or_prior({"PB-MSPA", "LB-PB-MSPA"})
            : rival_ema_plain({"PB-MSPA", "LB-PB-MSPA"});
        if (confident(r, time_ms))
            t_m.update((r / std::max(time_ms, 1e-9) - 1.0) * feat[F25_M_DENSITY]);
    }

    if (kernel == "LB-PM-BHash" || kernel == "LB-PB-MSPA") {
        std::string non_lb = (kernel == "LB-PM-BHash") ? "PM-BHash" : "PB-MSPA";
        double r = enable_counterfactual ? rival_ema_or_prior({non_lb})
            : rival_ema_plain({non_lb});
        if (confident(r, time_ms)) {
            t_var.update(-(r / std::max(time_ms, 1e-9) - 1.0) * feat[F25_DEGREE_VAR] * 0.2);
        }
        t_lb.update(-(r / std::max(time_ms, 1e-9) - 1.0) * feat[F25_DEGREE_X] * 0.4);
    }
}


void InferenceEngine::init_dta(const MatrixStats& ms, DTA_TYPE type) {
    dta = std::make_unique<DTA>(ms);
    // P1-9: variants gate improvements (default BASE_DTA enables all for
    // backward compat with run_inference.py --dta-type base|ucb|gradient).
    if (type == BASE_DTA) {
        dta->enable_forced_probe = false;
        dta->enable_counterfactual = false;
    } else if (type == UCB_DTA) {
        dta->enable_forced_probe = true;
        dta->enable_counterfactual = false;
    } else {  // GRADIENT_DTA: full improved stack
        dta->enable_forced_probe = true;
        dta->enable_counterfactual = true;
    }
}

std::pair<std::string, std::string> InferenceEngine::select_kernel(
    const FeatureVector& feat, bool x_decreasing, int x_nnz, int n)
{
    // Model-0 prefilter (same as basic execution_flow; duplicated here so the
    // improved engine is correct even when called directly).
    if (x_nnz == 1)
        return {"CPU", "PM-BHash"};
    if (n > 0 && x_nnz > n * 0.30f)
        return {"GPU", "SpMV"};

    if (mode == BASELINE) return {"CPU", "LB-MSPA"};

    if (mode == DTA_MODE && dta) {
        bool on_gpu = false;
        // P0-10: use accessor (iteration is protected).
        if (dta->get_iteration() == 0) {
            std::string plat = m1->predict_name(feat);
            on_gpu = (plat == "GPU");
        }

        std::string k = dta->select_kernel(feat);

        if (on_gpu && x_decreasing) {
            std::string k4 = m4->predict_name(feat);
            // P0-10: return k4 (universal one-copy decision), not k.
            if (k4 != "SpMV" && k4 != "Sort-Based SpMSpV") {
                return {"CPU", k4};
            }
        }
        // GPU iff selected kernel is a GPU kernel (SpMV family).
        bool is_gpu = (k == "SpMV" || k == "Sort-Based SpMSpV" || k == "Merge-Based SpMV");
        return {is_gpu ? "GPU" : "CPU", k};
    }

    // P1-8: sequential inference — depth-3/7 trees are ~µs; std::async
    // spawn overhead dominates per-BFS-level. predict_name is const and
    // thread-safe, but sequential is faster end-to-end (see SPEC.md).
    std::string platform   = m1->predict_name(feat);
    std::string cpu_kernel = m2->predict_name(feat);
    std::string uni_kernel = m4->predict_name(feat);

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
