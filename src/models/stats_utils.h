#pragma once
// P1-6: separated concerns — RunningStats (Welford), EmaTracker (EMA),
// SlidingWindow (capped deque), ThresholdAdaptor (clamped gradient step).
#include <algorithm>
#include <cmath>
#include <deque>

struct RunningStats {
    int n_obs{0};
    double mean{0.0};
    double M2{0.0};
    void update(double x) {
        n_obs++;
        double d = x - mean;
        mean += d / n_obs;
        M2 += d * (x - mean);
    }
    double std() const {
        if (n_obs < 2) return 0.0;
        return std::sqrt(std::max(0.0, M2 / (n_obs - 1)));
    }
};

struct EmaTracker {
    double ema{0.0};
    double alpha{0.3};
    bool init{false};
    void update(double x) {
        if (!init) { ema = x; init = true; }
        else ema = alpha * x + (1.0 - alpha) * ema;
    }
    void adapt_alpha(double z) {
        if (z > 2.5) alpha = std::min(0.75, alpha * 1.6);
        else alpha = std::max(0.20, alpha * 0.98);
    }
};

struct SlidingWindow {
    std::deque<double> w;
    int cap{20};
    void push(double x) {
        w.push_back(x);
        while ((int)w.size() > cap) w.pop_front();
    }
    int count() const { return (int)w.size(); }
    double ucb_bonus(int total_iters) const {
        int n = count();
        if (n == 0) return 1e6;
        double t = std::min(total_iters, cap);
        return std::sqrt(std::log(t + 1.0) / n);
    }
};

struct ThresholdAdaptor {
    double value{0.0}, lo{0.0}, hi{1.0}, lr{0.05}, momentum{0.0}, decay{0.99};
    void init(double v, double mn, double mx, double l, double d = 0.99) {
        value = v; lo = mn; hi = mx; lr = l; decay = d; momentum = 0.0;
    }
    void update(double grad) {
        momentum = 0.9 * momentum + 0.1 * grad;
        value = std::max(lo, std::min(hi, value + lr * momentum));
        lr = std::max(0.001, lr * decay);
    }
    void update_adaptive(double grad) {  // improved dampened beta
        double beta = 0.9 * std::exp(-std::fabs(grad) * 0.1);
        beta = std::max(0.3, std::min(0.9, beta));
        momentum = beta * momentum + (1.0 - beta) * grad;
        value = std::max(lo, std::min(hi, value + lr * momentum));
        lr = std::max(0.001, lr * decay);
    }
};
