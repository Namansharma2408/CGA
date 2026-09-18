#include "vector/dense_vector.h"
#include <numeric>
#include <cstring>

DenseVector::DenseVector(int n, float fill) : n(n), data(n, fill) {}

void DenseVector::fill_zero() { std::fill(data.begin(), data.end(), 0.0f); }

int DenseVector::count_nonzero() const {
    int cnt = 0;
    for (float v : data) if (v != 0.0f) ++cnt;
    return cnt;
}
