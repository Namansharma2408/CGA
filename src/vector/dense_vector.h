#pragma once
#include <vector>
#include <cstdint>

struct DenseVector {
    int n{0};
    std::vector<float> data;

    explicit DenseVector(int n, float fill = 0.0f);

    float  operator[](int i) const { return data[i]; }
    float& operator[](int i)       { return data[i]; }

    void fill_zero();
    int  count_nonzero() const;
};
