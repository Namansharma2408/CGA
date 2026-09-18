#pragma once
#include <vector>
#include <utility>
#include <cstdint>

class BucketHashTable {
    struct Entry {
        int key{-1};
        float val{0.0f};
    };
    std::vector<Entry> table;
    int capacity;
    int mask;

public:
    explicit BucketHashTable(int size_hint);

    void insert(int key, float val);

    void collect(std::vector<int>& out_idx, std::vector<float>& out_val) const;
};
