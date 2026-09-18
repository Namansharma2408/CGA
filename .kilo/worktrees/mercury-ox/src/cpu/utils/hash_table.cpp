#include "cpu/utils/hash_table.h"

BucketHashTable::BucketHashTable(int size_hint) {
    capacity = 1;
    int target = size_hint * 2 + 1;
    while (capacity < target) capacity <<= 1;
    mask = capacity - 1;
    table.resize(capacity);
}

void BucketHashTable::insert(int key, float val) {
    int h = (257 * key) & mask;
    while (true) {
        if (table[h].key == -1) {
            table[h].key = key;
            table[h].val = val;
            return;
        } else if (table[h].key == key) {
            table[h].val += val;
            return;
        }
        h = (h + 1) & mask;
    }
}

void BucketHashTable::collect(std::vector<int>& out_idx, std::vector<float>& out_val) const {
    out_idx.reserve(table.size() / 2);
    out_val.reserve(table.size() / 2);
    for (int i = 0; i < capacity; ++i) {
        if (table[i].key != -1) {
            out_idx.push_back(table[i].key);
            out_val.push_back(table[i].val);
        }
    }
}
