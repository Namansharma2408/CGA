#include "cpu/utils/bucket.h"

int bucket_index(int row_i, int n_bucket, int m) {
    int idx = (n_bucket * (row_i + 1) - 1) / m;
    return (idx < n_bucket) ? idx : n_bucket - 1;
}

int default_n_buckets(int n_threads) {
    return 4 * n_threads;
}
