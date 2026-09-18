#pragma once
#include <vector>
#include <cstdint>
#include <utility>

int bucket_index(int row_i, int n_bucket, int m);

int default_n_buckets(int n_threads = 8);

using Bucket = std::vector<std::pair<int, float>>;
