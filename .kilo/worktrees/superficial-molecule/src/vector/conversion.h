#pragma once
#include "vector/sparse_vector.h"
#include "vector/dense_vector.h"

DenseVector sparse_to_dense(const SparseVector& x);

SparseVector dense_to_sparse(const DenseVector& x);
