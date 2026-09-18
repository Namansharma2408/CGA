#include "vector/conversion.h"

DenseVector sparse_to_dense(const SparseVector& x) {
    DenseVector d(x.n, 0.0f);
    for (int k = 0; k < x.nnz; ++k)
        d.data[x.indices[k]] = x.values[k];
    return d;
}

SparseVector dense_to_sparse(const DenseVector& x) {
    SparseVector s(x.n);
    for (int i = 0; i < x.n; ++i)
        if (x.data[i] != 0.0f)
            s.push(i, x.data[i]);
    return s;
}
