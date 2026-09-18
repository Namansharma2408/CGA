#include "cpu/utils/load_balance.h"
#include <numeric>
#include <algorithm>

void static_load_balance(
    const CSCMatrix& A,
    std::vector<int>& xi,
    std::vector<float>& xv)
{
    std::vector<int> col_nnz(xi.size());
    for(size_t k = 0; k < xi.size(); ++k) {
        int j = xi[k];
        col_nnz[k] = A.indptr[j+1] - A.indptr[j];
    }

    std::vector<int> perm(xi.size());
    std::iota(perm.begin(), perm.end(), 0);

    std::sort(perm.begin(), perm.end(), [&](int a, int b) {
        return col_nnz[a] > col_nnz[b];
    });

    std::vector<int>   new_xi(xi.size());
    std::vector<float> new_xv(xv.size());
    for(size_t k = 0; k < perm.size(); ++k) {
        new_xi[k] = xi[perm[k]];
        new_xv[k] = xv[perm[k]];
    }

    xi = std::move(new_xi);
    xv = std::move(new_xv);
}
