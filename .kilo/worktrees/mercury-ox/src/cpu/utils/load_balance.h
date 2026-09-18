#pragma once
#include <vector>
#include "matrix/csc.h"

void static_load_balance(
    const CSCMatrix& A,
    std::vector<int>& xi,
    std::vector<float>& xv);
