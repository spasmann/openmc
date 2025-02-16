#ifndef OPENMC_RQMC_H
#define OPENMC_RQMC_H

#include "openmc/vector.h"

namespace openmc{

vector<double> rhalton(int64_t dim, uint64_t* seed, int64_t skip = 0);

} // namespace openmc

#endif // OPENMC_RQMC_H