#include "openmc/random_dist.h"
#include "openmc/vector.h"
#include "openmc/simulation.h"

namespace openmc {

//==============================================================================
// Randomized Halton (Owen, 2017)
//==============================================================================

// Implementation of the Fisher-Yates shuffle algorithm.
// Algorithm adapted from:
//    https://en.cppreference.com/w/cpp/algorithm/random_shuffle#Version_3
void fisher_yates_shuffle(vector<int64_t>& arr, uint64_t* seed)
{
  // Loop over the array from the last element down to the second
  for (int i = arr.size() - 1; i > 0; --i) {
    // Generate a random index in the range [0, i]
    int j = uniform_int_distribution(0, i, seed);
    std::swap(arr[i], arr[j]);
  }
}

// Function to generate randomized Halton sequence samples
//
// Algorithm adapted from:
//      A. B. Owen. A randomized halton algorithm in r. Arxiv, 6 2017.
//      URL https://arxiv.org/abs/1706.02808
vector<double> rhalton(int64_t dim, uint64_t* seed, int64_t skip = 0)
{
  if (dim > 10) {
    fatal_error("Halton sampling dimension too large");
  }
  int64_t b, res, dig;
  double b2r, ans;
const std::array<int64_t, 10> primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
  vector<double> halton(dim, 0.0);
  vector<int64_t> perm;

  for (int D = 0; D < dim; ++D) {
    b = primes[D];
    perm.resize(b);
    b2r = 1.0 / b;
    res = skip;
    ans = 0.0;

    while ((1.0 - b2r) < 1.0) {
      std::iota(perm.begin(), perm.end(), 0);
      fisher_yates_shuffle(perm, seed);
      dig = res % b;
      ans += perm[dig] * b2r;
      res = (res - dig) / b;
      b2r /= b;
    }

    halton[D] = ans;
  }

  return halton;
}

} // namespace openmc