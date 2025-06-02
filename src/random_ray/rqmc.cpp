#include "openmc/random_dist.h"
#include "openmc/vector.h"
#include "openmc/simulation.h"
#include "openmc/random_ray/rqmc.h"

namespace openmc {

constexpr float S = float(1.0/(1ul<<32));

//==============================================================================
// Hash Operation
//==============================================================================

uint32_t hash(uint32_t x)
{
    // finalizer from murmurhash3
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

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
vector<double> halton_rand(int64_t dim, uint64_t* seed, int64_t index)
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
    res = index;
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

//==============================================================================
// Halton Sequence
//==============================================================================
vector<double> halton(int64_t dim, int64_t index)
{
  if (dim > 10) {
    fatal_error("Halton sampling dimension too large");
  }
  int64_t b, res, dig;
  double b2r, ans;
  const std::array<int64_t, 10> primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
  vector<double> halton(dim, 0.0);

  for (int D = 0; D < dim; ++D) {
    b = primes[D];
    b2r = 1.0 / b;
    res = index+1;
    ans = 0.0;

    while ((1.0 - b2r) < 1.0) {
      ans += (res % b) * b2r;
      b2r /= b;
      res = floor(res/b);
    }

    halton[D] = ans;
  }

  return halton;
}

//==============================================================================
// Sobol Sequence (Hash Based)
//==============================================================================

uint32_t directions[5][32] = {
  0x80000000, 0x40000000, 0x20000000, 0x10000000,
  0x08000000, 0x04000000, 0x02000000, 0x01000000,
  0x00800000, 0x00400000, 0x00200000, 0x00100000,
  0x00080000, 0x00040000, 0x00020000, 0x00010000,
  0x00008000, 0x00004000, 0x00002000, 0x00001000,
  0x00000800, 0x00000400, 0x00000200, 0x00000100,
  0x00000080, 0x00000040, 0x00000020, 0x00000010,
  0x00000008, 0x00000004, 0x00000002, 0x00000001,

  0x80000000, 0xc0000000, 0xa0000000, 0xf0000000,
  0x88000000, 0xcc000000, 0xaa000000, 0xff000000,
  0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000,
  0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000,
  0x80008000, 0xc000c000, 0xa000a000, 0xf000f000,
  0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
  0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0,
  0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff,

  0x80000000, 0xc0000000, 0x60000000, 0x90000000,
  0xe8000000, 0x5c000000, 0x8e000000, 0xc5000000,
  0x68800000, 0x9cc00000, 0xee600000, 0x55900000,
  0x80680000, 0xc09c0000, 0x60ee0000, 0x90550000,
  0xe8808000, 0x5cc0c000, 0x8e606000, 0xc5909000,
  0x6868e800, 0x9c9c5c00, 0xeeee8e00, 0x5555c500,
  0x8000e880, 0xc0005cc0, 0x60008e60, 0x9000c590,
  0xe8006868, 0x5c009c9c, 0x8e00eeee, 0xc5005555,

  0x80000000, 0xc0000000, 0x20000000, 0x50000000,
  0xf8000000, 0x74000000, 0xa2000000, 0x93000000,
  0xd8800000, 0x25400000, 0x59e00000, 0xe6d00000,
  0x78080000, 0xb40c0000, 0x82020000, 0xc3050000,
  0x208f8000, 0x51474000, 0xfbea2000, 0x75d93000,
  0xa0858800, 0x914e5400, 0xdbe79e00, 0x25db6d00,
  0x58800080, 0xe54000c0, 0x79e00020, 0xb6d00050,
  0x800800f8, 0xc00c0074, 0x200200a2, 0x50050093,

  0x80000000, 0x40000000, 0x20000000, 0xb0000000,
  0xf8000000, 0xdc000000, 0x7a000000, 0x9d000000,
  0x5a800000, 0x2fc00000, 0xa1600000, 0xf0b00000,
  0xda880000, 0x6fc40000, 0x81620000, 0x40bb0000,
  0x22878000, 0xb3c9c000, 0xfb65a000, 0xddb2d000,
  0x78022800, 0x9c0b3c00, 0x5a0fb600, 0x2d0ddb00,
  0xa2878080, 0xf3c9c040, 0xdb65a020, 0x6db2d0b0,
  0x800228f8, 0x400b3cdc, 0x200fb67a, 0xb00ddb9d,
};


vector<uint32_t> sobol(int32_t dim, uint32_t index)
{
  vector<uint32_t> samples(dim, 0);

  for (int d = 0; d < dim; d++) {
    for (int bit = 0; bit < 32; bit++) {
      int mask = (index >> bit) & 1;
      samples[d] ^= mask * directions[d][bit];
    }
  }
  return samples;
}

vector<double> sobol_shuffled_scrambled(int32_t dim, uint64_t* seed, uint32_t index)
{
  if (dim > 5) {
    fatal_error("Sobol sampling dimension too large");
  }
  vector<double> samples(dim);

  uint32_t seed_32 = uint32_t(*seed); // cast to 32 bit
  uint32_t hseed = hash(seed_32);

  uint32_t i = nested_uniform_scramble_base2(index, hseed);
  vector<uint32_t> sobol_samples = sobol(dim, i);

  for (int d = 0; d < dim; d++) {
    samples[d] = double(nested_uniform_scramble_base2(sobol_samples[d], hash_combine(hseed, d)) * S);
  }

  return samples;
}

} // namespace openmc