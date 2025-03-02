#ifndef OPENMC_RQMC_H
#define OPENMC_RQMC_H

#pragma once

#include "openmc/vector.h"
#include <cstdint>

namespace openmc{

vector<double> halton_rand(int64_t dim, uint64_t* seed, int64_t index);

vector<double> halton(int64_t dim, int64_t index);

vector<uint32_t> sobol(int32_t dim, uint32_t index);

vector<double> sobol_shuffled_scrambled(int32_t dim, uint64_t* seed, uint32_t index);

inline uint32_t hash_combine(uint32_t seed, uint32_t v)
{
  return seed ^ (v + (seed << 6) + (seed >> 2));
}

inline uint32_t reverse_bits(uint32_t x)
{
  x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
  x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
  x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
  x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
  return ((x >> 16) | (x << 16));
}

inline uint32_t laine_karras_permutation(uint32_t x, uint32_t seed) {
  x += seed;
  x ^= x * 0x6c50b47cu;
  x ^= x * 0xb82f1e52u;
  x ^= x * 0xc7afe638u;
  x ^= x * 0x8d22f6e6u;
  return x;
}

inline uint32_t nested_uniform_scramble_base2(uint32_t x, uint32_t seed) {
  x = reverse_bits(x);
  x = laine_karras_permutation(x, seed);
  x = reverse_bits(x);
  return x;
}


} // namespace openmc

#endif // OPENMC_RQMC_H
