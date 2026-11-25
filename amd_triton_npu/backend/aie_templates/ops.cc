// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The aie_api calls tl.extra.air.aie's template ops make, one a function:
// operands native vectors by value (a 64-lane operand as two 32-lane
// halves), results stored through `o`. ops.mlir is the IR Peano's clang -O2
// emits for them (scripts/gen_aie_templates.py), which backend/aie_ops.py
// uses as each op's lowering.
#include <aie_api/aie.hpp>
using BV = aie::block_vector<bfp16ebs8, 64>;
using ACC = aie::accum<accfloat, 64>;
using VB = aie::vector<bfloat16, 64>;
using VF = aie::vector<float, 64>;

static inline VF f64(v32float a, v32float b) {
  return aie::concat(aie::vector<float, 32>(a), aie::vector<float, 32>(b));
}
static inline VB b64(v32bfloat16 a, v32bfloat16 b) {
  return aie::concat(aie::vector<bfloat16, 32>(a),
                     aie::vector<bfloat16, 32>(b));
}
static inline void sf(v32float *o, VF r) {
  o[0] = r.extract<32>(0);
  o[1] = r.extract<32>(1);
}
static inline void sb(v32bfloat16 *o, VB r) {
  o[0] = r.extract<32>(0);
  o[1] = r.extract<32>(1);
}

extern "C" {
// aie::mul(float, float) (fp32 emulated on bf16), to_vector<float>
void op_mul_ff(v32float a0, v32float a1, v32float b0, v32float b1,
               v32float *o) {
  sf(o, aie::mul(f64(a0, a1), f64(b0, b1)).to_vector<float>());
}
// aie::sub(bf16, bf16) of 64 lanes
void op_sub_bb(v32bfloat16 a0, v32bfloat16 a1, v32bfloat16 b0, v32bfloat16 b1,
               v32bfloat16 *o) {
  sb(o, aie::sub(b64(a0, a1), b64(b0, b1)));
}
// aie::max(bf16, bf16) of 64 lanes
void op_max_bb(v32bfloat16 a0, v32bfloat16 a1, v32bfloat16 b0, v32bfloat16 b1,
               v32bfloat16 *o) {
  sb(o, aie::max(b64(a0, a1), b64(b0, b1)));
}
// aie::inv(float) of 64 lanes
void op_inv(v32float a0, v32float a1, v32float *o) {
  sf(o, aie::inv(f64(a0, a1)));
}

// helpers of attention kernels
static inline v8bfloat16 row_max_(VB v) {
  VB t = aie::transpose(v, 8, 8);
  aie::vector<bfloat16, 32> a = aie::max(t.extract<32>(0), t.extract<32>(1));
  aie::vector<bfloat16, 16> b = aie::max(a.extract<16>(0), a.extract<16>(1));
  return aie::max(b.extract<8>(0), b.extract<8>(1));
}
static inline VB rows8a_(aie::vector<bfloat16, 8> v) {
  auto a = aie::interleave_zip(v, v, 1);
  aie::vector<bfloat16, 16> b = aie::concat(a.first, a.second);
  auto c = aie::interleave_zip(b, b, 2);
  aie::vector<bfloat16, 32> d = aie::concat(c.first, c.second);
  auto e = aie::interleave_zip(d, d, 4);
  return aie::concat(e.first, e.second);
}
// row_max(v): the max of each row of an 8x8 block
void op_row_max(v32bfloat16 a0, v32bfloat16 a1, v8bfloat16 *o) {
  o[0] = row_max_(b64(a0, a1));
}
// rows8(v) (attention's, by interleave_zip): lane r * 8 + c = v[r]
void op_rows8a(v8bfloat16 a, v32bfloat16 *o) {
  sb(o, rows8a_(aie::vector<bfloat16, 8>(a)));
}
// aie::max of 8 lanes
void op_max8(v8bfloat16 a, v8bfloat16 b, v8bfloat16 *o) {
  o[0] = aie::max(aie::vector<bfloat16, 8>(a), aie::vector<bfloat16, 8>(b));
}
// aie::concat of two 8-lane vectors
void op_cat8(v8bfloat16 a, v8bfloat16 b, v16bfloat16 *o) {
  o[0] = aie::concat(aie::vector<bfloat16, 8>(a), aie::vector<bfloat16, 8>(b));
}
// aie::sub of 16 lanes
void op_sub16(v16bfloat16 a, v16bfloat16 b, v16bfloat16 *o) {
  o[0] = aie::sub(aie::vector<bfloat16, 16>(a), aie::vector<bfloat16, 16>(b));
}
// aie::mul(v, s).to_vector<float>(), 16 and 64 lanes
void op_mulsc16(v16bfloat16 a, bfloat16 s, v16float *o) {
  o[0] = aie::mul(aie::vector<bfloat16, 16>(a), s).to_vector<float>();
}
void op_mulsc64(v32bfloat16 a0, v32bfloat16 a1, bfloat16 s, v32float *o) {
  sf(o, aie::mul(b64(a0, a1), s).to_vector<float>());
}
// aie::exp2<bfloat16>, 16 and 64 lanes
void op_exp2_16(v16float a, v16bfloat16 *o) {
  o[0] = aie::exp2<bfloat16>(aie::vector<float, 16>(a));
}
void op_exp2_64(v32float a0, v32float a1, v32bfloat16 *o) {
  sb(o, aie::exp2<bfloat16>(f64(a0, a1)));
}
// aie::le(d, 0), aie::gt(d, 0) of 64 int16 lanes: the mask's bits
void op_le0(v32int16 d0, v32int16 d1, unsigned long long *o) {
  o[0] = aie::le(aie::concat(aie::vector<int16, 32>(d0),
                             aie::vector<int16, 32>(d1)),
                 int16(0))
             .to_uint64();
}
void op_gt0(v32int16 d0, v32int16 d1, unsigned long long *o) {
  o[0] = aie::gt(aie::concat(aie::vector<int16, 32>(d0),
                             aie::vector<int16, 32>(d1)),
                 int16(0))
             .to_uint64();
}
// aie::select(a, b, m) of 64 bf16 lanes
void op_select64(v32bfloat16 a0, v32bfloat16 a1, v32bfloat16 b0, v32bfloat16 b1,
                 unsigned long long m, v32bfloat16 *o) {
  sb(o, aie::select(b64(a0, a1), b64(b0, b1), aie::mask<64>::from_uint64(m)));
}
// acc.to_vector<bfloat16>() of 64 lanes
void op_srs64(v32float a0, v32float a1, v32bfloat16 *o) {
  ACC c;
  c.from_vector(f64(a0, a1));
  sb(o, c.to_vector<bfloat16>());
}
// v.extract<8>(0) of 16 lanes
void op_ext8(v16bfloat16 a, v8bfloat16 *o) {
  o[0] = aie::vector<bfloat16, 16>(a).extract<8>(0);
}
}
