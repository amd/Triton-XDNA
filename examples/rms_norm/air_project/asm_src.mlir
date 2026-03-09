#loc = loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":24:0)
#loc1 = loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":49:35)
#loc3 = loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":48:23)
#loc6 = loc("/home/strixminipc/Triton-XDNA/sandbox/lib/python3.13/site-packages/triton/language/standard.py":293:36)
#loc7 = loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":45:20)
#loc11 = loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":40:16)
#loc12 = loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":43:17)
#loc15 = loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":52:21)
#loc16 = loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":52:16)
#loc17 = loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":53:13)
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#loc19 = loc("X"(#loc))
#loc20 = loc("Y"(#loc))
#loc21 = loc("M"(#loc))
#loc22 = loc("rstd"(#loc1))
#loc24 = loc("mean_sq"(#loc3))
#loc25 = loc("sum_sq"(#loc7))
#loc28 = loc("x"(#loc11))
#loc29 = loc("x_f32"(#loc12))
#loc32 = loc("y"(#loc15))
#loc33 = loc("y"(#loc16))
#loc34 = loc("y"(#loc17))
#loc35 = loc(callsite(#loc6 at #loc25))
module {
  func.func @rms_norm_kernel(%arg0: memref<*xbf16> {tt.divisibility = 16 : i32} loc("X"(#loc)), %arg1: memref<*xbf16> {tt.divisibility = 16 : i32} loc("Y"(#loc)), %arg2: i32 {tt.divisibility = 16 : i32} loc("M"(#loc)), %arg3: i32 loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":24:0), %arg4: i32 loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":24:0), %arg5: i32 loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":24:0), %arg6: i32 loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":24:0), %arg7: i32 loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":24:0), %arg8: i32 loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":24:0)) {
    %cst = arith.constant 9.99999974E-6 : f32 loc(#loc22)
    %c64 = arith.constant 64 : index loc(#loc23)
    %cst_0 = arith.constant 6.400000e+01 : f32 loc(#loc24)
    %c2_i32 = arith.constant 2 : i32 loc(#loc4)
    %cst_1 = arith.constant 0.000000e+00 : f32 loc(#loc36)
    %0 = tensor.empty() : tensor<2xf32> loc(#loc22)
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2xf32>) -> tensor<2xf32> loc(#loc22)
    %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<2xf32>) -> tensor<2xf32> loc(#loc24)
    %3 = arith.muli %arg6, %c2_i32 : i32 loc(#loc26)
    %4 = arith.index_cast %3 : i32 to index loc(#loc9)
    %5 = arith.muli %4, %c64 : index loc(#loc27)
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [2, 64], strides: [64, 1] : memref<*xbf16> to memref<2x64xbf16, strided<[64, 1], offset: ?>> loc(#loc23)
    %alloc = memref.alloc() : memref<2x64xbf16> loc(#loc28)
    memref.copy %reinterpret_cast, %alloc : memref<2x64xbf16, strided<[64, 1], offset: ?>> to memref<2x64xbf16> loc(#loc28)
    %6 = bufferization.to_tensor %alloc restrict writable : memref<2x64xbf16> to tensor<2x64xbf16> loc(#loc28)
    %7 = tensor.empty() : tensor<2x64xf32> loc(#loc29)
    %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<2x64xbf16>) outs(%7 : tensor<2x64xf32>) {
    ^bb0(%in: bf16 loc("x"(#loc11)), %out: f32 loc("x_f32"(#loc12))):
      %19 = arith.extf %in : bf16 to f32 loc(#loc29)
      linalg.yield %19 : f32 loc(#loc29)
    } -> tensor<2x64xf32> loc(#loc29)
    %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%8, %8 : tensor<2x64xf32>, tensor<2x64xf32>) outs(%8 : tensor<2x64xf32>) {
    ^bb0(%in: f32 loc("x_f32"(#loc12)), %in_3: f32 loc("x_f32"(#loc12)), %out: f32 loc("x_f32"(#loc12))):
      %19 = arith.mulf %in, %in_3 : f32 loc(#loc30)
      linalg.yield %19 : f32 loc(#loc30)
    } -> tensor<2x64xf32> loc(#loc30)
    %10 = tensor.empty() : tensor<64x2xf32> loc(#loc35)
    %transposed = linalg.transpose ins(%9 : tensor<2x64xf32>) outs(%10 : tensor<64x2xf32>) permutation = [1, 0]  loc(#loc35)
    %11 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<2xf32>) -> tensor<2xf32> loc(#loc35)
    %reduced = linalg.reduce ins(%transposed : tensor<64x2xf32>) outs(%11 : tensor<2xf32>) dimensions = [0] 
      (%in: f32 loc(callsite(#loc6 at #loc25)), %init: f32 loc(callsite(#loc6 at #loc25))) {
        %19 = arith.addf %in, %init : f32 loc(#loc35)
        linalg.yield %19 : f32 loc(#loc35)
      } loc(#loc35)
    %12 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%reduced, %2 : tensor<2xf32>, tensor<2xf32>) outs(%reduced : tensor<2xf32>) {
    ^bb0(%in: f32 loc(callsite(#loc6 at #loc25)), %in_3: f32 loc("mean_sq"(#loc3)), %out: f32 loc(callsite(#loc6 at #loc25))):
      %19 = arith.divf %in, %in_3 : f32 loc(#loc24)
      linalg.yield %19 : f32 loc(#loc24)
    } -> tensor<2xf32> loc(#loc24)
    %13 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%12, %1 : tensor<2xf32>, tensor<2xf32>) outs(%12 : tensor<2xf32>) {
    ^bb0(%in: f32 loc("mean_sq"(#loc3)), %in_3: f32 loc("rstd"(#loc1)), %out: f32 loc("mean_sq"(#loc3))):
      %19 = arith.addf %in, %in_3 : f32 loc(#loc22)
      linalg.yield %19 : f32 loc(#loc22)
    } -> tensor<2xf32> loc(#loc22)
    %14 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%13 : tensor<2xf32>) outs(%13 : tensor<2xf32>) {
    ^bb0(%in: f32 loc("rstd"(#loc1)), %out: f32 loc("rstd"(#loc1))):
      %19 = math.rsqrt %in : f32 loc(#loc31)
      linalg.yield %19 : f32 loc(#loc31)
    } -> tensor<2xf32> loc(#loc31)
    %expanded = tensor.expand_shape %14 [[0, 1]] output_shape [2, 1] : tensor<2xf32> into tensor<2x1xf32> loc(#loc32)
    %15 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<2x1xf32>) outs(%7 : tensor<2x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32 loc("y"(#loc15)), %out: f32 loc("y"(#loc16))):
      linalg.yield %in : f32 loc(#loc33)
    } -> tensor<2x64xf32> loc(#loc33)
    %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%8, %15 : tensor<2x64xf32>, tensor<2x64xf32>) outs(%8 : tensor<2x64xf32>) {
    ^bb0(%in: f32 loc("x_f32"(#loc12)), %in_3: f32 loc("y"(#loc16)), %out: f32 loc("x_f32"(#loc12))):
      %19 = arith.mulf %in, %in_3 : f32 loc(#loc33)
      linalg.yield %19 : f32 loc(#loc33)
    } -> tensor<2x64xf32> loc(#loc33)
    %17 = tensor.empty() : tensor<2x64xbf16> loc(#loc34)
    %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<2x64xf32>) outs(%17 : tensor<2x64xbf16>) {
    ^bb0(%in: f32 loc("y"(#loc16)), %out: bf16 loc("y"(#loc17))):
      %19 = arith.truncf %in : f32 to bf16 loc(#loc34)
      linalg.yield %19 : bf16 loc(#loc34)
    } -> tensor<2x64xbf16> loc(#loc34)
    %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%5], sizes: [2, 64], strides: [64, 1] : memref<*xbf16> to memref<2x64xbf16, strided<[64, 1], offset: ?>> loc(#loc9)
    bufferization.materialize_in_destination %18 in writable %reinterpret_cast_2 : (tensor<2x64xbf16>, memref<2x64xbf16, strided<[64, 1], offset: ?>>) -> () loc(#loc18)
    return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":40:20)
#loc4 = loc(unknown)
#loc5 = loc("/home/strixminipc/Triton-XDNA/sandbox/lib/python3.13/site-packages/triton/language/standard.py":263:15)
#loc8 = loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":34:22)
#loc9 = loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":54:17)
#loc10 = loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":39:30)
#loc13 = loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":44:19)
#loc14 = loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":49:25)
#loc18 = loc("/home/strixminipc/Triton-XDNA/examples/rms_norm/rms_norm.py":54:26)
#loc23 = loc("x"(#loc2))
#loc26 = loc("row_start"(#loc8))
#loc27 = loc("offsets"(#loc10))
#loc30 = loc("x_sq"(#loc13))
#loc31 = loc("rstd"(#loc14))
#loc36 = loc(callsite(#loc5 at #loc35))

