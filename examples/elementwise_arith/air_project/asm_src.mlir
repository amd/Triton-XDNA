#loc = loc("/home/strixminipc/Triton-XDNA/examples/elementwise_arith/elementwise_arith.py":86:1)
#loc5 = loc("/home/strixminipc/Triton-XDNA/examples/elementwise_arith/elementwise_arith.py":89:9)
#map = affine_map<(d0) -> (d0)>
#loc8 = loc("X"(#loc))
#loc9 = loc("OUT"(#loc))
#loc12 = loc("x"(#loc5))
module {
  func.func @square_kernel(%arg0: memref<*xi16> {tt.divisibility = 16 : i32} loc("X"(#loc)), %arg1: memref<*xi16> {tt.divisibility = 16 : i32} loc("OUT"(#loc)), %arg2: i32 loc("/home/strixminipc/Triton-XDNA/examples/elementwise_arith/elementwise_arith.py":86:1), %arg3: i32 loc("/home/strixminipc/Triton-XDNA/examples/elementwise_arith/elementwise_arith.py":86:1), %arg4: i32 loc("/home/strixminipc/Triton-XDNA/examples/elementwise_arith/elementwise_arith.py":86:1), %arg5: i32 loc("/home/strixminipc/Triton-XDNA/examples/elementwise_arith/elementwise_arith.py":86:1), %arg6: i32 loc("/home/strixminipc/Triton-XDNA/examples/elementwise_arith/elementwise_arith.py":86:1), %arg7: i32 loc("/home/strixminipc/Triton-XDNA/examples/elementwise_arith/elementwise_arith.py":86:1)) {
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %0 = arith.muli %arg5, %c1024_i32 : i32 loc(#loc10)
    %1 = arith.index_cast %0 : i32 to index loc(#loc3)
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [1024], strides: [1] : memref<*xi16> to memref<1024xi16, strided<[1], offset: ?>> loc(#loc11)
    %alloc = memref.alloc() : memref<1024xi16> loc(#loc12)
    memref.copy %reinterpret_cast, %alloc : memref<1024xi16, strided<[1], offset: ?>> to memref<1024xi16> loc(#loc12)
    %2 = bufferization.to_tensor %alloc restrict writable : memref<1024xi16> to tensor<1024xi16> loc(#loc12)
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [1024], strides: [1] : memref<*xi16> to memref<1024xi16, strided<[1], offset: ?>> loc(#loc3)
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%2, %2 : tensor<1024xi16>, tensor<1024xi16>) outs(%2 : tensor<1024xi16>) {
    ^bb0(%in: i16 loc("x"(#loc5)), %in_1: i16 loc("x"(#loc5)), %out: i16 loc("x"(#loc5))):
      %4 = arith.muli %in, %in_1 : i16 loc(#loc6)
      linalg.yield %4 : i16 loc(#loc6)
    } -> tensor<1024xi16> loc(#loc6)
    bufferization.materialize_in_destination %3 in writable %reinterpret_cast_0 : (tensor<1024xi16>, memref<1024xi16, strided<[1], offset: ?>>) -> () loc(#loc7)
    return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/strixminipc/Triton-XDNA/examples/elementwise_arith/elementwise_arith.py":88:15)
#loc3 = loc("/home/strixminipc/Triton-XDNA/examples/elementwise_arith/elementwise_arith.py":90:14)
#loc4 = loc("/home/strixminipc/Triton-XDNA/examples/elementwise_arith/elementwise_arith.py":89:17)
#loc6 = loc("/home/strixminipc/Triton-XDNA/examples/elementwise_arith/elementwise_arith.py":90:32)
#loc7 = loc("/home/strixminipc/Triton-XDNA/examples/elementwise_arith/elementwise_arith.py":90:5)
#loc10 = loc("offsets"(#loc2))
#loc11 = loc("x"(#loc4))

