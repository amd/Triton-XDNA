#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @rms_norm_kernel(%arg0: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    air.launch (%arg9, %arg10, %arg11) in (%arg12=%c16, %arg13=%c1, %arg14=%c1) args(%arg15=%arg0, %arg16=%arg1) : memref<*xbf16>, memref<*xbf16> {
      air.segment @rms_norm_kernel_0  args(%arg17=%arg10, %arg18=%arg15, %arg19=%arg16) : index, memref<*xbf16>, memref<*xbf16> {
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c2 = arith.constant 2 : index
        %c1_0 = arith.constant 1 : index
        %c128 = arith.constant 128 : index
        %0 = arith.muli %arg17, %c128 : index
        %alloc = memref.alloc() : memref<2x64xbf16, 1 : i32>
        air.dma_memcpy_nd (%alloc[] [] [], %arg18[%c0, %0] [%c2, %c64] [%c64, %c1_0]) {id = 1 : i32} : (memref<2x64xbf16, 1 : i32>, memref<*xbf16>)
        %alloc_1 = memref.alloc() : memref<2x64xbf16, 1>
        air.herd @herd_0  tile (%arg20, %arg21) in (%arg22=%c2, %arg23=%c1_0) args(%arg24=%alloc, %arg25=%alloc_1) : memref<2x64xbf16, 1 : i32>, memref<2x64xbf16, 1> {
          %c64_2 = arith.constant 64 : index
          %c1_3 = arith.constant 1 : index
          %c0_4 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %cst_5 = arith.constant 6.400000e+01 : f32
          %cst_6 = arith.constant 9.99999974E-6 : f32
          %subview = memref.subview %arg24[%arg20, 0] [1, 64] [1, 1] : memref<2x64xbf16, 1 : i32> to memref<1x64xbf16, strided<[64, 1], offset: ?>, 1 : i32>
          %alloc_7 = memref.alloc() : memref<1xf32, 2>
          memref.store %cst, %alloc_7[%c0_4] : memref<1xf32, 2>
          linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%subview : memref<1x64xbf16, strided<[64, 1], offset: ?>, 1 : i32>) outs(%alloc_7 : memref<1xf32, 2>) {
          ^bb0(%in: bf16, %out: f32):
            %1 = arith.extf %in : bf16 to f32
            %2 = arith.mulf %1, %1 : f32
            %3 = arith.addf %2, %out : f32
            linalg.yield %3 : f32
          }
          %alloc_8 = memref.alloc() : memref<1x64xbf16, 2>
          linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%subview, %alloc_7 : memref<1x64xbf16, strided<[64, 1], offset: ?>, 1 : i32>, memref<1xf32, 2>) outs(%alloc_8 : memref<1x64xbf16, 2>) {
          ^bb0(%in: bf16, %in_9: f32, %out: bf16):
            %1 = arith.divf %in_9, %cst_5 : f32
            %2 = arith.addf %1, %cst_6 : f32
            %3 = math.rsqrt %2 : f32
            %4 = arith.extf %in : bf16 to f32
            %5 = arith.mulf %4, %3 : f32
            %6 = arith.truncf %5 : f32 to bf16
            linalg.yield %6 : bf16
          }
          memref.dealloc %alloc_7 : memref<1xf32, 2>
          memref.dealloc %alloc_8 : memref<1x64xbf16, 2>
        }
        air.dma_memcpy_nd (%arg19[%c0, %0] [%c2, %c64] [%c64, %c1_0], %alloc_1[] [] []) {id = 2 : i32} : (memref<*xbf16>, memref<2x64xbf16, 1>)
        memref.dealloc %alloc_1 : memref<2x64xbf16, 1>
      }
    }
    return
  }
}
