#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    air.launch (%arg9, %arg10, %arg11) in (%arg12=%c4, %arg13=%c2, %arg14=%c1) args(%arg15=%arg0, %arg16=%arg1, %arg17=%arg2) : memref<*xf32>, memref<*xf32>, memref<*xf32> {
      air.segment @kernel_0  args(%arg18=%arg9, %arg19=%arg10, %arg20=%arg15, %arg21=%arg16, %arg22=%arg17) : index, index, memref<*xf32>, memref<*xf32>, memref<*xf32> {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %c256 = arith.constant 256 : index
        %c2_1 = arith.constant 2 : index
        %c4_2 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c8 = arith.constant 8 : index
        %c2048 = arith.constant 2048 : index
        %c32 = arith.constant 32 : index
        %0 = arith.muli %arg19, %c32 : index
        %1 = arith.muli %arg18, %c2048 : index
        %2 = arith.addi %1, %0 : index
        %alloc = memref.alloc() : memref<8x16x4x2xf32, 1 : i32>
        air.dma_memcpy_nd (%alloc[] [] [], %arg20[%c0, %c0, %c0, %2] [%c8, %c16, %c4_2, %c2_1] [%c256, %c2_1, %c64, %c1_0]) {id = 1 : i32} : (memref<8x16x4x2xf32, 1 : i32>, memref<*xf32>)
        %3 = bufferization.to_tensor %alloc restrict writable : memref<8x16x4x2xf32, 1 : i32> to tensor<8x16x4x2xf32>
        %alloc_3 = memref.alloc() : memref<8x16x4x2xf32, 1 : i32>
        air.dma_memcpy_nd (%alloc_3[] [] [], %arg21[%c0, %c0, %c0, %2] [%c8, %c16, %c4_2, %c2_1] [%c256, %c2_1, %c64, %c1_0]) {id = 2 : i32} : (memref<8x16x4x2xf32, 1 : i32>, memref<*xf32>)
        %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<8x16x4x2xf32, 1 : i32> to tensor<8x16x4x2xf32>
        %5 = bufferization.alloc_tensor() : tensor<8x16x4x2xf32>
        %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<8x16x4x2xf32>, tensor<8x16x4x2xf32>) outs(%5 : tensor<8x16x4x2xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %7 = arith.mulf %in, %in_4 : f32
          linalg.yield %7 : f32
        } -> tensor<8x16x4x2xf32>
        %reinterpret_cast = memref.reinterpret_cast %arg22 to offset: [%2], sizes: [8, 16, 4, 2], strides: [256, 2, 64, 1] : memref<*xf32> to memref<8x16x4x2xf32, strided<[256, 2, 64, 1], offset: ?>>
        bufferization.materialize_in_destination %6 in writable %reinterpret_cast : (tensor<8x16x4x2xf32>, memref<8x16x4x2xf32, strided<[256, 2, 64, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}
