#map = affine_map<()[s0] -> (s0 * 256)>
module {
  func.func @square_kernel(%arg0: memref<*xi16> {tt.divisibility = 16 : i32}, %arg1: memref<*xi16> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c1 = arith.constant 1 : index
    air.launch (%arg8, %arg9, %arg10) in (%arg11=%c1, %arg12=%c1, %arg13=%c1) args(%arg14=%arg0, %arg15=%arg1) : memref<*xi16>, memref<*xi16> {
      air.segment @square_kernel_0  args(%arg16=%arg8, %arg17=%arg14, %arg18=%arg15) : index, memref<*xi16>, memref<*xi16> {
        %c1024 = arith.constant 1024 : index
        %c4 = arith.constant 4 : index
        %c1_0 = arith.constant 1 : index
        %0 = arith.muli %arg16, %c1024 : index
        %alloc = memref.alloc() : memref<1024xi16, 1 : i32>
        air.dma_memcpy_nd (%alloc[] [] [], %arg17[%0] [%c1024] [%c1_0]) {id = 1 : i32} : (memref<1024xi16, 1 : i32>, memref<*xi16>)
        %alloc_1 = memref.alloc() : memref<1024xi16, 1>
        air.herd @herd_0  tile (%arg19, %arg20) in (%arg21=%c4, %arg22=%c1_0) args(%arg23=%alloc, %arg24=%alloc_1) : memref<1024xi16, 1 : i32>, memref<1024xi16, 1> {
          %1 = ub.poison : i16
          %c1_2 = arith.constant 1 : index
          %c0 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c32 = arith.constant 32 : index
          %2 = affine.apply #map()[%arg19]
          %alloc_3 = memref.alloc() : memref<256xi16, 2>
          air.dma_memcpy_nd (%alloc_3[] [] [], %arg23[%2] [%c256] [%c1_2]) {id = 1 : i32} : (memref<256xi16, 2>, memref<1024xi16, 1 : i32>)
          %alloc_4 = memref.alloc() : memref<256xi16, 2>
          scf.for %arg25 = %c0 to %c256 step %c32 {
            %subview = memref.subview %alloc_3[%arg25] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
            %subview_5 = memref.subview %alloc_4[%arg25] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
            %3 = vector.transfer_read %subview[%c0], %1 {in_bounds = [true]} : memref<32xi16, strided<[1], offset: ?>, 2>, vector<32xi16>
            %4 = arith.muli %3, %3 : vector<32xi16>
            vector.transfer_write %4, %subview_5[%c0] {in_bounds = [true]} : vector<32xi16>, memref<32xi16, strided<[1], offset: ?>, 2>
          }
          air.dma_memcpy_nd (%arg24[%2] [%c256] [%c1_2], %alloc_4[] [] []) {id = 2 : i32} : (memref<1024xi16, 1>, memref<256xi16, 2>)
          memref.dealloc %alloc_3 : memref<256xi16, 2>
          memref.dealloc %alloc_4 : memref<256xi16, 2>
        }
        air.dma_memcpy_nd (%arg18[%0] [%c1024] [%c1_0], %alloc_1[] [] []) {id = 2 : i32} : (memref<*xi16>, memref<1024xi16, 1>)
        memref.dealloc %alloc_1 : memref<1024xi16, 1>
      }
    }
    return
  }
}
