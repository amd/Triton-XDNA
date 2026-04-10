#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  aie.device(npu2) @sub_kernel_0 {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %lock_1_1 = aie.lock(%mem_tile_1_1, 1) {init = 4 : i32}
    %lock_1_1_0 = aie.lock(%mem_tile_1_1, 0) {init = 0 : i32}
    %lock_0_1 = aie.lock(%mem_tile_0_1, 1) {init = 4 : i32}
    %lock_0_1_1 = aie.lock(%mem_tile_0_1, 0) {init = 0 : i32}
    %lock_2_1 = aie.lock(%mem_tile_2_1, 1) {init = 4 : i32}
    %lock_2_1_2 = aie.lock(%mem_tile_2_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_5 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_6 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_7 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 5) {init = 1 : i32}
    %lock_0_3_8 = aie.lock(%tile_0_3, 4) {init = 0 : i32}
    %lock_0_3_9 = aie.lock(%tile_0_3, 3) {init = 1 : i32}
    %lock_0_3_10 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_11 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_12 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 5) {init = 1 : i32}
    %lock_0_4_13 = aie.lock(%tile_0_4, 4) {init = 0 : i32}
    %lock_0_4_14 = aie.lock(%tile_0_4, 3) {init = 1 : i32}
    %lock_0_4_15 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_16 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_17 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %lock_0_5 = aie.lock(%tile_0_5, 5) {init = 1 : i32}
    %lock_0_5_18 = aie.lock(%tile_0_5, 4) {init = 0 : i32}
    %lock_0_5_19 = aie.lock(%tile_0_5, 3) {init = 1 : i32}
    %lock_0_5_20 = aie.lock(%tile_0_5, 2) {init = 0 : i32}
    %lock_0_5_21 = aie.lock(%tile_0_5, 1) {init = 1 : i32}
    %lock_0_5_22 = aie.lock(%tile_0_5, 0) {init = 0 : i32}
    %buf14 = aie.buffer(%mem_tile_0_1) {sym_name = "buf14"} : memref<1024xi16, 1 : i32> 
    %buf13 = aie.buffer(%mem_tile_1_1) {sym_name = "buf13"} : memref<1024xi16, 1 : i32> 
    %buf12 = aie.buffer(%mem_tile_2_1) {sym_name = "buf12"} : memref<1024xi16, 1> 
    %buf11 = aie.buffer(%tile_0_5) {sym_name = "buf11"} : memref<256xi16, 2> 
    %buf10 = aie.buffer(%tile_0_5) {sym_name = "buf10"} : memref<256xi16, 2> 
    %buf9 = aie.buffer(%tile_0_5) {sym_name = "buf9"} : memref<256xi16, 2> 
    %buf8 = aie.buffer(%tile_0_4) {sym_name = "buf8"} : memref<256xi16, 2> 
    %buf7 = aie.buffer(%tile_0_4) {sym_name = "buf7"} : memref<256xi16, 2> 
    %buf6 = aie.buffer(%tile_0_4) {sym_name = "buf6"} : memref<256xi16, 2> 
    %buf5 = aie.buffer(%tile_0_3) {sym_name = "buf5"} : memref<256xi16, 2> 
    %buf4 = aie.buffer(%tile_0_3) {sym_name = "buf4"} : memref<256xi16, 2> 
    %buf3 = aie.buffer(%tile_0_3) {sym_name = "buf3"} : memref<256xi16, 2> 
    %buf2 = aie.buffer(%tile_0_2) {sym_name = "buf2"} : memref<256xi16, 2> 
    %buf1 = aie.buffer(%tile_0_2) {sym_name = "buf1"} : memref<256xi16, 2> 
    %buf0 = aie.buffer(%tile_0_2) {sym_name = "buf0"} : memref<256xi16, 2> 
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_22, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf9 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_21, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_5_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf11 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_20, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf10 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_18, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %0 = ub.poison : i16
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_21, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_20, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_18, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c32 {
        %subview = memref.subview %buf11[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %subview_23 = memref.subview %buf10[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %subview_24 = memref.subview %buf9[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %1 = vector.transfer_read %subview[%c0], %0 {in_bounds = [true]} : memref<32xi16, strided<[1], offset: ?>, 2>, vector<32xi16>
        %2 = vector.transfer_read %subview_23[%c0], %0 {in_bounds = [true]} : memref<32xi16, strided<[1], offset: ?>, 2>, vector<32xi16>
        %3 = arith.subi %1, %2 : vector<32xi16>
        vector.transfer_write %3, %subview_24[%c0] {in_bounds = [true]} : vector<32xi16>, memref<32xi16, strided<[1], offset: ?>, 2>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_5_19, Release, 1)
      aie.use_lock(%lock_0_5, Release, 1)
      aie.use_lock(%lock_0_5_22, Release, 1)
      cf.br ^bb1
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf6 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_16, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_4_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_15, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf7 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_13, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %0 = ub.poison : i16
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_16, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_15, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_13, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c32 {
        %subview = memref.subview %buf8[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %subview_23 = memref.subview %buf7[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %subview_24 = memref.subview %buf6[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %1 = vector.transfer_read %subview[%c0], %0 {in_bounds = [true]} : memref<32xi16, strided<[1], offset: ?>, 2>, vector<32xi16>
        %2 = vector.transfer_read %subview_23[%c0], %0 {in_bounds = [true]} : memref<32xi16, strided<[1], offset: ?>, 2>, vector<32xi16>
        %3 = arith.subi %1, %2 : vector<32xi16>
        vector.transfer_write %3, %subview_24[%c0] {in_bounds = [true]} : vector<32xi16>, memref<32xi16, strided<[1], offset: ?>, 2>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_4_14, Release, 1)
      aie.use_lock(%lock_0_4, Release, 1)
      aie.use_lock(%lock_0_4_17, Release, 1)
      cf.br ^bb1
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_11, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_3_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf5 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_10, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf4 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_8, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %0 = ub.poison : i16
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_11, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_10, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_8, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c32 {
        %subview = memref.subview %buf5[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %subview_23 = memref.subview %buf4[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %subview_24 = memref.subview %buf3[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %1 = vector.transfer_read %subview[%c0], %0 {in_bounds = [true]} : memref<32xi16, strided<[1], offset: ?>, 2>, vector<32xi16>
        %2 = vector.transfer_read %subview_23[%c0], %0 {in_bounds = [true]} : memref<32xi16, strided<[1], offset: ?>, 2>, vector<32xi16>
        %3 = arith.subi %1, %2 : vector<32xi16>
        vector.transfer_write %3, %subview_24[%c0] {in_bounds = [true]} : vector<32xi16>, memref<32xi16, strided<[1], offset: ?>, 2>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_3_9, Release, 1)
      aie.use_lock(%lock_0_3, Release, 1)
      aie.use_lock(%lock_0_3_12, Release, 1)
      cf.br ^bb1
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf0 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_6, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf2 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_5, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf1 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_3, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = ub.poison : i16
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c32 {
        %subview = memref.subview %buf2[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %subview_23 = memref.subview %buf1[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %subview_24 = memref.subview %buf0[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %1 = vector.transfer_read %subview[%c0], %0 {in_bounds = [true]} : memref<32xi16, strided<[1], offset: ?>, 2>, vector<32xi16>
        %2 = vector.transfer_read %subview_23[%c0], %0 {in_bounds = [true]} : memref<32xi16, strided<[1], offset: ?>, 2>, vector<32xi16>
        %3 = arith.subi %1, %2 : vector<32xi16>
        vector.transfer_write %3, %subview_24[%c0] {in_bounds = [true]} : vector<32xi16>, memref<32xi16, strided<[1], offset: ?>, 2>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2_4, Release, 1)
      aie.use_lock(%lock_0_2, Release, 1)
      aie.use_lock(%lock_0_2_7, Release, 1)
      cf.br ^bb1
    }
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_0_4, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 3, %tile_0_5, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_0_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_0_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_0_4, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 3, %tile_0_5, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %mem_tile_2_1, DMA : 1)
    aie.flow(%tile_0_4, DMA : 0, %mem_tile_2_1, DMA : 2)
    aie.flow(%tile_0_5, DMA : 0, %mem_tile_2_1, DMA : 3)
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_2, AcquireGreaterEqual, 4)
      aie.dma_bd(%buf12 : memref<1024xi16, 1>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 4)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf12 : memref<1024xi16, 1>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_2, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf12 : memref<1024xi16, 1>, 256, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_2, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 2, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf12 : memref<1024xi16, 1>, 512, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_2, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 3, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf12 : memref<1024xi16, 1>, 768, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_2, Release, 1)
      aie.next_bd ^bb10
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf14 : memref<1024xi16, 1 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf14 : memref<1024xi16, 1 : i32>, 256, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf14 : memref<1024xi16, 1 : i32>, 512, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 3, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf14 : memref<1024xi16, 1 : i32>, 768, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 0, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 4)
      aie.dma_bd(%buf14 : memref<1024xi16, 1 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_1, Release, 4)
      aie.next_bd ^bb10
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf13 : memref<1024xi16, 1 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf13 : memref<1024xi16, 1 : i32>, 256, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf13 : memref<1024xi16, 1 : i32>, 512, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 3, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf13 : memref<1024xi16, 1 : i32>, 768, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 0, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 4)
      aie.dma_bd(%buf13 : memref<1024xi16, 1 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_0, Release, 4)
      aie.next_bd ^bb10
    }
    aie.shim_dma_allocation @air_channel_5(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_1(%shim_noc_tile_1_0, MM2S, 0)
    aie.runtime_sequence @sub_kernel_0_sequence(%arg0: memref<*xi16>, %arg1: memref<*xi16>, %arg2: memref<*xi16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
      %0 = aiex.dma_configure_task_for @air_channel_0 {
        aie.dma_bd(%arg0 : memref<*xi16>, 0, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_1 {
        aie.dma_bd(%arg1 : memref<*xi16>, 0, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_5 {
        aie.dma_bd(%arg2 : memref<*xi16>, 0, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%1)
    }
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) {
    aie.runtime_sequence @sub_kernel(%arg0: memref<*xi16>, %arg1: memref<*xi16>, %arg2: memref<*xi16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
      aiex.configure @sub_kernel_0 {
        aiex.run @sub_kernel_0_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) : (memref<*xi16>, memref<*xi16>, memref<*xi16>, i32, i32, i32, i32, i32, i32)
      }
    }
  }
}
