#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  aie.device(npu2) @square_kernel_0 {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %lock_0_1 = aie.lock(%mem_tile_0_1, 1) {init = 4 : i32}
    %lock_0_1_0 = aie.lock(%mem_tile_0_1, 0) {init = 0 : i32}
    %lock_1_1 = aie.lock(%mem_tile_1_1, 1) {init = 4 : i32}
    %lock_1_1_1 = aie.lock(%mem_tile_1_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 3) {init = 1 : i32}
    %lock_0_3_5 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_6 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_7 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 3) {init = 1 : i32}
    %lock_0_4_8 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_9 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_10 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %lock_0_5 = aie.lock(%tile_0_5, 3) {init = 1 : i32}
    %lock_0_5_11 = aie.lock(%tile_0_5, 2) {init = 0 : i32}
    %lock_0_5_12 = aie.lock(%tile_0_5, 1) {init = 1 : i32}
    %lock_0_5_13 = aie.lock(%tile_0_5, 0) {init = 0 : i32}
    %buf9 = aie.buffer(%mem_tile_0_1) {sym_name = "buf9"} : memref<1024xi16, 1 : i32> 
    %buf8 = aie.buffer(%mem_tile_1_1) {sym_name = "buf8"} : memref<1024xi16, 1> 
    %buf7 = aie.buffer(%tile_0_5) {sym_name = "buf7"} : memref<256xi16, 2> 
    %buf6 = aie.buffer(%tile_0_5) {sym_name = "buf6"} : memref<256xi16, 2> 
    %buf5 = aie.buffer(%tile_0_4) {sym_name = "buf5"} : memref<256xi16, 2> 
    %buf4 = aie.buffer(%tile_0_4) {sym_name = "buf4"} : memref<256xi16, 2> 
    %buf3 = aie.buffer(%tile_0_3) {sym_name = "buf3"} : memref<256xi16, 2> 
    %buf2 = aie.buffer(%tile_0_3) {sym_name = "buf2"} : memref<256xi16, 2> 
    %buf1 = aie.buffer(%tile_0_2) {sym_name = "buf1"} : memref<256xi16, 2> 
    %buf0 = aie.buffer(%tile_0_2) {sym_name = "buf0"} : memref<256xi16, 2> 
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf6 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_12, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf7 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_11, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %0 = ub.poison : i16
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_5_12, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_5_11, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c32 {
        %subview = memref.subview %buf7[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %subview_14 = memref.subview %buf6[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %1 = vector.transfer_read %subview[%c0], %0 {in_bounds = [true]} : memref<32xi16, strided<[1], offset: ?>, 2>, vector<32xi16>
        %2 = arith.muli %1, %1 : vector<32xi16>
        vector.transfer_write %2, %subview_14[%c0] {in_bounds = [true]} : vector<32xi16>, memref<32xi16, strided<[1], offset: ?>, 2>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_5, Release, 1)
      aie.use_lock(%lock_0_5_13, Release, 1)
      cf.br ^bb1
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf4 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_9, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf5 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_8, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %0 = ub.poison : i16
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_4_9, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_4_8, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c32 {
        %subview = memref.subview %buf5[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %subview_14 = memref.subview %buf4[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %1 = vector.transfer_read %subview[%c0], %0 {in_bounds = [true]} : memref<32xi16, strided<[1], offset: ?>, 2>, vector<32xi16>
        %2 = arith.muli %1, %1 : vector<32xi16>
        vector.transfer_write %2, %subview_14[%c0] {in_bounds = [true]} : vector<32xi16>, memref<32xi16, strided<[1], offset: ?>, 2>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_4, Release, 1)
      aie.use_lock(%lock_0_4_10, Release, 1)
      cf.br ^bb1
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf2 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_6, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_5, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %0 = ub.poison : i16
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_3_6, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_3_5, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c32 {
        %subview = memref.subview %buf3[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %subview_14 = memref.subview %buf2[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %1 = vector.transfer_read %subview[%c0], %0 {in_bounds = [true]} : memref<32xi16, strided<[1], offset: ?>, 2>, vector<32xi16>
        %2 = arith.muli %1, %1 : vector<32xi16>
        vector.transfer_write %2, %subview_14[%c0] {in_bounds = [true]} : vector<32xi16>, memref<32xi16, strided<[1], offset: ?>, 2>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_3, Release, 1)
      aie.use_lock(%lock_0_3_7, Release, 1)
      cf.br ^bb1
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf0 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf1 : memref<256xi16, 2>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_2, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = ub.poison : i16
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c256 step %c32 {
        %subview = memref.subview %buf1[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %subview_14 = memref.subview %buf0[%arg0] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
        %1 = vector.transfer_read %subview[%c0], %0 {in_bounds = [true]} : memref<32xi16, strided<[1], offset: ?>, 2>, vector<32xi16>
        %2 = arith.muli %1, %1 : vector<32xi16>
        vector.transfer_write %2, %subview_14[%c0] {in_bounds = [true]} : vector<32xi16>, memref<32xi16, strided<[1], offset: ?>, 2>
      } {loop_annotation = #loop_annotation}
      aie.use_lock(%lock_0_2, Release, 1)
      aie.use_lock(%lock_0_2_4, Release, 1)
      cf.br ^bb1
    }
    air.channel @channel_0 []
    air.channel @channel_2 [1, 1]
    air.channel @channel_8 [1, 1]
    air.channel @channel_9 [1, 1]
    air.channel @channel_10 [1, 1]
    air.channel @channel_4 [1, 1]
    air.channel @channel_5 [1, 1]
    air.channel @channel_6 [1, 1]
    air.channel @channel_7 [1, 1]
    air.channel @channel_3 []
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_0_4, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 3, %tile_0_5, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %mem_tile_1_1, DMA : 1)
    aie.flow(%tile_0_4, DMA : 0, %mem_tile_1_1, DMA : 2)
    aie.flow(%tile_0_5, DMA : 0, %mem_tile_1_1, DMA : 3)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_1, AcquireGreaterEqual, 4)
      aie.dma_bd(%buf8 : memref<1024xi16, 1>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 4)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<1024xi16, 1>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<1024xi16, 1>, 256, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 2, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<1024xi16, 1>, 512, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_1, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 3, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<1024xi16, 1>, 768, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_1, Release, 1)
      aie.next_bd ^bb10
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf9 : memref<1024xi16, 1 : i32>, 0, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf9 : memref<1024xi16, 1 : i32>, 256, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf9 : memref<1024xi16, 1 : i32>, 512, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 3, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf9 : memref<1024xi16, 1 : i32>, 768, 256) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 0, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 4)
      aie.dma_bd(%buf9 : memref<1024xi16, 1 : i32>, 0, 1024) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_0, Release, 4)
      aie.next_bd ^bb10
    }
    aie.shim_dma_allocation @air_channel_3(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0(%shim_noc_tile_0_0, MM2S, 0)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 3 : i64, location = 0 : i64, row = -1 : i64}], sym_name = "square_kernel_0"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 4 : i64, sym_name = "herd_0"}
    }
  }
  air.channel @channel_0 []
  air.channel @channel_1 [4, 1]
  air.channel @channel_2 [4, 1]
  air.channel @channel_3 []
  func.func @square_kernel(%arg0: memref<*xi16> {tt.divisibility = 16 : i32}, %arg1: memref<*xi16> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg8, %arg9, %arg10) in (%arg11=%c1, %arg12=%c1, %arg13=%c1) args(%arg14=%arg0, %arg15=%arg1) : memref<*xi16>, memref<*xi16> attributes {id = 1 : i32} {
      %c1024 = arith.constant 1024 : index
      %c1_0 = arith.constant 1 : index
      %1 = arith.muli %arg8, %c1024 : index
      %2 = air.channel.put async  @channel_0[] (%arg14[%1] [%c1024] [%c1_0]) {id = 1 : i32, metadataArray = [{base = "air_channel_0", index = 0 : i32}]} : (memref<*xi16>)
      %3 = air.channel.get async  @channel_3[] (%arg15[%1] [%c1024] [%c1_0]) {id = 2 : i32, metadataArray = [{base = "air_channel_3", index = 0 : i32}]} : (memref<*xi16>)
      %4 = air.segment @square_kernel_0 async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c4 = arith.constant 4 : index
        %c768 = arith.constant 768 : index
        %c3 = arith.constant 3 : index
        %c512 = arith.constant 512 : index
        %c2 = arith.constant 2 : index
        %c256 = arith.constant 256 : index
        %c0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<1024xi16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1024xi16, 1 : i32>
          air.execute_terminator %alloc : memref<1024xi16, 1 : i32>
        }
        %5 = air.channel.get async [%async_token]  @channel_0[] (%results[] [] []) {id = 3 : i32} : (memref<1024xi16, 1 : i32>)
        %async_token_2, %results_3 = air.execute -> (memref<1024xi16, 1>) {
          %alloc = memref.alloc() : memref<1024xi16, 1>
          air.execute_terminator %alloc : memref<1024xi16, 1>
        }
        %6 = air.channel.put async [%5]  @channel_1[%c0, %c0] (%results[%c0] [%c256] [%c1_1]) {id = 4 : i32} : (memref<1024xi16, 1 : i32>)
        %7 = air.channel.put async [%5]  @channel_1[%c1_1, %c0] (%results[%c256] [%c256] [%c1_1]) {id = 5 : i32} : (memref<1024xi16, 1 : i32>)
        %8 = air.channel.put async [%5]  @channel_1[%c2, %c0] (%results[%c512] [%c256] [%c1_1]) {id = 6 : i32} : (memref<1024xi16, 1 : i32>)
        %9 = air.channel.put async [%5]  @channel_1[%c3, %c0] (%results[%c768] [%c256] [%c1_1]) {id = 7 : i32} : (memref<1024xi16, 1 : i32>)
        %10 = air.channel.get async [%async_token_2]  @channel_2[%c0, %c0] (%results_3[%c0] [%c256] [%c1_1]) {id = 8 : i32} : (memref<1024xi16, 1>)
        %11 = air.channel.get async [%async_token_2]  @channel_2[%c1_1, %c0] (%results_3[%c256] [%c256] [%c1_1]) {id = 9 : i32} : (memref<1024xi16, 1>)
        %12 = air.channel.get async [%async_token_2]  @channel_2[%c2, %c0] (%results_3[%c512] [%c256] [%c1_1]) {id = 10 : i32} : (memref<1024xi16, 1>)
        %13 = air.channel.get async [%async_token_2]  @channel_2[%c3, %c0] (%results_3[%c768] [%c256] [%c1_1]) {id = 11 : i32} : (memref<1024xi16, 1>)
        %14 = air.herd @herd_0 async [%5, %async_token_2]  tile (%arg16, %arg17) in (%arg18=%c1_1, %arg19=%c4) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %c32 = arith.constant 32 : index
          %c256_5 = arith.constant 256 : index
          %c0_6 = arith.constant 0 : index
          %16 = ub.poison : i16
          %async_token_7, %results_8 = air.execute -> (memref<256xi16, 2>) {
            %alloc = memref.alloc() : memref<256xi16, 2>
            air.execute_terminator %alloc : memref<256xi16, 2>
          }
          %17 = air.channel.get async [%async_token_7]  @channel_1[%arg17, %c0_6] (%results_8[] [] []) {id = 12 : i32} : (memref<256xi16, 2>)
          %async_token_9, %results_10 = air.execute -> (memref<256xi16, 2>) {
            %alloc = memref.alloc() : memref<256xi16, 2>
            air.execute_terminator %alloc : memref<256xi16, 2>
          }
          %18 = air.wait_all async [%17, %async_token_9] 
          %19 = scf.for %arg20 = %c0_6 to %c256_5 step %c32 iter_args(%arg21 = %18) -> (!air.async.token) {
            %subview = memref.subview %results_8[%arg20] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
            %subview_13 = memref.subview %results_10[%arg20] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
            %async_token_14, %results_15 = air.execute [%arg21] -> (vector<32xi16>) {
              %23 = vector.transfer_read %subview[%c0_6], %16 {in_bounds = [true]} : memref<32xi16, strided<[1], offset: ?>, 2>, vector<32xi16>
              air.execute_terminator %23 : vector<32xi16>
            }
            %21 = arith.muli %results_15, %results_15 : vector<32xi16>
            %async_token_16 = air.execute [%arg21] {
              vector.transfer_write %21, %subview_13[%c0_6] {in_bounds = [true]} : vector<32xi16>, memref<32xi16, strided<[1], offset: ?>, 2>
            }
            %22 = air.wait_all async [%async_token_14, %async_token_16] 
            scf.yield %22 : !air.async.token
          }
          %20 = air.channel.put async [%async_token_9]  @channel_2[%arg17, %c0_6] (%results_10[] [] []) {id = 13 : i32} : (memref<256xi16, 2>)
          %async_token_11 = air.execute [%17] {
            memref.dealloc %results_8 : memref<256xi16, 2>
          }
          %async_token_12 = air.execute [%20] {
            memref.dealloc %results_10 : memref<256xi16, 2>
          }
        }
        %15 = air.channel.put async [%14]  @channel_3[] (%results_3[] [] []) {id = 14 : i32} : (memref<1024xi16, 1>)
        %async_token_4 = air.execute [%15] {
          memref.dealloc %results_3 : memref<1024xi16, 1>
        }
        air.wait_all [%6, %7, %8, %9, %10, %11, %12, %13, %async_token_4]  {air.segment_end}
      }
    }
    return
  }
}
