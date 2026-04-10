#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  aie.device(npu2) @square_kernel_0 {
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
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
    %buf9 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf9"} : memref<1024xi16, 1 : i32> 
    %buf8 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf8"} : memref<1024xi16, 1> 
    %buf7 = aie.buffer(%tile_0_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf7"} : memref<256xi16, 2> 
    %buf6 = aie.buffer(%tile_0_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "buf6"} : memref<256xi16, 2> 
    %buf5 = aie.buffer(%tile_0_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf5"} : memref<256xi16, 2> 
    %buf4 = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "buf4"} : memref<256xi16, 2> 
    %buf3 = aie.buffer(%tile_0_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf3"} : memref<256xi16, 2> 
    %buf2 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "buf2"} : memref<256xi16, 2> 
    %buf1 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf1"} : memref<256xi16, 2> 
    %buf0 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "buf0"} : memref<256xi16, 2> 
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf6 : memref<256xi16, 2>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_5_12, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf7 : memref<256xi16, 2>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_5_11, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c0_i32 = arith.constant 0 : i32
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb4
      aie.use_lock(%lock_0_5_12, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_11, AcquireGreaterEqual, 1)
      cf.br ^bb2(%c0 : index)
    ^bb2(%0: index):  // 2 preds: ^bb1, ^bb3
      %1 = arith.cmpi slt, %0, %c256 : index
      cf.cond_br %1, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      %2 = vector.load %buf7[%0] : memref<256xi16, 2>, vector<32xi16>
      %3 = aievec.mul_elem %2, %2 : vector<32xi16>, vector<32xi16>, vector<32xi32>
      %4 = aievec.srs %3, %c0_i32 : vector<32xi32>, i32, vector<32xi16>
      vector.store %4, %buf6[%0] : memref<256xi16, 2>, vector<32xi16>
      %5 = arith.addi %0, %c32 : index
      cf.br ^bb2(%5 : index) {loop_annotation = #loop_annotation}
    ^bb4:  // pred: ^bb2
      aie.use_lock(%lock_0_5, Release, 1)
      aie.use_lock(%lock_0_5_13, Release, 1)
      cf.br ^bb1
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf4 : memref<256xi16, 2>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_4_9, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf5 : memref<256xi16, 2>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_4_8, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c0_i32 = arith.constant 0 : i32
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb4
      aie.use_lock(%lock_0_4_9, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_8, AcquireGreaterEqual, 1)
      cf.br ^bb2(%c0 : index)
    ^bb2(%0: index):  // 2 preds: ^bb1, ^bb3
      %1 = arith.cmpi slt, %0, %c256 : index
      cf.cond_br %1, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      %2 = vector.load %buf5[%0] : memref<256xi16, 2>, vector<32xi16>
      %3 = aievec.mul_elem %2, %2 : vector<32xi16>, vector<32xi16>, vector<32xi32>
      %4 = aievec.srs %3, %c0_i32 : vector<32xi32>, i32, vector<32xi16>
      vector.store %4, %buf4[%0] : memref<256xi16, 2>, vector<32xi16>
      %5 = arith.addi %0, %c32 : index
      cf.br ^bb2(%5 : index) {loop_annotation = #loop_annotation}
    ^bb4:  // pred: ^bb2
      aie.use_lock(%lock_0_4, Release, 1)
      aie.use_lock(%lock_0_4_10, Release, 1)
      cf.br ^bb1
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf2 : memref<256xi16, 2>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_3_6, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<256xi16, 2>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_3_5, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c0_i32 = arith.constant 0 : i32
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb4
      aie.use_lock(%lock_0_3_6, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_5, AcquireGreaterEqual, 1)
      cf.br ^bb2(%c0 : index)
    ^bb2(%0: index):  // 2 preds: ^bb1, ^bb3
      %1 = arith.cmpi slt, %0, %c256 : index
      cf.cond_br %1, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      %2 = vector.load %buf3[%0] : memref<256xi16, 2>, vector<32xi16>
      %3 = aievec.mul_elem %2, %2 : vector<32xi16>, vector<32xi16>, vector<32xi32>
      %4 = aievec.srs %3, %c0_i32 : vector<32xi32>, i32, vector<32xi16>
      vector.store %4, %buf2[%0] : memref<256xi16, 2>, vector<32xi16>
      %5 = arith.addi %0, %c32 : index
      cf.br ^bb2(%5 : index) {loop_annotation = #loop_annotation}
    ^bb4:  // pred: ^bb2
      aie.use_lock(%lock_0_3, Release, 1)
      aie.use_lock(%lock_0_3_7, Release, 1)
      cf.br ^bb1
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf0 : memref<256xi16, 2>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_2_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf1 : memref<256xi16, 2>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_2_2, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c0_i32 = arith.constant 0 : i32
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb4
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, 1)
      cf.br ^bb2(%c0 : index)
    ^bb2(%0: index):  // 2 preds: ^bb1, ^bb3
      %1 = arith.cmpi slt, %0, %c256 : index
      cf.cond_br %1, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      %2 = vector.load %buf1[%0] : memref<256xi16, 2>, vector<32xi16>
      %3 = aievec.mul_elem %2, %2 : vector<32xi16>, vector<32xi16>, vector<32xi32>
      %4 = aievec.srs %3, %c0_i32 : vector<32xi32>, i32, vector<32xi16>
      vector.store %4, %buf0[%0] : memref<256xi16, 2>, vector<32xi16>
      %5 = arith.addi %0, %c32 : index
      cf.br ^bb2(%5 : index) {loop_annotation = #loop_annotation}
    ^bb4:  // pred: ^bb2
      aie.use_lock(%lock_0_2, Release, 1)
      aie.use_lock(%lock_0_2_4, Release, 1)
      cf.br ^bb1
    }
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
      aie.dma_bd(%buf8 : memref<1024xi16, 1>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 0 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 4)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<1024xi16, 1>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_1_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<1024xi16, 1>, 256, 256) {bd_id = 24 : i32, next_bd_id = 24 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_1_1_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 2, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<1024xi16, 1>, 512, 256) {bd_id = 2 : i32, next_bd_id = 2 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_1_1_1, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 3, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<1024xi16, 1>, 768, 256) {bd_id = 25 : i32, next_bd_id = 25 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_1_1_1, Release, 1)
      aie.next_bd ^bb10
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf9 : memref<1024xi16, 1 : i32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf9 : memref<1024xi16, 1 : i32>, 256, 256) {bd_id = 24 : i32, next_bd_id = 24 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf9 : memref<1024xi16, 1 : i32>, 512, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 3, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf9 : memref<1024xi16, 1 : i32>, 768, 256) {bd_id = 25 : i32, next_bd_id = 25 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 0, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 4)
      aie.dma_bd(%buf9 : memref<1024xi16, 1 : i32>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 2 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_1_0, Release, 4)
      aie.next_bd ^bb10
    }
    aie.shim_dma_allocation @air_channel_3(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.runtime_sequence @square_kernel_0_sequence(%arg0: memref<*xi16>, %arg1: memref<*xi16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
      %0 = aiex.dma_configure_task_for @air_channel_0 {
        aie.dma_bd(%arg0 : memref<*xi16>, 0, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_3 {
        aie.dma_bd(%arg1 : memref<*xi16>, 0, 1024, [<size = 2, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      aiex.dma_free_task(%0)
      aiex.dma_await_task(%1)
    }
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) {
    aie.runtime_sequence @square_kernel(%arg0: memref<*xi16>, %arg1: memref<*xi16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
      aiex.configure @square_kernel_0 {
        aiex.run @square_kernel_0_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (memref<*xi16>, memref<*xi16>, i32, i32, i32, i32, i32, i32)
      }
    }
  }
}
