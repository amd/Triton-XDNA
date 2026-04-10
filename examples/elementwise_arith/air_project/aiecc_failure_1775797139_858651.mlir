#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  aie.device(npu2) @mul_kernel_0 {
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_2_1 = aie.tile(2, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
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
    %buf14 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf14"} : memref<1024xi16, 1 : i32> 
    %buf13 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf13"} : memref<1024xi16, 1 : i32> 
    %buf12 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf12"} : memref<1024xi16, 1> 
    %buf11 = aie.buffer(%tile_0_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf11"} : memref<256xi16, 2> 
    %buf10 = aie.buffer(%tile_0_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "buf10"} : memref<256xi16, 2> 
    %buf9 = aie.buffer(%tile_0_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "buf9"} : memref<256xi16, 2> 
    %buf8 = aie.buffer(%tile_0_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf8"} : memref<256xi16, 2> 
    %buf7 = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "buf7"} : memref<256xi16, 2> 
    %buf6 = aie.buffer(%tile_0_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "buf6"} : memref<256xi16, 2> 
    %buf5 = aie.buffer(%tile_0_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf5"} : memref<256xi16, 2> 
    %buf4 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "buf4"} : memref<256xi16, 2> 
    %buf3 = aie.buffer(%tile_0_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "buf3"} : memref<256xi16, 2> 
    %buf2 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf2"} : memref<256xi16, 2> 
    %buf1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "buf1"} : memref<256xi16, 2> 
    %buf0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "buf0"} : memref<256xi16, 2> 
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_22, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf9 : memref<256xi16, 2>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_5_21, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_5_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf11 : memref<256xi16, 2>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_5_20, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf10 : memref<256xi16, 2>, 0, 256) {bd_id = 2 : i32, next_bd_id = 2 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_5_18, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c0_i32 = arith.constant 0 : i32
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb4
      aie.use_lock(%lock_0_5_21, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_20, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_18, AcquireGreaterEqual, 1)
      cf.br ^bb2(%c0 : index)
    ^bb2(%0: index):  // 2 preds: ^bb1, ^bb3
      %1 = arith.cmpi slt, %0, %c256 : index
      cf.cond_br %1, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      %2 = vector.load %buf11[%0] : memref<256xi16, 2>, vector<32xi16>
      %3 = vector.load %buf10[%0] : memref<256xi16, 2>, vector<32xi16>
      %4 = aievec.mul_elem %2, %3 : vector<32xi16>, vector<32xi16>, vector<32xi32>
      %5 = aievec.srs %4, %c0_i32 : vector<32xi32>, i32, vector<32xi16>
      vector.store %5, %buf9[%0] : memref<256xi16, 2>, vector<32xi16>
      %6 = arith.addi %0, %c32 : index
      cf.br ^bb2(%6 : index) {loop_annotation = #loop_annotation}
    ^bb4:  // pred: ^bb2
      aie.use_lock(%lock_0_5_19, Release, 1)
      aie.use_lock(%lock_0_5, Release, 1)
      aie.use_lock(%lock_0_5_22, Release, 1)
      cf.br ^bb1
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf6 : memref<256xi16, 2>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_4_16, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_4_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<256xi16, 2>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_4_15, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf7 : memref<256xi16, 2>, 0, 256) {bd_id = 2 : i32, next_bd_id = 2 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_4_13, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c0_i32 = arith.constant 0 : i32
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb4
      aie.use_lock(%lock_0_4_16, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_15, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_13, AcquireGreaterEqual, 1)
      cf.br ^bb2(%c0 : index)
    ^bb2(%0: index):  // 2 preds: ^bb1, ^bb3
      %1 = arith.cmpi slt, %0, %c256 : index
      cf.cond_br %1, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      %2 = vector.load %buf8[%0] : memref<256xi16, 2>, vector<32xi16>
      %3 = vector.load %buf7[%0] : memref<256xi16, 2>, vector<32xi16>
      %4 = aievec.mul_elem %2, %3 : vector<32xi16>, vector<32xi16>, vector<32xi32>
      %5 = aievec.srs %4, %c0_i32 : vector<32xi32>, i32, vector<32xi16>
      vector.store %5, %buf6[%0] : memref<256xi16, 2>, vector<32xi16>
      %6 = arith.addi %0, %c32 : index
      cf.br ^bb2(%6 : index) {loop_annotation = #loop_annotation}
    ^bb4:  // pred: ^bb2
      aie.use_lock(%lock_0_4_14, Release, 1)
      aie.use_lock(%lock_0_4, Release, 1)
      aie.use_lock(%lock_0_4_17, Release, 1)
      cf.br ^bb1
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<256xi16, 2>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_3_11, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_3_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf5 : memref<256xi16, 2>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_3_10, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf4 : memref<256xi16, 2>, 0, 256) {bd_id = 2 : i32, next_bd_id = 2 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_3_8, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c0_i32 = arith.constant 0 : i32
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb4
      aie.use_lock(%lock_0_3_11, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_10, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_8, AcquireGreaterEqual, 1)
      cf.br ^bb2(%c0 : index)
    ^bb2(%0: index):  // 2 preds: ^bb1, ^bb3
      %1 = arith.cmpi slt, %0, %c256 : index
      cf.cond_br %1, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      %2 = vector.load %buf5[%0] : memref<256xi16, 2>, vector<32xi16>
      %3 = vector.load %buf4[%0] : memref<256xi16, 2>, vector<32xi16>
      %4 = aievec.mul_elem %2, %3 : vector<32xi16>, vector<32xi16>, vector<32xi32>
      %5 = aievec.srs %4, %c0_i32 : vector<32xi32>, i32, vector<32xi16>
      vector.store %5, %buf3[%0] : memref<256xi16, 2>, vector<32xi16>
      %6 = arith.addi %0, %c32 : index
      cf.br ^bb2(%6 : index) {loop_annotation = #loop_annotation}
    ^bb4:  // pred: ^bb2
      aie.use_lock(%lock_0_3_9, Release, 1)
      aie.use_lock(%lock_0_3, Release, 1)
      aie.use_lock(%lock_0_3_12, Release, 1)
      cf.br ^bb1
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf0 : memref<256xi16, 2>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_2_6, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf2 : memref<256xi16, 2>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_2_5, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf1 : memref<256xi16, 2>, 0, 256) {bd_id = 2 : i32, next_bd_id = 2 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_2_3, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c0_i32 = arith.constant 0 : i32
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb4
      aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
      cf.br ^bb2(%c0 : index)
    ^bb2(%0: index):  // 2 preds: ^bb1, ^bb3
      %1 = arith.cmpi slt, %0, %c256 : index
      cf.cond_br %1, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      %2 = vector.load %buf2[%0] : memref<256xi16, 2>, vector<32xi16>
      %3 = vector.load %buf1[%0] : memref<256xi16, 2>, vector<32xi16>
      %4 = aievec.mul_elem %2, %3 : vector<32xi16>, vector<32xi16>, vector<32xi32>
      %5 = aievec.srs %4, %c0_i32 : vector<32xi32>, i32, vector<32xi16>
      vector.store %5, %buf0[%0] : memref<256xi16, 2>, vector<32xi16>
      %6 = arith.addi %0, %c32 : index
      cf.br ^bb2(%6 : index) {loop_annotation = #loop_annotation}
    ^bb4:  // pred: ^bb2
      aie.use_lock(%lock_0_2_4, Release, 1)
      aie.use_lock(%lock_0_2, Release, 1)
      aie.use_lock(%lock_0_2_7, Release, 1)
      cf.br ^bb1
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_2, AcquireGreaterEqual, 4)
      aie.dma_bd(%buf12 : memref<1024xi16, 1>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 0 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 4)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf12 : memref<1024xi16, 1>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_2_1_2, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf12 : memref<1024xi16, 1>, 256, 256) {bd_id = 24 : i32, next_bd_id = 24 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_2_1_2, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(S2MM, 2, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf12 : memref<1024xi16, 1>, 512, 256) {bd_id = 2 : i32, next_bd_id = 2 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_2_1_2, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 3, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf12 : memref<1024xi16, 1>, 768, 256) {bd_id = 25 : i32, next_bd_id = 25 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_2_1_2, Release, 1)
      aie.next_bd ^bb10
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf14 : memref<1024xi16, 1 : i32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf14 : memref<1024xi16, 1 : i32>, 256, 256) {bd_id = 24 : i32, next_bd_id = 24 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf14 : memref<1024xi16, 1 : i32>, 512, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 3, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf14 : memref<1024xi16, 1 : i32>, 768, 256) {bd_id = 25 : i32, next_bd_id = 25 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 0, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 4)
      aie.dma_bd(%buf14 : memref<1024xi16, 1 : i32>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 2 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_0_1_1, Release, 4)
      aie.next_bd ^bb10
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf13 : memref<1024xi16, 1 : i32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf13 : memref<1024xi16, 1 : i32>, 256, 256) {bd_id = 24 : i32, next_bd_id = 24 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf13 : memref<1024xi16, 1 : i32>, 512, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 3, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf13 : memref<1024xi16, 1 : i32>, 768, 256) {bd_id = 25 : i32, next_bd_id = 25 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %4 = aie.dma_start(S2MM, 0, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 4)
      aie.dma_bd(%buf13 : memref<1024xi16, 1 : i32>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 2 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_1_1_0, Release, 4)
      aie.next_bd ^bb10
    }
    aie.shim_dma_allocation @air_channel_5(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_1(%shim_noc_tile_1_0, MM2S, 0)
    aie.runtime_sequence @mul_kernel_0_sequence(%arg0: memref<*xi16>, %arg1: memref<*xi16>, %arg2: memref<*xi16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
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
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<DMA : 2, North : 0>
      aie.connect<DMA : 3, North : 3>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<South : 3, North : 1>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<DMA : 2, North : 0>
      aie.connect<DMA : 3, North : 3>
    }
    %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0) {
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_2_0 = aie.shim_mux(%shim_noc_tile_2_0) {
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_2_1 = aie.switchbox(%mem_tile_2_1) {
      aie.connect<DMA : 0, South : 2>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 1, DMA : 1>
      aie.connect<North : 0, DMA : 2>
      aie.connect<North : 3, DMA : 3>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 5, North : 3>
      aie.connect<South : 0, North : 5>
      aie.connect<South : 3, North : 4>
      aie.connect<East : 3, DMA : 1>
      aie.connect<DMA : 0, East : 0>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<South : 5, North : 0>
      aie.connect<South : 4, North : 2>
      aie.connect<East : 2, DMA : 1>
      aie.connect<East : 0, North : 5>
      aie.connect<DMA : 0, East : 0>
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 2, North : 0>
      aie.connect<South : 5, DMA : 1>
      aie.connect<East : 2, North : 5>
      aie.connect<DMA : 0, East : 0>
      aie.connect<North : 0, East : 3>
    }
    %switchbox_0_5 = aie.switchbox(%tile_0_5) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 5, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %tile_1_2 = aie.tile(1, 2)
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<South : 1, West : 3>
      aie.connect<South : 5, North : 1>
      aie.connect<South : 0, North : 2>
      aie.connect<South : 3, North : 0>
      aie.connect<West : 0, East : 1>
      aie.connect<North : 3, East : 3>
    }
    %tile_1_3 = aie.tile(1, 3)
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<South : 1, West : 2>
      aie.connect<South : 2, West : 0>
      aie.connect<South : 0, North : 0>
      aie.connect<West : 0, East : 1>
      aie.connect<North : 1, South : 3>
    }
    %tile_1_4 = aie.tile(1, 4)
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<South : 0, West : 2>
      aie.connect<West : 0, South : 1>
      aie.connect<West : 3, East : 3>
    }
    %tile_2_2 = aie.tile(2, 2)
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<West : 1, South : 2>
      aie.connect<North : 3, South : 1>
      aie.connect<West : 3, South : 0>
      aie.connect<North : 0, South : 3>
    }
    %tile_2_3 = aie.tile(2, 3)
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
      aie.connect<West : 1, South : 3>
      aie.connect<North : 1, South : 0>
    }
    %tile_2_4 = aie.tile(2, 4)
    %switchbox_2_4 = aie.switchbox(%tile_2_4) {
      aie.connect<West : 3, South : 1>
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%tile_0_4 : Core, %switchbox_0_4 : Core)
    aie.wire(%tile_0_4 : DMA, %switchbox_0_4 : DMA)
    aie.wire(%switchbox_0_3 : North, %switchbox_0_4 : South)
    aie.wire(%tile_0_5 : Core, %switchbox_0_5 : Core)
    aie.wire(%tile_0_5 : DMA, %switchbox_0_5 : DMA)
    aie.wire(%switchbox_0_4 : North, %switchbox_0_5 : South)
    aie.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
    aie.wire(%shim_mux_1_0 : North, %switchbox_1_0 : South)
    aie.wire(%shim_noc_tile_1_0 : DMA, %shim_mux_1_0 : DMA)
    aie.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
    aie.wire(%mem_tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%mem_tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
    aie.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
    aie.wire(%switchbox_0_3 : East, %switchbox_1_3 : West)
    aie.wire(%tile_1_3 : Core, %switchbox_1_3 : Core)
    aie.wire(%tile_1_3 : DMA, %switchbox_1_3 : DMA)
    aie.wire(%switchbox_1_2 : North, %switchbox_1_3 : South)
    aie.wire(%switchbox_0_4 : East, %switchbox_1_4 : West)
    aie.wire(%tile_1_4 : Core, %switchbox_1_4 : Core)
    aie.wire(%tile_1_4 : DMA, %switchbox_1_4 : DMA)
    aie.wire(%switchbox_1_3 : North, %switchbox_1_4 : South)
    aie.wire(%switchbox_1_0 : East, %switchbox_2_0 : West)
    aie.wire(%shim_mux_2_0 : North, %switchbox_2_0 : South)
    aie.wire(%shim_noc_tile_2_0 : DMA, %shim_mux_2_0 : DMA)
    aie.wire(%switchbox_1_1 : East, %switchbox_2_1 : West)
    aie.wire(%mem_tile_2_1 : Core, %switchbox_2_1 : Core)
    aie.wire(%mem_tile_2_1 : DMA, %switchbox_2_1 : DMA)
    aie.wire(%switchbox_2_0 : North, %switchbox_2_1 : South)
    aie.wire(%switchbox_1_2 : East, %switchbox_2_2 : West)
    aie.wire(%tile_2_2 : Core, %switchbox_2_2 : Core)
    aie.wire(%tile_2_2 : DMA, %switchbox_2_2 : DMA)
    aie.wire(%switchbox_2_1 : North, %switchbox_2_2 : South)
    aie.wire(%switchbox_1_3 : East, %switchbox_2_3 : West)
    aie.wire(%tile_2_3 : Core, %switchbox_2_3 : Core)
    aie.wire(%tile_2_3 : DMA, %switchbox_2_3 : DMA)
    aie.wire(%switchbox_2_2 : North, %switchbox_2_3 : South)
    aie.wire(%switchbox_1_4 : East, %switchbox_2_4 : West)
    aie.wire(%tile_2_4 : Core, %switchbox_2_4 : Core)
    aie.wire(%tile_2_4 : DMA, %switchbox_2_4 : DMA)
    aie.wire(%switchbox_2_3 : North, %switchbox_2_4 : South)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) {
    aie.runtime_sequence @mul_kernel(%arg0: memref<*xi16>, %arg1: memref<*xi16>, %arg2: memref<*xi16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
      aiex.configure @mul_kernel_0 {
        aiex.run @mul_kernel_0_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) : (memref<*xi16>, memref<*xi16>, memref<*xi16>, i32, i32, i32, i32, i32, i32)
      }
    }
  }
}
