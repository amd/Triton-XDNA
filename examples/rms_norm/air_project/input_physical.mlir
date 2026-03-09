#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  aie.device(npu2) @rms_norm_kernel_0 {
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
    }
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
    }
    %lock_1_1 = aie.lock(%mem_tile_1_1, 1) {init = 1 : i32}
    %lock_1_1_0 = aie.lock(%mem_tile_1_1, 0) {init = 0 : i32}
    %buf5 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf5"} : memref<2x64xbf16, 1 : i32> 
    %buf4 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buf4"} : memref<2x64xbf16, 1> 
    %buf3 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "buf3"} : memref<1xf32, 2> 
    %buf2 = aie.buffer(%tile_0_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf2"} : memref<1x64xbf16, 2> 
    %buf1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "buf1"} : memref<1xf32, 2> 
    %buf0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "buf0"} : memref<1x64xbf16, 2> 
    memref.global "public" @__air_herd_arg_1 : memref<2x64xbf16, 1 : i32>
    %core_0_3 = aie.core(%tile_0_3) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : f32
      %cst_1 = arith.constant 6.400000e+01 : f32
      %cst_2 = arith.constant 9.99999974E-6 : f32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb7
      %0 = memref.get_global @__air_herd_arg_1 : memref<2x64xbf16, 1 : i32>
      %subview = memref.subview %0[1, 0] [1, 64] [1, 1] : memref<2x64xbf16, 1 : i32> to memref<1x64xbf16, strided<[64, 1], offset: 64>, 1 : i32>
      memref.store %cst, %buf3[%c0] : memref<1xf32, 2>
      cf.br ^bb2(%c0 : index)
    ^bb2(%1: index):  // 2 preds: ^bb1, ^bb3
      %2 = arith.cmpi slt, %1, %c64 : index
      cf.cond_br %2, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      %3 = memref.load %subview[%c0, %1] : memref<1x64xbf16, strided<[64, 1], offset: 64>, 1 : i32>
      %4 = memref.load %buf3[%c0] : memref<1xf32, 2>
      %5 = arith.extf %3 : bf16 to f32
      %6 = arith.mulf %5, %5 : f32
      %7 = arith.addf %6, %4 : f32
      memref.store %7, %buf3[%c0] : memref<1xf32, 2>
      %8 = arith.addi %1, %c1 : index
      cf.br ^bb2(%8 : index) {loop_annotation = #loop_annotation}
    ^bb4:  // pred: ^bb2
      cf.br ^bb5(%c0 : index)
    ^bb5(%9: index):  // 2 preds: ^bb4, ^bb6
      %10 = arith.cmpi slt, %9, %c64 : index
      cf.cond_br %10, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %11 = memref.load %subview[%c0, %9] : memref<1x64xbf16, strided<[64, 1], offset: 64>, 1 : i32>
      %12 = memref.load %buf3[%c0] : memref<1xf32, 2>
      %13 = arith.divf %12, %cst_1 : f32
      %14 = arith.addf %13, %cst_2 : f32
      %15 = math.rsqrt %14 : f32
      %16 = arith.extf %11 : bf16 to f32
      %17 = arith.mulf %16, %15 : f32
      %18 = arith.truncf %17 : f32 to bf16
      memref.store %18, %buf2[%c0, %9] : memref<1x64xbf16, 2>
      %19 = arith.addi %9, %c1 : index
      cf.br ^bb5(%19 : index) {loop_annotation = #loop_annotation}
    ^bb7:  // pred: ^bb5
      cf.br ^bb1
    }
    memref.global "public" @__air_herd_arg : memref<2x64xbf16, 1 : i32>
    %core_0_2 = aie.core(%tile_0_2) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : f32
      %cst_1 = arith.constant 6.400000e+01 : f32
      %cst_2 = arith.constant 9.99999974E-6 : f32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb7
      %0 = memref.get_global @__air_herd_arg : memref<2x64xbf16, 1 : i32>
      %subview = memref.subview %0[0, 0] [1, 64] [1, 1] : memref<2x64xbf16, 1 : i32> to memref<1x64xbf16, strided<[64, 1]>, 1 : i32>
      memref.store %cst, %buf1[%c0] : memref<1xf32, 2>
      cf.br ^bb2(%c0 : index)
    ^bb2(%1: index):  // 2 preds: ^bb1, ^bb3
      %2 = arith.cmpi slt, %1, %c64 : index
      cf.cond_br %2, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      %3 = memref.load %subview[%c0, %1] : memref<1x64xbf16, strided<[64, 1]>, 1 : i32>
      %4 = memref.load %buf1[%c0] : memref<1xf32, 2>
      %5 = arith.extf %3 : bf16 to f32
      %6 = arith.mulf %5, %5 : f32
      %7 = arith.addf %6, %4 : f32
      memref.store %7, %buf1[%c0] : memref<1xf32, 2>
      %8 = arith.addi %1, %c1 : index
      cf.br ^bb2(%8 : index) {loop_annotation = #loop_annotation}
    ^bb4:  // pred: ^bb2
      cf.br ^bb5(%c0 : index)
    ^bb5(%9: index):  // 2 preds: ^bb4, ^bb6
      %10 = arith.cmpi slt, %9, %c64 : index
      cf.cond_br %10, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %11 = memref.load %subview[%c0, %9] : memref<1x64xbf16, strided<[64, 1]>, 1 : i32>
      %12 = memref.load %buf1[%c0] : memref<1xf32, 2>
      %13 = arith.divf %12, %cst_1 : f32
      %14 = arith.addf %13, %cst_2 : f32
      %15 = math.rsqrt %14 : f32
      %16 = arith.extf %11 : bf16 to f32
      %17 = arith.mulf %16, %15 : f32
      %18 = arith.truncf %17 : f32 to bf16
      memref.store %18, %buf0[%c0, %9] : memref<1x64xbf16, 2>
      %19 = arith.addi %9, %c1 : index
      cf.br ^bb5(%19 : index) {loop_annotation = #loop_annotation}
    ^bb7:  // pred: ^bb5
      cf.br ^bb1
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf4 : memref<2x64xbf16, 1>, 0, 128) {bd_id = 0 : i32, next_bd_id = 0 : i32, task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    aie.shim_dma_allocation @air_channel_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.runtime_sequence @rms_norm_kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
      %0 = aiex.dma_configure_task_for @air_channel_0 {
        aie.dma_bd(%arg0 : memref<*xbf16>, 0, 128, [<size = 128, stride = 1>])
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_1 {
        aie.dma_bd(%arg1 : memref<*xbf16>, 0, 128, [<size = 128, stride = 1>])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
      %2 = aiex.dma_configure_task_for @air_channel_0 {
        aie.dma_bd(%arg0 : memref<*xbf16>, 0, 128, [<size = 128, stride = 1>])
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @air_channel_1 {
        aie.dma_bd(%arg1 : memref<*xbf16>, 0, 128, [<size = 128, stride = 1>])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%3)
      aiex.dma_await_task(%3)
      aiex.dma_free_task(%2)
      %4 = aiex.dma_configure_task_for @air_channel_0 {
        aie.dma_bd(%arg0 : memref<*xbf16>, 0, 128, [<size = 128, stride = 1>])
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @air_channel_1 {
        aie.dma_bd(%arg1 : memref<*xbf16>, 0, 128, [<size = 128, stride = 1>])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%5)
      aiex.dma_await_task(%5)
      aiex.dma_free_task(%4)
      %6 = aiex.dma_configure_task_for @air_channel_0 {
        aie.dma_bd(%arg0 : memref<*xbf16>, 0, 128, [<size = 128, stride = 1>])
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @air_channel_1 {
        aie.dma_bd(%arg1 : memref<*xbf16>, 0, 128, [<size = 128, stride = 1>])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%7)
      aiex.dma_await_task(%7)
      aiex.dma_free_task(%6)
    }
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
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
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<DMA : 0, South : 2>
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
    aie.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
    aie.wire(%shim_mux_1_0 : North, %switchbox_1_0 : South)
    aie.wire(%shim_noc_tile_1_0 : DMA, %shim_mux_1_0 : DMA)
    aie.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
    aie.wire(%mem_tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%mem_tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
}
