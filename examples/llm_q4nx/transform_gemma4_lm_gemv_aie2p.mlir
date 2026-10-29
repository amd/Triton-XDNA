// Schedule of gemma4_lm_gemv.gemv_q4 on AIE2P: one herd of NC x NR cores,
// each owning its rows of the LM head. Per row tile a core takes the row
// tile's packets into L1 one at a time, dequantizes a scale group (4 k tiles)
// just before it is used, and accumulates 8 k by 8 n products in f32; x is
// copied into every core's L1 once per dispatch and expanded there.
module attributes {transform.with_named_sequence} {
  // Slices moved up towards their producers, through index arithmetic and
  // collapse_shapes, so that the next fusion into a loop finds them.
  transform.named_sequence @bubble_up(%root: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.tensor.bubble_up_extract_slice
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
  transform.named_sequence @bubble_slices(%root: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %f1 = transform.apply_registered_pass "affine-expand-index-ops-as-affine" to %f : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f1 {
      transform.apply_patterns.tensor.bubble_up_extract_slice
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %f2 = transform.air.bubble_slice_through_collapse %f1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f2 {
      transform.apply_patterns.tensor.bubble_up_extract_slice
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // a packet's q, scale and min are one stream: one copy of the packet
    %mg = transform.air.merge_sibling_copies %f : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f { transform.apply_patterns.canonicalization } : !transform.any_op
    // x's expansion, once per dispatch: each core expands x into
    // its own L1, in a core loop of its own (which also keeps it out of the
    // elementwise fusion)
    %xb0 = transform.structured.match ops{["linalg.generic"]} filter_result_type = tensor<8x4x64x8x8xbf16> in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.annotate %xb0 "x_exp" : !transform.any_op
    transform.foreach %xb0 : !transform.any_op {
    ^bb0(%xb1: !transform.any_op):
      %xbt, %xbf = transform.structured.tile_using_forall %xb1 tile_sizes [1, 1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.annotate %xbf "pro_forall" : !transform.any_op
    }
    %rd = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %rdg = transform.structured.generalize %rd : (!transform.any_op) -> !transform.any_op
    %tr = transform.structured.match ops{["linalg.transpose"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %trg = transform.structured.generalize %tr : (!transform.any_op) -> !transform.any_op
    %f2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %f3 = transform.air.fuse_elementwise_linalg %f2 fold_reshapes : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f3 { transform.apply_patterns.canonicalization } : !transform.any_op
    // unit dims away, so the nibble shifts fuse into each dequant
    %f4 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f4 {
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %f5 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %f6 = transform.air.fuse_elementwise_linalg %f5 fold_reshapes : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f6 { transform.apply_patterns.canonicalization } : !transform.any_op
    // The prologue, once per dispatch: each core copies a run of x into
    // its L1 and expands it there.
    %pfs = transform.structured.match ops{["scf.forall"]} attributes{pro_forall} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %pfs : !transform.any_op {
    ^bb0(%pxf: !transform.any_op):
      %px1 = transform.structured.match ops{["linalg.generic"]} in %pxf : (!transform.any_op) -> !transform.any_op
      transform.annotate %px1 "x_exp" : !transform.any_op
      %pex = transform.get_producer_of_operand %px1[0] : (!transform.any_op) -> !transform.any_op
      %pexf, %pxf1 = transform.structured.fuse_into_containing_op %pex into %pxf : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %psl = transform.get_producer_of_operand %pexf[0] : (!transform.any_op) -> !transform.any_op
      %pslf, %pxf2 = transform.structured.fuse_into_containing_op %psl into %pxf1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    }
    %pf = transform.structured.match ops{["scf.forall"]} attributes{pro_forall} in %arg1 : (!transform.any_op) -> !transform.any_op
    %pf0, %pf1, %pf2 = transform.split_handle %pf : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %pf01 = transform.loop.fuse_sibling %pf1 into %pf0 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    %pf012 = transform.loop.fuse_sibling %pf2 into %pf01 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.annotate %pf012 "pro_forall" : !transform.any_op
    // x's runs as one copy into each core's L1
    %pxs2 = transform.structured.match ops{["linalg.generic"]} attributes{x_exp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %pxc = transform.air.coalesce_slices %pxs2 : (!transform.any_op) -> !transform.any_op
    transform.annotate %pxc "x_copy" : !transform.any_op
    // the expansion's reshape folded into its insert, so it writes the
    // core's xexp in place
    %pfo = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %pfo {
      transform.apply_patterns.tensor.fold_tensor_subset_ops
      transform.apply_patterns.tensor.reassociative_reshape_folding
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.include @bubble_slices failures(propagate) (%arg1) : (!transform.any_op) -> ()
    // Name the pieces: each run's multiply-accumulate (one per run of
    // packets), their zero starts and dequants, and the epilogue.
    %nr = transform.structured.match ops{["linalg.generic"]} filter_result_type = tensor<8x4x8x8x8xf32> in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.annotate %nr "red" : !transform.any_op
    %nf = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<8x4x8x8x8xf32> in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.annotate %nf "acc_fill" : !transform.any_op
    transform.foreach %nr : !transform.any_op {
    ^bb0(%nr1: !transform.any_op):
      %nc = transform.get_producer_of_operand %nr1[0] : (!transform.any_op) -> !transform.any_op
      %nd = transform.get_producer_of_operand %nc[0] : (!transform.any_op) -> !transform.any_op
      transform.annotate %nd "dequant" : !transform.any_op
    }
    %ns = transform.structured.match ops{["linalg.generic"]} filter_result_type = tensor<8x4x8x8xf32> in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.annotate %ns "epi_red" : !transform.any_op
    %nsf = transform.get_producer_of_operand %ns[3] : (!transform.any_op) -> !transform.any_op
    transform.annotate %nsf "epi_fill" : !transform.any_op
    %ne = transform.structured.match ops{["linalg.generic"]} filter_result_type = tensor<8x4x8x8xbf16> in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.annotate %ne "epi_cast" : !transform.any_op
    // One core per (column, row), everything of the row tile inside.
    %ct, %cf = transform.structured.tile_using_forall %ne tile_sizes [1, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %cf "core_forall" : !transform.any_op
    %fu4 = transform.structured.match ops{["linalg.generic"]} attributes{epi_red} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fu5 = transform.structured.match ops{["scf.forall"]} attributes{core_forall} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fu6, %fu7 = transform.structured.fuse_into_containing_op %fu4 into %fu5 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fu8 = transform.structured.match ops{["linalg.fill"]} attributes{epi_fill} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fu9 = transform.structured.match ops{["scf.forall"]} attributes{core_forall} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fu10, %fu11 = transform.structured.fuse_into_containing_op %fu8 into %fu9 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @bubble_up failures(propagate) (%arg1) : (!transform.any_op) -> ()
    %fu13 = transform.structured.match ops{["linalg.generic"]} attributes{red} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fu14 = transform.structured.match ops{["scf.forall"]} attributes{core_forall} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fu15, %fu16 = transform.structured.fuse_into_containing_op %fu13 into %fu14 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @bubble_slices failures(propagate) (%arg1) : (!transform.any_op) -> ()
    %fu20 = transform.structured.match ops{["linalg.generic"]} attributes{red} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fu21 = transform.structured.match ops{["scf.forall"]} attributes{core_forall} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fu22, %fu23 = transform.structured.fuse_into_containing_op %fu20 into %fu21 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @bubble_slices failures(propagate) (%arg1) : (!transform.any_op) -> ()
    %fu27 = transform.structured.match ops{["linalg.generic"]} attributes{red} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fu28 = transform.structured.match ops{["scf.forall"]} attributes{core_forall} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fu29, %fu30 = transform.structured.fuse_into_containing_op %fu27 into %fu28 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @bubble_slices failures(propagate) (%arg1) : (!transform.any_op) -> ()
    %fu34 = transform.structured.match ops{["linalg.fill"]} attributes{acc_fill} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fu35 = transform.structured.match ops{["scf.forall"]} attributes{core_forall} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fu36, %fu37 = transform.structured.fuse_into_containing_op %fu34 into %fu35 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @bubble_up failures(propagate) (%arg1) : (!transform.any_op) -> ()
    %fu39 = transform.structured.match ops{["linalg.generic"]} attributes{dequant} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fu40 = transform.structured.match ops{["scf.forall"]} attributes{core_forall} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fu41, %fu42 = transform.structured.fuse_into_containing_op %fu39 into %fu40 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @bubble_up failures(propagate) (%arg1) : (!transform.any_op) -> ()
    %ms44 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %ms44 {
      transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.include @bubble_slices failures(propagate) (%arg1) : (!transform.any_op) -> ()
    transform.include @bubble_slices failures(propagate) (%arg1) : (!transform.any_op) -> ()
    // Inside each core, K a packet (32 k tiles) at a time: the packet in L1.
    %tr56 = transform.structured.match ops{["linalg.generic"]} attributes{red} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %tr56 : !transform.any_op {
    ^bb0(%tr561: !transform.any_op):
      %tr54, %tr55 = transform.structured.tile_using_for %tr561 tile_sizes [32, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.annotate %tr55 "pkt_loop" : !transform.any_op
    }
    %bc57 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %bc58 = transform.air.bubble_slice_through_collapse %bc57 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %bc58 { transform.apply_patterns.canonicalization } : !transform.any_op
    %il59s = transform.structured.match ops{["scf.for"]} attributes{pkt_loop} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %il59s : !transform.any_op {
    ^bb0(%il59: !transform.any_op):
      %il60 = transform.structured.match ops{["linalg.generic"]} attributes{red} in %il59 : (!transform.any_op) -> !transform.any_op
      %il61 = transform.get_producer_of_operand %il60[0] : (!transform.any_op) -> !transform.any_op
      %il62 = transform.get_producer_of_operand %il61[0] : (!transform.any_op) -> !transform.any_op
      %il63 = transform.get_producer_of_operand %il62[0] : (!transform.any_op) -> !transform.any_op
      %il64, %il65 = transform.structured.fuse_into_containing_op %il63 into %il59 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    }
    transform.include @bubble_up failures(propagate) (%arg1) : (!transform.any_op) -> ()
    %ms67 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %ms67 {
      transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.include @bubble_slices failures(propagate) (%arg1) : (!transform.any_op) -> ()
    %pcs = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %pcs : !transform.any_op
    %pd = transform.structured.match ops{["linalg.generic"]} attributes{dequant} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %pd : !transform.any_op {
    ^bb0(%pd1: !transform.any_op):
      %pc = transform.air.coalesce_slices %pd1 : (!transform.any_op) -> !transform.any_op
      transform.annotate %pc "pkt_copy" : !transform.any_op
    }
    transform.include @bubble_slices failures(propagate) (%arg1) : (!transform.any_op) -> ()
    %fl = transform.structured.match ops{["scf.for"]} attributes{pkt_loop} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fl0, %fl1, %fl2 = transform.split_handle %fl : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %fl01 = transform.loop.fuse_sibling %fl0 into %fl1 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    %fl012 = transform.loop.fuse_sibling %fl01 into %fl2 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.annotate %fl012 "pkt_loop" : !transform.any_op
    transform.include @bubble_slices failures(propagate) (%arg1) : (!transform.any_op) -> ()
    // a packet's scale groups in turn (4 k tiles), each dequantized just before use
    %tr79 = transform.structured.match ops{["linalg.generic"]} attributes{red} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %tr79 : !transform.any_op {
    ^bb0(%tr791: !transform.any_op):
      %tr77, %tr78 = transform.structured.tile_using_for %tr791 tile_sizes [4, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.annotate %tr78 "g_loop" : !transform.any_op
    }
    %bc80 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %bc81 = transform.air.bubble_slice_through_collapse %bc80 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %bc81 { transform.apply_patterns.canonicalization } : !transform.any_op
    %il82s = transform.structured.match ops{["scf.for"]} attributes{g_loop} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %il82s : !transform.any_op {
    ^bb0(%il82: !transform.any_op):
      %il83 = transform.structured.match ops{["linalg.generic"]} attributes{red} in %il82 : (!transform.any_op) -> !transform.any_op
      %il84 = transform.get_producer_of_operand %il83[0] : (!transform.any_op) -> !transform.any_op
      %il85 = transform.get_producer_of_operand %il84[0] : (!transform.any_op) -> !transform.any_op
      %il86 = transform.get_producer_of_operand %il85[0] : (!transform.any_op) -> !transform.any_op
      %il87, %il88 = transform.structured.fuse_into_containing_op %il86 into %il82 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    }
    transform.include @bubble_slices failures(propagate) (%arg1) : (!transform.any_op) -> ()
    %tv92 = transform.structured.match ops{["linalg.generic"]} attributes{red} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %tv92 : !transform.any_op {
    ^bb0(%tv93: !transform.any_op):
      %mme_t, %tl94, %tl95, %tl96, %tl97, %tl98 = transform.structured.tile_using_for %tv93 tile_sizes [1, 1, 1, 1, 4, 0] interchange = [1, 2, 3, 4, 0, 5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    }
    %ud0 = transform.structured.match ops{["linalg.generic"]} attributes{red} in %arg1 : (!transform.any_op) -> !transform.any_op
    %ud1 = transform.get_parent_op %ud0 <{deduplicate, op_name = "scf.for"}> : (!transform.any_op) -> !transform.any_op
    transform.annotate %ud1 "mac_loop" : !transform.any_op
    transform.apply_patterns to %ud1 {
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    // what a core works on in its L1
    %e0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %e0 {
      transform.apply_patterns.tensor.fold_tensor_empty
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %pk = transform.structured.match ops{["linalg.copy"]} attributes{pkt_copy} in %arg1 : (!transform.any_op) -> !transform.any_op
    %pkb, %pkn = transform.structured.bufferize_to_allocation %pk <{memory_space = 2, bufferize_destination_only}> : !transform.any_op
    %xc = transform.structured.match ops{["linalg.copy"]} attributes{x_copy} in %arg1 : (!transform.any_op) -> !transform.any_op
    %xcb, %xcn = transform.structured.bufferize_to_allocation %xc <{memory_space = 2, bufferize_destination_only, emit_dealloc}> : !transform.any_op
    %al99 = transform.structured.match ops{["linalg.generic"]} attributes{epi_cast} in %arg1 : (!transform.any_op) -> !transform.any_op
    %al100, %al101 = transform.structured.bufferize_to_allocation %al99 <{memory_space = 2, bufferize_destination_only, emit_dealloc}> : !transform.any_op
    %al102 = transform.structured.match ops{["linalg.generic"]} attributes{dequant} in %arg1 : (!transform.any_op) -> !transform.any_op
    %al103, %al104 = transform.structured.bufferize_to_allocation %al102 <{memory_space = 2, bufferize_destination_only}> : !transform.any_op
    %ef = transform.structured.match ops{["linalg.fill"]} attributes{epi_fill} in %arg1 : (!transform.any_op) -> !transform.any_op
    %efb, %efn = transform.structured.bufferize_to_allocation %ef <{memory_space = 2, bufferize_destination_only, emit_dealloc}> : !transform.any_op
    // the expansion is written straight into each core's L1, where the row
    // tiles read it
    %xee = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.bufferization.eliminate_empty_tensors %xee : !transform.any_op
    %xe0 = transform.structured.match ops{["tensor.empty"]} filter_result_type = tensor<8x4x64x8x8xbf16> in %arg1 : (!transform.any_op) -> !transform.any_op
    %xe1 = transform.cast %xe0 : !transform.any_op to !transform.op<"tensor.empty">
    %xe2 = transform.bufferization.empty_tensor_to_alloc_tensor %xe1 : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">
    %xe3, %xe4 = transform.structured.bufferize_to_allocation %xe2 <{memory_space = 2}> : !transform.op<"bufferization.alloc_tensor">
    %em121 = transform.structured.match ops{["scf.forall"]} attributes{core_forall} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %em121 : !transform.any_op {
    ^bb0(%em122: !transform.any_op):
      %em123 = transform.structured.match ops{["tensor.empty"]} in %em122 : (!transform.any_op) -> !transform.any_op
      transform.foreach %em123 : !transform.any_op {
      ^bb0(%em124: !transform.any_op):
        %em125 = transform.cast %em124 : !transform.any_op to !transform.op<"tensor.empty">
        %em126 = transform.bufferization.empty_tensor_to_alloc_tensor %em125 : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">
        %em127, %em128 = transform.structured.bufferize_to_allocation %em126 <{memory_space = 2, emit_dealloc}> : !transform.op<"bufferization.alloc_tensor">
      }
    }
    %em131 = transform.structured.match ops{["tensor.empty"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      transform.foreach %em131 : !transform.any_op {
      ^bb0(%em132: !transform.any_op):
        %em133 = transform.cast %em132 : !transform.any_op to !transform.op<"tensor.empty">
        %em134 = transform.bufferization.empty_tensor_to_alloc_tensor %em133 : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">
        %em135, %em136 = transform.structured.bufferize_to_allocation %em134 <{memory_space = 1, emit_dealloc}> : !transform.op<"bufferization.alloc_tensor">
      }
    // Bufferize.
    %b0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %b0 { transform.apply_patterns.tensor.fold_tensor_empty
      transform.apply_patterns.canonicalization } : !transform.any_op
    transform.bufferization.eliminate_empty_tensors %b0 : !transform.any_op
    %b1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %b2 = transform.bufferization.one_shot_bufferize %b1 : (!transform.any_op) -> !transform.any_op
    %b3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %b3 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %b3 : !transform.any_op
    transform.apply_patterns to %b3 { transform.apply_patterns.canonicalization } : !transform.any_op
    %b4 = transform.air.remove_uninitialized_copy %b3 : (!transform.any_op) -> !transform.any_op
    %b5 = transform.air.eliminate_cascade_memcpy %b4 : (!transform.any_op) -> !transform.any_op
    // One vector op per tile: the dequant 64 lanes of one (nb, kb) tile,
    // the contraction one 4x8x8 matmul, the broadcast and fill a row.
    %tv137 = transform.structured.match ops{["linalg.generic"]} attributes{dequant} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %tv137 : !transform.any_op {
    ^bb0(%tv138: !transform.any_op):
      %dqv_t, %tl139, %tl140, %tl141, %tl142, %tl143, %tl144 = transform.structured.tile_using_for %tv138 tile_sizes [1, 1, 1, 1, 1, 1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    }
    %tv152 = transform.structured.match ops{["linalg.generic"]} attributes{epi_red} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %tv152 : !transform.any_op {
    ^bb0(%tv153: !transform.any_op):
      %epi_redv_t, %tl154, %tl155, %tl156, %tl157 = transform.structured.tile_using_for %tv153 tile_sizes [1, 1, 1, 4, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    }
    %tv158 = transform.structured.match ops{["linalg.generic"]} attributes{epi_cast} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %tv158 : !transform.any_op {
    ^bb0(%tv159: !transform.any_op):
      %epi_castv_t, %tl160, %tl161, %tl162 = transform.structured.tile_using_for %tv159 tile_sizes [1, 1, 4, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    }
    // The row tile's L3 transfers one per column, so each column's memtile
    // holds its own part (the 1-D x copy is left whole).
    %l3f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %l3f2 = transform.air.convert_memref_copy_to_linalg_copy %l3f : (!transform.any_op) -> !transform.any_op
    %l3c = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %l3c : !transform.any_op {
    ^bb0(%l3c1: !transform.any_op):
      transform.sequence %l3c1 : !transform.any_op failures(suppress) {
      ^bb1(%l3c2: !transform.any_op):
        // only the output row tile ([8, 4, 64]); the packet stream is split
        // per column by aircc already
        %l3m = transform.match.structured failures(propagate) %l3c2 : (!transform.any_op) -> !transform.any_op {
        ^bb2(%l3s: !transform.any_op):
          %l3d = transform.match.structured.dim %l3s[2] : (!transform.any_op) -> !transform.param<i64>
          %l3k = transform.param.constant 64 : i64 -> !transform.param<i64>
          transform.match.param.cmpi eq %l3d, %l3k : !transform.param<i64>
          transform.match.structured.yield %l3s : !transform.any_op
        }
        %l3t, %l3l = transform.structured.tile_using_for %l3m tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        // unrolled: each column's transfer at a constant offset, as a
        // transfer of its own
        transform.loop.unroll %l3l factor = 8 : !transform.any_op
      }
    }
    // The runs one after the other share their L1 buffers, so the cores
    // take every run's packets through one buffer and one channel.
    %pu = transform.structured.match ops{["scf.for"]} attributes{pkt_loop} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %pu factor = 2 : !transform.any_op
    %sh0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %sh1 = transform.apply_registered_pass "buffer-loop-hoisting" to %sh0 : (!transform.any_op) -> !transform.any_op
    %pkl = transform.structured.match ops{["memref.alloc"]} filter_result_type = memref<1x1x1x5120xi16, 2> in %arg1 : (!transform.any_op) -> !transform.any_op
    %pk0, %pk1, %pk2, %pk3, %pk4, %pk5 = transform.split_handle %pkl : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %pkga = transform.merge_handles %pk0, %pk2, %pk4 : !transform.any_op
    %pkgb = transform.merge_handles %pk1, %pk3, %pk5 : !transform.any_op
    %pka1 = transform.air.share_alloc %pkga : (!transform.any_op) -> !transform.any_op
    %pkb1 = transform.air.share_alloc %pkgb : (!transform.any_op) -> !transform.any_op
    %sh_1 = transform.structured.match ops{["memref.alloc"]} filter_result_type = memref<1x1x8x1x1x4x8x2x4xbf16, 2> in %arg1 : (!transform.any_op) -> !transform.any_op
    %sh_1k = transform.air.share_alloc %sh_1 : (!transform.any_op) -> !transform.any_op
    %tv163 = transform.structured.match ops{["linalg.generic"]} attributes{x_exp} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %tv163 : !transform.any_op {
    ^bb0(%tv164: !transform.any_op):
      %xev_t, %tl165, %tl166 = transform.structured.tile_using_for %tv164 tile_sizes [1, 4, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    }
    // the expansion's move into the cores' xexp is core code (L1 to L1)
    %xm2 = transform.structured.match ops{["linalg.copy"]} filter_operand_types = [memref<1x1x64x8x8xbf16, strided<[4096, 4096, 64, 8, 1], offset: ?>, 2>, memref<1x1x64x8x8xbf16, strided<[16384, 4096, 64, 8, 1], offset: ?>, 2>] in %arg1 : (!transform.any_op) -> !transform.any_op
    %xm3 = transform.structured.generalize %xm2 : (!transform.any_op) -> !transform.any_op
    transform.foreach %xm3 : !transform.any_op {
    ^bb0(%xm4: !transform.any_op):
      %xm5, %xm6, %xm7, %xm8 = transform.structured.tile_using_for %xm4 tile_sizes [1, 1, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    }
    // A core's copies to and from L2 are DMAs (its L1-to-L1 ones stay code).
    %dm0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %dm0 { transform.apply_patterns.canonicalization } : !transform.any_op
    %dmf = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %dmf : !transform.any_op {
    ^bb0(%dmf1: !transform.any_op):
      %dml = transform.structured.match ops{["linalg.copy"]} in %dmf1 : (!transform.any_op) -> !transform.any_op
      transform.foreach %dml : !transform.any_op {
      ^bb1(%dml1: !transform.any_op):
        %dmm = transform.structured.linalg_copy_to_memref %dml1 : (!transform.any_op) -> !transform.any_op
      }
      %dmc = transform.structured.match ops{["memref.copy"]} in %dmf1 : (!transform.any_op) -> !transform.any_op
      transform.foreach %dmc : !transform.any_op {
      ^bb1(%dmc1: !transform.any_op):
        %dmd = transform.air.copy_to_dma %dmc1 : (!transform.any_op) -> !transform.any_op
      }
    }
    %cr0 = transform.structured.match ops{["scf.for"]} attributes{mac_loop} in %arg1 : (!transform.any_op) -> !transform.any_op
    %cr1 = transform.get_parent_op %cr0 <{deduplicate, op_name = "scf.for"}> : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %cr0 factor = 4 : !transform.any_op
    transform.loop.unroll %cr1 factor = 2 : !transform.any_op
    // The foralls are herds: (column, row) of the array.
    %hf = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %hf : !transform.any_op {
    ^bb0(%hf1: !transform.any_op):
      %hp = transform.loop.forall_to_parallel %hf1 : (!transform.any_op) -> !transform.any_op
      %hh = transform.air.par_to_herd %hp : (!transform.any_op) -> !transform.any_op
      // one name: the same cores, the herds in turn
      %hname = transform.param.constant "gemv" -> !transform.any_param
      transform.annotate %hh "sym_name" = %hname : !transform.any_op, !transform.any_param
    }
    %hs = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %hv = transform.air.herd_vectorize %hs : (!transform.any_op) -> !transform.any_op
    %hc = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %hc {
      transform.apply_patterns.memref.fold_memref_alias_ops
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %hc : !transform.any_op
    %fw0 = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %fw0 : (!transform.any_op) -> ()
    %hs2 = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %hlin = transform.air.linearize_vectors %hs2 arch = "aie2p" f32_lanes = 64 contract : (!transform.any_op) -> !transform.any_op
    %vf0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %vf0 {
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.vector.drop_unit_dims_with_shape_cast
      transform.apply_patterns.canonicalization
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    %vf1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %vf2 = transform.air.fold_unit_extent_dims %vf1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
