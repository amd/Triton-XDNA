// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1  : (!transform.any_op) -> !transform.any_op

        // Bufferize result to shared (L2) memory allocation
        %buffer_res_shared, %new_fill = transform.structured.bufferize_to_allocation %fill
          {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

        // Find the copy operations to tile using for.
        %func_1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.air.convert_memref_copy_to_linalg_copy %func_1 : (!transform.any_op) -> !transform.any_op
        %copies = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %copy_1, %copy_2 = transform.split_handle %copies : (!transform.any_op<"linalg.copy">) -> (!transform.any_op<"linalg.copy">, !transform.any_op<"linalg.copy">)
        %tiled_copy_1, %tiled_copy_for_loop_1 =
          transform.structured.tile_using_for %copy_1 tile_sizes [0, 256]
          : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
        %tiled_copy_2, %tiled_copy_for_loop_2 =
          transform.structured.tile_using_for %copy_2 tile_sizes [256, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)

        // Second level tile to forall with tile_sizes.
        %matmul_1 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %tiled_matmul_1, %forall_1 =
          transform.structured.tile_using_forall %matmul_1 tile_sizes [64, 64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        // Run canonicalization
        %func_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_2 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_cse to %func_2 : !transform.any_op

        // Fuse fill operation into the forall loop.
        %fused_fill_1 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %fill_consumer = transform.get_consumers_of_result %fused_fill_1[0] : (!transform.any_op) -> (!transform.any_op)
        %fused_fill_2, %fused_loop_2 = transform.structured.fuse_into_containing_op %fused_fill_1 into %fill_consumer : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

        // Pack by applying data tiling, and the linalg.matmul becomes linalg.generic.
        %packed = transform.structured.pack %tiled_matmul_1 packed_sizes = [4, 4, 8]
          : (!transform.any_op) -> (!transform.any_op)

        // Transpose A matrix.
        %pack_producer_a = transform.get_producer_of_operand %packed[0]
          : (!transform.any_op) -> (!transform.any_op)
        %packed_a, %pack_a, %empty_unpack_a =
          transform.structured.pack_transpose %pack_producer_a with_compute_op(%packed)
          outer_perm = [1, 0] : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)

        // Transpose B matrix.
        %pack_producer_b = transform.get_producer_of_operand %packed_a[1]
          : (!transform.any_op) -> (!transform.any_op)
        %packed_b, %pack_b, %empty_unpack_b =
          transform.structured.pack_transpose %pack_producer_b with_compute_op(%packed_a)
          outer_perm = [1, 0] inner_perm = [1, 0] : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)

        // Transpose C matrix.
        %unpack = transform.get_consumers_of_result %packed_b[0]
          : (!transform.any_op) -> (!transform.any_op)
        %packed_c, %pack_c, %unpack_c =
          transform.structured.pack_transpose %unpack with_compute_op(%packed_b)
          outer_perm = [1, 0] : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)

        // Bufferize result to local memory allocation
        %buffer_c, %new_c = transform.structured.bufferize_to_allocation %pack_c
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

        // Tile the reduction loop.
        %tiled_reduction, %for_loop =
          transform.structured.tile_using_for %packed_c tile_sizes [0, 0, 4]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        // Fuse pack ops into the for loop.
        %fused_pack_a, %e1 = transform.structured.fuse_into_containing_op %pack_a into %for_loop
          : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %fused_pack_b, %e2 = transform.structured.fuse_into_containing_op %pack_b into %for_loop
          : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

        // Promote the inputs to local memory.
        %buffer_a, %new_a = transform.structured.bufferize_to_allocation %fused_pack_a
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
        %buffer_b, %new_b = transform.structured.bufferize_to_allocation %fused_pack_b
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

        // Run canonicalization
        %func_3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_3 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_cse to %func_3 : !transform.any_op

        
        // Bufferize
        %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!transform.any_op) -> !transform.any_op

        // Run canonicalization to remove redundant memcpy (with linalg.generic form) ops created, which can be deleted by canonicalizer. We have to run it again because the memrefs are unified in CSE pass, so we can truely remove redundant memcpy.
        %func6 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func6 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_cse to %func6 : !transform.any_op
        transform.apply_patterns to %func6 {
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %func_op_updated = transform.air.remove_uninitialized_copy %func6 : (!transform.any_op) -> !transform.any_op
        %func_op_updated_1 = transform.air.eliminate_cascade_memcpy %func_op_updated : (!transform.any_op) -> !transform.any_op

        // Tile linalg.generics for vectorization
        %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_generics, %vec_loops:3 =
          transform.structured.tile_using_for %linalg_generics tile_sizes [1, 1, 1, 0, 0, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)     

        // Tile linalg.fills for vectorized write
        %linalg_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_fills, %vec_fill_loops:2 =
          transform.structured.tile_using_for %linalg_fills tile_sizes [1, 1]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)         
    transform.yield
  }
}
