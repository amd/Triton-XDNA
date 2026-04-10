#!/bin/bash
set -e
# Repeater script for: LLVM lowering
echo "Original MLIR Diagnostics:"
cat << 'DIAGNOSTICS_EOF'
aievec.mul_elem conversion is not supported for AIE2p.

failed to legalize operation 'aievec.mul_elem' that was explicitly marked illegal: %21 = "aievec.mul_elem"(%20, %20) : (vector<32xi16>, vector<32xi16>) -> vector<32xi32>
DIAGNOSTICS_EOF
echo ""

MLIR_FILE='air_project/aiecc_failure_1775797174_862028.mlir'
PASS_PIPELINE='any(aie.device(aie-localize-locks,aie-normalize-address-spaces,aie-transform-bfp-types),aie-standard-lowering{device=square_kernel_0 tilecol=0 tilerow=5},aiex-standard-lowering,convert-aievec-to-llvm{aie-target=aie2p aie2-fp32-emulation-strategy=accuracy-safe},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,expand-strided-metadata,lower-affine,arith-expand{include-bf16=false include-f4e2m1=false include-f8e8m0=false},finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false},convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=true},convert-to-llvm{allow-pattern-rollback=true dynamic=true },convert-vector-to-llvm{enable-arm-bf16=false enable-arm-i8mm=false enable-arm-neon=false enable-arm-sve=false enable-x86=false force-32bit-vector-indices=true reassociate-fp-reductions=false use-vector-alignment=false vector-contract-lowering=dot vector-transpose-lowering=eltwise},convert-ub-to-llvm{index-bitwidth=0},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse)'
aie-opt --mlir-print-ir-after-all --mlir-disable-threading --pass-pipeline="$PASS_PIPELINE" "$MLIR_FILE"
