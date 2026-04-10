#!/bin/bash
set -e
# Repeater script for: resource allocation
echo "Original MLIR Diagnostics:"
cat << 'DIAGNOSTICS_EOF'
failed to legalize operation 'arith.subi' that was explicitly marked illegal: %120 = "arith.subi"(%118, %119) <{overflowFlags = #arith.overflow<none>}> : (vector<32xi16>, vector<32xi16>) -> vector<32xi16>
DIAGNOSTICS_EOF
echo ""

MLIR_FILE='air_project/aiecc_failure_1775797115_856352.mlir'
PASS_PIPELINE='any(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},unknown<RedundantLoadStoreOptimizationPass>,unknown<ReorderOperationsPass>,unknown<{anonymous}::CopyRemovalPass>,unknown<VectorBroadcastLoweringPass>,test-canonicalize-vector-for-aievec{aie-target=aie2p target-backend=llvmir},test-lower-vector-to-aievec{aie-target=aie2p target-backend=llvmir},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},unknown<ExtendUPDOpsPass>,cse,unknown<SimplifyUPDOpsPass>,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},test-aievec-optimize{aie-target=aie2p target-backend=llvmir},cse,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},aievec-convolution-analysis{print=false},test-aievec-convolution-optimize{aie-target=aie2p shift=0 target-backend=llvmir},cse,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},loop-invariant-code-motion,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},lower-affine,aie-canonicalize-device,aie.device(aie-assign-lock-ids,aie-register-objectFifos,aie-objectFifo-stateful-transform{dynamic-objFifos=false packet-sw-objFifos=false},aie-assign-bd-ids,aie-lower-cascade-flows,aie-lower-broadcast-packet,aie-lower-multicast,aie-assign-tile-controller-ids{column-wise-unique-ids=true},aie-generate-column-control-overlay{route-shim-to-tct=shim-only route-shim-to-tile-ctrl=false},aie-assign-buffer-addresses{alloc-scheme=},aie-assign-core-link-files,aie-vector-transfer-lowering{max-transfer-rank=4294967295}),convert-scf-to-cf{allow-pattern-rollback=true})'
aie-opt --mlir-print-ir-after-all --mlir-disable-threading --pass-pipeline="$PASS_PIPELINE" "$MLIR_FILE"
