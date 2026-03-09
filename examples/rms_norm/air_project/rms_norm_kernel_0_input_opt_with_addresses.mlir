#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>, llvm.target_triple = "aie2p"} {
  llvm.mlir.global external @buf0() {addr_space = 0 : i32} : !llvm.array<1 x array<64 x bf16>>
  llvm.mlir.global external @buf1() {addr_space = 0 : i32} : !llvm.array<1 x f32>
  llvm.mlir.global external @buf2() {addr_space = 0 : i32} : !llvm.array<1 x array<64 x bf16>>
  llvm.mlir.global external @buf3() {addr_space = 0 : i32} : !llvm.array<1 x f32>
  llvm.mlir.global external @buf4() {addr_space = 0 : i32} : !llvm.array<2 x array<64 x bf16>>
  llvm.mlir.global external @buf5() {addr_space = 0 : i32} : !llvm.array<2 x array<64 x bf16>>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.event(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.set.ctrl.reg(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @__air_herd_arg_1() {addr_space = 0 : i32} : !llvm.array<2 x array<64 x bf16>>
  llvm.mlir.global external @__air_herd_arg() {addr_space = 0 : i32} : !llvm.array<2 x array<64 x bf16>>
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @buf0 : !llvm.ptr
    %1 = llvm.mlir.addressof @buf1 : !llvm.ptr
    %2 = llvm.mlir.addressof @__air_herd_arg : !llvm.ptr
    %3 = llvm.mlir.constant(64 : index) : i32
    %4 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %5 = llvm.mlir.constant(6.400000e+01 : f32) : f32
    %6 = llvm.mlir.constant(9.99999974E-6 : f32) : f32
    %7 = llvm.mlir.constant(1 : index) : i32
    %8 = llvm.mlir.constant(0 : index) : i32
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb4
    %9 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x array<64 x bf16>>
    %10 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x f32>
    llvm.store %4, %10 : f32, !llvm.ptr
    llvm.br ^bb2(%8 : i32)
  ^bb2(%11: i32):  // 2 preds: ^bb1, ^bb3
    %12 = llvm.icmp "slt" %11, %3 : i32
    llvm.cond_br %12, ^bb3, ^bb4(%8 : i32)
  ^bb3:  // pred: ^bb2
    %13 = llvm.mul %8, %3 overflow<nsw, nuw> : i32
    %14 = llvm.add %13, %11 overflow<nsw, nuw> : i32
    %15 = llvm.getelementptr inbounds|nuw %9[%14] : (!llvm.ptr, i32) -> !llvm.ptr, bf16
    %16 = llvm.load %15 : !llvm.ptr -> bf16
    %17 = llvm.load %10 : !llvm.ptr -> f32
    %18 = llvm.fpext %16 : bf16 to f32
    %19 = llvm.fmul %18, %18 : f32
    %20 = llvm.fadd %19, %17 : f32
    llvm.store %20, %10 : f32, !llvm.ptr
    %21 = llvm.add %11, %7 : i32
    llvm.br ^bb2(%21 : i32) {loop_annotation = #loop_annotation}
  ^bb4(%22: i32):  // 2 preds: ^bb2, ^bb5
    %23 = llvm.icmp "slt" %22, %3 : i32
    llvm.cond_br %23, ^bb5, ^bb1
  ^bb5:  // pred: ^bb4
    %24 = llvm.mul %8, %3 overflow<nsw, nuw> : i32
    %25 = llvm.add %24, %22 overflow<nsw, nuw> : i32
    %26 = llvm.getelementptr inbounds|nuw %9[%25] : (!llvm.ptr, i32) -> !llvm.ptr, bf16
    %27 = llvm.load %26 : !llvm.ptr -> bf16
    %28 = llvm.load %10 : !llvm.ptr -> f32
    %29 = llvm.fdiv %28, %5 : f32
    %30 = llvm.fadd %29, %6 : f32
    %31 = "xllvm.intr.aie2p.invsqrt"(%30) : (f32) -> f32
    %32 = llvm.fpext %27 : bf16 to f32
    %33 = llvm.fmul %32, %31 : f32
    %34 = llvm.fptrunc %33 : f32 to bf16
    %35 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<64 x bf16>>
    %36 = llvm.getelementptr inbounds|nuw %35[%25] : (!llvm.ptr, i32) -> !llvm.ptr, bf16
    llvm.store %34, %36 : bf16, !llvm.ptr
    %37 = llvm.add %22, %7 : i32
    llvm.br ^bb4(%37 : i32) {loop_annotation = #loop_annotation}
  }
  llvm.func @core_0_3() {
    %0 = llvm.mlir.addressof @buf2 : !llvm.ptr
    %1 = llvm.mlir.addressof @buf3 : !llvm.ptr
    %2 = llvm.mlir.addressof @__air_herd_arg_1 : !llvm.ptr
    %3 = llvm.mlir.constant(64 : index) : i32
    %4 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %5 = llvm.mlir.constant(6.400000e+01 : f32) : f32
    %6 = llvm.mlir.constant(9.99999974E-6 : f32) : f32
    %7 = llvm.mlir.constant(1 : index) : i32
    %8 = llvm.mlir.constant(0 : index) : i32
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb4
    %9 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x array<64 x bf16>>
    %10 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x f32>
    llvm.store %4, %10 : f32, !llvm.ptr
    llvm.br ^bb2(%8 : i32)
  ^bb2(%11: i32):  // 2 preds: ^bb1, ^bb3
    %12 = llvm.icmp "slt" %11, %3 : i32
    llvm.cond_br %12, ^bb3, ^bb4(%8 : i32)
  ^bb3:  // pred: ^bb2
    %13 = llvm.getelementptr %9[64] : (!llvm.ptr) -> !llvm.ptr, bf16
    %14 = llvm.mul %8, %3 overflow<nsw, nuw> : i32
    %15 = llvm.add %14, %11 overflow<nsw, nuw> : i32
    %16 = llvm.getelementptr inbounds|nuw %13[%15] : (!llvm.ptr, i32) -> !llvm.ptr, bf16
    %17 = llvm.load %16 : !llvm.ptr -> bf16
    %18 = llvm.load %10 : !llvm.ptr -> f32
    %19 = llvm.fpext %17 : bf16 to f32
    %20 = llvm.fmul %19, %19 : f32
    %21 = llvm.fadd %20, %18 : f32
    llvm.store %21, %10 : f32, !llvm.ptr
    %22 = llvm.add %11, %7 : i32
    llvm.br ^bb2(%22 : i32) {loop_annotation = #loop_annotation}
  ^bb4(%23: i32):  // 2 preds: ^bb2, ^bb5
    %24 = llvm.icmp "slt" %23, %3 : i32
    llvm.cond_br %24, ^bb5, ^bb1
  ^bb5:  // pred: ^bb4
    %25 = llvm.getelementptr %9[64] : (!llvm.ptr) -> !llvm.ptr, bf16
    %26 = llvm.mul %8, %3 overflow<nsw, nuw> : i32
    %27 = llvm.add %26, %23 overflow<nsw, nuw> : i32
    %28 = llvm.getelementptr inbounds|nuw %25[%27] : (!llvm.ptr, i32) -> !llvm.ptr, bf16
    %29 = llvm.load %28 : !llvm.ptr -> bf16
    %30 = llvm.load %10 : !llvm.ptr -> f32
    %31 = llvm.fdiv %30, %5 : f32
    %32 = llvm.fadd %31, %6 : f32
    %33 = "xllvm.intr.aie2p.invsqrt"(%32) : (f32) -> f32
    %34 = llvm.fpext %29 : bf16 to f32
    %35 = llvm.fmul %34, %33 : f32
    %36 = llvm.fptrunc %35 : f32 to bf16
    %37 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<64 x bf16>>
    %38 = llvm.getelementptr inbounds|nuw %37[%27] : (!llvm.ptr, i32) -> !llvm.ptr, bf16
    llvm.store %36, %38 : bf16, !llvm.ptr
    %39 = llvm.add %23, %7 : i32
    llvm.br ^bb4(%39 : i32) {loop_annotation = #loop_annotation}
  }
}
