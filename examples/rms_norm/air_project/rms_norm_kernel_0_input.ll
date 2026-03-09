; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@buf0 = external global [1 x [64 x bfloat]]
@buf1 = external global [1 x float]
@buf2 = external global [1 x [64 x bfloat]]
@buf3 = external global [1 x float]
@buf4 = external global [2 x [64 x bfloat]]
@buf5 = external global [2 x [64 x bfloat]]
@__air_herd_arg_1 = external global [2 x [64 x bfloat]]
@__air_herd_arg = external global [2 x [64 x bfloat]]

declare void @debug_i32(i32)

; Unknown intrinsic
declare void @llvm.aie2p.event(i32)

; Unknown intrinsic
declare void @llvm.aie2p.put.ms(i32, i32)

; Unknown intrinsic
declare { i32, i32 } @llvm.aie2p.get.ss()

; Unknown intrinsic
declare void @llvm.aie2p.mcd.write.vec(<16 x i32>, i32)

; Unknown intrinsic
declare <16 x i32> @llvm.aie2p.scd.read.vec(i32)

; Unknown intrinsic
declare void @llvm.aie2p.acquire(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.release(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.set.ctrl.reg(i32, i32)

define void @core_0_2() {
  br label %1

1:                                                ; preds = %14, %0
  store float 0.000000e+00, ptr @buf1, align 4
  br label %2

2:                                                ; preds = %5, %1
  %3 = phi i32 [ %13, %5 ], [ 0, %1 ]
  %4 = icmp slt i32 %3, 64
  br i1 %4, label %5, label %14

5:                                                ; preds = %2
  %6 = add nuw nsw i32 0, %3
  %7 = getelementptr inbounds nuw bfloat, ptr @__air_herd_arg, i32 %6
  %8 = load bfloat, ptr %7, align 2
  %9 = load float, ptr @buf1, align 4
  %10 = fpext bfloat %8 to float
  %11 = fmul float %10, %10
  %12 = fadd float %11, %9
  store float %12, ptr @buf1, align 4
  %13 = add i32 %3, 1
  br label %2, !llvm.loop !1

14:                                               ; preds = %17, %2
  %15 = phi i32 [ %29, %17 ], [ 0, %2 ]
  %16 = icmp slt i32 %15, 64
  br i1 %16, label %17, label %1

17:                                               ; preds = %14
  %18 = add nuw nsw i32 0, %15
  %19 = getelementptr inbounds nuw bfloat, ptr @__air_herd_arg, i32 %18
  %20 = load bfloat, ptr %19, align 2
  %21 = load float, ptr @buf1, align 4
  %22 = fdiv float %21, 6.400000e+01
  %23 = fadd float %22, 0x3EE4F8B580000000
  %24 = call float @llvm.aie2p.invsqrt(float %23)
  %25 = fpext bfloat %20 to float
  %26 = fmul float %25, %24
  %27 = fptrunc float %26 to bfloat
  %28 = getelementptr inbounds nuw bfloat, ptr @buf0, i32 %18
  store bfloat %27, ptr %28, align 2
  %29 = add i32 %15, 1
  br label %14, !llvm.loop !1
}

define void @core_0_3() {
  br label %1

1:                                                ; preds = %14, %0
  store float 0.000000e+00, ptr @buf3, align 4
  br label %2

2:                                                ; preds = %5, %1
  %3 = phi i32 [ %13, %5 ], [ 0, %1 ]
  %4 = icmp slt i32 %3, 64
  br i1 %4, label %5, label %14

5:                                                ; preds = %2
  %6 = add nuw nsw i32 0, %3
  %7 = getelementptr inbounds nuw bfloat, ptr getelementptr inbounds nuw (i8, ptr @__air_herd_arg_1, i64 128), i32 %6
  %8 = load bfloat, ptr %7, align 2
  %9 = load float, ptr @buf3, align 4
  %10 = fpext bfloat %8 to float
  %11 = fmul float %10, %10
  %12 = fadd float %11, %9
  store float %12, ptr @buf3, align 4
  %13 = add i32 %3, 1
  br label %2, !llvm.loop !1

14:                                               ; preds = %17, %2
  %15 = phi i32 [ %29, %17 ], [ 0, %2 ]
  %16 = icmp slt i32 %15, 64
  br i1 %16, label %17, label %1

17:                                               ; preds = %14
  %18 = add nuw nsw i32 0, %15
  %19 = getelementptr inbounds nuw bfloat, ptr getelementptr inbounds nuw (i8, ptr @__air_herd_arg_1, i64 128), i32 %18
  %20 = load bfloat, ptr %19, align 2
  %21 = load float, ptr @buf3, align 4
  %22 = fdiv float %21, 6.400000e+01
  %23 = fadd float %22, 0x3EE4F8B580000000
  %24 = call float @llvm.aie2p.invsqrt(float %23)
  %25 = fpext bfloat %20 to float
  %26 = fmul float %25, %24
  %27 = fptrunc float %26 to bfloat
  %28 = getelementptr inbounds nuw bfloat, ptr @buf2, i32 %18
  store bfloat %27, ptr %28, align 2
  %29 = add i32 %15, 1
  br label %14, !llvm.loop !1
}

; Unknown intrinsic
declare float @llvm.aie2p.invsqrt(float)

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.mustprogress"}
