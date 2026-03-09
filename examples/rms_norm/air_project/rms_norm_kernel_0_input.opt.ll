; ModuleID = '/home/strixminipc/Triton-XDNA/examples/rms_norm/air_project/rms_norm_kernel_0_input.llpeanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@buf0 = external local_unnamed_addr global [1 x [64 x bfloat]]
@buf1 = external local_unnamed_addr global [1 x float]
@buf2 = external local_unnamed_addr global [1 x [64 x bfloat]]
@buf3 = external local_unnamed_addr global [1 x float]
@__air_herd_arg_1 = external local_unnamed_addr global [2 x [64 x bfloat]]
@__air_herd_arg = external local_unnamed_addr global [2 x [64 x bfloat]]

; Function Attrs: nofree noreturn nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none)
define void @core_0_2() local_unnamed_addr #0 {
  br label %.loopexit

.loopexit.loopexit:                               ; preds = %35
  br label %.loopexit, !llvm.loop !1

.loopexit:                                        ; preds = %.loopexit.loopexit, %0
  store float 0.000000e+00, ptr @buf1, align 4
  br label %4

.preheader:                                       ; preds = %4
  store float %33, ptr @buf1, align 4
  %1 = fmul float %33, 1.562500e-02
  %2 = fadd float %1, 0x3EE4F8B580000000
  %3 = tail call float @llvm.aie2p.invsqrt(float %2)
  br label %35

4:                                                ; preds = %4, %.loopexit
  %5 = phi i32 [ 0, %.loopexit ], [ %34, %4 ]
  %6 = phi float [ 0.000000e+00, %.loopexit ], [ %33, %4 ]
  %7 = trunc nuw i32 %5 to i20
  %8 = getelementptr inbounds bfloat, ptr @__air_herd_arg, i20 %7
  %9 = load bfloat, ptr %8, align 2
  %10 = fpext bfloat %9 to float
  %11 = fmul float %10, %10
  %12 = fadd float %6, %11
  %13 = or disjoint i32 %5, 1
  %14 = trunc nuw i32 %13 to i20
  %15 = getelementptr inbounds bfloat, ptr @__air_herd_arg, i20 %14
  %16 = load bfloat, ptr %15, align 2
  %17 = fpext bfloat %16 to float
  %18 = fmul float %17, %17
  %19 = fadd float %12, %18
  %20 = or disjoint i32 %5, 2
  %21 = trunc nuw i32 %20 to i20
  %22 = getelementptr inbounds bfloat, ptr @__air_herd_arg, i20 %21
  %23 = load bfloat, ptr %22, align 2
  %24 = fpext bfloat %23 to float
  %25 = fmul float %24, %24
  %26 = fadd float %19, %25
  %27 = or disjoint i32 %5, 3
  %28 = trunc nuw i32 %27 to i20
  %29 = getelementptr inbounds bfloat, ptr @__air_herd_arg, i20 %28
  %30 = load bfloat, ptr %29, align 2
  %31 = fpext bfloat %30 to float
  %32 = fmul float %31, %31
  %33 = fadd float %26, %32
  %34 = add nuw nsw i32 %5, 4
  %exitcond.not.3 = icmp eq i32 %34, 64
  br i1 %exitcond.not.3, label %.preheader, label %4, !llvm.loop !1

35:                                               ; preds = %35, %.preheader
  %36 = phi i32 [ 0, %.preheader ], [ %68, %35 ]
  %37 = trunc nuw i32 %36 to i20
  %38 = getelementptr inbounds bfloat, ptr @__air_herd_arg, i20 %37
  %39 = load bfloat, ptr %38, align 2
  %40 = fpext bfloat %39 to float
  %41 = fmul float %3, %40
  %42 = fptrunc float %41 to bfloat
  %43 = getelementptr inbounds bfloat, ptr @buf0, i20 %37
  store bfloat %42, ptr %43, align 2
  %44 = or disjoint i32 %36, 1
  %45 = trunc nuw i32 %44 to i20
  %46 = getelementptr inbounds bfloat, ptr @__air_herd_arg, i20 %45
  %47 = load bfloat, ptr %46, align 2
  %48 = fpext bfloat %47 to float
  %49 = fmul float %3, %48
  %50 = fptrunc float %49 to bfloat
  %51 = getelementptr inbounds bfloat, ptr @buf0, i20 %45
  store bfloat %50, ptr %51, align 2
  %52 = or disjoint i32 %36, 2
  %53 = trunc nuw i32 %52 to i20
  %54 = getelementptr inbounds bfloat, ptr @__air_herd_arg, i20 %53
  %55 = load bfloat, ptr %54, align 2
  %56 = fpext bfloat %55 to float
  %57 = fmul float %3, %56
  %58 = fptrunc float %57 to bfloat
  %59 = getelementptr inbounds bfloat, ptr @buf0, i20 %53
  store bfloat %58, ptr %59, align 2
  %60 = or disjoint i32 %36, 3
  %61 = trunc nuw i32 %60 to i20
  %62 = getelementptr inbounds bfloat, ptr @__air_herd_arg, i20 %61
  %63 = load bfloat, ptr %62, align 2
  %64 = fpext bfloat %63 to float
  %65 = fmul float %3, %64
  %66 = fptrunc float %65 to bfloat
  %67 = getelementptr inbounds bfloat, ptr @buf0, i20 %61
  store bfloat %66, ptr %67, align 2
  %68 = add nuw nsw i32 %36, 4
  %exitcond3.not.3 = icmp eq i32 %68, 64
  br i1 %exitcond3.not.3, label %.loopexit.loopexit, label %35, !llvm.loop !1
}

; Function Attrs: nofree noreturn nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none)
define void @core_0_3() local_unnamed_addr #0 {
  br label %.loopexit

.loopexit.loopexit:                               ; preds = %35
  br label %.loopexit, !llvm.loop !1

.loopexit:                                        ; preds = %.loopexit.loopexit, %0
  store float 0.000000e+00, ptr @buf3, align 4
  br label %4

.preheader:                                       ; preds = %4
  store float %33, ptr @buf3, align 4
  %1 = fmul float %33, 1.562500e-02
  %2 = fadd float %1, 0x3EE4F8B580000000
  %3 = tail call float @llvm.aie2p.invsqrt(float %2)
  br label %35

4:                                                ; preds = %4, %.loopexit
  %5 = phi i32 [ 0, %.loopexit ], [ %34, %4 ]
  %6 = phi float [ 0.000000e+00, %.loopexit ], [ %33, %4 ]
  %7 = trunc nuw i32 %5 to i20
  %8 = getelementptr inbounds [2 x [64 x bfloat]], ptr @__air_herd_arg_1, i20 0, i20 1, i20 %7
  %9 = load bfloat, ptr %8, align 2
  %10 = fpext bfloat %9 to float
  %11 = fmul float %10, %10
  %12 = fadd float %6, %11
  %13 = or disjoint i32 %5, 1
  %14 = trunc nuw i32 %13 to i20
  %15 = getelementptr inbounds [2 x [64 x bfloat]], ptr @__air_herd_arg_1, i20 0, i20 1, i20 %14
  %16 = load bfloat, ptr %15, align 2
  %17 = fpext bfloat %16 to float
  %18 = fmul float %17, %17
  %19 = fadd float %12, %18
  %20 = or disjoint i32 %5, 2
  %21 = trunc nuw i32 %20 to i20
  %22 = getelementptr inbounds [2 x [64 x bfloat]], ptr @__air_herd_arg_1, i20 0, i20 1, i20 %21
  %23 = load bfloat, ptr %22, align 2
  %24 = fpext bfloat %23 to float
  %25 = fmul float %24, %24
  %26 = fadd float %19, %25
  %27 = or disjoint i32 %5, 3
  %28 = trunc nuw i32 %27 to i20
  %29 = getelementptr inbounds [2 x [64 x bfloat]], ptr @__air_herd_arg_1, i20 0, i20 1, i20 %28
  %30 = load bfloat, ptr %29, align 2
  %31 = fpext bfloat %30 to float
  %32 = fmul float %31, %31
  %33 = fadd float %26, %32
  %34 = add nuw nsw i32 %5, 4
  %exitcond.not.3 = icmp eq i32 %34, 64
  br i1 %exitcond.not.3, label %.preheader, label %4, !llvm.loop !1

35:                                               ; preds = %35, %.preheader
  %36 = phi i32 [ 0, %.preheader ], [ %68, %35 ]
  %37 = trunc nuw i32 %36 to i20
  %38 = getelementptr inbounds [2 x [64 x bfloat]], ptr @__air_herd_arg_1, i20 0, i20 1, i20 %37
  %39 = load bfloat, ptr %38, align 2
  %40 = fpext bfloat %39 to float
  %41 = fmul float %3, %40
  %42 = fptrunc float %41 to bfloat
  %43 = getelementptr inbounds bfloat, ptr @buf2, i20 %37
  store bfloat %42, ptr %43, align 2
  %44 = or disjoint i32 %36, 1
  %45 = trunc nuw i32 %44 to i20
  %46 = getelementptr inbounds [2 x [64 x bfloat]], ptr @__air_herd_arg_1, i20 0, i20 1, i20 %45
  %47 = load bfloat, ptr %46, align 2
  %48 = fpext bfloat %47 to float
  %49 = fmul float %3, %48
  %50 = fptrunc float %49 to bfloat
  %51 = getelementptr inbounds bfloat, ptr @buf2, i20 %45
  store bfloat %50, ptr %51, align 2
  %52 = or disjoint i32 %36, 2
  %53 = trunc nuw i32 %52 to i20
  %54 = getelementptr inbounds [2 x [64 x bfloat]], ptr @__air_herd_arg_1, i20 0, i20 1, i20 %53
  %55 = load bfloat, ptr %54, align 2
  %56 = fpext bfloat %55 to float
  %57 = fmul float %3, %56
  %58 = fptrunc float %57 to bfloat
  %59 = getelementptr inbounds bfloat, ptr @buf2, i20 %53
  store bfloat %58, ptr %59, align 2
  %60 = or disjoint i32 %36, 3
  %61 = trunc nuw i32 %60 to i20
  %62 = getelementptr inbounds [2 x [64 x bfloat]], ptr @__air_herd_arg_1, i20 0, i20 1, i20 %61
  %63 = load bfloat, ptr %62, align 2
  %64 = fpext bfloat %63 to float
  %65 = fmul float %3, %64
  %66 = fptrunc float %65 to bfloat
  %67 = getelementptr inbounds bfloat, ptr @buf2, i20 %61
  store bfloat %66, ptr %67, align 2
  %68 = add nuw nsw i32 %36, 4
  %exitcond3.not.3 = icmp eq i32 %68, 64
  br i1 %exitcond3.not.3, label %.loopexit.loopexit, label %35, !llvm.loop !1
}

; Function Attrs: nofree nosync nounwind memory(none)
declare float @llvm.aie2p.invsqrt(float) #1

attributes #0 = { nofree noreturn nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) }
attributes #1 = { nofree nosync nounwind memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.mustprogress"}
