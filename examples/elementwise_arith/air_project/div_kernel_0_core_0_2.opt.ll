; ModuleID = 'air_project/div_kernel_0_core_0_2.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@buf0 = external local_unnamed_addr global [256 x float]
@buf1 = external local_unnamed_addr global [256 x float]
@buf2 = external local_unnamed_addr global [256 x float]

; Function Attrs: nofree noinline nosync nounwind memory(none)
define float @__aie2p_scalar_fdiv(float %0, float %1) local_unnamed_addr #0 {
  %3 = tail call float @llvm.aie2p.inv(float %1)
  %4 = fmul float %3, %0
  ret float %4
}

; Function Attrs: nounwind
declare void @llvm.aie2p.acquire(i32, i32) #1

; Function Attrs: nounwind
declare void @llvm.aie2p.release(i32, i32) #1

; Function Attrs: noreturn nounwind
define void @core_0_2() local_unnamed_addr #2 {
  br label %1

1:                                                ; preds = %76, %0
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %2

2:                                                ; preds = %1, %2
  %3 = phi i32 [ 0, %1 ], [ %74, %2 ]
  %4 = trunc nuw i32 %3 to i20
  %5 = getelementptr float, ptr @buf2, i20 %4
  %6 = load <16 x float>, ptr %5, align 64
  %7 = getelementptr float, ptr @buf1, i20 %4
  %8 = load <16 x float>, ptr %7, align 64
  %9 = extractelement <16 x float> %6, i64 0
  %10 = extractelement <16 x float> %8, i64 0
  %11 = tail call float @__aie2p_scalar_fdiv(float %9, float %10)
  %12 = insertelement <16 x float> poison, float %11, i64 0
  %13 = extractelement <16 x float> %6, i64 1
  %14 = extractelement <16 x float> %8, i64 1
  %15 = tail call float @__aie2p_scalar_fdiv(float %13, float %14)
  %16 = insertelement <16 x float> %12, float %15, i64 1
  %17 = extractelement <16 x float> %6, i64 2
  %18 = extractelement <16 x float> %8, i64 2
  %19 = tail call float @__aie2p_scalar_fdiv(float %17, float %18)
  %20 = insertelement <16 x float> %16, float %19, i64 2
  %21 = extractelement <16 x float> %6, i64 3
  %22 = extractelement <16 x float> %8, i64 3
  %23 = tail call float @__aie2p_scalar_fdiv(float %21, float %22)
  %24 = insertelement <16 x float> %20, float %23, i64 3
  %25 = extractelement <16 x float> %6, i64 4
  %26 = extractelement <16 x float> %8, i64 4
  %27 = tail call float @__aie2p_scalar_fdiv(float %25, float %26)
  %28 = insertelement <16 x float> %24, float %27, i64 4
  %29 = extractelement <16 x float> %6, i64 5
  %30 = extractelement <16 x float> %8, i64 5
  %31 = tail call float @__aie2p_scalar_fdiv(float %29, float %30)
  %32 = insertelement <16 x float> %28, float %31, i64 5
  %33 = extractelement <16 x float> %6, i64 6
  %34 = extractelement <16 x float> %8, i64 6
  %35 = tail call float @__aie2p_scalar_fdiv(float %33, float %34)
  %36 = insertelement <16 x float> %32, float %35, i64 6
  %37 = extractelement <16 x float> %6, i64 7
  %38 = extractelement <16 x float> %8, i64 7
  %39 = tail call float @__aie2p_scalar_fdiv(float %37, float %38)
  %40 = insertelement <16 x float> %36, float %39, i64 7
  %41 = extractelement <16 x float> %6, i64 8
  %42 = extractelement <16 x float> %8, i64 8
  %43 = tail call float @__aie2p_scalar_fdiv(float %41, float %42)
  %44 = insertelement <16 x float> %40, float %43, i64 8
  %45 = extractelement <16 x float> %6, i64 9
  %46 = extractelement <16 x float> %8, i64 9
  %47 = tail call float @__aie2p_scalar_fdiv(float %45, float %46)
  %48 = insertelement <16 x float> %44, float %47, i64 9
  %49 = extractelement <16 x float> %6, i64 10
  %50 = extractelement <16 x float> %8, i64 10
  %51 = tail call float @__aie2p_scalar_fdiv(float %49, float %50)
  %52 = insertelement <16 x float> %48, float %51, i64 10
  %53 = extractelement <16 x float> %6, i64 11
  %54 = extractelement <16 x float> %8, i64 11
  %55 = tail call float @__aie2p_scalar_fdiv(float %53, float %54)
  %56 = insertelement <16 x float> %52, float %55, i64 11
  %57 = extractelement <16 x float> %6, i64 12
  %58 = extractelement <16 x float> %8, i64 12
  %59 = tail call float @__aie2p_scalar_fdiv(float %57, float %58)
  %60 = insertelement <16 x float> %56, float %59, i64 12
  %61 = extractelement <16 x float> %6, i64 13
  %62 = extractelement <16 x float> %8, i64 13
  %63 = tail call float @__aie2p_scalar_fdiv(float %61, float %62)
  %64 = insertelement <16 x float> %60, float %63, i64 13
  %65 = extractelement <16 x float> %6, i64 14
  %66 = extractelement <16 x float> %8, i64 14
  %67 = tail call float @__aie2p_scalar_fdiv(float %65, float %66)
  %68 = insertelement <16 x float> %64, float %67, i64 14
  %69 = extractelement <16 x float> %6, i64 15
  %70 = extractelement <16 x float> %8, i64 15
  %71 = tail call float @__aie2p_scalar_fdiv(float %69, float %70)
  %72 = insertelement <16 x float> %68, float %71, i64 15
  %73 = getelementptr float, ptr @buf0, i20 %4
  store <16 x float> %72, ptr %73, align 64
  %74 = add nuw nsw i32 %3, 16
  %75 = icmp ult i32 %3, 240
  br i1 %75, label %2, label %76, !llvm.loop !1

76:                                               ; preds = %2
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  br label %1
}

; Function Attrs: nofree nosync nounwind memory(none)
declare float @llvm.aie2p.inv(float) #3

attributes #0 = { nofree noinline nosync nounwind memory(none) }
attributes #1 = { nounwind }
attributes #2 = { noreturn nounwind }
attributes #3 = { nofree nosync nounwind memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.mustprogress"}
