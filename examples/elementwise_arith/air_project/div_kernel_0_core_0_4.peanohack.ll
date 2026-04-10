; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@buf0 = external global [256 x float]
@buf1 = external global [256 x float]
@buf2 = external global [256 x float]
@buf3 = external global [256 x float]
@buf4 = external global [256 x float]
@buf5 = external global [256 x float]
@buf6 = external global [256 x float]
@buf7 = external global [256 x float]
@buf8 = external global [256 x float]
@buf9 = external global [256 x float]
@buf10 = external global [256 x float]
@buf11 = external global [256 x float]
@buf12 = external global [1024 x float]
@buf13 = external global [1024 x float]
@buf14 = external global [1024 x float]

; Function Attrs: noinline
define float @__aie2p_scalar_fdiv(float %0, float %1) #0 {
  %3 = call float @llvm.aie2p.inv(float %1)
  %4 = fmul float %0, %3
  ret float %4
}

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

define void @core_0_4() {
  br label %1

1:                                                ; preds = %76, %0
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %2

2:                                                ; preds = %5, %1
  %3 = phi i32 [ %75, %5 ], [ 0, %1 ]
  %4 = icmp slt i32 %3, 256
  br i1 %4, label %5, label %76

5:                                                ; preds = %2
  %6 = getelementptr float, ptr @buf8, i32 %3
  %7 = load <16 x float>, ptr %6
  %8 = getelementptr float, ptr @buf7, i32 %3
  %9 = load <16 x float>, ptr %8
  %10 = extractelement <16 x float> %7, i64 0
  %11 = extractelement <16 x float> %9, i64 0
  %12 = call float @__aie2p_scalar_fdiv(float %10, float %11)
  %13 = insertelement <16 x float> poison, float %12, i64 0
  %14 = extractelement <16 x float> %7, i64 1
  %15 = extractelement <16 x float> %9, i64 1
  %16 = call float @__aie2p_scalar_fdiv(float %14, float %15)
  %17 = insertelement <16 x float> %13, float %16, i64 1
  %18 = extractelement <16 x float> %7, i64 2
  %19 = extractelement <16 x float> %9, i64 2
  %20 = call float @__aie2p_scalar_fdiv(float %18, float %19)
  %21 = insertelement <16 x float> %17, float %20, i64 2
  %22 = extractelement <16 x float> %7, i64 3
  %23 = extractelement <16 x float> %9, i64 3
  %24 = call float @__aie2p_scalar_fdiv(float %22, float %23)
  %25 = insertelement <16 x float> %21, float %24, i64 3
  %26 = extractelement <16 x float> %7, i64 4
  %27 = extractelement <16 x float> %9, i64 4
  %28 = call float @__aie2p_scalar_fdiv(float %26, float %27)
  %29 = insertelement <16 x float> %25, float %28, i64 4
  %30 = extractelement <16 x float> %7, i64 5
  %31 = extractelement <16 x float> %9, i64 5
  %32 = call float @__aie2p_scalar_fdiv(float %30, float %31)
  %33 = insertelement <16 x float> %29, float %32, i64 5
  %34 = extractelement <16 x float> %7, i64 6
  %35 = extractelement <16 x float> %9, i64 6
  %36 = call float @__aie2p_scalar_fdiv(float %34, float %35)
  %37 = insertelement <16 x float> %33, float %36, i64 6
  %38 = extractelement <16 x float> %7, i64 7
  %39 = extractelement <16 x float> %9, i64 7
  %40 = call float @__aie2p_scalar_fdiv(float %38, float %39)
  %41 = insertelement <16 x float> %37, float %40, i64 7
  %42 = extractelement <16 x float> %7, i64 8
  %43 = extractelement <16 x float> %9, i64 8
  %44 = call float @__aie2p_scalar_fdiv(float %42, float %43)
  %45 = insertelement <16 x float> %41, float %44, i64 8
  %46 = extractelement <16 x float> %7, i64 9
  %47 = extractelement <16 x float> %9, i64 9
  %48 = call float @__aie2p_scalar_fdiv(float %46, float %47)
  %49 = insertelement <16 x float> %45, float %48, i64 9
  %50 = extractelement <16 x float> %7, i64 10
  %51 = extractelement <16 x float> %9, i64 10
  %52 = call float @__aie2p_scalar_fdiv(float %50, float %51)
  %53 = insertelement <16 x float> %49, float %52, i64 10
  %54 = extractelement <16 x float> %7, i64 11
  %55 = extractelement <16 x float> %9, i64 11
  %56 = call float @__aie2p_scalar_fdiv(float %54, float %55)
  %57 = insertelement <16 x float> %53, float %56, i64 11
  %58 = extractelement <16 x float> %7, i64 12
  %59 = extractelement <16 x float> %9, i64 12
  %60 = call float @__aie2p_scalar_fdiv(float %58, float %59)
  %61 = insertelement <16 x float> %57, float %60, i64 12
  %62 = extractelement <16 x float> %7, i64 13
  %63 = extractelement <16 x float> %9, i64 13
  %64 = call float @__aie2p_scalar_fdiv(float %62, float %63)
  %65 = insertelement <16 x float> %61, float %64, i64 13
  %66 = extractelement <16 x float> %7, i64 14
  %67 = extractelement <16 x float> %9, i64 14
  %68 = call float @__aie2p_scalar_fdiv(float %66, float %67)
  %69 = insertelement <16 x float> %65, float %68, i64 14
  %70 = extractelement <16 x float> %7, i64 15
  %71 = extractelement <16 x float> %9, i64 15
  %72 = call float @__aie2p_scalar_fdiv(float %70, float %71)
  %73 = insertelement <16 x float> %69, float %72, i64 15
  %74 = getelementptr float, ptr @buf6, i32 %3
  store <16 x float> %73, ptr %74
  %75 = add i32 %3, 16
  br label %2, !llvm.loop !1

76:                                               ; preds = %2
  call void @llvm.aie2p.release(i32 51, i32 1)
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  br label %1
}

; Unknown intrinsic
declare float @llvm.aie2p.inv(float)

attributes #0 = { noinline }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.mustprogress"}
