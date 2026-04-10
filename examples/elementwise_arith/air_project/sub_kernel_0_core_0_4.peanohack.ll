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

1:                                                ; preds = %22, %0
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %2

2:                                                ; preds = %5, %1
  %3 = phi i32 [ %21, %5 ], [ 0, %1 ]
  %4 = icmp slt i32 %3, 256
  br i1 %4, label %5, label %22

5:                                                ; preds = %2
  %6 = getelementptr float, ptr @buf8, i32 %3
  %7 = load <16 x float>, ptr %6
  %8 = getelementptr float, ptr @buf7, i32 %3
  %9 = load <16 x float>, ptr %8
  %10 = bitcast <16 x float> %7 to <8 x i64>
  %11 = bitcast <16 x float> %9 to <8 x i64>
  %12 = shufflevector <8 x i64> %10, <8 x i64> %10, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %13 = shufflevector <8 x i64> %11, <8 x i64> %11, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %14 = bitcast <32 x i64> %12 to <64 x float>
  %15 = bitcast <32 x i64> %13 to <64 x float>
  %16 = call <64 x float> @llvm.aie2p.ACC2048.accfloat.sub.conf(<64 x float> %14, <64 x float> %15, i32 60)
  %17 = bitcast <64 x float> %16 to <32 x i64>
  %18 = shufflevector <32 x i64> %17, <32 x i64> %17, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %19 = bitcast <8 x i64> %18 to <16 x float>
  %20 = getelementptr float, ptr @buf6, i32 %3
  store <16 x float> %19, ptr %20
  %21 = add i32 %3, 16
  br label %2, !llvm.loop !1

22:                                               ; preds = %2
  call void @llvm.aie2p.release(i32 51, i32 1)
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  br label %1
}

; Unknown intrinsic
declare <64 x float> @llvm.aie2p.ACC2048.accfloat.sub.conf(<64 x float>, <64 x float>, i32)

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.mustprogress"}
