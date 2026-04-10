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
@buf8 = external global [1024 x float]
@buf9 = external global [1024 x float]

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

define void @core_0_5() {
  call void @llvm.aie2p.set.ctrl.reg(i32 9, i32 1)
  call void @llvm.aie2p.set.ctrl.reg(i32 1, i32 0)
  br label %1

1:                                                ; preds = %13, %0
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  br label %2

2:                                                ; preds = %5, %1
  %3 = phi i32 [ %12, %5 ], [ 0, %1 ]
  %4 = icmp slt i32 %3, 256
  br i1 %4, label %5, label %13

5:                                                ; preds = %2
  %6 = getelementptr float, ptr @buf7, i32 %3
  %7 = load <16 x float>, ptr %6
  %8 = call <16 x bfloat> @llvm.aie2p.v16accfloat.to.v16bf16(<16 x float> %7)
  %9 = shufflevector <16 x bfloat> %8, <16 x bfloat> %8, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %10 = call <16 x float> @llvm.aie2p.I512.I512.ACC512.bf.mul.conf(<32 x bfloat> %9, <32 x bfloat> %9, i32 60)
  %11 = getelementptr float, ptr @buf6, i32 %3
  store <16 x float> %10, ptr %11
  %12 = add i32 %3, 16
  br label %2, !llvm.loop !1

13:                                               ; preds = %2
  call void @llvm.aie2p.release(i32 51, i32 1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  br label %1
}

; Unknown intrinsic
declare <16 x bfloat> @llvm.aie2p.v16accfloat.to.v16bf16(<16 x float>)

; Unknown intrinsic
declare <16 x float> @llvm.aie2p.I512.I512.ACC512.bf.mul.conf(<32 x bfloat>, <32 x bfloat>, i32)

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.mustprogress"}
