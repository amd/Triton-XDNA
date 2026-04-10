; ModuleID = 'air_project/square_kernel_0_core_0_2.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@buf0 = external local_unnamed_addr global [256 x float]
@buf1 = external local_unnamed_addr global [256 x float]

; Function Attrs: nounwind
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: nounwind
declare void @llvm.aie2p.release(i32, i32) #0

; Function Attrs: nounwind memory(inaccessiblemem: write)
declare void @llvm.aie2p.set.ctrl.reg(i32, i32) #1

; Function Attrs: noreturn nounwind
define void @core_0_2() local_unnamed_addr #2 {
  tail call void @llvm.aie2p.set.ctrl.reg(i32 9, i32 1)
  tail call void @llvm.aie2p.set.ctrl.reg(i32 1, i32 0)
  br label %1

1:                                                ; preds = %13, %0
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  br label %2

2:                                                ; preds = %1, %2
  %3 = phi i32 [ 0, %1 ], [ %11, %2 ]
  %4 = trunc nuw i32 %3 to i20
  %5 = getelementptr float, ptr @buf1, i20 %4
  %6 = load <16 x float>, ptr %5, align 64
  %7 = tail call <16 x bfloat> @llvm.aie2p.v16accfloat.to.v16bf16(<16 x float> %6)
  %8 = shufflevector <16 x bfloat> %7, <16 x bfloat> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %9 = tail call <16 x float> @llvm.aie2p.I512.I512.ACC512.bf.mul.conf(<32 x bfloat> %8, <32 x bfloat> %8, i32 60)
  %10 = getelementptr float, ptr @buf0, i20 %4
  store <16 x float> %9, ptr %10, align 64
  %11 = add nuw nsw i32 %3, 16
  %12 = icmp ult i32 %3, 240
  br i1 %12, label %2, label %13, !llvm.loop !1

13:                                               ; preds = %2
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  br label %1
}

; Function Attrs: nofree nounwind memory(inaccessiblemem: read)
declare <16 x bfloat> @llvm.aie2p.v16accfloat.to.v16bf16(<16 x float>) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare <16 x float> @llvm.aie2p.I512.I512.ACC512.bf.mul.conf(<32 x bfloat>, <32 x bfloat>, i32) #4

attributes #0 = { nounwind }
attributes #1 = { nounwind memory(inaccessiblemem: write) }
attributes #2 = { noreturn nounwind }
attributes #3 = { nofree nounwind memory(inaccessiblemem: read) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.mustprogress"}
