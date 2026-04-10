; ModuleID = 'air_project/sub_kernel_0_core_0_4.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@buf6 = external local_unnamed_addr global [256 x float]
@buf7 = external local_unnamed_addr global [256 x float]
@buf8 = external local_unnamed_addr global [256 x float]

; Function Attrs: nounwind
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: nounwind
declare void @llvm.aie2p.release(i32, i32) #0

; Function Attrs: noreturn nounwind
define void @core_0_4() local_unnamed_addr #1 {
  br label %1

1:                                                ; preds = %19, %0
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %2

2:                                                ; preds = %1, %2
  %3 = phi i32 [ 0, %1 ], [ %17, %2 ]
  %4 = trunc nuw i32 %3 to i20
  %5 = getelementptr float, ptr @buf8, i20 %4
  %6 = load <8 x i64>, ptr %5, align 64
  %7 = getelementptr float, ptr @buf7, i20 %4
  %8 = load <8 x i64>, ptr %7, align 64
  %9 = shufflevector <8 x i64> %6, <8 x i64> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %10 = shufflevector <8 x i64> %8, <8 x i64> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %11 = bitcast <32 x i64> %9 to <64 x float>
  %12 = bitcast <32 x i64> %10 to <64 x float>
  %13 = tail call <64 x float> @llvm.aie2p.ACC2048.accfloat.sub.conf(<64 x float> %11, <64 x float> %12, i32 60)
  %14 = bitcast <64 x float> %13 to <32 x i64>
  %15 = shufflevector <32 x i64> %14, <32 x i64> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = getelementptr float, ptr @buf6, i20 %4
  store <8 x i64> %15, ptr %16, align 64
  %17 = add nuw nsw i32 %3, 16
  %18 = icmp ult i32 %3, 240
  br i1 %18, label %2, label %19, !llvm.loop !1

19:                                               ; preds = %2
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  br label %1
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare <64 x float> @llvm.aie2p.ACC2048.accfloat.sub.conf(<64 x float>, <64 x float>, i32) #2

attributes #0 = { nounwind }
attributes #1 = { noreturn nounwind }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.mustprogress"}
