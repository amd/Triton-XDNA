module {
  air.channel @channel_0 []
  air.channel @channel_1 [4, 1]
  air.channel @channel_2 [4, 1]
  air.channel @channel_3 []
  func.func @square_kernel(%arg0: memref<*xi16> {tt.divisibility = 16 : i32}, %arg1: memref<*xi16> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg8, %arg9, %arg10) in (%arg11=%c1, %arg12=%c1, %arg13=%c1) args(%arg14=%arg0, %arg15=%arg1) : memref<*xi16>, memref<*xi16> attributes {id = 1 : i32} {
      %c1024 = arith.constant 1024 : index
      %c1_0 = arith.constant 1 : index
      %1 = arith.muli %arg8, %c1024 : index
      %2 = air.channel.put async  @channel_0[] (%arg14[%1] [%c1024] [%c1_0]) {id = 1 : i32} : (memref<*xi16>)
      %3 = air.channel.get async  @channel_3[] (%arg15[%1] [%c1024] [%c1_0]) {id = 2 : i32} : (memref<*xi16>)
      %4 = air.segment @square_kernel_0 async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c4 = arith.constant 4 : index
        %c768 = arith.constant 768 : index
        %c3 = arith.constant 3 : index
        %c512 = arith.constant 512 : index
        %c2 = arith.constant 2 : index
        %c256 = arith.constant 256 : index
        %c0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<1024xi16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1024xi16, 1 : i32>
          air.execute_terminator %alloc : memref<1024xi16, 1 : i32>
        }
        %5 = air.channel.get async [%async_token]  @channel_0[] (%results[] [] []) {id = 3 : i32} : (memref<1024xi16, 1 : i32>)
        %async_token_2, %results_3 = air.execute -> (memref<1024xi16, 1>) {
          %alloc = memref.alloc() : memref<1024xi16, 1>
          air.execute_terminator %alloc : memref<1024xi16, 1>
        }
        %6 = air.channel.put async [%5]  @channel_1[%c0, %c0] (%results[%c0] [%c256] [%c1_1]) {id = 4 : i32} : (memref<1024xi16, 1 : i32>)
        %7 = air.channel.put async [%5]  @channel_1[%c1_1, %c0] (%results[%c256] [%c256] [%c1_1]) {id = 5 : i32} : (memref<1024xi16, 1 : i32>)
        %8 = air.channel.put async [%5]  @channel_1[%c2, %c0] (%results[%c512] [%c256] [%c1_1]) {id = 6 : i32} : (memref<1024xi16, 1 : i32>)
        %9 = air.channel.put async [%5]  @channel_1[%c3, %c0] (%results[%c768] [%c256] [%c1_1]) {id = 7 : i32} : (memref<1024xi16, 1 : i32>)
        %10 = air.channel.get async [%async_token_2]  @channel_2[%c0, %c0] (%results_3[%c0] [%c256] [%c1_1]) {id = 8 : i32} : (memref<1024xi16, 1>)
        %11 = air.channel.get async [%async_token_2]  @channel_2[%c1_1, %c0] (%results_3[%c256] [%c256] [%c1_1]) {id = 9 : i32} : (memref<1024xi16, 1>)
        %12 = air.channel.get async [%async_token_2]  @channel_2[%c2, %c0] (%results_3[%c512] [%c256] [%c1_1]) {id = 10 : i32} : (memref<1024xi16, 1>)
        %13 = air.channel.get async [%async_token_2]  @channel_2[%c3, %c0] (%results_3[%c768] [%c256] [%c1_1]) {id = 11 : i32} : (memref<1024xi16, 1>)
        %14 = air.herd @herd_0 async [%5, %async_token_2]  tile (%arg16, %arg17) in (%arg18=%c1_1, %arg19=%c4) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %c32 = arith.constant 32 : index
          %c256_5 = arith.constant 256 : index
          %c0_6 = arith.constant 0 : index
          %16 = ub.poison : i16
          %async_token_7, %results_8 = air.execute -> (memref<256xi16, 2>) {
            %alloc = memref.alloc() : memref<256xi16, 2>
            air.execute_terminator %alloc : memref<256xi16, 2>
          }
          %17 = air.channel.get async [%async_token_7]  @channel_1[%arg17, %c0_6] (%results_8[] [] []) {id = 12 : i32} : (memref<256xi16, 2>)
          %async_token_9, %results_10 = air.execute -> (memref<256xi16, 2>) {
            %alloc = memref.alloc() : memref<256xi16, 2>
            air.execute_terminator %alloc : memref<256xi16, 2>
          }
          %18 = air.wait_all async [%17, %async_token_9] 
          %19 = scf.for %arg20 = %c0_6 to %c256_5 step %c32 iter_args(%arg21 = %18) -> (!air.async.token) {
            %subview = memref.subview %results_8[%arg20] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
            %subview_13 = memref.subview %results_10[%arg20] [32] [1] : memref<256xi16, 2> to memref<32xi16, strided<[1], offset: ?>, 2>
            %async_token_14, %results_15 = air.execute [%arg21] -> (vector<32xi16>) {
              %23 = vector.transfer_read %subview[%c0_6], %16 {in_bounds = [true]} : memref<32xi16, strided<[1], offset: ?>, 2>, vector<32xi16>
              air.execute_terminator %23 : vector<32xi16>
            }
            %21 = arith.muli %results_15, %results_15 : vector<32xi16>
            %async_token_16 = air.execute [%arg21] {
              vector.transfer_write %21, %subview_13[%c0_6] {in_bounds = [true]} : vector<32xi16>, memref<32xi16, strided<[1], offset: ?>, 2>
            }
            %22 = air.wait_all async [%async_token_14, %async_token_16] 
            scf.yield %22 : !air.async.token
          }
          %20 = air.channel.put async [%async_token_9]  @channel_2[%arg17, %c0_6] (%results_10[] [] []) {id = 13 : i32} : (memref<256xi16, 2>)
          %async_token_11 = air.execute [%17] {
            memref.dealloc %results_8 : memref<256xi16, 2>
          }
          %async_token_12 = air.execute [%20] {
            memref.dealloc %results_10 : memref<256xi16, 2>
          }
        }
        %15 = air.channel.put async [%14]  @channel_3[] (%results_3[] [] []) {id = 14 : i32} : (memref<1024xi16, 1>)
        %async_token_4 = air.execute [%15] {
          memref.dealloc %results_3 : memref<1024xi16, 1>
        }
        air.wait_all [%6, %7, %8, %9, %10, %11, %12, %13, %async_token_4]  {air.segment_end}
      }
    }
    return
  }
}
