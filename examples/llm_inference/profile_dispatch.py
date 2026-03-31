"""
Profile where time is spent during LLM decode with NPU offloading.
Measures CPU matmul, NPU dispatch, attention, and overhead separately.
"""
import os, sys, time, types
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
import benchmark

# Check torch threading
print(f"torch.get_num_threads() = {torch.get_num_threads()}")
print(f"torch.get_num_interop_threads() = {torch.get_num_interop_threads()}")
print(f"torch.__config__.parallel_info():")
print(torch.__config__.parallel_info())

# ---- Triton kernel (same as llm_real.py) ----
import triton
import triton.language as tl

@triton.jit
def swiglu_kernel(GATE, UP, OUT, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    gate = tl.load(GATE + offsets[:])
    up = tl.load(UP + offsets[:])
    gate_f32 = gate.to(tl.float32)
    silu_gate = (gate_f32 * tl.sigmoid(gate_f32)).to(gate.dtype)
    tl.store(OUT + offsets[:], silu_gate * up)

def set_transform(name):
    os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = os.path.join(SCRIPT_DIR, name)

def npu_swiglu(gate_flat, up_flat):
    set_transform("transform_binary_aie2p.mlir")
    out = torch.empty_like(gate_flat)
    N = gate_flat.numel()
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    swiglu_kernel[grid](gate_flat, up_flat, out, N, BLOCK_SIZE=1024)
    return out

# ---- Profiling data ----
profile_data = {
    "cpu_proj_ns": [],          # gate_proj + up_proj time
    "npu_swiglu_ns": [],        # NPU dispatch time
    "cpu_downproj_ns": [],      # down_proj time
    "contiguous_view_ns": [],   # .contiguous().view(-1) overhead
}

def patched_mlp_forward_profiled(self, hidden_states):
    t0 = time.perf_counter_ns()
    gate = self.gate_proj(hidden_states)
    up = self.up_proj(hidden_states)
    t1 = time.perf_counter_ns()
    
    shape = gate.shape
    gate_flat = gate.contiguous().view(-1)
    up_flat = up.contiguous().view(-1)
    t2 = time.perf_counter_ns()
    
    activated = npu_swiglu(gate_flat, up_flat).view(shape)
    t3 = time.perf_counter_ns()
    
    result = self.down_proj(activated)
    t4 = time.perf_counter_ns()
    
    profile_data["cpu_proj_ns"].append(t1 - t0)
    profile_data["contiguous_view_ns"].append(t2 - t1)
    profile_data["npu_swiglu_ns"].append(t3 - t2)
    profile_data["cpu_downproj_ns"].append(t4 - t3)
    return result

# ---- Main ----
if __name__ == "__main__":
    MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model.eval()
    print(f"  hidden={model.config.hidden_size}, intermediate={model.config.intermediate_size}, layers={model.config.num_hidden_layers}")

    # Activate NPU backend
    benchmark.select_npu_backend()
    for layer in model.model.layers:
        layer.mlp.forward = types.MethodType(patched_mlp_forward_profiled, layer.mlp)

    # Warm up (JIT compile)
    print("\nWarming up (JIT compile)...")
    warmup_ids = tokenizer("Hello", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        _ = model.generate(warmup_ids, max_new_tokens=2, do_sample=False)
    
    # Clear profile data from warmup
    for k in profile_data:
        profile_data[k].clear()

    # Generate tokens with profiling
    print("\nProfiling 5 decode tokens...")
    messages = [{"role": "user", "content": "What is an NPU?"}]
    chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_input, return_tensors="pt")
    generated_ids = inputs["input_ids"].clone()
    past_key_values = None
    
    token_times = []
    with torch.no_grad():
        for i in range(6):  # 1 prefill + 5 decode
            t_tok = time.perf_counter()
            if past_key_values is None:
                outputs = model(input_ids=generated_ids, use_cache=True)
            else:
                outputs = model(input_ids=generated_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            dt = time.perf_counter() - t_tok
            token_times.append(dt)
            tok_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(f"  Token {i}: \"{tok_text}\"  ({dt:.3f}s)")

    # Report profiling results
    print(f"\n{'='*70}")
    print("  PROFILING RESULTS (decode tokens only, per-layer averages)")
    print(f"{'='*70}")
    
    n_layers = model.config.num_hidden_layers
    # Only look at decode tokens (skip prefill data)
    # First n_layers entries are from warmup prefill in profiling,
    # but we cleared data after warmup. So data starts from the
    # actual generation: first n_layers × seqlen entries are prefill,
    # then each subsequent n_layers entries are one decode token.
    
    # Prefill is token 0, decode is tokens 1-5
    # For token 0 (prefill), we process the whole prompt sequence
    # For tokens 1-5, we process 1 token each = n_layers MLP calls each
    
    # Total data points: token0(prefill) has n_layers calls, tokens 1-5 have n_layers each
    # Decode data starts at index n_layers (after prefill)
    total_calls = len(profile_data["cpu_proj_ns"])
    print(f"  Total MLP calls recorded: {total_calls}")
    print(f"  Expected: {n_layers} × {len(token_times)} tokens = {n_layers * len(token_times)}")
    
    # Take only decode data (skip first token's n_layers entries)
    if total_calls >= n_layers * 2:
        decode_start = n_layers  # skip prefill
        
        cpu_proj = profile_data["cpu_proj_ns"][decode_start:]
        cont_view = profile_data["contiguous_view_ns"][decode_start:]
        npu_swiglu = profile_data["npu_swiglu_ns"][decode_start:]
        cpu_down = profile_data["cpu_downproj_ns"][decode_start:]
        
        avg_proj = sum(cpu_proj) / len(cpu_proj) / 1e6  # ms
        avg_view = sum(cont_view) / len(cont_view) / 1e6
        avg_swiglu = sum(npu_swiglu) / len(npu_swiglu) / 1e6
        avg_down = sum(cpu_down) / len(cpu_down) / 1e6
        total_per_layer = avg_proj + avg_view + avg_swiglu + avg_down
        
        print(f"\n  Per MLP layer (avg across {len(cpu_proj)} calls):")
        print(f"    CPU gate+up proj:     {avg_proj:8.2f} ms  ({avg_proj/total_per_layer*100:5.1f}%)")
        print(f"    Contiguous+view:      {avg_view:8.2f} ms  ({avg_view/total_per_layer*100:5.1f}%)")
        print(f"    NPU SwiGLU dispatch:  {avg_swiglu:8.2f} ms  ({avg_swiglu/total_per_layer*100:5.1f}%)")
        print(f"    CPU down_proj:        {avg_down:8.2f} ms  ({avg_down/total_per_layer*100:5.1f}%)")
        print(f"    Total per layer:      {total_per_layer:8.2f} ms")
        
        # Per-token totals
        decode_times = token_times[1:]  # skip prefill
        avg_token = sum(decode_times) / len(decode_times)
        mlp_per_token = total_per_layer * n_layers
        non_mlp_per_token = (avg_token * 1000) - mlp_per_token
        
        print(f"\n  Per decode token (avg):")
        print(f"    Total MLP time:       {mlp_per_token:8.1f} ms  ({mlp_per_token/(avg_token*1000)*100:5.1f}%)")
        print(f"    Non-MLP (attn+other): {non_mlp_per_token:8.1f} ms  ({non_mlp_per_token/(avg_token*1000)*100:5.1f}%)")
        print(f"    Total per token:      {avg_token*1000:8.1f} ms")
        
        # Breakdown of NPU dispatch only
        npu_total = sum(npu_swiglu) / 1e6  # ms
        n_decode_tokens = len(decode_times)
        print(f"\n  NPU dispatch summary:")
        print(f"    Total NPU time:       {npu_total:8.1f} ms across {n_decode_tokens} tokens")
        print(f"    NPU time per token:   {npu_total/n_decode_tokens:8.1f} ms")
        print(f"    NPU time per dispatch:{avg_swiglu:8.2f} ms")
    else:
        print(f"  Not enough data to analyze (got {total_calls}, need >= {n_layers * 2})")
