# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""End to end: a Triton prefill feeding mlir-air's Q4NX fused decode.

    python llama32_1b_q4nx_inference.py --backend npu --max-tokens 20
    python llama32_1b_q4nx_inference.py --prefill-only     # no decode build needed

The prefill is this repo's -- Triton kernels on XDNA (prefill.py, kernels.py).
The decode is mlir-air's fused Q4NX decode, run unmodified: this script writes
the KV handoff npz that its `generate()` already loads, and neutralizes only
the step that would have produced it.

Prerequisite for generation (not for --prefill-only): the decode templates
must be built once.

    cd mlir-air-local/programming_examples/llms/llama32_1b_q4nx
    make compile-decode
"""

import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402


def _air_inference_module():
    """mlir-air's llama32_1b_q4nx_inference, importable and unmodified."""
    import importlib.util

    config._add_air_paths()
    path = os.path.join(
        str(config._air_llms_root()),
        "llama32_1b_q4nx",
        "llama32_1b_q4nx_inference.py",
    )
    spec = importlib.util.spec_from_file_location("q4nx_inference", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["q4nx_inference"] = mod
    spec.loader.exec_module(mod)
    return mod


def _tokenizer_dir(air):
    """The tokenizer mlir-air's driver uses, or the Hub checkpoint."""
    if air is not None:
        return air._TOKENIZER
    return os.environ.get(
        "Q4NX_TOKENIZER_DIR",
        os.path.expanduser("~/q4nx_data/tokenizer/Llama-3.2-1B"),
    )


class _TimedOps:
    """Wrap an ops backend so each operator's wall time is accumulated."""

    def __init__(self, inner):
        self._inner, self.t = inner, {}
        self.enabled = getattr(inner, "enabled", "-")

    def __getattr__(self, name):
        fn = getattr(self._inner, name)
        if not callable(fn):
            return fn

        def timed(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0

        return timed

    def report(self, total):
        for k, v in sorted(self.t.items(), key=lambda kv: -kv[1]):
            print(f"[profile] {k:10s} {v * 1000:8.1f} ms  {100 * v / total:5.1f}%")
        print(f"[profile] {'TOTAL':10s} {total * 1000:8.1f} ms")


def run_prefill(ids, backend, ops, max_seq, model, kv_path, profile=False):
    """Our prefill -> the handoff npz. Returns (first_token, P)."""
    import torch

    from model import LlamaPrefill
    from prefill import build_ops

    m = LlamaPrefill(
        ops=build_ops(backend, ops),
        n_layers=config.N_LAYERS,
        max_seq=max_seq,
        model=model,
    )
    if profile:
        m.ops = _TimedOps(m.ops)
    t0 = time.time()
    m.load_weights()
    t_load = time.time() - t0
    if profile:  # one warm pass so timings exclude compilation
        m.prefill(ids)
        m.clear_context()
        m.ops.t.clear()
    t0 = time.time()
    logits = m.prefill(ids)
    t_run = time.time() - t0
    if profile:
        m.ops.report(t_run)
    first = int(torch.argmax(logits))
    if kv_path is not None:
        m.save_kv_npz(kv_path, first, ids)
    print(
        f"[triton-prefill] backend={backend} ops={getattr(m.ops, 'enabled', '-')} "
        f"P={len(ids)} load {t_load:.1f}s prefill {t_run:.2f}s first={first}",
        flush=True,
    )
    return first, len(ids)


def make_session_class(air, backend, ops, model, n_layers):
    """mlir-air's Session with OUR prefill in place of theirs.

    `Session.run_turn` only ever calls `prefill`, `kv_view` and
    `clear_context` on its prefiller, and LlamaPrefill provides all three with
    the same semantics, so nothing below run_turn needs to change. The decode
    stays exactly as mlir-air builds it.
    """
    from model import LlamaPrefill
    from prefill import build_ops

    class TritonSession(air.Session):
        def __init__(self, seq_len=2048):
            import time

            import numpy as np

            self.np = np
            self.seq_len = seq_len
            t0 = time.perf_counter()
            print("[session] loading prefill weights (once)...", flush=True)
            self.prefiller = LlamaPrefill(
                ops=build_ops(backend, ops),
                n_layers=n_layers,
                max_seq=seq_len,
                model=model,
            )
            self.prefiller.load_weights()
            print(
                f"[session] Triton prefill resident ({time.perf_counter() - t0:.2f}s); "
                f"building decode...",
                flush=True,
            )
            self.dec = air.FusedDecoder(staircase=air._staircase_on())
            self.attn_maxl = self.dec.ATTN_MAXL
            # One throwaway prefill so the first real turn is warm: it compiles
            # and caches every kernel and opens the XRT session the launcher
            # now keeps alive. clear_context() either side leaves no state.
            t_w = time.perf_counter()
            self.prefiller.clear_context()
            self.prefiller.prefill([128000])
            self.prefiller.clear_context()
            # And one throwaway decode, for the same reason: the first dispatch
            # of a process costs ~100 ms against a steady-state ~18 (the array
            # is configured for this design for the first time), which on a
            # short answer is most of the wall clock and reads as a slow
            # runtime. Position 0 of the KV cache is the scratch, and every
            # turn's seed_kv overwrites it before it is read.
            self.dec.dispatch(128000, 0)
            print(
                f"[session] ready: Triton prefill + AIR decode resident "
                f"(ATTN_MAXL={self.attn_maxl}, warmup {time.perf_counter() - t_w:.2f}s).",
                flush=True,
            )

    return TritonSession


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("cpu", "npu"), default="npu")
    ap.add_argument("--ops", default="all", help="NPU ops (see prefill.py)")
    ap.add_argument("--prompt", default=None, help="token ids, comma separated")
    ap.add_argument("--text", default=None, help="prompt text (needs transformers)")
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--model", default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument(
        "--seq-len", type=int, default=2048, help="passed through to generate()"
    )
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="multi-turn chat REPL (/clear resets, /exit quits)",
    )
    ap.add_argument("--system", default=None, help="system prompt for --interactive")
    ap.add_argument(
        "--profile", action="store_true", help="per-operator prefill timing"
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument(
        "--prefill-only",
        action="store_true",
        help="run the Triton prefill and check the gate, without the decode "
        "(so it needs no `make compile-decode`)",
    )
    args = ap.parse_args(argv)

    air = None if args.prefill_only else _air_inference_module()

    # On HSA the decode is ours too: mlir-air's decoder dispatches through
    # pyxrt, and HSA needs the instruction stream patched per token instead.
    # See hsa_decode.py.
    if os.environ.get("AMD_TRITON_NPU_RUNTIME") == "hsa" and air is not None:
        from hsa_decode import make_hsa_decoder_class

        air.FusedDecoder = make_hsa_decoder_class(
            air, str(config._air_llms_root().parent / "fused_decode")
        )

    if args.interactive:
        # Swap our prefill into mlir-air's Session, then hand off to its REPL.
        air.Session = make_session_class(
            air, args.backend, args.ops, args.model, config.N_LAYERS
        )
        air.interactive_chat(
            system=args.system,
            seq_len=args.seq_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            greedy=args.greedy,
        )
        return 0

    if args.text is not None:
        from transformers import AutoTokenizer

        ids = AutoTokenizer.from_pretrained(_tokenizer_dir(air)).encode(args.text)
    elif args.prompt:
        ids = [int(t) for t in args.prompt.split(",")]
    else:
        ids = list(config.PROMPT)

    if args.prefill_only:
        first, _ = run_prefill(
            ids,
            args.backend,
            args.ops,
            args.max_seq,
            args.model,
            None,
            profile=args.profile,
        )
        if ids != list(config.PROMPT):
            return 0
        ok = first == config.EXPECT_FIRST
        print(
            f"[triton-prefill] first token {first} "
            f"(expect {config.EXPECT_FIRST}) -- {'PASS' if ok else 'FAIL'}"
        )
        return 0 if ok else 1

    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    kv_path = tmp.name
    try:
        first, P = run_prefill(
            ids,
            args.backend,
            args.ops,
            args.max_seq,
            args.model,
            kv_path,
            profile=args.profile,
        )
        if ids == list(config.PROMPT):
            verdict = "PASS" if first == config.EXPECT_FIRST else "FAIL"
            print(
                f"[triton-prefill] first token {first} "
                f"(expect {config.EXPECT_FIRST}) -- {verdict}",
                flush=True,
            )

        # The npz is already written, so the decode's own prefill step has
        # nothing left to do. Everything after it -- seed_kv, the dispatch
        # loop, sampling -- is mlir-air's, untouched.
        air.run_prefill = lambda *a, **k: None

        t0 = time.time()
        out = air.generate(
            ids,
            args.max_tokens,
            args.seq_len,
            kv_path,
            greedy=args.greedy,
        )
        dt = time.time() - t0
        print(f"[e2e] {len(out)} tokens in {dt:.2f}s", flush=True)
        print(f"[e2e] ids {out}", flush=True)
        try:
            from transformers import AutoTokenizer

            tk = AutoTokenizer.from_pretrained(_tokenizer_dir(air))
            print(f"[e2e] {tk.decode(ids)!r} ->  {tk.decode(out)!r}", flush=True)
        except Exception as e:  # tokenizer is a convenience, not the gate
            print(f"[e2e] (no tokenizer: {e})", flush=True)
    finally:
        os.unlink(kv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
