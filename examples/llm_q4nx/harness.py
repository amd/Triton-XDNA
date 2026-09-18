# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The driver every Q4NX example shares: a Triton prefill feeding mlir-air's
fused decode.

A model's own directory supplies two things and nothing else -- a `ModelSpec`
(what the decode build and mlir-air's driver need to be told) and a prefill
class (the forward, and the constants it is written against). Everything here
is the same for all of them: argument parsing, the KV handoff, the gate, the
chat session, and the swap that puts our prefill under mlir-air's `generate()`.

The prefill class must provide what mlir-air's `Session.run_turn` calls on its
prefiller -- `prefill`, `kv_view`, `clear_context` -- plus `load_weights` and
`save_kv_npz`. `model.py` in each example is that class.
"""

import argparse
import importlib.util
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import airsrc  # noqa: E402
import decode_build  # noqa: E402

# run_tests.py grades this exit code as a skip, not a failure -- the autotools
# convention, and what examples/gpt2 uses. These examples need a multi-gigabyte
# weight bundle from the Hub, so a runner with no network declines the test
# rather than reporting a defect it has not found.
SKIP_EXIT_CODE = 77


class ExampleUnavailable(Exception):
    """The example cannot run here (weights, tokenizer or decode build)."""


def air_inference_module(model):
    """mlir-air's driver for this model, importable and unmodified."""
    # Before the import: mlir-air's module reads its decode-shape selection at
    # import time, so a later choice would not be seen. See
    # airsrc.select_decode_artifact for why these examples fix it rather than
    # taking mlir-air's default.
    airsrc.select_decode_artifact()

    # Also before the import, and for the same reason: the drivers that take
    # this read it into a module-level constant. See `ModelSpec.decode_dir_env`.
    if model.decode_dir_env:
        os.environ[model.decode_dir_env] = airsrc.fused_decode_dir()

    airsrc.add_air_paths(model.air_package, *model.extra_packages)
    path = os.path.join(
        str(airsrc.air_llms_root()), model.air_package, model.air_inference
    )
    spec = importlib.util.spec_from_file_location("q4nx_inference", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["q4nx_inference"] = mod
    spec.loader.exec_module(mod)
    return mod


def tokenizer_dir(air, model):
    """The tokenizer mlir-air's driver uses, or this model's checkpoint.

    The drivers spell it differently -- the 1B exports `_TOKENIZER`, the 3B
    `TOKENIZER_DEFAULT` -- so try both before falling back. The tokenizer is a
    convenience for printing text, never the gate, so an unknown spelling
    degrades to the fallback rather than failing the run.
    """
    for attr in ("_TOKENIZER", "TOKENIZER_DEFAULT"):
        if air is not None and hasattr(air, attr):
            return getattr(air, attr)
    return os.environ.get(
        "Q4NX_TOKENIZER_DIR", os.path.expanduser(model.tokenizer_fallback)
    )


def check_host_memory(spec):
    """Decline before allocating if this host cannot hold the weights.

    The prefill dequantizes to bf16 on the host, and an undersized host does
    not raise -- it gets SIGKILLed partway through the load, which no `except`
    can turn into the skip this would otherwise report. So the check has to be
    up front and approximate. Linux only; elsewhere it passes and the caller
    takes its chances, as it did before.
    """
    if not spec.min_host_gib:
        return
    try:
        with open("/proc/meminfo") as f:
            avail_kib = next(
                int(line.split()[1]) for line in f if line.startswith("MemAvailable:")
            )
    except (OSError, StopIteration):
        return
    avail = avail_kib / 1024 / 1024
    if avail < spec.min_host_gib:
        raise ExampleUnavailable(
            f"{spec.name} needs about {spec.min_host_gib:.0f} GiB of host memory "
            f"for its dequantized bf16 weights; this host has {avail:.1f} GiB "
            f"available"
        )


def run_prefill(
    prefill_cls, cfg, ids, backend, ops, max_seq, model, kv_path, profile=False
):
    """Our prefill -> the handoff npz.

    Returns `(first_token, P, prefiller)`. The prefiller is returned because
    mlir-air's per-model drivers do not agree on how the handoff is made: the
    1B reads the npz, while the 3B is handed this object and calls
    `kv_view()` on it. See `ModelSpec.driver_api`. `kv_path=None` skips the
    npz, which is what the object-handoff models and `--prefill-only` want.
    """
    import torch

    m = prefill_cls(
        backend=backend,
        ops=ops,
        n_layers=cfg.N_LAYERS,
        max_seq=max_seq,
        model=model,
        expect_model=cfg.MODEL_NAME,
    )
    m.timer.enabled = profile
    t0 = time.time()
    try:
        m.load_weights()
    except Exception as e:  # noqa: BLE001 -- any failure to obtain weights
        raise ExampleUnavailable(f"cannot load the q4nx weights: {e}") from e
    t_load = time.time() - t0
    if profile:  # one warm pass so timings exclude compilation
        m.prefill(ids)
        m.clear_context()
        m.timer.reset()
    t0 = time.time()
    logits = m.prefill(ids)
    t_run = time.time() - t0
    if profile:
        m.timer.report(t_run)
    first = int(torch.argmax(logits))
    if kv_path is not None:
        m.save_kv_npz(kv_path, first, ids)
    # The model is named here, not left to the caller's shell trace. Every
    # Q4NX gate otherwise prints identical text -- same P, same first token --
    # so a loop that ran one model twice would read as two passes.
    print(
        f"[triton-prefill] model={cfg.MODEL_NAME} backend={backend} "
        f"ops={sorted(m.enabled)} P={len(ids)} load {t_load:.1f}s "
        f"prefill {t_run:.2f}s first={first}",
        flush=True,
    )
    return first, len(ids), m


def make_session_class(air, prefill_cls, cfg, backend, ops, model):
    """mlir-air's Session with OUR prefill in place of theirs.

    `Session.run_turn` only ever calls `prefill`, `kv_view` and
    `clear_context` on its prefiller, and the prefill class provides all three
    with the same semantics, so nothing below run_turn needs to change. The
    decode stays exactly as mlir-air builds it.
    """

    class TritonSession(air.Session):
        def __init__(self, seq_len=2048):
            import time

            import numpy as np

            self.np = np
            self.seq_len = seq_len
            t0 = time.perf_counter()
            print("[session] loading prefill weights (once)...", flush=True)
            self.prefiller = prefill_cls(
                backend=backend,
                ops=ops,
                n_layers=cfg.N_LAYERS,
                max_seq=seq_len,
                model=model,
                expect_model=cfg.MODEL_NAME,
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
            self.prefiller.prefill([cfg.BOS])
            self.prefiller.clear_context()
            # And one throwaway decode, for the same reason: the first dispatch
            # of a process costs ~100 ms against a steady-state ~18 (the array
            # is configured for this design for the first time), which on a
            # short answer is most of the wall clock and reads as a slow
            # runtime. Position 0 of the KV cache is the scratch, and every
            # turn's seed_kv overwrites it before it is read.
            self.dec.dispatch(cfg.BOS, 0)
            print(
                f"[session] ready: Triton prefill + AIR decode resident "
                f"(ATTN_MAXL={self.attn_maxl}, warmup {time.perf_counter() - t_w:.2f}s).",
                flush=True,
            )

    return TritonSession


def generate_via_prefiller(air, spec, cfg, args, ids, prefiller):
    """Generation for drivers that take the prefill object rather than a file.

    mlir-air's 3B driver hands its `generate_stream` a prefiller and calls
    `clear_context()`, `prefill()` and `kv_view()` on it -- the interface our
    prefill already has, so it goes in directly with nothing to neutralize.

    Two details that are easy to get wrong and silent when wrong:

    * `min_prefill=1`. Their default is `PREFILL_MIN_TOKENS` (96), below which
      the prompt is replayed token-by-token through the decode and the
      prefiller is never touched. That is the right default for them -- it is
      faster for short prompts -- but it would mean the six-token gate prompt
      exercised none of the Triton prefill this example exists to test, and
      still printed a plausible answer.
    * `stop_on_eos=False`. The npz path generates a fixed count, so leaving
      their EOS stop on would make the two paths' token counts differ for
      reasons that have nothing to do with the kernels.
    """
    dec = getattr(air, spec.decoder_class)(
        args.model or cfg.MODEL_DEFAULT,
        airsrc.fused_decode_dir(),
        model_type=spec.model_type,
    )
    gen, t_prompt, t_gen = air.generate_stream(
        dec,
        None,  # tokenizer: only used for streaming output, which we do not do
        ids,
        args.max_tokens,
        stream=False,
        prefiller=prefiller,
        min_prefill=1,
        stop_on_eos=False,
    )
    print(
        f"[e2e] prompt {t_prompt:.2f}s, decode {t_gen:.2f}s "
        f"({len(gen) / t_gen:.2f} tok/s)",
        flush=True,
    )
    return gen


def generate_via_kv_arrays(air, spec, cfg, args, ids, prefiller, first, ttft):
    """Generation for drivers whose `generate()` has no handoff parameter.

    Qwen3-4B's driver does its own prefill unconditionally and hands the result
    to the decode loop, so there is nothing to pass in -- but the prefill it
    calls is one function, `_prefill_npu(prompt, model, seq_len=None) ->
    (K, V, first, ttft)`, and replacing it puts our KV in front of their decode
    without reaching past their public `generate()`. That is the same move the
    npz path makes on `run_prefill`, and it is preferred to calling their
    `_decode_loop` directly: `generate()` also builds the decoder, checks P
    against ATTN_MAXL and prints the throughput lines the benchmark scrapes.

    `stop_on_eos=False` for the same reason as the prefiller path: the other
    routes generate a fixed count, and an EOS stop here would make the token
    counts differ for reasons unrelated to the kernels.
    """
    K, V = prefiller.kv_stack()
    # Their fourth return value is the time-to-first-token they print on a line
    # the nightly benchmark scrapes. It is our prefill that spent it, so the
    # measured wall clock goes back in rather than a zero that would read as a
    # free prefill.
    air._prefill_npu = lambda prompt, model, seq_len=None: (K, V, first, ttft)
    return air.generate(
        list(ids),
        args.max_tokens,
        model=args.model or cfg.MODEL_DEFAULT,
        greedy=args.greedy,
        stop_on_eos=False,
    )


def build_parser(doc):
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument("--backend", choices=("cpu", "npu"), default="npu")
    ap.add_argument(
        "--ops", default="all", help="NPU ops: all, or a comma-separated subset"
    )
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
    return ap


def main(spec, cfg, prefill_cls, doc=None, argv=None):
    """Run one model end to end.

    Args:
        spec: its `registry.ModelSpec`.
        cfg: its `config` module -- N_LAYERS, PROMPT, EXPECT_FIRST.
        prefill_cls: its prefill class, from `model.py`.
    """
    args = build_parser(doc).parse_args(argv)

    if os.environ.get("AMD_TRITON_NPU_RUNTIME") == "hsa" and not spec.supports_hsa:
        # Up front, for the same reason --interactive is below: a property of
        # the model, not of anything on disk. Without this the run loads
        # weights for minutes and then dies inside hsa_decode.py on
        # `air.FusedDecoder`, an attribute this model's driver does not define.
        raise SystemExit(
            f"AMD_TRITON_NPU_RUNTIME=hsa is not supported for {spec.name}. The "
            f"HSA decode adapter subclasses mlir-air's `FusedDecoder`, which "
            f"only its npz-API drivers define; this model's driver names its "
            f"decoder {spec.decoder_class or 'something else'}. Use the "
            f"default XRT runtime."
        )

    if args.interactive and spec.driver_api != "npz":
        # Before anything is imported or loaded: mlir-air's
        # Session/interactive_chat pair exists only on the npz drivers. The
        # prefiller ones ship their own `repl`, built around `generate_stream`
        # and a decoder this harness constructs differently, so there is nothing
        # to swap our prefill into. This is a property of the model, so say so
        # before asking anyone to rebuild artifacts or wait for a weight load.
        raise SystemExit(
            f"--interactive is not supported for {spec.name}: mlir-air's driver "
            f"for it exposes no Session/interactive_chat to host our prefill. "
            f"Use --max-tokens for a single turn."
        )

    check_host_memory(spec)

    air = None if args.prefill_only else air_inference_module(spec)

    if air is not None:
        # One model's artifacts under another's name decode to fluent nonsense
        # rather than failing, so check before the decoder loads them.
        decode_build.check_stamp(spec)

    # On HSA the decode is ours too: mlir-air's decoder dispatches through
    # pyxrt, and HSA needs the instruction stream patched per token instead.
    # See hsa_decode.py.
    if os.environ.get("AMD_TRITON_NPU_RUNTIME") == "hsa" and air is not None:
        from hsa_decode import make_hsa_decoder_class

        air.FusedDecoder = make_hsa_decoder_class(air, airsrc.fused_decode_dir())

    if args.interactive:
        # Swap our prefill into mlir-air's Session, then hand off to its REPL.
        air.Session = make_session_class(
            air, prefill_cls, cfg, args.backend, args.ops, args.model
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

        ids = AutoTokenizer.from_pretrained(tokenizer_dir(air, spec)).encode(args.text)
    elif args.prompt:
        ids = [int(t) for t in args.prompt.split(",")]
    else:
        ids = list(cfg.PROMPT)

    def gate(first):
        """The canonical prompt's first token. Returns True if it holds."""
        ok = first == cfg.EXPECT_FIRST
        print(
            f"[triton-prefill] first token {first} "
            f"(expect {cfg.EXPECT_FIRST}) for {cfg.MODEL_NAME} "
            f"-- {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
        return ok

    if args.prefill_only:
        first, _, _ = run_prefill(
            prefill_cls,
            cfg,
            ids,
            args.backend,
            args.ops,
            args.max_seq,
            args.model,
            None,
            profile=args.profile,
        )
        if ids != list(cfg.PROMPT):
            return 0
        return 0 if gate(first) else 1

    # Only the npz drivers read a handoff file. Writing one for the others
    # would be a large K/V dump nothing opens -- [N_LAYERS, P, DK] per tensor.
    if spec.driver_api == "npz":
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        tmp.close()
        kv_path = tmp.name
    else:
        kv_path = None
    try:
        t_pf = time.time()
        first, P, prefiller = run_prefill(
            prefill_cls,
            cfg,
            ids,
            args.backend,
            args.ops,
            args.max_seq,
            args.model,
            kv_path,
            profile=args.profile,
        )
        ttft = time.time() - t_pf
        # Stop rather than decode from a prefill already known to be wrong.
        # Generation would still produce fluent text -- the decode is fed a KV
        # cache, not a verdict -- so a caller reading the exit status would be
        # told the run succeeded. --prefill-only gates the same way.
        if ids == list(cfg.PROMPT) and not gate(first):
            return 1

        t0 = time.time()
        if spec.driver_api == "prefiller":
            out = generate_via_prefiller(air, spec, cfg, args, ids, prefiller)
        elif spec.driver_api == "kv_arrays":
            out = generate_via_kv_arrays(
                air, spec, cfg, args, ids, prefiller, first, ttft
            )
        else:
            # The npz is already written, so the decode's own prefill step has
            # nothing left to do. Everything after it -- seed_kv, the dispatch
            # loop, sampling -- is mlir-air's, untouched.
            air.run_prefill = lambda *a, **k: None
            out = air.generate(
                ids, args.max_tokens, args.seq_len, kv_path, greedy=args.greedy
            )
        dt = time.time() - t0
        print(f"[e2e] {len(out)} tokens in {dt:.2f}s", flush=True)
        print(f"[e2e] ids {out}", flush=True)

        # The decode gate. Everything above this line is prefill: the
        # first-token check cannot see the builder environment, -DMODEL_TYPE,
        # GLU_SLICE, the core stack, or which driver API was used, because all
        # of those live in the artifact this generation just ran.
        golden = getattr(cfg, "EXPECT_IDS", None)
        if golden and args.greedy and ids == list(cfg.PROMPT):
            got = list(out)[: len(golden)]
            if got != list(golden):
                print(
                    f"[e2e] decode MISMATCH for {cfg.MODEL_NAME}\n"
                    f"  expected {list(golden)}\n"
                    f"  got      {got}",
                    flush=True,
                )
                return 1
            print(
                f"[e2e] decode ids match the recorded {len(golden)} for "
                f"{cfg.MODEL_NAME} -- PASS",
                flush=True,
            )
        try:
            from transformers import AutoTokenizer

            tk = AutoTokenizer.from_pretrained(tokenizer_dir(air, spec))
            print(f"[e2e] {tk.decode(ids)!r} ->  {tk.decode(out)!r}", flush=True)
        except Exception as e:  # tokenizer is a convenience, not the gate
            print(f"[e2e] (no tokenizer: {e})", flush=True)
    finally:
        if kv_path is not None:
            os.unlink(kv_path)
    return 0


def run(spec, cfg, prefill_cls, doc=None, argv=None):
    """`main`, with the skip convention applied. For an example's __main__."""
    try:
        return main(spec, cfg, prefill_cls, doc=doc, argv=argv)
    except ExampleUnavailable as e:
        print(f"SKIP: {e}", flush=True)
        return SKIP_EXIT_CODE
