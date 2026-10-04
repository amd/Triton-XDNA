# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Resolve a transform script's references into `transform_library/`.

Stdlib-only on purpose, like `msvc.py`: it is a pure text rewrite, and keeping
it out of `driver.py` lets a test load it straight off disk without importing
the backend -- whose package loads air's LLVM, which then collides with any
other LLVM the process loads (`LLVM ERROR: Option ... already exists`).
"""

import os
import re


def _inject_transform_library(user_script):
    """
    Process library references in a user transform script.

    Two mechanisms:
    1. transform.include calls are expanded inline (parameter substitution,
       SSA renaming) to avoid segfaults in mlir-air's transform interpreter
       when resolving transform.include across region boundaries.
    2. foreach_match @name symbol references are resolved by injecting the
       referenced named_sequence definitions into the module (these cannot
       be inlined because foreach_match resolves symbols at runtime).

    Args:
        user_script: The user's transform script as a string

    Returns:
        str: The processed script
    """
    has_includes = "transform.include" in user_script
    has_foreach_match = "foreach_match" in user_script
    if not has_includes and not has_foreach_match:
        return user_script

    # Load library content from transform_library/ directory
    lib_dir = os.path.join(os.path.dirname(__file__), "transform_library")
    if not os.path.isdir(lib_dir):
        return user_script
    parts = []
    for fname in sorted(os.listdir(lib_dir)):
        if fname.endswith(".mlir"):
            with open(os.path.join(lib_dir, fname), "r") as f:
                parts.append(f.read())
    lib_content = "\n".join(parts)

    # Parse all named sequences: full text (for injection) and decomposed (for inlining)
    full_seq_pattern = re.compile(
        r"((?://[^\n]*\n)*"
        r"transform\.named_sequence\s+@(\w+)\s*\([^)]*\)"
        r"(?:\s*->\s*!transform\.any_op)?"
        r"\s*\{.*?\n\})",
        re.DOTALL,
    )
    full_sequences = {}
    for m in full_seq_pattern.finditer(lib_content):
        full_sequences[m.group(2)] = m.group(1)

    # Parse inlinable sequences (readonly or consumed param, for transform.include)
    inline_seq_pattern = re.compile(
        r"transform\.named_sequence\s+@(\w+)\s*\(\s*"
        r"%(\w+)\s*:\s*!transform\.any_op\s*\{transform\.(?:readonly|consumed)\}\s*\)"
        r"(\s*->\s*!transform\.any_op)?"
        r"\s*\{(.*?)\n\}",
        re.DOTALL,
    )
    sequences = {}
    for match in inline_seq_pattern.finditer(lib_content):
        name = match.group(1)
        param = match.group(2)
        has_result = match.group(3) is not None
        body = match.group(4)
        sequences[name] = (param, body, has_result)

    if not sequences and not full_sequences:
        return user_script

    # Inline transform.include calls to avoid mlir-air segfaults
    include_pattern = re.compile(
        r"(?:(%\w+)\s*=\s*)?"
        r"transform\.include\s+@(\w+)\s+"
        r"failures\(\w+\)\s*"
        r"\((%\w+)\)\s*"
        r":\s*\(!transform\.any_op\)\s*->\s*"
        r"(?:!transform\.any_op|\(\s*\))"
    )

    _counter = [0]

    def _expand(text, depth=0):
        if depth > 20 or "transform.include" not in text:
            return text

        def _replace_include(m):
            result_var = m.group(1)
            seq_name = m.group(2)
            actual_arg = m.group(3)

            if seq_name not in sequences:
                return m.group(0)

            param, body, has_result = sequences[seq_name]
            expanded = body.replace(f"%{param}", actual_arg)

            yield_match = re.search(
                r"transform\.yield(?:\s+(%\w+)\s*:\s*!transform\.any_op)?",
                expanded,
            )
            if yield_match:
                yielded_var = yield_match.group(1)
                expanded = expanded[: yield_match.start()].rstrip()
                if result_var and yielded_var:
                    expanded = expanded.replace(yielded_var, result_var)

            suffix = f"_lib{_counter[0]}"
            _counter[0] += 1
            local_vars = set(re.findall(r"%(\w+)", expanded))
            actual_name = actual_arg.lstrip("%")
            result_name = result_var.lstrip("%") if result_var else ""
            skip = {actual_name, result_name, "__", ""}
            for var in local_vars:
                if var not in skip and not var.startswith("_lib"):
                    expanded = re.sub(
                        rf"(?<!\w)%{re.escape(var)}(?!\w)",
                        f"%{var}{suffix}",
                        expanded,
                    )

            return expanded

        text = include_pattern.sub(_replace_include, text)
        return _expand(text, depth + 1)

    result = _expand(user_script) if has_includes else user_script

    # Inject named sequences referenced by foreach_match (symbol references
    # that cannot be inlined — they must exist as definitions in the module).
    if has_foreach_match or "foreach_match" in result:
        all_refs = set(re.findall(r"@(\w+)", result))
        all_refs.discard("__transform_main")
        # Transitively resolve dependencies
        needed = set()
        worklist = [n for n in all_refs if n in full_sequences]
        while worklist:
            name = worklist.pop()
            if name in needed:
                continue
            needed.add(name)
            for dep in re.findall(r"@(\w+)", full_sequences[name]):
                if dep in full_sequences and dep not in needed:
                    worklist.append(dep)
        # Inject definitions for all unresolved @name references
        # (matchers/actions referenced by foreach_match, plus their deps)
        if needed:
            module_marker = "module attributes {transform.with_named_sequence} {"
            idx = result.find(module_marker)
            if idx != -1:
                insert_pos = idx + len(module_marker)
                injection = "\n\n".join(
                    full_sequences[n] for n in full_sequences if n in needed
                )
                result = (
                    result[:insert_pos]
                    + "\n\n"
                    + injection
                    + "\n\n"
                    + result[insert_pos:]
                )

    return result
