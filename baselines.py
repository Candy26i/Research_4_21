# baselines.py
# -*- coding: utf-8 -*-
"""
Baselines for the tool-routing research experiment.

Evaluates three non-learned policies on the test split using the same
shaped reward as routing_aware_reward, so comparisons are apples-to-apples.

Baselines:
  1) no_tool        - manager never calls any tool; just directly picks an answer
                      (using a small inference-time LLM pass).
  2) random_route   - first tool is chosen uniformly at random from TOOL_REGISTRY.
  3) fixed_route    - deterministic chain: fast_solver -> deep_reasoner.
  4) all_tools      - calls every tool exactly once (upper-bound cost).

For each baseline, we:
  - generate completions by directly simulating the manager+tool flow
  - score them with routing_aware_reward
  - report accuracy, avg_cost_per_correct, routing_entropy, tool_appropriateness_rate

Intended use:
  python baselines.py \
    --task_name medqa \
    --data_path medqa \
    --split_path splits_medqa_1000.json \
    --split_key test_ids \
    --manager_base_model Qwen/Qwen3-8B \
    --tool_base_model   Qwen/Qwen3-8B \
    --fast_solver_tool_out   fast_solver_tool_medqa \
    --deep_reasoner_tool_out deep_reasoner_tool_medqa \
    --medical_kb_tool_out    medical_kb_tool_medqa \
    --answer_critic_tool_out answer_critic_tool_medqa \
    --baseline all \
    --out_dir baselines_medqa_test
"""

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse everything from the main script
from agents_routing_research_v1 import (
    # Config
    ANSWER_LABELS, ANSWER_CANONICAL_TO_TOKEN,
    TASK_NAME,
    # Runtime setup
    ID2EX, ALLOWED_TOOL_IDS,
    TOOL_REGISTRY, TOOL_ORDER,
    _run_tool, init_tool_agents,
    configure_task, configure_cuda_runtime,
    resolve_data_path_arg, read_json, load_raw_task,
    # Reward + analysis
    routing_aware_reward, analyze_routing_trace,
    # Runtime helpers
    runtime_device, runtime_dtype, set_seed, ensure_dir,
    write_json, write_jsonl, render_chat_messages, dumps_json,
    parse_answer_label_lastline,
    build_manager_system_prompt, _format_choices_block,
    # Tool registry
    TOOL_CACHE, TOOL_RAW_CACHE,
)
from agents_routing_research_v1 import MAX_MANAGER_TOOL_CALLS  # noqa


def _build_answer_only_messages(eid: int, q: str, ctx: str, choices: Dict[str, str]) -> List[Dict[str, str]]:
    """Prompt for a direct answer (no tool routing). Last line must be ANSWER_<LABEL>."""
    answer_lines = "\n".join([f"  ANSWER_{ANSWER_CANONICAL_TO_TOKEN[lab]}" for lab in ANSWER_LABELS])
    choices_block = _format_choices_block(choices)
    sys = (
        "You are a medical expert. Given a question, reason briefly and give a final answer. "
        "The last line of your response must be exactly one of:\n"
        f"{answer_lines}\n"
        "Nothing after that line."
    )
    user = (
        f"Example ID: {eid}\n\n"
        f"Question:\n{q}\n\n"
        f"{choices_block}"
        f"Context:\n{ctx}\n\n"
        f"Answer directly. Final line must be ANSWER_<label>."
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


@torch.no_grad()
def _llm_generate(model, tok, messages: List[Dict[str, str]], max_new_tokens: int = 512, temperature: float = 0.0) -> str:
    prompt = render_chat_messages(tok, messages, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    do_sample = (temperature > 1e-6)
    kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = max(float(temperature), 1e-6)
    out = model.generate(**inputs, **kwargs)
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def _synthesize_completion_messages(
    tool_calls: List[str],
    tool_outputs: List[str],
    final_answer_text: str,
) -> List[Dict[str, Any]]:
    """Build a list of messages that looks like what the trainer's reward function
    consumes (assistant with tool_calls + tool results + final assistant text).
    This lets us reuse routing_aware_reward as-is.
    """
    msgs = []
    for i, (name, out) in enumerate(zip(tool_calls, tool_outputs)):
        call_id = f"call_{i}"
        msgs.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": dumps_json({"example_id": 0})},
            }],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": out,
        })
    # Final assistant turn
    msgs.append({"role": "assistant", "content": final_answer_text})
    return msgs


def _make_answer_prompt(q: str, ctx: str, choices: Dict[str, str], tool_outputs_block: str) -> List[Dict[str, str]]:
    """Prompt for the manager's final answer turn given (question, context, tool outputs)."""
    answer_lines = "\n".join([f"  ANSWER_{ANSWER_CANONICAL_TO_TOKEN[lab]}" for lab in ANSWER_LABELS])
    choices_block = _format_choices_block(choices)
    sys = (
        "You have already consulted some tools (their JSON outputs are below). "
        "Synthesize their results and give a final answer. "
        f"Final line must be exactly one of:\n{answer_lines}\n"
        "Nothing after that line."
    )
    user = (
        f"Question:\n{q}\n\n"
        f"{choices_block}"
        f"Context:\n{ctx}\n\n"
        f"Tool outputs:\n{tool_outputs_block}\n\n"
        "Decide the answer."
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


def _baseline_policy_routes(baseline: str) -> List[str]:
    """Return the deterministic tool sequence for a baseline, or [] for 'no_tool'
    or 'random_route' (handled specially)."""
    if baseline == "no_tool":
        return []
    if baseline == "random_route":
        return []  # handled by caller
    if baseline == "fixed_route":
        return ["fast_solver_tool", "deep_reasoner_tool"]
    if baseline == "all_tools":
        return TOOL_ORDER[:MAX_MANAGER_TOOL_CALLS]
    raise ValueError(f"Unknown baseline: {baseline}")


def run_baseline(
    baseline: str,
    rows: List[Dict[str, Any]],
    manager_model,
    manager_tok,
    out_dir: str,
    seed: int = 42,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Run one baseline over all rows, log trace rows in the same format as
    routing_aware_reward, and return aggregate metrics."""
    rng = random.Random(seed)
    ensure_dir(out_dir)
    raw_trace_path = os.path.join(out_dir, f"baseline_{baseline}_trace.jsonl")

    all_completions = []
    all_gts = []
    all_eids = []

    for r in rows:
        eid = int(r["example_id"])
        q, ctx = r["question"], r["context"]
        choices = r.get("choices", {}) or {}

        # 1. Decide tool sequence
        if baseline == "random_route":
            k = rng.randint(1, MAX_MANAGER_TOOL_CALLS)
            routes = [rng.choice(TOOL_ORDER) for _ in range(k)]
        else:
            routes = _baseline_policy_routes(baseline)

        # 2. Execute tool calls
        tool_outputs = []
        for tname in routes:
            # _run_tool reads from ID2EX, not from the row directly
            out = _run_tool(tname, eid)
            tool_outputs.append(out)

        # 3. Build the final-answer prompt
        if routes:
            tool_outputs_block = "\n".join([
                f"- {nm}: {out}" for nm, out in zip(routes, tool_outputs)
            ])
            final_msgs = _make_answer_prompt(q, ctx, choices, tool_outputs_block)
        else:
            final_msgs = _build_answer_only_messages(eid, q, ctx, choices)

        final_text = _llm_generate(
            manager_model, manager_tok, final_msgs,
            max_new_tokens=512, temperature=temperature,
        )

        # 4. Synthesize the messages into the reward-function format
        completion_msgs = _synthesize_completion_messages(routes, tool_outputs, final_text)
        all_completions.append(completion_msgs)
        all_gts.append(r["ground_truth"])
        all_eids.append(eid)

    # Score with the shared reward fn (writes to RAW_TRACE_JSONL if set)
    # We set RAW_TRACE_JSONL globally so routing_aware_reward logs there
    import agents_routing_research_v1 as M
    old_trace = M.RAW_TRACE_JSONL
    old_fail = M.FAIL_BUFFER_JSONL
    M.RAW_TRACE_JSONL = raw_trace_path
    M.FAIL_BUFFER_JSONL = None  # skip fail buffer for baselines

    # Clear file so we get a clean trace
    with open(raw_trace_path, "w", encoding="utf-8"):
        pass

    try:
        rewards = routing_aware_reward(
            prompts=[None] * len(all_completions),
            completions=all_completions,
            ground_truth=all_gts,
            example_id=all_eids,
        )
    finally:
        M.RAW_TRACE_JSONL = old_trace
        M.FAIL_BUFFER_JSONL = old_fail

    # Aggregate
    summary_json = os.path.join(out_dir, f"baseline_{baseline}_summary.json")
    summary_txt = os.path.join(out_dir, f"baseline_{baseline}_summary.txt")
    summary = analyze_routing_trace(raw_trace_path, out_json=summary_json, out_txt=summary_txt)
    summary["baseline"] = baseline
    summary["mean_reward"] = float(sum(rewards) / max(1, len(rewards)))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_name", type=str, default="medqa",
                    choices=["pubmedqa", "medqa", "medxpertqa_text", "generic"])
    ap.add_argument("--label_space", type=str, default="")
    ap.add_argument("--data_path", type=str, default="")
    ap.add_argument("--split_path", type=str, required=True)
    ap.add_argument("--split_key", type=str, default="test_ids",
                    choices=["train_ids", "dev_ids", "test_ids"])
    ap.add_argument("--max_eval", type=int, default=0,
                    help="If >0, evaluate only the first N rows of the chosen split.")

    ap.add_argument("--manager_base_model", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tool_base_model", type=str, default="Qwen/Qwen3-0.6B")

    ap.add_argument("--fast_solver_tool_out",    type=str, default="fast_solver_tool_adapter")
    ap.add_argument("--deep_reasoner_tool_out",  type=str, default="deep_reasoner_tool_adapter")
    ap.add_argument("--medical_kb_tool_out",     type=str, default="medical_kb_tool_adapter")
    ap.add_argument("--answer_critic_tool_out",  type=str, default="answer_critic_tool_adapter")

    ap.add_argument("--baseline", type=str, default="all",
                    choices=["all", "no_tool", "random_route", "fixed_route", "all_tools"])
    ap.add_argument("--out_dir", type=str, default="baselines_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    configure_cuda_runtime()
    configure_task(args.task_name, args.label_space)
    data_path = resolve_data_path_arg(args.data_path, args.task_name)
    print(f"[BASELINE] task={args.task_name} data_path={data_path}")

    set_seed(args.seed)
    rows = load_raw_task(data_path)
    splits = read_json(args.split_path)
    split_ids = set(splits[args.split_key])
    eval_rows = [r for r in rows if int(r["example_id"]) in split_ids]
    if args.max_eval > 0:
        eval_rows = eval_rows[: args.max_eval]
    print(f"[BASELINE] evaluating {len(eval_rows)} rows from {args.split_key}")

    # Populate ID2EX + ALLOWED_TOOL_IDS so _run_tool works
    import agents_routing_research_v1 as M
    M.ID2EX.clear()
    for r in rows:
        M.ID2EX[int(r["example_id"])] = {
            "question": r["question"],
            "context": r["context"],
            "choices": r.get("choices", {}) or {},
        }
    M.ALLOWED_TOOL_IDS = set(split_ids)
    for tname in TOOL_ORDER:
        TOOL_CACHE[tname].clear()
        TOOL_RAW_CACHE[tname].clear()

    # Init tool runtime
    adapter_paths = {
        "fast_solver_tool": args.fast_solver_tool_out,
        "deep_reasoner_tool": args.deep_reasoner_tool_out,
        "medical_kb_tool": args.medical_kb_tool_out,
        "answer_critic_tool": args.answer_critic_tool_out,
    }
    init_tool_agents(args.tool_base_model, adapter_paths, device=runtime_device())

    # Init manager LLM for direct answering
    mtok = AutoTokenizer.from_pretrained(args.manager_base_model, trust_remote_code=True)
    mtok.padding_side = "left"
    if mtok.pad_token_id is None and mtok.eos_token_id is not None:
        mtok.pad_token_id = mtok.eos_token_id
    mmodel = AutoModelForCausalLM.from_pretrained(
        args.manager_base_model, torch_dtype=runtime_dtype(), trust_remote_code=True,
    ).to(runtime_device())
    mmodel.eval()
    for p in mmodel.parameters():
        p.requires_grad_(False)

    ensure_dir(args.out_dir)
    baselines_to_run = (
        ["no_tool", "random_route", "fixed_route", "all_tools"]
        if args.baseline == "all" else [args.baseline]
    )

    all_summaries = {}
    for bname in baselines_to_run:
        print(f"\n[BASELINE] running {bname} ...")
        t0 = time.time()
        summary = run_baseline(
            baseline=bname,
            rows=eval_rows,
            manager_model=mmodel,
            manager_tok=mtok,
            out_dir=args.out_dir,
            seed=args.seed,
            temperature=args.temperature,
        )
        dt = time.time() - t0
        all_summaries[bname] = summary
        acc = summary.get("accuracy", 0.0)
        apc = summary.get("avg_cost_per_correct", float("inf"))
        mr  = summary.get("mean_reward", 0.0)
        print(f"[BASELINE] {bname}: acc={acc:.4f}  avg_cost_per_correct={apc:.2f}  "
              f"mean_reward={mr:.4f}  ({dt:.1f}s)")

    # Write comparative table
    table_path = os.path.join(args.out_dir, "baselines_comparison.json")
    write_json(table_path, all_summaries)
    print(f"\n[BASELINE] wrote comparison to {table_path}")

    # Print tabular summary
    print("\n" + "=" * 88)
    print(f"{'baseline':<16}{'acc':>8}{'mean_R':>10}{'avg_cost':>12}{'cost/correct':>14}{'entropy':>10}")
    print("-" * 88)
    for bname, s in all_summaries.items():
        if s.get("num_rows", 0) == 0:
            continue
        print(
            f"{bname:<16}"
            f"{s['accuracy']:>8.4f}"
            f"{s['mean_reward']:>10.4f}"
            f"{s['avg_cost_per_completion']:>12.3f}"
            f"{s['avg_cost_per_correct']:>14.2f}"
            f"{s['routing_entropy_normalized']:>10.4f}"
        )
    print("=" * 88)


if __name__ == "__main__":
    main()
