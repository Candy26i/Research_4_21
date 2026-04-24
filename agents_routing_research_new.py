# agents_as_tools_three_tools_belief_v2.py
# -*- coding: utf-8 -*-
"""
Three-tool agentic GRPO pipeline with differentiated tools for routing research.

Tools:
  1) fast_solver_tool     - quick small-model candidate generation (cost=1.0)
  2) deep_reasoner_tool   - careful large-model reasoning over candidates (cost=10.0)
  3) answer_critic_tool   - adversarial critique of a favored answer (cost=5.0)

Manager (trained via GRPO) must emit a BELIEF_STATE JSON block before each
tool call or final answer. Shaped reward rewards correctness, valid format,
tool diversity, and belief-state quality; penalizes repeated tools, budget
overruns, and fake tool-call text.

Fixes over the previous draft:
- is_main_process() is a function (dynamic), not an import-time constant
- fcntl file lock on fail buffer / raw trace writes
- Tool artifact regex tightened (no false positives on explanations)
- Shaped reward rebalanced so belief bonus doesn't dominate correctness
- Full multi-GPU + optional vLLM server support
- Works on pubmedqa, medqa, medxpertqa_text out of the box
"""

import argparse
import glob
import hashlib
import importlib
import inspect
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# FIX: fcntl is POSIX-only. On Windows (where multi-GPU training is not
# practical anyway due to NCCL limitations), skip file locking entirely.
try:
    import fcntl
    _FCNTL_AVAILABLE = True
except ImportError:
    fcntl = None
    _FCNTL_AVAILABLE = False

import numpy as np
import torch
import transformers
from packaging.version import Version


def _load_local_dotenv(dotenv_path: str = "", override: bool = False) -> None:
    """Minimal .env loader so TEACHER_* vars work without extra dependencies."""
    if not dotenv_path:
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    path = os.path.abspath(dotenv_path)
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                value = value.strip()
                if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                    value = value[1:-1]
                if override or key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


_load_local_dotenv()

DATASETS_AVAILABLE = False
DATASETS_IMPORT_ERROR: Optional[Exception] = None
try:
    from datasets import Dataset, load_dataset
    DATASETS_AVAILABLE = True
except Exception as _datasets_import_error:
    Dataset = Any
    load_dataset = None
    DATASETS_IMPORT_ERROR = _datasets_import_error

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Trainer,
    TrainingArguments,
)

TRL_AVAILABLE = False
TRL_IMPORT_ERROR: Optional[Exception] = None
try:
    from trl import GRPOConfig, GRPOTrainer
    TRL_AVAILABLE = True
except Exception as _trl_import_error:
    GRPOConfig = None
    GRPOTrainer = None
    TRL_IMPORT_ERROR = _trl_import_error

try:
    from peft import LoraConfig, PeftModel, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


# =========================================================
# Runtime helpers
# =========================================================
MIN_TRANSFORMERS = "4.53.0"
MIN_TRL = "0.19.0"


def require_clean_runtime() -> None:
    import trl
    tf_ver = str(getattr(transformers, "__version__", "0"))
    trl_ver = str(getattr(trl, "__version__", "0"))
    if Version(tf_ver) < Version(MIN_TRANSFORMERS):
        raise RuntimeError(f"transformers>={MIN_TRANSFORMERS} required, found {tf_ver}")
    if Version(trl_ver) < Version(MIN_TRL):
        raise RuntimeError(f"trl>={MIN_TRL} required, found {trl_ver}")
    if is_main_process():
        print(f"[ENV] transformers={tf_ver} trl={trl_ver}")


def require_trl(stage_name: str) -> None:
    if TRL_AVAILABLE:
        return
    detail = "" if TRL_IMPORT_ERROR is None else f" Original: {type(TRL_IMPORT_ERROR).__name__}: {TRL_IMPORT_ERROR}"
    raise RuntimeError(f"{stage_name} requires `trl`.{detail}")


def require_datasets(stage_name: str) -> None:
    if DATASETS_AVAILABLE:
        return
    detail = "" if DATASETS_IMPORT_ERROR is None else f" Original: {type(DATASETS_IMPORT_ERROR).__name__}: {DATASETS_IMPORT_ERROR}"
    raise RuntimeError(f"{stage_name} requires `datasets`.{detail}")


def get_local_rank() -> int:
    try:
        return int(os.environ.get("LOCAL_RANK", "-1"))
    except Exception:
        return -1


def get_global_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except Exception:
        return 0


def get_world_size() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        return 1


def is_main_process() -> bool:
    """FIX: dynamic rank detection. Accelerate sets RANK after imports complete."""
    return get_global_rank() == 0


def configure_cuda_runtime() -> None:
    local_rank = get_local_rank()
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)


def runtime_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = get_local_rank()
    if local_rank >= 0:
        return f"cuda:{local_rank}"
    return "cuda"


def runtime_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _import_optional_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _signature_parameter_names(callable_obj: Any) -> Optional[set]:
    try:
        return set(inspect.signature(callable_obj).parameters.keys())
    except Exception:
        return None


def _filter_supported_kwargs(callable_obj: Any, kwargs: Dict[str, Any], label: str) -> Dict[str, Any]:
    supported = _signature_parameter_names(callable_obj)
    if supported is None:
        return dict(kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted([k for k in kwargs.keys() if k not in filtered])
    if dropped and is_main_process():
        print(f"[{label}] skipped unsupported kwargs: {', '.join(dropped)}")
    return filtered


def _trainer_processing_kwargs(processing_obj: Any) -> Dict[str, Any]:
    if not TRL_AVAILABLE:
        return {}
    supported = _signature_parameter_names(GRPOTrainer.__init__) or set()
    if "processing_class" in supported:
        return {"processing_class": processing_obj}
    if "tokenizer" in supported:
        return {"tokenizer": processing_obj}
    return {}


def validate_distributed_runtime(stage_name: str, require_cuda: bool = False, use_vllm: bool = False) -> None:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(f"{stage_name} requires CUDA but torch.cuda unavailable.")
    if get_world_size() > 1 and not torch.cuda.is_available():
        raise RuntimeError(f"{stage_name} WORLD_SIZE>1 but torch.cuda unavailable.")
    if use_vllm:
        if os.name == "nt":
            raise RuntimeError("vLLM not supported on native Windows.")
        if _import_optional_module("vllm") is None:
            raise RuntimeError(f"{stage_name} requested vLLM but `vllm` not importable.")


def validate_grpo_batch_geometry(per_device_train_bs: int, grad_accum: int, num_generations: int) -> None:
    world_size = max(1, get_world_size())
    effective_batch = int(per_device_train_bs) * world_size * int(grad_accum)
    if effective_batch <= 0:
        raise RuntimeError("GRPO effective batch size must be positive.")
    if int(num_generations) <= 0:
        raise RuntimeError("GRPO num_generations must be positive.")
    if effective_batch % int(num_generations) != 0:
        raise RuntimeError(
            f"Invalid GRPO batch geometry: per_device({per_device_train_bs}) * world_size({world_size}) * "
            f"grad_accum({grad_accum}) = {effective_batch}, must divide num_generations({num_generations})."
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =========================================================
# JSON / IO helpers
# =========================================================
def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl_locked(path: str, rows: List[Dict[str, Any]]) -> None:
    """FIX: fcntl-locked append for multi-rank safety on POSIX.
    On Windows fcntl is unavailable; skip locking (Windows multi-GPU is
    already impractical, and single-GPU has no contention)."""
    if not rows:
        return
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        if _FCNTL_AVAILABLE:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
        try:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
        finally:
            if _FCNTL_AVAILABLE:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass


def dumps_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    try:
                        obj = json.loads(chunk)
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        return None
    return None


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text", "")))
            else:
                parts.append(dumps_json(item) if isinstance(item, (dict, list)) else str(item))
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        return dumps_json(content)
    return str(content)


def _fallback_render_messages(messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
    parts = []
    for m in messages:
        role = str(m.get("role", "")).strip() or "user"
        if role == "assistant" and isinstance(m.get("tool_calls"), list):
            content = _message_content_to_text(m.get("content"))
            if content:
                parts.append(f"assistant: {content}")
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                payload = {"name": str(fn.get("name", "")).strip(), "arguments": fn.get("arguments", "{}")}
                parts.append(f"assistant_tool_call: {dumps_json(payload)}")
            continue
        if role == "tool":
            name = str(m.get("name", "tool")).strip() or "tool"
            parts.append(f"tool[{name}]: {_message_content_to_text(m.get('content'))}")
            continue
        parts.append(f"{role}: {_message_content_to_text(m.get('content'))}")
    if add_generation_prompt:
        parts.append("assistant: ")
    return "\n".join(parts)


def render_chat_messages(tokenizer: Any, messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        except Exception:
            return _fallback_render_messages(messages, add_generation_prompt)
    except Exception:
        return _fallback_render_messages(messages, add_generation_prompt)


# =========================================================
# Task config
# =========================================================
TASK_NAME = "pubmedqa"
ANSWER_LABELS: List[str] = ["yes", "no", "maybe"]
ANSWER_TOKEN_TO_CANONICAL: Dict[str, str] = {"YES": "yes", "NO": "no", "MAYBE": "maybe"}
ANSWER_CANONICAL_TO_TOKEN: Dict[str, str] = {"yes": "YES", "no": "NO", "maybe": "MAYBE"}
MEDQA_REGIONS: List[str] = []
MEDQA_REGION_ALIASES: Dict[str, str] = {
    "us": "US",
    "usa": "US",
    "u.s.": "US",
    "mainland": "Mainland",
    "cn": "Mainland",
    "china": "Mainland",
    "taiwan": "Taiwan",
    "tw": "Taiwan",
}
MEDQA_REGION_QBANK_FILENAMES: Dict[str, str] = {
    "US": "US_qbank.jsonl",
    "Mainland": "chinese_qbank.jsonl",
    "Taiwan": "taiwanese_qbank.jsonl",
}
ANSWER_LASTLINE_RE = re.compile(
    r"^\s*(?:answer\s*[:=\-]?\s*)?ANSWER_(YES|NO|MAYBE)\b[^\w]*$",
    re.IGNORECASE,
)


def _label_to_token(label: str) -> str:
    s = str(label).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError(f"Invalid label: {label!r}")
    return s.upper()


def _default_labels_for_task(task_name: str) -> List[str]:
    t = (task_name or "").strip().lower()
    if t == "medqa":
        return ["A", "B", "C", "D", "E"]
    if t == "medxpertqa_text":
        return list("ABCDEFGHIJ")
    if t == "pubmedqa":
        return ["yes", "no", "maybe"]
    return ["yes", "no", "maybe"]


def _build_answer_regex(tokens: List[str]) -> re.Pattern:
    alts = "|".join([re.escape(t) for t in sorted(tokens, key=len, reverse=True)])
    return re.compile(
        rf"^\s*(?:answer\s*[:=\-]?\s*)?ANSWER_({alts})\b[^\w]*$",
        re.IGNORECASE,
    )


def _parse_label_space_arg(label_space: str) -> List[str]:
    if not label_space or not label_space.strip():
        return []
    return [p.strip() for p in label_space.split(",") if p.strip()]


def _canonicalize_medqa_region(value: Any) -> Optional[str]:
    s = str(value).strip()
    if not s:
        return None
    key = s.lower()
    return MEDQA_REGION_ALIASES.get(key)


def _normalize_medqa_regions_arg(medqa_regions: Any) -> List[str]:
    if medqa_regions is None:
        return []
    if isinstance(medqa_regions, str):
        raw_items = [p.strip() for p in medqa_regions.split(",") if p.strip()]
    elif isinstance(medqa_regions, (list, tuple, set)):
        raw_items = [str(x).strip() for x in medqa_regions if str(x).strip()]
    else:
        raw_items = [str(medqa_regions).strip()]

    if any(x.lower() == "all" for x in raw_items):
        return []

    out: List[str] = []
    seen = set()
    bad = []
    for raw in raw_items:
        canon = _canonicalize_medqa_region(raw)
        if canon is None:
            bad.append(raw)
            continue
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    if bad:
        allowed = ", ".join(sorted(set(MEDQA_REGION_ALIASES.values())))
        raise ValueError(f"Unsupported --medqa_regions value(s): {bad}. Allowed: {allowed}, or all.")
    return out


def configure_medqa_regions(medqa_regions: Any) -> List[str]:
    global MEDQA_REGIONS
    MEDQA_REGIONS = _normalize_medqa_regions_arg(medqa_regions)
    return MEDQA_REGIONS


def _normalize_label(raw: Any) -> str:
    s = str(raw).strip()
    if not s:
        return s
    tok = _label_to_token(s)
    if tok in ANSWER_TOKEN_TO_CANONICAL:
        return ANSWER_TOKEN_TO_CANONICAL[tok]
    if s in ANSWER_CANONICAL_TO_TOKEN:
        return s
    return s


def configure_task(task_name: str, label_space: str = "") -> Tuple[str, List[str]]:
    global TASK_NAME, ANSWER_LABELS, ANSWER_TOKEN_TO_CANONICAL, ANSWER_CANONICAL_TO_TOKEN, ANSWER_LASTLINE_RE
    t = (task_name or "pubmedqa").strip().lower()
    labels = _parse_label_space_arg(label_space) or _default_labels_for_task(t)

    token_to_canonical: Dict[str, str] = {}
    canonical_to_token: Dict[str, str] = {}
    canonical_labels: List[str] = []

    for lb in labels:
        canonical = str(lb).strip()
        if not canonical:
            continue
        token = _label_to_token(canonical)
        if token in token_to_canonical and token_to_canonical[token] != canonical:
            raise ValueError(f"Label collision token={token}")
        token_to_canonical[token] = canonical
        canonical_to_token[canonical] = token
        if canonical not in canonical_labels:
            canonical_labels.append(canonical)

    if not canonical_labels:
        raise ValueError("Empty label space.")

    TASK_NAME = t
    ANSWER_LABELS = canonical_labels
    ANSWER_TOKEN_TO_CANONICAL = token_to_canonical
    ANSWER_CANONICAL_TO_TOKEN = canonical_to_token
    ANSWER_LASTLINE_RE = _build_answer_regex(list(token_to_canonical.keys()))
    return TASK_NAME, ANSWER_LABELS


# =========================================================
# Dataset loading
# =========================================================
def _read_json_or_jsonl(path: str) -> Any:
    p = str(path)
    if p.lower().endswith(".jsonl"):
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSONL parse error in {p}:{i}: {e}") from e
        return rows
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _default_data_path_for_task(task_name: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    t = (task_name or "").strip().lower()
    if t == "medqa":
        return os.path.join(base_dir, "MedQA", "data_clean", "questions")
    if t == "medxpertqa_text":
        return os.path.join(base_dir, "MedXpertQA", "Text")
    return os.path.join(base_dir, "Pubmedqa", "pqal_question_context_groundtruth.json")


def resolve_data_path_arg(data_path_arg: str, task_name: str) -> str:
    arg = (data_path_arg or "").strip()
    if not arg:
        return _default_data_path_for_task(task_name)
    alias = arg.lower()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if alias in {"pubmedqa", "pubmed", "pqal"}:
        return os.path.join(base_dir, "Pubmedqa")
    if alias in {"medqa"}:
        return os.path.join(base_dir, "MedQA")
    if alias in {"medxpertqa_text", "medxpertqa", "medxpert"}:
        return os.path.join(base_dir, "MedXpertQA", "Text")
    return arg


def _medqa_region_dir_candidates(path: str, requested_regions: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    p = os.path.abspath(path)
    if not os.path.isdir(p):
        return []

    direct_region = _canonicalize_medqa_region(os.path.basename(p))
    if direct_region is not None:
        return [(direct_region, p)]

    questions_root = p
    if os.path.basename(p).lower() != "questions":
        nested = os.path.join(p, "data_clean", "questions")
        if os.path.isdir(nested):
            questions_root = nested

    regions = list(requested_regions or [])
    if not regions:
        discovered: List[str] = []
        seen = set()
        for name in sorted(os.listdir(questions_root)):
            region = _canonicalize_medqa_region(name)
            full = os.path.join(questions_root, name)
            if region is None or not os.path.isdir(full) or region in seen:
                continue
            seen.add(region)
            discovered.append(region)
        regions = discovered

    out: List[Tuple[str, str]] = []
    for region in regions:
        region_dir = os.path.join(questions_root, region)
        if os.path.isdir(region_dir):
            out.append((region, region_dir))
    return out


def _canonical_medqa_region_files(region: str, region_dir: str) -> List[str]:
    lower = region.lower()
    clean_candidates = [
        os.path.join(region_dir, f"{lower}_clean_all.jsonl"),
        os.path.join(region_dir, "clean_all.jsonl"),
    ]
    for fp in clean_candidates:
        if os.path.isfile(fp):
            return [fp]

    split_candidates = [
        os.path.join(region_dir, "train.jsonl"),
        os.path.join(region_dir, "dev.jsonl"),
        os.path.join(region_dir, "test.jsonl"),
    ]
    split_files = [fp for fp in split_candidates if os.path.isfile(fp)]
    if split_files:
        return split_files

    qbank_name = MEDQA_REGION_QBANK_FILENAMES.get(region, "")
    if qbank_name:
        qbank_fp = os.path.join(region_dir, qbank_name)
        if os.path.isfile(qbank_fp):
            return [qbank_fp]

    fallback = sorted([
        fp for fp in glob.glob(os.path.join(region_dir, "*.jsonl"))
        if os.path.isfile(fp)
        and os.path.basename(fp) not in {".DS_Store"}
        and "4_options" not in fp
        and "metamap" not in fp.lower()
    ])
    return fallback[:1]


def _discover_medqa_files(path: str, requested_regions: Optional[List[str]] = None) -> List[str]:
    p = os.path.abspath(path)
    if os.path.isfile(p):
        return [p]
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Data path not found: {path}")

    files: List[str] = []
    for region, region_dir in _medqa_region_dir_candidates(p, requested_regions=requested_regions):
        files.extend(_canonical_medqa_region_files(region, region_dir))
    if files:
        return files

    # Final fallback for unusual layouts: still avoid obviously derived helper files.
    fallback = sorted([
        fp for fp in glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)
        if os.path.isfile(fp)
        and "4_options" not in fp
        and "metamap" not in fp.lower()
    ])
    if fallback:
        return fallback
    raise FileNotFoundError(f"No canonical MedQA json/jsonl under: {path}")


def _discover_data_files(path: str, task_name: str, medqa_regions: Optional[List[str]] = None) -> List[str]:
    p = os.path.abspath(path)
    if os.path.isfile(p):
        return [p]
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Data path not found: {path}")
    t = (task_name or "").strip().lower()

    if t == "pubmedqa":
        preferred = [
            os.path.join(p, "pqal_question_context_groundtruth.json"),
            os.path.join(p, "Pubmedqa", "pqal_question_context_groundtruth.json"),
            os.path.join(p, "pubmedqa", "pqal_question_context_groundtruth.json"),
        ]
        for c in preferred:
            if os.path.isfile(c):
                return [c]
        fallback = sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
        if fallback:
            return [fallback[0]]
        raise FileNotFoundError(f"No PubMedQA json under: {path}")

    if t == "medqa":
        return _discover_medqa_files(p, requested_regions=medqa_regions)

    if t == "medxpertqa_text":
        preferred = [
            os.path.join(p, "dev.jsonl"),
            os.path.join(p, "test.jsonl"),
            os.path.join(p, "Text", "dev.jsonl"),
            os.path.join(p, "Text", "test.jsonl"),
        ]
        files = [x for x in preferred if os.path.isfile(x)]
        if files:
            return files
        fallback = sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True))
        fallback = [x for x in fallback if os.path.isfile(x)]
        if fallback:
            return fallback
        raise FileNotFoundError(f"No MedXpertQA Text jsonl under: {path}")

    files = sorted([x for x in glob.glob(os.path.join(p, "**", "*.json*"), recursive=True) if os.path.isfile(x)])
    if files:
        return files
    raise FileNotFoundError(f"No json/jsonl under: {path}")


def _sorted_choice_items(choices: Dict[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not isinstance(choices, dict):
        return out
    for k, v in choices.items():
        kk = str(k).strip()
        vv = str(v).strip()
        if kk:
            out.append((kk, vv))
    out.sort(key=lambda x: x[0])
    return out


def _build_default_context(ex: Dict[str, Any], choices: Dict[str, str], effective_task: str) -> str:
    parts: List[str] = []
    if "context" in ex and str(ex.get("context", "")).strip():
        parts.append(str(ex.get("context", "")).strip())
    if choices:
        choice_lines = [f"{k}. {v}" for k, v in _sorted_choice_items(choices)]
        parts.append("Options:\n" + "\n".join(choice_lines))
    if effective_task == "medxpertqa_text":
        for key, prefix in [
            ("medical_task", "Medical task"),
            ("body_system", "Body system"),
            ("question_type", "Question type"),
        ]:
            val = str(ex.get(key, "")).strip()
            if val:
                parts.append(f"{prefix}: {val}")
    meta = str(ex.get("meta_info", "")).strip()
    if meta:
        parts.append(f"Meta: {meta}")
    phrases = ex.get("metamap_phrases", [])
    if isinstance(phrases, list) and phrases:
        pv = ", ".join([str(x) for x in phrases if str(x).strip()])[:1000]
        if pv:
            parts.append(f"MetaMap phrases: {pv}")
    return "\n\n".join([p for p in parts if p.strip()])


def _next_unique_id(seen: set, next_auto: int, candidate: Optional[Any]) -> Tuple[int, int]:
    if candidate is not None and str(candidate).strip():
        try:
            eid = int(candidate)
        except Exception:
            eid = None
        if eid is not None and eid not in seen:
            seen.add(eid)
            return eid, max(next_auto, eid + 1)
    while next_auto in seen:
        next_auto += 1
    eid = int(next_auto)
    seen.add(eid)
    return eid, next_auto + 1


def _infer_medqa_region_from_source_file(source_file: str) -> str:
    parts = [p.strip().lower() for p in re.split(r"[\\/]+", os.path.normpath(source_file)) if p.strip()]
    for part in parts:
        canon = _canonicalize_medqa_region(part)
        if canon is not None:
            return canon
    return ""


def load_raw_dataset(path: str, task_name: str = "", medqa_regions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    effective_task = (task_name or TASK_NAME).strip().lower()
    files = _discover_data_files(path, effective_task, medqa_regions=medqa_regions)
    rows: List[Dict[str, Any]] = []
    seen_ids: set = set()
    next_auto_id = 0
    medqa_regions = list(medqa_regions or [])

    for fp in files:
        raw = _read_json_or_jsonl(fp)
        if isinstance(raw, dict):
            iterable = raw.items()
            keyed = True
        elif isinstance(raw, list):
            iterable = enumerate(raw)
            keyed = False
        else:
            raise ValueError(f"Unsupported type in {fp}: {type(raw)}")

        for k, ex in iterable:
            if not isinstance(ex, dict):
                raise ValueError(f"Unsupported ex type in {fp} key {k!r}: {type(ex)}")
            eid_candidate = k if keyed else ex.get("example_id", ex.get("id", None))
            eid, next_auto_id = _next_unique_id(seen_ids, next_auto_id, eid_candidate)
            question = str(ex.get("question", "")).strip()
            choices = ex.get("choices", ex.get("options", {}))
            if not isinstance(choices, dict):
                choices = {}
            norm_choices = {str(kk).strip(): str(vv).strip() for kk, vv in choices.items() if str(kk).strip()}
            context = str(ex.get("context", "")).strip()
            if not context:
                context = _build_default_context(ex, norm_choices, effective_task)
            gt_raw = ex.get("ground_truth", ex.get("answer_idx", ex.get("label", ex.get("answer_label", ""))))
            answer_field = str(ex.get("answer", "")).strip()
            if not str(gt_raw).strip() and answer_field in norm_choices:
                gt_raw = answer_field
            gt = _normalize_label(gt_raw)
            answer_text = answer_field
            if answer_text in norm_choices:
                answer_text = str(norm_choices.get(answer_text, "")).strip()
            if not answer_text and gt and gt in norm_choices:
                answer_text = str(norm_choices.get(gt, "")).strip()
            rows.append({
                "example_id": eid,
                "raw_id": str(ex.get("id", ex.get("example_id", eid))),
                "question": question,
                "context": context,
                "ground_truth": gt,
                "answer_text": answer_text,
                "choices": norm_choices,
                "task_name": TASK_NAME,
                "source_file": fp,
                "medqa_region": _infer_medqa_region_from_source_file(fp) if effective_task == "medqa" else "",
                "meta_info": str(ex.get("meta_info", "")),
                "medical_task": str(ex.get("medical_task", "")),
                "body_system": str(ex.get("body_system", "")),
                "question_type": str(ex.get("question_type", "")),
            })

    allowed = set(ANSWER_LABELS)
    cleaned_rows: List[Dict[str, Any]] = []
    dropped_missing_q = 0
    dropped_bad_label = 0
    for r in rows:
        if not r["question"]:
            dropped_missing_q += 1
            continue
        if r["ground_truth"] not in allowed:
            dropped_bad_label += 1
            continue
        cleaned_rows.append(r)

    if effective_task == "medqa" and medqa_regions:
        wanted = set(medqa_regions)
        filtered_rows = [r for r in cleaned_rows if r.get("medqa_region", "") in wanted]
        if is_main_process():
            region_counts = Counter([r.get("medqa_region", "") or "unknown" for r in cleaned_rows])
            print(f"[DATA][MedQA] region_filter={medqa_regions} before={len(cleaned_rows)} counts={dict(region_counts)}")
            print(f"[DATA][MedQA] kept_after_region_filter={len(filtered_rows)}")
        cleaned_rows = filtered_rows

    if not cleaned_rows:
        raise ValueError(
            f"No valid rows from {path}. missing_q={dropped_missing_q} bad_label={dropped_bad_label}"
        )
    if (dropped_missing_q or dropped_bad_label) and is_main_process():
        print(f"[DATA] dropped missing_q={dropped_missing_q} bad_label={dropped_bad_label} kept={len(cleaned_rows)}")
    return cleaned_rows


def load_raw_task(path: str) -> List[Dict[str, Any]]:
    return load_raw_dataset(path=path, task_name=TASK_NAME, medqa_regions=MEDQA_REGIONS)


# =========================================================
# Splits
# =========================================================
def _alloc_counts_stratified(label_counts: Dict[str, int], target: int) -> Dict[str, int]:
    labels = sorted(label_counts.keys())
    total = sum(label_counts.values())
    if total == 0:
        return {lab: 0 for lab in labels}
    floor_counts = {}
    frac = {}
    for lab in labels:
        x = target * (label_counts[lab] / total)
        floor_counts[lab] = int(np.floor(x))
        frac[lab] = x - floor_counts[lab]
    remainder = target - sum(floor_counts.values())
    order = sorted(labels, key=lambda lab: frac[lab], reverse=True)
    i = 0
    while remainder > 0 and i < 100000:
        lab = order[i % len(order)]
        floor_counts[lab] += 1
        remainder -= 1
        i += 1
    return floor_counts


def make_splits(rows: List[Dict[str, Any]], test_size: int = 200, dev_size: int = 160, seed: int = 42) -> Dict[str, List[int]]:
    if test_size + dev_size >= len(rows):
        raise ValueError("test_size + dev_size must be < total")
    rng = random.Random(seed)
    by_label: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        by_label[r["ground_truth"]].append(r["example_id"])
    for lab in by_label:
        rng.shuffle(by_label[lab])
    full_counts = {lab: len(ids) for lab, ids in by_label.items()}
    test_counts = _alloc_counts_stratified(full_counts, test_size)
    test_ids = set()
    remaining_by_label: Dict[str, List[int]] = {}
    for lab, ids in by_label.items():
        n = min(test_counts.get(lab, 0), len(ids))
        test_ids.update(ids[:n])
        remaining_by_label[lab] = ids[n:]
    if len(test_ids) < test_size:
        need = test_size - len(test_ids)
        pool = []
        for lab in remaining_by_label:
            pool.extend(remaining_by_label[lab])
        rng.shuffle(pool)
        take = pool[:need]
        test_ids.update(take)
        take_set = set(take)
        for lab in remaining_by_label:
            remaining_by_label[lab] = [x for x in remaining_by_label[lab] if x not in take_set]
    rem_counts = {lab: len(ids) for lab, ids in remaining_by_label.items()}
    dev_counts = _alloc_counts_stratified(rem_counts, dev_size)
    dev_ids = set()
    train_ids = set()
    for lab, ids in remaining_by_label.items():
        n = min(dev_counts.get(lab, 0), len(ids))
        dev_ids.update(ids[:n])
        train_ids.update(ids[n:])
    if len(dev_ids) < dev_size:
        need = dev_size - len(dev_ids)
        pool = list(train_ids)
        rng.shuffle(pool)
        take = pool[:need]
        dev_ids.update(take)
        train_ids.difference_update(set(take))
    return {
        "train_ids": sorted(list(train_ids)),
        "dev_ids": sorted(list(dev_ids)),
        "test_ids": sorted(list(test_ids)),
    }


def subsample_rows(rows: List[Dict[str, Any]], max_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    max_samples = int(max_samples)
    if max_samples <= 0 or max_samples >= len(rows):
        return rows
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_label[str(r["ground_truth"])].append(r)
    for lb in by_label:
        rng.shuffle(by_label[lb])
    wanted = _alloc_counts_stratified({lb: len(items) for lb, items in by_label.items()}, max_samples)
    sampled: List[Dict[str, Any]] = []
    for lb, items in by_label.items():
        sampled.extend(items[:wanted.get(lb, 0)])
    if len(sampled) < max_samples:
        sampled_ids = {int(x["example_id"]) for x in sampled}
        rest = [r for r in rows if int(r["example_id"]) not in sampled_ids]
        rng.shuffle(rest)
        sampled.extend(rest[: max_samples - len(sampled)])
    sampled = sampled[:max_samples]
    sampled.sort(key=lambda x: int(x["example_id"]))
    return sampled


def _row_sample_uid(row: Dict[str, Any]) -> str:
    payload = {
        "question": str(row.get("question", "")).strip(),
        "ground_truth": str(row.get("ground_truth", "")).strip(),
        "answer_text": str(row.get("answer_text", row.get("answer", ""))).strip(),
        "choices": {k: v for k, v in _sorted_choice_items(row.get("choices", row.get("options", {})) or {})},
    }
    raw = dumps_json(payload)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _split_example_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    options = dict(row.get("choices", row.get("options", {})) or {})
    answer_idx = str(row.get("ground_truth", row.get("answer_idx", ""))).strip()
    answer_text = str(row.get("answer_text", row.get("answer", ""))).strip()
    if not answer_text and answer_idx and answer_idx in options:
        answer_text = str(options.get(answer_idx, "")).strip()
    return {
        "example_id": int(row.get("example_id", -1)),
        "sample_uid": _row_sample_uid(row),
        "raw_id": str(row.get("raw_id", row.get("example_id", ""))),
        "question": str(row.get("question", "")).strip(),
        "answer": answer_text,
        "answer_idx": answer_idx,
        "ground_truth": answer_idx,
        "options": options,
        "choices": options,
        "context": str(row.get("context", "")).strip(),
        "task_name": str(row.get("task_name", TASK_NAME)).strip(),
        "source_file": str(row.get("source_file", "")).strip(),
        "medqa_region": str(row.get("medqa_region", "")).strip(),
        "meta_info": str(row.get("meta_info", "")).strip(),
        "medical_task": str(row.get("medical_task", "")).strip(),
        "body_system": str(row.get("body_system", "")).strip(),
        "question_type": str(row.get("question_type", "")).strip(),
    }


def attach_split_examples(splits: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    id2row = {int(r["example_id"]): r for r in rows}
    out = dict(splits)
    for split_name in ["train", "dev", "test"]:
        ids = [int(x) for x in splits.get(f"{split_name}_ids", [])]
        out[f"{split_name}_examples"] = [
            _split_example_from_row(id2row[eid])
            for eid in ids
            if eid in id2row
        ]
    return out


def _normalize_split_example(ex: Dict[str, Any], fallback_id: int) -> Dict[str, Any]:
    choices = ex.get("choices", ex.get("options", {}))
    if not isinstance(choices, dict):
        choices = {}
    norm_choices = {str(k).strip(): str(v).strip() for k, v in choices.items() if str(k).strip()}
    answer_idx = str(ex.get("ground_truth", ex.get("answer_idx", ""))).strip()
    answer_text = str(ex.get("answer_text", ex.get("answer", ""))).strip()
    if not answer_text and answer_idx in norm_choices:
        answer_text = str(norm_choices.get(answer_idx, "")).strip()
    sample = {
        "example_id": int(ex.get("example_id", fallback_id)),
        "sample_uid": str(ex.get("sample_uid", "")).strip(),
        "raw_id": str(ex.get("raw_id", ex.get("example_id", fallback_id))),
        "question": str(ex.get("question", "")).strip(),
        "context": str(ex.get("context", "")).strip(),
        "ground_truth": answer_idx,
        "answer_text": answer_text,
        "choices": norm_choices,
        "task_name": str(ex.get("task_name", TASK_NAME)).strip() or TASK_NAME,
        "source_file": str(ex.get("source_file", "")).strip(),
        "medqa_region": str(ex.get("medqa_region", "")).strip(),
        "meta_info": str(ex.get("meta_info", "")).strip(),
        "medical_task": str(ex.get("medical_task", "")).strip(),
        "body_system": str(ex.get("body_system", "")).strip(),
        "question_type": str(ex.get("question_type", "")).strip(),
    }
    if not sample["sample_uid"]:
        sample["sample_uid"] = _row_sample_uid(sample)
    return sample


def get_split_examples(splits: Dict[str, Any], split_name: str) -> List[Dict[str, Any]]:
    raw = splits.get(f"{split_name}_examples", [])
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(raw):
        if isinstance(ex, dict):
            out.append(_normalize_split_example(ex, fallback_id=i))
    return out


def get_rows_for_split(splits: Dict[str, Any], split_name: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    embedded = get_split_examples(splits, split_name)
    if embedded:
        return embedded
    id2row = {int(r["example_id"]): r for r in rows}
    return [
        id2row[int(eid)]
        for eid in splits.get(f"{split_name}_ids", [])
        if int(eid) in id2row
    ]


def resolve_tool_base_models_for_stage(
    stage_name: str,
    default_tool_base_model: str = "",
    fast_solver_base_model: str = "",
    deep_reasoner_base_model: str = "",
    answer_critic_base_model: str = "",
) -> Dict[str, str]:
    stage = str(stage_name or "").strip()
    default_model = str(default_tool_base_model or "").strip()
    fast_model = str(fast_solver_base_model or "").strip()
    deep_model = str(deep_reasoner_base_model or "").strip()
    critic_model = str(answer_critic_base_model or "").strip()

    out: Dict[str, str] = {}

    if stage in {"train_fast_solver_tool", "train_manager_grpo"}:
        if not fast_model:
            raise ValueError(f"{stage} requires --fast_solver_base_model.")
        out["fast_solver_tool"] = fast_model

    if stage in {"train_deep_reasoner_tool", "train_manager_grpo"}:
        model_name = deep_model or default_model
        if not model_name:
            raise ValueError(f"{stage} requires --tool_base_model or --deep_reasoner_base_model.")
        out["deep_reasoner_tool"] = model_name

    if stage in {"train_answer_critic_tool", "train_manager_grpo"}:
        model_name = critic_model or default_model
        if not model_name:
            raise ValueError(f"{stage} requires --tool_base_model or --answer_critic_base_model.")
        out["answer_critic_tool"] = model_name

    return out


def build_split_scope_metadata(data_path: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "task_name": TASK_NAME,
        "resolved_data_path": os.path.abspath(data_path),
        "num_rows": len(rows),
        "source_files": sorted(set([str(r.get("source_file", "")).strip() for r in rows if str(r.get("source_file", "")).strip()])),
    }
    if TASK_NAME == "medqa":
        meta["medqa_regions"] = list(MEDQA_REGIONS)
        meta["medqa_region_counts"] = dict(Counter([r.get("medqa_region", "") or "unknown" for r in rows]))
    return meta


def validate_split_scope(splits: Dict[str, Any], data_path: str) -> None:
    meta = splits.get("dataset_scope")
    if not isinstance(meta, dict):
        return

    expected_task = str(meta.get("task_name", "")).strip().lower()
    if expected_task and expected_task != TASK_NAME:
        raise RuntimeError(
            f"Split file task mismatch: split built for {expected_task}, current task is {TASK_NAME}."
        )

    expected_regions = list(meta.get("medqa_regions", [])) if TASK_NAME == "medqa" else []
    if TASK_NAME == "medqa" and expected_regions != list(MEDQA_REGIONS):
        raise RuntimeError(
            f"Split file MedQA region scope mismatch: split built with {expected_regions or ['ALL']}, "
            f"current run uses {MEDQA_REGIONS or ['ALL']}. Reuse the same --medqa_regions or remake splits."
        )

    expected_path = str(meta.get("resolved_data_path", "")).strip()
    current_path = os.path.abspath(data_path)
    if expected_path and expected_path != current_path and is_main_process():
        print(f"[WARN] split path differs: split={expected_path} current={current_path}")


# =========================================================
# Sentence selection
# =========================================================
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def split_into_sentences(context: str) -> List[str]:
    if not context:
        return []
    txt = context.replace("\n", " ").strip()
    txt = re.sub(r"\s+", " ", txt)
    parts = re.split(r"(?<=[\.\?\!;])\s+", txt)
    sents = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) < 20:
            continue
        if len(p) > 500:
            p = p[:500]
        sents.append(p)
    if not sents and txt:
        sents = [txt[:500]]
    return sents


def overlap_score(q_words: List[str], s_words: List[str]) -> float:
    if not q_words or not s_words:
        return 0.0
    qs = set(q_words)
    ss = set(s_words)
    inter = len(qs.intersection(ss))
    return inter / (1.0 + 0.05 * len(ss))


def build_candidates(question: str, context: str, top_k: int, rng: random.Random) -> List[Dict[str, Any]]:
    q_words = tokenize_words(question)
    sents = split_into_sentences(context)
    cands = []
    for i, s in enumerate(sents):
        s_words = tokenize_words(s)
        sc = overlap_score(q_words, s_words) + rng.uniform(-0.02, 0.02)
        cands.append({"sid": i, "text": s, "score": float(sc)})
    cands.sort(key=lambda x: x["score"], reverse=True)
    return cands[:top_k]


def pick_evidence(candidates: List[Dict[str, Any]], n_min: int, n_max: int, rng: random.Random) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    n = rng.randint(n_min, n_max)
    n = min(n, len(candidates))
    scores = np.array([max(0.001, c["score"] + 0.05) for c in candidates], dtype=np.float32)
    probs = scores / scores.sum()
    idxs = rng.choices(list(range(len(candidates))), weights=probs.tolist(), k=n * 2)
    uniq, seen = [], set()
    for ix in idxs:
        if ix in seen:
            continue
        uniq.append(ix)
        seen.add(ix)
        if len(uniq) >= n:
            break
    ev = [candidates[ix] for ix in uniq]
    ev.sort(key=lambda x: x["score"], reverse=True)
    return ev[:n]


# =========================================================
# Tool registry (description-based, for research on tool routing)
# Each tool has:
#   system_prompt  - what the tool does (used when running the tool)
#   description    - for the manager to read in its prompt
#   when_to_use    - guidance for the manager
#   when_not_to_use
#   cost           - rollout cost (units are arbitrary; used in cost-aware reward)
#   capabilities   - tags used for routing-appropriateness bonus
#   max_new_tokens - generation length cap when running the tool
# =========================================================

FAST_SOLVER_SYS = (
    "You are a fast solver tool (small model, quick answer).\n"
    "Given a medical question and candidate answers, quickly propose which one seems right "
    "and why. Return ONLY a JSON object with keys:\n"
    "  top_guess: str (the letter or label)\n"
    "  top_guess_rationale: str (<=200 chars)\n"
    "  alternative_guesses: list[{label:str, reason:str}] (0~3 items)\n"
    "  confidence: float 0.0~1.0\n"
    "Be fast. No deep reasoning required.\n"
)

DEEP_REASONER_SYS = (
    "You are a deep reasoner tool (careful, thorough analysis).\n"
    "Given a medical question and candidate answers, analyze each candidate, compare them, "
    "and identify the most defensible choice. Return ONLY a JSON object with keys:\n"
    "  candidate_analyses: list[{label:str, supporting_reasoning:str, "
    "weaknesses:str}] (one entry per candidate)\n"
    "  most_defensible: str (label)\n"
    "  key_discriminators: list[str] (what facts would change the answer)\n"
    "  remaining_uncertainty: str\n"
    "  confidence: float 0.0~1.0\n"
)

ANSWER_CRITIC_SYS = (
    "You are an answer critic tool.\n"
    "Given a medical question, the candidate answers, and a currently-favored answer, "
    "try to FALSIFY the favored answer: find a reason it could be wrong, alternative "
    "interpretations, overlooked clinical features. Return ONLY a JSON object with keys:\n"
    "  favored_answer_weaknesses: list[str] (<=3 items)\n"
    "  better_alternatives: list[{label:str, why:str}] (0~2 items)\n"
    "  clinical_features_overlooked: list[str]\n"
    "  would_change_answer_if: str (what evidence would flip the answer)\n"
    "  confidence_that_favored_is_wrong: float 0.0~1.0\n"
)

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "fast_solver_tool": {
        "system_prompt": FAST_SOLVER_SYS,
        "description": (
            "A quick small-model solver that proposes a top guess with brief rationale. "
            "Fast and cheap."
        ),
        "when_to_use": (
            "For easy questions or as a first pass to generate candidate answers to "
            "critique later. Also as a sanity check."
        ),
        "when_not_to_use": (
            "When the question requires careful clinical reasoning across many candidates, "
            "or when precise medical knowledge is required."
        ),
        "cost": 1.0,
        "capabilities": ["quick_answer", "candidate_generation"],
        "max_new_tokens": 300,
    },
    "deep_reasoner_tool": {
        "system_prompt": DEEP_REASONER_SYS,
        "description": (
            "A careful large-model reasoner that analyzes each candidate answer, compares "
            "them, and identifies the most defensible. Slow and expensive."
        ),
        "when_to_use": (
            "For difficult clinical reasoning with multiple plausible candidates, or when "
            "you need systematic comparison across options."
        ),
        "when_not_to_use": (
            "For easy questions (wasteful) or when you just need a factual lookup."
        ),
        "cost": 10.0,
        "capabilities": ["deep_reasoning", "candidate_comparison", "differential_dx"],
        "max_new_tokens": 600,
    },
    "answer_critic_tool": {
        "system_prompt": ANSWER_CRITIC_SYS,
        "description": (
            "An answer critic that tries to falsify a currently-favored candidate: finds "
            "weaknesses, overlooked features, alternative interpretations."
        ),
        "when_to_use": (
            "After you have a top guess and want to stress-test it, especially when "
            "multiple candidates seem close."
        ),
        "when_not_to_use": (
            "Before you have any candidate answer to critique (use fast_solver or "
            "deep_reasoner first)."
        ),
        "cost": 5.0,
        "capabilities": ["critique", "falsification", "second_opinion"],
        "max_new_tokens": 400,
    },
}

# For backward compat with existing code that iterates TOOL_ORDER and TOOL_SPECS
TOOL_ORDER = list(TOOL_REGISTRY.keys())
TOOL_SPECS = {name: info["system_prompt"] for name, info in TOOL_REGISTRY.items()}

MAX_MANAGER_TOOL_CALLS = 3
WEAK_UNCERTAINTY_FLAGS = ["weak_supervision_generation"]


# =========================================================
# OpenAI-compatible chat client (for optional GPT teacher synthesis)
# Works with OpenAI, Anthropic via a bridge, or any OpenAI-compatible local server.
# =========================================================
class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.timeout = int(timeout)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        import requests  # keep dependency optional
        url = self.base_url
        if not url.endswith("/v1"):
            url = url + "/v1"
        url = url + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "messages": messages, "temperature": float(temperature)}
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return ""


def get_teacher_client_from_env() -> Optional[OpenAICompatClient]:
    """Returns a teacher client if TEACHER_BASE_URL and TEACHER_MODEL are set.

    Environment variables:
        TEACHER_BASE_URL  - API endpoint (e.g. https://api.openai.com)
        TEACHER_API_KEY   - API key (or empty for local servers that don't need auth)
        TEACHER_MODEL     - model id (e.g. gpt-4o-mini, gpt-4o)
        TEACHER_TIMEOUT   - request timeout in seconds (default 60)
    """
    base_url = os.environ.get("TEACHER_BASE_URL", "").strip()
    api_key = os.environ.get("TEACHER_API_KEY", "").strip()
    model = os.environ.get("TEACHER_MODEL", "").strip()
    timeout = int(os.environ.get("TEACHER_TIMEOUT", "60"))
    if not base_url or not model:
        return None
    return OpenAICompatClient(base_url=base_url, api_key=api_key, model=model, timeout=timeout)


def _gpt_call_with_retry(
    teacher: OpenAICompatClient,
    messages: List[Dict[str, str]],
    tool_name: str,
    max_retries: int = 3,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Call teacher API, retry on JSON parse failure, raise on persistent failure."""
    last_raw = None
    for attempt in range(max_retries):
        try:
            raw = teacher.chat(messages, temperature=temperature)
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"GPT teacher call for {tool_name} failed after {max_retries} retries: {type(e).__name__}: {e}"
                ) from e
            continue
        last_raw = raw
        obj = extract_first_json(raw)
        if obj is not None:
            return obj
    raise ValueError(
        f"GPT teacher returned non-JSON for {tool_name} after {max_retries} retries.\n"
        f"Last raw response: {last_raw}"
    )


# =========================================================
# Tool SFT data builder
# =========================================================
def _normalize_fast_solver(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj["top_guess"] = str(obj.get("top_guess", ""))[:60]
    obj["top_guess_rationale"] = str(obj.get("top_guess_rationale", ""))[:200]
    alts = obj.get("alternative_guesses", [])
    norm_alts = []
    if isinstance(alts, list):
        for it in alts[:3]:
            if isinstance(it, dict):
                norm_alts.append({
                    "label": str(it.get("label", ""))[:20],
                    "reason": str(it.get("reason", ""))[:160],
                })
    obj["alternative_guesses"] = norm_alts
    try:
        obj["confidence"] = float(obj.get("confidence", 0.5))
    except Exception:
        obj["confidence"] = 0.5
    obj["confidence"] = max(0.0, min(1.0, obj["confidence"]))
    return obj


def _normalize_deep_reasoner(obj: Dict[str, Any]) -> Dict[str, Any]:
    cas = obj.get("candidate_analyses", [])
    norm_cas = []
    if isinstance(cas, list):
        for it in cas[:10]:
            if isinstance(it, dict):
                norm_cas.append({
                    "label": str(it.get("label", ""))[:20],
                    "supporting_reasoning": str(it.get("supporting_reasoning", ""))[:200],
                    "weaknesses": str(it.get("weaknesses", ""))[:200],
                })
    obj["candidate_analyses"] = norm_cas
    obj["most_defensible"] = str(obj.get("most_defensible", ""))[:20]
    kd = obj.get("key_discriminators", [])
    obj["key_discriminators"] = [str(x)[:160] for x in kd[:5]] if isinstance(kd, list) else []
    obj["remaining_uncertainty"] = str(obj.get("remaining_uncertainty", ""))[:200]
    try:
        obj["confidence"] = float(obj.get("confidence", 0.6))
    except Exception:
        obj["confidence"] = 0.6
    obj["confidence"] = max(0.0, min(1.0, obj["confidence"]))
    return obj


def _normalize_answer_critic(obj: Dict[str, Any]) -> Dict[str, Any]:
    fw = obj.get("favored_answer_weaknesses", [])
    obj["favored_answer_weaknesses"] = [str(x)[:200] for x in fw[:3]] if isinstance(fw, list) else []
    ba = obj.get("better_alternatives", [])
    norm_ba = []
    if isinstance(ba, list):
        for it in ba[:2]:
            if isinstance(it, dict):
                norm_ba.append({
                    "label": str(it.get("label", ""))[:20],
                    "why": str(it.get("why", ""))[:200],
                })
    obj["better_alternatives"] = norm_ba
    cfo = obj.get("clinical_features_overlooked", [])
    obj["clinical_features_overlooked"] = [str(x)[:200] for x in cfo[:5]] if isinstance(cfo, list) else []
    obj["would_change_answer_if"] = str(obj.get("would_change_answer_if", ""))[:200]
    try:
        obj["confidence_that_favored_is_wrong"] = float(obj.get("confidence_that_favored_is_wrong", 0.3))
    except Exception:
        obj["confidence_that_favored_is_wrong"] = 0.3
    obj["confidence_that_favored_is_wrong"] = max(0.0, min(1.0, obj["confidence_that_favored_is_wrong"]))
    # Legacy "confidence" key for compatibility with some downstream code
    obj["confidence"] = 1.0 - obj["confidence_that_favored_is_wrong"]
    return obj


def _weak_tool_target(tool_name: str, q: str, ctx: str, candidates: List[Dict[str, Any]], evidence: List[Dict[str, Any]], choices: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Heuristic targets when GPT synthesis is not used. Low-quality but free."""
    choice_labels = sorted(list((choices or {}).keys())) or (ANSWER_LABELS or [])
    first_label = choice_labels[0] if choice_labels else ""

    if tool_name == "fast_solver_tool":
        alts = []
        for lb in choice_labels[1:3]:
            alts.append({"label": lb, "reason": "Alternative candidate worth consideration."})
        return _normalize_fast_solver({
            "top_guess": first_label,
            "top_guess_rationale": "Heuristic first-choice fallback (weak supervision).",
            "alternative_guesses": alts,
            "confidence": 0.3,
        })

    if tool_name == "deep_reasoner_tool":
        cas = []
        for lb in choice_labels[:5]:
            val = (choices or {}).get(lb, "")
            cas.append({
                "label": lb,
                "supporting_reasoning": f"Option {lb}: {val[:140]}" if val else f"Option {lb}",
                "weaknesses": "Weak supervision; reasoning not yet performed.",
            })
        return _normalize_deep_reasoner({
            "candidate_analyses": cas,
            "most_defensible": first_label,
            "key_discriminators": ["Clinical features differentiating candidates."],
            "remaining_uncertainty": "High; needs real reasoning.",
            "confidence": 0.3,
        })

    if tool_name == "answer_critic_tool":
        alts = []
        if len(choice_labels) >= 2:
            alts.append({"label": choice_labels[1], "why": "Alternative worth consideration."})
        return _normalize_answer_critic({
            "favored_answer_weaknesses": ["Weak supervision; critique not performed."],
            "better_alternatives": alts,
            "clinical_features_overlooked": [],
            "would_change_answer_if": "Additional clinical reasoning performed.",
            "confidence_that_favored_is_wrong": 0.4,
        })

    raise ValueError(f"Unknown tool_name={tool_name}")


def build_tool_sft_data_from_splits(
    data_path: str,
    split_path: str,
    out_dir: str,
    seed: int = 42,
    top_k: int = 20,
    variants_train: int = 3,
    variants_dev: int = 2,
    ev_min: int = 3,
    ev_max: int = 6,
    synth_mode: str = "weak",   # "weak" or "gpt"
    gpt_temperature: float = 0.2,
    gpt_max_retries: int = 3,
) -> Dict[str, Tuple[str, str]]:
    set_seed(seed)
    splits = read_json(split_path)
    validate_split_scope(splits, data_path)
    train_rows = get_split_examples(splits, "train")
    dev_rows = get_split_examples(splits, "dev")
    if not train_rows or not dev_rows:
        raise RuntimeError(
            "Split file must contain embedded train_examples and dev_examples. "
            "Remake splits with the current make_splits stage."
        )

    # Teacher setup
    teacher: Optional[OpenAICompatClient] = None
    if synth_mode == "gpt":
        teacher = get_teacher_client_from_env()
        if teacher is None:
            raise ValueError(
                "synth_mode='gpt' requires TEACHER_BASE_URL and TEACHER_MODEL env vars. "
                "Set them, or use --tool_synth_mode weak."
            )
        print(f"[GPT SYNTH] teacher model={teacher.model} base_url={teacher.base_url}")

    out_map: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {
        name: ([], []) for name in TOOL_ORDER
    }

    def _make_tool_target(tool_name: str, q: str, ctx: str,
                          candidates: List[Dict[str, Any]],
                          evidence: List[Dict[str, Any]],
                          choices: Dict[str, str]) -> Dict[str, Any]:
        """Build the target for one (tool, example) pair. GPT or weak."""
        if synth_mode == "gpt" and teacher is not None:
            sys_prompt = TOOL_SPECS[tool_name]
            user = _tool_input_user_message(tool_name, 0, q, ctx, choices)
            # strip the synthetic "Example ID: 0\n" prefix for teacher (not needed)
            user = user.replace("Example ID: 0\n", "", 1)
            obj = _gpt_call_with_retry(
                teacher=teacher,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": user}],
                tool_name=tool_name,
                max_retries=gpt_max_retries,
                temperature=gpt_temperature,
            )
            return _normalize_tool_output(tool_name, obj)
        # weak mode
        return _weak_tool_target(tool_name, q, ctx, candidates, evidence, choices=choices)

    def add_one(ex: Dict[str, Any], variants: int, is_dev: bool) -> None:
        eid = int(ex["example_id"])
        q, ctx = ex["question"], ex["context"]
        choices = ex.get("choices", {}) or {}
        base_rng = random.Random(seed * 100000 + eid)
        for _ in range(variants):
            rng = random.Random(base_rng.randint(0, 10**9))
            candidates = build_candidates(q, ctx, top_k=top_k, rng=rng)
            evidence = pick_evidence(candidates, n_min=ev_min, n_max=ev_max, rng=rng)
            for tool_name in TOOL_ORDER:
                sys_prompt = TOOL_SPECS[tool_name]
                try:
                    obj = _make_tool_target(tool_name, q, ctx, candidates, evidence, choices)
                except Exception as e:
                    raise RuntimeError(
                        f"Tool SFT target generation failed: eid={eid} tool={tool_name} "
                        f"{type(e).__name__}: {e}"
                    ) from e
                user = _tool_input_user_message(tool_name, eid, q, ctx, choices)
                row = {
                    "example_id": eid,
                    "prompt": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
                    "response": dumps_json(obj),
                }
                if is_dev:
                    out_map[tool_name][1].append(row)
                else:
                    out_map[tool_name][0].append(row)

    total_train = len(train_rows)
    total_dev = len(dev_rows)
    if synth_mode == "gpt":
        print(f"[GPT SYNTH] will make ~{total_train * variants_train * len(TOOL_ORDER) + total_dev * variants_dev * len(TOOL_ORDER)} API calls")

    for i, ex in enumerate(sorted(train_rows, key=lambda item: int(item["example_id"]))):
        add_one(ex, variants_train, is_dev=False)
        if synth_mode == "gpt" and (i + 1) % 25 == 0:
            print(f"[GPT SYNTH] train progress: {i+1}/{total_train}")
    for i, ex in enumerate(sorted(dev_rows, key=lambda item: int(item["example_id"]))):
        add_one(ex, variants_dev, is_dev=True)
        if synth_mode == "gpt" and (i + 1) % 25 == 0:
            print(f"[GPT SYNTH] dev progress: {i+1}/{total_dev}")

    ensure_dir(out_dir)
    file_map = {}
    for tool_name in TOOL_ORDER:
        train_rows, dev_rows = out_map[tool_name]
        train_path = os.path.join(out_dir, f"{tool_name}_train.jsonl")
        dev_path = os.path.join(out_dir, f"{tool_name}_dev.jsonl")
        write_jsonl(train_path, train_rows)
        write_jsonl(dev_path, dev_rows)
        print(f"[TOOL SFT DATA] {tool_name}: train/dev = {len(train_rows)} / {len(dev_rows)}")
        file_map[tool_name] = (train_path, dev_path)
    return file_map


# =========================================================
# SFT tokenization + training
# =========================================================
def tokenize_sft_dataset(ds: Dataset, tokenizer: Any, max_seq_len: int) -> Dataset:
    eos = tokenizer.eos_token or ""

    def _normalize_response_messages(response: Any) -> List[Dict[str, Any]]:
        if isinstance(response, str):
            return [{"role": "assistant", "content": response}]
        if isinstance(response, dict):
            msg = dict(response)
            msg.setdefault("role", "assistant")
            return [msg]
        if isinstance(response, list):
            out = []
            for item in response:
                if not isinstance(item, dict):
                    raise ValueError(f"Unsupported resp type: {type(item)}")
                msg = dict(item)
                msg.setdefault("role", "assistant")
                out.append(msg)
            return out
        raise ValueError(f"Unsupported resp type: {type(response)}")

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt_msgs = ex["prompt"]
        response_msgs = _normalize_response_messages(ex["response"])
        prompt_text = render_chat_messages(tokenizer, prompt_msgs, add_generation_prompt=True)
        full_text = render_chat_messages(tokenizer, prompt_msgs + response_msgs, add_generation_prompt=False) + eos
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full = tokenizer(full_text, add_special_tokens=False)
        input_ids = full["input_ids"][:max_seq_len]
        attention_mask = full["attention_mask"][:max_seq_len]
        prompt_len = min(len(prompt_ids), max_seq_len)
        labels = ([-100] * prompt_len) + input_ids[prompt_len:]
        labels = labels[:max_seq_len]
        if len(labels) < len(input_ids):
            labels += [-100] * (len(input_ids) - len(labels))
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return ds.map(_map, remove_columns=ds.column_names)


def train_sft_agent(
    tool_base_model: str,
    train_jsonl: str,
    dev_jsonl: str,
    out_dir: str,
    seed: int = 42,
    max_seq_len: int = 2048,
    lr: float = 2e-4,
    epochs: int = 2,
    per_device_bs: int = 1,
    grad_accum: int = 8,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> None:
    require_datasets("train_sft_agent")
    set_seed(seed)
    validate_distributed_runtime("train_sft_agent", require_cuda=(get_world_size() > 1))

    tok = AutoTokenizer.from_pretrained(tool_base_model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        tool_base_model, torch_dtype=runtime_dtype(), trust_remote_code=True,
    )
    model.config.use_cache = False

    if use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft required for LoRA.")
        common = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        present = set(name.split(".")[-1] for name, _ in model.named_modules())
        target_modules = [m for m in common if m in present] or ["q_proj", "v_proj"]
        lconf = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=target_modules,
        )
        model = get_peft_model(model, lconf)
        if is_main_process():
            print(f"[LoRA] target_modules={target_modules}")

    ds = load_dataset("json", data_files={"train": train_jsonl, "validation": dev_jsonl})
    train_ds = tokenize_sft_dataset(ds["train"], tok, max_seq_len=max_seq_len)
    dev_ds = tokenize_sft_dataset(ds["validation"], tok, max_seq_len=max_seq_len)

    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, return_tensors="pt")
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        seed=seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds, data_collator=collator)
    trainer.train()
    ensure_dir(out_dir)
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    if is_main_process():
        print(f"[SFT] saved to: {out_dir}")


# =========================================================
# Tool runtime
# =========================================================
TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*.+?\s*</tool_call>", re.IGNORECASE | re.DOTALL)
TOOLS_TAG_RE = re.compile(r"<tools>.*?</tools>", re.IGNORECASE | re.DOTALL)
TOOL_CALLS_FIELD_RE = re.compile(r'"tool_calls"\s*:', re.IGNORECASE)
# FIX: require ( or { after tool name to avoid false positives on explanations
PLAIN_TOOL_NAME_RE = re.compile(
    r"^\s*(fast_solver_tool|deep_reasoner_tool|answer_critic_tool)\s*[\(\{]",
    re.IGNORECASE | re.MULTILINE,
)

ID2EX: Dict[int, Dict[str, Any]] = {}
TOOL_CACHE: Dict[str, Dict[int, str]] = {name: {} for name in TOOL_ORDER}
TOOL_RAW_CACHE: Dict[str, Dict[int, str]] = {name: {} for name in TOOL_ORDER}
ALLOWED_TOOL_IDS: Optional[set] = None
FAIL_BUFFER_JSONL: Optional[str] = None
RAW_TRACE_JSONL: Optional[str] = None


def _append_raw_trace_rows(rows: List[Dict[str, Any]]) -> None:
    if not rows or not RAW_TRACE_JSONL:
        return
    if not is_main_process():
        return
    append_jsonl_locked(RAW_TRACE_JSONL, rows)


def final_has_tool_call_artifacts(text: str) -> bool:
    if not text:
        return False
    txt = str(text)
    return bool(
        TOOL_CALL_TAG_RE.search(txt)
        or TOOLS_TAG_RE.search(txt)
        or TOOL_CALLS_FIELD_RE.search(txt)
        or PLAIN_TOOL_NAME_RE.search(txt)
    )


def ensure_list(x: Any, n: int) -> List[Any]:
    if isinstance(x, list):
        if len(x) == n:
            return x
        if len(x) == 0:
            return [None] * n
        return (x * ((n // len(x)) + 1))[:n]
    return [x] * n


def extract_stats(completion_msgs: Any) -> Dict[str, Any]:
    if not isinstance(completion_msgs, list):
        txt = _message_content_to_text(completion_msgs)
        has_tool_text = final_has_tool_call_artifacts(txt)
        return {
            "assistant_texts": [txt],
            "tool_msg_count": 0,
            "tool_call_count": 0,
            "tool_names": [],
            "tool_payloads": [],
            "tool_call_names": [],
            "tool_call_payloads": [],
            "last_assistant_text": txt,
            "last_assistant_has_tool_calls": False,
            "fake_tool_text_attempt": bool(has_tool_text),
        }
    assistant_msgs = [m for m in completion_msgs if isinstance(m, dict) and m.get("role") == "assistant"]
    tool_msgs = [m for m in completion_msgs if isinstance(m, dict) and m.get("role") == "tool"]
    tool_msg_count = len(tool_msgs)

    tool_call_count = 0
    tool_call_names = []
    tool_call_payloads = []
    for m in assistant_msgs:
        tc = m.get("tool_calls")
        if isinstance(tc, list):
            tool_call_count += len(tc)
            for item in tc:
                if isinstance(item, dict):
                    fn = item.get("function", {})
                    tool_call_names.append(str(fn.get("name", "")))
                    tool_call_payloads.append(str(fn.get("arguments", "")))

    tool_names = []
    tool_payloads = []
    for m in tool_msgs:
        tool_names.append("" if m.get("name") is None else str(m.get("name")))
        tool_payloads.append(_message_content_to_text(m.get("content")))

    assistant_texts = [_message_content_to_text(m.get("content")) for m in assistant_msgs]
    last_assistant_text = assistant_texts[-1] if assistant_texts else ""
    last_assistant_has_tool_calls = bool(assistant_msgs[-1].get("tool_calls")) if assistant_msgs else False
    any_tool_artifacts_anywhere = any(final_has_tool_call_artifacts(t) for t in assistant_texts)
    fake_tool_text_attempt = bool(any_tool_artifacts_anywhere and (tool_msg_count == 0 and tool_call_count == 0))

    return {
        "assistant_texts": assistant_texts,
        "tool_msg_count": tool_msg_count,
        "tool_call_count": tool_call_count,
        "tool_names": tool_names,
        "tool_payloads": tool_payloads,
        "tool_call_names": tool_call_names,
        "tool_call_payloads": tool_call_payloads,
        "last_assistant_text": last_assistant_text,
        "last_assistant_has_tool_calls": last_assistant_has_tool_calls,
        "fake_tool_text_attempt": fake_tool_text_attempt,
    }


@dataclass
class FrozenAgent:
    tool_base_model: str
    adapter_path: Optional[str] = None
    device: str = "cpu"
    max_new_tokens: int = 512

    def __post_init__(self):
        self.tok = AutoTokenizer.from_pretrained(self.tool_base_model, trust_remote_code=True)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.tok.padding_side = "left"
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            self.tool_base_model, torch_dtype=dtype, trust_remote_code=True,
        ).to(self.device)
        if self.adapter_path:
            if not PEFT_AVAILABLE:
                raise RuntimeError("peft not available.")
            model = PeftModel.from_pretrained(model, self.adapter_path).to(self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

    @torch.no_grad()
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        prompt = render_chat_messages(self.tok, messages, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        do_sample = (temperature > 1e-6)
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tok.pad_token_id,
            "eos_token_id": self.tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(float(temperature), 1e-6)
        out = self.model.generate(**inputs, **gen_kwargs)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


@dataclass
class SharedToolBase:
    """Multiple adapters on a single base model, switched at generation time."""
    tool_base_model: str
    adapter_paths: Dict[str, Optional[str]]
    device: str = "cpu"

    def __post_init__(self):
        if not PEFT_AVAILABLE:
            raise RuntimeError("SharedToolBase requires peft.")
        self.tok = AutoTokenizer.from_pretrained(self.tool_base_model, trust_remote_code=True)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.tok.padding_side = "left"

        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        base_model = AutoModelForCausalLM.from_pretrained(
            self.tool_base_model, torch_dtype=dtype, trust_remote_code=True,
        ).to(self.device)

        primary_name, primary_path = None, None
        for name in TOOL_ORDER:
            p = self.adapter_paths.get(name)
            if p:
                primary_name, primary_path = name, p
                break
        if primary_path is None:
            raise RuntimeError("SharedToolBase requires at least one adapter path.")

        model = PeftModel.from_pretrained(base_model, primary_path, adapter_name=primary_name).to(self.device)
        self.adapter_names: Dict[str, Optional[str]] = {}
        for name in TOOL_ORDER:
            p = self.adapter_paths.get(name)
            if not p:
                self.adapter_names[name] = None
            elif p == primary_path:
                self.adapter_names[name] = primary_name
            else:
                model.load_adapter(p, adapter_name=name)
                self.adapter_names[name] = name

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.lock = threading.Lock()

    @torch.no_grad()
    def generate(self, tool_name: str, messages: List[Dict[str, str]], max_new_tokens: int, temperature: float = 0.0) -> str:
        adapter_name = self.adapter_names.get(tool_name)
        if adapter_name is None:
            raise RuntimeError(f"Adapter for `{tool_name}` not loaded.")
        prompt = render_chat_messages(self.tok, messages, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        do_sample = (temperature > 1e-6)
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": self.tok.pad_token_id,
            "eos_token_id": self.tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(float(temperature), 1e-6)
        with self.lock:
            self.model.set_adapter(adapter_name)
            out = self.model.generate(**inputs, **gen_kwargs)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


@dataclass
class SharedToolView:
    shared_base: SharedToolBase
    tool_name: str
    max_new_tokens: int = 512

    @torch.no_grad()
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        return self.shared_base.generate(self.tool_name, messages, self.max_new_tokens, temperature)


_shared_tool_base: Optional[SharedToolBase] = None
_tool_agents: Dict[str, Any] = {}


def init_tool_agents(tool_base_models: Dict[str, str], adapter_paths: Dict[str, str], device: str):
    global _shared_tool_base, _tool_agents
    shared_model_names = {str(tool_base_models.get(name, "")).strip() for name in TOOL_ORDER}
    can_share = bool(
        PEFT_AVAILABLE
        and all(adapter_paths.get(name) for name in TOOL_ORDER)
        and len(shared_model_names) == 1
        and "" not in shared_model_names
    )
    if can_share and _shared_tool_base is None:
        try:
            shared_model_name = next(iter(shared_model_names))
            _shared_tool_base = SharedToolBase(
                tool_base_model=shared_model_name,
                adapter_paths=adapter_paths,
                device=device,
            )
            _tool_agents = {}
            for tool_name in TOOL_ORDER:
                info = TOOL_REGISTRY.get(tool_name, {})
                max_new = int(info.get("max_new_tokens", 420))
                _tool_agents[tool_name] = SharedToolView(
                    _shared_tool_base, tool_name, max_new_tokens=max_new
                )
            if is_main_process():
                print("[TOOLS] runtime=shared_base adapters=" + ",".join(TOOL_ORDER))
            return
        except Exception as e:
            if is_main_process():
                print(f"[WARN] shared base failed -> split models. {type(e).__name__}: {e}")

    _tool_agents = {}
    for tool_name in TOOL_ORDER:
        info = TOOL_REGISTRY.get(tool_name, {})
        max_new = int(info.get("max_new_tokens", 420))
        _tool_agents[tool_name] = FrozenAgent(
            tool_base_model=str(tool_base_models.get(tool_name, "")).strip(),
            adapter_path=adapter_paths.get(tool_name),
            device=device,
            max_new_tokens=max_new,
        )
    if is_main_process():
        print("[TOOLS] runtime=split_models")


def _tool_guard(eid: int) -> Optional[str]:
    if ALLOWED_TOOL_IDS is not None and eid not in ALLOWED_TOOL_IDS:
        return dumps_json({"error": f"example_id {eid} not allowed in current split"})
    return None


def _tool_input_user_message(tool_name: str, eid: int, q: str, ctx: str, choices: Dict[str, str]) -> str:
    """Build the user message for a tool call. New tools care about q + choices, not context sentences."""
    choice_lines = ""
    if choices:
        choice_lines = "\n".join([f"  {k}. {v}" for k, v in sorted(choices.items())])
        choice_lines = f"\nCANDIDATE ANSWERS:\n{choice_lines}\n"

    if tool_name == "answer_critic_tool":
        # The critic needs a currently-favored answer. Default to the first choice
        # at SFT time (teacher can still override). At runtime the manager should
        # include the favored answer in the question or pass it via the prompt.
        favored = sorted(choices.keys())[0] if choices else "A"
        return f"Example ID: {eid}\nQUESTION:\n{q}\n{choice_lines}\nCURRENTLY_FAVORED: {favored}\n"

    return f"Example ID: {eid}\nQUESTION:\n{q}\n{choice_lines}"


def _fallback_tool_output(tool_name: str, q: str, ctx: str, candidates: List[Dict[str, Any]], evidence: List[Dict[str, Any]], choices: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    obj = _weak_tool_target(tool_name, q, ctx, candidates, evidence, choices=choices)
    # Explicitly signal invalid output
    if tool_name == "fast_solver_tool":
        obj["confidence"] = 0.0
    elif tool_name == "deep_reasoner_tool":
        obj["remaining_uncertainty"] = "Tool output failed to parse; answer unavailable."
        obj["confidence"] = 0.0
    elif tool_name == "answer_critic_tool":
        obj["favored_answer_weaknesses"] = (obj.get("favored_answer_weaknesses") or []) + ["invalid_tool_output"]
        obj["confidence"] = 0.0
        obj["confidence_that_favored_is_wrong"] = 0.0
    return obj


def _normalize_tool_output(tool_name: str, obj: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "fast_solver_tool":
        return _normalize_fast_solver(obj)
    if tool_name == "deep_reasoner_tool":
        return _normalize_deep_reasoner(obj)
    if tool_name == "answer_critic_tool":
        return _normalize_answer_critic(obj)
    raise ValueError(f"Unknown tool_name={tool_name}")


def _run_tool(tool_name: str, example_id: int) -> str:
    eid = int(example_id)
    guard = _tool_guard(eid)
    if guard is not None:
        TOOL_RAW_CACHE[tool_name][eid] = guard
        return guard
    if eid in TOOL_CACHE[tool_name]:
        return TOOL_CACHE[tool_name][eid]

    ex = ID2EX.get(eid)
    if ex is None:
        out = dumps_json({"error": "example_id not found"})
        TOOL_CACHE[tool_name][eid] = out
        TOOL_RAW_CACHE[tool_name][eid] = out
        return out

    q, ctx = ex["question"], ex["context"]
    choices = ex.get("choices", {}) or {}
    rng = random.Random(hash((tool_name, eid)) % (10**9))
    candidates = build_candidates(q, ctx, top_k=20, rng=rng)
    evidence = pick_evidence(candidates, 3, 6, rng)
    sys_prompt = TOOL_SPECS[tool_name]
    user_msg = _tool_input_user_message(tool_name, eid, q, ctx, choices)
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}]
    raw = _tool_agents[tool_name].generate(msgs, temperature=0.0) if tool_name in _tool_agents else ""
    TOOL_RAW_CACHE[tool_name][eid] = raw
    obj = extract_first_json(raw)
    if obj is None:
        obj = _fallback_tool_output(tool_name, q, ctx, candidates, evidence, choices=choices)
    obj = _normalize_tool_output(tool_name, obj)
    out = dumps_json(obj)
    TOOL_CACHE[tool_name][eid] = out

    _append_raw_trace_rows([{
        "ts": int(time.time()),
        "agent": tool_name,
        "event": "tool_call",
        "example_id": eid,
        "raw_output": raw,
        "normalized_output": out,
    }])
    return out


def fast_solver_tool(example_id: int) -> str:
    """Quick small-model solver. Fast and cheap.

    Returns a top guess + brief rationale + a few alternatives. Use for easy
    questions or to generate initial candidates that other tools can critique.

    Args:
        example_id: dataset example id from the user message.

    Returns:
        JSON string with top_guess, top_guess_rationale, alternative_guesses, confidence.
    """
    return _run_tool("fast_solver_tool", example_id)


def deep_reasoner_tool(example_id: int) -> str:
    """Careful large-model reasoner. Slow and expensive.

    Systematically analyzes each candidate answer, compares them, and identifies
    the most defensible. Use for hard clinical reasoning across multiple plausible
    candidates.

    Args:
        example_id: dataset example id from the user message.

    Returns:
        JSON string with candidate_analyses, most_defensible, key_discriminators,
        remaining_uncertainty, confidence.
    """
    return _run_tool("deep_reasoner_tool", example_id)


def answer_critic_tool(example_id: int) -> str:
    """Adversarial answer critic. Tries to falsify the currently-favored answer,
    find overlooked features, suggest better alternatives.

    Use AFTER you have a top guess and want to stress-test it. Do not use before
    you have any candidate.

    Args:
        example_id: dataset example id from the user message.

    Returns:
        JSON string with favored_answer_weaknesses, better_alternatives,
        clinical_features_overlooked, would_change_answer_if, confidence_that_favored_is_wrong.
    """
    return _run_tool("answer_critic_tool", example_id)


# =========================================================
# Manager prompt with belief state
# =========================================================
BELIEF_PREFIX = "BELIEF_STATE:"


def build_manager_system_prompt() -> str:
    if TASK_NAME == "medqa":
        task_line = "You are a manager agent solving medical multiple-choice questions."
    elif TASK_NAME == "medxpertqa_text":
        task_line = "You are a manager agent solving expert-level medical multiple-choice questions."
    elif TASK_NAME == "pubmedqa":
        task_line = "You are a manager agent solving PubMedQA-style clinical yes/no/maybe questions."
    else:
        task_line = "You are a manager agent solving question-answering tasks."

    answer_lines = "\n".join([f"  ANSWER_{ANSWER_CANONICAL_TO_TOKEN[lab]}" for lab in ANSWER_LABELS])

    # Description-based tool guidance, built from TOOL_REGISTRY
    tool_guidance_lines = []
    for name, info in TOOL_REGISTRY.items():
        tool_guidance_lines.append(
            f"- {name} (cost={info['cost']:.1f}, capabilities={info['capabilities']}):\n"
            f"    {info['description']}\n"
            f"    WHEN TO USE: {info['when_to_use']}\n"
            f"    WHEN NOT TO USE: {info['when_not_to_use']}"
        )
    tool_guidance_block = "\n".join(tool_guidance_lines)
    tool_name_list = ", ".join(TOOL_REGISTRY.keys())

    return (
        task_line + "\n\n"
        f"Tools available (native tool-calling only): {tool_name_list}\n"
        f"You may call up to {MAX_MANAGER_TOOL_CALLS} tools total. Each tool has a cost — "
        f"minimizing cost while staying accurate is part of your objective.\n\n"
        f"{tool_guidance_block}\n\n"
        "Before every tool call and before your final answer, emit a BELIEF_STATE JSON block:\n"
        "BELIEF_STATE:\n"
        "{\n"
        '  "problem_type_guess": str,           // e.g. "simple_recall", "differential_dx", "drug_dosing"\n'
        '  "required_capabilities": list[str],  // e.g. ["knowledge_recall", "candidate_comparison"]\n'
        '  "known_facts": list[str],\n'
        '  "current_hypotheses": list[str],     // candidate answers you are considering\n'
        '  "favored_answer": str,               // your current best guess, if any\n'
        '  "uncertainties": list[str],\n'
        '  "tools_called": list[str],\n'
        '  "remaining_budget": int,\n'
        '  "recommended_next_tool": str,        // the tool name you plan to call next, or "answer"\n'
        '  "why_this_tool": str                  // brief justification\n'
        "}\n\n"
        "Routing rules:\n"
        "- Match required_capabilities to tool capabilities when choosing.\n"
        "- Prefer cheaper tools when cheap tools suffice. Save expensive tools for hard cases.\n"
        "- answer_critic_tool requires a favored_answer first; do not call it blindly.\n"
        "- Avoid calling the same tool twice.\n"
        "- Answer directly when confident to save cost.\n"
        "- Never write <tool_call> tags or tool JSON in plain text. Use native tool calls only.\n"
        "- Tool arguments: pass the exact Example ID from the user message.\n\n"
        "The final answer's last line must be exactly one of:\n"
        f"{answer_lines}\n"
        "Nothing after that line.\n"
    )


MANAGER_SYSTEM = build_manager_system_prompt()


def _format_choices_block(choices: Optional[Dict[str, str]]) -> str:
    if not isinstance(choices, dict) or not choices:
        return ""
    items = _sorted_choice_items(choices)
    if not items:
        return ""
    return "Choices:\n" + "\n".join([f"{k}. {v}" for k, v in items]) + "\n\n"


def build_initial_belief_state(eid: int, q: str, ctx: str) -> Dict[str, Any]:
    q_words = tokenize_words(q)
    ctx_words = tokenize_words(ctx)
    return {
        "problem_type_guess": "multiple_choice_medical" if len(ANSWER_LABELS) >= 4 else "clinical_qa",
        "required_capabilities": ["candidate_generation"],
        "known_facts": [f"example_id={eid}", f"question_words={len(q_words)}", f"context_words={len(ctx_words)}"],
        "current_hypotheses": [],
        "favored_answer": "",
        "uncertainties": ["Have not yet analyzed the question."],
        "tools_called": [],
        "remaining_budget": MAX_MANAGER_TOOL_CALLS,
        "recommended_next_tool": "fast_solver_tool",
        "why_this_tool": "Start with a cheap candidate generation pass before committing to expensive reasoning.",
    }


def build_manager_messages(
    eid: int, q: str, ctx: str, choices: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    choices_block = _format_choices_block(choices)
    belief = build_initial_belief_state(eid, q, ctx)
    user_text = (
        f"Example ID: {eid}\n\n"
        f"Question:\n{q}\n\n"
        f"{choices_block}"
        f"Context:\n{ctx}\n\n"
        f"Current Planner Memory / Belief State:\n{dumps_json(belief)}\n\n"
        "Decide the next best action. If you call a tool, pass the exact Example ID shown above.\n"
        "Before any tool call or before the final answer, write a BELIEF_STATE block first.\n"
    )
    return [
        {"role": "system", "content": MANAGER_SYSTEM},
        {"role": "user", "content": user_text},
    ]


def parse_belief_state_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    idx = text.find(BELIEF_PREFIX)
    if idx < 0:
        return None
    sub = text[idx + len(BELIEF_PREFIX):].strip()
    obj = extract_first_json(sub)
    if not isinstance(obj, dict):
        return None
    required = {
        "problem_type_guess", "known_facts", "current_hypotheses", "uncertainties",
        "tools_called", "remaining_budget", "recommended_next_tool", "why_this_tool",
    }
    # Accept partial belief states too, but reward full ones more
    return obj


# =========================================================
# FIX: shaped reward with correctness dominance
# =========================================================
def parse_answer_label_lastline(text: str) -> Optional[str]:
    if not text:
        return None
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return None
    last_line = lines[-1]
    m = ANSWER_LASTLINE_RE.match(last_line)
    if not m:
        return None
    tok = m.group(1).upper()
    return ANSWER_TOKEN_TO_CANONICAL.get(tok)


def _route_pattern(tool_call_names: List[str]) -> str:
    if not tool_call_names:
        return "direct"
    return "->".join([x for x in tool_call_names if x])


def _tool_diversity_bonus(names: List[str]) -> float:
    uniq = [x for x in names if x]
    if not uniq:
        return 0.0
    uniq_set = set(uniq)
    if len(uniq_set) == 1:
        return 0.0
    return 0.05 * float(len(uniq_set) - 1)


def _repeated_tool_penalty(names: List[str]) -> float:
    if not names:
        return 0.0
    c = Counter([x for x in names if x])
    penalty = 0.0
    for _, v in c.items():
        if v > 1:
            penalty += 0.08 * float(v - 1)
    return penalty


def _budget_penalty(tool_call_count: int) -> float:
    if tool_call_count <= MAX_MANAGER_TOOL_CALLS:
        return 0.0
    return 0.20 * float(tool_call_count - MAX_MANAGER_TOOL_CALLS)


def _belief_quality_bonus(belief_obj: Optional[Dict[str, Any]]) -> float:
    """Cap belief bonus so it can't dominate correctness signal.
    Max 0.10 total across all sub-bonuses."""
    if belief_obj is None:
        return 0.0
    bonus = 0.03
    if isinstance(belief_obj.get("uncertainties"), list) and len(belief_obj.get("uncertainties")) > 0:
        bonus += 0.01
    if isinstance(belief_obj.get("known_facts"), list) and len(belief_obj.get("known_facts")) > 0:
        bonus += 0.01
    if isinstance(belief_obj.get("current_hypotheses"), list) and len(belief_obj.get("current_hypotheses")) > 0:
        bonus += 0.02
    if isinstance(belief_obj.get("required_capabilities"), list) and len(belief_obj.get("required_capabilities")) > 0:
        bonus += 0.03  # research-specific: reward explicit capability naming
    return min(bonus, 0.10)


def _recommended_tool_alignment_bonus(belief_obj: Optional[Dict[str, Any]], route_names: List[str]) -> float:
    if belief_obj is None or not route_names:
        return 0.0
    rec = str(belief_obj.get("recommended_next_tool", "")).strip()
    if rec and rec == route_names[0]:
        return 0.05
    return 0.0


def _routing_appropriateness_bonus(belief_obj: Optional[Dict[str, Any]], route_names: List[str]) -> float:
    """Reward the manager when its chosen tool's declared capabilities match the
    required_capabilities it claimed to need. This gives dense signal on whether
    the routing decision was 'on-topic'.

    Max 0.10 per call, 0.15 total cap across all calls.
    """
    if belief_obj is None or not route_names:
        return 0.0
    required = belief_obj.get("required_capabilities") or []
    if not isinstance(required, list) or not required:
        return 0.0
    required_set = {str(x).strip().lower() for x in required if str(x).strip()}
    if not required_set:
        return 0.0

    bonus = 0.0
    for tool_name in route_names:
        info = TOOL_REGISTRY.get(tool_name)
        if not info:
            continue
        tool_caps = {str(c).strip().lower() for c in info.get("capabilities", [])}
        overlap = len(required_set & tool_caps)
        if overlap > 0:
            bonus += min(0.05 + 0.025 * (overlap - 1), 0.10)
    return min(bonus, 0.15)


def _tool_cost_sum(route_names: List[str]) -> float:
    """Sum of TOOL_REGISTRY costs for all tools called in this completion."""
    total = 0.0
    for nm in route_names:
        info = TOOL_REGISTRY.get(nm)
        if info:
            total += float(info.get("cost", 0.0))
    return total


def _answer_critic_ordering_penalty(route_names: List[str]) -> float:
    """answer_critic_tool requires something to critique. If it's called first
    (nothing to critique yet), penalize."""
    if not route_names:
        return 0.0
    if route_names[0] == "answer_critic_tool":
        return 0.10
    return 0.0


def routing_aware_reward(prompts=None, completions=None, ground_truth=None, example_id=None, **kwargs):
    """Cost-aware shaped reward for the tool-routing research experiment.

    Components (approximate ranges):
      +0.10   valid format (parsed ANSWER_*)
      +1.00   correct label                         <-- dominant
      +0.05   at least one tool call
      +up to 0.15   tool diversity
      +up to 0.10   belief quality
      +0.05         recommended-tool self-consistency
      +up to 0.15   routing appropriateness (caps × overlap)
      -0.01 × sum(tool_cost)                        <-- cost-aware term
      -budget penalty                                (exceeds MAX_MANAGER_TOOL_CALLS)
      -repeated-tool penalty
      -0.10   answer_critic called before any candidate
      -0.25   last turn mixes tool-call artifacts with final answer
      -0.35   fake plaintext tool-call attempt
      -0.05   invalid final format

    Design:
      correctness ($$+1.00$$) dominates all shaping signals,
      but shaping gives dense gradient during cold start.

      cost penalty is small ($$\\approx 0.01$$ per cost-unit): a single
      deep_reasoner_tool call (cost 10) costs the manager 0.10 reward,
      which is worth paying if it flips one wrong answer to correct.
    """
    # cost coefficient — tunable
    COST_COEF = 0.01

    n = len(completions)
    gts = ensure_list(ground_truth, n)
    exids = ensure_list(example_id, n)

    rewards = []
    fail_rows = []
    trace_rows = []

    for c, gt, eid in zip(completions, gts, exids):
        gt = _normalize_label(gt)
        st = extract_stats(c)
        last_text = st["last_assistant_text"]
        pred = parse_answer_label_lastline(last_text)
        valid_format = pred is not None
        final_has_artifacts = bool(
            st["last_assistant_has_tool_calls"] or final_has_tool_call_artifacts(last_text)
        )
        fake_tool_text = bool(st.get("fake_tool_text_attempt"))
        route_names = st.get("tool_call_names", [])
        tool_call_count = int(st["tool_call_count"])
        route_pattern = _route_pattern(route_names)
        tool_cost = _tool_cost_sum(route_names)

        belief_obj = None
        for txt in reversed(st.get("assistant_texts", [])):
            b = parse_belief_state_from_text(txt)
            if b is not None:
                belief_obj = b
                break

        reward = 0.0
        if valid_format:
            reward += 0.10
        correct = bool(valid_format and pred == gt)
        if correct:
            reward += 1.00
        if tool_call_count > 0:
            reward += 0.05
        reward += _tool_diversity_bonus(route_names)
        reward += _belief_quality_bonus(belief_obj)
        reward += _recommended_tool_alignment_bonus(belief_obj, route_names)
        reward += _routing_appropriateness_bonus(belief_obj, route_names)

        # penalties
        reward -= COST_COEF * tool_cost
        reward -= _budget_penalty(tool_call_count)
        reward -= _repeated_tool_penalty(route_names)
        reward -= _answer_critic_ordering_penalty(route_names)
        if final_has_artifacts:
            reward -= 0.25
        if fake_tool_text:
            reward -= 0.35
        if not valid_format:
            reward -= 0.05

        reward = float(max(-1.0, min(1.5, reward)))
        rewards.append(reward)

        row = {
            "ts": int(time.time()),
            "agent": "manager",
            "event": "completion",
            "example_id": int(eid) if eid is not None else None,
            "ground_truth": gt,
            "pred": pred,
            "correct": correct,
            "reward": reward,
            "valid_format": bool(valid_format),
            "route_pattern": route_pattern,
            "tool_call_count": tool_call_count,
            "tool_call_names": route_names,
            "tool_names": st.get("tool_names", []),
            "tool_cost_sum": tool_cost,
            "final_has_tool_artifacts": bool(final_has_artifacts),
            "fake_tool_text_attempt": bool(fake_tool_text),
            "belief_state_present": bool(belief_obj is not None),
            "belief_state": belief_obj,
            "last_assistant_text": last_text,
        }
        trace_rows.append(row)
        if reward < 1.0 and is_main_process() and FAIL_BUFFER_JSONL:
            fail_rows.append(row)

    if fail_rows and is_main_process() and FAIL_BUFFER_JSONL:
        append_jsonl_locked(FAIL_BUFFER_JSONL, fail_rows)
    _append_raw_trace_rows(trace_rows)
    return rewards


# =========================================================
# Routing analysis
# =========================================================
def analyze_routing_trace(raw_trace_jsonl: str, out_json: str = "", out_txt: str = "") -> Dict[str, Any]:
    """Research-grade routing analysis. Computes:
      - accuracy
      - avg_cost_per_correct
      - routing_entropy (over first-tool choices)
      - tool_appropriateness_rate (fraction of correct routes where chosen tool's
        capabilities overlap with required_capabilities from belief)
      - per-route accuracy + cost + count
      - belief_state_present_rate
      - tool usage histograms
    """
    rows = []
    with open(raw_trace_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("agent") == "manager" and obj.get("event") == "completion":
                rows.append(obj)

    if not rows:
        summary = {"num_rows": 0}
    else:
        n = len(rows)
        route_counter = Counter([str(r.get("route_pattern", "unknown")) for r in rows])
        first_tool_counter = Counter()
        by_route_reward: Dict[str, List[float]] = defaultdict(list)
        by_route_correct: Dict[str, List[int]] = defaultdict(list)
        by_route_cost: Dict[str, List[float]] = defaultdict(list)
        tool_counter = Counter()
        belief_present = 0
        n_correct = 0
        total_cost = 0.0
        total_cost_correct = 0.0
        n_appropriate_routes = 0
        n_routes_with_caps = 0

        for r in rows:
            route = str(r.get("route_pattern", "unknown"))
            names = r.get("tool_call_names", []) or []
            cost = float(r.get("tool_cost_sum", 0.0))
            pred = r.get("pred")
            gt = r.get("ground_truth")
            is_correct = pred is not None and pred == gt
            belief = r.get("belief_state") or {}

            by_route_reward[route].append(float(r.get("reward", 0.0)))
            by_route_correct[route].append(int(is_correct))
            by_route_cost[route].append(cost)
            tool_counter.update(names)
            first_tool_counter.update([names[0]] if names else ["direct"])
            belief_present += int(bool(r.get("belief_state_present")))
            total_cost += cost
            if is_correct:
                n_correct += 1
                total_cost_correct += cost

            # routing appropriateness (research metric)
            required = belief.get("required_capabilities") or []
            if isinstance(required, list) and required:
                required_set = {str(x).strip().lower() for x in required if str(x).strip()}
                if required_set:
                    n_routes_with_caps += 1
                    for tn in names:
                        info = TOOL_REGISTRY.get(tn)
                        if not info:
                            continue
                        caps = {str(c).strip().lower() for c in info.get("capabilities", [])}
                        if required_set & caps:
                            n_appropriate_routes += 1
                            break

        # Routing entropy over first-tool choices
        total_first = sum(first_tool_counter.values())
        if total_first > 0:
            probs = [v / total_first for v in first_tool_counter.values()]
            import math as _m
            entropy = -sum(p * _m.log(p + 1e-12) for p in probs)
            # Normalize by log(k) for comparability across tool-pool sizes
            max_ent = _m.log(max(1, len(first_tool_counter)))
            routing_entropy_normalized = entropy / max_ent if max_ent > 0 else 0.0
        else:
            entropy = 0.0
            routing_entropy_normalized = 0.0

        summary = {
            "num_rows": n,
            "accuracy": n_correct / n,
            "avg_cost_per_completion": total_cost / n,
            "avg_cost_per_correct": (total_cost_correct / n_correct) if n_correct else float("inf"),
            "routing_entropy": entropy,
            "routing_entropy_normalized": routing_entropy_normalized,
            "tool_appropriateness_rate": (
                n_appropriate_routes / n_routes_with_caps if n_routes_with_caps else 0.0
            ),
            "belief_state_present_rate": belief_present / n,
            "route_counts": dict(route_counter),
            "first_tool_distribution": dict(first_tool_counter),
            "tool_call_frequency": dict(tool_counter),
            "route_accuracy": {k: sum(v) / max(1, len(v)) for k, v in by_route_correct.items()},
            "route_avg_reward": {k: sum(v) / max(1, len(v)) for k, v in by_route_reward.items()},
            "route_avg_cost": {k: sum(v) / max(1, len(v)) for k, v in by_route_cost.items()},
        }

    if out_json:
        write_json(out_json, summary)
    if out_txt:
        ensure_dir(os.path.dirname(out_txt) or ".")
        lines = [f"num_rows: {summary.get('num_rows', 0)}"]
        if summary.get("num_rows", 0) > 0:
            lines += [
                f"accuracy: {summary['accuracy']:.4f}",
                f"avg_cost_per_completion: {summary['avg_cost_per_completion']:.4f}",
                f"avg_cost_per_correct: {summary['avg_cost_per_correct']:.4f}",
                f"routing_entropy: {summary['routing_entropy']:.4f}",
                f"routing_entropy_normalized: {summary['routing_entropy_normalized']:.4f}",
                f"tool_appropriateness_rate: {summary['tool_appropriateness_rate']:.4f}",
                f"belief_state_present_rate: {summary['belief_state_present_rate']:.4f}",
                "route_counts:",
            ]
            for k, v in sorted(summary.get("route_counts", {}).items(), key=lambda x: -x[1]):
                lines.append(f"  {k}: {v}")
            lines.append("first_tool_distribution:")
            for k, v in sorted(summary.get("first_tool_distribution", {}).items(), key=lambda x: -x[1]):
                lines.append(f"  {k}: {v}")
            lines.append("route_accuracy:")
            for k, v in sorted(summary.get("route_accuracy", {}).items(), key=lambda x: -x[1]):
                lines.append(f"  {k}: {v:.4f}")
            lines.append("route_avg_cost:")
            for k, v in sorted(summary.get("route_avg_cost", {}).items(), key=lambda x: -x[1]):
                lines.append(f"  {k}: {v:.4f}")
            lines.append("route_avg_reward:")
            for k, v in sorted(summary.get("route_avg_reward", {}).items(), key=lambda x: -x[1]):
                lines.append(f"  {k}: {v:.4f}")
            lines.append("tool_call_frequency:")
            for k, v in sorted(summary.get("tool_call_frequency", {}).items(), key=lambda x: -x[1]):
                lines.append(f"  {k}: {v}")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    return summary


# =========================================================
# Manager GRPO
# =========================================================
def train_manager_grpo_from_splits(
    manager_base_model: str,
    tool_base_model: str,
    data_path: str,
    split_path: str,
    save_dir: str,
    extractor_adapter_unused: Optional[str] = None,  # deprecated placeholder
    fast_solver_adapter: str = "",
    deep_reasoner_adapter: str = "",
    answer_critic_adapter: str = "",
    fast_solver_base_model: str = "",
    deep_reasoner_base_model: str = "",
    answer_critic_base_model: str = "",
    seed: int = 42,
    per_device_train_bs: int = 1,
    grad_accum: int = 4,
    max_prompt_length: int = 2048,
    max_completion_length: int = 1024,
    temperature: float = 0.7,
    num_generations: int = 8,
    grpo_beta: float = 0.01,
    fail_buffer_jsonl: str = "",
    raw_trace_jsonl: str = "",
    use_wandb: bool = False,
    wandb_project: str = "agents_as_tools_three_grpo",
    wandb_entity: str = "",
    wandb_run_name: str = "",
    wandb_mode: str = "online",
    use_vllm: bool = False,
    vllm_mode: str = "server",
    vllm_server_base_url: str = "http://127.0.0.1:8000",
    vllm_gpu_memory_utilization: float = 0.35,
    vllm_enable_sleep_mode: bool = False,
    manager_use_lora: bool = True,
    manager_lora_r: int = 16,
    manager_lora_alpha: int = 32,
    manager_lora_dropout: float = 0.05,
    manager_gradient_checkpointing: bool = True,
) -> None:
    require_clean_runtime()
    require_trl("train_manager_grpo")
    require_datasets("train_manager_grpo")
    validate_distributed_runtime(
        "train_manager_grpo", require_cuda=(get_world_size() > 1 or use_vllm), use_vllm=use_vllm,
    )
    validate_grpo_batch_geometry(per_device_train_bs, grad_accum, num_generations)
    set_seed(seed)

    if is_main_process():
        print(
            f"[RUNTIME] world_size={get_world_size()} rank={get_global_rank()} "
            f"local_rank={get_local_rank()} device={runtime_device()}"
        )
        if use_vllm:
            print(f"[VLLM] mode={vllm_mode} base_url={vllm_server_base_url}")

    global FAIL_BUFFER_JSONL, RAW_TRACE_JSONL, ALLOWED_TOOL_IDS
    FAIL_BUFFER_JSONL = (fail_buffer_jsonl.strip() or os.path.join(save_dir, "fail_buffer.jsonl"))
    RAW_TRACE_JSONL = (raw_trace_jsonl.strip() or os.path.join(save_dir, "train_raw_trace.jsonl"))

    splits = read_json(split_path)
    validate_split_scope(splits, data_path)
    train_rows = get_split_examples(splits, "train")
    if not train_rows:
        raise RuntimeError(
            "Split file must contain embedded train_examples. "
            "Remake splits with the current make_splits stage."
        )
    train_ids = {int(r["example_id"]) for r in train_rows}

    ID2EX.clear()
    for r in train_rows:
        ID2EX[int(r["example_id"])] = {"question": r["question"], "context": r["context"], "choices": r.get("choices", {})}

    ALLOWED_TOOL_IDS = set(train_ids)
    for tool_name in TOOL_ORDER:
        TOOL_CACHE[tool_name].clear()
        TOOL_RAW_CACHE[tool_name].clear()

    adapter_paths = {
        "fast_solver_tool": fast_solver_adapter,
        "deep_reasoner_tool": deep_reasoner_adapter,
        "answer_critic_tool": answer_critic_adapter,
    }
    tool_base_models = resolve_tool_base_models_for_stage(
        stage_name="train_manager_grpo",
        default_tool_base_model=tool_base_model,
        fast_solver_base_model=fast_solver_base_model,
        deep_reasoner_base_model=deep_reasoner_base_model,
        answer_critic_base_model=answer_critic_base_model,
    )
    if is_main_process():
        print(f"[TOOLS] base_models={tool_base_models}")
    init_tool_agents(tool_base_models, adapter_paths=adapter_paths, device=runtime_device())

    if use_wandb:
        try:
            import wandb  # noqa
        except Exception as e:
            raise RuntimeError("wandb not installed.") from e
        if wandb_project.strip():
            os.environ["WANDB_PROJECT"] = wandb_project.strip()
        if wandb_entity.strip():
            os.environ["WANDB_ENTITY"] = wandb_entity.strip()
        os.environ["WANDB_MODE"] = (wandb_mode or "online").strip().lower()
        _save_dir_tag = os.path.basename(save_dir.rstrip("/\\"))
        run_name = wandb_run_name.strip() or f"grpo3_{_save_dir_tag}_{int(time.time())}"
        os.environ["WANDB_NAME"] = run_name
        if is_main_process():
            print(f"[WANDB] project={os.environ.get('WANDB_PROJECT','')} run={run_name}")
    else:
        os.environ.setdefault("WANDB_DISABLED", "true")

    if is_main_process():
        ensure_dir(save_dir)
        for p in [FAIL_BUFFER_JSONL, RAW_TRACE_JSONL]:
            ensure_dir(os.path.dirname(p) or ".")
            if os.environ.get("LOG_APPEND", "0") == "0":
                with open(p, "w", encoding="utf-8"):
                    pass
        print(f"[LOGS] fail_buffer -> {FAIL_BUFFER_JSONL}")
        print(f"[LOGS] raw_trace   -> {RAW_TRACE_JSONL}")

    if get_world_size() > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    manager_tok = AutoTokenizer.from_pretrained(manager_base_model, trust_remote_code=True)
    manager_tok.padding_side = "left"
    if manager_tok.pad_token_id is None and manager_tok.eos_token_id is not None:
        manager_tok.pad_token_id = manager_tok.eos_token_id

    dataset = Dataset.from_list(sorted(train_rows, key=lambda item: int(item["example_id"])))

    def preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        eid = int(ex["example_id"])
        msgs = build_manager_messages(eid, ex["question"], ex["context"], choices=ex.get("choices", {}))
        return {
            "prompt": msgs,
            "ground_truth": ex["ground_truth"],
            "example_id": eid,
            "raw_id": ex.get("raw_id", str(eid)),
        }

    train_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    grpo_kwargs = {
        "output_dir": save_dir,
        "remove_unused_columns": False,
        "per_device_train_batch_size": int(per_device_train_bs),
        "gradient_accumulation_steps": int(grad_accum),
        "max_prompt_length": int(max_prompt_length),
        "max_completion_length": int(max_completion_length),
        "num_generations": int(num_generations),
        "temperature": float(temperature),
        "do_sample": True,
        "beta": float(grpo_beta),
        "scale_rewards": "group",
        "bf16": torch.cuda.is_available(),
        "logging_steps": 1,
        "log_completions": True,
        "num_completions_to_print": None,
        "log_unique_prompts": False,
        "report_to": (["wandb"] if use_wandb else []),
        "max_tool_calling_iterations": int(MAX_MANAGER_TOOL_CALLS),
        "chat_template_kwargs": {"enable_thinking": False},
        "gradient_checkpointing": bool(manager_gradient_checkpointing),
        "use_vllm": bool(use_vllm),
    }
    if use_vllm:
        grpo_kwargs.update({
            "vllm_mode": vllm_mode,
            "vllm_server_base_url": vllm_server_base_url,
            "vllm_gpu_memory_utilization": float(vllm_gpu_memory_utilization),
            "vllm_enable_sleep_mode": bool(vllm_enable_sleep_mode),
        })

    grpo_args = GRPOConfig(**_filter_supported_kwargs(GRPOConfig.__init__, grpo_kwargs, "GRPOConfig"))

    manager_model = AutoModelForCausalLM.from_pretrained(
        manager_base_model, torch_dtype=runtime_dtype(), trust_remote_code=True,
    )
    manager_model.config.use_cache = False
    manager_model.generation_config = GenerationConfig.from_model_config(manager_model.config)
    manager_model.generation_config.do_sample = True
    manager_model.generation_config.temperature = float(temperature)
    manager_model.generation_config.top_p = 1.0
    manager_model.generation_config.top_k = 0
    manager_model.generation_config.pad_token_id = manager_tok.pad_token_id
    manager_model.generation_config.eos_token_id = manager_tok.eos_token_id
    if not hasattr(manager_model, "warnings_issued") or manager_model.warnings_issued is None:
        manager_model.warnings_issued = {}

    if manager_use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("--mgr_use_lora needs peft.")
        common = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        present = set(name.split(".")[-1] for name, _ in manager_model.named_modules())
        target_modules = [m for m in common if m in present] or ["q_proj", "v_proj"]
        lconf = LoraConfig(
            r=manager_lora_r, lora_alpha=manager_lora_alpha, lora_dropout=manager_lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=target_modules,
        )
        manager_model = get_peft_model(manager_model, lconf)
        if is_main_process():
            print(f"[MANAGER LoRA] target_modules={target_modules}")

    if manager_gradient_checkpointing:
        try:
            manager_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            manager_model.gradient_checkpointing_enable()
        except Exception:
            pass
        if hasattr(manager_model, "enable_input_require_grads"):
            try:
                manager_model.enable_input_require_grads()
            except Exception:
                pass

    trainer_kwargs = {
        "model": manager_model,
        "args": grpo_args,
        "train_dataset": train_dataset,
        "reward_funcs": [routing_aware_reward],
        "rollout_func": None,
        "tools": [fast_solver_tool, deep_reasoner_tool, answer_critic_tool],
    }
    trainer_kwargs.update(_trainer_processing_kwargs(manager_tok))
    if is_main_process():
        print("[TRAINER] native tools =", TOOL_ORDER)

    trainer = GRPOTrainer(**_filter_supported_kwargs(GRPOTrainer.__init__, trainer_kwargs, "GRPOTrainer"))
    trainer.train()

    if is_main_process():
        ensure_dir(save_dir)
        trainer.model.save_pretrained(save_dir)
        manager_tok.save_pretrained(save_dir)
        print(f"[GRPO] saved manager to: {save_dir}")


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", type=str, required=True,
        choices=[
            "make_splits", "build_tool_sft",
            "train_fast_solver_tool", "train_deep_reasoner_tool",
            "train_answer_critic_tool",
            "train_manager_grpo", "analyze_routing",
        ],
    )
    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--manager_base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tool_base_model", type=str, default="")
    parser.add_argument("--fast_solver_base_model", type=str, default="")
    parser.add_argument("--deep_reasoner_base_model", type=str, default="")
    parser.add_argument("--answer_critic_base_model", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument(
        "--task_name", type=str, default="pubmedqa",
        choices=["pubmedqa", "medqa", "medxpertqa_text", "generic"],
    )
    parser.add_argument("--label_space", type=str, default="")
    parser.add_argument(
        "--medqa_regions", type=str, default="",
        help="Comma-separated MedQA region filter. Example: US or US,Taiwan. Empty means all regions.",
    )

    # split
    parser.add_argument("--split_path", type=str, default="splits_task.json")
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--dev_size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=-1)

    # tool SFT data
    parser.add_argument("--tool_sft_out_dir", type=str, default="tool_sft_data_three")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--tool_variants_train", type=int, default=3)
    parser.add_argument("--tool_variants_dev", type=int, default=2)
    parser.add_argument("--ev_min", type=int, default=3)
    parser.add_argument("--ev_max", type=int, default=6)
    parser.add_argument(
        "--tool_synth_mode", type=str, default="weak", choices=["weak", "gpt"],
        help="weak: heuristic targets (no API). gpt: use teacher API via TEACHER_BASE_URL/TEACHER_API_KEY/TEACHER_MODEL env vars.",
    )
    parser.add_argument("--tool_synth_gpt_temperature", type=float, default=0.2)
    parser.add_argument("--tool_synth_gpt_max_retries", type=int, default=3)

    # tool SFT train
    parser.add_argument("--tool_lr", type=float, default=2e-4)
    parser.add_argument("--tool_epochs", type=int, default=2)
    parser.add_argument("--tool_bs", type=int, default=1)
    parser.add_argument("--tool_grad_accum", type=int, default=8)
    parser.add_argument("--tool_max_seq_len", type=int, default=2048)
    parser.add_argument("--tool_use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast_solver_tool_out", type=str, default="fast_solver_tool_adapter")
    parser.add_argument("--deep_reasoner_tool_out", type=str, default="deep_reasoner_tool_adapter")
    parser.add_argument("--answer_critic_tool_out", type=str, default="answer_critic_tool_adapter")

    # manager GRPO
    parser.add_argument("--manager_out", type=str, default="manager_grpo_three")
    parser.add_argument("--mgr_bs", type=int, default=1)
    parser.add_argument("--mgr_grad_accum", type=int, default=4)
    parser.add_argument("--mgr_max_prompt_length", type=int, default=2048)
    parser.add_argument("--mgr_max_completion_length", type=int, default=1536)
    parser.add_argument("--mgr_temperature", type=float, default=0.7)
    parser.add_argument("--mgr_num_generations", type=int, default=8)
    parser.add_argument("--grpo_beta", type=float, default=0.01)
    parser.add_argument("--fail_buffer_jsonl", type=str, default="")
    parser.add_argument("--raw_trace_jsonl", type=str, default="")
    parser.add_argument("--grpo_use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="agents_as_tools_three_grpo")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline"])
    parser.add_argument("--mgr_use_vllm", action="store_true")
    parser.add_argument("--mgr_vllm_mode", type=str, default="server", choices=["server", "colocate"])
    parser.add_argument("--mgr_vllm_server_base_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--mgr_vllm_gpu_memory_utilization", type=float, default=0.35)
    parser.add_argument("--mgr_vllm_enable_sleep_mode", action="store_true")
    parser.add_argument("--mgr_use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mgr_lora_r", type=int, default=16)
    parser.add_argument("--mgr_lora_alpha", type=int, default=32)
    parser.add_argument("--mgr_lora_dropout", type=float, default=0.05)
    parser.add_argument("--mgr_gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)

    # analyze
    parser.add_argument("--routing_summary_json", type=str, default="")
    parser.add_argument("--routing_summary_txt", type=str, default="")

    args = parser.parse_args()
    configure_cuda_runtime()

    if args.base_model.strip():
        shared = args.base_model.strip()
        args.manager_base_model = shared
        args.tool_base_model = shared

    configured_task, configured_labels = configure_task(args.task_name, args.label_space)
    configured_medqa_regions = configure_medqa_regions(args.medqa_regions)
    if is_main_process():
        print(f"[TASK] task={configured_task} labels={configured_labels}")
        if configured_task == "medqa":
            print(f"[TASK][MedQA] regions={configured_medqa_regions or ['ALL']}")
    data_path = resolve_data_path_arg(args.data_path, configured_task)
    if is_main_process():
        print(f"[DATA] {data_path}")
        print(f"[MODELS] manager={args.manager_base_model} default_tool={args.tool_base_model or '[unset]'}")

    global MANAGER_SYSTEM
    MANAGER_SYSTEM = build_manager_system_prompt()
    set_seed(args.seed)

    if args.stage == "make_splits":
        rows = load_raw_task(data_path)
        print(f"[SPLIT] loaded rows = {len(rows)}")
        if args.max_samples > 0:
            sample_seed = args.seed if args.sample_seed < 0 else args.sample_seed
            rows = subsample_rows(rows, max_samples=args.max_samples, seed=sample_seed)
            print(f"[SPLIT] subsampled = {len(rows)}")
        splits = make_splits(rows, test_size=args.test_size, dev_size=args.dev_size, seed=args.seed)
        splits = attach_split_examples(splits, rows)
        splits["dataset_scope"] = build_split_scope_metadata(data_path, rows)
        write_json(args.split_path, splits)
        print(f"[SPLIT] train/dev/test = {len(splits['train_ids'])}/{len(splits['dev_ids'])}/{len(splits['test_ids'])}")
        print(f"[SPLIT] wrote -> {args.split_path}")
        return

    if args.stage == "build_tool_sft":
        file_map = build_tool_sft_data_from_splits(
            data_path=data_path, split_path=args.split_path, out_dir=args.tool_sft_out_dir,
            seed=args.seed, top_k=args.top_k,
            variants_train=args.tool_variants_train, variants_dev=args.tool_variants_dev,
            ev_min=args.ev_min, ev_max=args.ev_max,
            synth_mode=args.tool_synth_mode,
            gpt_temperature=args.tool_synth_gpt_temperature,
            gpt_max_retries=args.tool_synth_gpt_max_retries,
        )
        print("[TOOL SFT DATA] file_map =", file_map)
        return

    tool_stage_map = {
        "train_fast_solver_tool": ("fast_solver_tool", args.fast_solver_tool_out),
        "train_deep_reasoner_tool": ("deep_reasoner_tool", args.deep_reasoner_tool_out),
        "train_answer_critic_tool": ("answer_critic_tool", args.answer_critic_tool_out),
    }
    if args.stage in tool_stage_map:
        resolved_tool_base_models = resolve_tool_base_models_for_stage(
            stage_name=args.stage,
            default_tool_base_model=args.tool_base_model,
            fast_solver_base_model=args.fast_solver_base_model,
            deep_reasoner_base_model=args.deep_reasoner_base_model,
            answer_critic_base_model=args.answer_critic_base_model,
        )
        if is_main_process():
            print(f"[MODELS][tools] {resolved_tool_base_models}")
        tool_name, out_dir = tool_stage_map[args.stage]
        train_jsonl = os.path.join(args.tool_sft_out_dir, f"{tool_name}_train.jsonl")
        dev_jsonl = os.path.join(args.tool_sft_out_dir, f"{tool_name}_dev.jsonl")
        train_sft_agent(
            tool_base_model=resolved_tool_base_models[tool_name],
            train_jsonl=train_jsonl, dev_jsonl=dev_jsonl, out_dir=out_dir,
            seed=args.seed, max_seq_len=args.tool_max_seq_len,
            lr=args.tool_lr, epochs=args.tool_epochs,
            per_device_bs=args.tool_bs, grad_accum=args.tool_grad_accum,
            use_lora=args.tool_use_lora,
        )
        return

    if args.stage == "train_manager_grpo":
        resolved_tool_base_models = resolve_tool_base_models_for_stage(
            stage_name=args.stage,
            default_tool_base_model=args.tool_base_model,
            fast_solver_base_model=args.fast_solver_base_model,
            deep_reasoner_base_model=args.deep_reasoner_base_model,
            answer_critic_base_model=args.answer_critic_base_model,
        )
        if is_main_process():
            print(f"[MODELS][tools] {resolved_tool_base_models}")
        fb = args.fail_buffer_jsonl.strip() or os.path.join(args.manager_out, "fail_buffer.jsonl")
        rt = args.raw_trace_jsonl.strip() or os.path.join(args.manager_out, "train_raw_trace.jsonl")
        train_manager_grpo_from_splits(
            manager_base_model=args.manager_base_model,
            tool_base_model=args.tool_base_model,
            data_path=data_path, split_path=args.split_path, save_dir=args.manager_out,
            fast_solver_adapter=args.fast_solver_tool_out,
            deep_reasoner_adapter=args.deep_reasoner_tool_out,
            answer_critic_adapter=args.answer_critic_tool_out,
            fast_solver_base_model=resolved_tool_base_models["fast_solver_tool"],
            deep_reasoner_base_model=resolved_tool_base_models["deep_reasoner_tool"],
            answer_critic_base_model=resolved_tool_base_models["answer_critic_tool"],
            seed=args.seed,
            per_device_train_bs=args.mgr_bs, grad_accum=args.mgr_grad_accum,
            max_prompt_length=args.mgr_max_prompt_length,
            max_completion_length=args.mgr_max_completion_length,
            temperature=args.mgr_temperature, num_generations=args.mgr_num_generations,
            grpo_beta=args.grpo_beta,
            fail_buffer_jsonl=fb, raw_trace_jsonl=rt,
            use_wandb=args.grpo_use_wandb, wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity, wandb_run_name=args.wandb_run_name,
            wandb_mode=args.wandb_mode,
            use_vllm=args.mgr_use_vllm, vllm_mode=args.mgr_vllm_mode,
            vllm_server_base_url=args.mgr_vllm_server_base_url,
            vllm_gpu_memory_utilization=args.mgr_vllm_gpu_memory_utilization,
            vllm_enable_sleep_mode=args.mgr_vllm_enable_sleep_mode,
            manager_use_lora=args.mgr_use_lora,
            manager_lora_r=args.mgr_lora_r, manager_lora_alpha=args.mgr_lora_alpha,
            manager_lora_dropout=args.mgr_lora_dropout,
            manager_gradient_checkpointing=args.mgr_gradient_checkpointing,
        )
        return

    if args.stage == "analyze_routing":
        rt = args.raw_trace_jsonl.strip() or os.path.join(args.manager_out, "train_raw_trace.jsonl")
        out_json = args.routing_summary_json.strip() or os.path.join(args.manager_out, "routing_summary.json")
        out_txt = args.routing_summary_txt.strip() or os.path.join(args.manager_out, "routing_summary.txt")
        summary = analyze_routing_trace(rt, out_json=out_json, out_txt=out_txt)
        print("[ROUTING]", json.dumps(summary, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
