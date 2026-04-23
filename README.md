# Tool-Routing Research with GRPO (Current Repo)

This README reflects the **current** design of `agents_routing_research_v1.py`.

Goal:
- Train a GRPO manager to decide whether to answer directly or call one of 4 tools.
- Make routing cost-aware: expensive tools should be used only when they improve accuracy.
- Analyze routing behavior after training from the raw trace.

Current runnable pipeline in this workspace:
- `make_splits`
- `build_tool_sft`
- `train_fast_solver_tool`
- `train_deep_reasoner_tool`
- `train_medical_kb_tool`
- `train_answer_critic_tool`
- `train_manager_grpo`
- `analyze_routing`

Important workspace note:
- Earlier notes referenced `baselines.py`, but that file is **not present** in this repo.

---

## 1. Tool Pool

The manager can route to 4 differentiated tools:

| Tool | What it does | Cost | Capability tags |
|---|---|---:|---|
| `fast_solver_tool` | quick top guess + alternatives | 1.0 | `quick_answer`, `candidate_generation` |
| `deep_reasoner_tool` | careful candidate-by-candidate analysis | 10.0 | `deep_reasoning`, `candidate_comparison`, `differential_dx` |
| `medical_kb_tool` | factual medical recall, definitions, mechanisms, thresholds | 2.0 | `knowledge_recall`, `definitions`, `guidelines` |
| `answer_critic_tool` | adversarially stress-test a favored answer | 5.0 | `critique`, `falsification`, `second_opinion` |

The reward still makes correctness dominant, but adds:
- a small cost penalty
- a routing appropriateness bonus
- a belief-state quality bonus
- penalties for repeated tools, fake tool text, budget overrun, and bad ordering

---

## 2. Data Scope for MedQA

`--task_name medqa --data_path medqa` now resolves to:

```text
MedQA/data_clean/questions
```

By default, that scans **all** MedQA regions found there:
- `US`
- `Mainland`
- `Taiwan`

If you want **US-only**, you must pass:

```bash
--medqa_regions US
```

This matters because:
- `--max_samples` is applied **after** region filtering
- the split file stores dataset scope metadata
- later stages must reuse the same `--medqa_regions`, or the script will stop with a scope mismatch

`--max_samples` means:
- total number of examples before `train/dev/test` split
- not the train set size

Example with defaults:
- `--max_samples 500 --test_size 200 --dev_size 160` -> `140 train / 160 dev / 200 test`
- `--max_samples 1000 --test_size 200 --dev_size 160` -> `640 train / 160 dev / 200 test`

---

## 3. Install

```bash
pip install "transformers>=4.53" "trl>=0.19" "datasets>=2.19" "peft>=0.11" "accelerate>=0.30"
pip install numpy packaging requests
```

Optional:

```bash
pip install wandb vllm
```

Runtime notes:
- `vllm` is **not supported on native Windows** in this script.
- Multi-GPU training is most practical on Linux.
- For quick bring-up, you can still run split creation and tool-data generation on Windows.

PowerShell note:
- Replace Bash line continuation `\` with PowerShell backtick `` ` ``, or put each command on one line.

---

## 4. Recommended Run Modes

### 4.1 Smoke Run

Use this to confirm the pipeline works end to end:
- `US-only`
- `500` total examples
- smaller base model if needed

### 4.2 Research Run

Use this for the main experiment:
- `US-only`
- `1000` total examples
- `Qwen/Qwen3-8B` if your GPUs can support it

If you are memory-constrained, start with `Qwen/Qwen3-0.6B` for both manager and tools, then scale up.

---

## 5. End-to-End Commands

### 5.1 Create Splits

#### US-only, 500 total examples

```bash
python agents_routing_research_v1.py \
  --stage make_splits \
  --task_name medqa \
  --data_path medqa \
  --medqa_regions US \
  --split_path splits_medqa_us_500.json \
  --max_samples 500 \
  --test_size 200 \
  --dev_size 160 \
  --seed 42
```

#### US-only, 1000 total examples

```bash
python agents_routing_research_v1.py \
  --stage make_splits \
  --task_name medqa \
  --data_path medqa \
  --medqa_regions US \
  --split_path splits_medqa_us_1000.json \
  --max_samples 1000 \
  --test_size 200 \
  --dev_size 160 \
  --seed 42
```

What to expect in logs:
- a region breakdown for MedQA
- confirmation that only `US` examples were kept
- final train/dev/test sizes

### 5.2 Build Tool SFT Data

GPT teacher mode is recommended for MedQA. Weak heuristic supervision is usually too poor for 5-way MCQ training.

Set teacher env vars first:

```bash
# Bash
export TEACHER_BASE_URL="https://api.openai.com"
export TEACHER_API_KEY="sk-..."
export TEACHER_MODEL="gpt-4o-mini"
```

```powershell
# PowerShell
$env:TEACHER_BASE_URL = "https://api.openai.com"
$env:TEACHER_API_KEY = "sk-..."
$env:TEACHER_MODEL = "gpt-4o-mini"
```

Then build tool SFT data for the 1000-example US split:

```bash
python agents_routing_research_v1.py \
  --stage build_tool_sft \
  --task_name medqa \
  --data_path medqa \
  --medqa_regions US \
  --split_path splits_medqa_us_1000.json \
  --tool_sft_out_dir tool_sft_medqa_us_1000 \
  --tool_variants_train 2 \
  --tool_variants_dev 1 \
  --tool_synth_mode gpt \
  --tool_synth_gpt_temperature 0.2 \
  --tool_synth_gpt_max_retries 3
```

Expected outputs:
- `tool_sft_medqa_us_1000/fast_solver_tool_train.jsonl`
- `tool_sft_medqa_us_1000/fast_solver_tool_dev.jsonl`
- `tool_sft_medqa_us_1000/deep_reasoner_tool_train.jsonl`
- `tool_sft_medqa_us_1000/deep_reasoner_tool_dev.jsonl`
- `tool_sft_medqa_us_1000/medical_kb_tool_train.jsonl`
- `tool_sft_medqa_us_1000/medical_kb_tool_dev.jsonl`
- `tool_sft_medqa_us_1000/answer_critic_tool_train.jsonl`
- `tool_sft_medqa_us_1000/answer_critic_tool_dev.jsonl`

### 5.3 Train the 4 Tool Adapters

Each tool trains independently from the generated JSONL files.

#### Bash loop

```bash
for TOOL in fast_solver deep_reasoner medical_kb answer_critic; do
  python agents_routing_research_v1.py \
    --stage train_${TOOL}_tool \
    --tool_base_model Qwen/Qwen3-8B \
    --tool_sft_out_dir tool_sft_medqa_us_1000 \
    --${TOOL}_tool_out ${TOOL}_tool_medqa_us_1000 \
    --tool_epochs 2 \
    --tool_lr 2e-4 \
    --tool_bs 1 \
    --tool_grad_accum 8 \
    --tool_max_seq_len 2048 \
    --tool_use_lora
done
```

#### PowerShell

Run the four stages explicitly in PowerShell:

```powershell
python agents_routing_research_v1.py --stage train_fast_solver_tool --tool_base_model Qwen/Qwen3-8B --tool_sft_out_dir tool_sft_medqa_us_1000 --fast_solver_tool_out fast_solver_tool_medqa_us_1000 --tool_epochs 2 --tool_lr 2e-4 --tool_bs 1 --tool_grad_accum 8 --tool_max_seq_len 2048 --tool_use_lora
python agents_routing_research_v1.py --stage train_deep_reasoner_tool --tool_base_model Qwen/Qwen3-8B --tool_sft_out_dir tool_sft_medqa_us_1000 --deep_reasoner_tool_out deep_reasoner_tool_medqa_us_1000 --tool_epochs 2 --tool_lr 2e-4 --tool_bs 1 --tool_grad_accum 8 --tool_max_seq_len 2048 --tool_use_lora
python agents_routing_research_v1.py --stage train_medical_kb_tool --tool_base_model Qwen/Qwen3-8B --tool_sft_out_dir tool_sft_medqa_us_1000 --medical_kb_tool_out medical_kb_tool_medqa_us_1000 --tool_epochs 2 --tool_lr 2e-4 --tool_bs 1 --tool_grad_accum 8 --tool_max_seq_len 2048 --tool_use_lora
python agents_routing_research_v1.py --stage train_answer_critic_tool --tool_base_model Qwen/Qwen3-8B --tool_sft_out_dir tool_sft_medqa_us_1000 --answer_critic_tool_out answer_critic_tool_medqa_us_1000 --tool_epochs 2 --tool_lr 2e-4 --tool_bs 1 --tool_grad_accum 8 --tool_max_seq_len 2048 --tool_use_lora
```

Each stage writes one LoRA adapter directory such as:
- `fast_solver_tool_medqa_us_1000`
- `deep_reasoner_tool_medqa_us_1000`
- `medical_kb_tool_medqa_us_1000`
- `answer_critic_tool_medqa_us_1000`

### 5.4 Train the GRPO Manager

Important:
- Reuse the **same** `--split_path` and `--medqa_regions` that you used in `make_splits`.
- If you built a US-only split, you must also pass `--medqa_regions US` here.

#### 2-GPU example, no vLLM

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch \
  --config_file trl_multi_gpu_2gpu_bf16.yaml \
  agents_routing_research_v1.py \
  --stage train_manager_grpo \
  --task_name medqa \
  --data_path medqa \
  --medqa_regions US \
  --split_path splits_medqa_us_1000.json \
  --manager_base_model Qwen/Qwen3-8B \
  --tool_base_model Qwen/Qwen3-8B \
  --fast_solver_tool_out fast_solver_tool_medqa_us_1000 \
  --deep_reasoner_tool_out deep_reasoner_tool_medqa_us_1000 \
  --medical_kb_tool_out medical_kb_tool_medqa_us_1000 \
  --answer_critic_tool_out answer_critic_tool_medqa_us_1000 \
  --manager_out manager_grpo_routing_medqa_us_1000 \
  --mgr_bs 1 \
  --mgr_grad_accum 4 \
  --mgr_num_generations 8 \
  --mgr_max_prompt_length 3000 \
  --mgr_max_completion_length 1536 \
  --mgr_temperature 0.7 \
  --grpo_beta 0.01 \
  --mgr_use_lora \
  --mgr_gradient_checkpointing \
  --grpo_use_wandb \
  --wandb_project medqa_routing_research \
  --wandb_run_name costaware_us1000_run1
```

Batch geometry check for the example above:
- `per_device_bs * world_size * grad_accum`
- `1 * 2 * 4 = 8`
- divisible by `num_generations = 8`

#### Small bring-up example

If you just want to confirm the manager stage starts, use a smaller model and a smaller split:

```bash
accelerate launch \
  --config_file trl_multi_gpu_2gpu_bf16.yaml \
  agents_routing_research_v1.py \
  --stage train_manager_grpo \
  --task_name medqa \
  --data_path medqa \
  --medqa_regions US \
  --split_path splits_medqa_us_500.json \
  --manager_base_model Qwen/Qwen3-0.6B \
  --tool_base_model Qwen/Qwen3-0.6B \
  --fast_solver_tool_out fast_solver_tool_medqa_us_500 \
  --deep_reasoner_tool_out deep_reasoner_tool_medqa_us_500 \
  --medical_kb_tool_out medical_kb_tool_medqa_us_500 \
  --answer_critic_tool_out answer_critic_tool_medqa_us_500 \
  --manager_out manager_grpo_routing_medqa_us_500 \
  --mgr_bs 1 \
  --mgr_grad_accum 4 \
  --mgr_num_generations 8 \
  --mgr_use_lora \
  --mgr_gradient_checkpointing
```

### 5.5 Analyze Routing

```bash
python agents_routing_research_v1.py \
  --stage analyze_routing \
  --manager_out manager_grpo_routing_medqa_us_1000
```

This writes:
- `manager_grpo_routing_medqa_us_1000/routing_summary.json`
- `manager_grpo_routing_medqa_us_1000/routing_summary.txt`

The summary includes:
- `accuracy`
- `avg_cost_per_completion`
- `avg_cost_per_correct`
- `routing_entropy`
- `routing_entropy_normalized`
- `tool_appropriateness_rate`
- `belief_state_present_rate`
- per-route accuracy, reward, and cost
- tool call frequencies

---

## 6. Minimal "What Do I Run?" Answer

If you want the shortest practical path for the main experiment:

1. Create a US-only split.
2. Build tool SFT data from that split.
3. Train the 4 tool adapters.
4. Train the GRPO manager with the same split and region filter.
5. Run `analyze_routing`.

Concrete split command:

```bash
python agents_routing_research_v1.py --stage make_splits --task_name medqa --data_path medqa --medqa_regions US --split_path splits_medqa_us_1000.json --max_samples 1000 --test_size 200 --dev_size 160 --seed 42
```

Concrete tool-data command:

```bash
python agents_routing_research_v1.py --stage build_tool_sft --task_name medqa --data_path medqa --medqa_regions US --split_path splits_medqa_us_1000.json --tool_sft_out_dir tool_sft_medqa_us_1000 --tool_variants_train 2 --tool_variants_dev 1 --tool_synth_mode gpt --tool_synth_gpt_temperature 0.2 --tool_synth_gpt_max_retries 3
```

After that, train the 4 tools and the manager with the matching paths shown above.

---

## 7. Research Notes

What this setup can support:
- cost-aware routing experiments
- routing entropy analysis
- tool usage distribution analysis
- belief-state quality and routing appropriateness analysis

What it does **not** currently support in this workspace:
- built-in baseline evaluation script
- native Windows `vllm`

---

## 8. Debugging Checklist

If something fails, check:

- `trl` version is at least `0.19.0`
- `transformers` version is at least `4.53.0`
- `--mgr_num_generations` divides `mgr_bs * world_size * mgr_grad_accum`
- the split file and current run use the same `--medqa_regions`
- the tool adapter directories exist before manager training
- the OpenAI teacher env vars are set if using `--tool_synth_mode gpt`
- `vllm` is not enabled on native Windows

Typical mistakes:

- Forgetting `--medqa_regions US` in a later stage after creating a US-only split
- Reusing `splits_medqa_1000.json` that was built from all-region MedQA
- Training tools into one directory name and pointing manager training at another

---

## 9. Current Recommendation

For this repo, the most consistent MedQA setup is:
- `task_name = medqa`
- `data_path = medqa`
- `medqa_regions = US`
- `max_samples = 1000`
- `split_path = splits_medqa_us_1000.json`

If you only want a quick validation run:
- keep `medqa_regions = US`
- use `max_samples = 500`
- optionally use `Qwen/Qwen3-0.6B` first
