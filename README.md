# MedQA US 500 Clean 流程说明

这份说明对应当前的 [agents_routing_research_v1.py](/c:/Users/yyn07/Desktop/MedQA/agents_routing_research_v1.py)。

当前这条线只做一件事：

- 只用 `MedQA US`
- 只取 `500` 条
- split 固定为 `320 train / 80 dev / 100 test`
- 只保留 3 个工具
  - `fast_solver_tool`
  - `deep_reasoner_tool`
  - `answer_critic_tool`
- `medical_kb_tool` 已完全移除

## 1. 现在代码的原则

这版代码尽量不做隐式 fallback，按你现在的预期走。

### 1.1 数据源原则

MedQA 现在优先读取 clean/canonical 数据源：

```text
MedQA/data_clean/questions/US/us_clean_all.jsonl
```

不会再把 `4_options`、`metamap` 之类的衍生文件混进主流程。

### 1.2 split 原则

`make_splits` 生成的 split 文件不再只是：

- `train_ids`
- `dev_ids`
- `test_ids`

还会直接写入：

- `train_examples`
- `dev_examples`
- `test_examples`

每条样本都带：

- `question`
- `answer`
- `answer_idx`
- `options`
- `context`
- `source_file`

后续：

- `build_tool_sft`
- `train_manager_grpo`

都要求 split 文件里必须有这些嵌入样本。

如果 split 里没有 `train_examples/dev_examples`，脚本会直接报错，不再回退到“重新扫全量数据 + 用 ID 回查”。

### 1.3 模型配置原则

现在模型配置也尽量显式：

- `train_fast_solver_tool` 必须传 `--fast_solver_base_model`
- `train_deep_reasoner_tool` 必须传 `--tool_base_model`，或者显式传 `--deep_reasoner_base_model`
- `train_answer_critic_tool` 必须传 `--tool_base_model`，或者显式传 `--answer_critic_base_model`
- `train_manager_grpo` 必须传
  - `--fast_solver_base_model`
  - `--tool_base_model`
  - 如果 deep / critic 想和默认工具模型不同，再单独传 override

你当前的设计就是：

- `fast_solver = Qwen/Qwen3-0.6B`
- `deep_reasoner = Qwen/Qwen3-8B`
- `answer_critic = Qwen/Qwen3-8B`
- `manager = Qwen/Qwen3-8B`

### 1.4 Teacher 合成原则

如果你用：

```text
--tool_synth_mode gpt
```

那么 teacher 生成失败时现在会直接报错，不再自动降级成 weak supervision。

这意味着：

- 训练数据要么是 clean teacher data
- 要么就是你明确选择 `weak`

不会再混出半 clean 半 weak 的 SFT 数据。

## 2. 你现在这条线的标准产物

建议统一用下面这套名字：

- split：`splits_medqa_us_500_clean.json`
- tool SFT 数据：`tool_sft_medqa_us_500_clean`
- fast solver adapter：`fast_solver_tool_medqa_us_500_clean`
- deep reasoner adapter：`deep_reasoner_tool_medqa_us_500_clean`
- answer critic adapter：`answer_critic_tool_medqa_us_500_clean`
- manager：`manager_grpo_medqa_us_500_clean`

## 3. 整个流程怎么走

就是 6 步：

1. 生成 clean split
2. 构建 3-tool SFT 数据
3. 训练 fast solver
4. 训练 deep reasoner
5. 训练 answer critic
6. 训练 GRPO manager

下面是每一步的标准命令。

## 4. 第一步：生成 clean split

这一步会：

- 只取 `US`
- 只抽 `500` 条
- 切成 `320 / 80 / 100`
- 把样本原文直接写进 split

命令：

```powershell
python agents_routing_research_v1.py --stage make_splits --task_name medqa --data_path medqa --medqa_regions US --split_path splits_medqa_us_500_clean.json --max_samples 500 --test_size 100 --dev_size 80 --seed 42
```

预期输出：

- [splits_medqa_us_500_clean.json](/c:/Users/yyn07/Desktop/MedQA/splits_medqa_us_500_clean.json)

并且日志里应该看到：

```text
train/dev/test = 320/80/100
```

## 5. 第二步：构建 3-tool SFT 数据

### 5.1 推荐模式

推荐用 GPT teacher。

先设置环境变量：

```powershell
$env:TEACHER_BASE_URL = "https://api.openai.com"
$env:TEACHER_API_KEY = "sk-..."
$env:TEACHER_MODEL = "gpt-4o-mini"
```

### 5.2 构建命令

```powershell
python agents_routing_research_v1.py --stage build_tool_sft --task_name medqa --data_path medqa --medqa_regions US --split_path splits_medqa_us_500_clean.json --tool_sft_out_dir tool_sft_medqa_us_500_clean --tool_variants_train 2 --tool_variants_dev 1 --tool_synth_mode gpt --tool_synth_gpt_temperature 0.2 --tool_synth_gpt_max_retries 3
```

说明：

- 这里不需要传工具模型参数
- 这里只是生成工具 SFT 数据，不训练模型
- 它只会读 split 文件里的 `train_examples/dev_examples`

预期输出目录：

- `tool_sft_medqa_us_500_clean`

里面只会有 6 个文件：

- `fast_solver_tool_train.jsonl`
- `fast_solver_tool_dev.jsonl`
- `deep_reasoner_tool_train.jsonl`
- `deep_reasoner_tool_dev.jsonl`
- `answer_critic_tool_train.jsonl`
- `answer_critic_tool_dev.jsonl`

## 6. 第三步：训练 fast solver

你的设计里 `fast_solver` 固定是 `0.6B`。

命令：

```powershell
python agents_routing_research_v1.py --stage train_fast_solver_tool --fast_solver_base_model Qwen/Qwen3-0.6B --tool_sft_out_dir tool_sft_medqa_us_500_clean --fast_solver_tool_out fast_solver_tool_medqa_us_500_clean --tool_epochs 2 --tool_lr 2e-4 --tool_bs 1 --tool_grad_accum 8 --tool_max_seq_len 2048 --tool_use_lora
```

这里不需要再传 `--tool_base_model`。

## 7. 第四步：训练 deep reasoner

你的设计里 `deep_reasoner` 用大模型。

命令：

```powershell
python agents_routing_research_v1.py --stage train_deep_reasoner_tool --tool_base_model Qwen/Qwen3-8B --tool_sft_out_dir tool_sft_medqa_us_500_clean --deep_reasoner_tool_out deep_reasoner_tool_medqa_us_500_clean --tool_epochs 2 --tool_lr 2e-4 --tool_bs 1 --tool_grad_accum 8 --tool_max_seq_len 2048 --tool_use_lora
```

如果你以后想让 deep reasoner 和 `--tool_base_model` 不一样，再改成：

```powershell
--deep_reasoner_base_model ...
```

## 8. 第五步：训练 answer critic

你的设计里 `answer_critic` 也用大模型。

命令：

```powershell
python agents_routing_research_v1.py --stage train_answer_critic_tool --tool_base_model Qwen/Qwen3-8B --tool_sft_out_dir tool_sft_medqa_us_500_clean --answer_critic_tool_out answer_critic_tool_medqa_us_500_clean --tool_epochs 2 --tool_lr 2e-4 --tool_bs 1 --tool_grad_accum 8 --tool_max_seq_len 2048 --tool_use_lora
```

如果以后想单独改 answer critic 模型，再显式传：

```powershell
--answer_critic_base_model ...
```

## 9. 第六步：训练 GRPO manager

这是最后一步。

你当前设计是：

- manager：`Qwen/Qwen3-8B`
- fast solver：`Qwen/Qwen3-0.6B`
- deep reasoner：`Qwen/Qwen3-8B`
- answer critic：`Qwen/Qwen3-8B`

对应命令：

```powershell
accelerate launch --config_file accelerate_configs/trl_multi_gpu_2gpu_bf16.yaml agents_routing_research_v1.py --stage train_manager_grpo --task_name medqa --data_path medqa --medqa_regions US --split_path splits_medqa_us_500_clean.json --manager_base_model Qwen/Qwen3-8B --tool_base_model Qwen/Qwen3-8B --fast_solver_base_model Qwen/Qwen3-0.6B --fast_solver_tool_out fast_solver_tool_medqa_us_500_clean --deep_reasoner_tool_out deep_reasoner_tool_medqa_us_500_clean --answer_critic_tool_out answer_critic_tool_medqa_us_500_clean --manager_out manager_grpo_medqa_us_500_clean --mgr_bs 1 --mgr_grad_accum 4 --mgr_num_generations 8 --mgr_max_prompt_length 2048 --mgr_max_completion_length 1536 --mgr_use_lora
```

说明：

- 这里 `--tool_base_model` 是 deep reasoner 和 answer critic 的公共模型
- `fast_solver` 还是单独显式指定 `--fast_solver_base_model`
- 这里也只会读 split 文件里的 `train_examples`

## 10. 推荐执行顺序

你直接照这个顺序跑就行：

1. `make_splits`
2. `build_tool_sft`
3. `train_fast_solver_tool`
4. `train_deep_reasoner_tool`
5. `train_answer_critic_tool`
6. `train_manager_grpo`

## 11. 现在不要再用的旧东西

不要再用下面这些旧配置或旧产物：

- `train_medical_kb_tool`
- `medical_kb_tool_*`
- `tool_sft_data_four`
- `manager_grpo_four`
- `500 -> 140/160/200` 那套旧 split

## 12. 一句话总结

你现在这条线就是：

- `US only`
- `500 clean samples`
- `320/80/100`
- `3 tools`
- `fast_solver = 0.6B`
- `deep_reasoner / answer_critic / manager = 8B`
- `split 文件是唯一可信样本来源`
