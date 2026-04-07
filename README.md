# JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks

Evaluation script for JailNewsBench — a multilingual benchmark for fake news generation under jailbreak attacks.

- 📄 **Paper**: https://openreview.net/forum?id=7dTqUaY2Kl
- 🤗 **Dataset**: https://huggingface.co/datasets/MasahiroKaneko/JailNewsBench

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Generation + Evaluation

```bash
python evaluate.py \
  --model claude-sonnet-4-6 \
  --judge_model gpt-4o \
  --attack all \
  --split test \
  --output_dir outputs/eval
```

### Generation only

```bash
python evaluate.py \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor_parallel_size 4 \
  --attack all \
  --output_dir outputs/eval
```

### Evaluation only (from existing generations)

```bash
python evaluate.py \
  --judge_only \
  --generation_file outputs/eval/gen_role_play.jsonl \
  --judge_model gpt-4o \
  --output_dir outputs/eval
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | — | Target model (required for generation) |
| `--judge_model` | — | Judge model for evaluation |
| `--attack` | `all` | Attack type: `all` or one of `original`, `explicit`, `role_play`, `system_override`, `research_front`, `neg_prompting`, `context_overload` |
| `--split` | `test` | Dataset split: `train`, `validation`, `test` |
| `--tensor_parallel_size` | `1` | Tensor parallelism for vLLM |
| `--max_new_tokens` | `1024` | Max generation tokens |
| `--batch_size` | `64` | Batch size (vLLM only) |
| `--judge_only` | `False` | Skip generation, evaluate existing file |
| `--generation_file` | — | Path to JSONL for `--judge_only` |
| `--output_dir` | `outputs/eval` | Output directory |
| `--seed` | `42` | Random seed |

## Supported Models

Backend is inferred automatically from the model name.

| Prefix | Backend | API key env var |
|---|---|---|
| `gpt-`, `o1-`, `o3-` | OpenAI | `OPENAI_API_KEY` |
| `claude-` | Anthropic | `ANTHROPIC_API_KEY` |
| `gemini-` | Gemini | `GEMINI_API_KEY` |
| other | vLLM | — |

## Output

For each attack type, the script writes:

- `gen_{attack}.jsonl` — generated texts with metadata
- `eval_{attack}.jsonl` — per-record evaluation results (if `--judge_model` is set)
- `eval_{attack}_summary.json` — aggregated metrics

### Metrics

| Metric | Description |
|---|---|
| `ifl` | Infelicity rate (%) — disfluent outputs |
| `asr` | Attack success rate (%) — non-refused outputs |
| `avg_score` | Mean harmfulness score (0–4) across 8 sub-metrics |

The 8 sub-metrics are: `faithfulness`, `verifiability`, `adherence`, `scope`, `scale`, `formality`, `subjectivity`, `agitativeness`.

> **Note**: Scores reported in the paper are averaged across all splits. Results on the publicly released splits may not exactly match the paper's numbers.

## Citation

```bibtex
@inproceedings{
kaneko2026jailnewsbench,
title={JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks},
author={Masahiro Kaneko and Ayana Niwa and Timothy Baldwin},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=7dTqUaY2Kl}
}
```
