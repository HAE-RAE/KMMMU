# KMMMU Evaluation Tutorial

This repository provides a minimal, tutorial-style evaluation pipeline for the **HAERAE-HUB/KMMMU** dataset using:

* **vLLM (offline inference)** with `LLM.generate`
* **Math-Verify** for structured answer parsing
* **LiteLLM** for LLM-as-a-Judge evaluation

The goal is clarity and simplicity. There is no over-engineering, no defensive try/except logic, and no batching logic. The full dataset is processed in a single forward pass and a single judge call.

---

## Files

### `eval_tutorial.ipynb`

The main notebook that:

1. Loads the KMMMU dataset from Hugging Face
2. Runs multimodal inference with vLLM
3. Forces answers in `$\\boxed{...}$` format
4. Parses outputs using Math-Verify `parse`
5. Sends `(question, gold, raw prediction, parsed prediction)` to an LLM judge
6. Computes accuracy and saves results to CSV

The notebook is structured with markdown explanations for each stage of the pipeline.

---

### `kmmmu_utils.py`

Utility module containing:

* System prompts
* Image utilities (e.g., `fetch_image`, optional image helpers)
* Shared configuration components

This keeps the notebook clean and focused on the evaluation flow.

---

## Requirements

Install the following:

```bash
pip install datasets vllm litellm "math-verify[inference]"
```

You also need:

* A local GPU setup compatible with vLLM
* An API key for your LiteLLM provider (e.g., `OPENAI_API_KEY`)

