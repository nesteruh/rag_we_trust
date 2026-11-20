# In RAG We Trust?

Final project for *Natural Language Processing* (Prof. Alfio Ferrara, Università degli Studi di Milano, AY 2025-26), project P5.

Measures how retrieval-augmented generation (RAG) systems handle poisoned or conflicting evidence.

## Repository layout

| Path | Contents |
|---|---|
| `src/ragtrust/` | The code, one module per layer: `data`, `prompts`, `llm`, `metrics`, `poison`, `retrieval`, `experiment`, `runs`, `analysis`, `stats` |
| `notebooks/` | Numbered notebooks that import from `ragtrust` and tell the story of the project |
| `data/poison/` | The question selection, the generated poison set (published: LLM generation cannot be reproduced exactly) and the excluded items |
| `data/handcheck/` | Hand checks of the poison items and of the answer scoring |
| `results/` | Final result files (`*_final.csv` with the settings next to them as `.json`), `tables/`, `figures/` |

| Notebook | What |
|---|---|
| `00_question_selection` | Closed-book run on all 7,405 questions; stratified selection of 400 known + 400 unknown |
| `01_data_and_baselines` | Closed-book, gold-only and 10-paragraph baselines |
| `02_poison_generation` | False and true versions of the answer paragraph (writer: gemma4:12b) |
| `03_controlled_poisoning` | m false versions out of 4, prompts and source labels |
| `04_retrieval_poisoning` | BM25 and dense retrieval over a collection that contains the poison |
| `05_self_consistency` | 5 samples per question at temperature 0.7 |
| `06_analysis` | Main results: dose-response, the five hypothesis tests, prompt and labels, retrieval, figures|

## Setup

Requires Python 3.11+ and [Ollama](https://ollama.com/) with `llama3.1:8b`, `qwen2.5:7b`, `gemma4:12b` and `nomic-embed-text`.

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Linux / macOS:         source .venv/bin/activate
pip install -e .
```

## Reproducing the results

**From the published result files:** run `notebooks/06_analysis.ipynb`.
It rebuilds every table in `results/tables/` and every figure in `results/figures/`.

```bash
python -m ipykernel install --user --name ragtrust        # once: a Jupyter kernel that uses this environment
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=ragtrust notebooks/06_analysis.ipynb
```

**Environment of the published results:** Ollama 0.40.16; model ids `llama3.1:8b` 46e0c10c039e, `qwen2.5:7b` 845dbda0ea48,
`gemma4:12b` 6114515d63c1, `nomic-embed-text` 0a109f422b47 (`results/environment.json`); Python packages in `requirements.lock.txt`.

**From scratch** (the final run of step 3 took about 5 hours on one consumer GPU; steps 1 and 2 take additional hours):
1. `notebooks/00_question_selection.ipynb`: stratified question selection
2. `notebooks/02_poison_generation.ipynb`: poison generation
3. `notebooks/01`, `03`, `04`, `05`: baselines, controlled poisoning, retrieval poisoning, self-consistency
4. `notebooks/06_analysis.ipynb`

Generation uses temperature 0 and a fixed seed, but Ollama is not fully deterministic: about 8% of answers differ
between two runs, so a new run reproduces the results within noise, not exactly. Result files are cached by name;
delete a file in `results/` to run that step again.

## Main results

Share of answers that give the false answer when m of the 4 versions of the answer paragraph are false
(questions answered correctly at m = 0; 670 poison items after excluding 81 found bad in a hand check):

| Model | m = 1 | m = 2 | m = 3 | m = 4 |
|---|---|---|---|---|
| llama3.1:8b | 0.07 | 0.22 | 0.50 | 0.80 |
| qwen2.5:7b | 0.04 | 0.29 | 0.62 | 0.89 |

A "check consistency" prompt has no detectable effect (m = 2); a source label on the false versions
("Wikipedia" vs "anonymous online forum") changes the share by 16-18 points at m = 2; knowing the answer protects
llama but not significantly qwen; qwen is easier to poison than llama; a real retriever dilutes the poison.
All numbers, tests and figures: `notebooks/06_analysis.ipynb`.

## AI usage

The use of AI assistants is declared in the AI usage disclaimer of the report.
