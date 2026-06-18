# meta2perf-sr-deap

This repository contains the code, data, and experiment artifacts for a DEAP-based genetic programming pipeline that learns closed-form symbolic regression models for estimating machine-learning classification performance.

The prediction target is the Matthews Correlation Coefficient (MCC). Each meta-instance combines dataset meta-features, model descriptors, and the MCC obtained by a machine-learning algorithm on a sampled version of a classification dataset.

## Repository Contents

```text
.
├── README.md
└── src/
    ├── exp_perf_beam_grid_search.py
    ├── exp_perf_beam_grid_search_top10.py
    ├── exp_perf_beam_grid_search_top10_s1.py
    ├── exp_perf_beam_grid_search_top10_s2.py
    ├── exp_perf_beam_grid_search_top10_s3.py
    ├── plot_reg_model.py
    ├── meta_dataset/
    │   └── data.csv
    ├── plots/
    │   └── pred_vs_true.pdf
    ├── results/
    │   ├── beam_grid_hyperparameter_search_summary.csv
    │   ├── beam_grid_refinement_top10_summary.csv
    │   ├── exp_perf_beam_grid_search_top10_s1/
    │   │   └── summary.csv
    │   ├── exp_perf_beam_grid_search_top10_s2/
    │   │   └── summary.csv
    │   └── exp_perf_beam_grid_search_top10_s3/
    │       └── summary.csv
    └── utils/
        ├── ga_initialization.py
        └── gp_custom_operators.py
```

## Data

The file `src/meta_dataset/data.csv` contains 25,836 meta-instances and 24 columns. The descriptive columns are:

- `Seed`
- `Dataset`
- `Sample Size`
- `Model`

The remaining columns contain dataset meta-features, model descriptors, and the target variable `MCC`.

## Experiment Workflow

The repository implements a three-step search and evaluation workflow.

### 1. Beam-Guided Hyperparameter Search

`src/exp_perf_beam_grid_search.py` performs the first screening stage. It evaluates a budget-limited sequence of configurations from a predefined grid using a beam-guided search strategy.

Main settings:

- Maximum tree size: 400 nodes
- Maximum tree height: 30
- Initialization method: OBLESA
- Screening budget: 100 GP generations per configuration
- Maximum evaluated configurations: 300
- Base random seed: 42

Main output:

- `src/results/beam_grid_hyperparameter_search_summary.csv`

### 2. Top-10 Refinement

`src/exp_perf_beam_grid_search_top10.py` reads the first-stage summary, selects the top 10 configurations, and reruns them with a larger budget.

Main settings:

- Refinement budget: 500 GP generations per selected configuration
- Number of selected configurations: 10
- Checkpoint resume enabled

Main output:

- `src/results/beam_grid_refinement_top10_summary.csv`

### 3. Independent Final Runs

The scripts below run manually selected configurations for 5,000 GP generations:

- `src/exp_perf_beam_grid_search_top10_s1.py`
- `src/exp_perf_beam_grid_search_top10_s2.py`
- `src/exp_perf_beam_grid_search_top10_s3.py`

Their summaries are stored in:

- `src/results/exp_perf_beam_grid_search_top10_s1/summary.csv`
- `src/results/exp_perf_beam_grid_search_top10_s2/summary.csv`
- `src/results/exp_perf_beam_grid_search_top10_s3/summary.csv`

In the included results, configuration `s2` obtained the highest test `R^2` among the three independent final runs.

## Plotting the Selected Model

`src/plot_reg_model.py` evaluates the selected symbolic expression and generates a predicted-versus-true MCC plot.

Output:

- `src/plots/pred_vs_true.pdf`

## Running the Code

The scripts use paths relative to `src/`, so run them from inside that directory:

```bash
cd src
python exp_perf_beam_grid_search.py
python exp_perf_beam_grid_search_top10.py
python exp_perf_beam_grid_search_top10_s1.py
python exp_perf_beam_grid_search_top10_s2.py
python exp_perf_beam_grid_search_top10_s3.py
python plot_reg_model.py
```

Optional generated folders such as `cache/`, `checkpoints/`, and `logs/` may be created when the GP scripts are executed.