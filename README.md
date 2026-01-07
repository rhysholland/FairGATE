# fairGATE <img src="https://img.shields.io/badge/CRAN-Downloadable-brightgreen" alt="Status Badge" align="right"/>

**fairGATE** is an R package for training and analysing **fairness-aware gated neural networks** (Mixture-of-Experts models) designed for **subgroup-aware prediction and interpretability** in many types of data, including but not limited to: health, clinical and economic data. Developed under the supervision of Dr Raquel Iniesta [https://www.kcl.ac.uk/people/raquel-iniesta]

### ⬇️ Now Downloadable via CRAN in RStudio

```r
install.packages("fairGATE")
```

---

## 🧠 Overview

Modern clinical prediction models often perform unevenly across demographic or clinical subgroups (e.g., sex, age, ethnicity).  
**fairGATE** implements a flexible Gated Neural Network (GNN) architecture that explicitly models subgroup-specific experts while maintaining a shared gating mechanism.  
This allows users to:

- Train predictive models that **mitigate subgroup bias**
- Analyse **expert specialisation** and **gate behaviour**
- Evaluate **fairness metrics** and **interpretability** in model outputs
- Export model predictions for **fairness evaluation** in a format directly compatable
  with **IBM's AIF360 toolkit** (see: https://github.com/Trusted-AI/AIF360)

The package provides a full, reproducible pipeline:
1. **Data preparation:** `prepare_data()`  
2. **Model training:** `train_gnn()`  
3. **Result analysis:** `analyse_gnn_results()` and `analyse_experts()`  
4. **Visualisation:** `plot_sankey()` and other diagnostic plots
5. **Fairness evaluation** `export_f360_csv` for use in IBM's AIF360

---

## 🎯 Purpose

The goal of **fairGATE** is to make *fair, interpretable, and subgroup-aware deep learning* accessible to health and clinical researchers.  
It is designed for use with binary outcomes and subgroup variables such as gender, ethnicity, income, or treatment group.  
The methods draw on:
- Jordan and Jacobs (1994) <doi:10.1162/neco.1994.6.2.181>  
- Hardt, Price, and Srebro (2016) <doi:10.1145/3157382>  
- Iniesta, Stahl, and McGuffin (2016) <doi:10.1016/j.jad.2016.03.016>

---

# fairGATE User Guide: Function Reference

This guide provides a concise overview of the core functions within **fairGATE**, designed to help users prepare data, train fairness-aware Gated Neural Networks (GNNs), analyse results, and export fairness metrics. Each section summarises the purpose, arguments, and outputs of each function with practical examples.

---

## 🧩 Data Preparation

### `prepare_data()`
Prepares a dataframe for GNN training by cleaning, encoding, and scaling features. It defines the binary outcome and sensitive group variable, returning a structured list ready for use in `train_gnn()`.

#### **Usage**
```r
prepared_data <- prepare_data(
  data,
  outcome_var,
  group_var,
  cols_to_remove = NULL
)
```

#### **Arguments**
- **data**: A dataframe containing the raw dataset.
- **outcome_var**: Name of the binary outcome column (coded 0/1).
- **group_var**: Name of the sensitive attribute column. Can be a **string or numeric variable** and can include **two or more subgroups** — the model automatically detects and handles them.
- **cols_to_remove** *(optional)*: Columns to exclude from the model (IDs, text fields, etc.).

#### **Returns**
A list containing:
- `X`: Scaled numeric feature matrix.
- `y`: Binary outcome vector.
- `group`: Encoded group vector.
- `feature_names`: Column names of the feature matrix.
- `subject_ids`: Optional subject identifiers.

#### **Example**
```r
prepared_data_chd <- prepare_data(
  data = sim_df_encoded,
  outcome_var = "chd_event",
  group_var = "race",
  cols_to_remove = "patient_id"
)
```

---

## ⚙️ Model Training

### `train_gnn()`
Trains a fairness-aware Gated Neural Network (GNN) using a fairness-constrained loss to reduce subgroup disparities in predictions. Supports optional hyperparameter tuning, repeated splits, and automatic fairness evaluation.

#### **Usage**
```r
final_fit <- train_gnn(
  prepared_data = prepared_data_chd,
  hyper_grid = tuning_grid,
  num_repeats = 7,
  epochs = 140,
  run_tuning = TRUE,
  verbose = TRUE
)
```

#### **Arguments**
- **prepared_data**: List from `prepare_data()`.
- **hyper_grid**: Dataframe with tuning parameters (`lr`, `hidden_dim`, `dropout_rate`, `lambda`, `temperature`).
- **num_repeats**: Number of train/test splits for robustness.
- **epochs**: Training epochs per run.
- **output_dir**: Directory for saving outputs (if enabled).
- **run_tuning**: Whether to run a tuning loop over the grid.
- **best_params**: Manually supplied parameters if tuning is skipped.
- **save_outputs**: If `TRUE`, saves predictions and weights as CSV/RDS.
- **seed**: Random seed for reproducibility.
- **verbose**: Prints progress updates.

#### **Setting up tuning grid and manual parameters**
```r
# Define a modest tuning grid for quick search
tuning_grid <- expand.grid(
  lr = c(5e-4, 1e-3, 2e-3),
  hidden_dim = c(64L, 128L),
  dropout_rate = c(0.2, 0.3),
  lambda = c(0.6, 0.8),
  temperature = c(0.7, 1.0),
  KEEP.OUT.ATTRS = FALSE
)

# Or define best_params directly for a single configuration
best_params <- data.frame(
  lr = 1e-3,
  hidden_dim = 128L,
  dropout_rate = 0.3,
  lambda = 0.6,
  temperature = 0.7
)

# Example without tuning (uses best_params)
final_fit <- train_gnn(
  prepared_data = prepared_data_chd,
  run_tuning = FALSE,
  best_params = best_params,
  num_repeats = 5,
  epochs = 120,
  verbose = TRUE
)
```

#### **Returns**
A list containing:
- `final_results`: Predictions and true outcomes per subject.
- `gate_weights`: Gate probabilities and entropy measures.
- `expert_weights`: Input-layer weights for each expert.
- `performance_summary`: AUC and Brier Score summary.
- `aif360_data`: Data formatted for fairness metric computation.
- `tuning_results`: Grid search outcomes when tuning enabled.

---

## 🔍 Model Evaluation & Analysis

### `analyse_gnn_results()`
Generates model performance plots and statistical summaries, including ROC and calibration curves, gate weight distributions, and entropy analyses.

#### **Usage**
```r
res <- analyse_gnn_results(
  gnn_results = final_fit,
  prepared_data = prepared_data_chd,
  create_roc_plot = TRUE,
  create_calibration_plot = TRUE,
  analyse_gate_weights = TRUE,
  analyse_gate_entropy = TRUE,
  verbose = TRUE
)
```

#### **Viewing outputs**
```r
# View overall metrics
res$metrics_table

# Display diagnostic plots
print(res$roc_plot)
print(res$calibration_plot)
print(res$gate_density_plot)
print(res$entropy_density_plot)

# Inspect gate weight test summaries
lapply(res$gate_weight_tests, function(x) x$method)
```

---

### `analyse_experts()`
Analyses subgroup-specific expert weights to identify which features most strongly influence predictions within each group.

#### **Usage**
```r
exp_res <- analyse_experts(
  gnn_results = final_fit,
  prepared_data = prepared_data_chd,
  top_n_features = 20,
  verbose = TRUE
)
```

#### **Viewing outputs**
```r
# Top features per subgroup
head(exp_res$means_by_group_wide)

# Pairwise comparisons between groups
names(exp_res$pairwise_differences)

# Example: display the Black vs White comparison
head(exp_res$pairwise_differences[["Black_vs_White"]])
```

#### **Returns**
A list containing:
- `all_weights`: Feature weights across experts and repeats.
- `means_by_group_wide`: Mean feature importances per group.
- `pairwise_differences`: Feature weight differences between groups.

---

## 🌐 Fairness & Export Tools

### `export_f360()`
Exports predictions and group information into a format compatible with IBM’s **AIF360** fairness toolkit.

#### **Usage**
```r
export_f360_csv(
  gnn_results = final_fit,
  prepared_data = prepared_data_chd,
  path = "outputs/fairness360_input.csv",
  include_gate_cols = TRUE,
  threshold = 0.5,
  verbose = TRUE
)
```

#### **Returns**
Writes a CSV file ready for import into AIF360 or Python fairness workflows.

---

### `plot_sankey()`
Creates a Sankey diagram visualising routing between actual groups, learned feature profiles, and assigned experts.

#### **Usage**
```r
p <- plot_sankey(
  prepared_data = prepared_data_chd,
  gnn_results = final_fit,
  expert_results = exp_res,
  verbose = TRUE
)

print(p)
```

#### **Returns**
A `ggplot2` Sankey chart showing group-to-expert routing patterns.

---

**fairGATE** provides a transparent and modular framework for fairness-aware machine learning, enabling users to explore subgroup-specific model behaviour, interpret gating mechanisms, and quantify fairness across sensitive attributes with ease.

---

## 📘 Example Workflow

See the vignette for a full worked example using simulated clinical data:

### [👉View Vignette Here](https://htmlpreview.github.io/?https://github.com/rhysholland/FairGATE/blob/main/doc/introduction-to-fairGATE.html)

Example (simplified):

```r
library(fairGATE)

# 1. Prepare data
prepared <- prepare_data(df, outcome = "remission", group = "sex")

# 2. Train model
results <- train_gnn(prepared, hyper_grid = grid, num_repeats = 20, epochs = 300)

# 3. Analyse experts
expert_results <- analyse_experts(results, prepared_data = prepared)

# 4. Visualise routing
plot_sankey(raw_data = df, gnn_results = results, expert_results = expert_results,
            group_mappings = c("0"="Male", "1"="Female"), group_var = "sex")
