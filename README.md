# fairGATE <img src="https://img.shields.io/badge/status-CRAN_ready-brightgreen" alt="Status Badge" align="right"/>

**fairGATE** is an R package for training and analysing **fairness-aware gated neural networks** (Mixture-of-Experts models) designed for **subgroup-aware prediction and interpretability** in many types of data, including but not limited to: health, clinical and economic data.

---

## 🧠 Overview

Modern clinical prediction models often perform unevenly across demographic or clinical subgroups (e.g., sex, age, ethnicity).  
**fairGATE** implements a flexible Gated Neural Network (GNN) architecture that explicitly models subgroup-specific experts while maintaining a shared gating mechanism.  
This allows users to:

- Train predictive models that **mitigate subgroup bias**
- Analyse **expert specialisation** and **gate behaviour**
- Evaluate **fairness metrics** and **interpretability** in model outputs

The package provides a full, reproducible pipeline:
1. **Data preparation:** `prepare_data()`  
2. **Model training:** `train_gnn()`  
3. **Result analysis:** `analyse_gnn_results()` and `analyse_experts()`  
4. **Visualisation:** `plot_sankey()` and other diagnostic plots

---

## 🎯 Purpose

The goal of **fairGATE** is to make *fair, interpretable, and subgroup-aware deep learning* accessible to health and clinical researchers.  
It is designed for use with binary outcomes and subgroup variables such as gender, ethnicity, income, or treatment group.  
The methods draw on:
- Jordan and Jacobs (1994) <doi:10.1162/neco.1994.6.2.181>  
- Hardt, Price, and Srebro (2016) <doi:10.1145/3157382>  
- Iniesta, Stahl, and McGuffin (2016) <doi:10.1016/j.jad.2016.03.016>

---

## 📖 Quick Reference

A complete list of functions, their inputs, outputs, and workflow order can be found in:  
👉 **[`fairGATE_function_reference.csv`](fairGATE_function_reference.csv)**

---

## 📘 Example Workflow

See the vignette for a full worked example using simulated clinical data:

👉 [**Vignette: Introduction to fairGATE**](vignettes/)

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
