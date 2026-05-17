# Code Submission Notes

This project can be reproduced from the single entrypoint below:

```bash
python script/run_reproducible_pipeline.py
```

For a faster run that stops after data cleaning, parameter fitting, and the
canonical equal-weight optimisation, use:

```bash
python script/run_reproducible_pipeline.py --core-only
```

## Pipeline Order

The entrypoint executes the modelling scripts in this order:

1. clean domestic price events and 10-trading-day crude-oil windows;
2. estimate task-1 oil-price transmission and structural adjustment parameters;
3. export the task-1 theoretical adjustment sequence;
4. prepare task-2 monthly and event-level inputs;
5. fit the task-2 social-welfare loss parameters;
6. generate the equal-weight optimal policy path;
7. run alpha diagnostics, fitted alpha functions, policy comparisons, and task-3
   simple-rule robustness checks.

The generated manifest is written to:

```text
result/pipeline_manifest.csv
```

Key final outputs are under `result/`, including:

- `final_prediction_compare_scheme1.csv`
- `task2_optimization_equal_weight_final_step001_updated.csv`
- `task2_before_after_adjustment_total_summary.csv`
- `task2_policy_comparison_key_metrics.csv`
- `task3_simple_rule_strategy_comparison.csv`
- `task3_robustness_scenario_summary.csv`
