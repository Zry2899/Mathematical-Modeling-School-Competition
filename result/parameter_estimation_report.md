# 参数估计报告

## 1. 数据文件来源和使用字段

- 主数据来源：`result/domestic_events_clean.csv + result/oil_10_workday_average.csv`
- 已扫描文件：
  - `data\03_monthly_macro.csv`
  - `data\04_monthly_oil_supply.csv`
  - `data\05_monthly_product_consumption.csv`
  - `data\07_refinery_cost_assumptions.csv`
  - `data\basket-daily.csv`
  - `data\brent-daily.csv`
  - `data\china_cpi_ppi_monthly.csv`
  - `data\china_crude_oil_import_monthly.csv`
  - `data\cny_usd_exchange_rate.csv`
  - `data\rare-domastic.csv`
  - `data\wti-daily.csv`
  - `result\backtest_diesel.csv`
  - `result\backtest_gasoline.csv`
  - `result\domestic_events_clean.csv`
  - `result\final_prediction_compare_scheme1.csv`
  - `result\oil_10_workday_average.csv`
  - `result\parameter_estimation_diesel.csv`
  - `result\parameter_estimation_gasoline.csv`
  - `result\pipeline_manifest.csv`
  - `result\structural_backtest_diesel.csv`
  - `result\structural_backtest_gasoline.csv`
  - `result\structural_parameters_diesel.csv`
  - `result\structural_parameters_gasoline.csv`
  - `result\task2_alpha_analysis_sample_summary.csv`
  - `result\task2_alpha_correlation_tests.csv`
  - `result\task2_alpha_diesel_combined_metrics_by_direction.csv`
  - `result\task2_alpha_diesel_combined_predictions.csv`
  - `result\task2_alpha_diesel_combined_refit_params.csv`
  - `result\task2_alpha_diesel_downward_function_fit_data.csv`
  - `result\task2_alpha_diesel_downward_function_overlay_data.csv`
  - `result\task2_alpha_diesel_downward_function_params.csv`
  - `result\task2_alpha_diesel_upward_function_fit_data.csv`
  - `result\task2_alpha_diesel_upward_function_overlay_data.csv`
  - `result\task2_alpha_diesel_upward_function_params.csv`
  - `result\task2_alpha_gasoline_combined_metrics_by_direction.csv`
  - `result\task2_alpha_gasoline_combined_predictions.csv`
  - `result\task2_alpha_gasoline_combined_refit_params.csv`
  - `result\task2_alpha_gasoline_downward_function_fit_data.csv`
  - `result\task2_alpha_gasoline_downward_function_overlay_data.csv`
  - `result\task2_alpha_gasoline_downward_function_params.csv`
  - `result\task2_alpha_gasoline_upward_function_fit_data.csv`
  - `result\task2_alpha_gasoline_upward_function_overlay_data.csv`
  - `result\task2_alpha_gasoline_upward_function_params.csv`
  - `result\task2_alpha_partial_effects.csv`
  - `result\task2_alpha_partial_effects_summary.csv`
  - `result\task2_before_after_adjustment_alpha_path.csv`
  - `result\task2_before_after_adjustment_dimension_mean_summary.csv`
  - `result\task2_before_after_adjustment_event_detail.csv`
  - `result\task2_before_after_adjustment_total_summary.csv`
  - `result\task2_cpi_regression_data.csv`
  - `result\task2_data_audit.csv`
  - `result\task2_event_model_input.csv`
  - `result\task2_event_model_input_clean.csv`
  - `result\task2_event_model_input_forecast.csv`
  - `result\task2_fitted_parameters.csv`
  - `result\task2_loss_function_smoke_test.csv`
  - `result\task2_missing_data_detail.csv`
  - `result\task2_monthly_model_data.csv`
  - `result\task2_monthly_model_data_clean.csv`
  - `result\task2_monthly_model_data_forecast.csv`
  - `result\task2_optimization_equal_weight.csv`
  - `result\task2_optimization_equal_weight_20260516_142852.csv`
  - `result\task2_optimization_equal_weight_20260516_144910.csv`
  - `result\task2_optimization_equal_weight_20260516_144958.csv`
  - `result\task2_optimization_equal_weight_20260516_145033.csv`
  - `result\task2_optimization_equal_weight_20260516_145132.csv`
  - `result\task2_optimization_equal_weight_20260516_150519.csv`
  - `result\task2_optimization_equal_weight_20260516_155952.csv`
  - `result\task2_optimization_equal_weight_20260516_160644.csv`
  - `result\task2_optimization_equal_weight_final_step001.csv`
  - `result\task2_optimization_equal_weight_final_step001_summary.csv`
  - `result\task2_optimization_equal_weight_final_step001_updated.csv`
  - `result\task2_optimization_equal_weight_forecast.csv`
  - `result\task2_optimization_equal_weight_forecast_final_step001.csv`
  - `result\task2_optimization_equal_weight_forecast_final_step001_summary.csv`
  - `result\task2_optimization_equal_weight_forecast_summary.csv`
  - `result\task2_optimization_equal_weight_forecast_updated.csv`
  - `result\task2_optimization_equal_weight_summary.csv`
  - `result\task2_optimization_equal_weight_summary_updated.csv`
  - `result\task2_optimization_equal_weight_updated.csv`
  - `result\task2_policy_comparison_component_summary.csv`
  - `result\task2_policy_comparison_dimension_summary.csv`
  - `result\task2_policy_comparison_event_losses.csv`
  - `result\task2_policy_comparison_key_metrics.csv`
  - `result\task2_policy_comparison_monthly_cpi.csv`
  - `result\task2_raw_loss_distribution.xlsx`
  - `result\task2_raw_loss_distribution_summary.csv`
  - `result\task2_raw_loss_histograms.xlsx`
  - `result\task2_raw_loss_histogram_bins.csv`
  - `result\task2_raw_loss_history.csv`
  - `result\task2_raw_loss_monthly_history.csv`
  - `result\task2_sensitivity_comparison_step001.csv`
  - `result\task2_sensitivity_consumer_priority_step001.csv`
  - `result\task2_sensitivity_consumer_priority_step001_summary.csv`
  - `result\task2_sensitivity_cpi_priority_step001.csv`
  - `result\task2_sensitivity_cpi_priority_step001_summary.csv`
  - `result\task2_sensitivity_energy_priority_step001.csv`
  - `result\task2_sensitivity_energy_priority_step001_summary.csv`
  - `result\task2_sensitivity_firm_priority_step001.csv`
  - `result\task2_sensitivity_firm_priority_step001_summary.csv`
  - `result\task2_sensitivity_forecast_comparison_step001.csv`
  - `result\task2_sensitivity_forecast_consumer_priority_step001.csv`
  - `result\task2_sensitivity_forecast_consumer_priority_step001_summary.csv`
  - `result\task2_sensitivity_forecast_cpi_priority_step001.csv`
  - `result\task2_sensitivity_forecast_cpi_priority_step001_summary.csv`
  - `result\task2_sensitivity_forecast_energy_priority_step001.csv`
  - `result\task2_sensitivity_forecast_energy_priority_step001_summary.csv`
  - `result\task2_sensitivity_forecast_firm_priority_step001.csv`
  - `result\task2_sensitivity_forecast_firm_priority_step001_summary.csv`
  - `result\task2_sensitivity_forecast_summary_step001.csv`
  - `result\task2_sensitivity_forecast_volatility_priority_step001.csv`
  - `result\task2_sensitivity_forecast_volatility_priority_step001_summary.csv`
  - `result\task2_sensitivity_summary_step001.csv`
  - `result\task2_sensitivity_volatility_priority_step001.csv`
  - `result\task2_sensitivity_volatility_priority_step001_summary.csv`
  - `result\task2_six_alpha_version_comparison_step001.csv`
  - `result\task2_weight_scenarios.csv`
  - `result\task3_robustness_dimension_means.csv`
  - `result\task3_robustness_scenario_summary.csv`
  - `result\task3_rule_fit_metrics.csv`
  - `result\task3_simple_rule_alpha_path.csv`
  - `result\task3_simple_rule_dimension_mean_summary.csv`
  - `result\task3_simple_rule_strategy_comparison.csv`
  - `result\theta_special_diesel.csv`
  - `result\theta_special_gasoline.csv`
  - `result\theta_summary.csv`
  - `result\transmission_asymmetry_by_range.csv`
  - `result\transmission_asymmetry_overall.csv`
  - `result\transmission_asymmetry_ratio_by_range.csv`
  - `result\transmission_asymmetry_ratio_overall.csv`

字段识别结果：
- date: `date`
- gasoline_change: `gasoline_change`
- diesel_change: `diesel_change`
- is_special_regulated: `is_special_regulated`
- wti: `wti_mean`
- brent: `brent_mean`
- basket: `basket_mean`
- exchange_rate: `exchange_rate`

## 2. 干净样本筛选规则

clean_sample 同时满足：当前样本非特殊调控、上一期非特殊调控、上一期实际调价幅度不小于 50 元/吨、当前实际调价幅度不小于 50 元/吨、当前和上一期均非机制切换样本。

## 3. 回归模型公式

令 `X1_t = mu_t * nu * WTI_t`，`X2_t = mu_t * nu * Brent_t`，`X3_t = mu_t * nu * Basket_t`。

拟合调价幅度模型：

```text
Delta_P_t = gamma1 * Delta_X1_t + gamma2 * Delta_X2_t + gamma3 * Delta_X3_t + intercept
```

并令：

```text
alpha = gamma1 + gamma2 + gamma3
w_i = gamma_i / alpha
```

## 4. 汽油参数估计结果

- gamma1: 0.5768407600145631
- gamma2: 0.09660931423282237
- gamma3: 0.5441209430083251
- alpha: 1.2175710172557106
- w1_wti: 0.47376354384215497
- w2_brent: 0.07934593782510575
- w3_basket: 0.4468905183327393
- intercept: -0.07631600032304924
- MAE: 48.11884059704756
- RMSE: 65.13669013100164
- R2: 0.9291238881244535
- n: 161

## 5. 柴油参数估计结果

- gamma1: 0.5567132324693373
- gamma2: 0.0926964125333925
- gamma3: 0.5223455345126062
- alpha: 1.171755179515336
- w1_wti: 0.47511053691233207
- w2_brent: 0.07910902734113263
- w3_basket: 0.44578043574653536
- intercept: -0.003550484392768405
- MAE: 46.55951606842106
- RMSE: 62.65436828048324
- R2: 0.9291823264498265
- n: 161

## 6. 完整时序回代误差

汽油：
- normal_sample_MAE: 57.66227492161392
- all_sample_MAE: 61.913373063488294
- special_sample_MAE: 608.179484294345

柴油：
- normal_sample_MAE: 55.5374351055767
- all_sample_MAE: 59.60832461442565
- special_sample_MAE: 582.7176265015155

## 7. 特殊调控 theta_i 计算结果

详见：
- `result/theta_special_gasoline.csv`
- `result/theta_special_diesel.csv`

## 8. theta_i 是否近似常数

| fuel_type | valid_n | theta_mean | theta_std | theta_cv | judgement |
| --- | --- | --- | --- | --- | --- |
| gasoline | 2 | 1.85079 | 1.29572 | 0.700092 | theta_i 波动较大，不建议设为常数 |
| diesel | 2 | 1.82702 | 1.27216 | 0.696301 | theta_i 波动较大，不建议设为常数 |
| combined | 2 | 1.83891 | 1.28394 | 0.698209 | 综合 theta_i 波动较大，不建议设为常数 |

## 9. 如果模型效果仍然较差，可能原因

- 调价窗口是否严格使用了 10 个工作日，而不是 10 个自然日。
- 国内调价日期是否区分公告日和生效日。
- 特殊调控标记是否准确。
- 40、80、130 美元分段利润函数是否应纳入差分模型。
- 累计未调幅度是否按政策逻辑正确递推。
- 官方一揽子油种权重和部分成本项并未公开，公开数据只能近似识别。
