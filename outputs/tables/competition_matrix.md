# Competition matrix

Test MAE of the model selected on validation among all models, baselines included (rows: input window V, columns: output window H).

|   V \ H | 1                          | 5                          | 30                         | 90                         |
|--------:|:---------------------------|:---------------------------|:---------------------------|:---------------------------|
|       5 | 0.012296 (mlp_deep)        | 0.005599 (historical_mean) | 0.002322 (mlp_deep)        | 0.001269 (historical_mean) |
|      10 | 0.012286 (historical_mean) | 0.005600 (historical_mean) | 0.002322 (mlp_deep)        | 0.001269 (historical_mean) |
|      30 | 0.012290 (historical_mean) | 0.005603 (historical_mean) | 0.002329 (mlp_deep)        | 0.001269 (historical_mean) |
|      90 | 0.012309 (historical_mean) | 0.005614 (historical_mean) | 0.002325 (historical_mean) | 0.001271 (historical_mean) |

## Winners

|   input_window |   output_window | model           | family   |   mae_val |   mae_test |   n_params | best_baseline   |   dm_p_value |    n_seeds |   mae_test_mean |   mae_test_std |
|---------------:|----------------:|:----------------|:---------|----------:|-----------:|-----------:|:----------------|-------------:|-----------:|----------------:|---------------:|
|              5 |               1 | mlp_deep        | dense    |  0.009011 |   0.012296 |      10263 | historical_mean |     0.451802 |   3.000000 |        0.012318 |       0.000029 |
|              5 |               5 | historical_mean | baseline |  0.004135 |   0.005599 |         23 | historical_mean |     1.000000 | nan        |      nan        |     nan        |
|              5 |              30 | mlp_deep        | dense    |  0.001702 |   0.002322 |      10263 | historical_mean |     0.889283 |   3.000000 |        0.002323 |       0.000001 |
|              5 |              90 | historical_mean | baseline |  0.000924 |   0.001269 |         23 | historical_mean |     1.000000 | nan        |      nan        |     nan        |
|             10 |               1 | historical_mean | baseline |  0.009013 |   0.012286 |         23 | historical_mean |     1.000000 | nan        |      nan        |     nan        |
|             10 |               5 | historical_mean | baseline |  0.004135 |   0.005600 |         23 | historical_mean |     1.000000 | nan        |      nan        |     nan        |
|             10 |              30 | mlp_deep        | dense    |  0.001702 |   0.002322 |      17623 | historical_mean |     0.830156 |   3.000000 |        0.002321 |       0.000001 |
|             10 |              90 | historical_mean | baseline |  0.000925 |   0.001269 |         23 | historical_mean |     1.000000 | nan        |      nan        |     nan        |
|             30 |               1 | historical_mean | baseline |  0.009023 |   0.012290 |         23 | historical_mean |     1.000000 | nan        |      nan        |     nan        |
|             30 |               5 | historical_mean | baseline |  0.004139 |   0.005603 |         23 | historical_mean |     1.000000 | nan        |      nan        |     nan        |
|             30 |              30 | mlp_deep        | dense    |  0.001702 |   0.002329 |      47063 | historical_mean |     0.257088 |   3.000000 |        0.002325 |       0.000003 |
|             30 |              90 | historical_mean | baseline |  0.000924 |   0.001269 |         23 | historical_mean |     1.000000 | nan        |      nan        |     nan        |
|             90 |               1 | historical_mean | baseline |  0.009030 |   0.012309 |         23 | historical_mean |     1.000000 | nan        |      nan        |     nan        |
|             90 |               5 | historical_mean | baseline |  0.004139 |   0.005614 |         23 | historical_mean |     1.000000 | nan        |      nan        |     nan        |
|             90 |              30 | historical_mean | baseline |  0.001704 |   0.002325 |         23 | historical_mean |     1.000000 | nan        |      nan        |     nan        |
|             90 |              90 | historical_mean | baseline |  0.000925 |   0.001271 |         23 | historical_mean |     1.000000 | nan        |      nan        |     nan        |

## Best neural network per cell

Same selection rule restricted to the four network families; `dm_p_value` tests it against the cell's best validation baseline and the seed columns show its test MAE over independent re-trainings.

|   V \ H | 1                   | 5                   | 30                  | 90                  |
|--------:|:--------------------|:--------------------|:--------------------|:--------------------|
|       5 | 0.012296 (mlp_deep) | 0.005606 (cnn_deep) | 0.002322 (mlp_deep) | 0.001267 (mlp_deep) |
|      10 | 0.012301 (mlp_deep) | 0.005601 (conv_gru) | 0.002322 (mlp_deep) | 0.001270 (mlp_deep) |
|      30 | 0.012291 (cnn_deep) | 0.005617 (cnn_deep) | 0.002329 (mlp_deep) | 0.001269 (mlp_deep) |
|      90 | 0.012314 (cnn_deep) | 0.005614 (conv_gru) | 0.002331 (mlp_deep) | 0.001274 (mlp_deep) |

|   input_window |   output_window | model    | family        |   mae_val |   mae_test |   n_params | best_baseline   |   dm_p_value |   n_seeds |   mae_test_mean |   mae_test_std |
|---------------:|----------------:|:---------|:--------------|----------:|-----------:|-----------:|:----------------|-------------:|----------:|----------------:|---------------:|
|              5 |               1 | mlp_deep | dense         |  0.009011 |   0.012296 |      10263 | historical_mean |     0.451802 |         3 |        0.012318 |       0.000029 |
|              5 |               5 | cnn_deep | convolutional |  0.004137 |   0.005606 |       9815 | historical_mean |     0.389320 |         3 |        0.005606 |       0.000001 |
|              5 |              30 | mlp_deep | dense         |  0.001702 |   0.002322 |      10263 | historical_mean |     0.889283 |         3 |        0.002323 |       0.000001 |
|              5 |              90 | mlp_deep | dense         |  0.000925 |   0.001267 |      10263 | historical_mean |     0.798074 |         3 |        0.001268 |       0.000001 |
|             10 |               1 | mlp_deep | dense         |  0.009025 |   0.012301 |      17623 | historical_mean |     0.000642 |         3 |        0.012304 |       0.000002 |
|             10 |               5 | conv_gru | mixed         |  0.004138 |   0.005601 |      10487 | historical_mean |     0.590415 |         3 |        0.005602 |       0.000001 |
|             10 |              30 | mlp_deep | dense         |  0.001702 |   0.002322 |      17623 | historical_mean |     0.830156 |         3 |        0.002321 |       0.000001 |
|             10 |              90 | mlp_deep | dense         |  0.000925 |   0.001270 |      17623 | historical_mean |     0.624845 |         3 |        0.001269 |       0.000001 |
|             30 |               1 | cnn_deep | convolutional |  0.009025 |   0.012291 |       9815 | historical_mean |     0.933430 |         3 |        0.012293 |       0.000001 |
|             30 |               5 | cnn_deep | convolutional |  0.004142 |   0.005617 |       9815 | historical_mean |     0.333159 |         3 |        0.005604 |       0.000009 |
|             30 |              30 | mlp_deep | dense         |  0.001702 |   0.002329 |      47063 | historical_mean |     0.257088 |         3 |        0.002325 |       0.000003 |
|             30 |              90 | mlp_deep | dense         |  0.000925 |   0.001269 |      47063 | historical_mean |     0.977039 |         3 |        0.001270 |       0.000001 |
|             90 |               1 | cnn_deep | convolutional |  0.009033 |   0.012314 |       9815 | historical_mean |     0.327073 |         3 |        0.012313 |       0.000002 |
|             90 |               5 | conv_gru | mixed         |  0.004141 |   0.005614 |      10487 | historical_mean |     0.982673 |         3 |        0.005615 |       0.000002 |
|             90 |              30 | mlp_deep | dense         |  0.001705 |   0.002331 |     135383 | historical_mean |     0.043867 |         3 |        0.002330 |       0.000001 |
|             90 |              90 | mlp_deep | dense         |  0.000926 |   0.001274 |     135383 | historical_mean |     0.353408 |         3 |        0.001274 |       0.000000 |
