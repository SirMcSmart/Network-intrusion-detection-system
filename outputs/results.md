# Results (2026-11-17T15:45:51Z)

## KDDTest+: attack vs normal, threshold calibrated to 2% FPR on validation

| Model | Detection rate | FPR | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Isolation Forest | 0.6229 | 0.0212 | 0.9749 | 0.7601 | 0.9493 | 0.9605 |
| XGBoost | 0.8543 | 0.0476 | 0.9596 | 0.9039 | 0.9678 | 0.9705 |
| LSTM | 0.7566 | 0.0821 | 0.9241 | 0.8320 | 0.9496 | 0.9559 |

## KDDTest+: attack vs normal, each model's default decision

| Model | Detection rate | FPR | Precision | F1 |
|---|---|---|---|---|
| Isolation Forest | 0.6664 | 0.0269 | 0.9704 | 0.7902 |
| XGBoost | 0.6543 | 0.0287 | 0.9678 | 0.7807 |
| LSTM | 0.6477 | 0.0547 | 0.9400 | 0.7669 |

## KDDTest+: share flagged per category (calibrated threshold)

| Model | Normal | DoS | Probe | R2L | U2R | Novel attacks | Known attacks |
|---|---|---|---|---|---|---|---|
| Isolation Forest | 0.0212 | 0.7467 | 0.9054 | 0.0403 | 0.6100 | 0.5907 | 0.6362 |
| XGBoost | 0.0476 | 0.9694 | 0.9938 | 0.4223 | 0.8200 | 0.7896 | 0.8810 |
| LSTM | 0.0821 | 0.9200 | 0.7633 | 0.3377 | 0.3500 | 0.5131 | 0.8571 |

## KDDTest+: 5-class (supervised models)

| Model | Accuracy | Macro F1 | Normal recall | DoS recall | Probe recall | R2L recall | U2R recall |
|---|---|---|---|---|---|---|---|
| XGBoost | 0.7775 | 0.5318 | 0.9713 | 0.8458 | 0.6592 | 0.0661 | 0.0550 |
| LSTM | 0.7560 | 0.5428 | 0.9453 | 0.7894 | 0.6530 | 0.1376 | 0.0850 |

## KDDTest-21: attack vs normal, threshold calibrated to 2% FPR on validation

| Model | Detection rate | FPR | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Isolation Forest | 0.5053 | 0.0943 | 0.9602 | 0.6621 | 0.7716 | 0.9423 |
| XGBoost | 0.8072 | 0.1891 | 0.9506 | 0.8730 | 0.8367 | 0.9457 |
| LSTM | 0.6779 | 0.3541 | 0.8961 | 0.7719 | 0.7513 | 0.9246 |

## KDDTest-21: attack vs normal, each model's default decision

| Model | Detection rate | FPR | Precision | F1 |
|---|---|---|---|---|
| Isolation Forest | 0.5625 | 0.1190 | 0.9552 | 0.7080 |
| XGBoost | 0.5425 | 0.1292 | 0.9498 | 0.6906 |
| LSTM | 0.5338 | 0.2402 | 0.9092 | 0.6727 |

## KDDTest-21: share flagged per category (calibrated threshold)

| Model | Normal | DoS | Probe | R2L | U2R | Novel attacks | Known attacks |
|---|---|---|---|---|---|---|---|
| Isolation Forest | 0.0943 | 0.5737 | 0.9059 | 0.0403 | 0.6100 | 0.5904 | 0.4518 |
| XGBoost | 0.1891 | 0.9475 | 0.9938 | 0.4223 | 0.8200 | 0.7890 | 0.8186 |
| LSTM | 0.3541 | 0.8625 | 0.7614 | 0.3377 | 0.3500 | 0.5118 | 0.7821 |

## KDDTest-21: 5-class (supervised models)

| Model | Accuracy | Macro F1 | Normal recall | DoS recall | Probe recall | R2L recall | U2R recall |
|---|---|---|---|---|---|---|---|
| XGBoost | 0.5769 | 0.4441 | 0.8708 | 0.7351 | 0.6565 | 0.0661 | 0.0550 |
| LSTM | 0.5370 | 0.4420 | 0.7598 | 0.6382 | 0.6503 | 0.1376 | 0.0850 |
