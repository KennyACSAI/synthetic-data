# Synthetic Earthquake Dataset Report

## Dataset Summary

Total events: 15979
Real events: 15932
Synthetic events: 47

## Synthetic Data Methods

1. **Bootstrap method**: 17 events
   - Created by scaling up moderate (M5-6) real events
   - Sample weight: 0.3

2. **Physics-based method**: 20 events
   - Generated using fault geometry and physical parameters
   - Based on Gutenberg-Richter relation with b-value = 0.77
   - Sample weight: 0.5

3. **Simple method**: 10 events
   - Created by spatial jittering from template events
   - Sample weight: 0.2

## Magnitude Distribution

```
method      bootstrap  physics  real  simple
mag_range                                   
(3.0, 4.0]          0        0  1598       0
(4.0, 5.0]          0        0   136       0
(5.0, 6.0]          0        0    16       0
(6.0, 7.0]          5       17     1       5
(7.0, 8.0]         12        3     0       5
```

## Time Period

Date range: 2003-01-01 00:00:00 to 2025-08-27 00:33:48

## Cross-Validation Folds

The dataset is divided into time-based CV folds:

- Fold 0 (2003-2005): 1547 events
- Fold 1 (2006-2008): 1289 events
- Fold 2 (2009-2011): 3397 events
- Fold 3 (2012-2014): 3647 events
- Fold 4 (2015-2017): 2192 events
- Fold 5 (2018-2020): 1558 events
- Fold 6 (2021-2025): 2349 events

## Usage for Forecasting

When training earthquake forecasting models:

1. Use the `sample_weight` column to give appropriate importance to each event
2. Use the `is_synthetic` column to differentiate between real and synthetic events
3. Use the `cv_fold` column for time-based cross-validation
4. Consider evaluating model performance with and without synthetic data
