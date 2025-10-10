# Random Forest Classifier for Rust using Linfa

This document explains how we implemented a Random Forest Classifier in Rust using the `linfa`
ecosystem, specifically the `linfa-ensemble` and `linfa-trees` crates.

## Overview

The `linfa-ensemble` crate doesn't provide a Random Forest Classifier out of the box, but it does
provide an `EnsembleLearner` that implements **bootstrap aggregating (bagging)** with majority
voting. A Random Forest is essentially:

**Random Forest = Decision Trees + Bagging**

So we can create a Random Forest by combining:

- `DecisionTree` from `linfa-trees` (base estimator)
- `EnsembleLearner` from `linfa-ensemble` (handles bootstrap sampling and voting)

## Implementation

### Key Components

1. **`RandomForestClassifier`** - Builder pattern for configuration
   - `n_estimators`: Number of trees in the forest (default: 100)
   - `max_depth`: Maximum depth of each tree (default: 10)
   - `min_samples_split`: Minimum samples to split a node (default: 2)
   - `bootstrap_proportion`: Proportion of data for each bootstrap sample (default: 1.0)
   - `random_state`: Seed for reproducibility (default: None)

2. **`FittedRandomForest`** - Trained model that makes predictions

### Architecture from linfa-ensemble

The `EnsembleLearner` (from the algorithm.rs file) provides:

- **Bootstrap sampling**: Creates multiple training sets by sampling with replacement
- **Parallel training**: Trains multiple models on different bootstrap samples
- **Majority voting**: Aggregates predictions using a voting mechanism

```rust
// From linfa-ensemble/src/algorithm.rs
impl<F, T, M> PredictInplace<Array2<F>, T> for EnsembleLearner<M> {
    fn predict_inplace(&self, x: &Array2<F>, y: &mut T) {
        // Generate predictions from each model
        let predictions = self.generate_predictions(x);

        // Use HashMap to count votes for each class
        let mut prediction_maps = y_array.map(|_| HashMap::new());
        for prediction in predictions {
            // Count votes for each prediction
            Zip::from(&mut prediction_maps)
                .and(&p_arr)
                .for_each(|map, val| *map.entry(*val).or_insert(0) += 1);
        }

        // Pick the class with the most votes
        let agg_preds = prediction_maps.map(|map|
            map.iter().max_by_key(|(_, v)| **v).unwrap().0
        );
        // ... assign to y
    }
}
```

## Usage Example

```rust
use ndarray::{Array1, Array2};

// Create and configure the classifier
let rf = RandomForestClassifier::new()
    .n_estimators(100)      // 100 trees
    .max_depth(5)           // Max depth of 5
    .random_state(1);       // For reproducibility

// Fit the model
let model = rf.fit(x_train, y_train)?;

// Make predictions
let predictions = model.predict(&x_test);
```

## Type Requirements

The label type `L` must satisfy:

- `'static + Clone + Copy + Ord + Label`
- `Label` is from the `linfa` crate

For the Titanic dataset, we use `usize` which implements all required traits.

## Data Conversion

Since we use Polars for data handling and linfa uses ndarray, we need conversion functions in both
directions:

### Polars → ndarray (for model input)

```rust
// DataFrame -> Array2<f64>
fn dataframe_to_array2(df: &DataFrame) -> Result<Array2<f64>> {
    let nrows = df.height();
    let ncols = df.width();
    let mut data = Vec::with_capacity(nrows * ncols);

    for col in df.get_columns() {
        let col_data = col.cast(&DataType::Float64)?;
        let ca = col_data.f64()?;
        for val in ca.iter() {
            data.push(val.unwrap_or(0.0)); // Handle nulls as 0.0
        }
    }

    // Polars stores column-major, so we need to reshape accordingly
    Ok(Array2::from_shape_vec((ncols, nrows), data)?.reversed_axes())
}

// Series -> Array1<usize> (for labels)
fn series_to_array1(series: &Series) -> Result<Array1<usize>> {
    let ca = series.i64()?;
    let vec: Vec<usize> = ca.iter().map(|v| v.unwrap_or(0) as usize).collect();
    Ok(Array1::from_vec(vec))
}
```

### ndarray → Polars (for predictions/output)

```rust
// Array1<usize> predictions -> DataFrame (for CSV export)
fn array1_to_dataframe(
    predictions: &Array1<usize>,
    passenger_ids: &Array1<i64>,
) -> Result<DataFrame> {
    let survived: Vec<i64> = predictions.iter().map(|&x| x as i64).collect();
    let ids: Vec<i64> = passenger_ids.iter().copied().collect();

    let df = DataFrame::new(vec![
        Series::new("PassengerId".into(), ids).into(),
        Series::new("Survived".into(), survived).into(),
    ])?;

    Ok(df)
}
```

### Writing predictions to CSV

```rust
// Convert predictions to DataFrame
let submission_df = array1_to_dataframe(&predictions, &passenger_ids_array)?;

// Write to CSV using Polars
let mut file = std::fs::File::create("data/submission.csv")?;
CsvWriter::new(&mut file)
    .include_header(true)
    .with_separator(b',')
    .finish(&mut submission_df.clone())?;
```

## Results on Titanic Dataset

- **Training Shape**: (891 samples, 19 features after one-hot encoding)
- **Test Shape**: (418 samples, 20 features)
- **Training Accuracy**: 78.68%
- **Predictions**: 152 survived, 266 did not survive
- **Output**: CSV file with PassengerId and Survived columns (ready for Kaggle submission)

## Comparison to sklearn

| Feature            | sklearn.RandomForestClassifier | This Implementation             |
| ------------------ | ------------------------------ | ------------------------------- |
| Base Estimator     | DecisionTreeClassifier         | linfa_trees::DecisionTree       |
| Bagging            | Built-in                       | linfa_ensemble::EnsembleLearner |
| Voting             | Majority voting                | Majority voting via HashMap     |
| Bootstrap          | ✓                              | ✓                               |
| n_estimators       | ✓                              | ✓                               |
| max_depth          | ✓                              | ✓                               |
| min_samples_split  | ✓                              | ✓ (via min_weight_split)        |
| Feature subsetting | ✓                              | ✗ (not yet in linfa-trees)      |

## Dependencies

```toml
[dependencies]
linfa = "0.8.0"
linfa-ensemble = "0.8.0"
linfa-trees = "0.8.0"
ndarray = "0.16"
rand = "0.8"
rand_xoshiro = "0.6"
```

## Limitations

1. **No feature subsetting**: The current implementation doesn't randomly select features at each
   split (which is part of what makes Random Forests "random"). This would need to be added to
   `linfa-trees`.

2. **Feature shape mismatch**: Notice the test set has 20 features vs 19 in training. This is
   because one-hot encoding may create different columns. You'll need to align features between
   train and test sets.

3. **Memory**: All trees are kept in memory. For very large forests, this could be an issue.

## Future Improvements

- Add feature subsetting at each split (true Random Forest behavior)
- Add out-of-bag (OOB) error estimation
- Add feature importance calculation
- Handle categorical features more elegantly
- Parallelize tree training (linfa may already do this internally)

## References

- [linfa-ensemble GitHub](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-ensemble)
- [Kaggle Titanic Tutorial](https://www.kaggle.com/code/alexisbcook/titanic-tutorial)
- [Random Forest (Wikipedia)](https://en.wikipedia.org/wiki/Random_forest)
