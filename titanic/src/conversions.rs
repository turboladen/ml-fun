//! Conversion utilities between Polars DataFrames/Series and ndarray Arrays
//!
//! This module provides helper functions to convert between Polars' data structures
//! and ndarray's Array types, which is necessary when using Polars for data handling
//! and linfa for machine learning.

use anyhow::Result;
use ndarray::{Array1, Array2, s};
use polars::prelude::*;

/// Convert a Polars DataFrame to an ndarray Array2<f64>
///
/// This function converts all columns in the DataFrame to f64 and creates a 2D array
/// where rows represent samples and columns represent features.
///
/// # Arguments
/// * `df` - The Polars DataFrame to convert
///
/// # Returns
/// * `Result<Array2<f64>>` - A 2D array with shape (n_samples, n_features)
///
/// # Notes
/// - Null values are replaced with 0.0
/// - All columns are cast to Float64
/// - Polars stores data in column-major format, so we need to transpose
///
/// # Example
/// ```ignore
/// let df = training_data.get_big_x()?;
/// let x_train = dataframe_to_array2(&df)?;
/// println!("Shape: {:?}", x_train.dim());
/// ```
pub fn dataframe_to_array2(df: &DataFrame) -> Result<Array2<f64>> {
    let nrows = df.height();
    let ncols = df.width();

    let mut data = Vec::with_capacity(nrows * ncols);

    for col in df.get_columns() {
        // Convert each column to f64
        let col_data = col.cast(&DataType::Float64)?;
        let ca = col_data.f64()?;

        for val in ca.iter() {
            data.push(val.unwrap_or(0.0)); // Handle nulls as 0.0
        }
    }

    // Polars stores column-major, so we need to reshape accordingly
    Ok(Array2::from_shape_vec((ncols, nrows), data)?.reversed_axes())
}

/// Convert a Polars Series to an ndarray Array1<usize>
///
/// This function extracts integer values from a Series and converts them to usize,
/// which is the label type expected by linfa models.
///
/// # Arguments
/// * `series` - The Polars Series to convert (should contain i64 values)
///
/// # Returns
/// * `Result<Array1<usize>>` - A 1D array of labels
///
/// # Notes
/// - Expects series to contain i64 values
/// - Null values are replaced with 0
/// - Values are cast to usize
///
/// # Example
/// ```ignore
/// let y_series = training_data.get_survived()?;
/// let y_train = series_to_array1(&y_series)?;
/// ```
pub fn series_to_array1(series: &Series) -> Result<Array1<usize>> {
    let ca = series.i64()?;
    let vec: Vec<usize> = ca.iter().map(|v| v.unwrap_or(0) as usize).collect();

    Ok(Array1::from_vec(vec))
}

/// Convert a Polars Series to an ndarray Array1<i64>
///
/// Similar to `series_to_array1` but preserves i64 type for IDs.
///
/// # Arguments
/// * `series` - The Polars Series to convert
///
/// # Returns
/// * `Result<Array1<i64>>` - A 1D array of i64 values
///
/// # Example
/// ```ignore
/// let ids_series = testing_data.get_passenger_ids()?;
/// let ids = series_to_array1_i64(&ids_series)?;
/// ```
pub fn series_to_array1_i64(series: &Series) -> Result<Array1<i64>> {
    let ca = series.i64()?;
    let vec: Vec<i64> = ca.iter().map(|v| v.unwrap_or(0)).collect();

    Ok(Array1::from_vec(vec))
}

/// Convert ndarray predictions and IDs to a Polars DataFrame
///
/// This function creates a DataFrame suitable for Kaggle submission format,
/// with PassengerId and Survived columns.
///
/// # Arguments
/// * `predictions` - Array of predicted class labels (0 or 1)
/// * `passenger_ids` - Array of passenger IDs corresponding to predictions
///
/// # Returns
/// * `Result<DataFrame>` - DataFrame with PassengerId and Survived columns
///
/// # Example
/// ```ignore
/// let submission_df = array1_to_dataframe(&predictions, &passenger_ids)?;
/// ```
pub fn array1_to_dataframe(
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

/// Write a DataFrame to a CSV file
///
/// # Arguments
/// * `df` - The DataFrame to write
/// * `path` - Path to the output CSV file
///
/// # Returns
/// * `Result<()>` - Success or error
///
/// # Example
/// ```ignore
/// write_csv_file(&submission_df, "data/submission.csv")?;
/// ```
pub fn write_csv_file(df: &mut DataFrame, path: &str) -> Result<()> {
    let mut file = std::fs::File::create(path)?;

    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(df)?;

    Ok(())
}

/// Calculate accuracy given predictions and actual labels
pub fn calculate_accuracy(predictions: &Array1<usize>, actual: &Array1<usize>) -> f64 {
    let correct = predictions
        .iter()
        .zip(actual.iter())
        .filter(|(pred, act)| pred == act)
        .count();
    correct as f64 / actual.len() as f64
}

/// Split arrays into training and validation sets
///
/// This function splits feature and target arrays into training and validation sets
/// based on the provided ratio. The split is sequential (not shuffled).
///
/// # Arguments
/// * `x` - Feature matrix (rows = samples, columns = features)
/// * `y` - Target vector (class labels)
/// * `ratio` - Proportion for training set (e.g., 0.8 = 80% train, 20% validation)
///
/// # Returns
/// * Tuple of (x_train, x_val, y_train, y_val)
///
/// # Example
/// ```ignore
/// let (x_train, x_val, y_train, y_val) = train_test_split(x, y, 0.8);
/// println!("Training samples: {}, Validation samples: {}", x_train.nrows(), x_val.nrows());
/// ```
pub fn train_test_split(
    x: Array2<f64>,
    y: Array1<usize>,
    ratio: f32,
) -> (Array2<f64>, Array2<f64>, Array1<usize>, Array1<usize>) {
    let n_samples = x.nrows();
    let split_idx = (n_samples as f32 * ratio) as usize;

    // Split features
    let x_train = x.slice(s![..split_idx, ..]).to_owned();
    let x_val = x.slice(s![split_idx.., ..]).to_owned();

    // Split targets
    let y_train = y.slice(s![..split_idx]).to_owned();
    let y_val = y.slice(s![split_idx..]).to_owned();

    (x_train, x_val, y_train, y_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_array1_to_dataframe() {
        let predictions = arr1(&[0, 1, 0, 1]);
        let ids = arr1(&[892, 893, 894, 895]);

        let df = array1_to_dataframe(&predictions, &ids).unwrap();

        assert_eq!(df.height(), 4);
        assert_eq!(df.width(), 2);
        assert!(df.column("PassengerId").is_ok());
        assert!(df.column("Survived").is_ok());
    }

    #[test]
    fn test_series_to_array1() {
        let series = Series::new("test".into(), vec![0i64, 1i64, 0i64, 1i64]);
        let array = series_to_array1(&series).unwrap();

        assert_eq!(array.len(), 4);
        assert_eq!(array[0], 0);
        assert_eq!(array[1], 1);
    }

    #[test]
    fn test_series_to_array1_with_nulls() {
        let series = Series::new("test".into(), vec![Some(1i64), None, Some(0i64)]);
        let array = series_to_array1(&series).unwrap();

        assert_eq!(array.len(), 3);
        assert_eq!(array[0], 1);
        assert_eq!(array[1], 0); // null becomes 0
        assert_eq!(array[2], 0);
    }

    #[test]
    fn test_train_test_split() {
        use ndarray::{arr1, arr2};

        let x = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]);
        let y = arr1(&[0, 1, 0, 1, 0]);

        let (x_train, x_val, y_train, y_val) = train_test_split(x, y, 0.8);

        // With 5 samples and 0.8 ratio, we should get 4 training and 1 validation
        assert_eq!(x_train.nrows(), 4);
        assert_eq!(x_val.nrows(), 1);
        assert_eq!(y_train.len(), 4);
        assert_eq!(y_val.len(), 1);

        // Check that the split happened correctly
        assert_eq!(x_train[[0, 0]], 1.0);
        assert_eq!(x_val[[0, 0]], 9.0);
        assert_eq!(y_train[0], 0);
        assert_eq!(y_val[0], 0);
    }
}
