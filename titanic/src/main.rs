mod conversions;
mod data;
pub mod random_forest;
mod testing_data;
mod training_data;

use crate::{
    conversions::*, data::Data, random_forest::RandomForestClassifier, testing_data::TestingData,
    training_data::TrainingData,
};
use polars::prelude::col;

fn main() -> anyhow::Result<()> {
    println!("=== Loading Titanic Data ===");
    let training_data = TrainingData::try_new()?;

    // println!("\n=== Initial data inspection ===");
    // let first_train = training_data.lazy_frame_cloned().first().collect()?;
    // println!("First row of training data: {}", first_train);
    // training_data.percentage_of_sex_who_survived("female")?;
    // training_data.percentage_of_sex_who_survived("male")?;

    let (x_train, x_validation, y_train, y_validation) =
        prepare_for_test_train_split(&training_data)?;

    // Train the Random Forest Classifier on training split
    let model = {
        println!("\n=== Training Random Forest Classifier (on training split) ===");
        let rf = RandomForestClassifier::new()
            .n_estimators(100)
            .max_depth(5)
            .random_state(1);

        println!("Fitting model with {} trees...", 100);
        rf.fit(x_train.clone(), y_train.clone())?
    };
    println!("Model trained successfully!");

    // Evaluate on validation set
    println!("\n=== Validation Performance ===");
    let val_predictions = model.predict(&x_validation);
    let val_accuracy = calculate_accuracy(&val_predictions, &y_validation);
    println!("Validation accuracy: {:.2}%", val_accuracy * 100.0);

    // Calculate training accuracy on the split
    println!("\n=== Training Accuracy (on training split) ===");
    let train_predictions = model.predict(&x_train);
    let train_accuracy = calculate_accuracy(&train_predictions, &y_train);
    println!("Training accuracy: {:.2}%", train_accuracy * 100.0);

    // Now retrain on the FULL dataset for final predictions
    println!("\n=== Retraining on Full Dataset for Final Predictions ===");
    let x_full_df = training_data.get_feature_matrix([
        col("Pclass"),
        col("Sex"),
        col("Age"),
        col("Fare"),
        col("SibSp"),
        col("Parch"),
    ])?;
    let x_full = dataframe_to_array2(&x_full_df)?;
    let y_full_series = training_data.get_col_as_series("Survived")?;
    let y_full = series_to_array1(&y_full_series)?;

    let final_model = {
        let rf = RandomForestClassifier::new()
            .n_estimators(100)
            .max_depth(5)
            .random_state(1);

        println!("Training on all {} samples...", x_full.nrows());
        rf.fit(x_full, y_full)?
    };
    println!("Final model trained on full dataset!");

    let testing_data = TestingData::try_new()?;
    // let first_test = testing_data.lazy_frame_cloned().first().collect()?;
    // println!("First row of test data: {}", first_test);

    // Make predictions on the test set
    let predictions = {
        println!("\n=== Making Predictions on Test Data ===");
        let x_test_df = testing_data.get_feature_matrix([
            col("Pclass"),
            col("Sex"),
            col("Age"),
            col("Fare"),
            col("SibSp"),
            col("Parch"),
        ])?;
        let x_test = dataframe_to_array2(&x_test_df)?;
        println!("Test features shape: {:?}", x_test.dim());

        final_model.predict(&x_test)
    };
    println!("Generated {} predictions", predictions.len());
    // println!("First 10 predictions: {:?}", &predictions.slice(s![..10]));

    // Get PassengerIds from test data for the submission file
    let mut submission_df = {
        let passenger_ids_series = testing_data.get_col_as_series("PassengerId")?;
        let passenger_ids = series_to_array1_i64(&passenger_ids_series)?;

        // Convert predictions to DataFrame
        array1_to_dataframe(&predictions, &passenger_ids)?
    };
    println!("\nSubmission DataFrame:");
    println!("{}", submission_df);

    // Write to CSV
    write_csv_file(&mut submission_df, "data/submission.csv")?;
    println!("\n✅ Predictions saved to data/submission.csv");

    println!("\n=== Summary ===");
    println!(
        "Survived predictions: {}",
        predictions.iter().filter(|&&x| x == 1).count()
    );
    println!(
        "Did not survive predictions: {}",
        predictions.iter().filter(|&&x| x == 0).count()
    );

    Ok(())
}

fn prepare_for_test_train_split(
    training_data: &TrainingData,
) -> anyhow::Result<(
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
    ndarray::Array1<usize>,
    ndarray::Array1<usize>,
)> {
    // Get features (X) and labels (y) for training
    println!("\n=== Preparing Training Data ===");
    let x_df = training_data.get_feature_matrix([
        col("Pclass"),
        col("Sex"),
        col("Age"),
        col("Fare"),
        col("SibSp"),
        col("Parch"),
    ])?;
    let x = dataframe_to_array2(&x_df)?;
    println!("Full dataset features shape: {:?}", x.dim());

    let y = {
        let y_series = training_data.get_col_as_series("Survived")?;
        series_to_array1(&y_series)?
    };
    println!("Full dataset labels shape: {:?}", y.dim());

    // Split into training and validation sets
    println!("\n=== Splitting Data for Validation ===");
    let (x_train, x_validation, y_train, y_validation) = train_test_split(x, y, 0.8);
    println!("Training set: {} samples", x_train.nrows());
    println!("Validation set: {} samples", x_validation.nrows());

    Ok((x_train, x_validation, y_train, y_validation))
}
