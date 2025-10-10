mod conversions;
mod data;
pub mod random_forest;
mod testing_data;
mod training_data;

use crate::{
    conversions::*, data::Data, random_forest::RandomForestClassifier, testing_data::TestingData,
    training_data::TrainingData,
};
use ndarray::s;
use polars::prelude::col;

fn main() -> anyhow::Result<()> {
    println!("=== Loading Titanic Data ===");
    let training_data = TrainingData::try_new()?;
    let testing_data = TestingData::try_new()?;

    println!("\n=== Initial data inspection ===");
    let first_train = training_data.lazy_frame_cloned().first().collect()?;
    println!("First row of training data: {}", first_train);
    training_data.percentage_of_sex_who_survived("female")?;
    training_data.percentage_of_sex_who_survived("male")?;

    let first_test = testing_data.lazy_frame_cloned().first().collect()?;
    println!("First row of test data: {}", first_test);

    // Get features (X) and labels (y) for training
    println!("\n=== Preparing Training Data ===");
    let x_train_df =
        training_data.get_big_x([col("Pclass"), col("Sex"), col("SibSp"), col("Parch")])?;
    let x_train = dataframe_to_array2(&x_train_df)?;
    println!("Training features shape: {:?}", x_train.dim());

    let y_train_series = training_data.get_col_as_series("Survived")?;
    let y_train = series_to_array1(&y_train_series)?;
    println!("Training labels shape: {:?}", y_train.dim());

    // Train the Random Forest Classifier
    let model = {
        println!("\n=== Training Random Forest Classifier ===");
        let rf = RandomForestClassifier::new()
            .n_estimators(100)
            .max_depth(5)
            .random_state(1);

        println!("Fitting model with {} trees...", 100);
        rf.fit(x_train.clone(), y_train.clone())?
    };
    println!("Model trained successfully!");

    // Make predictions on the test set
    let predictions = {
        println!("\n=== Making Predictions on Test Data ===");
        let x_test_df =
            testing_data.get_big_x([col("Pclass"), col("Sex"), col("SibSp"), col("Parch")])?;
        let x_test = dataframe_to_array2(&x_test_df)?;
        println!("Test features shape: {:?}", x_test.dim());

        model.predict(&x_test)
    };
    println!("Generated {} predictions", predictions.len());
    println!("First 10 predictions: {:?}", &predictions.slice(s![..10]));

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

    // Calculate training accuracy
    let accuracy = {
        println!("\n=== Training Accuracy ===");
        let train_predictions = model.predict(&x_train);
        let correct = train_predictions
            .iter()
            .zip(y_train.iter())
            .filter(|(pred, actual)| pred == actual)
            .count();
        correct as f64 / y_train.len() as f64
    };
    println!("Training accuracy: {:.2}%", accuracy * 100.0);

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
