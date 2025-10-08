mod testing_data;
mod training_data;

use crate::{testing_data::TrainingData, training_data::TestingData};

fn main() -> anyhow::Result<()> {
    let training_data = TrainingData::try_new()?;
    let first_train = training_data.lazy_frame_cloned().first().collect()?;
    println!("First row of training data: {}", first_train);

    training_data.percentage_of_sex_who_survived("female")?;
    training_data.percentage_of_sex_who_survived("male")?;

    println!("--- Test Data ---");
    let test_data = TestingData::try_new()?;
    let first_test = test_data.lazy_frame_cloned().first().collect()?;
    println!("First row of test data: {}", first_test);

    Ok(())
}
