mod testing_data;
mod training_data;

use crate::{testing_data::TestingData, training_data::TrainingData};

fn main() -> anyhow::Result<()> {
    let training_data = TrainingData::try_new()?;
    let first_train = training_data.lazy_frame_cloned().first().collect()?;
    println!("First row of training data: {}", first_train);

    training_data.percentage_of_sex_who_survived("female")?;
    training_data.percentage_of_sex_who_survived("male")?;

    println!("--- Test Data ---");
    let testing_data = TestingData::try_new()?;
    let first_test = testing_data.lazy_frame_cloned().first().collect()?;
    println!("First row of test data: {}", first_test);

    println!("--- Big X ---");
    let big_x = training_data.get_big_x()?;
    dbg!("Big X from training data:\n{}", big_x);
    let big_x_test = testing_data.get_big_x()?;
    dbg!("Big X from testing data:\n{}", big_x_test);

    Ok(())
}
