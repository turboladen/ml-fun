use polars::prelude::*;

fn main() -> anyhow::Result<()> {
    do_train_data()?;
    do_test_data()?;

    Ok(())
}

fn do_train_data() -> anyhow::Result<()> {
    let train_data = LazyCsvReader::new(PlPath::from_str("data/train.csv"))
        .with_has_header(true)
        .finish()?;

    let first_train = train_data.clone().first().collect()?;
    println!("First row of training data: {}", first_train);

    percentage_of_sex_who_survived(train_data.clone(), "female")?;
    percentage_of_sex_who_survived(train_data.clone(), "male")?;

    Ok(())
}

fn percentage_of_sex_who_survived(train_data: LazyFrame, sex: &str) -> anyhow::Result<()> {
    let women = train_data
        .filter(col("Sex").eq(lit(sex)))
        .select([
            col("Survived").sum().alias("survived_sum"),
            col("Survived").count().alias("total_count"),
        ])
        .collect()?;

    let rate = women.column("survived_sum")?.get(0)?.try_extract::<f64>()?
        / women.column("total_count")?.get(0)?.try_extract::<f64>()?;

    println!(
        "Percentage of {sex}s who survived: {:.2}% ({rate})",
        rate * 100.0
    );

    Ok(())
}

fn do_test_data() -> anyhow::Result<()> {
    let test_data = LazyCsvReader::new(PlPath::from_str("data/test.csv"))
        .with_has_header(true)
        .finish()?;

    let first_test = test_data.first().collect()?;
    println!("First row of test data: {}", first_test);

    Ok(())
}
