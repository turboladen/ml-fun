use polars::prelude::*;

fn main() -> anyhow::Result<()> {
    do_test_data()?;
    do_train_data()?;

    Ok(())
}

fn do_train_data() -> anyhow::Result<()> {
    let train_data = LazyCsvReader::new(PlPath::from_str("data/train.csv"))
        .with_has_header(true)
        .finish()?;

    // let first_train = train_data.first().collect()?;
    // dbg!(first_train);

    let women = train_data
        .filter(col("Sex").eq(lit("female")))
        // .select([col("Survived")])
        .select([
            col("Survived").sum().alias("survived_sum"),
            col("Survived").count().alias("total_count"),
        ])
        .collect()?;

    // let survived_sum: f64 = women.column("Survived")?.sum_reduce()?.into();
    //
    // let total_count = women.height() as f64;
    // let rate_women = survived_sum / total_count;

    // let women = train_data
    //     .filter(col("Sex").eq("female"))
    //     .select([col("Survived")]);
    //
    // let rate_women = women.sum::<Option<u32>>().collect()? / women.count().collect()?;
    // dbg!(women);

    let rate = women.column("survived_sum")?.get(0)?.try_extract::<f64>()?
        / women.column("total_count")?.get(0)?.try_extract::<f64>()?;
    dbg!(rate);
    Ok(())
}

fn do_test_data() -> anyhow::Result<()> {
    let test_data = LazyCsvReader::new(PlPath::from_str("data/test.csv"))
        .with_has_header(true)
        .finish()?;

    let first_test = test_data.first().collect()?;
    dbg!(first_test);

    Ok(())
}
