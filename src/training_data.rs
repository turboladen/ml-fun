use polars::prelude::*;

pub struct TrainingData {
    lazy_frame: LazyFrame,
}

impl TrainingData {
    pub fn try_new() -> anyhow::Result<Self> {
        let lazy_frame = LazyCsvReader::new(PlPath::from_str("data/train.csv"))
            .with_has_header(true)
            .finish()?;

        Ok(Self { lazy_frame })
    }

    pub fn percentage_of_sex_who_survived(&self, sex: &str) -> anyhow::Result<()> {
        let women = self
            .lazy_frame
            .clone()
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

    pub fn lazy_frame_cloned(&self) -> LazyFrame {
        self.lazy_frame.clone()
    }

    pub fn get_big_x(&self) -> anyhow::Result<DataFrame> {
        let big_x = self
            .lazy_frame
            .clone()
            .select([col("Pclass"), col("Sex"), col("SibSp"), col("Parch")])
            .collect()?;

        Ok(big_x.to_dummies(None, false, false)?)
    }
}
