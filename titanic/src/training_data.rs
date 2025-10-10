use polars::prelude::{LazyCsvReader, LazyFileListReader, LazyFrame, PlPath, col, lit};

use crate::data::Data;

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
}

impl Data for TrainingData {
    fn lazy_frame_cloned(&self) -> polars::prelude::LazyFrame {
        self.lazy_frame.clone()
    }
}
