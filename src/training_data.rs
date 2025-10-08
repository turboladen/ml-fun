use polars::prelude::*;

pub struct TestingData {
    lazy_frame: LazyFrame,
}

impl TestingData {
    pub fn try_new() -> anyhow::Result<Self> {
        let lazy_frame = LazyCsvReader::new(PlPath::from_str("data/test.csv"))
            .with_has_header(true)
            .finish()?;

        Ok(Self { lazy_frame })
    }

    pub fn lazy_frame_cloned(&self) -> LazyFrame {
        self.lazy_frame.clone()
    }
}
