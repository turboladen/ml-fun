use polars::prelude::{LazyCsvReader, LazyFileListReader, LazyFrame, PlPath};

use crate::data::Data;

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
}

impl Data for TestingData {
    fn lazy_frame_cloned(&self) -> polars::prelude::LazyFrame {
        self.lazy_frame.clone()
    }
}
