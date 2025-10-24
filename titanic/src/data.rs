use polars::{
    frame::DataFrame,
    prelude::{DataFrameOps, Expr, LazyFrame, Series, col},
};

pub trait Data {
    fn lazy_frame_cloned(&self) -> LazyFrame;

    fn get_col_as_series(&self, col_name: &str) -> anyhow::Result<Series> {
        let df = self.lazy_frame_cloned().select([col(col_name)]).collect()?;

        Ok(df.column(col_name)?.clone().as_series().unwrap().clone())
    }

    fn get_feature_matrix<E>(&self, exprs: E) -> anyhow::Result<DataFrame>
    where
        E: AsRef<[Expr]>,
    {
        let big_x = self.lazy_frame_cloned().select(exprs.as_ref()).collect()?;

        Ok(big_x.to_dummies(None, false, false)?)
    }
}
