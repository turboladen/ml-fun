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
        let df = self.lazy_frame_cloned().select(exprs.as_ref()).collect()?;

        // Separate categorical and numeric columns
        let mut result_dfs = Vec::new();

        for col_name in df.get_column_names() {
            let column = df.column(col_name)?;

            // Only create dummies for string/categorical columns
            if column.dtype().is_numeric() {
                // Keep numeric columns as-is, but fill nulls with 0
                let filled = self
                    .lazy_frame_cloned()
                    .select([col(col_name.as_str()).fill_null(0)])
                    .collect()?;
                result_dfs.push(filled);
            } else {
                // Create dummy variables for categorical columns
                let col_df = self
                    .lazy_frame_cloned()
                    .select([col(col_name.as_str())])
                    .collect()?;
                let dummies = col_df.to_dummies(None, false, false)?;
                result_dfs.push(dummies);
            }
        }

        // Horizontally concatenate all dataframes
        let mut result = result_dfs[0].clone();
        for df in &result_dfs[1..] {
            result = result.hstack(df.get_columns())?;
        }

        Ok(result)
    }
}
