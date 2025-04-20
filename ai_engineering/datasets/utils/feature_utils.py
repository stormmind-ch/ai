import polars as pl


def df_column_checker(df: pl.DataFrame, required_columns:set[str]):
    actual_columns = set(df.columns)

    if not required_columns.issubset(actual_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}. Found: {actual_columns}")