from fastapi import FastAPI, UploadFile, File
import pandas as pd
from io import BytesIO

app = FastAPI()


@app.get("/")
def root():
    return {"message": "API работает"}


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * mis_val / len(df)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table.columns = ['Missing Values', '% of Total Values']

    mis_val_table = mis_val_table[
        mis_val_table['% of Total Values'] != 0
    ].sort_values('% of Total Values', ascending=False).round(1)

    return mis_val_table


@app.post("/describe/dataset")
async def describe_dataset(file: UploadFile = File(...)):
    # читаем содержимое файла
    content = await file.read()
    df = pd.read_csv(BytesIO(content))

    rows_count = len(df)
    columns = list(df.columns)
    columns_count = len(columns)

    missing_df = missing_values_table(df)

    missing_columns = list(
        missing_df[missing_df['% of Total Values'] > 50].index
    )

    top_missing = missing_df.head(5).reset_index().rename(
        columns={"index": "column"}
    ).to_dict(orient="records")

    duplicates = int(df.duplicated().sum())

    has_from = "From" in columns
    has_to = "To" in columns

    sample_rows = df.head(2).to_dict(orient="records")

    return {
        "file_name": file.filename,
        "rows": rows_count,
        "columns_count": columns_count,
        "columns": columns,
        "has_from": has_from,
        "has_to": has_to,
        "duplicates_count": duplicates,
        "missing_summary": {
            "columns_with_missing": len(missing_df),
            "top_missing_columns": top_missing,
            "columns_above_50_percent_missing": missing_columns
        },
        "sample_rows": sample_rows
    }
