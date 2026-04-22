from fastapi import FastAPI
from typing import Dict
import pandas as pd

app = FastAPI()


@app.get("/")
def root():
    return {"message": "API работает"}


def missing_values_table(df):
    # Общее число пропусков
    mis_val = df.isnull().sum()

    # Процент пропусков
    mis_val_percent = 100 * mis_val / len(df)

    # Таблица
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Переименование колонок
    mis_val_table.columns = ['Missing Values', '% of Total Values']

    # Убираем нули и сортируем
    mis_val_table = mis_val_table[
        mis_val_table['% of Total Values'] != 0
    ].sort_values('% of Total Values', ascending=False).round(1)

    return mis_val_table


@app.post("/describe/dataset")
def describe_dataset(data: Dict):

    rows = data.get("rows_data", [])

    if not rows:
        return {"error": "Нет данных"}

    df = pd.DataFrame(rows)

    # базовая инфа
    rows_count = len(df)
    columns = list(df.columns)
    columns_count = len(columns)

    # пропуски
    missing_df = missing_values_table(df)

    # список колонок с >50% пропусков
    missing_columns = list(
        missing_df[missing_df['% of Total Values'] > 50].index
    )

    # топ пропусков
    top_missing = missing_df.head(5).reset_index().rename(
        columns={"index": "column"}
    ).to_dict(orient="records")

    # дубликаты
    duplicates = int(df.duplicated().sum())

    # проверка зарплаты
    has_from = "From" in columns
    has_to = "To" in columns

    # примеры строк
    sample_rows = df.head(2).to_dict(orient="records")

    return {
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
