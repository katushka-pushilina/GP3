from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


app = FastAPI(title='HH.ru Agent API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def read_csv_file(content: bytes) -> pd.DataFrame:
    encodings = ['utf-8', 'utf-8-sig', 'cp1251', 'windows-1251', 'latin1']
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc, on_bad_lines='skip')
        except Exception as e:
            last_error = e

    raise ValueError(f'Не удалось прочитать CSV. Последняя ошибка: {last_error}')


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.drop_duplicates()

    if 'From' in data.columns and 'To' in data.columns:
        data = data[~(data['From'].isna() & data['To'].isna())]

    for col in ['From', 'To']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    text_cols = data.select_dtypes(include='object').columns
    for col in text_cols:
        data[col] = data[col].astype(str).str.strip()
        data[col] = data[col].replace({'': np.nan, 'nan': np.nan, 'None': np.nan})

    return data


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if 'From' in data.columns and 'To' in data.columns:
        data['salary_mean'] = data[['From', 'To']].mean(axis=1)

    if 'Description' in data.columns:
        data['has_description'] = data['Description'].notna().astype(int)
        data['description_length'] = data['Description'].fillna('').astype(str).str.len()

    if 'Keys' in data.columns:
        data['has_keys'] = data['Keys'].notna().astype(int)

    return data


def summarize_df(df: pd.DataFrame) -> dict:
    return {
        'rows': int(df.shape[0]),
        'columns_count': int(df.shape[1]),
        'columns': list(df.columns),
        'missing': {col: int(val) for col, val in df.isna().sum().to_dict().items()},
        'missing_percent': {
            col: round(float(val), 2)
            for col, val in ((df.isna().mean() * 100).to_dict()).items()
        },
        'duplicates': int(df.duplicated().sum()),
        'sample': df.head(5).replace({np.nan: None}).to_dict(orient='records'),
    }


@app.get('/')
def root():
    return {'message': 'HH.ru Agent API работает'}


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/analyze')
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    df = read_csv_file(content)
    return summarize_df(df)


@app.post('/preprocess')
async def preprocess(file: UploadFile = File(...)):
    content = await file.read()
    df = read_csv_file(content)

    before_summary = summarize_df(df)
    cleaned = clean_dataframe(df)
    featured = add_features(cleaned)
    after_summary = summarize_df(featured)

    return {
        'before': before_summary,
        'after': after_summary,
        'new_columns': [col for col in featured.columns if col not in df.columns],
    }


@app.post('/train')
async def train_model(
    file: UploadFile = File(...),
    target: str = Form('Salary'),
):
    content = await file.read()
    df = read_csv_file(content)
    df = clean_dataframe(df)
    df = add_features(df)

    if target not in df.columns:
        return {
            'error': f'Колонка {target} не найдена в датасете',
            'available_columns': list(df.columns),
        }

    data = df.copy()
    data = data[data[target].notna()].copy()

    if data[target].dtype == bool:
        data[target] = data[target].astype(int)

    drop_cols = []
    for col in ['DescriptionHTML', 'SummaryHTML', 'Unnamed: 0']:
        if col in data.columns:
            drop_cols.append(col)

    X = data.drop(columns=[target] + drop_cols, errors='ignore')
    y = data[target]

    empty_cols = [col for col in X.columns if X[col].isna().all()]
    X = X.drop(columns=empty_cols, errors='ignore')

    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ]
    )

    models = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=150, random_state=42),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(pd.Series(y).unique()) > 1 else None
    )

    results = []

    for model_name, model in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')

        results.append({
            'model': model_name,
            'accuracy': round(float(acc), 4),
            'f1_weighted': round(float(f1), 4),
        })

    best_model = sorted(results, key=lambda x: x['f1_weighted'], reverse=True)[0]

    return {
        'target': target,
        'rows_used': int(data.shape[0]),
        'features_used': list(X.columns),
        'results': results,
        'best_model': best_model,
    }
