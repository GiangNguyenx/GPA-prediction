import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

RAW_PATH = '../data/raw'
PROCESSED_PATH = '../data/processed'

BIN_MAP = {
    'yes': 1, 'no': 0,
    'father': 1, 'mother': 0,
    'GT3': 1, 'LE3': 0,
    'A': 1, 'T': 0, 'F': 0, 'M': 1,
    'course': 0, 'home': 1, 'reputation': 2,
    'at_home': 0, 'health': 1, 'services': 2, 'teacher': 4, 'other': 3
}

CONVERT_BINARY_COLS = [
    'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
    'internet', 'romantic', 'Fjob', 'Mjob', 'reason', 'guardian',
    'Pstatus', 'famsize', 'sex'
]


def load_data():
    mat_df = pd.read_csv(os.path.join(RAW_PATH, 'student-mat.csv'), sep=';')
    por_df = pd.read_csv(os.path.join(RAW_PATH, 'student-por.csv'), sep=';')

    if mat_df.columns.equals(por_df.columns):
        df = pd.concat([mat_df, por_df], axis=0, ignore_index=True)
    else:
        raise ValueError("Columns of both datasets do not match!")

    return df


def preprocess_data(df):
    df = df.drop(['school', 'address'], axis=1)

    for col in CONVERT_BINARY_COLS:
        df[col] = df[col].map(BIN_MAP)

    features = df.drop('G3', axis=1)
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(features)

    processed_df = pd.DataFrame(normalized_data, columns=features.columns)
    processed_df['G3'] = df['G3'].values

    return processed_df


def save_processed_data(df):
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    df.to_csv(os.path.join(PROCESSED_PATH, 'data_processed.csv'), index=False)


def main():
    raw_df = load_data()
    processed_df = preprocess_data(raw_df)
    save_processed_data(processed_df)


if __name__ == "__main__":
    main()
