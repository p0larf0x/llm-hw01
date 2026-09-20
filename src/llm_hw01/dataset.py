from pathlib import Path

import kagglehub
import pandas as pd

def load_dataset() -> pd.DataFrame:
    data_dir = Path("data/raw/rutweetcorp")
    data_dir.mkdir(parents=True, exist_ok=True)

    positive_file = data_dir / "positive.csv"
    negative_file = data_dir / "negative.csv"
    # если файлы не существуют, загружаем их с Kaggle
    if not positive_file.exists() or not negative_file.exists():
        kagglehub.dataset_download("maximsuvorov/rutweetcorp", output_dir=str(data_dir), force_download=True)
    # загружаем данные из CSV файлов
    positive_df = pd.read_csv(positive_file, usecols=["ttext"])
    negative_df = pd.read_csv(negative_file, usecols=["ttext"])
    
    positive_df["target"] = 1
    negative_df["target"] = 0
    # объединяем положительные и отрицательные примеры в один дф
    df = pd.concat([positive_df, negative_df], ignore_index=True)
    # print(f"Dataset shape: {df.shape}")
    # print(f"Missing values:\n{df.isna().sum()}")
    # print(f"class balance:\n{df['target'].value_counts()}")
    # print(f"Duplicates: {df['ttext'].duplicated().sum()}")

    # label_counts = df.groupby('ttext')['target'].nunique()
    # conflicting_labels = label_counts[label_counts > 1]
    # print(f"Conflicting labels:\n{conflicting_labels}")
    
    # удаляем дубликаты, оставляя только первый экземпляр
    df = df.drop_duplicates(subset='ttext', keep='first')
    # сбрасываем индекс после удаления дубликатов для корректного отображения
    df = df.reset_index(drop=True)

    return df.reset_index(drop=True)