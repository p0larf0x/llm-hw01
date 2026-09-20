# %%

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np

from llm_hw01.dataset import load_dataset
from llm_hw01.preprocessing import (
    clean_text,
    stem_text,
)
# %%

# повторяем все что было в task2.py, но без валидации и подбора параметров
def task3():
    df = load_dataset()

    X = df["ttext"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
        shuffle=True,
    )

    X_train_clean = X_train.apply(clean_text)
    X_test_clean = X_test.apply(clean_text)

    X_train_stemmed = X_train_clean.apply(stem_text)
    X_test_stemmed = X_test_clean.apply(stem_text)


    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_stemmed)
    feature_names = vectorizer.get_feature_names_out()
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="tanh",
        solver="adam",
        early_stopping=True,
        n_iter_no_change=5,
        max_iter=100,
        random_state=42,
)
    mlp.fit(X_train_tfidf, y_train)
    # извлекаем веса первого слоя
    first_layer_weights = mlp.coefs_[0]
    print("feature names shape:", feature_names.shape)
    first_layer_weights = mlp.coefs_[0]
    print("First layer weights shape:", first_layer_weights.shape)
    # извлекаем веса второго слоя
    second_layer_weights = mlp.coefs_[1]
    print("Second layer weights shape:", second_layer_weights.shape)
    # выводим первые 10 весов первого и второго слоя
    #for i in range(10):
    #    print(f"Feature: {feature_names[i]},\n Weight first: {first_layer_weights[i]},\n Weight second: {second_layer_weights[i]}\n\n")

    # вычисляем важность признаков как произведение весов первого и второго слоя
    feature_scores = (first_layer_weights @ second_layer_weights)[:, 0]
    negative_scores = np.where(feature_scores < 0)[0]
    negative_indices = negative_scores[np.argsort(feature_scores[negative_scores])[:20]]
    print("\nTop 20 negative features:")
    for idx in negative_indices:
        print(f"Feature: {feature_names[idx]},\n Score: {feature_scores[idx]}")

    positive_scores = np.where(feature_scores > 0)[0]
    positive_indices = positive_scores[np.argsort(feature_scores[positive_scores])[::-1][:20]]
    print("\nTop 20 positive features:")
    for idx in positive_indices:
        print(f"Feature: {feature_names[idx]},\n Score: {feature_scores[idx]}")

if __name__ == "__main__":
    task3()