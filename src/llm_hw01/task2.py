# %%

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from llm_hw01.dataset import load_dataset
from llm_hw01.preprocessing import (
    clean_text,
    stem_text,
)
# %%


def task2():
    df = load_dataset()

    X = df["ttext"]
    y = df["target"]
    # делим данные на обучающую и тестовую выборки
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
    # делим обучающую выборку на обучающую и валидационную
    X_train_mlp, X_val, y_train_mlp, y_val = train_test_split(
        X_train_stemmed,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )
    # преобразование текста в числовой формат
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_mlp)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test_stemmed)

    # print(f"Train TF-IDF shape: {X_train_tfidf.shape}")
    # print(f"Validation TF-IDF shape: {X_val_tfidf.shape}")
    # print(f"Test TF-IDF shape: {X_test_tfidf.shape}")

    # создаем и обучаем MLP классификатор
    # данный блок использовался для подбора гиперпараметров
    # результаты подбора выведены в TASK2.md
    mlp = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="tanh",
        solver="adam",
        early_stopping=True,
        n_iter_no_change=5,
        max_iter=100,
        random_state=42,
    )
    mlp.fit(X_train_tfidf, y_train_mlp)
    y_val_pred = mlp.predict(X_val_tfidf)
    print(
        f"MLP validation accuracy: "
        f"{accuracy_score(y_val, y_val_pred)}"
    )

    print(
        "MLP validation classification report:"
    )

    print(
        classification_report(
            y_val,
            y_val_pred,
        )
    )
    print(f"MLP iterations: {mlp.n_iter_}")
    print(f"MLP final loss: {mlp.loss_}")
    # проверяем качество модели на тестовой выборке результаты выведены в TASK2.md
    y_test_pred = mlp.predict(X_test_tfidf)
    print(
        f"MLP test accuracy: "
        f"{accuracy_score(y_test, y_test_pred)}"
    )
    print(
        "MLP test classification report:"
    )
    print(
        classification_report(
            y_test,
            y_test_pred,
        )
    )

if __name__ == "__main__":
    task2()
# %%
