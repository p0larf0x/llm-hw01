# %%

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from llm_hw01.dataset import load_dataset
from llm_hw01.preprocessing import (
    clean_text,
    stem_text,
    lemmatize_text,
)
# %%

def task1():
    #######
    # результаты экспериментов с различными методами предобработки текста выведены в TASK1.md
    #######
    df = load_dataset()
    X = df['ttext']
    y = df['target']
    # делим данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)
    # print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    # print(f"Train balance: {y_train.value_counts(normalize=True)}")
    # print(f"Test balance: {y_test.value_counts(normalize=True)}")

    # создаем пайплайн с TF-IDF векторизацией и логистической регрессией
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    # обучаем модель на грязных данных
    model.fit(X_train, y_train)
    # делаем предсказания на грязных данных
    y_pred = model.predict(X_test)
    print(f"accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"classification report:\n{classification_report(y_test, y_pred)}")

    X_train_clean = X_train.apply(clean_text)
    X_test_clean = X_test.apply(clean_text)
    # print(f"Blank tweets in X_train: {X_train.str.strip().eq('').sum()}")
    # print(f"Blank tweets in X_test: {X_test.str.strip().eq('').sum()}")

    # создаем пайплайн с TF-IDF векторизацией и логистической регрессией для очищенных данных
    clean_model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    # обучаем модель на очищенных данных
    clean_model.fit(X_train_clean, y_train)
    y_pred_clean = clean_model.predict(X_test_clean)
    print(f"accuracy (cleaned): {accuracy_score(y_test, y_pred_clean)}")
    print(f"classification report (cleaned):\n{classification_report(y_test, y_pred_clean)}")

    X_train_stemmed = X_train_clean.apply(stem_text)
    X_test_stemmed = X_test_clean.apply(stem_text)
    stemmed_model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    # проверяем, что обучения на стеммированных данных
    stemmed_model.fit(X_train_stemmed, y_train)
    y_pred_stemmed = stemmed_model.predict(X_test_stemmed)
    print(f"accuracy (stemmed): {accuracy_score(y_test, y_pred_stemmed)}")
    print(f"classification report (stemmed):\n{classification_report(y_test, y_pred_stemmed)}")
    # проверяем, что обучения на лемматизированных данных
    X_train_lemmatized = X_train_clean.apply(lemmatize_text)
    X_test_lemmatized = X_test_clean.apply(lemmatize_text)
    lemmatized_model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    lemmatized_model.fit(X_train_lemmatized, y_train)
    y_pred_lemmatized = lemmatized_model.predict(X_test_lemmatized)
    print(f"accuracy (lemmatized): {accuracy_score(y_test, y_pred_lemmatized)}")
    print(f"classification report (lemmatized):\n{classification_report(y_test, y_pred_lemmatized)}")

if __name__ == "__main__":
    task1()
