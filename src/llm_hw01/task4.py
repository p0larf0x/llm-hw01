# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from llm_hw01.dataset import load_dataset
from llm_hw01.preprocessing import (
    clean_text,
    stem_text,
)

# %%

def task4():
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

    # преобразование текста в числовой формат
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_stemmed)
    X_test_tfidf = vectorizer.transform(X_test_stemmed)

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
    mlp.fit(X_train_tfidf, y_train)

    y_test_pred = mlp.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average="weighted")
    recall = recall_score(y_test, y_test_pred, average="weighted")
    f1 = f1_score(y_test, y_test_pred, average="weighted")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_test_pred))
    cm = confusion_matrix(
        y_test,
        y_test_pred,
    )

    tn, fp, fn, tp = cm.ravel()
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")
    print(f"True Positive: {tp}")
    print(f"True Negative: {tn}")

    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred,display_labels=["Negative", "Positive"])
    plt.title("Confusion Matrix")
    plt.savefig("./images/confusion_matrix.png", dpi=300, bbox_inches='tight')
    #plt.show()

    test_sentences = [
        "Сегодня прекрасный день, я очень счастлив",
        "Спасибо, мне всё очень понравилось",
        "Я ненавижу этот ужасный день",
        "Мне очень грустно и плохо",
        "Этот фильм совсем не плохой",
        "Как же здорово снова заболеть",
        "Прекрасно, автобус опять опоздал на час",
        "ЕДиная россия - делает вбросы на выборах",
        "Ерундистика какая-то, но мне понравилось",
        "Я не могу поверить, что это произошло, это не приколько",
    ]
    
    processed_sentences = []
    for sentence in test_sentences:
        cleaned_sentence = clean_text(sentence)
        stemmed_sentence = stem_text(cleaned_sentence)
        processed_sentences.append(stemmed_sentence)
    # for original, processed in zip(test_sentences, processed_sentences):
    #     print(f"Original: {original}")
    #     print(f"Processed: {processed}")
    #     print()
    manual_tfidf = vectorizer.transform(processed_sentences)
    manual_tfidf_pred = mlp.predict(manual_tfidf)
    manual_probabilities = mlp.predict_proba(manual_tfidf)
    for sentence, prediction, probability in zip(
        test_sentences,
        manual_tfidf_pred,
        manual_probabilities,
    ):
        sentiment = (
            "Positive"
            if prediction == 1
            else "Negative"
        )

        print(f"Text: {sentence}")
        print(f"Prediction: {sentiment}")
        print(
            f"Negative: {probability[0]:.4f}, "
            f"Positive: {probability[1]:.4f}"
        )
        print()

if __name__ == "__main__":
    task4()
# %%
