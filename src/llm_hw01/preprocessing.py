import re
import pymorphy3
from nltk.stem.snowball import RussianStemmer

stemmer = RussianStemmer()
morph = pymorphy3.MorphAnalyzer()


def clean_text(text: str) -> str:
    # нижний регистр
    text = text.lower()
    # удаление ссылок
    text = re.sub(r"http\S+", " ", text)
    # удаление упоминаний
    text = re.sub(r"@\w+", " ", text)
    # удаление хэштегов
    text = re.sub(r"#(\w+)", r"\1", text)
    # удаление пунктуации
    text = re.sub(r"[^\w\s]", " ", text)
    # удаление лишних пробелов
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# используем стемминг для уменьшения слов до их основы
def stem_text(text: str) -> str:
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)
# используем лемматизацию для приведения слов к их нормальной форме
def lemmatize_text(text: str) -> str:
    words = text.split()
    lemmatized_words = [
        morph.parse(word)[0].normal_form
                        for word in words
                        ]

    return ' '.join(lemmatized_words)