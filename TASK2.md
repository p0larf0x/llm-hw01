## Эксперименты с гиперпараметрами MLP и TF-IDF

Все значения Accuracy ниже получены на одной и той же validation-выборке.
Test-выборка при подборе гиперпараметров не использовалась.

### 1. Подбор архитектуры MLP

Параметры TF-IDF: `max_features=10000`, `ngram_range=(1, 2)`.
Оптимизатор: `Adam`, функция активации: `ReLU`, `early_stopping=True`.

| Hidden layers | Validation Accuracy |
|---|---:|
| `(16,)` | **0.74229** |
| `(32,)` | 0.74085 |
| `(64,)` | 0.74051 |
| `(128,)` | 0.73594 |
| `(256,)` | 0.73913 |
| `(64, 64)` | 0.73855 |
| `(128, 64)` | 0.73812 |

Лучший результат на этом этапе: один скрытый слой из 16 нейронов.

### 2. Совместный подбор `max_features` и размера скрытого слоя

Параметры: `ngram_range=(1, 2)`, `activation=ReLU`, `solver=Adam`.

| max_features | `(16,)` | `(32,)` |
|---:|---:|---:|
| 5 000 | 0.73657 | 0.73625 |
| 10 000 | 0.74229 | 0.74085 |
| 20 000 | 0.74588 | **0.74703** |

Лучший результат: `max_features=20000`, `hidden_layer_sizes=(32,)`.

### 3. Подбор `min_df`

Параметры: `max_features=20000`, `hidden_layer_sizes=(32,)`, `ReLU + Adam`.

| min_df | Validation Accuracy |
|---:|---:|
| `1` (default) | **0.74703** |
| `2` | 0.74628 |
| `3` | 0.74634 |
| `5` | 0.74594 |

Фильтрация редких признаков не улучшила результат, поэтому оставлено `min_df=1`.

### 4. Подбор `max_df`

Параметры: `max_features=20000`, `min_df=1`, `hidden_layer_sizes=(32,)`, `ReLU + Adam`.

| max_df | Validation Accuracy |
|---:|---:|
| `1.0` (default) | **0.74703** |
| `0.95` | 0.74703 |
| `0.90` | 0.74703 |

Изменение `max_df` не повлияло на результат.

### 5. Подбор функции активации и оптимизатора

Параметры TF-IDF: `max_features=20000`, `ngram_range=(1, 2)`.  
Архитектура MLP: `hidden_layer_sizes=(32,)`.

| Activation | Solver | Learning rate | Validation Accuracy |
|---|---|---:|---:|
| `relu` | `adam` | `0.001` | 0.74703 |
| `tanh` | `adam` | `0.001` | **0.74749** |
| `relu` | `sgd` | `0.001` | 0.73913 |
| `relu` | `sgd` | `0.01` | 0.74657 |
| `tanh` | `sgd` | `0.01` | 0.74720 |

Лучший результат показала комбинация `tanh + Adam`.

## Итоговая конфигурация

```python
TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
)
### Финальный результат

После подбора гиперпараметров была выбрана следующая конфигурация:

- `max_features=20000`
- `ngram_range=(1, 2)`
- `hidden_layer_sizes=(32,)`
- `activation="tanh"`
- `solver="adam"`
- `early_stopping=True`

Результаты:

| Dataset | Accuracy |
|---|---:|
| Validation | 0.74749 |
| Test | **0.74609** |

Разница между validation и test составила около 0.14 процентного пункта

Финальный test classification report:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 0 | 0.74 | 0.74 | 0.74 |
| 1 | 0.75 | 0.75 | 0.75 |

Итоговая accuracy на test: **0.74609**.