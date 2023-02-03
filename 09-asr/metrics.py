import re
import numpy as np
from typing import List

# Будем использовать эту функцию для нормализации текстов перед замером CER / WER
ALLOWED_SYMBOLS = re.compile(r"(^[a-zа-я\s]+$)")


def normalize_text(text: str) -> str:
    """
    В датасетах, иногда встречается '-', 'ё', апострофы и большие буквы. А мы хотим, чтобы:
        WER("Ростов-на-дону", "ростов на дону") == 0
        WER("It's", "it s") == 0
        WER("ёлки палки", "елки палки") == 0
    Поэтому заменяем в target'ах 'ё' на 'е', а '-' на ' ' и т. д.
    Кроме того на всякий случай удаляем лишние пробелы.
    И проверяем что в получившейся строке только допустимые символы.
    """
    assert isinstance(text, str)
    text = text.lower().strip().replace("ё", "е")
    text = re.sub(r"\W+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().split(" ")
    text = " ".join(word for word in text if len(word) > 0)
    assert (text == "") or ALLOWED_SYMBOLS.match(text)
    return text


def wer(ground_truth: str, predicted: str) -> float:
    ground_truth_words = ground_truth.split()
    predicted_words = predicted.split()
    d = np.zeros((len(ground_truth_words) + 1, len(predicted_words) + 1))
    for i in range(len(ground_truth_words) + 1):
        for j in range(len(predicted_words) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(ground_truth_words) + 1):
        for j in range(1, len(predicted_words) + 1):
            if ground_truth_words[i - 1] == predicted_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return float(d[len(ground_truth_words)][len(predicted_words)])


def cer(ground_truth: str, predicted: str) -> float:
    dp = [[0] * (len(ground_truth) + 1) for _ in range(len(predicted) + 1)]
    for i in range(len(predicted) + 1):
        dp[i][0] = i
    for j in range(len(ground_truth) + 1):
        dp[0][j] = j
    for i in range(1, len(predicted) + 1):
        for j in range(1, len(ground_truth) + 1):
            if predicted[i - 1] == ground_truth[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[-1][-1]


# Функции для расчета relative CER / WER
# В функции нужно подавать строки обработанные методом normalize_text
def relative_cer(ground_truth: str, predicted: str) -> float:
    assert isinstance(ground_truth, str)
    assert isinstance(predicted, str)
    return min(1, cer(ground_truth, predicted) / (len(ground_truth) + 1e-10))


def relative_wer(ground_truth: str, predicted: str) -> float:
    assert isinstance(ground_truth, str)
    assert isinstance(predicted, str)
    gt_len = ground_truth.count(" ") + 1
    return min(1, wer(ground_truth, predicted) / (gt_len + 1e-10))


# Функции для расчета ORACLE relative CER / WER - тут мы выбираем лучшую гипотезу из beam'a
# В функции нужно подавать строки обработанные методом normalize_text
def oracle_relative_cer(ground_truth: str, predicted: List[str]) -> float:
    return min(relative_cer(ground_truth, hypo) for hypo in predicted)


def oracle_relative_wer(ground_truth: str, predicted: List[str]) -> float:
    return min(relative_wer(ground_truth, hypo) for hypo in predicted)
