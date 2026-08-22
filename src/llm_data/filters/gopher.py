from collections import Counter
from statistics import median

import daft
from daft import DataFrame, DataType, col

GOPHER_SCORE_COLUMNS = (
    "fraction_of_characters_in_most_common_2grams",
    "fraction_of_characters_in_most_common_3grams",
    "fraction_of_characters_in_most_common_4grams",
    "fraction_of_characters_in_duplicate_5grams",
    "fraction_of_characters_in_duplicate_6grams",
    "fraction_of_characters_in_duplicate_7grams",
    "fraction_of_characters_in_duplicate_8grams",
    "fraction_of_characters_in_duplicate_9grams",
    "fraction_of_characters_in_duplicate_10grams",
    "character_count",
    "word_count",
    "median_word_length",
    "symbol_to_word_ratio",
    "fraction_of_words_with_alpha_character",
    "required_word_count",
    "fraction_of_lines_starting_with_bullet_point",
    "fraction_of_lines_ending_with_ellipsis",
    "fraction_of_duplicate_lines",
    "fraction_of_characters_in_duplicate_lines",
    "fraction_of_duplicate_paragraphs",
    "fraction_of_characters_in_duplicate_paragraphs",
)

REQUIRED_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}

GOPHER_MIN_THRESHOLDS = {
    "word_count": 50,
    "median_word_length": 3,
    "fraction_of_words_with_alpha_character": 0.8,
    "required_word_count": 2,
}

GOPHER_MAX_THRESHOLDS = {
    "word_count": 100000,
    "median_word_length": 10,
    "symbol_to_word_ratio": 0.1,
    "fraction_of_lines_starting_with_bullet_point": 0.9,
    "fraction_of_lines_ending_with_ellipsis": 0.3,
    "fraction_of_duplicate_lines": 0.3,
    "fraction_of_characters_in_duplicate_lines": 0.3,
    "fraction_of_duplicate_paragraphs": 0.3,
    "fraction_of_characters_in_duplicate_paragraphs": 0.2,
    "fraction_of_characters_in_most_common_2grams": 0.2,
    "fraction_of_characters_in_most_common_3grams": 0.18,
    "fraction_of_characters_in_most_common_4grams": 0.16,
    "fraction_of_characters_in_duplicate_5grams": 0.15,
    "fraction_of_characters_in_duplicate_6grams": 0.14,
    "fraction_of_characters_in_duplicate_7grams": 0.13,
    "fraction_of_characters_in_duplicate_8grams": 0.12,
    "fraction_of_characters_in_duplicate_9grams": 0.11,
    "fraction_of_characters_in_duplicate_10grams": 0.10,
}


# Scoring ported from https://github.com/allenai/dolma/blob/v1.2.1/python/dolma/taggers/gopher.py
# Paragraph rules adapted from https://github.com/NVIDIA-NeMo/Curator/blob/a4470c6fe9b20ec98eb0839939c5e89de8aca3e5/nemo_curator/stages/text/filters/heuristic/repetition/repetition.py
@daft.func(
    return_dtype=DataType.struct(
        {name: DataType.float64() for name in GOPHER_SCORE_COLUMNS}
    ),
    unnest=True,
)
def gopher_scores(text: str | None) -> dict[str, float]:
    scores = dict.fromkeys(GOPHER_SCORE_COLUMNS, 0.0)
    if not text:
        return scores

    words = text.split()
    word_count = len(words)
    word_char_count = sum(map(len, words))
    scores.update(
        character_count=len(text),
        word_count=word_count,
        median_word_length=median(map(len, words)) if words else 0.0,
        symbol_to_word_ratio=sum("#" in word or "…" in word for word in words)
        / max(word_count, 1),
        fraction_of_words_with_alpha_character=sum(
            any(character.isalpha() for character in word) for word in words
        )
        / max(word_count, 1),
        required_word_count=sum(word in REQUIRED_WORDS for word in words),
    )

    for n in range(2, 11):
        counts = Counter(zip(*(words[i:] for i in range(n))))
        if not counts:
            continue
        if n < 5:
            ngram, count = counts.most_common(1)[0]
            scores[f"fraction_of_characters_in_most_common_{n}grams"] = (
                count * sum(map(len, ngram)) / max(word_char_count, 1)
            )
        else:
            character_counts = [
                (count, count * sum(map(len, ngram))) for ngram, count in counts.items()
            ]
            scores[f"fraction_of_characters_in_duplicate_{n}grams"] = sum(
                characters for count, characters in character_counts if count > 1
            ) / max(sum(characters for _, characters in character_counts), 1)

    lines = text.split("\n")
    line_counts = Counter(lines)
    paragraphs = text.split("\n\n")
    unique_paragraphs = set(paragraphs)
    scores.update(
        fraction_of_lines_starting_with_bullet_point=sum(
            line.startswith(("*", "-")) for line in lines
        )
        / len(lines),
        fraction_of_lines_ending_with_ellipsis=sum(line.endswith("…") for line in lines)
        / len(lines),
        fraction_of_duplicate_lines=sum(
            count for count in line_counts.values() if count > 1
        )
        / len(lines),
        fraction_of_characters_in_duplicate_lines=sum(
            len(line) * count for line, count in line_counts.items() if count > 1
        )
        / max(word_char_count, 1),
        fraction_of_duplicate_paragraphs=1 - len(unique_paragraphs) / len(paragraphs),
        fraction_of_characters_in_duplicate_paragraphs=1
        - sum(map(len, unique_paragraphs)) / max(sum(map(len, paragraphs)), 1),
    )
    return scores


class GopherFilter:
    def __init__(
        self,
        input_column: str = "text",
    ):
        self.input_column = input_column

    def __call__(self, df: DataFrame) -> DataFrame:
        df = df.select("*", gopher_scores(col(self.input_column)))
        predicate = daft.lit(True)
        for column, threshold in GOPHER_MIN_THRESHOLDS.items():
            predicate &= col(column) >= threshold
        for column, threshold in GOPHER_MAX_THRESHOLDS.items():
            predicate &= col(column) <= threshold
        return df.where(predicate)
