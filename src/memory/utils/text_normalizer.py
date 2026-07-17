import re
import string


class TextNormalizer:
    @staticmethod
    def normalize_answer(text: str) -> str:
        if text is None:
            return ""

        text = str(text)

        def lower(s: str) -> str:
            return s.lower()

        def remove_punctuation(s: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in s if ch not in exclude)

        def remove_articles(s: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", s)

        def white_space_fix(s: str) -> str:
            return " ".join(s.split())

        return white_space_fix(
            remove_articles(
                remove_punctuation(
                    lower(text)
                )
            )
        )