from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from .knowledge_schema import SquadSample, KnowledgeContext


class SquadLoader:
    """
    Loader for SQuAD v1.1 / v2.0.

    Supported formats:
    - Official SQuAD JSON:
        train-v1.1.json
        dev-v1.1.json
        train-v2.0.json
        dev-v2.0.json

    - Optional simple CSV format:
        columns should include context, question, answer/answers.

    Recommended usage for this project:

        loader = SquadLoader(dataset_dir="data")

        samples = loader.load_samples_from_file(
            file_path="data/train-v2.0.json",
            limit=20,
            answerable_only=True,
        )

    Or:

        samples = loader.load_samples(
            split="train",
            limit=20,
            answerable_only=True,
        )
    """

    def __init__(
        self,
        dataset_name: str = "akashdesarda/squad-v11",
        dataset_dir: Optional[str | Path] = "data",
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_dir = Path(dataset_dir) if dataset_dir is not None else None

    # ------------------------------------------------------------------
    # Dataset path resolution
    # ------------------------------------------------------------------

    def download_dataset(self) -> Path:
        """
        Optional KaggleHub download.

        This is kept for compatibility, but for the current project it is better
        to use official SQuAD JSON files placed under data/.
        """

        try:
            import kagglehub
        except ImportError as error:
            raise ImportError(
                "kagglehub is not installed. Install it with `pip install kagglehub`, "
                "or use local SQuAD JSON files under data/."
            ) from error

        path = kagglehub.dataset_download(self.dataset_name)
        self.dataset_dir = Path(path)

        return self.dataset_dir

    def get_dataset_dir(self) -> Path:
        """
        Return dataset directory.

        If dataset_dir is None, try KaggleHub download.
        """

        if self.dataset_dir is None:
            return self.download_dataset()

        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.dataset_dir.resolve()}"
            )

        return self.dataset_dir

    def find_data_file(self, split: str = "validation") -> Path:
        """
        Find SQuAD data file from dataset_dir.

        Supported split aliases:
        - train / training
        - dev / validation / val

        This method supports both v1.1 and v2.0 file names.
        """

        dataset_dir = self.get_dataset_dir()
        normalized_split = self._normalize_split(split)

        files = [
            path
            for path in dataset_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".csv"}
        ]

        if not files:
            raise FileNotFoundError(
                f"No .json or .csv files found under: {dataset_dir.resolve()}"
            )

        files = sorted(files, key=lambda path: str(path).lower())
        candidate_names = self._candidate_file_names(normalized_split)

        # 1. Exact filename match.
        for candidate_name in candidate_names:
            for file_path in files:
                if file_path.name.lower() == candidate_name.lower():
                    return file_path

        # 2. Keyword match.
        keywords = self._split_keywords(normalized_split)

        for file_path in files:
            file_name = file_path.name.lower()
            if any(keyword in file_name for keyword in keywords):
                return file_path

        available = "\n".join(str(path) for path in files)

        raise FileNotFoundError(
            f"Could not find SQuAD file for split='{split}'.\n"
            f"Dataset dir: {dataset_dir.resolve()}\n"
            f"Candidate names: {candidate_names}\n"
            f"Available files:\n{available}"
        )

    # ------------------------------------------------------------------
    # Public loading methods
    # ------------------------------------------------------------------

    def load_samples(
        self,
        split: str = "validation",
        limit: Optional[int] = None,
        answerable_only: bool = True,
    ) -> list[SquadSample]:
        """
        Load samples by split from dataset_dir.
        """

        data_file = self.find_data_file(split)

        return self.load_samples_from_file(
            file_path=data_file,
            limit=limit,
            answerable_only=answerable_only,
        )

    def load_samples_from_file(
        self,
        file_path: str | Path,
        limit: Optional[int] = None,
        answerable_only: bool = True,
    ) -> list[SquadSample]:
        """
        Load SQuAD samples from a specific file.

        File can be:
        - official SQuAD JSON
        - simple CSV
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"SQuAD file does not exist: {file_path.resolve()}"
            )

        suffix = file_path.suffix.lower()

        if suffix == ".json":
            return self.load_samples_from_json_file(
                file_path=file_path,
                limit=limit,
                answerable_only=answerable_only,
            )

        if suffix == ".csv":
            return self.load_samples_from_csv_file(
                file_path=file_path,
                limit=limit,
                answerable_only=answerable_only,
            )

        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. Expected .json or .csv."
        )

    # ------------------------------------------------------------------
    # JSON loading
    # ------------------------------------------------------------------

    def load_samples_from_json_file(
        self,
        file_path: str | Path,
        limit: Optional[int] = None,
        answerable_only: bool = True,
    ) -> list[SquadSample]:
        """
        Load official SQuAD v1.1 / v2.0 JSON.

        SQuAD v1.1:
            all questions are answerable.

        SQuAD v2.0:
            some questions have:
                is_impossible = true
                answers = []

        If answerable_only=True, unanswerable questions are skipped.
        """

        file_path = Path(file_path)

        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected SQuAD JSON root to be a dict, got {type(data)}."
            )

        if "data" not in data:
            raise ValueError(
                f"Invalid SQuAD JSON: missing top-level 'data' field in {file_path}"
            )

        articles = data.get("data", [])

        if not isinstance(articles, list):
            raise ValueError(
                f"Invalid SQuAD JSON: top-level 'data' field must be a list."
            )

        samples: list[SquadSample] = []

        for article in articles:
            title = str(article.get("title", "")).strip()

            paragraphs = article.get("paragraphs", [])

            if not isinstance(paragraphs, list):
                continue

            for paragraph in paragraphs:
                context = str(paragraph.get("context", "")).strip()
                qas = paragraph.get("qas", [])

                if not context or not isinstance(qas, list):
                    continue

                for qa in qas:
                    sample_id = str(qa.get("id", "")).strip()
                    question = str(qa.get("question", "")).strip()
                    is_impossible = bool(qa.get("is_impossible", False))

                    answers = self._extract_answers_from_json_qa(qa)

                    if answerable_only:
                        if is_impossible:
                            continue
                        if not answers:
                            continue

                    if not question:
                        continue

                    sample = SquadSample(
                        sample_id=sample_id,
                        title=title,
                        context=context,
                        question=question,
                        answers=answers,
                    )

                    samples.append(sample)

                    if limit is not None and len(samples) >= limit:
                        return samples

        return samples

    @staticmethod
    def _extract_answers_from_json_qa(qa: dict[str, Any]) -> list[str]:
        """
        Extract answer texts from a SQuAD QA object.

        Expected format:
            "answers": [
                {"text": "...", "answer_start": 123}
            ]
        """

        raw_answers = qa.get("answers", [])

        if not isinstance(raw_answers, list):
            return []

        answers: list[str] = []

        for answer in raw_answers:
            if isinstance(answer, dict):
                text = answer.get("text", "")
                if isinstance(text, str) and text.strip():
                    answers.append(text.strip())
            elif isinstance(answer, str) and answer.strip():
                answers.append(answer.strip())

        # Deduplicate while preserving order.
        return list(dict.fromkeys(answers))

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    def load_samples_from_csv_file(
        self,
        file_path: str | Path,
        limit: Optional[int] = None,
        answerable_only: bool = True,
    ) -> list[SquadSample]:
        """
        Load a simple CSV version of SQuAD.

        This is a fallback parser. Official JSON is preferred.
        """

        file_path = Path(file_path)
        samples: list[SquadSample] = []

        with file_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise ValueError(f"CSV file has no header: {file_path}")

            field_map = {
                self._normalize_column_name(field): field
                for field in reader.fieldnames
            }

            for row_index, row in enumerate(reader):
                sample_id = self._read_csv_value(
                    row=row,
                    field_map=field_map,
                    aliases=["id", "sample_id", "qas_id", "question_id"],
                    default=f"{file_path.stem}_{row_index:06d}",
                )

                title = self._read_csv_value(
                    row=row,
                    field_map=field_map,
                    aliases=["title", "topic", "article_title"],
                    default="unknown",
                )

                context = self._read_csv_value(
                    row=row,
                    field_map=field_map,
                    aliases=["context", "passage", "paragraph", "content"],
                    default="",
                )

                question = self._read_csv_value(
                    row=row,
                    field_map=field_map,
                    aliases=["question", "query"],
                    default="",
                )

                answer = self._read_csv_value(
                    row=row,
                    field_map=field_map,
                    aliases=["answer", "answer_text", "text", "answers"],
                    default="",
                )

                answers = [str(answer).strip()] if str(answer).strip() else []

                if answerable_only and not answers:
                    continue

                if not str(context).strip() or not str(question).strip():
                    continue

                samples.append(
                    SquadSample(
                        sample_id=str(sample_id).strip(),
                        title=str(title).strip() or "unknown",
                        context=str(context).strip(),
                        question=str(question).strip(),
                        answers=answers,
                    )
                )

                if limit is not None and len(samples) >= limit:
                    return samples

        return samples

    # ------------------------------------------------------------------
    # Context extraction
    # ------------------------------------------------------------------

    def extract_unique_contexts(
        self,
        samples: list[SquadSample],
    ) -> list[KnowledgeContext]:
        """
        Extract deduplicated contexts from SQuAD samples.

        Multiple questions may share the same context.
        This method groups them by context_hash.
        """

        contexts_by_hash: dict[str, KnowledgeContext] = {}

        for sample in samples:
            context_hash = sample.context_hash

            if context_hash not in contexts_by_hash:
                context_id = self._make_context_id(len(contexts_by_hash))

                contexts_by_hash[context_hash] = KnowledgeContext.from_sample(
                    sample=sample,
                    context_id=context_id,
                )
            else:
                contexts_by_hash[context_hash].add_sample_id(sample.sample_id)

        return list(contexts_by_hash.values())

    def load_unique_contexts(
        self,
        split: str = "validation",
        limit: Optional[int] = None,
        answerable_only: bool = True,
    ) -> list[KnowledgeContext]:
        """
        Load samples and extract unique contexts.
        """

        samples = self.load_samples(
            split=split,
            limit=limit,
            answerable_only=answerable_only,
        )

        return self.extract_unique_contexts(samples)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_split(split: str) -> str:
        split = split.lower().strip()

        if split in {"train", "training"}:
            return "train"

        if split in {"dev", "validation", "valid", "val"}:
            return "dev"

        raise ValueError(
            f"Unsupported split '{split}'. Use 'train' or 'validation/dev'."
        )

    @staticmethod
    def _candidate_file_names(normalized_split: str) -> list[str]:
        if normalized_split == "train":
            return [
                "train-v2.0.json",
                "train_v2.0.json",
                "training-v2.0.json",
                "train-v1.1.json",
                "train_v1.1.json",
                "training-v1.1.json",
                "train.json",
                "train.csv",
            ]

        if normalized_split == "dev":
            return [
                "dev-v2.0.json",
                "dev_v2.0.json",
                "validation-v2.0.json",
                "dev-v1.1.json",
                "dev_v1.1.json",
                "validation-v1.1.json",
                "dev.json",
                "validation.json",
                "dev.csv",
                "validation.csv",
            ]

        raise ValueError(f"Unsupported normalized split: {normalized_split}")

    @staticmethod
    def _split_keywords(normalized_split: str) -> list[str]:
        if normalized_split == "train":
            return ["train", "training"]

        if normalized_split == "dev":
            return ["dev", "validation", "valid", "val"]

        raise ValueError(f"Unsupported normalized split: {normalized_split}")

    @staticmethod
    def _make_context_id(index: int) -> str:
        return f"squad_ctx_{index:06d}"

    @staticmethod
    def _normalize_column_name(column_name: str) -> str:
        return (
            column_name.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

    @classmethod
    def _read_csv_value(
        cls,
        row: dict[str, Any],
        field_map: dict[str, str],
        aliases: list[str],
        default: Any = None,
    ) -> Any:
        for alias in aliases:
            normalized_alias = cls._normalize_column_name(alias)
            original_field = field_map.get(normalized_alias)

            if original_field is None:
                continue

            value = row.get(original_field)

            if value is not None and str(value).strip():
                return value

        return default