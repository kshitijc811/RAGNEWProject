"""Small, local retrieval engine for the College Knowledge Assistant.

The engine deliberately uses extractive answers: it can only quote or
summarize retrieved passages and returns an explicit not-found result below
the relevance threshold. This keeps the demo usable without an API key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int | None = None


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[SearchResult, ...]
    found: bool
    conflict: bool = False


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_terms(text: str) -> str:
    replacements = {
        "examinations": "examination",
        "exams": "examination",
        "percentage": "percent",
        "titles": "books",
        "borrowed": "borrow",
    }
    normalized = text.lower()
    for old, new in replacements.items():
        normalized = re.sub(rf"\b{old}\b", new, normalized)
    return normalized


def _chunks(text: str, source: str, page: int | None = None) -> list[Chunk]:
    paragraphs = [_clean(p) for p in re.split(r"\n\s*\n", text) if _clean(p)]
    result: list[Chunk] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) <= 120:
            result.append(Chunk(paragraph, source, page))
            continue
        for start in range(0, len(words), 100):
            part = " ".join(words[start : start + 120])
            if part:
                result.append(Chunk(part, source, page))
    return result


def load_documents(paths: Iterable[str | Path]) -> list[Chunk]:
    """Load .txt and .pdf files, preserving source names and PDF page numbers."""
    chunks: list[Chunk] = []
    for path_value in paths:
        path = Path(path_value)
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("PDF support requires the pypdf package.") from exc
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                chunks.extend(_chunks(page.extract_text() or "", path.name, page_number))
        elif path.suffix.lower() in {".txt", ".md"}:
            chunks.extend(_chunks(path.read_text(encoding="utf-8"), path.name))
        else:
            raise ValueError(f"Unsupported document type: {path.name}")
    if not chunks:
        raise ValueError("No readable text was found in the supplied documents.")
    return chunks


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("The knowledge base needs at least one text chunk.")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )
        self.matrix = self.vectorizer.fit_transform(
            [_normalize_terms(chunk.text) for chunk in chunks]
        )

    def search(self, question: str, k: int = 5) -> list[SearchResult]:
        if not question.strip():
            return []
        query = self.vectorizer.transform([_normalize_terms(question)])
        scores = cosine_similarity(query, self.matrix).ravel()
        indices = scores.argsort()[::-1][:k]
        return [
            SearchResult(self.chunks[index], float(scores[index]))
            for index in indices
            if scores[index] > 0
        ]

    @staticmethod
    def _sentence(text: str, question: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        stop_terms = {
            "what", "when", "where", "which", "who", "how", "many", "from",
            "the", "and", "can", "does", "with", "for", "are", "this",
        }
        question_terms = set(
            re.findall(r"[a-z0-9]{3,}", _normalize_terms(question))
        ) - stop_terms
        if not sentences:
            return text
        return max(
            sentences,
            key=lambda sentence: len(
                question_terms
                & (
                    set(re.findall(r"[a-z0-9]{3,}", _normalize_terms(sentence)))
                    - stop_terms
                )
            ),
        )

    @staticmethod
    def _has_conflict(results: list[SearchResult], question: str) -> bool:
        if " and " in _normalize_terms(question):
            return False
        if len(results) < 2 or results[0].score < 0.12 or results[1].score < 0.12:
            return False
        first = KnowledgeBase._sentence(results[0].chunk.text, question)
        for candidate in results[1:]:
            if candidate.chunk.source == results[0].chunk.source:
                continue
            second = KnowledgeBase._sentence(candidate.chunk.text, question)
            first_terms = set(re.findall(r"[a-z0-9]{4,}", first.lower()))
            second_terms = set(re.findall(r"[a-z0-9]{4,}", second.lower()))
            shared = first_terms & second_terms
            overlap = len(shared) / max(1, min(len(first_terms), len(second_terms)))
            if overlap >= 0.35 and len(shared) >= 3 and first.lower() != second.lower():
                return True
        return False

    def answer(self, question: str, k: int = 5, threshold: float = 0.05) -> Answer:
        results = self.search(question, k)
        relevant = tuple(result for result in results if result.score >= threshold)
        if not relevant:
            return Answer(
                "I could not find this information in the uploaded college documents.",
                (),
                False,
            )
        conflict = self._has_conflict(list(relevant), question)
        if conflict:
            text = (
                "The uploaded documents contain conflicting information about this "
                "question. Please review the cited passages and confirm the current "
                "policy with the college."
            )
        else:
            selected = []
            seen_sources: set[str] = set()
            for result in relevant:
                if result.chunk.source not in seen_sources:
                    selected.append(self._sentence(result.chunk.text, question))
                    seen_sources.add(result.chunk.source)
            text = " ".join(selected[:3])
        return Answer(text, relevant, True, conflict)
