from pathlib import Path

from rag import Chunk, KnowledgeBase, load_documents


ROOT = Path(__file__).parents[1]
SAMPLES = ROOT / "sample_documents"


def kb():
    return KnowledgeBase(load_documents(sorted(SAMPLES.glob("*.txt"))))


def test_clear_answer_has_source():
    answer = kb().answer("What is the attendance requirement?")
    assert answer.found
    assert "75 percent" in answer.text
    assert answer.sources[0].chunk.source == "academic_handbook.txt"


def test_two_documents_are_combined():
    answer = kb().answer("When do examinations happen and what percentage is the exam?")
    names = {source.chunk.source for source in answer.sources}
    assert {"academic_calendar.txt", "assessment_policy.txt"} <= names


def test_unrelated_question_is_not_answered():
    assert not kb().answer("Who won the football world cup in 2018?").found


def test_different_wording_retrieves_source():
    answer = kb().answer("How many titles can an undergraduate borrow from the library?")
    assert answer.found
    assert "four books" in answer.text


def test_conflicting_documents_are_flagged():
    conflict = KnowledgeBase([
        Chunk("The library fine is 10 rupees per day.", "rules-a.txt"),
        Chunk("The library fine is 25 rupees per day.", "rules-b.txt"),
    ])
    answer = conflict.answer("How much is the library fine?")
    assert answer.found and answer.conflict
    assert len(answer.sources) == 2


def test_all_five_sample_documents_load():
    assert len(list(SAMPLES.glob("*.txt"))) >= 5
    assert len(kb().chunks) >= 5


def test_pdf_extension_is_supported_by_loader_contract(tmp_path):
    text_file = tmp_path / "notice.txt"
    text_file.write_text("The notice says the office opens at noon.", encoding="utf-8")
    assert load_documents([text_file])[0].source == "notice.txt"


def test_empty_question_returns_not_found():
    assert not kb().answer("   ").found


def test_sources_are_ranked_by_relevance():
    assert kb().answer("When is the library open?").sources[0].chunk.source == "library_rules.txt"


def test_re_evaluation_window_is_grounded():
    answer = kb().answer("How long do I have to request a re-evaluation?")
    assert answer.found
    assert "seven calendar days" in answer.text
