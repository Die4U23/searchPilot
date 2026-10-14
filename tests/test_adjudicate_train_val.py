"""Negative controls for independent annotation audit boundaries."""

import pytest
from scripts.adjudicate_train_val import (
    disagreements,
    read_csv,
    sha256,
    verify_inputs,
    verify_output,
)


def records(split="train", tool_grade="3", claude_grade="0"):
    queries = [{"query_id": "q", "split": split}]
    tool = [{"query_id": "q", "item_id": "N1", "grade": tool_grade}]
    claude = [{"query_id": "q", "item_id": "N1", "grade": claude_grade}]
    return queries, tool, claude


def test_test_disagreements_are_excluded():
    *_, selected, count = disagreements(*records(split="test"))
    assert selected == []
    assert count == 0


def test_adjacent_disagreements_are_excluded():
    *_, selected, count = disagreements(*records(tool_grade="2", claude_grade="1"))
    assert selected == []
    assert count == 1


def test_duplicate_labels_rejected():
    queries, tool, claude = records()
    with pytest.raises(ValueError, match="Duplicate"):
        disagreements(queries, tool + tool, claude)


@pytest.mark.parametrize("grade", ["4", "-1", "1.0", ""])
def test_invalid_grades_rejected(grade):
    with pytest.raises(ValueError, match="Invalid query or grade"):
        disagreements(*records(tool_grade=grade))


def test_unknown_query_rejected():
    _, tool, claude = records()
    with pytest.raises(ValueError, match="Invalid query or grade"):
        disagreements([], tool, claude)


def test_missing_label_pair_rejected():
    queries, tool, _ = records()
    with pytest.raises(ValueError, match="coverage differs"):
        disagreements(queries, tool, [])


def test_changed_input_rejected(tmp_path):
    path = tmp_path / "input.csv"
    path.write_bytes(b"original")
    expected = {"input.csv": sha256(b"original")}
    verify_inputs(tmp_path, expected)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        verify_inputs(tmp_path, expected)


@pytest.mark.parametrize("suffix", [".csv", ".md"])
def test_text_input_lf_crlf_equivalent_but_content_change_rejected(tmp_path, suffix):
    path = tmp_path / f"input{suffix}"
    canonical = "query,grade\n含空格 ,1\n".encode()
    expected = {path.name: sha256(canonical)}
    for data in (canonical, canonical.replace(b"\n", b"\r\n")):
        path.write_bytes(data)
        verify_inputs(tmp_path, expected)
    path.write_bytes(canonical.replace(b" ,1", b",1").replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="SHA-256 changed"):
        verify_inputs(tmp_path, expected)


def test_binary_input_newlines_are_not_normalized(tmp_path):
    path = tmp_path / "items.parquet"
    path.write_bytes(b"binary\r\nbytes\x00")
    expected = {path.name: sha256(path.read_bytes())}
    verify_inputs(tmp_path, expected)
    path.write_bytes(b"binary\nbytes\x00")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        verify_inputs(tmp_path, expected)


def test_output_lf_crlf_equivalent_but_changed_grade_rejected(tmp_path):
    path = tmp_path / "output.csv"
    expected = b"query_id,item_id,grade\nq_001,N1,1\n"
    for data in (expected, expected.replace(b"\n", b"\r\n")):
        path.write_bytes(data)
        verify_output(path, expected)
    path.write_bytes(expected.replace(b"N1,1", b"N1,3").replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="CSV differs"):
        verify_output(path, expected)


def test_output_trailing_whitespace_is_not_ignored(tmp_path):
    path = tmp_path / "output.csv"
    expected = b"query_id,item_id,grade\nq_001,N1,1\n"
    path.write_bytes(expected + b" ")
    with pytest.raises(ValueError, match="CSV differs"):
        verify_output(path, expected)


def test_csv_quoted_multiline_field_uses_canonical_newlines(tmp_path):
    path = tmp_path / "input.csv"
    canonical = b'query_id,query_text\nq_001,"first\nsecond "\n'
    path.write_bytes(canonical)
    lf_rows = read_csv(path)
    path.write_bytes(canonical.replace(b"\n", b"\r\n"))
    assert read_csv(path) == lf_rows == [{"query_id": "q_001", "query_text": "first\nsecond "}]
