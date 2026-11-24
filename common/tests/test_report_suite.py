from tools.report_suite import build_table, find_reports, parse_report, format_table


def _write_report(path, n_samples, n_questions, n_mismatches):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"Clips checked: {n_samples}\nIssues found: {n_mismatches}\n\n"
        f"RESULT: {n_samples} samples, {n_questions} questions, {n_mismatches} mismatches\n",
        encoding="utf8",
    )


def test_parse_report_extracts_result_line(tmp_path):
    p = tmp_path / "validation_report.txt"
    _write_report(p, 10, 100, 0)
    parsed = parse_report(p)
    assert parsed == dict(n_samples=10, n_questions=100, n_mismatches=0)


def test_parse_report_returns_none_without_result_line(tmp_path):
    p = tmp_path / "validation_report.txt"
    p.write_text("no result line here", encoding="utf8")
    assert parse_report(p) is None


def test_find_reports_excludes_common_and_tools(tmp_path):
    _write_report(tmp_path / "ds_a" / "validation_report.txt", 5, 20, 0)
    _write_report(tmp_path / "ds_a" / "tests" / "sub_sample" / "validation_report.txt", 3, 12, 0)
    _write_report(tmp_path / "common" / "tests" / "validation_report.txt", 1, 1, 0)  # must be excluded
    _write_report(tmp_path / "tools" / "validation_report.txt", 1, 1, 0)  # must be excluded

    found = find_reports(tmp_path)
    rels = {str(p.relative_to(tmp_path)) for p in found}
    assert "ds_a/validation_report.txt" in rels
    assert "ds_a/tests/sub_sample/validation_report.txt" in rels
    assert not any("common" in r for r in rels)
    assert not any(r.startswith("tools") for r in rels)


def test_build_table_and_format_table_report_pass_and_fail(tmp_path):
    _write_report(tmp_path / "ds_a" / "validation_report.txt", 10, 100, 0)
    _write_report(tmp_path / "ds_b" / "validation_report.txt", 5, 50, 2)

    rows = build_table(tmp_path)
    statuses = {r["dataset"]: r["status"] for r in rows}
    assert statuses["ds_a"] == "PASS"
    assert statuses["ds_b"] == "FAIL"

    table_text = format_table(rows)
    assert "PASS" in table_text and "FAIL" in table_text
    assert "2 report(s)" in table_text
    assert "1 passing" in table_text


def test_format_table_empty():
    assert "No validation_report.txt" in format_table([])
