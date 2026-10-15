"""Evidence-graded self-critique: keep rule, hunk evidence, verdict parsing."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from mira.core.passes import (
    _MAX_HUNK_EVIDENCE_CHARS,
    _critique_keep,
    _hunk_evidence,
    self_critique,
)
from mira.models import FileChangeType, FileDiff, HunkInfo, ReviewComment, Severity


def _comment(
    severity: Severity = Severity.WARNING,
    confidence: float = 0.9,
    line: int = 10,
    title: str = "t",
) -> ReviewComment:
    return ReviewComment(
        path="app.py",
        line=line,
        end_line=None,
        severity=severity,
        category="bug",
        title=title,
        body="body",
        confidence=confidence,
    )


class TestKeepRule:
    def test_proven_always_kept(self):
        c = _comment(severity=Severity.NITPICK, confidence=0.1)
        assert _critique_keep({"evidence": "proven"}, c)

    def test_unsupported_always_dropped(self):
        c = _comment(severity=Severity.BLOCKER, confidence=1.0)
        assert not _critique_keep({"evidence": "unsupported"}, c)

    def test_plausible_kept_only_when_severe_and_confident(self):
        assert _critique_keep({"evidence": "plausible"}, _comment(Severity.WARNING, confidence=0.8))
        assert _critique_keep({"evidence": "plausible"}, _comment(Severity.BLOCKER, 0.95))
        assert not _critique_keep({"evidence": "plausible"}, _comment(Severity.WARNING, 0.7))
        assert not _critique_keep({"evidence": "plausible"}, _comment(Severity.SUGGESTION, 0.95))

    def test_legacy_keep_boolean_honored(self):
        c = _comment()
        assert _critique_keep({"keep": True}, c)
        assert not _critique_keep({"keep": False}, c)
        assert not _critique_keep({}, c)


class TestHunkEvidence:
    def _file(self) -> FileDiff:
        return FileDiff(
            path="app.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[
                HunkInfo(1, 5, 1, 5, "hunk-one content"),
                HunkInfo(20, 10, 8, 10, "hunk-two content"),
            ],
        )

    def test_picks_covering_hunk(self):
        c = _comment(line=10)
        assert _hunk_evidence(c, [self._file()]) == "hunk-two content"

    def test_no_match_returns_empty(self):
        c = _comment(line=100)
        assert _hunk_evidence(c, [self._file()]) == ""
        assert _hunk_evidence(c, None) == ""
        assert _hunk_evidence(c, []) == ""

    def test_long_hunks_truncated(self):
        f = FileDiff(
            path="app.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[HunkInfo(1, 5, 1, 50, "x" * 5000)],
        )
        out = _hunk_evidence(_comment(line=10), [f])
        assert len(out) < 1300

    def test_late_cited_line_is_kept_inside_evidence_cap(self):
        content = "@@ -1,80 +1,80 @@\n" + "".join(
            f"+line {line:02d} {'x' * 30}\n" for line in range(1, 81)
        )
        f = FileDiff(
            path="app.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[HunkInfo(1, 80, 1, 80, content)],
        )

        out = _hunk_evidence(_comment(line=78), [f])

        assert "+line 78" in out
        assert len(out) <= _MAX_HUNK_EVIDENCE_CHARS

    def test_removed_rows_do_not_advance_target_line(self):
        content = (
            "@@ -10,102 +10,3 @@\n"
            " context line 10\n"
            + "".join(f"-removed {i}\n" for i in range(1, 101))
            + "+target line 11\n"
            " context line 12\n"
        )
        f = FileDiff(
            path="app.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[HunkInfo(10, 102, 10, 3, content)],
        )

        out = _hunk_evidence(_comment(line=11), [f])

        assert "+target line 11" in out

    @pytest.mark.parametrize("separator", ["\u2028", "\v", "\r"])
    def test_non_lf_separators_do_not_split_diff_rows(self, separator):
        content = "@@ -1 +1 @@\n" + "-old" + separator + "x" * 1600 + "\n+actual target\n"
        f = FileDiff(
            path="app.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[HunkInfo(1, 1, 1, 1, content)],
        )

        out = _hunk_evidence(_comment(line=1), [f])

        assert "+actual target" in out
        assert len(out) <= _MAX_HUNK_EVIDENCE_CHARS

    def test_oversized_cited_row_is_visibly_truncated(self):
        content = "@@ -1 +1 @@\n+" + "x" * (_MAX_HUNK_EVIDENCE_CHARS + 100)
        f = FileDiff(
            path="app.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[HunkInfo(1, 1, 1, 1, content)],
        )

        out = _hunk_evidence(_comment(line=1), [f])

        assert out.endswith("\n…")
        assert len(out) <= _MAX_HUNK_EVIDENCE_CHARS

    def test_final_cited_row_fits_with_only_prefix_marker(self):
        anchor = "+" + "z" * 1196 + "\n"
        content = "@@ -1 +1 @@\n-" + "x" * 1600 + "\n" + anchor
        f = FileDiff(
            path="app.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[HunkInfo(1, 1, 1, 1, content)],
        )

        out = _hunk_evidence(_comment(line=1), [f])

        assert out == "…\n" + anchor
        assert len(out) == _MAX_HUNK_EVIDENCE_CHARS

    def test_later_hunk_is_selected_without_earlier_context(self):
        first = "@@ -1,20 +1,20 @@\n" + "".join(
            f"+first {line} {'x' * 50}\n" for line in range(1, 21)
        )
        second = "@@ -100,20 +100,20 @@\n" + "".join(
            f"+second {line} {'y' * 50}\n" for line in range(100, 120)
        )
        f = FileDiff(
            path="app.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[HunkInfo(1, 20, 1, 20, first), HunkInfo(100, 20, 100, 20, second)],
        )

        out = _hunk_evidence(_comment(line=118), [f])

        assert "+second 118" in out
        assert "+first" not in out
        assert len(out) <= _MAX_HUNK_EVIDENCE_CHARS

    def test_short_hunk_is_returned_unchanged(self):
        content = "@@ -8,2 +8,2 @@\n context\n+changed\n"
        f = FileDiff(
            path="app.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[HunkInfo(8, 2, 8, 2, content)],
        )

        assert _hunk_evidence(_comment(line=9), [f]) == content


class TestEvidenceGradedCritique:
    @pytest.mark.asyncio
    async def test_critic_receives_late_cited_line_from_long_hunk(self):
        content = "@@ -1,80 +1,80 @@\n" + "".join(
            f"+line {line:02d} {'x' * 30}\n" for line in range(1, 81)
        )
        f = FileDiff(
            path="app.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[HunkInfo(1, 80, 1, 80, content)],
        )
        comment = _comment(line=78, severity=Severity.BLOCKER)
        critic = AsyncMock()
        critic.complete_with_tools.return_value = json.dumps(
            {"verdicts": [{"index": 0, "evidence": "proven", "reason": "shown in diff"}]}
        )

        kept = await self_critique(AsyncMock(), [comment], indexing_llm=critic, diff_files=[f])

        prompt = critic.complete_with_tools.call_args.kwargs["messages"][0]["content"]
        assert "+line 78" in prompt
        assert kept == [comment]

    @pytest.mark.asyncio
    async def test_grades_applied_through_llm_flow(self, monkeypatch):
        comments = [
            _comment(title="proven bug"),
            _comment(title="plausible low-sev", severity=Severity.SUGGESTION),
            _comment(title="unsupported claim", severity=Severity.BLOCKER),
        ]
        response = json.dumps(
            {
                "verdicts": [
                    {"index": 0, "evidence": "proven", "reason": "shown in diff"},
                    {"index": 1, "evidence": "plausible", "reason": "depends on caller"},
                    {"index": 2, "evidence": "unsupported", "reason": "code contradicts"},
                ]
            }
        )
        from mira.llm import provider as provider_mod

        captured = {}

        async def fake_complete_with_tools(self, messages, tools, temperature=None):
            captured["prompt"] = messages[0]["content"]
            return response

        monkeypatch.setattr(
            provider_mod.LLMProvider, "complete_with_tools", fake_complete_with_tools
        )

        diff_files = [
            FileDiff(
                path="app.py",
                change_type=FileChangeType.MODIFIED,
                hunks=[HunkInfo(1, 5, 1, 50, "+    def should_block?(email)")],
            )
        ]
        kept = await self_critique(AsyncMock(), comments, diff_files=diff_files)

        assert [c.title for c in kept] == ["proven bug"]
        # The critic saw the actual hunk, not just the citation.
        assert "should_block?" in captured["prompt"]

    @pytest.mark.asyncio
    async def test_audit_records_dropped_with_evidence_and_reason(self, monkeypatch):
        comments = [
            _comment(title="kept bug"),
            _comment(title="dropped claim", severity=Severity.BLOCKER),
        ]
        response = json.dumps(
            {
                "verdicts": [
                    {"index": 0, "evidence": "proven", "reason": "shown"},
                    {"index": 1, "evidence": "unsupported", "reason": "code contradicts"},
                ]
            }
        )
        from mira.llm import provider as provider_mod

        async def fake(self, messages, tools, temperature=None):
            return response

        monkeypatch.setattr(provider_mod.LLMProvider, "complete_with_tools", fake)

        audit: list[dict] = []
        await self_critique(AsyncMock(), comments, audit=audit)

        assert len(audit) == 1
        entry = audit[0]
        assert entry["stage"] == "self_critique"
        assert entry["title"] == "dropped claim"
        assert entry["reason"] == "unsupported: code contradicts"

    @pytest.mark.asyncio
    async def test_audit_records_missing_verdict(self, monkeypatch):
        comments = [_comment(title="no verdict for me")]
        from mira.llm import provider as provider_mod

        async def fake(self, messages, tools, temperature=None):
            return json.dumps({"verdicts": []})

        monkeypatch.setattr(provider_mod.LLMProvider, "complete_with_tools", fake)

        audit: list[dict] = []
        kept = await self_critique(AsyncMock(), comments, audit=audit)

        assert kept == []
        assert audit[0]["reason"].startswith("no-verdict:")
