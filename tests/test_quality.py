"""Tests for the translation-quality machinery."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from conftest import ScriptedBackend, lines_reply
from mangatl.domain.models import GlossaryEntry
from mangatl.quality.glossary import Glossary
from mangatl.quality.prompt import (
    HonorificPolicy,
    StyleGuide,
    build_critique_prompt,
    build_translation_prompt,
    parse_lines_response,
)
from mangatl.quality.refine import Refiner, RefineSettings

if TYPE_CHECKING:
    from pathlib import Path

from mangatl.quality.scoring import (
    best_candidate,
    chrf,
    length_sanity,
    score_candidate,
)


class TestChrf:
    """Character n-gram F-score."""

    def test_identical_scores_one(self) -> None:
        """Identical scores one."""
        assert chrf("hello world", "hello world") == pytest.approx(1.0)

    def test_disjoint_scores_zero(self) -> None:
        """Disjoint scores zero."""
        assert chrf("aaaa", "bbbb") == pytest.approx(0.0)

    def test_partial_between(self) -> None:
        """Partial between."""
        score = chrf("hello there", "hello world")
        assert 0.0 < score < 1.0

    def test_whitespace_ignored(self) -> None:
        """Whitespace ignored."""
        assert chrf("a b c", "abc") == pytest.approx(1.0)

    def test_both_empty(self) -> None:
        """Both empty."""
        assert chrf("", "") == pytest.approx(0.0)

    def test_one_empty(self) -> None:
        """One empty."""
        assert chrf("", "abc") == pytest.approx(0.0)

    def test_short_strings_skip_high_orders(self) -> None:
        """Short strings skip high orders."""
        assert chrf("ab", "ab") == pytest.approx(1.0)

    def test_max_n_one(self) -> None:
        """Max n one."""
        assert chrf("abc", "cba", max_n=1) == pytest.approx(1.0)

    def test_invalid_max_n(self) -> None:
        """Invalid max n."""
        with pytest.raises(ValueError, match="max_n must be"):
            chrf("a", "b", max_n=0)

    def test_japanese(self) -> None:
        """Japanese."""
        assert chrf("こんにちは", "こんにちは") == pytest.approx(1.0)
        assert 0.0 < chrf("こんにちは", "こんばんは") < 1.0


class TestLengthSanity:
    """Length-ratio plausibility."""

    def test_plausible(self) -> None:
        """Plausible."""
        assert length_sanity("こんにちは", "Hello there") == pytest.approx(1.0)

    def test_both_empty(self) -> None:
        """Both empty."""
        assert length_sanity("", "") == pytest.approx(1.0)

    def test_empty_source_with_output(self) -> None:
        """Empty source with output."""
        assert length_sanity("", "something") == pytest.approx(0.0)

    def test_empty_target(self) -> None:
        """Empty target."""
        assert length_sanity("こんにちは", "") == pytest.approx(0.0)

    def test_truncated_penalised(self) -> None:
        """Truncated penalised."""
        assert 0.0 < length_sanity("あ" * 100, "hi") < 1.0

    def test_runaway_penalised(self) -> None:
        """Runaway penalised."""
        assert 0.0 < length_sanity("あ", "x" * 100) < 1.0


class TestScoreCandidate:
    """Round-trip candidate scoring."""

    def test_perfect_round_trip_scores_high(self) -> None:
        """Perfect round trip scores high."""
        score = score_candidate("こんにちは", "Hello there", "こんにちは")
        assert score > 0.9

    def test_bad_round_trip_scores_low(self) -> None:
        """Bad round trip scores low."""
        assert score_candidate("こんにちは", "Hello", "さようなら") < 0.5

    def test_best_candidate_picks_highest(self) -> None:
        """Best candidate picks highest."""
        target, score = best_candidate(
            "こんにちは",
            [("Goodbye", "さようなら"), ("Hello there", "こんにちは")],
        )
        assert target == "Hello there"
        assert score > 0.9

    def test_best_candidate_needs_candidates(self) -> None:
        """Best candidate needs candidates."""
        with pytest.raises(ValueError, match="at least one candidate"):
            best_candidate("x", [])


class TestGlossary:
    """Series glossary behaviour."""

    def test_add_len_iter(self) -> None:
        """Add len iter."""
        g = Glossary([GlossaryEntry(source="兄", target="Nii-san")])
        assert len(g) == 1
        assert [e.source for e in g] == ["兄"]

    def test_later_entry_replaces(self) -> None:
        """Later entry replaces."""
        g = Glossary(
            [
                GlossaryEntry(source="兄", target="Brother"),
                GlossaryEntry(source="兄", target="Nii-san"),
            ],
        )
        assert len(g) == 1
        assert g.entries[0].target == "Nii-san"

    def test_longest_source_first(self) -> None:
        """Longest source first."""
        g = Glossary(
            [
                GlossaryEntry(source="兄", target="Nii-san"),
                GlossaryEntry(source="お兄ちゃん", target="Onii-chan"),
            ],
        )
        assert [e.source for e in g.entries] == ["お兄ちゃん", "兄"]

    def test_same_length_sorts_by_source(self) -> None:
        """Same length sorts by source."""
        g = Glossary(
            [
                GlossaryEntry(source="bb", target="B"),
                GlossaryEntry(source="aa", target="A"),
            ],
        )
        assert [e.source for e in g.entries] == ["aa", "bb"]

    def test_remove(self) -> None:
        """Remove."""
        g = Glossary([GlossaryEntry(source="兄", target="Nii-san")])
        assert g.remove("兄")
        assert not g.remove("兄")

    def test_relevant_to(self) -> None:
        """Relevant to."""
        g = Glossary(
            [
                GlossaryEntry(source="兄", target="Nii-san"),
                GlossaryEntry(source="姉", target="Nee-san"),
            ],
        )
        assert [e.source for e in g.relevant_to(["兄さん、待って"])] == ["兄"]

    def test_violations(self) -> None:
        """Violations."""
        g = Glossary([GlossaryEntry(source="兄", target="Nii-san")])
        assert g.violations("兄さん", "Big brother")
        assert not g.violations("兄さん", "nii-san, wait")
        assert not g.violations("こんにちは", "Hello")

    def test_json_round_trip(self) -> None:
        """Json round trip."""
        g = Glossary([GlossaryEntry(source="兄", target="Nii-san", note="older")])
        again = Glossary.from_json(g.to_json())
        assert again.entries[0].note == "older"

    def test_from_json_rejects_non_array(self) -> None:
        """From json rejects non array."""
        with pytest.raises(TypeError, match="must be an array"):
            Glossary.from_json('{"a": 1}')

    def test_from_json_rejects_non_object_entry(self) -> None:
        """From json rejects non object entry."""
        with pytest.raises(TypeError, match="must be objects"):
            Glossary.from_json("[1, 2]")

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Save and load."""
        path = tmp_path / "g.json"
        Glossary([GlossaryEntry(source="兄", target="Nii-san")]).save(path)
        assert len(Glossary.load(path)) == 1

    def test_load_missing_is_empty(self, tmp_path: Path) -> None:
        """Load missing is empty."""
        assert len(Glossary.load(tmp_path / "absent.json")) == 0


class TestPrompt:
    """Prompt construction and reply parsing."""

    def test_translation_prompt_contains_everything(self) -> None:
        """Translation prompt contains everything."""
        prompt = build_translation_prompt(
            ["こんにちは", "またね"],
            context=["Earlier line"],
            glossary=[GlossaryEntry(source="兄", target="Nii-san", note="older")],
            style=StyleGuide(honorifics=HonorificPolicy.KEEP, notes="shonen"),
        )
        assert "こんにちは" in prompt
        assert "Earlier line" in prompt
        assert "Nii-san" in prompt
        assert "older" in prompt
        assert "shonen" in prompt
        assert "exactly 2 strings" in prompt

    def test_translation_prompt_empty_lines_rejected(self) -> None:
        """Translation prompt empty lines rejected."""
        with pytest.raises(ValueError, match="zero lines"):
            build_translation_prompt([])

    def test_no_glossary_or_context(self) -> None:
        """No glossary or context."""
        prompt = build_translation_prompt(["あ"])
        assert "(none)" in prompt

    @pytest.mark.parametrize(
        "policy",
        [HonorificPolicy.KEEP, HonorificPolicy.DROP, HonorificPolicy.ADAPT],
    )
    def test_every_honorific_policy_renders(self, policy: HonorificPolicy) -> None:
        """Every honorific policy renders."""
        prompt = build_translation_prompt(["あ"], style=StyleGuide(honorifics=policy))
        assert "honorific" in prompt.lower() or "relationship" in prompt.lower()

    def test_sfx_toggle(self) -> None:
        """Sfx toggle."""
        keep = build_translation_prompt(["あ"], style=StyleGuide(keep_sfx=True))
        translate = build_translation_prompt(["あ"], style=StyleGuide(keep_sfx=False))
        assert "untranslated" in keep
        assert "onomatopoeia" in translate

    def test_context_truncated(self) -> None:
        """Context truncated."""
        prompt = build_translation_prompt(["あ"], context=[f"L{i}" for i in range(100)])
        assert "L99" in prompt
        assert "L0\n" not in prompt

    def test_critique_prompt(self) -> None:
        """Critique prompt."""
        prompt = build_critique_prompt(["こんにちは"], ["Hello"])
        assert "こんにちは" in prompt
        assert "Hello" in prompt

    @pytest.mark.parametrize(("src", "draft"), [([], []), (["a"], ["x", "y"])])
    def test_critique_prompt_length_mismatch(self, src: list[str], draft: list[str]) -> None:
        """Critique prompt length mismatch."""
        with pytest.raises(ValueError, match="length mismatch"):
            build_critique_prompt(src, draft)

    def test_parse_clean_reply(self) -> None:
        """Parse clean reply."""
        assert parse_lines_response('{"lines": ["a", "b"]}', 2) == ["a", "b"]

    def test_parse_reply_with_fence_and_prose(self) -> None:
        """Parse reply with fence and prose."""
        raw = 'Sure!\n```json\n{"lines": ["a"]}\n```\nHope that helps.'
        assert parse_lines_response(raw, 1) == ["a"]

    def test_parse_pads_short_reply(self) -> None:
        """Parse pads short reply."""
        assert parse_lines_response('{"lines": ["a"]}', 3) == ["a", "", ""]

    def test_parse_truncates_long_reply(self) -> None:
        """Parse truncates long reply."""
        assert parse_lines_response('{"lines": ["a", "b", "c"]}', 2) == ["a", "b"]

    def test_parse_coerces_non_strings(self) -> None:
        """Parse coerces non strings."""
        assert parse_lines_response('{"lines": [1, null]}', 2) == ["1", "None"]

    def test_parse_no_json(self) -> None:
        """Parse no json."""
        with pytest.raises(ValueError, match="no JSON object"):
            parse_lines_response("nothing here", 1)

    def test_parse_reversed_braces(self) -> None:
        """Parse reversed braces."""
        with pytest.raises(ValueError, match="no JSON object"):
            parse_lines_response("} {", 1)

    def test_parse_invalid_json(self) -> None:
        """Parse invalid json."""
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_lines_response('{"lines": [oops]}', 1)

    def test_parse_missing_lines_key(self) -> None:
        """Parse missing lines key."""
        with pytest.raises(TypeError, match="lacked a 'lines' array"):
            parse_lines_response('{"text": "hi"}', 1)


class TestRefineSettings:
    """Settings validation."""

    def test_default(self) -> None:
        """Default."""
        settings = RefineSettings()
        assert settings.critique
        assert settings.candidates == 2

    def test_bad_candidate_count(self) -> None:
        """Bad candidate count."""
        with pytest.raises(ValueError, match="candidates must be"):
            RefineSettings(candidates=0)


class TestRefiner:
    """The multi-pass refinement loop."""

    def test_empty_input_short_circuits(self) -> None:
        """Empty input short circuits."""
        backend = ScriptedBackend([])
        report = Refiner(backend).run([])
        assert report.lines == []
        assert report.passes == []
        assert backend.prompts == []

    def test_draft_only(self) -> None:
        """Draft only."""
        backend = ScriptedBackend([lines_reply(["Hello"])])
        refiner = Refiner(backend, settings=RefineSettings(critique=False, repair_glossary=False))
        report = refiner.run(["こんにちは"])
        assert report.lines == ["Hello"]
        assert report.passes == ["draft"]

    def test_draft_then_critique(self) -> None:
        """Draft then critique."""
        backend = ScriptedBackend([lines_reply(["Hi"]), lines_reply(["Hello there"])])
        refiner = Refiner(backend, settings=RefineSettings(repair_glossary=False))
        report = refiner.run(["こんにちは"])
        assert report.lines == ["Hello there"]
        assert report.passes == ["draft", "critique"]

    def test_glossary_repair_triggers(self) -> None:
        """Glossary repair triggers."""
        backend = ScriptedBackend(
            [lines_reply(["Big brother, wait"]), lines_reply(["Nii-san, wait"])],
        )
        refiner = Refiner(backend, settings=RefineSettings(critique=False))
        glossary = Glossary([GlossaryEntry(source="兄", target="Nii-san")])
        report = refiner.run(["兄さん、待って"], glossary=glossary)
        assert report.lines == ["Nii-san, wait"]
        assert report.repaired == [0]
        assert "glossary-repair" in report.passes

    def test_glossary_repair_skipped_when_compliant(self) -> None:
        """Glossary repair skipped when compliant."""
        backend = ScriptedBackend([lines_reply(["Nii-san, wait"])])
        refiner = Refiner(backend, settings=RefineSettings(critique=False))
        glossary = Glossary([GlossaryEntry(source="兄", target="Nii-san")])
        report = refiner.run(["兄さん、待って"], glossary=glossary)
        assert report.repaired == []
        assert "glossary-repair" not in report.passes

    def test_round_trip_scores(self) -> None:
        """Round trip scores."""
        backend = ScriptedBackend([lines_reply(["Hello"]), lines_reply(["こんにちは"])])
        refiner = Refiner(
            backend,
            settings=RefineSettings(critique=False, repair_glossary=False, round_trip=True),
        )
        report = refiner.run(["こんにちは"])
        assert len(report.scores) == 1
        assert report.scores[0] > 0.9
        assert "round-trip" in report.passes

    def test_round_trip_skips_blank_drafts(self) -> None:
        """Round trip skips blank drafts."""
        backend = ScriptedBackend([lines_reply([""])])
        refiner = Refiner(
            backend,
            settings=RefineSettings(critique=False, repair_glossary=False, round_trip=True),
        )
        report = refiner.run(["こんにちは"])
        assert report.scores == [0.0]

    def test_relevant_glossary_only_is_injected(self) -> None:
        """Relevant glossary only is injected."""
        backend = ScriptedBackend([lines_reply(["Hello"])])
        refiner = Refiner(backend, settings=RefineSettings(critique=False, repair_glossary=False))
        glossary = Glossary(
            [
                GlossaryEntry(source="兄", target="Nii-san"),
                GlossaryEntry(source="姉", target="Nee-san"),
            ],
        )
        refiner.run(["兄さん"], glossary=glossary)
        prompt = backend.prompts[0][1]
        assert "Nii-san" in prompt
        assert "Nee-san" not in prompt

    def test_backend_exhaustion_yields_blank(self) -> None:
        """Backend exhaustion yields blank."""
        backend = ScriptedBackend([])
        refiner = Refiner(backend, settings=RefineSettings(critique=False, repair_glossary=False))
        assert refiner.run(["あ"]).lines == [""]

    def test_style_guide_is_passed_through(self) -> None:
        """Style guide is passed through."""
        backend = ScriptedBackend([lines_reply(["Hello"])])
        refiner = Refiner(
            backend,
            settings=RefineSettings(critique=False, repair_glossary=False),
            style=StyleGuide(honorifics=HonorificPolicy.DROP, notes="seinen"),
        )
        refiner.run(["あ"])
        assert "seinen" in backend.prompts[0][1]

    def test_scripted_backend_reply_is_json(self) -> None:
        """Scripted backend reply is json."""
        backend = ScriptedBackend([])
        assert json.loads(backend.chat("s", "u"))["lines"] == [""]
