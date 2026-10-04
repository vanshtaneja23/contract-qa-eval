from typing import Any

import pytest

from contract_eval.cuad import apply_manifest, build_manifest, parse_cuad, select_subset


def _record(title: str, context: str, qas: list[dict[str, Any]]) -> dict[str, Any]:
    return {"title": title, "paragraphs": [{"context": context, "qas": qas}]}


CTX = "This Agreement is governed by the laws of Delaware. No non-compete applies."
RAW = {
    "data": [
        _record(
            "ACME-AGREEMENT",
            CTX,
            [
                {
                    "id": "ACME-AGREEMENT__Governing Law",
                    "question": 'Highlight the parts related to "Governing Law"',
                    "is_impossible": False,
                    "answers": [
                        {"answer_start": CTX.index("governed"), "text": "governed by the laws of Delaware"}
                    ],
                },
                {
                    "id": "ACME-AGREEMENT__Cap On Liability",
                    "question": 'Highlight the parts related to "Cap On Liability"',
                    "is_impossible": True,
                    "answers": [],
                },
            ],
        )
    ]
}


def test_parse_extracts_category_spans_and_impossible_flag() -> None:
    (c,) = parse_cuad(RAW)
    law, cap = c.questions
    assert law.category == "Governing Law" and not law.is_impossible
    assert CTX[law.spans[0].start : law.spans[0].end] == "governed by the laws of Delaware"
    assert cap.category == "Cap On Liability" and cap.is_impossible and cap.spans == ()


def test_parse_rejects_misaligned_gold_span() -> None:
    bad = {"data": [_record("X", CTX, [{**RAW["data"][0]["paragraphs"][0]["qas"][0]}])]}
    bad["data"][0]["paragraphs"][0]["qas"][0]["answers"] = [{"answer_start": 0, "text": "governed"}]
    with pytest.raises(ValueError, match="does not match"):
        parse_cuad(bad)


def _many(n: int) -> list[Any]:
    return parse_cuad({"data": [_record(f"C{i:03d}", f"text {i}", []) for i in range(n)]})


def test_subset_is_deterministic_and_order_independent() -> None:
    contracts = _many(50)
    a = select_subset(contracts, 10, seed=42)
    b = select_subset(list(reversed(contracts)), 10, seed=42)
    assert [c.title for c in a] == [c.title for c in b]
    assert [c.title for c in select_subset(contracts, 10, seed=7)] != [c.title for c in a]


def test_manifest_round_trip_and_tamper_detection() -> None:
    contracts = _many(20)
    subset = select_subset(contracts, 5, seed=1)
    manifest = build_manifest(subset, seed=1)
    assert apply_manifest(contracts, manifest) == subset

    manifest["contracts"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed"):
        apply_manifest(contracts, manifest)
