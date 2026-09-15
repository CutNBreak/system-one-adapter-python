"""Provider schemas preserve caller-defined question IDs and choice labels."""

import msgspec
import pytest

from system_one_adapter import Choice, Noul
from system_one_adapter._schema import create_llm_output_model, create_raw_output_schema
from system_one_adapter._utils.probability_normalization import AnswerMode

SCHEMA_KEYWORDS = ["title", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]


@pytest.mark.parametrize("answer_mode", ["probabilities", "discrete"])
def test_question_ids_can_match_schema_keywords(answer_mode: AnswerMode) -> None:
    questions = {key: Noul(instructions=f"Evaluate {key}.") for key in SCHEMA_KEYWORDS}
    output_model = create_llm_output_model(questions, answer_mode)
    schema = create_raw_output_schema(output_model)
    answers = schema["$defs"]["TypeSafeAnswers"]

    assert schema["properties"]["answers"] == {"$ref": "#/$defs/TypeSafeAnswers"}
    assert "Use these property names verbatim" in answers["description"]
    assert set(answers["properties"]) == set(answers["required"]) == set(questions)
    for key, answer in answers["properties"].items():
        assert questions[key].instructions in answer["description"]
        assert not set(answer) & set(SCHEMA_KEYWORDS)

    payload = {"answers": dict.fromkeys(questions, True if answer_mode == "discrete" else 0.8)}
    assert msgspec.to_builtins(msgspec.json.decode(msgspec.json.encode(payload), type=output_model)) == payload


def test_probability_labels_can_match_schema_keywords() -> None:
    criteria = {key: f"The {key} option." for key in SCHEMA_KEYWORDS}
    output_model = create_llm_output_model({"level": Choice(criteria=criteria)}, "probabilities")
    schema = create_raw_output_schema(output_model)
    probabilities = schema["$defs"]["ProbabilityMap0"]

    assert set(probabilities["properties"]) == set(probabilities["required"]) == set(criteria)
    assert "title" not in probabilities
    for key, probability in probabilities["properties"].items():
        assert probability["description"] == criteria[key]
        assert not set(probability) & set(SCHEMA_KEYWORDS)

    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(
            msgspec.json.encode({"answers": {"level": dict.fromkeys(criteria, 2)}}),
            type=output_model,
        )
