"""Tests for the TeacherStudentOptimizer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rosettastone.core.types import PromptPair
from rosettastone.optimize.pipeline_config import PipelineConfig, PipelineModuleConfig


def _make_prompt_pair(prompt: str = "Hello", response: str = "Hi") -> PromptPair:
    return PromptPair(prompt=prompt, response=response, source_model="openai/gpt-4o")


def _make_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        name="test_pipeline",
        modules=[
            PipelineModuleConfig(
                name="step1",
                prompt_template="You are helpful.",
                input_fields=["user_query"],
                output_fields=["step1_result"],
                depends_on=[],
            ),
            PipelineModuleConfig(
                name="step2",
                prompt_template="Refine the result.",
                input_fields=["step1_result"],
                output_fields=["final_answer"],
                depends_on=["step1"],
            ),
        ],
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-haiku-4-5",
    )


def _make_migration_config() -> MagicMock:
    config = MagicMock()
    config.source_model = "openai/gpt-4o"
    config.target_model = "anthropic/claude-haiku-4-5"
    return config


class TestTeacherStudentOptimizer:
    """Tests for TeacherStudentOptimizer."""

    def test_optimize_raises_not_implemented(self) -> None:
        """optimize() must raise NotImplementedError — this class is pipeline-only."""
        from rosettastone.optimize.teacher_student import TeacherStudentOptimizer

        optimizer = TeacherStudentOptimizer()
        with pytest.raises(NotImplementedError, match="pipeline_optimize"):
            optimizer.optimize(
                train_set=[_make_prompt_pair()],
                val_set=[_make_prompt_pair()],
                config=_make_migration_config(),
            )

    def test_generate_teacher_demos_returns_prompt_pair_list(self) -> None:
        """generate_teacher_demos() returns a list of PromptPairs with correct length."""
        from rosettastone.optimize.teacher_student import TeacherStudentOptimizer

        train_set = [
            _make_prompt_pair("What is 2+2?"),
            _make_prompt_pair("What is the sky color?"),
        ]
        pipeline_config = _make_pipeline_config()

        mock_lm = MagicMock()
        mock_lm.return_value = "4"

        with patch("dspy.LM", return_value=mock_lm) as mock_lm_cls:
            mock_predict_result = MagicMock()
            mock_predict_result.completions = MagicMock()
            # dspy.LM.__call__ returns a list of strings
            mock_lm.return_value = ["teacher response"]

            optimizer = TeacherStudentOptimizer()
            demos = optimizer.generate_teacher_demos(train_set, "openai/gpt-4o", pipeline_config)

        assert len(demos) == len(train_set)
        for demo in demos:
            assert isinstance(demo, PromptPair)

    def test_generate_teacher_demos_empty_train_set(self) -> None:
        """generate_teacher_demos() returns empty list when train_set is empty."""
        from rosettastone.optimize.teacher_student import TeacherStudentOptimizer

        pipeline_config = _make_pipeline_config()
        optimizer = TeacherStudentOptimizer()

        with patch("dspy.LM", return_value=MagicMock(return_value=[""])):
            demos = optimizer.generate_teacher_demos([], "openai/gpt-4o", pipeline_config)

        assert demos == []

    def test_generate_teacher_demos_uses_first_module_input_field(self) -> None:
        """generate_teacher_demos() uses the first input field of the first module as prompt field."""
        from rosettastone.optimize.teacher_student import TeacherStudentOptimizer

        pipeline_config = _make_pipeline_config()
        # First module has input_fields=["user_query"]

        train_set = [_make_prompt_pair("test prompt")]
        captured_calls: list[dict] = []

        def fake_lm_call(*args, **kwargs):  # type: ignore[no-untyped-def]
            captured_calls.append({"args": args, "kwargs": kwargs})
            return ["teacher answer"]

        mock_lm_instance = MagicMock(side_effect=fake_lm_call)

        with patch("dspy.LM", return_value=mock_lm_instance):
            optimizer = TeacherStudentOptimizer()
            demos = optimizer.generate_teacher_demos(train_set, "openai/gpt-4o", pipeline_config)

        # Verify the LM was called and a demo was produced
        assert len(demos) == 1
        assert demos[0].prompt == "test prompt"

    def test_pipeline_optimize_calls_optimize_pipeline(self) -> None:
        """pipeline_optimize() calls optimize_pipeline with teacher demos."""
        from rosettastone.optimize.teacher_student import TeacherStudentOptimizer

        pipeline_config = _make_pipeline_config()
        migration_config = _make_migration_config()
        train_set = [_make_prompt_pair("q1"), _make_prompt_pair("q2")]

        expected_result = {"step1": "optimized instructions 1", "step2": "optimized instructions 2"}
        teacher_demos = [_make_prompt_pair("q1", "teacher ans 1"), _make_prompt_pair("q2", "teacher ans 2")]

        optimizer = TeacherStudentOptimizer()

        with patch.object(optimizer, "generate_teacher_demos", return_value=teacher_demos) as mock_gen, \
             patch(
                 "rosettastone.optimize.teacher_student.optimize_pipeline",
                 return_value=expected_result,
             ) as mock_opt:
            result = optimizer.pipeline_optimize(pipeline_config, train_set, migration_config)

        mock_gen.assert_called_once_with(train_set, migration_config.source_model, pipeline_config)
        mock_opt.assert_called_once_with(pipeline_config, teacher_demos, migration_config)
        assert result == expected_result

    def test_pipeline_optimize_returns_dict(self) -> None:
        """pipeline_optimize() returns a dict[str, str]."""
        from rosettastone.optimize.teacher_student import TeacherStudentOptimizer

        pipeline_config = _make_pipeline_config()
        migration_config = _make_migration_config()

        teacher_demos = [_make_prompt_pair("q", "a")]
        fake_result = {"step1": "instructions", "step2": "more instructions"}

        optimizer = TeacherStudentOptimizer()

        with patch.object(optimizer, "generate_teacher_demos", return_value=teacher_demos), \
             patch(
                 "rosettastone.optimize.teacher_student.optimize_pipeline",
                 return_value=fake_result,
             ):
            result = optimizer.pipeline_optimize(pipeline_config, [], migration_config)

        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(key, str)
            assert isinstance(val, str)

    def test_teacher_demo_source_model_set_correctly(self) -> None:
        """Teacher demos should have source_model set to the source LLM model string."""
        from rosettastone.optimize.teacher_student import TeacherStudentOptimizer

        pipeline_config = _make_pipeline_config()
        train_set = [_make_prompt_pair("some prompt")]
        source_model = "openai/gpt-4o"

        with patch("dspy.LM", return_value=MagicMock(return_value=["teacher response"])):
            optimizer = TeacherStudentOptimizer()
            demos = optimizer.generate_teacher_demos(train_set, source_model, pipeline_config)

        assert len(demos) == 1
        assert demos[0].source_model == source_model
