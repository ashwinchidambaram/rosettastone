"""Tests for shadow proxy behavior."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from rosettastone.shadow.config import ShadowConfig


class TestShadowProxy:
    def test_primary_response_returned_on_success(self, tmp_path):
        """Primary model response is returned immediately."""

        import scripts.shadow_proxy as proxy_module

        shadow_cfg = ShadowConfig(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
        )
        proxy_module._shadow_config = shadow_cfg
        proxy_module._log_dir = tmp_path

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "primary answer"
        mock_response.usage.total_tokens = 10
        mock_response._hidden_params = {"response_cost": 0.001}

        with patch("litellm.completion", return_value=mock_response):
            from fastapi.testclient import TestClient

            client = TestClient(proxy_module.app_proxy)
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "primary answer"

    def test_shadow_failure_does_not_block_primary(self, tmp_path):
        """Shadow call raising an exception does not affect primary response."""
        import scripts.shadow_proxy as proxy_module

        shadow_cfg = ShadowConfig(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
        )
        proxy_module._shadow_config = shadow_cfg
        proxy_module._log_dir = tmp_path

        primary_mock = MagicMock()
        primary_mock.choices = [MagicMock()]
        primary_mock.choices[0].message.content = "ok"
        primary_mock.usage.total_tokens = 5
        primary_mock._hidden_params = {}

        call_count = 0

        def _litellm_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return primary_mock
            raise RuntimeError("shadow model down")

        with patch("litellm.completion", side_effect=_litellm_side_effect):
            from fastapi.testclient import TestClient

            client = TestClient(proxy_module.app_proxy)
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "test"}]},
            )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "ok"

    def test_log_entry_written(self, tmp_path):
        """A ShadowLogEntry JSONL file is written after each request."""
        import scripts.shadow_proxy as proxy_module
        from rosettastone.shadow.log_format import read_log_entries

        shadow_cfg = ShadowConfig(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
        )
        proxy_module._shadow_config = shadow_cfg
        proxy_module._log_dir = tmp_path

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "answer"
        mock_resp.usage.total_tokens = 8
        mock_resp._hidden_params = {"response_cost": 0.0}

        with patch("litellm.completion", return_value=mock_resp):
            from fastapi.testclient import TestClient

            client = TestClient(proxy_module.app_proxy)
            client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

        # Wait briefly for background task
        import time

        time.sleep(0.2)

        entries = read_log_entries(tmp_path)
        assert len(entries) >= 1
        assert entries[0].source_model == "openai/gpt-4o"
