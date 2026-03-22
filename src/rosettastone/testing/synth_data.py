"""Synthetic data generator — calls source model to produce ground-truth baselines."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import litellm

from rosettastone.testing.domains import ALL_DOMAINS, DomainSpec

logger = logging.getLogger(__name__)


@dataclass
class GeneratedPair:
    messages: list[dict[str, str]]
    response_text: str
    output_type: str


class SyntheticDataGenerator:
    """Generate prompt/response pairs by calling a real LLM for ground-truth responses."""

    def __init__(
        self,
        source_model: str,
        domains: list[DomainSpec] | None = None,
    ) -> None:
        self.source_model = source_model
        self.domains = domains or ALL_DOMAINS

    def generate(self) -> list[GeneratedPair]:
        """Generate all pairs across all domains. Calls the source model via LiteLLM."""
        pairs: list[GeneratedPair] = []

        for domain in self.domains:
            for template in domain.templates:
                for fill in template.fill_values:
                    user_content = template.user_template.format(**fill)
                    messages: list[dict[str, str]] = [
                        {"role": "system", "content": template.system_message},
                        {"role": "user", "content": user_content},
                    ]

                    try:
                        response = litellm.completion(
                            model=self.source_model,
                            messages=messages,
                            temperature=0.0,
                        )
                        response_text = response.choices[0].message.content or ""
                    except Exception:
                        logger.exception(
                            "Failed to generate response for %s domain, prompt: %s",
                            domain.name,
                            user_content[:80],
                        )
                        continue

                    pairs.append(
                        GeneratedPair(
                            messages=messages,
                            response_text=response_text,
                            output_type=template.expected_output_type,
                        )
                    )
                    logger.debug(
                        "Generated %s pair (%s): %d chars",
                        domain.name,
                        template.expected_output_type,
                        len(response_text),
                    )

        logger.info("Generated %d synthetic pairs from %s", len(pairs), self.source_model)
        return pairs
