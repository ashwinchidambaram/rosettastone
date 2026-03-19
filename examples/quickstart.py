"""Minimal Python usage example for RosettaStone."""

from rosettastone import Migrator, MigrationConfig

config = MigrationConfig(
    source_model="openai/gpt-4o",
    target_model="anthropic/claude-sonnet-4",
    data_path="examples/sample_data.jsonl",
)

migrator = Migrator(config)
result = migrator.run()

print(f"Confidence: {result.confidence_score:.0%}")
print(f"Improvement over baseline: +{result.improvement:.0%}")
print(f"Cost: ${result.cost_usd:.2f}")
