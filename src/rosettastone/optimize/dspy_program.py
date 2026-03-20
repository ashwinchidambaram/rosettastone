"""DSPy module definition for the migration program."""

import dspy


class MigrationSignature(dspy.Signature):  # type: ignore[misc]
    """Given a prompt, produce a response that matches the behavioral style of the source model."""

    prompt: str = dspy.InputField(desc="The user prompt to respond to")
    response: str = dspy.OutputField(desc="Response matching source model behavior")


class MigrationProgram(dspy.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(MigrationSignature)

    def forward(self, prompt: str) -> dspy.Prediction:
        return self.predict(prompt=prompt)
