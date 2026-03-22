"""Prompt template domains for synthetic E2E test data generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptTemplate:
    system_message: str
    user_template: str
    fill_values: list[dict[str, str]]
    expected_output_type: str


@dataclass
class DomainSpec:
    name: str
    templates: list[PromptTemplate]
    count: int = 0  # computed from templates × fill_values

    def __post_init__(self) -> None:
        self.count = sum(len(t.fill_values) for t in self.templates)


# --- Domain definitions ---

_JSON_EXTRACTION = DomainSpec(
    name="json_extraction",
    templates=[
        PromptTemplate(
            system_message="Respond only with valid JSON. No explanation.",
            user_template="Extract name, age, and city from: '{text}'",
            fill_values=[
                {"text": "Maria Garcia, 34, Barcelona"},
                {"text": "John Smith is 28 years old and lives in New York"},
                {"text": "Yuki Tanaka (age 41) — Tokyo resident"},
                {"text": "Ahmed Hassan, born 1990, Cairo, Egypt"},
            ],
            expected_output_type="json",
        ),
        PromptTemplate(
            system_message="Respond only with valid JSON. No explanation.",
            user_template="Parse this event into JSON with fields date, event, location: '{text}'",
            fill_values=[
                {"text": "Concert at Madison Square Garden on March 15, 2025"},
                {"text": "Team meeting in Conference Room B, tomorrow at 2pm"},
                {"text": "Wedding reception, June 20 2025, Grand Hotel ballroom"},
                {"text": "Hackathon at Google Campus, April 5-6 2025"},
            ],
            expected_output_type="json",
        ),
    ],
)

_CLASSIFICATION = DomainSpec(
    name="classification",
    templates=[
        PromptTemplate(
            system_message=(
                "Respond with only the label: Positive, Negative, or Mixed. Nothing else."
            ),
            user_template="Classify sentiment: '{text}'",
            fill_values=[
                {"text": "Food was decent but service was terribly slow"},
                {"text": "Absolutely loved every minute of this movie!"},
                {"text": "The product broke after two days. Complete waste of money."},
                {"text": "It's okay, nothing special but does the job"},
            ],
            expected_output_type="classification",
        ),
        PromptTemplate(
            system_message=(
                "Respond with only one category: Bug, Feature, Question, or Documentation."
                " Nothing else."
            ),
            user_template="Classify this GitHub issue: '{text}'",
            fill_values=[
                {"text": "App crashes when clicking the submit button on mobile"},
                {"text": "Can we add dark mode support?"},
                {"text": "How do I configure the Redis connection?"},
                {"text": "The API docs are missing the rate limit section"},
            ],
            expected_output_type="classification",
        ),
    ],
)

_SHORT_QA = DomainSpec(
    name="short_qa",
    templates=[
        PromptTemplate(
            system_message="Answer concisely in 2-3 sentences.",
            user_template="{question}",
            fill_values=[
                {"question": "What is the difference between TCP and UDP?"},
                {"question": "Explain what a hash table is."},
                {"question": "What is the CAP theorem in distributed systems?"},
                {"question": "What is the difference between a stack and a queue?"},
            ],
            expected_output_type="short_text",
        ),
        PromptTemplate(
            system_message="Answer concisely in 2-3 sentences.",
            user_template="{question}",
            fill_values=[
                {"question": "What is the purpose of an index in a database?"},
                {"question": "What does idempotent mean in the context of APIs?"},
                {"question": "What is the difference between authentication and authorization?"},
                {"question": "What is a race condition?"},
            ],
            expected_output_type="short_text",
        ),
    ],
)

_LONG_EXPLANATION = DomainSpec(
    name="long_explanation",
    templates=[
        PromptTemplate(
            system_message="Provide a detailed explanation in 3-5 paragraphs.",
            user_template="{topic}",
            fill_values=[
                {"topic": "Explain microservices vs monolithic architectures"},
                {"topic": "Explain how garbage collection works in modern programming languages"},
                {"topic": "Explain the pros and cons of NoSQL vs relational databases"},
                {"topic": "Explain how HTTPS and TLS work to secure web traffic"},
            ],
            expected_output_type="long_text",
        ),
        PromptTemplate(
            system_message="Provide a detailed explanation in 3-5 paragraphs.",
            user_template="{topic}",
            fill_values=[
                {"topic": "Explain event-driven architecture and when to use it"},
                {"topic": "Explain how container orchestration with Kubernetes works"},
                {"topic": "Explain the principles of clean code and why they matter"},
                {"topic": "Explain CI/CD pipelines and their benefits for software teams"},
            ],
            expected_output_type="long_text",
        ),
    ],
)

_CODE_GENERATION = DomainSpec(
    name="code_generation",
    templates=[
        PromptTemplate(
            system_message=(
                "Write only Python code. Include a brief docstring but no other explanation."
            ),
            user_template="{task}",
            fill_values=[
                {"task": "Write a Python function to check if a number is prime"},
                {"task": "Write a Python function that reverses a linked list"},
                {"task": "Write a Python function to find the longest common subsequence"},
            ],
            expected_output_type="short_text",
        ),
        PromptTemplate(
            system_message=(
                "Write only Python code. Include a brief docstring but no other explanation."
            ),
            user_template="{task}",
            fill_values=[
                {"task": "Write a Python class that implements a basic LRU cache"},
                {"task": "Write a Python function to merge two sorted lists into one sorted list"},
                {"task": "Write a Python function that validates an email address using regex"},
            ],
            expected_output_type="short_text",
        ),
    ],
)

_REWRITING = DomainSpec(
    name="rewriting",
    templates=[
        PromptTemplate(
            system_message="Rewrite the given text as requested. Output only the rewritten text.",
            user_template="Rewrite for a 5th grader: '{text}'",
            fill_values=[
                {
                    "text": "Photosynthesis is the biochemical process by which chloroplasts "
                    "convert light energy into chemical energy stored in glucose."
                },
                {
                    "text": "The mitochondria facilitate oxidative phosphorylation to "
                    "produce adenosine triphosphate from metabolic substrates."
                },
                {
                    "text": "Quantum entanglement describes a phenomenon where particles "
                    "become correlated such that the state of one instantly influences the other."
                },
            ],
            expected_output_type="short_text",
        ),
        PromptTemplate(
            system_message="Rewrite the given text as requested. Output only the rewritten text.",
            user_template="Make this more formal and professional: '{text}'",
            fill_values=[
                {"text": "Hey, the server's down again. Can someone fix it ASAP?"},
                {"text": "This code is a total mess. We need to redo the whole thing."},
                {"text": "The new feature is pretty cool but it's kinda buggy still."},
            ],
            expected_output_type="short_text",
        ),
    ],
)


ALL_DOMAINS: list[DomainSpec] = [
    _JSON_EXTRACTION,
    _CLASSIFICATION,
    _SHORT_QA,
    _LONG_EXPLANATION,
    _CODE_GENERATION,
    _REWRITING,
]
