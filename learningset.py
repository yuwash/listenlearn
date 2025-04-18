import dataclasses
import re

import instructor
import openai
import pydantic

import common


class AuthlessOpenAI(openai.OpenAI):
    @property
    def auth_headers(self):
        return {}  # Don’t set Bearer.


@dataclasses.dataclass(frozen=True)
class OpenAIProvider:
    base_url: str
    model: str
    authless: bool = False

    def get_client(self):
        openai_client = (
            AuthlessOpenAI(
                api_key="",
                base_url=self.base_url,
            )
            if self.authless
            else openai.OpenAI(
                # Use api_key from environment variable.
                base_url=self.base_url,
            )
        )
        return instructor.from_openai(openai_client)


class LearningItem(pydantic.BaseModel):
    target_language_text: str
    fallback_language_text: str


import csv
import functools


class LearningSet(pydantic.BaseModel):
    items: list[LearningItem]

    def extend(self, other: "LearningSet") -> None:
        self.items.extend(other.items)


def learning_set_to_csv(learning_set: LearningSet, output_file: str) -> None:
    with open(output_file, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["target_language_text", "fallback_language_text"]
        )  # Write header
        for item in learning_set.items:
            writer.writerow([item.target_language_text, item.fallback_language_text])


@dataclasses.dataclass(frozen=True)
class OpenAILearningSetExtraction:
    openai_provider: "OpenAIProvider"
    prompt_template: str

    def extract(self, text: str, mode: common.LearningMode) -> LearningSet:
        client = self.openai_provider.get_client()
        fallback_language = mode.fallback_params.language
        prompt = self.prompt_template.format(
            fallback_language=fallback_language, text=text
        )
        return client.chat.completions.create(
            model=self.openai_provider.model,
            response_model=LearningSet,
            messages=[{"role": "user", "content": prompt}],
        )


EXTRACT_SENTENCES_PROMPT = (
    "Please split the text into sentences. "
    "Please use the language with code '{fallback_language}' as fallback language. "
    "Please include the original sentence as the target language. "
    "Please translate the sentences into the fallback language. "
    "Text to process: {text}"
)

EXTRACT_SHORT_CHUNKS_PROMPT = (
    "Please split the text into short chunks. "
    "The chunk should consist of around 1 to 3 words and should be translatable by retaining the meaning. "
    "Compound words should be kept intact. "
    "The chunk shouldn’t be longer than around 20 characters. "
    "Very common words can be ignored unless appearing within a compound word. "
    "Please use the language with code '{fallback_language}' as fallback language. "
    "Please include the original chunk as the target language. "
    "Please translate the chunk into the fallback language. "
    "Text to process: {text}"
)
