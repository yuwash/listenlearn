import argparse
import asyncio
import csv
import dataclasses
import json
import os
import os.path
import sys
import tempfile
import contextlib
from collections import deque

import appdirs
import edge_tts
import edge_tts.typing

import common
import learningset


def get_user_data_dir() -> str:
    app_author = "yuwash.eu"
    app_name = "listenlearn"
    return appdirs.user_data_dir(app_name, app_author)


@dataclasses.dataclass
class Config:
    learning_modes: dict[str, common.LearningMode]
    default_learning_mode: common.LearningMode
    openai_providers: dict[str, learningset.OpenAIProvider]
    default_openai_provider: learningset.OpenAIProvider

    @classmethod
    def load(cls) -> "Config":
        user_data_dir = get_user_data_dir()
        config_file_path = os.path.join(user_data_dir, "config.json")
        data = json.load(open(config_file_path))
        learning_modes = {
            mode_name: common.LearningMode(
                target_params=common.TTSParams(**mode_config["target_params"]),
                fallback_params=common.TTSParams(**mode_config["fallback_params"]),
            )
            for mode_name, mode_config in data["learning_modes"].items()
        }
        default_learning_mode_name = data["default_learning_mode"]
        openai_providers = {
            provider_name: learningset.OpenAIProvider(
                base_url=provider_config["base_url"],
                model=provider_config["model"],
                authless=provider_config["authless"],
            )
            for provider_name, provider_config in data["openai_providers"].items()
        }
        default_openai_provider = openai_providers[
            data["default_openai_provider"]
        ]
        return cls(
            learning_modes=learning_modes,
            default_learning_mode=learning_modes[default_learning_mode_name],
            openai_providers=openai_providers,
            default_openai_provider=default_openai_provider,
        )


@dataclasses.dataclass(frozen=True)
class LearningModeTTS:
    mode: common.LearningMode
    voices_manager: edge_tts.VoicesManager
    target_voice: edge_tts.typing.VoicesManagerVoice
    fallback_voice: edge_tts.typing.VoicesManagerVoice

    @classmethod
    async def create(cls, mode: common.LearningMode) -> "LearningModeTTS":
        voices_manager = await edge_tts.VoicesManager.create()
        target_voice = voices_manager.find(
            Gender=mode.target_params.voice_gender, Language=mode.target_params.language
        )[0]
        fallback_voice = voices_manager.find(
            Gender=mode.fallback_params.voice_gender,
            Language=mode.fallback_params.language,
        )[0]
        return cls(
            mode=mode,
            voices_manager=voices_manager,
            target_voice=target_voice,
            fallback_voice=fallback_voice,
        )

    async def _communicate_and_save(
        self,
        text: str,
        output_file: str,
        voice_name: str,
        tts_params: common.TTSParams,
    ) -> None:
        rate_str = f"{int(tts_params.rate * 100 - 100):+d}%"
        volume_str = f"{int(tts_params.volume * 100 - 100):+d}%"
        pitch_str = f"{int(tts_params.pitch_delta_hz):+d}Hz"
        communicate = edge_tts.Communicate(
            text, voice_name, rate=rate_str, volume=volume_str, pitch=pitch_str
        )
        await communicate.save(output_file)

    async def communicate_and_save_target_language(
        self, text: str, output_file: str
    ) -> None:
        await self._communicate_and_save(
            text,
            output_file,
            voice_name=self.target_voice["Name"],
            tts_params=self.mode.target_params,
        )

    async def communicate_and_save_fallback_language(
        self, text: str, output_file: str
    ) -> None:
        await self._communicate_and_save(
            text,
            output_file,
            voice_name=self.fallback_voice["Name"],
            tts_params=self.mode.fallback_params,
        )


async def learningset_command(
    args: argparse.Namespace, mode: common.LearningMode, config: Config
) -> None:
    text = sys.stdin.read()
    print("Extracting sentences...")

    sentences_extractor = learningset.OpenAILearningSetExtraction(
        openai_provider=config.default_openai_provider,
        prompt_template=learningset.EXTRACT_SENTENCES_PROMPT,
    )
    sentences_set = sentences_extractor.extract(text, mode)

    total = len(sentences_set.items)
    result_set = learningset.LearningSet(items=[])
    for i, item in enumerate(sentences_set.items):
        print(f"Processing {i + 1}/{total}: {item.target_language_text}")
        sentence_text = item.target_language_text

        short_chunks_extractor = learningset.OpenAILearningSetExtraction(
            openai_provider=config.default_openai_provider,
            prompt_template=learningset.EXTRACT_SHORT_CHUNKS_PROMPT,
        )
        result_set.extend(short_chunks_extractor.extract(sentence_text, mode))
        result_set.items.append(item)  # Add the sentence itself.
    learningset.learning_set_to_csv(result_set, args.out)


@dataclasses.dataclass(frozen=True)
class TTSItem:
    target_language_text: str
    fallback_language_text: str
    target_audio_file: str
    fallback_audio_file: str

    @property
    def measure(self) -> int:
        return 1 if len(self.target_language_text) < 50 else 2


class ReviewQueue:
    def __init__(self) -> None:
        self._queue: deque[list[TTSItem]] = deque()
        """
        Keyed by minimum wait time.
        Wait time is measured by a unit of learning item with at least
        50 characters (probably a sentence) or 2 shorter items.
        """

    def __len__(self) -> int:
        return len(self._queue)

    def add(self, item: TTSItem, wait_time: int) -> None:
        try:
            self._queue[wait_time].append(item)
        except IndexError:
            self._queue.extend((wait_time - len(self._queue)) * [[]] + [[item]])

    def progress(self, progress: int) -> None:
        effective_progress = min(progress, len(self._queue) - 1)
        if effective_progress <= 0:
            return
        for __ in range(effective_progress):
            self._queue[0].extend(self._queue[1])
            del self._queue[1]

    def pop(self) -> list[TTSItem]:
        try:
            items = self._queue[0]
        except IndexError:
            return []

        self._queue[0] = []
        # Not actually popping as that would change the wait times.
        return items


async def settts_command(args: argparse.Namespace, mode: common.LearningMode) -> None:
    tts = await LearningModeTTS.create(mode)
    concat_list = []
    review_queue = ReviewQueue()

    def review():
        while next_queue := review_queue.pop():
            for queue_item in next_queue:
                if queue_item.measure > 1:
                    concat_list.append(queue_item.target_audio_file)
                else:
                    concat_list.extend(
                        [
                            queue_item.target_audio_file,
                            queue_item.fallback_audio_file,
                            queue_item.target_audio_file,
                        ]
                    )
            added_count = len(next_queue)
            progress = (
                sum(item.measure for item in next_queue)
                if len(review_queue) > added_count
                # No need to iterate as the (potentially
                # larger) correct progress count won’t make a
                # difference.
                else added_count
            )
            review_queue.progress(progress)

    with open(args.csv_file) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row

        if args.cache:
            tmpdir = args.cache
            os.makedirs(tmpdir, exist_ok=True)
            tmpdir_context = contextlib.nullcontext(tmpdir)
        else:
            tmpdir_context = tempfile.TemporaryDirectory()

        with tmpdir_context as tmpdir:
            for i, row in enumerate(reader):
                review()

                tts_item = TTSItem(
                    target_language_text=row[0],
                    fallback_language_text=row[1],
                    target_audio_file=os.path.join(tmpdir, f"target_{i}.mp3"),
                    fallback_audio_file=os.path.join(tmpdir, f"fallback_{i}.mp3"),
                )

                if not os.path.exists(tts_item.target_audio_file):
                    await tts.communicate_and_save_target_language(
                        tts_item.target_language_text, tts_item.target_audio_file
                    )
                if not os.path.exists(tts_item.fallback_audio_file):
                    await tts.communicate_and_save_fallback_language(
                        tts_item.fallback_language_text, tts_item.fallback_audio_file
                    )

                concat_list.extend(
                    [
                        tts_item.target_audio_file,
                        tts_item.fallback_audio_file,
                        tts_item.target_audio_file,
                        tts_item.target_audio_file,
                    ]
                )
                review_queue.progress(tts_item.measure)
                review_queue.add(tts_item, 2)

            while 0 < len(review_queue):
                # Process any remaining items.
                review()

            with open(args.out, "wb") as out_file:
                for audio_file in concat_list:
                    with open(audio_file, "rb") as audio:
                        out_file.write(audio.read())


async def tts_command(args: argparse.Namespace, mode: common.LearningMode) -> None:
    text = sys.stdin.read()
    tts = await LearningModeTTS.create(mode)
    await tts.communicate_and_save_target_language(text, args.out)
    # await tts.communicate_and_save_fallback_language(text, output_file)


if __name__ == "__main__":
    config = Config.load()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mode", type=str, help="Learning mode name as defined in config.json"
    )
    subparsers = parser.add_subparsers(dest="command")

    # create the parser for the "tts" command
    parser_tts = subparsers.add_parser("tts")
    parser_tts.add_argument("-o", "--out", type=str, required=True, help="Output file")

    # create the parser for the "learningset" command
    parser_learningset = subparsers.add_parser("learningset")
    parser_learningset.add_argument(
        "-o", "--out", type=str, required=True, help="Output file"
    )

    # create the parser for the "settts" command
    parser_settts = subparsers.add_parser("settts")
    parser_settts.add_argument(
        "csv_file", type=str, help="CSV file containing the learning set"
    )
    parser_settts.add_argument(
        "-o", "--out", type=str, required=True, help="Output file"
    )
    parser_settts.add_argument(
        "--cache", type=str, help="Cache directory for TTS files"
    )

    args = parser.parse_args()

    mode = (
        config.learning_modes[args.mode] if args.mode else config.default_learning_mode
    )
    if args.command == "tts":
        asyncio.run(tts_command(args, mode))
    elif args.command == "learningset":
        asyncio.run(learningset_command(args, mode, config))
    elif args.command == "settts":
        asyncio.run(settts_command(args, mode))
