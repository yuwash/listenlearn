import argparse
import asyncio
import dataclasses
import json
import os.path
import sys

import appdirs
import edge_tts
import edge_tts.typing

import common


def get_user_data_dir() -> str:
    app_author = "yuwash.eu"
    app_name = "listenlearn"
    return appdirs.user_data_dir(app_name, app_author)


@dataclasses.dataclass
class Config:
    learning_modes: dict[str, common.LearningMode]
    default_learning_mode: common.LearningMode

    @classmethod
    def load(cls) -> "Config":
        user_data_dir = get_user_data_dir()
        config_file_path = os.path.join(user_data_dir, "config.json")
        data = json.load(open(config_file_path))
        learning_modes = {
            mode_name: common.LearningMode(**mode_config)
            for mode_name, mode_config in data["learning_modes"].items()
        }
        default_learning_mode_name = data["default_learning_mode"]
        return cls(
            learning_modes=learning_modes,
            default_learning_mode=learning_modes[default_learning_mode_name],
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
        target_voice = voices_manager.find(Gender=mode.target_voice_gender, Language=mode.target_language)[0]
        fallback_voice = voices_manager.find(Gender=mode.fallback_voice_gender, Language=mode.fallback_language)[0]
        return cls(
            mode=mode,
            voices_manager=voices_manager,
            target_voice=target_voice,
            fallback_voice=fallback_voice,
        )

    async def _communicate_and_save(self, text: str, output_file: str, voice_name: str) -> None:
        communicate = edge_tts.Communicate(text, voice_name)
        await communicate.save(output_file)

    async def communicate_and_save_target_language(self, text: str, output_file: str) -> None:
        await self._communicate_and_save(text, output_file, self.target_voice["Name"])

    async def communicate_and_save_fallback_language(self, text: str, output_file: str) -> None:
        await self._communicate_and_save(text, output_file, self.fallback_voice["Name"])


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

    args = parser.parse_args()

    mode = (
        config.learning_modes[args.mode]
        if args.mode else config.default_learning_mode
    )
    if args.command == "tts":
        asyncio.run(tts_command(args))
