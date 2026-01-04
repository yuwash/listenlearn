import dataclasses
import enum
import typing


@dataclasses.dataclass(frozen=True)
class TTSParams:
    language: str
    voice_gender: str = "Female"
    rate: float = 1.0
    volume: float = 1.0
    pitch_delta_hz: float = 0.0


class PlaybackItem(enum.StrEnum):
    Target = "target"
    Fallback = "fallback"


@dataclasses.dataclass(frozen=True)
class LearningMode:
    target_params: TTSParams
    fallback_params: TTSParams
    initial_playback_order: typing.Sequence[PlaybackItem] = (
        PlaybackItem.Target,
        PlaybackItem.Fallback,
        PlaybackItem.Target,
        PlaybackItem.Target,
    )
    short_target_review_playback_order: typing.Sequence[PlaybackItem] = (
        PlaybackItem.Target,
        PlaybackItem.Fallback,
        PlaybackItem.Target,
    )
    long_target_review_playback_order: typing.Sequence[PlaybackItem] = (
        PlaybackItem.Target,
    )
