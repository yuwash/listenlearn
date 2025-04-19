import dataclasses


@dataclasses.dataclass(frozen=True)
class TTSParams:
    language: str
    voice_gender: str = "Female"
    rate: float = 1.0
    volume: float = 1.0
    pitch_delta_hz: float = 0.0


@dataclasses.dataclass(frozen=True)
class LearningMode:
    target_params: TTSParams
    fallback_params: TTSParams
