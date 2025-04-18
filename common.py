import dataclasses


@dataclasses.dataclass(frozen=True)
class LearningMode:
    target_language: str
    fallback_language: str
    target_voice_gender: str = "Female"
    fallback_voice_gender: str = "Female"
