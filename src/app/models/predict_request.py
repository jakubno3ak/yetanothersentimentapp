from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Text must be a non-empty string")
        return v