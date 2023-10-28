from pydantic import BaseModel


class RequestParams(BaseModel):
    descriptions: list[str]


class ResponsePredict(BaseModel):
    descriptions: list[str]
    predict: list[str]
    predict_quality: list[str]

