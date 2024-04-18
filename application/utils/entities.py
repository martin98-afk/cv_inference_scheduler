
from typing import Optional, List

from pydantic import BaseModel


class VideoStreamParams(BaseModel):
    video_stream_url: Optional[str] = None
    callback_urls: Optional[List[str]] = []
    interval: int = 5


class ImageParams(BaseModel):
    image: Optional[str] = None


class InferenceParams(BaseModel):
    score_threshold: float = 0.15
    nms_threshold: float = 0.3
    eta: float = 0.5


class RequestParams(BaseModel):
    model_name: str = ""
    params: ImageParams = ImageParams()
    stream_params: VideoStreamParams = VideoStreamParams()
    inference_params: Optional[InferenceParams] = None


class AlgorithmConfig(BaseModel):
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    box: Optional[List[float]] = None
    keypoints: Optional[List[List[float]]] = None
    segments: Optional[List[List[float]]] = None


class SushineResponse(BaseModel):
    status: int
    message: Optional[str] = None
    algorithm_result: Optional[List[AlgorithmConfig]] = None
    processed_image: Optional[str] = None