import onnxruntime as ort
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from loguru import logger

from application.inference.semantic_segmentation.segmentation import YOLOv8Seg
from .model_loader import load_onnx_model, load_pytorch_model
from ..inference.keypoint_detection.keypoint_detection import YOLOv8Pose
from ..inference.object_detection.object_detection import YOLOv8
from ..utils.base64utils import base64_to_image
from ..utils.entities import RequestParams, SushineResponse

local_app = FastAPI()
model: ort.InferenceSession = None
model_name: str = ""
model_path: str = ""
model_task: str = ""
model_config: str = ""


def start_service(_model_name: str, _model_path: str, _model_task: str, device: str, _model_config: str, port: int):
    global model, model_name, model_task, model_path, model_config
    # Load the ONNX model
    model_name = _model_name
    model_path = _model_path
    model_task = _model_task
    if model_path.endswith("onnx"):
        model = load_onnx_model(model_path, device)
    elif model_path.endswith("pt"):
        model = load_pytorch_model(model_path, device)
    else:
        error_msg = f"{model_name}'s model type not support!"
        logger.error(error_msg)
        raise Exception(error_msg)

    model_config = _model_config

    uvicorn.run(
        local_app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
        timeout_keep_alive=30
    )


@local_app.get('/health')
async def is_healthy():
    return JSONResponse({"status": model is not None})


@local_app.post("/inf")
async def inference_image(params: RequestParams) -> Response:
    image = base64_to_image(params.params.image)
    if model_task == "object_detection":
        detector = YOLOv8(model, data_config=model_config)
        bbox, score, obj_class = detector(image, conf_threshold=0.25, iou_threshold=0.45)

        response = SushineResponse(
            status=200,
            algorithm_result=[
                {
                    "class_name": detector.classes[obj_class[i]],
                    "box": bbox[i],
                    "confidence": score[i]
                }
                for i in range(len(obj_class))
            ]
        )

    elif model_task == "keypoint_detection":
        detector = YOLOv8Pose(model, data_config=model_config)
        bbox, keypoint1, keypoint2, obj_class, scores = detector(image, conf_threshold=0.8, iou_threshold=0.45)
        response = SushineResponse(
            status=200,
            algorithm_result=[
                {
                    "class_name": detector.classes[obj_class[i]],
                    "confidence": scores[i],
                    "box": bbox[i],
                    "keypoints": [[x, y] for x, y in zip(keypoint1[i], keypoint2[i])]
                }
                for i in range(len(obj_class))
            ]
        )

    elif model_task == "semantic_segmentation":
        detector = YOLOv8Seg(model, data_config=model_config)
        boxes, segments, masks = detector(image, conf_threshold=0.8, iou_threshold=0.45)

        response = SushineResponse(
            status=200,
            algorithm_result=[
                {
                    "class_name": detector.classes[int(boxes[i, -1])],
                    "confidence": boxes[i, -2],
                    "box": boxes[i, :-2].tolist(),
                    "segments": segments[i]
                }
                for i in range(len(segments))
            ]
        )
    else:
        response = SushineResponse(
            status=500,
            message="model task not support!"
        )

    return JSONResponse(response.json())


if __name__ == '__main__':
    pass
