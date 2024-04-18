import os
import signal
import time
import traceback
from multiprocessing import Process
from typing import Optional, Dict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import httpx

from ..utils.base64utils import image_to_base64
from ..utils.dataclass import StreamInstance
from ..utils.entities import RequestParams, SushineResponse
from .model_manager import ModelManager
from ..utils.uuid import random_uuid
from ..video_stream.video_camera import RTSCapture
from ..video_stream.video_stream_manager import VideoStreamManager

manager = ModelManager()
client: Optional[httpx.AsyncClient] = None
pid_dict: Dict[str, StreamInstance] = {}
video_stream_manager = VideoStreamManager()


def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()


@app.on_event("startup")
async def on_startup():
    global client
    client = httpx.AsyncClient()


@app.on_event("shutdown")
async def on_shutdown():
    await client.aclose()


@app.get("/models")
async def list_models():
    return JSONResponse({
        "code": 0,
        "message": "",
        "data": [
            {
                "name": key,
                "model_path": value.model_path,
                "port": value.port,
                "healthy": (await manager.check_model_status(key)).value
            }
            for key, value in manager.model_dict.items()
        ]
    })


@app.delete("/")
def delete_process(rtsp_url: str, model_name: str):
    model_port = manager.model_dict[model_name].port
    video_stream_manager.remove_task(rtsp_url, model_port)
    if len(video_stream_manager.streams[rtsp_url].tasks) == 0:
        os.kill(pid_dict[rtsp_url].process.pid, signal.SIGKILL)
        pid_dict.pop(rtsp_url)
    else:
        restart_stream_process(rtsp_url)
    logger.info(f"stop stream detection process! task: {model_name}, url: {rtsp_url}")

    return JSONResponse({"status": "detection process stops!"})


@app.post("/inf")
async def unified_inference(params: RequestParams, request: Request):
    try:
        model_port = manager.model_dict[params.model_name].port
    except:
        return SushineResponse(
            status=503,
            message="model name not found in model list!"
        )

    logger.info(f"Receive Post request, {params.model_name} ==> {model_port}")

    # Check if the request is for stream processing
    if params.stream_params.video_stream_url:
        return await handle_stream(params, model_port)
    # Otherwise, handle it as an image processing request
    else:
        return await handle_image(params, request, model_port)


def stream_detect(rtsp_url: str, stream_manager: VideoStreamManager) -> None:
    video_stream = RTSCapture.create(rtsp_url)
    video_stream.start_read()
    while True:
        _, image = video_stream.read_latest_frame()
        if image is not None:
            image = image_to_base64(image)
            for task in stream_manager.streams[rtsp_url].tasks:
                if not task.check_task():
                    task.start(image)


def restart_stream_process(rtsp_url: str):
    pid = random_uuid()
    # 获取模型所在端口
    if rtsp_url in pid_dict:
        os.kill(pid_dict[rtsp_url].process.pid, signal.SIGKILL)
        pid_dict.pop(rtsp_url)
        time.sleep(2)

    process = Process(target=stream_detect,
                      name=pid,
                      args=(rtsp_url, video_stream_manager))
    process.start()

    pid_dict[rtsp_url] = StreamInstance(
        pid=pid,
        process=process
    )


async def handle_stream(params: RequestParams, model_port: int) -> JSONResponse:
    # Stream handling logic (similar to the original @app.post("/inf/stream"))

    rtsp_url = params.stream_params.video_stream_url
    video_stream_manager.add_task(rtsp_url, model_port,
                                  params.stream_params.callback_urls,
                                  params.stream_params.interval)
    restart_stream_process(rtsp_url)
    logger.info(f"start stream detection process! task: {params.model_name}, url: {rtsp_url}")

    return JSONResponse({"status": f"{params.model_name} detection process starts!"})


async def handle_image(params: RequestParams, request: Request, model_port: int):
    # Image handling logic (similar to the original @app.post("/inf/image"))
    req = client.build_request(
        "POST",
        f"http://127.0.0.1:{model_port}/inf",
        content=request.stream(),
        headers=request.headers.raw,
        timeout=300
    )

    try:
        resp = await client.send(req, stream=True)
        await resp.aread()
    except httpx.ConnectError:
        logger.exception(f"Model Unavailable : {params.model_name}")
        return SushineResponse(
            status=503,
            message=f"Model Unavailable : {params.model_name}"
        )
    except Exception:

        logger.exception(f"inference error!")
        return SushineResponse(
            status=500,
            message=traceback.format_exc()
        )

    if resp.content == b"Internal Server Error":
        return SushineResponse(
            status=500,
            message="model inference failed!"
        )

    return resp.json()
