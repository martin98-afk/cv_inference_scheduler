
import json
import time
from threading import Thread
from typing import List, Dict

import httpx
from loguru import logger

from application.utils.entities import RequestParams


class VideoStreamManager:
    def __init__(self):
        self.streams = {}  # Key: rtsp_url, Value: StreamContext

    def add_task(self, rtsp_url, model_port, callback_urls, interval):
        if rtsp_url not in self.streams:
            self.streams[rtsp_url] = StreamContext(rtsp_url)
        self.streams[rtsp_url].add_task(model_port, callback_urls, interval)

    def remove_task(self, rtsp_url, model_port):
        if rtsp_url in self.streams:
            self.streams[rtsp_url].remove_task(model_port)
            if self.streams[rtsp_url].is_empty():
                del self.streams[rtsp_url]


class StreamContext:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.tasks = []  # List of tasks

    def add_task(self, model_port, callback_urls, interval):
        self.tasks.append(Task(model_port, callback_urls, interval))

    def remove_task(self, model_port):
        self.tasks = [task for task in self.tasks if task.model_port != model_port]

    def is_empty(self):
        return len(self.tasks) == 0


class Task:
    def __init__(self, model_port: int, callback_urls: list[str], interval: float):
        self.model_port = model_port
        self.callback_urls = callback_urls
        self.interval = interval
        self.thread = None

    def _callback(self, urls: List[str], msg: Dict):
        callback_client = httpx.Client()

        for url in urls:
            try:
                callback_client.post(url, content=msg)
            except httpx.HTTPError:
                logger.error(f"callback to {url} failed!")
            except httpx.ConnectError:
                logger.error(f"{url} is not connectable!")

        callback_client.close()

    def start(self, image: str):
        self.thread = Thread(target=self.task_routine, args=(image,))
        self.thread.start()

    def check_task(self):
        if self.thread is None:
            return False
        return self.thread.is_alive()

    def task_routine(self, image: str):
        params = RequestParams(
            params={"image": image}
        )
        resp = httpx.Client().post(
            url=f"http://127.0.0.1:{self.model_port}/inf",
            content=params.json(),
            timeout=300
        )
        self._callback(self.callback_urls, json.loads(resp.content.decode('utf-8')))
        time.sleep(self.interval)
