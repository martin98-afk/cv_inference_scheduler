"""
@author: mading
@license: (C) Copyright: LUCULENT Corporation Limited.
@contact: mading@luculent.net
@file: model_manager.py
@time: 2023/11/3 11:02
@desc: 
"""
import asyncio
import os
import signal
import socket
from multiprocessing import Process
from typing import Dict, Optional

import httpx
from loguru import logger

from ..utils.config import Config
from ..local_service.model_service import start_service
from ..utils.dataclass import ModelInstance, ModelStatus


class ModelManager:

    def __init__(self):
        self.model_dict: Dict[str, ModelInstance] = {}
        self.current_port = Config.get("http.port")
        self.raw_env = os.environ.copy()

    @staticmethod
    def check_port(port: int):
        s = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            s.bind(("", port))
            return True
        except socket.error:
            logger.error(f"Port {port} is in use, try next")
            return False
        finally:
            if s:
                s.close()

    @property
    def _next_port(self):
        self.current_port += 1
        if not self.check_port(self.current_port): return self._next_port
        return self.current_port

    def start_model(self,
                    model_name: str,
                    model_path: str,
                    model_task: str,
                    device: str,
                    model_config: str,
                    port: Optional[int] = None):
        if port is None:
            port = self._next_port

        process = Process(
            target=start_service,
            name=model_name,
            args=(
                model_name,
                model_path,
                model_task,
                device,
                model_config,
                port
            )
        )
        process.start()
        logger.info(f"Model {model_name} started on port {port}")
        self.model_dict[model_name] = ModelInstance(
            model_name=model_name,
            model_path=model_path,
            model_task=model_task,
            device=device,
            model_config=model_config,
            port=port,
            process=process,
        )

    def start_model_from_config(self):
        root_path = Config.get("root-path")
        model_list = Config.get("model-list")
        for model in model_list.values():
            self.start_model(
                model_name=model["model_name"],
                model_path=os.path.join(root_path, "weights", model["model_path"]),
                model_task=model["model_task"],
                device=model["device"],
                model_config=os.path.join(root_path, "yml", model["model_config"])
            )

        signal.signal(signal.SIGINT, self.stop_all_models)
        signal.signal(signal.SIGILL, self.stop_all_models)
        signal.signal(signal.SIGTERM, self.stop_all_models)

    def stop_model(self, model_name: str):
        if model_name not in self.model_dict:
            return
        os.kill(self.model_dict[model_name].process.pid, signal.SIGKILL)
        logger.info(f"Model {model_name} stopped")
        return self.model_dict.pop(model_name)

    def stop_all_models(self, signum, frame):
        for model_name in list(self.model_dict.keys()):
            self.stop_model(model_name)

    def restart_model(self, model_name: str):
        instance = self.stop_model(model_name)
        self.start_model(
            model_name=model_name,
            model_path=instance.model_path,
            model_task=instance.model_task,
            device=instance.device,
            model_config=instance.model_config,
            port=instance.port
        )

    async def check_model_status(self, model_name: str):
        instance = self.model_dict.get(model_name, None)
        if instance is None:
            return
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"http://127.0.0.1:{instance.port}/health")
                if resp.status_code == 200:
                    data: dict = resp.json()
                    status = data.get("status", False)
                else:
                    status = False
        except Exception:
            status = False

        if status:
            instance.status = ModelStatus.HEALTHY
        elif instance.status == ModelStatus.HEALTHY:
            instance.status = ModelStatus.UNHEALTHY

        return instance.status

    async def check_all_status(self):
        all_healthy = True

        for model_name in list(self.model_dict.keys()):
            status = await self.check_model_status(model_name)
            if status == ModelStatus.HEALTHY:
                continue
            elif status == ModelStatus.UNHEALTHY:
                logger.error(f"Model {model_name} is not healthy, restarting!")
                self.restart_model(model_name)
            elif status == ModelStatus.BOOTING:
                logger.info(f"Model {model_name} is still booting!")
            all_healthy = False

        return all_healthy

    async def wait_for_healthy(self):
        while True:
            all_healthy = await self.check_all_status()

            if all_healthy:
                logger.info("All models are healthy!")
                break

            logger.info("Waiting for all models to be healthy!")
            await asyncio.sleep(5)