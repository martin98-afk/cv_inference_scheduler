"""
@author: mading
@license: (C) Copyright: LUCULENT Corporation Limited.
@contact: mading@luculent.net
@file: start_object_detection_server.py
@time: 2023/11/3 11:18
@desc: 
"""
import asyncio
import signal

import uvicorn

from application import app, manager
from application.utils.config import Config


async def main():

    manager.start_model_from_config()

    await manager.wait_for_healthy()

    is_running = True
    async def _health_checker():
        while is_running:
            await manager.check_all_status()
            await asyncio.sleep(5)

    def _stop(_, __):
        global is_running
        is_running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    uvicorn_config = uvicorn.Config(
        app=app,
        host=Config.get("http.host"),
        port=Config.get("http.port"),
        log_level="info",
        access_log=False,
        timeout_keep_alive=Config.get("http.timeout-keep-alive"),
    )
    uvicorn_server = uvicorn.Server(config=uvicorn_config)
    uvicorn_task = asyncio.create_task(uvicorn_server.serve())
    checker_task = asyncio.create_task(_health_checker())
    done, pending = await asyncio.wait(
        [uvicorn_task, checker_task], return_when=asyncio.tasks.FIRST_COMPLETED
    )
    for task in done:
        task.result()
    for task in pending:
        task.cancel()


if __name__ == "__main__":
    asyncio.run(main())