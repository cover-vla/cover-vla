"""
Adapted from https://github.com/Physical-Intelligence/openpi
"""

import asyncio
import logging
import traceback

import websockets.asyncio.server
import websockets.frames

from policy_server_client import msgpack_numpy
from src.utils.monitor import setup_logger


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self.logger = setup_logger(
            main_rank=True,
            filename=None,
            name="policy_server",
        )
        self.logger.setLevel(logging.INFO)

    def serve_forever(self) -> None:
        """Starts the server and runs it forever. This is a blocking call."""
        self.logger.info(f"Starting server on {self._host}:{self._port}")
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        self.logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())

                # Check if the client is requesting a new model checkpoint
                new_model_path = obs.get("new_model_path", None)
                if new_model_path is not None:
                    self._policy.switch_model(new_model_path)
                    self.logger.info(f"Loaded new model checkpoint: {new_model_path}")
                    await websocket.send(packer.pack({"status": "model switched"}))
                    continue # no actual observation will be sent with this, so we skip the rest of the loop

                # Check if the client is requesting a reset
                if obs.get("reset", False):
                    self._policy.reset()
                    await websocket.send(packer.pack({"status": "reset"}))
                    continue

                action = self._policy.select_action(obs)

                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                self.logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise
