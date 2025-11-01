"""
Adapted from https://github.com/Physical-Intelligence/openpi
"""

import logging
import time
from typing import (
    Dict,
    Tuple,
)

import websockets.sync.client
from typing_extensions import override

from policy_server_client import base_policy as _base_policy
from policy_server_client import msgpack_numpy


class WebsocketPolicyClient(_base_policy.BasePolicy):
    """Websocket client for the policy server."""

    def __init__(self, host: str, port: int):
        """Initialize the WebsocketPolicyClient.

        Args:
            host (str): The hostname of the policy server.
            port (int): The port number of the policy server.
        """
        self.host = host
        self.port = port
        self.logger = logging.getLogger("websockets.client")
        self._uri = f"ws://{self.host}:{self.port}"
        self._ws, self._server_metadata = self._wait_for_server()
        self._packer = msgpack_numpy.Packer()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        self.logger.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                conn = websockets.sync.client.connect(self._uri,
                                                      compression=None,
                                                      max_size=None,
                                                      ping_timeout=None,)
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                self.logger.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        """Reset the policy and associated env adapter to its initial state."""
        self._ws.send(self._packer.pack({"reset": True}))
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def switch_model(self, new_model_path) -> None:
        """Switch the model to a new checkpoint step."""
        self._ws.send(self._packer.pack({"new_model_path": new_model_path}))
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)
