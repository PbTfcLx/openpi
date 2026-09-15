import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        api_key: str | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        # Optional shared-secret authentication. When set, clients must send an
        # `Authorization: Api-Key <key>` header (see openpi_client.WebsocketClientPolicy).
        # This is strongly recommended when the server is reachable from the internet.
        self._api_key = api_key
        # Serialize policy.infer() calls. The policy keeps a JAX PRNG key that is
        # mutated on every call, so concurrent inference is not thread-safe.
        # Multiple connections may still be served concurrently at the I/O level;
        # only the actual model forward pass is serialized.
        self._infer_lock = asyncio.Lock()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._process_request,
        ) as server:
            await server.serve_forever()

    def _process_request(
        self, connection: _server.ServerConnection, request: _server.Request
    ) -> _server.Response | None:
        """Answer plain HTTP requests (health check / auth) before the websocket upgrade."""
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        if self._api_key is not None:
            # Mirrors the header sent by openpi_client.WebsocketClientPolicy(api_key=...).
            if request.headers.get("Authorization") != f"Api-Key {self._api_key}":
                logger.warning("Rejected connection from %s: missing or invalid API key", connection.remote_address)
                return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Unauthorized\n")
        # Continue with the normal request handling.
        return None

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                # Run inference off the event loop so that keepalive pings and
                # other connections are not starved while the (synchronous)
                # policy computes. The lock keeps the policy's internal RNG safe
                # under concurrent requests.
                async with self._infer_lock:
                    action = await asyncio.to_thread(self._policy.infer, obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise
