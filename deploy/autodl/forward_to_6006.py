#!/usr/bin/env python3
"""把 AutoDL 自定义服务端口（默认 6006）转发到 openpi websocket 服务端口（默认 8000）。

背景：AutoDL 实例没有独立公网 IP，只有 6006 / 6008 两个端口会被映射到公网地址
（形如 https://uXXXX-xxx.<region>.seetacloud.com:8443，可在环境变量 AutoDLService6006URL 中看到）。
所以链路是：

    外部客户端 --wss://uXXX...:8443--> AutoDL 反向代理 --TCP--> 实例内 6006 --TCP--> openpi 8000

AutoDL 镜像里通常没有 socat / nginx，这里用纯 Python 做一个轻量 TCP 转发。

用法：
    python deploy/autodl/forward_to_6006.py                     # 6006 -> 8000
    python deploy/autodl/forward_to_6006.py --target-port 8000  # 指定后端端口
    nohup python deploy/autodl/forward_to_6006.py >> logs/forward_6006.log 2>&1 &
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal

logger = logging.getLogger("forward_to_6006")


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """把 reader 的数据搬到 writer，直到任一端关闭。"""
    try:
        while data := await reader.read(64 * 1024):
            writer.write(data)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.close()


async def _handle(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target_host: str,
    target_port: int,
) -> None:
    peer = client_writer.get_extra_info("peername")
    try:
        target_reader, target_writer = await asyncio.open_connection(target_host, target_port)
    except OSError as exc:
        logger.error("无法连接后端 %s:%s（%s）；请确认 serve_policy 已启动", target_host, target_port, exc)
        client_writer.close()
        return

    logger.info("新连接 %s -> %s:%s", peer, target_host, target_port)
    try:
        await asyncio.gather(
            _pipe(client_reader, target_writer),
            _pipe(target_reader, client_writer),
        )
    finally:
        with contextlib.suppress(Exception):
            client_writer.close()
        logger.info("连接结束 %s", peer)


async def _serve(listen_host: str, listen_port: int, target_host: str, target_port: int) -> None:
    server = await asyncio.start_server(
        lambda r, w: _handle(r, w, target_host, target_port), listen_host, listen_port
    )
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    logger.info("TCP 转发已启动：%s -> %s:%s", addrs, target_host, target_port)
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--listen-host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--listen-port", type=int, default=6006, help="监听端口，AutoDL 需为 6006 或 6008（默认 6006）")
    parser.add_argument("--target-host", default="127.0.0.1", help="openpi 服务地址（默认 127.0.0.1）")
    parser.add_argument("--target-port", type=int, default=8000, help="openpi 服务端口（默认 8000）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, loop.stop)

    with contextlib.suppress(asyncio.CancelledError, KeyboardInterrupt):
        loop.run_until_complete(_serve(args.listen_host, args.listen_port, args.target_host, args.target_port))
    logger.info("转发服务已退出")


if __name__ == "__main__":
    main()
