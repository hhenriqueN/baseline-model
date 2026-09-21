"""UDP command sender for the FlightGear "generic" input protocol.

FlightGear is the UDP *server* here (--generic=socket,in,...) -- it listens
on its own bound port; we send it one line per control cycle.
"""
from __future__ import annotations

import socket

from controller.protocol import encode_command_line


class CommandSender:
    def __init__(self, port: int, host: str = "127.0.0.1"):
        self.port = port
        self.host = host
        self._sock: socket.socket | None = None
        self.n_sent = 0

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, values: dict):
        line = encode_command_line(values)
        self._sock.sendto(line, (self.host, self.port))
        self.n_sent += 1

    def stop(self):
        if self._sock:
            self._sock.close()
