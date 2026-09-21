"""UDP telemetry receiver for the FlightGear "generic" output protocol.

FlightGear is the UDP *client* here (--generic=socket,out,...) -- it pushes
one line per its own update tick to our bound port. A background thread
drains the socket continuously and keeps only the MOST RECENT decoded
sample, so the control loop never operates on a stale backlog if it is ever
briefly slower than FlightGear's output rate.
"""
from __future__ import annotations

import math
import socket
import threading
import time
from dataclasses import dataclass

from controller.protocol import decode_telemetry_line


@dataclass
class Telemetry:
    fields: dict
    received_wall_time: float  # time.monotonic() when this sample was received

    def get(self, name, default=None):
        return self.fields.get(name, default)

    def is_finite(self) -> bool:
        return all(math.isfinite(v) for v in self.fields.values() if isinstance(v, float))


class TelemetryReceiver:
    def __init__(self, port: int, host: str = "127.0.0.1"):
        self.port = port
        self.host = host
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Telemetry | None = None
        self.n_received = 0
        self.n_decode_errors = 0

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.2)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        buf = ""
        while not self._stop.is_set():
            try:
                data, _addr = self._sock.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                break
            buf += data.decode("ascii", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                fields = decode_telemetry_line(line)
                if fields is None:
                    self.n_decode_errors += 1
                    continue
                self.n_received += 1
                with self._lock:
                    self._latest = Telemetry(fields=fields, received_wall_time=time.monotonic())

    def latest(self) -> Telemetry | None:
        with self._lock:
            return self._latest

    def stop(self):
        self._stop.set()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=2.0)
