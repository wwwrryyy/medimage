"""
sharp_resilience.py
===================
SHARP Paradigm - Edge Fault Tolerance & Offline Findings Log.

Guarantees 100% local edge operation during network outages, degraded connectivity,
or hardware faults. Stores diagnostic findings locally in SQLite and synchronizes
with the clinical records server across a pluggable FindingTransport seam, marking
records as synced only after transport delivery is confirmed.
"""

from contextlib import contextmanager
import json
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Sequence, Union
import urllib.request


FindingTransport = Union[
    Callable[[List[Dict[str, Any]]], bool],
    "FindingTransportProtocol",
]


class FindingTransportProtocol(Protocol):
    """Seam for delivering a batch of unsynchronized findings to a remote sink."""

    def __call__(self, records: List[Dict[str, Any]]) -> bool:
        ...


class HttpJsonFindingTransport:
    """
    HTTP POST JSON adapter for FindingTransport.
    Sends {"findings": [...]} to the configured clinical records endpoint and returns
    True only on a 2xx HTTP response.
    """

    def __init__(self, endpoint_url: str, timeout_sec: float = 5.0):
        self.endpoint_url = endpoint_url
        self.timeout_sec = float(timeout_sec)

    def __call__(self, records: List[Dict[str, Any]]) -> bool:
        payload = json.dumps({"findings": records}).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False


class SHARPOfflineResilienceManager:
    """
    Local SQLite findings log and optional background synchronization manager.

    Records are marked synced=1 strictly by ID and only after the configured
    transport confirms delivery.
    """

    def __init__(
        self,
        db_path: str = "scan_telemetry.db",
        transport: Optional[FindingTransport] = None,
        sync_interval_sec: float = 5.0,
        start_monitor: bool = False,
    ):
        self.db_path = db_path
        self.transport = transport
        self.sync_interval_sec = float(sync_interval_sec)
        self.is_connected = transport is not None
        self.auto_sync_running = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.monitor_thread: Optional[threading.Thread] = None

        self._init_db()
        if start_monitor:
            self._start_network_monitor()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initializes local SQLite database schema for offline log persistence."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS anomaly_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    roi_id INTEGER,
                    roi_type TEXT,
                    confidence REAL,
                    mse_score REAL,
                    bbox_json TEXT,
                    synced INTEGER DEFAULT 0
                )
                """
            )

    def log_anomaly_offline(self, roi_dict: Dict[str, Any]) -> int:
        """
        Logs an anomaly ROI finding locally into the offline buffer.
        Returns the inserted row ID.
        """
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO anomaly_logs (timestamp, roi_id, roi_type, confidence, mse_score, bbox_json, synced)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    time.time(),
                    int(roi_dict["id"]),
                    str(roi_dict["type"]),
                    float(roi_dict["confidence"]),
                    float(roi_dict["score"]),
                    json.dumps(list(roi_dict["bbox"])),
                ),
            )
            return int(cursor.lastrowid or 0)

    def get_pending_sync_count(self) -> int:
        """Returns number of un-synchronized offline telemetry records."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM anomaly_logs WHERE synced = 0").fetchone()
            return int(row[0]) if row else 0

    def get_recent_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetches most recent logged findings."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, roi_id, roi_type, confidence, mse_score, bbox_json, synced
                FROM anomaly_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

        logs: List[Dict[str, Any]] = []
        for r in rows:
            logs.append(
                {
                    "log_id": r[0],
                    "timestamp": time.strftime("%H:%M:%S", time.localtime(r[1])),
                    "roi_id": r[2],
                    "type": r[3],
                    "confidence": r[4],
                    "mse_score": r[5],
                    "bbox": json.loads(r[6]),
                    "synced": bool(r[7]),
                }
            )
        return logs

    def sync_pending(self, batch_limit: int = 100) -> int:
        """
        Attempts to synchronize unsynced findings via the configured transport.
        Marks records as synced=1 ONLY if delivery succeeds.
        Returns the number of records synchronized in this call.
        """
        if not self.is_connected:
            return 0

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT id, timestamp, roi_id, roi_type, confidence, mse_score, bbox_json
                    FROM anomaly_logs
                    WHERE synced = 0
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (int(batch_limit),),
                ).fetchall()

            if not rows:
                return 0

            records = [
                {
                    "log_id": r[0],
                    "timestamp": r[1],
                    "roi_id": r[2],
                    "type": r[3],
                    "confidence": r[4],
                    "mse_score": r[5],
                    "bbox": json.loads(r[6]),
                }
                for r in rows
            ]

            if self.transport is not None:
                try:
                    delivered = bool(self.transport(records))
                except Exception:
                    delivered = False
            else:
                # Simulated connection mode (via toggle_simulated_connection without transport)
                delivered = True

            if not delivered:
                return 0

            ids = [r[0] for r in rows]
            placeholders = ",".join("?" for _ in ids)
            with self._connect() as conn:
                conn.execute(
                    f"UPDATE anomaly_logs SET synced = 1 WHERE id IN ({placeholders})",
                    ids,
                )
            return len(ids)

    def _start_network_monitor(self) -> None:
        """Launches a background heartbeat thread checking for pending syncs."""
        if self.auto_sync_running:
            return
        self._stop_event.clear()
        self.auto_sync_running = True
        self.monitor_thread = threading.Thread(target=self._network_loop, daemon=True)
        self.monitor_thread.start()

    def _network_loop(self) -> None:
        while self.auto_sync_running and not self._stop_event.wait(self.sync_interval_sec):
            if self.is_connected and self.get_pending_sync_count() > 0:
                self.sync_pending()

    def _synchronize_logs(self) -> int:
        """Backward-compatible alias for sync_pending()."""
        return self.sync_pending()

    def toggle_simulated_connection(self) -> bool:
        self.is_connected = not self.is_connected
        return self.is_connected

    def close(self) -> None:
        """Cleanly stops the background monitor thread if running."""
        self.auto_sync_running = False
        self._stop_event.set()
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        self.monitor_thread = None

    def __enter__(self) -> "SHARPOfflineResilienceManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# Domain-aligned alias matching GLOSSARY.md ("Finding")
FindingsLog = SHARPOfflineResilienceManager
