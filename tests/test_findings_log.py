"""Unit tests for FindingsLog / SHARPOfflineResilienceManager across transport adapters."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_python"))
from sharp_resilience import FindingsLog


def _sample_finding(idx: int = 1):
    return {
        "id": idx,
        "bbox": [10, 20, 30, 40],
        "score": 0.14,
        "type": "LARGE ANOMALY REGION",
        "confidence": 0.85,
    }


def test_failed_transport_preserves_unsynced_records(tmp_path):
    db_path = str(tmp_path / "findings.db")
    attempts = []

    def failing_transport(records):
        attempts.append(records)
        return False

    with FindingsLog(db_path=db_path, transport=failing_transport) as log:
        log.log_anomaly_offline(_sample_finding(1))
        log.log_anomaly_offline(_sample_finding(2))
        assert log.get_pending_sync_count() == 2

        synced_count = log.sync_pending()
        assert synced_count == 0
        assert len(attempts) == 1
        assert log.get_pending_sync_count() == 2
        assert all(not r["synced"] for r in log.get_recent_logs())


def test_successful_transport_marks_only_delivered_batch_as_synced(tmp_path):
    db_path = str(tmp_path / "findings.db")
    delivered_batches = []

    def ok_transport(records):
        delivered_batches.append(records)
        return True

    with FindingsLog(db_path=db_path, transport=ok_transport) as log:
        log.log_anomaly_offline(_sample_finding(1))
        log.log_anomaly_offline(_sample_finding(2))
        log.log_anomaly_offline(_sample_finding(3))

        # Sync first 2 records only
        synced = log.sync_pending(batch_limit=2)
        assert synced == 2
        assert len(delivered_batches) == 1
        assert [r["roi_id"] for r in delivered_batches[0]] == [1, 2]
        assert log.get_pending_sync_count() == 1

        # Sync remaining record
        synced_rest = log.sync_pending(batch_limit=2)
        assert synced_rest == 1
        assert log.get_pending_sync_count() == 0
        assert all(r["synced"] for r in log.get_recent_logs())


def test_transport_raising_exception_does_not_mark_synced(tmp_path):
    db_path = str(tmp_path / "findings.db")

    def raising_transport(_records):
        raise ConnectionError("network unreachable")

    with FindingsLog(db_path=db_path, transport=raising_transport) as log:
        log.log_anomaly_offline(_sample_finding(1))
        assert log.sync_pending() == 0
        assert log.get_pending_sync_count() == 1


def test_pathology_data_persisted_and_synced(tmp_path):
    db_path = str(tmp_path / "findings.db")
    delivered_batches = []

    def transport(records):
        delivered_batches.append(records)
        return True

    finding_with_pathology = {
        "id": 42,
        "bbox": [15, 25, 64, 64],
        "score": 0.22,
        "type": "LARGE ANOMALY REGION",
        "confidence": 0.91,
        "pathology": {
            "label": "CNV",
            "confidence": 0.94,
            "probabilities": {
                "CNV": 0.94,
                "DME": 0.02,
                "DRUSEN": 0.03,
                "NORMAL": 0.01,
            },
        },
    }

    with FindingsLog(db_path=db_path, transport=transport) as log:
        log.log_anomaly_offline(finding_with_pathology)

        # Verify offline retrieval
        recent = log.get_recent_logs()
        assert len(recent) == 1
        rec = recent[0]
        assert "pathology" in rec
        assert rec["pathology"]["label"] == "CNV"
        assert rec["pathology"]["confidence"] == 0.94
        assert rec["pathology"]["probabilities"]["CNV"] == 0.94

        # Verify transport sync receives pathology
        synced = log.sync_pending()
        assert synced == 1
        assert len(delivered_batches) == 1
        synced_rec = delivered_batches[0][0]
        assert synced_rec["roi_id"] == 42
        assert synced_rec["pathology"]["label"] == "CNV"
        assert synced_rec["pathology"]["confidence"] == 0.94

