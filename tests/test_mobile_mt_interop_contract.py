import pytest

from auto_annotation_tool.ranking.mobile_mt_invocations import (
    group_mt_invocations,
)


def attempt(**changes):
    row = {
        "id": "attempt-1",
        "subject_key": "subject-1",
        "session_id": "session-1",

        "mt_status": "NOT_RUN",
        "mt_executed": False,
        "mt_invocation_id": "",
        "mt_detection_count": "",
        "mt_detection_index": "",

        "stale_or_cancelled": False,
        "cancel_reason": "",
        "execution_error": "",
    }

    row.update(changes)
    return row


@pytest.mark.parametrize(
    "row",
    [
        # NOT_RUN = backend MT nie został uruchomiony.
        attempt(
            mt_status="NOT_RUN",
            mt_executed=False,
            mt_invocation_id="inv-1",
        ),
        attempt(
            mt_status="NOT_RUN",
            mt_executed=True,
            mt_invocation_id="inv-1",
        ),
        attempt(
            mt_status="NOT_RUN",
            mt_executed=False,
            mt_detection_count=0,
        ),
        attempt(
            mt_status="NOT_RUN",
            mt_executed=False,
            mt_detection_index=0,
        ),

        # Rzeczywiste wywołanie bez detekcji.
        attempt(
            mt_status="NO_DETECTION",
            mt_executed=True,
            mt_invocation_id="inv-1",
            mt_detection_count="",
        ),

        # Detekcja musi znać indeks i całkowitą liczbę detekcji.
        attempt(
            mt_status="VALID_QUAD",
            mt_executed=True,
            mt_invocation_id="inv-1",
            mt_detection_count=1,
            mt_detection_index="",
        ),
        attempt(
            mt_status="VALID_QUAD",
            mt_executed=True,
            mt_invocation_id="inv-1",
            mt_detection_count="",
            mt_detection_index=0,
        ),

        # Wykonane MT musi mieć identyfikator wywołania.
        attempt(
            mt_status="NO_DETECTION",
            mt_executed=True,
            mt_invocation_id="",
            mt_detection_count=0,
        ),
    ],
)
def test_samples_v2_rejects_contradictory_mt_records(row):
    with pytest.raises(ValueError):
        group_mt_invocations(
            {row["id"]: row},
            "session-1",
            strict_contract=True,
        )


def test_samples_v2_accepts_not_run():
    row = attempt()

    groups = group_mt_invocations(
        {row["id"]: row},
        "session-1",
        strict_contract=True,
    )

    group = groups["attempt-1"]

    assert group.executed is False
    assert group.detection_count == 0
    assert group.legacy_identity is True


def test_samples_v2_accepts_zero_detection_invocation():
    row = attempt(
        mt_status="NO_DETECTION",
        mt_executed=True,
        mt_invocation_id="inv-1",
        mt_detection_count=0,
    )

    groups = group_mt_invocations(
        {row["id"]: row},
        "session-1",
        strict_contract=True,
    )

    group = groups["inv-1"]

    assert group.executed is True
    assert group.detection_count == 0
    assert group.legacy_identity is False


def test_samples_v2_accepts_one_detection():
    row = attempt(
        mt_status="VALID_QUAD",
        mt_executed=True,
        mt_invocation_id="inv-1",
        mt_detection_count=1,
        mt_detection_index=0,
    )

    groups = group_mt_invocations(
        {row["id"]: row},
        "session-1",
        strict_contract=True,
    )

    group = groups["inv-1"]

    assert group.executed is True
    assert group.detection_count == 1
    assert group.legacy_identity is False


def test_samples_v2_accepts_three_detections():
    rows = {}

    for index in range(3):
        row = attempt(
            id=f"attempt-{index}",
            subject_key=f"subject-{index}",
            mt_status="VALID_QUAD",
            mt_executed=True,
            mt_invocation_id="inv-1",
            mt_detection_count=3,
            mt_detection_index=index,
        )
        rows[row["id"]] = row

    groups = group_mt_invocations(
        rows,
        "session-1",
        strict_contract=True,
    )

    group = groups["inv-1"]

    assert group.executed is True
    assert group.detection_count == 3
    assert len(group.records) == 3