import pandas as pd
from src.config import GROUP_COLUMN
from src.train import group_aware_split


def test_group_split_has_no_segment_overlap() -> None:
    rows = []
    for segment_id in range(50):
        for index in range(10):
            rows.append({"segment_id": segment_id, "target": int(index < 3), "pressure": 70.0 + index, "flow_rate": 4.0, "temperature": 30.0, "pump_speed": 1000.0, "energy_consumption": 25.0, "valve_status": 1, "pump_state": 1, "compressor_state": 1})
    train, test = group_aware_split(pd.DataFrame(rows))
    train_ids, test_ids = set(train[GROUP_COLUMN]), set(test[GROUP_COLUMN])
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) == 40
    assert len(test_ids) == 10
