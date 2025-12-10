from dataclasses import dataclass
from datetime import datetime

import pandas as pd


class LocalData:
    @dataclass
    class DateRange:
        start: datetime
        end: datetime
        label: str

    # YEAR-MOTH-DAY: pls 2025.02.01 not for e.g. 2025.2.1

    data_items = [
        DateRange(
            start=pd.to_datetime("2025-11-20"),
            end=pd.to_datetime("2025-12-02"),
            label="Test 1"
        ),
        DateRange(
            start=pd.to_datetime("2025-11-28"),
            end=pd.to_datetime("2025-12-05"),
            label="Test 2"
        ),
        DateRange(
            start=pd.to_datetime("2025-09-20"),
            end=pd.to_datetime("2025-11-12"),
            label="Test 3"
        ),
        DateRange(
            start=pd.to_datetime("2025-12-07"),
            end=pd.to_datetime("2025-12-23"),
            label="Test 4"
        ),
        DateRange(
            start=pd.to_datetime("2024-12-07"),
            end=pd.to_datetime("2024-12-23"),
            label="Test 5"
        ),
    ]
