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

    # Pls set previous year to 1999 in every entry (year will be overwritten in app)
    # Pls set current year to 2000 in every entry (year will be overwritten in app)
    # Pls set next year to 2001 in every entry (year will be overwritten in app)

    data_items = [
        DateRange(
            start=pd.to_datetime("2000-11-20"),
            end=pd.to_datetime("2000-12-02"),
            label="Test 1"
        ),
        DateRange(
            start=pd.to_datetime("2000-11-28"),
            end=pd.to_datetime("2000-12-05"),
            label="Test 2"
        ),
        DateRange(
            start=pd.to_datetime("2000-09-20"),
            end=pd.to_datetime("2000-11-12"),
            label="Test 3"
        ),
        DateRange(
            start=pd.to_datetime("2000-12-07"),
            end=pd.to_datetime("2000-12-23"),
            label="Test 4"
        ),
        DateRange(
            start=pd.to_datetime("2000-12-07"),
            end=pd.to_datetime("2000-12-23"),
            label="Test 5"
        ),
    ]

    @staticmethod
    def overwrite_years(data_items):
        current_year = datetime.now().year
        previous_year = current_year - 1
        next_year = current_year + 1

        for item in data_items:
            if item.start.year == 2000:
                item.start = item.start.replace(year=current_year)
            elif item.start.year == 1999:
                item.start = item.start.replace(year=previous_year)
            elif item.start.year == 2001:
                item.start = item.start.replace(year=next_year)

            if item.end.year == 2000:
                item.end = item.end.replace(year=current_year)
            elif item.end.year == 1999:
                item.end = item.end.replace(year=previous_year)
            elif item.end.year == 2001:
                item.end = item.end.replace(year=next_year)

        return data_items
