from datetime import UTC, datetime

import pandas as pd
import pytest

from insider_tracker.db.database import get_engine, get_session
from insider_tracker.db.models import Company, Summary
from insider_tracker.ui.dashboard_data import (
    BUY,
    SALE,
    build_day_chart,
    count_by_day,
    latest_summary,
    prepare_trades,
    table_rows,
    window_stats,
)


def fake_matched(**changes) -> dict:
    return {
        "transaction_date": pd.Timestamp("2026-10-20"),
        "company": "Testbolaget AB",
        "person_name": "Fejk Fejksson",
        "position": "Verkställande direktör (VD)",
        "window": "after",
        "days_from_report": 4,
        "is_related": False,
        "transaction_type": BUY,
        "volume": 100.0,
        "price": 23.5,
        "currency": "SEK",
        "amount_sek": 2350.0,
        "uncertain_reason": "",
        "source_url": "https://example.test/export",
        **changes,
    }


def prepared(*rows: dict) -> pd.DataFrame:
    return prepare_trades(pd.DataFrame(rows))


def test_person_names_never_reach_the_dashboard():
    trades = prepared(fake_matched())

    assert "person_name" not in trades.columns
    assert "Fejk Fejksson" not in trades.to_string()
    assert "Fejk Fejksson" not in table_rows(trades).to_string()


def test_labels_are_in_swedish():
    trades = prepared(
        fake_matched(),
        fake_matched(position="Styrelseledamot", window="before"),
        fake_matched(position="Ekonomichef/finanschef/finansdirektör"),
        fake_matched(position="Annan ledande befattningshavare"),
    )

    assert trades["position_label"].tolist() == ["Vd", "Styrelse", "Ekonomichef", "Övriga"]
    assert trades["window_label"].tolist()[:2] == ["Efter rapporten", "Före stoppet"]


def test_stats_divide_buys_by_the_length_of_each_window():
    trades = prepared(
        fake_matched(),
        fake_matched(),
        fake_matched(window="before", days_from_report=-40),
        fake_matched(transaction_type=SALE),
    )

    stats = window_stats(trades)

    assert stats["after"] == pytest.approx(2 / 15)
    assert stats["before"] == pytest.approx(1 / 14)


def test_stats_are_zero_without_trades():
    assert window_stats(prepared(fake_matched()).iloc[0:0]) == {"before": 0, "after": 0}


def test_count_by_day_separates_buys_and_sales():
    trades = prepared(fake_matched(), fake_matched(), fake_matched(transaction_type=SALE))

    counts = count_by_day(trades).set_index("transaction_type")

    assert counts.loc[BUY, "trades"] == 2
    assert counts.loc[SALE, "trades"] == 1


def test_table_has_swedish_headers_and_latest_first():
    trades = prepared(
        fake_matched(),
        fake_matched(transaction_date=pd.Timestamp("2026-10-25")),
    )

    table = table_rows(trades)

    assert table.iloc[0]["Datum"] == pd.Timestamp("2026-10-25")
    assert "Källa" in table.columns


def test_chart_can_be_built_with_data_and_with_only_one_type():
    both = prepared(fake_matched(), fake_matched(transaction_type=SALE))
    only_buys = prepared(fake_matched())

    assert "vconcat" in build_day_chart(both).to_dict()
    assert "vconcat" in build_day_chart(only_buys).to_dict()


def test_latest_summary_returns_the_newest_for_the_right_company():
    engine = get_engine("sqlite://")
    with get_session(engine) as session:
        first = Company(key="LEI1", name="Testbolaget AB", lei="LEI1")
        other = Company(key="LEI2", name="Annat AB", lei="LEI2")
        session.add_all([first, other])
        session.flush()
        session.add_all(
            [
                Summary(
                    company_id=first.id,
                    summary_text="Gammal",
                    model_used="m",
                    trades_used=1,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
                Summary(
                    company_id=first.id,
                    summary_text="Ny",
                    model_used="m",
                    trades_used=1,
                    created_at=datetime(2026, 2, 1, tzinfo=UTC),
                ),
                Summary(
                    company_id=other.id,
                    summary_text="Annat",
                    model_used="m",
                    trades_used=1,
                    created_at=datetime(2026, 3, 1, tzinfo=UTC),
                ),
            ]
        )

    with get_session(engine) as session:
        assert latest_summary(session, "Testbolaget AB").summary_text == "Ny"
        assert latest_summary(session, "Saknas AB") is None
