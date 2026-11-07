from datetime import date

import pandas as pd
import pytest
from sqlalchemy import func, select

from insider_tracker.db.database import get_engine, get_session
from insider_tracker.db.models import Company, Report, Trade
from insider_tracker.db.queries import read_reports, read_trades
from insider_tracker.pipeline.loader import (
    find_fetch_start,
    load_report_rows,
    load_trades,
    trade_fingerprints,
)


def fake_trade(**changes) -> dict:
    return {
        "published_at": pd.Timestamp("2026-10-21 12:00:00"),
        "company": "Testbolaget AB",
        "lei": "LEI1",
        "person_name": "Fejk Fejksson",
        "position": "Styrelseledamot",
        "is_related": False,
        "is_correction": False,
        "is_share_program": False,
        "transaction_type": "Förvärv",
        "instrument_type": "Aktie",
        "instrument_name": "Testbolaget B",
        "isin": "SE0000000001",
        "transaction_date": pd.Timestamp("2026-10-20"),
        "volume": 100.0,
        "volume_unit": "Antal",
        "price": 23.5,
        "currency": "SEK",
        "venue": "NASDAQ STOCKHOLM AB",
        "source_url": "https://example.test/export",
        **changes,
    }


def fake_reports(*report_dates: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "company_id": "9",
            "orderbook_id": "111",
            "name": "Testbolaget B",
            "isin": "SE0000000001",
            "ticker": "TEST B",
            "market_list": "Large Cap Stockholm",
            "report_date": pd.to_datetime(list(report_dates)),
            "report_type": "INTERIM_REPORT",
            "is_confirmed": "True",
            "source_url": "https://example.test/report",
        }
    )


@pytest.fixture
def engine():
    return get_engine("sqlite://")


def count(engine, model) -> int:
    with get_session(engine) as session:
        return session.scalar(select(func.count()).select_from(model))


def test_loading_the_same_trades_twice_adds_nothing(engine):
    trades = pd.DataFrame([fake_trade(), fake_trade(person_name="Fejk Annan")])

    with get_session(engine) as session:
        assert load_trades(session, trades) == 2
    with get_session(engine) as session:
        assert load_trades(session, trades) == 0

    assert count(engine, Trade) == 2


def test_identical_rows_in_one_export_are_both_kept(engine):
    trades = pd.DataFrame([fake_trade(), fake_trade()])

    with get_session(engine) as session:
        assert load_trades(session, trades) == 2
    with get_session(engine) as session:
        assert load_trades(session, trades) == 0

    assert count(engine, Trade) == 2


def test_fingerprint_changes_when_trade_changes():
    trades = pd.DataFrame([fake_trade(), fake_trade(volume=200.0)])

    first, second = trade_fingerprints(trades)

    assert first != second


def test_company_is_created_once_per_lei(engine):
    trades = pd.DataFrame([fake_trade(), fake_trade(person_name="Fejk Annan", company="Nytt namn")])

    with get_session(engine) as session:
        load_trades(session, trades)

    assert count(engine, Company) == 1


def test_company_without_lei_is_found_by_name(engine):
    trades = pd.DataFrame([fake_trade(lei=None, company="Utan LEI AB")])

    with get_session(engine) as session:
        load_trades(session, trades)
    with get_session(engine) as session:
        company = session.scalar(select(Company))

        assert company.key == "Utan LEI AB"
        assert company.lei is None


def test_trades_come_back_from_database_with_same_values_and_empty_price(engine):
    trades = pd.DataFrame([fake_trade(price=None)])

    with get_session(engine) as session:
        load_trades(session, trades)
    with get_session(engine) as session:
        stored = read_trades(session)

    assert stored.loc[0, "company"] == "Testbolaget AB"
    assert stored.loc[0, "transaction_date"] == pd.Timestamp("2026-10-20")
    assert stored.loc[0, "source_url"] == "https://example.test/export"
    assert pd.isna(stored.loc[0, "price"])
    assert not stored.loc[0, "is_related"]


def test_reports_are_only_saved_once(engine):
    reports = fake_reports("2026-10-16", "2027-01-22")

    with get_session(engine) as session:
        assert load_report_rows(session, reports) == 2
    with get_session(engine) as session:
        assert load_report_rows(session, reports) == 0

    assert count(engine, Report) == 2


def test_reports_from_own_list_without_orderbook_or_ticker_can_be_saved(engine):
    own_list = fake_reports("2026-10-16").drop(columns=["orderbook_id", "ticker"])

    with get_session(engine) as session:
        assert load_report_rows(session, own_list) == 1
    with get_session(engine) as session:
        stored = read_reports(session)

    assert stored.loc[0, "isin"] == "SE0000000001"
    assert stored.loc[0, "orderbook_id"] == ""


def test_own_list_reuses_the_id_of_a_known_company(engine):
    own_list = fake_reports("2027-01-28").assign(company_id="eget-id")

    with get_session(engine) as session:
        load_report_rows(session, fake_reports("2026-10-16"))
    with get_session(engine) as session:
        assert load_report_rows(session, own_list) == 1
    with get_session(engine) as session:
        stored = read_reports(session)

    assert set(stored["company_id"]) == {"9"}


def test_same_report_in_own_list_is_not_saved_twice(engine):
    with get_session(engine) as session:
        load_report_rows(session, fake_reports("2026-10-16"))
    with get_session(engine) as session:
        assert load_report_rows(session, fake_reports("2026-10-16").assign(company_id="eget")) == 0


def test_reports_come_back_with_company_id_and_dates(engine):
    with get_session(engine) as session:
        load_report_rows(session, fake_reports("2026-10-16"))
    with get_session(engine) as session:
        stored = read_reports(session)

    assert stored.loc[0, "company_id"] == "9"
    assert stored.loc[0, "report_date"] == pd.Timestamp("2026-10-16")
    assert stored.loc[0, "is_confirmed"]


def test_first_fetch_starts_before_the_first_report_window(engine):
    with get_session(engine) as session:
        start = find_fetch_start(session, fake_reports("2026-10-16"))

    assert start == date(2026, 9, 2)


def test_later_fetch_continues_from_latest_published_trade(engine):
    with get_session(engine) as session:
        load_trades(session, pd.DataFrame([fake_trade()]))
    with get_session(engine) as session:
        start = find_fetch_start(session, fake_reports("2026-01-01"))

    assert start == date(2026, 10, 14)
