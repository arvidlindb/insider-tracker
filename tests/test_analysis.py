import pandas as pd
import pytest

from insider_tracker.pipeline.analysis import coverage, position_group, summarize_windows


def fake_trade(**changes) -> dict:
    # Ett köp på 3 000 000 SEK dagen efter rapporten.
    return {
        "published_at": pd.Timestamp("2026-10-21 12:00:00"),
        "company": "Testbolaget AB",
        "lei": "LEI1",
        "person_name": "Fejk Fejksson",
        "position": "Verkställande direktör (VD)",
        "is_related": False,
        "is_correction": False,
        "is_share_program": False,
        "transaction_type": "Förvärv",
        "instrument_type": "Aktie",
        "instrument_name": "Testbolaget B",
        "isin": "SE0000000001",
        "transaction_date": pd.Timestamp("2026-10-17"),
        "volume": 30_000.0,
        "volume_unit": "Antal",
        "price": 100.0,
        "currency": "SEK",
        "venue": "NASDAQ STOCKHOLM AB",
        "source_url": "https://example.test/export",
        **changes,
    }


def fake_reports() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "company_id": ["9"],
            "isin": ["SE0000000001"],
            "market_list": ["Large Cap Stockholm"],
            "report_date": pd.to_datetime(["2026-10-16"]),
            "report_type": ["INTERIM_REPORT"],
        }
    )


@pytest.mark.parametrize(
    ("position", "group"),
    [
        ("Verkställande direktör (VD)", "ceo"),
        ("Vice verkställande direktör", "other"),
        ("Ekonomichef/finanschef/finansdirektör", "cfo"),
        ("Styrelseledamot", "board"),
        ("Styrelseordförande", "board"),
        ("Arbetstagarrepresentant i styrelsen eller arbetstagarsuppleant", "other"),
        ("Annan ledande befattningshavare", "other"),
        (None, "other"),
    ],
)
def test_position_group(position, group):
    assert position_group(position) == group


def test_summary_counts_trades_per_window_and_position():
    trades = pd.DataFrame(
        [
            fake_trade(),
            fake_trade(
                person_name="Fejk Styrelse",
                position="Styrelseledamot",
                transaction_type="Avyttring",
                transaction_date=pd.Timestamp("2026-09-10"),
            ),
        ]
    )

    summary = summarize_windows(trades, fake_reports()).set_index("window")

    assert summary.loc["after", "position_group"] == "ceo"
    assert summary.loc["after", "trades"] == 1
    assert summary.loc["after", "amount_sek"] == 3_000_000.0
    assert summary.loc["after", "trades_per_window_day"] == pytest.approx(1 / 15)
    assert summary.loc["before", "position_group"] == "board"
    assert summary.loc["before", "trades_per_window_day"] == pytest.approx(1 / 14)


def test_summary_leaves_out_trades_during_the_trading_stop_and_filtered_ones():
    trades = pd.DataFrame(
        [
            fake_trade(transaction_date=pd.Timestamp("2026-10-01")),
            fake_trade(person_name="Fejk A", transaction_type="Tilldelning"),
        ]
    )

    assert summarize_windows(trades, fake_reports()).empty


def test_summary_counts_sale_without_known_holding_as_uncertain():
    trades = pd.DataFrame([fake_trade(transaction_type="Avyttring")])

    summary = summarize_windows(trades, fake_reports())

    assert summary.loc[0, "uncertain_trades"] == 1


def test_coverage_is_the_share_of_trades_with_a_known_company():
    trades = pd.DataFrame([fake_trade(), fake_trade(isin="SE0000000002", lei="LEI2")])

    assert coverage(trades, fake_reports()) == 0.5
