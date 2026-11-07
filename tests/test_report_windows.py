import pandas as pd

from insider_tracker.pipeline.report_windows import assign_company_ids, match_trades_to_windows


def fake_reports(*report_dates: str, company_id: str = "9", isin: str = "SE0000000001"):
    return pd.DataFrame(
        {
            "company_id": company_id,
            "isin": isin,
            "report_date": pd.to_datetime(list(report_dates)),
            "report_type": "INTERIM_REPORT",
        }
    )


def fake_trades(*transaction_dates: str, isin: str = "SE0000000001", lei: str = "LEI1"):
    return pd.DataFrame(
        {
            "isin": isin,
            "lei": lei,
            "transaction_date": pd.to_datetime(list(transaction_dates)),
        }
    )


def test_company_is_found_by_isin():
    trades = assign_company_ids(fake_trades("2026-10-20"), fake_reports("2026-10-16"))

    assert trades.loc[0, "company_id"] == "9"


def test_lei_connects_other_instruments_to_the_same_company():
    trades = pd.concat(
        [fake_trades("2026-10-20"), fake_trades("2026-10-20", isin="SE0000000002")],
        ignore_index=True,
    )

    identified = assign_company_ids(trades, fake_reports("2026-10-16"))

    assert identified["company_id"].tolist() == ["9", "9"]


def test_trade_from_unknown_company_stays_empty():
    trades = fake_trades("2026-10-20", isin="SE0000000003", lei="LEI2")

    identified = assign_company_ids(trades, fake_reports("2026-10-16"))

    assert identified["company_id"].isna().all()


def test_windows_include_the_right_days():
    trades = fake_trades(
        "2026-09-01",  # dag -45, för tidigt
        "2026-09-02",  # dag -44, första dagen före
        "2026-09-15",  # dag -31, sista dagen före
        "2026-09-16",  # dag -30, handelsstoppet har börjat
        "2026-10-15",  # dag -1, handelsstopp
        "2026-10-16",  # dag 0, första dagen efter
        "2026-10-30",  # dag 14, sista dagen efter
        "2026-10-31",  # dag 15, för sent
    )

    matched = match_trades_to_windows(trades, fake_reports("2026-10-16"))

    days = matched.groupby("window")["days_from_report"].apply(list)
    assert days["before"] == [-44, -31]
    assert days["after"] == [0, 14]


def test_trade_is_measured_against_the_reports_of_its_own_company():
    reports = pd.concat(
        [
            fake_reports("2026-10-16"),
            fake_reports("2026-05-01", company_id="10", isin="SE0000000009"),
        ],
        ignore_index=True,
    )

    matched = match_trades_to_windows(fake_trades("2026-05-02"), reports)

    assert matched.empty


def test_trade_in_windows_of_two_reports_gets_one_row_per_report():
    matched = match_trades_to_windows(
        fake_trades("2026-10-20"), fake_reports("2026-10-16", "2026-11-25")
    )

    assert sorted(matched["window"]) == ["after", "before"]
    assert matched["report_date"].nunique() == 2


def test_old_window_label_is_replaced():
    trades = fake_trades("2026-10-20").assign(window="before")

    matched = match_trades_to_windows(trades, fake_reports("2026-10-16"))

    assert matched["window"].tolist() == ["after"]
