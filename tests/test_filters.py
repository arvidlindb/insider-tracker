import pandas as pd

from insider_tracker.pipeline.filters import filter_trades

BUY = "Förvärv"
SELL = "Avyttring"


def fake_trade(**changes) -> dict:
    # Standardraden är ett köp på 3 000 000 SEK, över alla beloppsgränser.
    return {
        "published_at": pd.Timestamp("2026-10-21 12:00:00"),
        "company": "Testbolaget AB",
        "lei": "LEI1",
        "person_name": "Fejk Fejksson",
        "position": "Verkställande direktör (VD)",
        "is_related": False,
        "is_correction": False,
        "is_share_program": False,
        "transaction_type": BUY,
        "instrument_type": "Aktie",
        "instrument_name": "Testbolaget B",
        "isin": "SE0000000001",
        "transaction_date": pd.Timestamp("2026-10-20"),
        "volume": 30_000.0,
        "volume_unit": "Antal",
        "price": 100.0,
        "currency": "SEK",
        "venue": "NASDAQ STOCKHOLM AB",
        "source_url": "https://example.test/export",
        **changes,
    }


def run(*rows: dict, market_list: str | None = None):
    reports = pd.DataFrame(
        {"company_id": ["9"], "isin": ["SE0000000001"], "market_list": [market_list]}
    )
    return filter_trades(pd.DataFrame(rows), reports)


def test_normal_market_purchase_is_kept_without_note():
    filtered, removed = run(fake_trade())

    assert len(filtered) == 1
    assert filtered.loc[0, "uncertain_reason"] == ""
    assert removed["transaction_type"] == 0


def test_step_1_removes_everything_that_is_not_a_market_trade_in_shares():
    filtered, removed = run(
        fake_trade(),
        fake_trade(person_name="Fejk A", transaction_type="Tilldelning"),
        fake_trade(person_name="Fejk B", transaction_type="Teckning"),
        fake_trade(person_name="Fejk C", instrument_type="Teckningsoption"),
        fake_trade(person_name="Fejk D", is_share_program=True),
        fake_trade(person_name="Fejk E", venue="Utanför handelsplats"),
    )

    assert filtered["person_name"].tolist() == ["Fejk Fejksson"]
    assert removed["transaction_type"] == 5


def test_step_1_keeps_row_with_missing_field_and_marks_it():
    filtered, _ = run(fake_trade(venue=None))

    assert len(filtered) == 1
    assert "handelsplats saknas" in filtered.loc[0, "uncertain_reason"]


def test_step_2_merges_with_volume_weighted_price():
    filtered, removed = run(
        fake_trade(volume=10_000.0, price=100.0),
        fake_trade(volume=30_000.0, price=200.0),
    )

    assert len(filtered) == 1
    assert filtered.loc[0, "volume"] == 40_000.0
    assert filtered.loc[0, "price"] == 175.0
    assert filtered.loc[0, "merged_rows"] == 2
    assert removed["merged"] == 1


def test_step_2_keeps_buy_and_sell_on_same_day_apart():
    filtered, removed = run(fake_trade(), fake_trade(transaction_type=SELL))

    assert len(filtered) == 2
    assert removed["merged"] == 0


def test_step_2_keeps_related_persons_trade_apart():
    filtered, _ = run(fake_trade(), fake_trade(is_related=True))

    assert len(filtered) == 2


def test_step_3_uses_limit_for_large_cap():
    kept, _ = run(fake_trade(volume=20_000.0), market_list="Large Cap Stockholm")
    dropped, removed = run(fake_trade(volume=19_999.0), market_list="Large Cap Stockholm")

    assert len(kept) == 1
    assert kept.loc[0, "amount_sek"] == 2_000_000.0
    assert dropped.empty
    assert removed["amount"] == 1


def test_step_3_uses_limit_for_mid_and_small_cap():
    assert len(run(fake_trade(volume=10_000.0), market_list="Mid Cap Stockholm")[0]) == 1
    assert run(fake_trade(volume=9_999.0), market_list="Mid Cap Stockholm")[0].empty
    assert len(run(fake_trade(volume=2_500.0), market_list="Small Cap Stockholm")[0]) == 1
    assert run(fake_trade(volume=2_499.0), market_list="Small Cap Stockholm")[0].empty


def test_step_3_uses_limit_for_unknown_segment():
    assert len(run(fake_trade(volume=5_000.0))[0]) == 1
    assert run(fake_trade(volume=4_999.0))[0].empty


def test_step_3_reads_small_company_venue_for_unknown_company():
    filtered, _ = run(fake_trade(isin="SE0000000002", lei="LEI2", volume=2_500.0,
                                 venue="SPOTLIGHT STOCK MARKET"))

    assert len(filtered) == 1
    assert filtered.loc[0, "segment"] == "small"


def test_step_3_keeps_other_currency_and_marks_it():
    filtered, _ = run(fake_trade(currency="EUR", volume=1.0))

    assert len(filtered) == 1
    assert "inte i SEK" in filtered.loc[0, "uncertain_reason"]


def test_step_3_keeps_missing_or_zero_price_and_marks_it():
    filtered, _ = run(
        fake_trade(price=None),
        fake_trade(person_name="Fejk A", price=0.0),
    )

    assert len(filtered) == 2
    assert filtered["uncertain_reason"].str.contains("pris eller volym saknas").all()


def test_step_4_marks_single_seller_but_keeps_the_row():
    filtered, removed = run(fake_trade(transaction_type=SELL))

    assert len(filtered) == 1
    assert "innehav saknas" in filtered.loc[0, "uncertain_reason"]
    assert removed["sale_condition"] == 0


def test_step_4_two_sellers_within_30_days_are_not_uncertain():
    filtered, _ = run(
        fake_trade(transaction_type=SELL),
        fake_trade(
            transaction_type=SELL,
            person_name="Fejk A",
            transaction_date=pd.Timestamp("2026-11-19"),
        ),
    )

    assert filtered["uncertain_reason"].tolist() == ["", ""]


def test_step_4_sellers_more_than_30_days_apart_are_uncertain():
    filtered, _ = run(
        fake_trade(transaction_type=SELL),
        fake_trade(
            transaction_type=SELL,
            person_name="Fejk A",
            transaction_date=pd.Timestamp("2026-11-20"),
        ),
    )

    assert filtered["uncertain_reason"].str.contains("innehav saknas").all()


def test_same_person_selling_twice_does_not_count_as_two_sellers():
    filtered, _ = run(
        fake_trade(transaction_type=SELL),
        fake_trade(transaction_type=SELL, transaction_date=pd.Timestamp("2026-10-25")),
    )

    assert filtered["uncertain_reason"].str.contains("innehav saknas").all()
