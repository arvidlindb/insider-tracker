import pandas as pd

from insider_tracker.pipeline.parser import clean_fi_df
from insider_tracker.providers.fi_client import parse_csv

DEFAULT_ROW = {
    "Publiceringsdatum": "2026-11-03 17:44:26",
    "Emittent": "Testbolaget AB",
    "LEI-kod": "LEI1",
    "Anmälningsskyldig": "Närstående Fejksson",
    "Person i ledande ställning": "Fejk Fejksson",
    "Befattning": "Verkställande\xa0direktör\xa0(VD)",
    "Närstående": "Ja",
    "Korrigering": "",
    "Beskrivning av korrigering": "",
    "Är förstagångsrapportering": "Ja",
    "Är kopplad till aktieprogram": "",
    "Karaktär": "Förvärv",
    "Instrumenttyp": "Aktie",
    "Instrumentnamn": "Testbolaget B",
    "ISIN": "SE0000000001",
    "Transaktionsdatum": "2026-11-02 00:00:00",
    "Volym": "23116,0",
    "Volymsenhet": "Antal",
    "Pris": "23,6297",
    "Valuta": "SEK",
    "Handelsplats": "XSTO",
    "Status": "Aktuell",
}


def fake_raw(*changes: dict) -> pd.DataFrame:
    """Bygger rådata via riktiga CSV-inläsningen, en rad per ändring av standardraden."""
    rows = [{**DEFAULT_ROW, **change} for change in changes]
    lines = [";".join(DEFAULT_ROW) + ";"] + [";".join(row.values()) + ";" for row in rows]
    raw = parse_csv("\r\n".join(lines).encode("utf-16-le"))
    raw["source_url"] = "https://example.test/export"
    return raw


def test_clean_converts_types():
    trade = clean_fi_df(fake_raw({})).iloc[0]

    assert trade["volume"] == 23116.0
    assert trade["price"] == 23.6297
    assert trade["transaction_date"] == pd.Timestamp("2026-11-02")
    assert trade["published_at"] == pd.Timestamp("2026-11-03 17:44:26")
    assert trade["source_url"] == "https://example.test/export"


def test_clean_handles_spaces_in_numbers_and_invalid_values():
    trades = clean_fi_df(fake_raw({"Volym": "1\xa0234,5", "Pris": "okänt"}))

    assert trades.loc[0, "volume"] == 1234.5
    assert pd.isna(trades.loc[0, "price"])


def test_clean_turns_ja_and_empty_into_true_and_false():
    trade = clean_fi_df(fake_raw({})).iloc[0]

    assert trade["is_related"]
    assert not trade["is_correction"]
    assert not trade["is_share_program"]


def test_clean_replaces_hard_spaces_in_text():
    assert clean_fi_df(fake_raw({})).loc[0, "position"] == "Verkställande direktör (VD)"


def test_clean_drops_replaced_rows_and_keeps_the_correction():
    trades = clean_fi_df(
        fake_raw(
            {"Status": "Reviderad", "Volym": "100,0"},
            {"Korrigering": "Ja", "Volym": "200,0"},
        )
    )

    assert trades["volume"].tolist() == [200.0]
    assert trades.loc[0, "is_correction"]


def test_clean_drops_rows_without_valid_transaction_date():
    trades = clean_fi_df(fake_raw({"Transaktionsdatum": "ogiltigt"}, {}))

    assert len(trades) == 1


def test_clean_leaves_out_the_person_who_filed_the_report():
    trades = clean_fi_df(fake_raw({}))

    assert "Närstående Fejksson" not in trades.to_string()


def test_clean_keeps_window_when_it_exists():
    raw = fake_raw({}).assign(window="before")

    assert clean_fi_df(raw).loc[0, "window"] == "before"
    assert "window" not in clean_fi_df(fake_raw({})).columns
