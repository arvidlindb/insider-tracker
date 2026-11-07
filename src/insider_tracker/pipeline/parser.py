"""Gör om FI:s rådata till en ren tabell med rätt datatyper."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Bara de här kolumnerna följer med. Anmälningsskyldig utelämnas med flit, eftersom det kan
# vara en närstående privatperson. Beskrivningen av en korrigering är fritext och utelämnas också.
COLUMN_NAMES = {
    "Publiceringsdatum": "published_at",
    "Emittent": "company",
    "LEI-kod": "lei",
    "Person i ledande ställning": "person_name",
    "Befattning": "position",
    "Närstående": "is_related",
    "Korrigering": "is_correction",
    "Är kopplad till aktieprogram": "is_share_program",
    "Karaktär": "transaction_type",
    "Instrumenttyp": "instrument_type",
    "Instrumentnamn": "instrument_name",
    "ISIN": "isin",
    "Transaktionsdatum": "transaction_date",
    "Volym": "volume",
    "Volymsenhet": "volume_unit",
    "Pris": "price",
    "Valuta": "currency",
    "Handelsplats": "venue",
    "source_url": "source_url",
}

TEXT_COLUMNS = ["company", "person_name", "position", "transaction_type", "instrument_type",
                "instrument_name"]
NUMBER_COLUMNS = ["volume", "price"]
DATE_COLUMNS = ["published_at", "transaction_date"]
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
FLAG_COLUMNS = ["is_related", "is_correction", "is_share_program"]


def to_number(values: pd.Series) -> pd.Series:
    """Gör om text som "1 234,50" till tal. Ogiltiga värden blir tomma."""
    # Det hårda mellanslaget skrivs ut för sig, eftersom \s inte alltid täcker det i pandas.
    without_spaces = values.str.replace("[\\s\xa0]", "", regex=True)
    return pd.to_numeric(without_spaces.str.replace(",", ".", regex=False), errors="coerce")


def clean_fi_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Rensar affärer från fi_client och returnerar en ny tabell med engelska kolumnnamn."""
    # När en affär rättas blir den gamla raden "Reviderad" och den nya "Aktuell".
    # Behöll vi båda skulle samma affär räknas två gånger.
    current = raw[raw["Status"] != "Reviderad"]

    # Fönstret finns bara om affärerna hämtats med fetch_report_windows.
    columns = [*COLUMN_NAMES, *(["window"] if "window" in raw.columns else [])]
    trades = current[columns].rename(columns=COLUMN_NAMES)

    # FI använder hårda mellanslag i vissa texter, till exempel i befattningar.
    for column in TEXT_COLUMNS:
        trades[column] = trades[column].str.replace("\xa0", " ", regex=False).str.strip()

    for column in NUMBER_COLUMNS:
        trades[column] = to_number(trades[column])

    for column in DATE_COLUMNS:
        trades[column] = pd.to_datetime(trades[column], format=DATE_FORMAT, errors="coerce")

    # I FI:s fil står det "Ja" eller ingenting, så en tom ruta betyder nej.
    for column in FLAG_COLUMNS:
        trades[column] = trades[column].isin(["Ja"])

    # Utan transaktionsdatum går affären inte att placera i något fönster. Saknat pris får
    # däremot vara kvar, eftersom affären fortfarande säger något.
    missing_date = trades["transaction_date"].isna()
    if missing_date.any():
        logger.warning("Tog bort %d affärer utan giltigt transaktionsdatum", missing_date.sum())

    return trades[~missing_date].reset_index(drop=True)
