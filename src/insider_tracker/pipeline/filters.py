"""Filtrerar fram de insynsaffärer som är intressanta att analysera.

Stegen körs i ordning och räknar hur många rader som försvinner i varje steg:
    1. Bara marknadsköp och marknadsförsäljningar av aktier.
    2. Samma person, bolag, dag och affärstyp slås ihop till en rad.
    3. Affären måste nå beloppsgränsen för bolagets segment.
    4. Försäljningar måste uppfylla ett extra villkor.

Vi hittar aldrig på data. Saknas det uppgifter för att avgöra ett filter behålls raden,
och kolumnen uncertain_reason berättar varför den är osäker.

Kör så här:
    python -m insider_tracker.pipeline.filters
"""

from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

from insider_tracker.pipeline.parser import clean_fi_df
from insider_tracker.pipeline.report_windows import assign_company_ids, load_reports
from insider_tracker.providers.fi_client import USER_AGENT, fetch_trades_in_chunks

OUTPUT_PATH = Path("data/filtered_trades.csv")

KEPT_TRANSACTION_TYPES = {"Förvärv", "Avyttring"}
SALE = "Avyttring"
SHARE = "Aktie"
OFF_MARKET = "Utanför handelsplats"

MINIMUM_AMOUNT_SEK = {"small": 250_000, "mid": 1_000_000, "large": 2_000_000, "unknown": 500_000}
LIST_SEGMENTS = {
    "Large Cap": "large",
    "Mid Cap": "mid",
    "Small Cap": "small",
    "NGM": "small",
    "First North": "small",
    "Spotlight": "small",
}
SMALL_COMPANY_VENUES = ("SPOTLIGHT", "NORDIC GROWTH", "FIRST NORTH")

SALE_CLUSTER_DAYS = 30

# Köp och försäljning samma dag är olika affärer, så de får inte räknas ihop. Detsamma gäller
# valutor, och en närståendes affär, eftersom den görs av någon annan än personen själv.
MERGE_KEYS = [
    "person_name",
    "company",
    "transaction_date",
    "transaction_type",
    "currency",
    "is_related",
]


def add_note(trades: pd.DataFrame, mask: pd.Series, note: str) -> None:
    """Skriver på varför raderna i mask är osäkra."""
    trades.loc[mask, "uncertain_reason"] = trades.loc[mask, "uncertain_reason"] + note + "; "


def market_segment(market_list: object, venue: object) -> str:
    """Bestämmer bolagets segment, i första hand från marknadslistan i rapportdatumen."""
    for name, segment in LIST_SEGMENTS.items():
        if name in str(market_list):
            return segment
    # Nasdaq Stockholm säger inget om storleken, men vissa handelsplatser gör det.
    if any(name in str(venue).upper() for name in SMALL_COMPANY_VENUES):
        return "small"
    return "unknown"


def add_segments(trades: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """Lägger till kolumnen segment för varje affär."""
    lists = reports.drop_duplicates("company_id").set_index("company_id")["market_list"]
    market_lists = trades["company_id"].map(lists)
    segments = [
        market_segment(market_list, venue)
        for market_list, venue in zip(market_lists, trades["venue"], strict=True)
    ]
    return trades.assign(segment=segments)


def keep_market_share_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Steg 1: behåller köp och försäljningar av aktier på marknaden.

    Interna överföringar har ingen egen markering hos FI. De flesta bör fångas av
    "Utanför handelsplats", men det går inte att vara helt säker.
    """
    type_missing = trades["transaction_type"].isna()
    instrument_missing = trades["instrument_type"].isna()
    venue_missing = trades["venue"].isna()

    keep = (
        (trades["transaction_type"].isin(KEPT_TRANSACTION_TYPES) | type_missing)
        & (trades["instrument_type"].eq(SHARE) | instrument_missing)
        & ~trades["is_share_program"]
        & ~trades["venue"].eq(OFF_MARKET)
    )

    kept = trades[keep].copy()
    add_note(kept, type_missing[keep], "affärstyp saknas")
    add_note(kept, instrument_missing[keep], "instrumenttyp saknas")
    add_note(kept, venue_missing[keep], "handelsplats saknas")
    return kept


def merge_same_day_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Steg 2: slår ihop en persons affärer samma dag, med volymvägt snittpris."""
    has_amount = trades["price"].gt(0) & trades["volume"].notna()
    frame = trades.assign(
        priced_amount=(trades["volume"] * trades["price"]).where(has_amount),
        priced_volume=trades["volume"].where(has_amount),
        amount_unknown=~has_amount,
        merged_rows=1,
    )

    aggregations = {column: "first" for column in frame.columns if column not in MERGE_KEYS}
    aggregations.update(
        volume="sum",
        priced_amount="sum",
        priced_volume="sum",
        amount_unknown="max",
        merged_rows="sum",
        published_at="max",
        is_correction="max",
        uncertain_reason=lambda notes: "".join(dict.fromkeys(notes)),
    )
    merged = frame.groupby(MERGE_KEYS, dropna=False, sort=False).agg(aggregations).reset_index()

    # Är volymen noll går det inte att räkna ut något snittpris, och då blir priset tomt.
    merged["price"] = merged["priced_amount"] / merged["priced_volume"].where(
        merged["priced_volume"] > 0
    )
    return merged.drop(columns=["priced_amount", "priced_volume"])


def keep_large_enough_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Steg 3: tar bara bort en affär när vi säkert vet att den är under beloppsgränsen."""
    trades = trades.copy()
    is_sek = trades["currency"].eq("SEK")
    amount_known = ~trades["amount_unknown"].astype(bool)

    add_note(trades, ~is_sek, "beloppet är inte i SEK")
    add_note(trades, ~amount_known, "pris eller volym saknas")

    trades["amount_sek"] = (trades["volume"] * trades["price"]).where(is_sek & amount_known)
    minimum = trades["segment"].map(MINIMUM_AMOUNT_SEK)
    too_small = trades["amount_sek"].lt(minimum)
    return trades[~too_small].drop(columns="amount_unknown")


def has_other_seller_nearby(trades: pd.DataFrame) -> pd.Series:
    """Svarar för varje försäljning om en annan person sålt i bolaget inom 30 dagar."""
    result = pd.Series(False, index=trades.index)
    sales = trades[trades["transaction_type"].eq(SALE)]
    limit = pd.Timedelta(days=SALE_CLUSTER_DAYS)

    for _, group in sales.groupby("company", dropna=False):
        for index, sale in group.iterrows():
            nearby = (group["transaction_date"] - sale["transaction_date"]).abs() <= limit
            result.loc[index] = group.loc[nearby, "person_name"].ne(sale["person_name"]).any()
    return result


def check_sales(trades: pd.DataFrame) -> pd.DataFrame:
    """Steg 4: markerar försäljningar vars villkor inte kan bekräftas.

    En försäljning klarar sig om personen säljer minst halva innehavet eller om fler
    insiders säljer. FI:s register visar inte innehavet, så det första villkoret går inte
    att pröva. Därför kan inga rader tas bort här, bara markeras som osäkra.
    """
    trades = trades.copy()
    unconfirmed = trades["transaction_type"].eq(SALE) & ~has_other_seller_nearby(trades)
    add_note(trades, unconfirmed, "innehav saknas för 50 %-villkoret")
    return trades


def filter_trades(trades: pd.DataFrame, reports: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Kör alla steg. Returnerar kvarvarande affärer och hur många rader varje steg tog bort."""
    # Bolagen kopplas innan något filtreras bort, så att LEI-koppling får så många träffar som
    # möjligt. Osäkerhetskolumnen börjar tom på alla rader.
    prepared = add_segments(assign_company_ids(trades, reports), reports)
    prepared = prepared.assign(uncertain_reason="").astype({"uncertain_reason": object})

    market_trades = keep_market_share_trades(prepared)
    merged = merge_same_day_trades(market_trades)
    large_enough = keep_large_enough_trades(merged)
    checked = check_sales(large_enough)

    checked["uncertain_reason"] = checked["uncertain_reason"].str.rstrip("; ")
    removed = {
        "transaction_type": len(prepared) - len(market_trades),
        "merged": len(market_trades) - len(merged),
        "amount": len(merged) - len(large_enough),
        "sale_condition": 0,
    }
    return checked.reset_index(drop=True), removed


def main() -> None:
    reports = load_reports()
    start = reports["report_date"].min().date() - timedelta(days=44)

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60) as client:
        raw = fetch_trades_in_chunks(client, start, date.today())

    trades = clean_fi_df(raw)
    filtered, removed = filter_trades(trades, reports)

    # Filen innehåller personnamn och ligger därför bara i data/, som inte följer med till git.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(OUTPUT_PATH, index=False)

    print(f"Rader från FI: {len(raw)}")
    print(f"Borttagna som ersatta eller utan datum: {len(raw) - len(trades)}")
    print(f"Borttagna på typ, instrument eller handelsplats: {removed['transaction_type']}")
    print(f"Försvunna vid sammanslagning: {removed['merged']}")
    print(f"Borttagna under beloppsgränsen: {removed['amount']}")
    print(f"Borttagna på säljvillkoret: {removed['sale_condition']}")
    print(f"\nKvar: {len(filtered)}, sparade i {OUTPUT_PATH}")
    print(filtered["transaction_type"].value_counts().to_string())
    print(f"\nOsäkra rader: {filtered['uncertain_reason'].ne('').sum()}")
    print(filtered.loc[filtered["uncertain_reason"].ne(""), "uncertain_reason"]
          .value_counts().to_string())


if __name__ == "__main__":
    main()
