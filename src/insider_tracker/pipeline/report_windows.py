"""Kopplar insynsaffärer till rätt bolag och rapport, och märker dem med fönster.

Namnen i rapportkalendern ("Investor B") och hos FI ("Investor AB") stämmer sällan överens,
så vi jämför inte namn. Kopplingen görs i två steg:
    1. Affärens ISIN jämförs med aktien vi har rapportdatum för.
    2. Träffarna visar bolagets LEI-kod, som sedan fångar bolagets andra instrument,
       till exempel optioner och andra aktieslag.

Kör så här:
    python -m insider_tracker.pipeline.report_windows
"""

from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

from insider_tracker.pipeline.parser import clean_fi_df
from insider_tracker.providers.fi_client import USER_AGENT, WINDOWS, fetch_trades_in_chunks

REPORTS_PATH = Path("data/report_dates.csv")


def load_reports(path: Path = REPORTS_PATH) -> pd.DataFrame:
    """Läser en CSV med rapportdatum."""
    reports = pd.read_csv(path, dtype=str)
    reports["report_date"] = pd.to_datetime(reports["report_date"])
    return reports


def assign_company_ids(trades: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """Lägger till kolumnen company_id. Den är tom för affärer vi inte har något bolag för."""
    isin_to_company = reports.drop_duplicates("isin").set_index("isin")["company_id"]
    by_isin = trades["isin"].map(isin_to_company)

    # Alla affärer med en LEI-kod som fick träff på ISIN hör till samma bolag.
    known = pd.DataFrame({"lei": trades["lei"], "company_id": by_isin}).dropna()
    lei_to_company = known.drop_duplicates("lei").set_index("lei")["company_id"]
    by_lei = trades["lei"].map(lei_to_company)

    # Kolumnerna kan få olika datatyper när en av dem är helt tom, så de görs lika först.
    return trades.assign(company_id=by_isin.astype(object).fillna(by_lei.astype(object)))


def match_trades_to_windows(trades: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """Returnerar en rad per affär och rapport där affären ligger inom ett fönster.

    Ligger en affär i fönstren för två rapporter blir det två rader, en för varje rapport.
    """
    identified = assign_company_ids(trades, reports).dropna(subset=["company_id"])

    # En äldre fönstermärkning från fi_client gäller inte längre, så den tas bort.
    identified = identified.drop(columns="window", errors="ignore")

    paired = identified.merge(
        reports[["company_id", "report_date", "report_type"]], on="company_id"
    )
    paired["days_from_report"] = (paired["transaction_date"] - paired["report_date"]).dt.days

    frames = []
    for window, (first_day, last_day) in WINDOWS.items():
        in_window = paired["days_from_report"].between(first_day, last_day)
        frames.append(paired[in_window].assign(window=window))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    reports = load_reports()

    # Äldsta rapporten minus det tidigaste fönstret ger var hämtningen ska börja.
    earliest_day = min(first_day for first_day, _ in WINDOWS.values())
    start = reports["report_date"].min().date() + timedelta(days=earliest_day)

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60) as client:
        trades = clean_fi_df(fetch_trades_in_chunks(client, start, date.today()))

    identified = assign_company_ids(trades, reports)
    matched = match_trades_to_windows(trades, reports)
    unmatched = identified[identified["company_id"].isna()]

    # Bara antal och bolagsnamn skrivs ut, aldrig personnamn.
    print(f"Hämtade affärer sedan {start}: {len(trades)}")
    print(f"Kopplade till ett bolag: {identified['company_id'].notna().sum()}")
    print(f"Bolag med minst en affär: {identified['company_id'].nunique()}")
    print(f"Rapporter totalt: {reports['company_id'].nunique()} bolag")
    print("\nAffärer i fönstren:")
    print(matched["window"].value_counts().to_string())
    print("\nBolag utan koppling, flest affärer först:")
    print(unmatched["company"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
