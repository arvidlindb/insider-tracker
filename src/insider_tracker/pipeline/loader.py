"""Sparar rapportdatum och insynsaffärer i databasen.

Första körningen hämtar hela perioden från FI. Sedan hämtas bara det senaste, och rader som
redan finns sparas inte igen.

Kör så här:
    python -m insider_tracker.pipeline.loader
"""

import argparse
import hashlib
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from insider_tracker.db.database import get_engine, get_session
from insider_tracker.db.models import Company, FetchLog, Report, Trade
from insider_tracker.pipeline.parser import clean_fi_df
from insider_tracker.pipeline.report_windows import REPORTS_PATH, load_reports
from insider_tracker.providers.fi_client import USER_AGENT, WINDOWS, fetch_trades_in_chunks

FINGERPRINT_COLUMNS = [
    "published_at",
    "company",
    "lei",
    "person_name",
    "is_related",
    "transaction_type",
    "instrument_name",
    "isin",
    "transaction_date",
    "volume",
    "price",
    "currency",
    "venue",
]

# Säkerhetsmarginal så att affärer som publicerats sent i förra körningen inte missas.
REFETCH_MARGIN_DAYS = 7


def trade_fingerprints(trades: pd.DataFrame) -> list[str]:
    """Räknar ut ett fingeravtryck för varje affär.

    Två rader kan se likadana ut och ändå vara olika affärer, till exempel när en order
    delas upp. Därför ingår ett löpnummer bland identiska rader. Då sparas båda, medan en
    ny hämtning av samma rader ändå ger samma fingeravtryck.
    """
    seen = Counter()
    fingerprints = []
    for values in trades[FINGERPRINT_COLUMNS].itertuples(index=False, name=None):
        key = "|".join(str(value) for value in values)
        fingerprints.append(hashlib.sha256(f"{key}|{seen[key]}".encode()).hexdigest())
        seen[key] += 1
    return fingerprints


def to_stored(value: object) -> object:
    """Gör om tomma värden och pandas-datum till sådant som databasen förstår."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def get_or_create_company(session: Session, companies: dict, row: dict) -> Company:
    """Hittar bolaget, eller skapar det första gången det dyker upp."""
    name = to_stored(row["company"]) or "Okänt bolag"
    lei = to_stored(row["lei"])
    key = lei or name
    if key not in companies:
        companies[key] = Company(key=key, name=name, lei=lei)
        session.add(companies[key])
    return companies[key]


def load_trades(session: Session, trades: pd.DataFrame) -> int:
    """Sparar affärerna och returnerar hur många som var nya."""
    existing = set(session.scalars(select(Trade.fingerprint)))
    companies = {company.key: company for company in session.scalars(select(Company))}

    new_rows = 0
    fingerprints = trade_fingerprints(trades)
    for row, fingerprint in zip(trades.to_dict("records"), fingerprints, strict=True):
        if fingerprint in existing:
            continue
        session.add(
            Trade(
                fingerprint=fingerprint,
                company=get_or_create_company(session, companies, row),
                published_at=to_stored(row["published_at"]),
                transaction_date=row["transaction_date"].date(),
                person_name=to_stored(row["person_name"]),
                position=to_stored(row["position"]),
                is_related=row["is_related"],
                is_correction=row["is_correction"],
                is_share_program=row["is_share_program"],
                transaction_type=to_stored(row["transaction_type"]),
                instrument_type=to_stored(row["instrument_type"]),
                instrument_name=to_stored(row["instrument_name"]),
                isin=to_stored(row["isin"]),
                volume=to_stored(row["volume"]),
                volume_unit=to_stored(row["volume_unit"]),
                price=to_stored(row["price"]),
                currency=to_stored(row["currency"]),
                venue=to_stored(row["venue"]),
                source_url=row["source_url"],
            )
        )
        new_rows += 1
    return new_rows


def load_report_rows(session: Session, reports: pd.DataFrame) -> int:
    """Sparar rapportdagarna och returnerar hur många som var nya."""
    keys = select(Report.source_company_id, Report.report_date, Report.report_type)
    existing = {tuple(key) for key in session.execute(keys)}

    # Finns bolaget redan, via samma ISIN, används dess id. Annars hamnar nya rapportdatum
    # under ett annat id än bolagets affärer, och då kopplas de aldrig ihop.
    known_ids = dict(session.execute(select(Report.isin, Report.source_company_id)).all())

    new_rows = 0
    for row in reports.to_dict("records"):
        company_id = known_ids.get(row["isin"], row["company_id"])
        report_date = row["report_date"].date()
        if (company_id, report_date, row["report_type"]) in existing:
            continue
        existing.add((company_id, report_date, row["report_type"]))
        session.add(
            Report(
                source_company_id=company_id,
                # En egen lista har varken orderbok eller ticker, så de får vara tomma.
                orderbook_id=to_stored(row.get("orderbook_id")) or "",
                name=to_stored(row["name"]),
                isin=to_stored(row["isin"]),
                ticker=to_stored(row.get("ticker")),
                market_list=to_stored(row["market_list"]),
                report_date=report_date,
                report_type=row["report_type"],
                is_confirmed=row["is_confirmed"] == "True",
                source_url=row["source_url"],
            )
        )
        new_rows += 1
    return new_rows


def find_fetch_start(session: Session, reports: pd.DataFrame) -> date:
    """Tar reda på varifrån FI-hämtningen ska börja."""
    latest = session.scalar(select(func.max(Trade.published_at)))
    if latest is not None:
        return latest.date() - timedelta(days=REFETCH_MARGIN_DAYS)

    earliest_day = min(first_day for first_day, _ in WINDOWS.values())
    return reports["report_date"].min().date() + timedelta(days=earliest_day)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparar rapportdatum och affärer i databasen.")
    parser.add_argument("--reports", type=Path, default=REPORTS_PATH, help="CSV med rapportdatum")
    args = parser.parse_args()

    reports = load_reports(args.reports)
    engine = get_engine()
    today = date.today()

    with get_session(engine) as session:
        new_reports = load_report_rows(session, reports)
        session.flush()
        start = find_fetch_start(session, reports)

        with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60) as client:
            raw = fetch_trades_in_chunks(client, start, today)

        new_trades = load_trades(session, clean_fi_df(raw))
        session.add(
            FetchLog(from_date=start, to_date=today, rows_fetched=len(raw), rows_new=new_trades)
        )
        session.flush()

        total_trades = session.scalar(select(func.count()).select_from(Trade))
        total_reports = session.scalar(select(func.count()).select_from(Report))

    print(f"Hämtade från FI sedan {start}: {len(raw)} rader")
    print(f"Nya affärer: {new_trades}, totalt {total_trades}")
    print(f"Nya rapportdagar: {new_reports}, totalt {total_reports}")


if __name__ == "__main__":
    main()
