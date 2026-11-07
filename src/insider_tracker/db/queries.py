"""Läser tillbaka data från databasen som pandas-tabeller."""

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from insider_tracker.db.models import Company, Report, Trade


def read_trades(session: Session) -> pd.DataFrame:
    """Hämtar alla affärer med samma kolumner som clean_fi_df ger."""
    statement = select(
        Trade.published_at,
        Company.name.label("company"),
        Company.lei,
        Trade.person_name,
        Trade.position,
        Trade.is_related,
        Trade.is_correction,
        Trade.is_share_program,
        Trade.transaction_type,
        Trade.instrument_type,
        Trade.instrument_name,
        Trade.isin,
        Trade.transaction_date,
        Trade.volume,
        Trade.volume_unit,
        Trade.price,
        Trade.currency,
        Trade.venue,
        Trade.source_url,
    ).join(Trade.company)
    trades = pd.read_sql(statement, session.connection())

    trades["published_at"] = pd.to_datetime(trades["published_at"])
    trades["transaction_date"] = pd.to_datetime(trades["transaction_date"])
    for column in ["is_related", "is_correction", "is_share_program"]:
        trades[column] = trades[column].astype(bool)
    return trades


def read_reports(session: Session) -> pd.DataFrame:
    """Hämtar alla rapportdagar med samma kolumner som filen med rapportdatum har."""
    statement = select(
        Report.source_company_id.label("company_id"),
        Report.orderbook_id,
        Report.name,
        Report.isin,
        Report.ticker,
        Report.market_list,
        Report.report_date,
        Report.report_type,
        Report.is_confirmed,
        Report.source_url,
    )
    reports = pd.read_sql(statement, session.connection())
    reports["report_date"] = pd.to_datetime(reports["report_date"])
    return reports
