"""Ett API som låter andra program läsa insynsaffärerna som JSON.

API:t läser bara. Det visar samma filtrerade affärer som dashboarden, utan personnamn, och
kan aldrig ändra databasen eller göra betalda AI-anrop.

Kör så här och öppna http://localhost:8000/docs för att prova anropen:
    uvicorn insider_tracker.api.main:app --reload
"""

from collections.abc import Iterator
from datetime import date, datetime
from functools import lru_cache
from typing import Annotated, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from insider_tracker.db.database import get_engine, get_session
from insider_tracker.providers.fi_client import WINDOWS
from insider_tracker.ui.dashboard_data import latest_summary, load_trades

MAX_LIMIT = 500

Window = Literal["before", "after"]
TransactionType = Literal["Förvärv", "Avyttring"]
PositionGroup = Literal["Vd", "Ekonomichef", "Styrelse", "Övriga"]

app = FastAPI(
    title="Insider Tracker API",
    description=(
        "Insynsaffärer från FI:s register runt bolagens rapporter. Datan visar när affärer "
        "skett, inte varför. Inga investeringsråd."
    ),
)


class Company(BaseModel):
    name: str
    trades: int


class Trade(BaseModel):
    """En affär. Fälten är valda med flit, och personnamn finns inte med."""

    transaction_date: date
    company: str
    window: Window
    days_from_report: int
    report_date: date
    position: str | None
    position_group: str
    is_related: bool
    transaction_type: str
    volume: float | None
    price: float | None
    currency: str | None
    amount_sek: float | None
    uncertain_reason: str
    source_url: str


class WindowCount(BaseModel):
    window: Window
    position_group: str
    transaction_type: str
    trades: int
    trades_per_window_day: float


class SummaryOut(BaseModel):
    company: str
    text: str
    model: str
    created_at: datetime


@lru_cache(maxsize=1)
def _engine():
    return get_engine()


@lru_cache(maxsize=1)
def _cached_trades() -> pd.DataFrame:
    # Datan läses en gång per start, eftersom filtret tar en stund. Starta om för ny data.
    with get_session(_engine()) as session:
        return load_trades(session)


def get_trades() -> pd.DataFrame:
    """Ger affärerna. Testerna byter ut den här mot fejkad data via dependency injection."""
    return _cached_trades()


def get_db() -> Iterator[Session]:
    with get_session(_engine()) as session:
        yield session


Trades = Annotated[pd.DataFrame, Depends(get_trades)]


def _records(frame: pd.DataFrame) -> list[dict]:
    """Gör om tabellen till rader där tomma värden blir null i JSON."""
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/companies")
def list_companies(trades: Trades) -> list[Company]:
    """Bolag med affärer i fönstren, flest affärer först."""
    counts = trades.groupby("company").size().sort_values(ascending=False, kind="stable")
    return [Company(name=name, trades=int(count)) for name, count in counts.items()]


@app.get("/trades")
def list_trades(
    trades: Trades,
    company: str | None = None,
    window: Window | None = None,
    transaction_type: TransactionType | None = None,
    position_group: PositionGroup | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 100,
) -> list[Trade]:
    """Affärer i fönstren, de senaste först. Taket på antal rader skyddar mot för stora svar."""
    selected = trades
    if company is not None:
        selected = selected[selected["company"].str.lower().eq(company.lower())]
    if window is not None:
        selected = selected[selected["window"].eq(window)]
    if transaction_type is not None:
        selected = selected[selected["transaction_type"].eq(transaction_type)]
    if position_group is not None:
        selected = selected[selected["position_label"].eq(position_group)]

    latest = selected.sort_values("transaction_date", ascending=False, kind="stable").head(limit)
    rows = latest.assign(
        position_group=latest["position_label"],
        transaction_date=latest["transaction_date"].dt.date,
        report_date=latest["report_date"].dt.date,
    )
    return [Trade(**row) for row in _records(rows[list(Trade.model_fields)])]


@app.get("/windows")
def window_counts(trades: Trades) -> list[WindowCount]:
    """Antal affärer per fönster, befattning och typ. Även per dag, då fönstren är olika långa."""
    counts = (
        trades.groupby(["window", "position_label", "transaction_type"])
        .size()
        .rename("trades")
        .reset_index()
    )
    days = {name: last - first + 1 for name, (first, last) in WINDOWS.items()}
    return [
        WindowCount(
            window=row["window"],
            position_group=row["position_label"],
            transaction_type=row["transaction_type"],
            trades=row["trades"],
            trades_per_window_day=row["trades"] / days[row["window"]],
        )
        for row in counts.to_dict("records")
    ]


@app.get("/companies/{company}/summary")
def company_summary(company: str, session: Annotated[Session, Depends(get_db)]) -> SummaryOut:
    """Bolagets senaste AI-sammanfattning. Nya skapas i dashboarden, aldrig via API:t."""
    summary = latest_summary(session, company)
    if summary is None:
        raise HTTPException(status_code=404, detail="Det finns ingen sammanfattning för bolaget.")
    return SummaryOut(
        company=company,
        text=summary.summary_text,
        model=summary.model_used,
        created_at=summary.created_at,
    )
