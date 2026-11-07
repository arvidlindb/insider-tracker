"""Tar fram data och diagram till dashboarden. Själva sidan ligger i app.py.

Allt som går att testa utan webbläsare ligger här. Personnamn tas bort direkt när datan läses,
så att de aldrig kan hamna på sidan.
"""

import altair as alt
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from insider_tracker.db.models import Company, Summary
from insider_tracker.db.queries import read_reports, read_trades
from insider_tracker.pipeline.analysis import position_group
from insider_tracker.pipeline.filters import filter_trades
from insider_tracker.pipeline.report_windows import match_trades_to_windows
from insider_tracker.providers.fi_client import WINDOWS

POSITION_LABELS = {"ceo": "Vd", "cfo": "Ekonomichef", "board": "Styrelse", "other": "Övriga"}
WINDOW_LABELS = {"before": "Före stoppet", "after": "Efter rapporten"}

BUY = "Förvärv"
SALE = "Avyttring"

# Färgerna är kontrollerade med valideringsverktyget, i både ljust och mörkt läge.
TYPE_COLORS = {BUY: "#2a78d6", SALE: "#d95926"}

FIRST_DAY = min(first for first, _ in WINDOWS.values())
LAST_DAY = max(last for _, last in WINDOWS.values())
TRADING_STOP = {"start": -30, "end": -1}

TABLE_COLUMNS = {
    "transaction_date": "Datum",
    "company": "Bolag",
    "window_label": "Fönster",
    "days_from_report": "Dag",
    "position_label": "Befattning",
    "is_related": "Närstående",
    "transaction_type": "Typ",
    "volume": "Volym",
    "price": "Pris",
    "currency": "Valuta",
    "amount_sek": "Belopp (SEK)",
    "uncertain_reason": "Osäkerhet",
    "source_url": "Källa",
}


def prepare_trades(matched: pd.DataFrame) -> pd.DataFrame:
    """Tar bort personnamn och lägger till svenska etiketter."""
    return matched.drop(columns="person_name").assign(
        position_label=matched["position"].map(position_group).map(POSITION_LABELS),
        window_label=matched["window"].map(WINDOW_LABELS),
    )


def load_trades(session: Session) -> pd.DataFrame:
    """Läser databasen, filtrerar och placerar affärerna i fönster."""
    reports = read_reports(session)
    filtered, _ = filter_trades(read_trades(session), reports)
    return prepare_trades(match_trades_to_windows(filtered, reports))


def window_stats(trades: pd.DataFrame) -> dict:
    """Räknar köp per dag i varje fönster, eftersom fönstren är olika långa."""
    buys = trades[trades["transaction_type"].eq(BUY)]
    stats = {}
    for window, (first_day, last_day) in WINDOWS.items():
        days = last_day - first_day + 1
        stats[window] = buys["window"].eq(window).sum() / days
    return stats


def count_by_day(trades: pd.DataFrame) -> pd.DataFrame:
    """Räknar affärer per dag från rapporten och typ av affär."""
    return (
        trades.groupby(["days_from_report", "transaction_type"])
        .size()
        .rename("trades")
        .reset_index()
    )


def table_rows(trades: pd.DataFrame) -> pd.DataFrame:
    """Gör tabellen med svenska rubriker, de senaste affärerna först."""
    latest = trades.sort_values("transaction_date", ascending=False, kind="stable")
    return latest[list(TABLE_COLUMNS)].rename(columns=TABLE_COLUMNS)


def latest_summary(session: Session, company: str) -> Summary | None:
    """Hämtar bolagets senaste sparade sammanfattning, om det finns någon."""
    statement = (
        select(Summary)
        .join(Company, Summary.company_id == Company.id)
        .where(Company.name == company)
        .order_by(Summary.created_at.desc(), Summary.id.desc())
    )
    return session.scalars(statement).first()


def _day_panel(counts: pd.DataFrame, transaction_type: str, title: str) -> alt.LayerChart:
    """Bygger en panel. Köp och försäljningar får varsin, så att staplarna kan jämföras rakt av."""
    data = counts[counts["transaction_type"].eq(transaction_type)]
    x_scale = alt.Scale(domain=[FIRST_DAY - 1, LAST_DAY + 1], nice=False)

    def day_axis(field: str) -> alt.X:
        # Alla lager delar samma axel, så att fältet och staplarna hamnar på rätt ställe.
        return alt.X(field, scale=x_scale, title="Dagar från rapporten")

    # Det grå fältet visar handelsstoppet, så att det inte ser ut som att datan saknas där.
    stop = (
        alt.Chart(pd.DataFrame([TRADING_STOP]))
        .mark_rect(color="#8a8a86", opacity=0.14)
        .encode(x=day_axis("start:Q"), x2="end:Q")
    )
    stop_label = (
        alt.Chart(pd.DataFrame([{"day": -15.5, "text": "Handelsstopp"}]))
        .mark_text(color="#8a8a86", baseline="top", dy=4)
        .encode(x=day_axis("day:Q"), y=alt.value(0), text="text:N")
    )
    report_line = (
        alt.Chart(pd.DataFrame([{"day": 0}]))
        .mark_rule(color="#8a8a86", strokeDash=[4, 3])
        .encode(x=day_axis("day:Q"))
    )
    bars = (
        alt.Chart(data)
        .mark_bar(color=TYPE_COLORS[transaction_type], size=7, cornerRadiusEnd=4)
        .encode(
            x=day_axis("days_from_report:Q"),
            y=alt.Y("trades:Q", title="Antal affärer", axis=alt.Axis(tickMinStep=1)),
            tooltip=[
                alt.Tooltip("days_from_report:Q", title="Dag från rapporten"),
                alt.Tooltip("transaction_type:N", title="Typ"),
                alt.Tooltip("trades:Q", title="Antal"),
            ],
        )
    )
    return alt.layer(stop, stop_label, report_line, bars).properties(height=170, title=title)


def build_day_chart(trades: pd.DataFrame) -> alt.VConcatChart:
    """Visar köp och försäljningar per dag från rapporten, på samma skala."""
    counts = count_by_day(trades)
    return alt.vconcat(
        _day_panel(counts, BUY, "Köp"),
        _day_panel(counts, SALE, "Försäljningar"),
    ).resolve_scale(y="shared")
