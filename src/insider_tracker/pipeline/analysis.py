"""Räknar köp och försäljningar före och efter rapporter, per befattning.

Resultatet visar när affärer skett, inte varför. Fönstren är också olika långa, och därför
räknas antalet även om till affärer per dag.

Kör så här:
    python -m insider_tracker.pipeline.analysis
"""

from pathlib import Path

import pandas as pd

from insider_tracker.db.database import get_engine, get_session
from insider_tracker.db.queries import read_reports, read_trades
from insider_tracker.pipeline.filters import filter_trades
from insider_tracker.pipeline.report_windows import assign_company_ids, match_trades_to_windows
from insider_tracker.providers.fi_client import WINDOWS

OUTPUT_PATH = Path("data/window_analysis.csv")
GROUP_COLUMNS = ["window", "position_group", "transaction_type"]


def position_group(position: object) -> str:
    """Delar in FI:s befattningar i grupper."""
    text = str(position).lower()
    # Dessa titlar innehåller ord som annars skulle hamna i fel grupp.
    if "arbetstagar" in text or "vice" in text or "suppleant" in text:
        return "other"
    if "verkställande direktör" in text or "(vd)" in text:
        return "ceo"
    if "ekonomichef" in text or "finanschef" in text or "finansdirektör" in text:
        return "cfo"
    if "styrelse" in text:
        return "board"
    return "other"


def summarize_windows(trades: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """Filtrerar affärerna, placerar dem i fönster och räknar ihop dem per grupp.

    Beloppet räknar bara affärer i SEK. Hur många som saknar belopp syns i uncertain_trades.
    """
    filtered, _ = filter_trades(trades, reports)
    matched = match_trades_to_windows(filtered, reports)
    matched = matched.assign(
        position_group=matched["position"].map(position_group),
        is_uncertain=matched["uncertain_reason"].ne(""),
    )

    summary = (
        matched.groupby(GROUP_COLUMNS)
        .agg(
            trades=("is_uncertain", "size"),
            persons=("person_name", "nunique"),
            companies=("company", "nunique"),
            amount_sek=("amount_sek", "sum"),
            uncertain_trades=("is_uncertain", "sum"),
        )
        .reset_index()
    )

    window_days = {name: last - first + 1 for name, (first, last) in WINDOWS.items()}
    # Är tabellen tom tror pandas att kolumnen är text, så datatypen anges uttryckligen.
    days = summary["window"].map(window_days).astype(float)
    summary["trades_per_window_day"] = summary["trades"] / days
    return summary


def coverage(trades: pd.DataFrame, reports: pd.DataFrame) -> float:
    """Andelen affärer som går att koppla till ett bolag med rapportdatum."""
    if trades.empty:
        return 0.0
    return assign_company_ids(trades, reports)["company_id"].notna().mean()


def main() -> None:
    with get_session(get_engine()) as session:
        trades = read_trades(session)
        reports = read_reports(session)

    summary = summarize_windows(trades, reports)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print(f"Affärer i databasen: {len(trades)}")
    print(f"Bolag med rapportdatum: {reports['company_id'].nunique()}")
    print(f"Täckning: {coverage(trades, reports):.0%} av affärerna har ett bolag med rapportdatum")
    print("\nDatan visar när affärer skett, inte varför.\n")
    print(summary.round(2).to_string(index=False))
    print(f"\nSparat i {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
