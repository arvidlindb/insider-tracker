"""Dashboard över insynsaffärer runt bolagens rapporter.

Kör så här:
    streamlit run src/insider_tracker/ui/app.py
"""

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from insider_tracker.ai.summarizer import SummaryError, save_summary, summarize_company
from insider_tracker.db.database import get_engine, get_session
from insider_tracker.ui.dashboard_data import (
    BUY,
    POSITION_LABELS,
    SALE,
    WINDOW_LABELS,
    build_day_chart,
    latest_summary,
    load_trades,
    table_rows,
    window_stats,
)

ALL_COMPANIES = "Alla bolag"


@st.cache_resource
def engine():
    return get_engine()


@st.cache_data(ttl=600)
def cached_trades() -> pd.DataFrame:
    """Sparar resultatet en stund, eftersom Streamlit kör om hela filen vid varje klick."""
    with get_session(engine()) as session:
        return load_trades(session)


def show_summary(company: str, trades: pd.DataFrame) -> None:
    """Visar bolagets sammanfattning och låter användaren skapa en ny."""
    st.subheader("AI-sammanfattning")

    if st.button("Skapa ny sammanfattning", help="Gör ett betalt anrop till Claude."):
        load_dotenv()
        try:
            with st.spinner("Claude skriver..."):
                result = summarize_company(anthropic.Anthropic(), trades, company)
            with get_session(engine()) as session:
                save_summary(session, company, result)
        except anthropic.AuthenticationError:
            st.error("API-nyckeln godkändes inte. Kontrollera .env.")
        except anthropic.RateLimitError:
            st.error("För många anrop just nu, eller slut på krediter. Försök igen senare.")
        except (anthropic.AnthropicError, SummaryError) as error:
            st.error(f"Kunde inte skapa sammanfattningen: {type(error).__name__}")

    with get_session(engine()) as session:
        summary = latest_summary(session, company)
        if summary is None:
            st.info("Det finns ingen sammanfattning av det här bolaget än.")
        else:
            st.markdown(summary.summary_text)
            st.caption(f"Modell: {summary.model_used}. Skapad: {summary.created_at:%Y-%m-%d %H:%M}.")


def main() -> None:
    st.set_page_config(page_title="Insider Tracker", layout="wide")
    st.title("Insider Tracker")
    st.caption(
        "Insynsaffärer från FI:s register runt bolagens rapporter. "
        "Datan visar när affärer skett, inte varför. Inga investeringsråd."
    )

    trades = cached_trades()
    if trades.empty:
        st.warning("Det finns inga affärer. Kör loader-kommandot först.")
        return

    companies = sorted(trades["company"].dropna().unique())
    with st.sidebar:
        st.header("Filter")
        company = st.selectbox("Bolag", [ALL_COMPANIES, *companies])
        positions = st.multiselect(
            "Befattning", list(POSITION_LABELS.values()), default=list(POSITION_LABELS.values())
        )
        types = st.multiselect("Typ", [BUY, SALE], default=[BUY, SALE])
        windows = st.multiselect(
            "Fönster", list(WINDOW_LABELS.values()), default=list(WINDOW_LABELS.values())
        )
        include_related = st.checkbox("Ta med närstående", value=True)
        include_uncertain = st.checkbox("Ta med osäkra rader", value=True)

    selected = trades[
        trades["position_label"].isin(positions)
        & trades["transaction_type"].isin(types)
        & trades["window_label"].isin(windows)
    ]
    if company != ALL_COMPANIES:
        selected = selected[selected["company"].eq(company)]
    if not include_related:
        selected = selected[~selected["is_related"]]
    if not include_uncertain:
        selected = selected[selected["uncertain_reason"].eq("")]

    stats = window_stats(selected)
    columns = st.columns(5)
    columns[0].metric("Affärer", f"{len(selected):,}".replace(",", " "))
    columns[1].metric("Köp", int(selected["transaction_type"].eq(BUY).sum()))
    columns[2].metric("Försäljningar", int(selected["transaction_type"].eq(SALE).sum()))
    columns[3].metric("Köp per dag före", f"{stats['before']:.2f}".replace(".", ","))
    columns[4].metric("Köp per dag efter", f"{stats['after']:.2f}".replace(".", ","))
    st.caption(
        "Fönstren är dag −44 till −31 och dag 0 till +14 runt rapporten. Dag 0 är osäker, eftersom "
        "affären kan ha gjorts före publiceringen. En affär kan räknas mot två rapporter."
    )

    if selected.empty:
        st.info("Inga affärer matchar filtren.")
    else:
        st.subheader("Affärer per dag från rapporten")
        st.altair_chart(build_day_chart(selected), width="stretch")

        st.subheader("Affärerna")
        st.dataframe(
            table_rows(selected),
            hide_index=True,
            width="stretch",
            column_config={
                "Datum": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "Volym": st.column_config.NumberColumn(format="localized"),
                "Pris": st.column_config.NumberColumn(format="%.2f"),
                "Belopp (SEK)": st.column_config.NumberColumn(format="localized"),
                "Källa": st.column_config.LinkColumn(display_text="FI"),
            },
        )

    if company != ALL_COMPANIES:
        # Sammanfattningen bygger på alla bolagets affärer, oavsett filtren ovan.
        show_summary(company, trades[trades["company"].eq(company)])


if __name__ == "__main__":
    main()
