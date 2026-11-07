"""Hämtar insynsaffärer från FI:s register som pandas-tabeller.

Fönstren runt en rapport räknas i dagar, där rapportdagen är dag 0:
    före:  dag -44 till -31, alltså två veckor innan handelsstoppet börjar
    efter: dag 0 till +14

Här hämtas bara rådata. Allt läses in som text och görs om till tal och datum i parsern.

Kör så här:
    python -m insider_tracker.providers.fi_client --report-date 2026-10-16
"""

import argparse
import io
import time
from datetime import date, timedelta

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

EXPORT_URL = "https://marknadssok.fi.se/Publiceringsklient/sv-SE/Search/Search"
USER_AGENT = "insider-tracker/0.1"
MAX_ROWS_PER_SEARCH = 1000

WINDOWS = {"before": (-44, -31), "after": (0, 14)}

# FI låter oss söka på publiceringsdatum, men fönstren gäller när affären gjordes.
# Affärer publiceras efter att de gjorts, så vi söker en vecka längre och filtrerar sedan.
# En affär som anmälts ännu senare än så missas.
PUBLICATION_MARGIN_DAYS = 7

TEMPORARY_STATUS_CODES = {429, 500, 502, 503, 504}

# Kolumnerna som CLI:t skriver ut. Personnamn lämnas med flit utanför.
DISPLAY_COLUMNS = [
    "window",
    "Emittent",
    "Karaktär",
    "Transaktionsdatum",
    "Volym",
    "Pris",
    "Valuta",
]


def is_temporary_error(error: BaseException) -> bool:
    """Svarar på om felet kan gå över av sig självt, så att ett nytt försök är rimligt."""
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in TEMPORARY_STATUS_CODES
    return isinstance(error, httpx.TransportError)


@retry(
    retry=retry_if_exception(is_temporary_error),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=5, max=60),
    reraise=True,
)
def download_csv(client: httpx.Client, from_date: date, to_date: date) -> httpx.Response:
    """Laddar ner FI:s export för ett intervall av publiceringsdatum."""
    params = {
        "SearchFunctionType": "Insyn",
        "Publiceringsdatum.From": from_date.isoformat(),
        "Publiceringsdatum.To": to_date.isoformat(),
        "button": "export",
    }
    response = client.get(EXPORT_URL, params=params)
    response.raise_for_status()
    return response


def parse_csv(content: bytes) -> pd.DataFrame:
    """Läser FI:s CSV till en tabell där alla värden fortfarande är text."""
    # Filen saknar BOM, så kodningen måste anges exakt. Vanlig "utf-16" räcker inte i pandas.
    trades = pd.read_csv(io.BytesIO(content), encoding="utf-16-le", sep=";", dtype=str)

    # Varje rad slutar med ett extra semikolon, vilket ger en tom kolumn på slutet.
    trades = trades.loc[:, ~trades.columns.str.startswith("Unnamed")]

    # Om FI skickar en felsida i stället för CSV vill vi märka det direkt.
    if "Transaktionsdatum" not in trades.columns:
        raise ValueError("Svaret från FI ser inte ut som insynsregistrets CSV.")
    return trades


def fetch_trades(client: httpx.Client, from_date: date, to_date: date) -> pd.DataFrame:
    """Hämtar alla affärer som publicerats mellan två datum, båda inräknade.

    FI ger högst 1000 rader per sökning. Får vi så många vet vi inte om något saknas,
    så då delar vi intervallet på mitten och hämtar halvorna var för sig.
    """
    response = download_csv(client, from_date, to_date)
    trades = parse_csv(response.content)
    trades["source_url"] = str(response.url)

    if len(trades) < MAX_ROWS_PER_SEARCH:
        return trades

    if from_date == to_date:
        raise ValueError(f"FI:s radtak nåddes för en enda dag: {from_date}")

    middle = from_date + (to_date - from_date) // 2
    first_half = fetch_trades(client, from_date, middle)
    second_half = fetch_trades(client, middle + timedelta(days=1), to_date)
    return pd.concat([first_half, second_half], ignore_index=True)


def fetch_trades_in_chunks(
    client: httpx.Client,
    from_date: date,
    to_date: date,
    chunk_days: int = 14,
    delay: float = 2.0,
) -> pd.DataFrame:
    """Hämtar en längre period i små bitar, med en paus mellan anropen.

    Två veckor ryms normalt under radtaket, så varje anrop blir litet och FI belastas mindre.
    """
    frames = []
    start = from_date
    while start <= to_date:
        end = min(start + timedelta(days=chunk_days - 1), to_date)
        frames.append(fetch_trades(client, start, end))
        start = end + timedelta(days=1)
        if start <= to_date:
            time.sleep(delay)
    return pd.concat(frames, ignore_index=True)


def fetch_report_windows(client: httpx.Client, report_date: date) -> pd.DataFrame:
    """Hämtar affärerna i båda fönstren runt en rapportdag och märker dem med fönstrets namn.

    Tabellen innehåller alla bolag som handlat under fönstren, inte bara bolaget som
    rapporterade den dagen.
    """
    frames = []
    for window, (first_day, last_day) in WINDOWS.items():
        start = report_date + timedelta(days=first_day)
        end = report_date + timedelta(days=last_day)

        trades = fetch_trades(client, start, end + timedelta(days=PUBLICATION_MARGIN_DAYS))

        # Här är det affärens eget datum som räknas, inte när den publicerades.
        # Ett ogiltigt datum blir tomt och hamnar därför utanför fönstret.
        transaction_dates = pd.to_datetime(trades["Transaktionsdatum"], errors="coerce")
        in_window = transaction_dates.dt.normalize().between(pd.Timestamp(start), pd.Timestamp(end))
        frames.append(trades[in_window].assign(window=window))

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hämtar insynsaffärer runt en rapportdag.")
    parser.add_argument("--report-date", type=date.fromisoformat, required=True)
    args = parser.parse_args()

    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(headers=headers, timeout=60) as client:
        trades = fetch_report_windows(client, args.report_date)

    print(trades["window"].value_counts().to_string())
    latest = trades.sort_values("Transaktionsdatum", ascending=False).head(10)
    print(latest[DISPLAY_COLUMNS].to_string(index=False))


if __name__ == "__main__":
    main()
