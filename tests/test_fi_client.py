from datetime import date

import httpx
import pytest
from tenacity import wait_none

from insider_tracker.providers import fi_client
from insider_tracker.providers.fi_client import (
    download_csv,
    fetch_report_windows,
    fetch_trades,
    fetch_trades_in_chunks,
    parse_csv,
)

HEADER = (
    "Publiceringsdatum;Emittent;LEI-kod;Anmälningsskyldig;Person i ledande ställning;Befattning;"
    "Närstående;Korrigering;Beskrivning av korrigering;Är förstagångsrapportering;"
    "Är kopplad till aktieprogram;Karaktär;Instrumenttyp;Instrumentnamn;ISIN;Transaktionsdatum;"
    "Volym;Volymsenhet;Pris;Valuta;Handelsplats;Status;"
)


def fake_csv(*transaction_dates: str) -> bytes:
    """Bygger en CSV i FI:s format, med fejkade värden och ett extra semikolon per rad."""
    rows = [
        f"2026-11-03 17:44:26;Testbolaget AB;LEI1;Fejk Fejksson;Fejk Fejksson;VD;;;;Ja;Nej;Förvärv;"
        f"Aktie;Testbolaget B;SE0000000001;{day} 00:00:00;100,0;Antal;23,50;SEK;XSTO;Aktuell;"
        for day in transaction_dates
    ]
    return "\r\n".join([HEADER, *rows]).encode("utf-16-le")


def client_with(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_parse_csv_reads_fi_format_and_drops_empty_last_column():
    trades = parse_csv(fake_csv("2026-11-02"))

    assert len(trades.columns) == 22
    assert trades.loc[0, "Karaktär"] == "Förvärv"
    assert trades.loc[0, "Pris"] == "23,50"


def test_parse_csv_rejects_response_that_is_not_fi_csv():
    with pytest.raises(ValueError):
        parse_csv("<html>Fel</html>".encode("utf-16-le"))


def test_fetch_trades_sends_dates_and_keeps_source_url():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["SearchFunctionType"] == "Insyn"
        assert request.url.params["Publiceringsdatum.From"] == "2026-11-01"
        assert request.url.params["Publiceringsdatum.To"] == "2026-11-03"
        assert request.url.params["button"] == "export"
        return httpx.Response(200, content=fake_csv("2026-11-02"))

    trades = fetch_trades(client_with(handler), date(2026, 11, 1), date(2026, 11, 3))

    assert len(trades) == 1
    assert trades.loc[0, "source_url"].startswith(fi_client.EXPORT_URL)


def test_fetch_trades_splits_interval_when_row_limit_is_reached(monkeypatch):
    monkeypatch.setattr(fi_client, "MAX_ROWS_PER_SEARCH", 2)

    def handler(request: httpx.Request) -> httpx.Response:
        first = request.url.params["Publiceringsdatum.From"]
        last = request.url.params["Publiceringsdatum.To"]
        if first != last:
            return httpx.Response(200, content=fake_csv(first, first))
        return httpx.Response(200, content=fake_csv(first))

    trades = fetch_trades(client_with(handler), date(2026, 11, 1), date(2026, 11, 2))

    assert len(trades) == 2
    assert trades["Transaktionsdatum"].str[:10].tolist() == ["2026-11-01", "2026-11-02"]


def test_fetch_trades_raises_when_a_single_day_hits_row_limit(monkeypatch):
    monkeypatch.setattr(fi_client, "MAX_ROWS_PER_SEARCH", 1)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=fake_csv("2026-11-01"))

    with pytest.raises(ValueError):
        fetch_trades(client_with(handler), date(2026, 11, 1), date(2026, 11, 1))


def test_chunks_cover_the_whole_period_without_overlap():
    searched = []

    def handler(request: httpx.Request) -> httpx.Response:
        first = request.url.params["Publiceringsdatum.From"]
        searched.append((first, request.url.params["Publiceringsdatum.To"]))
        return httpx.Response(200, content=fake_csv(first))

    trades = fetch_trades_in_chunks(
        client_with(handler), date(2026, 1, 1), date(2026, 1, 31), chunk_days=14, delay=0
    )

    assert searched == [
        ("2026-01-01", "2026-01-14"),
        ("2026-01-15", "2026-01-28"),
        ("2026-01-29", "2026-01-31"),
    ]
    assert len(trades) == 3


def test_report_windows_filter_on_transaction_date():
    searched = []

    def handler(request: httpx.Request) -> httpx.Response:
        first = request.url.params["Publiceringsdatum.From"]
        searched.append((first, request.url.params["Publiceringsdatum.To"]))
        if first == "2026-09-02":
            # Första och sista dagen är med. Dagen efter är handelsstoppet redan igång.
            return httpx.Response(200, content=fake_csv("2026-09-02", "2026-09-15", "2026-09-16"))
        return httpx.Response(200, content=fake_csv("2026-10-16", "2026-10-30", "2026-10-31"))

    trades = fetch_report_windows(client_with(handler), date(2026, 10, 16))

    # Sökningen går en vecka längre än fönstret, eftersom affärer publiceras i efterhand.
    assert searched == [("2026-09-02", "2026-09-22"), ("2026-10-16", "2026-11-06")]

    days = trades.groupby("window")["Transaktionsdatum"].apply(lambda s: s.str[:10].tolist())
    assert days["before"] == ["2026-09-02", "2026-09-15"]
    assert days["after"] == ["2026-10-16", "2026-10-30"]


def test_download_retries_temporary_error(monkeypatch):
    monkeypatch.setattr(download_csv.retry, "wait", wait_none())
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, content=fake_csv("2026-11-02"))

    response = download_csv(client_with(handler), date(2026, 11, 1), date(2026, 11, 2))

    assert response.status_code == 200
    assert len(calls) == 2


def test_download_does_not_retry_permanent_error(monkeypatch):
    monkeypatch.setattr(download_csv.retry, "wait", wait_none())
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        download_csv(client_with(handler), date(2026, 11, 1), date(2026, 11, 2))

    assert len(calls) == 1
