from datetime import UTC, datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from insider_tracker.api.main import MAX_LIMIT, app, get_db, get_trades
from insider_tracker.db.database import get_session
from insider_tracker.db.models import Base, Company, Summary


def fake_trade(**changes) -> dict:
    return {
        "transaction_date": pd.Timestamp("2026-10-19"),
        "company": "Testbolaget AB",
        "person_name": "Fejk Fejksson",
        "window": "after",
        "days_from_report": 3,
        "report_date": pd.Timestamp("2026-10-16"),
        "position": "Styrelseledamot",
        "position_label": "Styrelse",
        "is_related": False,
        "transaction_type": "Förvärv",
        "volume": 30_000.0,
        "price": 100.0,
        "currency": "SEK",
        "amount_sek": 3_000_000.0,
        "uncertain_reason": "",
        "source_url": "https://example.test/export",
        **changes,
    }


@pytest.fixture
def client():
    trades = pd.DataFrame(
        [
            fake_trade(),
            fake_trade(transaction_date=pd.Timestamp("2026-10-20")),
            fake_trade(transaction_type="Avyttring", currency="EUR", amount_sek=None),
            fake_trade(
                company="Annat AB",
                window="before",
                days_from_report=-40,
                position="Verkställande direktör (VD)",
                position_label="Vd",
            ),
        ]
    )
    # FastAPI kör anropen i en annan tråd. En databas i minnet måste därför dela en enda
    # anslutning mellan trådarna, annars ser anropet en tom databas.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with get_session(engine) as session:
        company = Company(key="LEI1", name="Testbolaget AB", lei="LEI1")
        session.add(company)
        session.flush()
        session.add(
            Summary(
                company_id=company.id,
                summary_text="Fejkad sammanfattning.",
                model_used="claude-haiku-5-5",
                trades_used=3,
                created_at=datetime(2026, 11, 1, tzinfo=UTC),
            )
        )

    def fake_db():
        with get_session(engine) as session:
            yield session

    # Dependency injection: API:t får fejkad data i stället för den riktiga databasen.
    app.dependency_overrides[get_trades] = lambda: trades
    app.dependency_overrides[get_db] = fake_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_companies_are_sorted_by_number_of_trades(client):
    companies = client.get("/companies").json()

    assert companies == [{"name": "Testbolaget AB", "trades": 3}, {"name": "Annat AB", "trades": 1}]


def test_trades_are_latest_first_and_contain_no_person_names(client):
    response = client.get("/trades")

    trades = response.json()
    assert trades[0]["transaction_date"] == "2026-10-20"
    assert "person_name" not in trades[0]
    assert "Fejk Fejksson" not in response.text


def test_trades_can_be_filtered(client):
    by_company = client.get("/trades", params={"company": "annat ab"}).json()
    sales = client.get("/trades", params={"transaction_type": "Avyttring"}).json()
    before = client.get("/trades", params={"window": "before"}).json()
    ceo = client.get("/trades", params={"position_group": "Vd"}).json()

    assert [trade["company"] for trade in by_company] == ["Annat AB"]
    assert len(sales) == 1
    assert sales[0]["amount_sek"] is None
    assert len(before) == 1
    assert len(ceo) == 1


def test_limit_has_a_ceiling(client):
    assert len(client.get("/trades", params={"limit": 1}).json()) == 1
    assert client.get("/trades", params={"limit": MAX_LIMIT + 1}).status_code == 422
    assert client.get("/trades", params={"limit": 0}).status_code == 422


def test_invalid_filter_value_is_rejected(client):
    assert client.get("/trades", params={"window": "under"}).status_code == 422


def test_window_counts_include_trades_per_day(client):
    counts = client.get("/windows").json()

    after_buys = next(
        row for row in counts
        if row["window"] == "after" and row["transaction_type"] == "Förvärv"
    )
    assert after_buys["trades"] == 2
    assert after_buys["trades_per_window_day"] == pytest.approx(2 / 15)


def test_summary_is_returned_and_missing_summary_gives_404(client):
    found = client.get("/companies/Testbolaget AB/summary")
    missing = client.get("/companies/Annat AB/summary")

    assert found.status_code == 200
    assert found.json()["text"] == "Fejkad sammanfattning."
    assert found.json()["model"] == "claude-haiku-5-5"
    assert missing.status_code == 404
