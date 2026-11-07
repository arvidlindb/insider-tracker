"""Databasens tabeller."""

from datetime import UTC, date, datetime

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Company(Base):
    """Ett bolag i FI:s register. Vi känner igen det på LEI-koden, eller på namnet om den saknas."""

    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String, unique=True)
    name: Mapped[str]
    lei: Mapped[str | None]

    trades: Mapped[list["Trade"]] = relationship(back_populates="company")


class Trade(Base):
    """En insynsaffär. Fingeravtrycket hindrar att samma rad sparas två gånger."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    published_at: Mapped[datetime | None]
    transaction_date: Mapped[date] = mapped_column(index=True)
    person_name: Mapped[str | None]
    position: Mapped[str | None]
    is_related: Mapped[bool]
    is_correction: Mapped[bool]
    is_share_program: Mapped[bool]
    transaction_type: Mapped[str | None]
    instrument_type: Mapped[str | None]
    instrument_name: Mapped[str | None]
    isin: Mapped[str | None] = mapped_column(index=True)
    volume: Mapped[float | None]
    volume_unit: Mapped[str | None]
    price: Mapped[float | None]
    currency: Mapped[str | None]
    venue: Mapped[str | None]
    source_url: Mapped[str]

    company: Mapped[Company] = relationship(back_populates="trades")


class Report(Base):
    """En rapportdag. Sparas här, så att historiken finns kvar även om källan slutar visa den."""

    __tablename__ = "reports"
    __table_args__ = (UniqueConstraint("source_company_id", "report_date", "report_type"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    source_company_id: Mapped[str] = mapped_column(String, index=True)
    orderbook_id: Mapped[str]
    name: Mapped[str | None]
    isin: Mapped[str | None]
    ticker: Mapped[str | None]
    market_list: Mapped[str | None]
    report_date: Mapped[date]
    report_type: Mapped[str]
    is_confirmed: Mapped[bool]
    source_url: Mapped[str]


class Summary(Base):
    """En AI-genererad sammanfattning. Modellen sparas, så att man ser vad som skrev texten."""

    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    summary_text: Mapped[str]
    model_used: Mapped[str]
    trades_used: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))


class FetchLog(Base):
    """Vad som hämtades och när, så att man i efterhand kan se vad systemet visste."""

    __tablename__ = "fetch_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    fetched_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))
    from_date: Mapped[date]
    to_date: Mapped[date]
    rows_fetched: Mapped[int]
    rows_new: Mapped[int]
