"""Kopplingen till databasen."""

import os
from collections.abc import Iterator
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from insider_tracker.db.models import Base

DEFAULT_DATABASE_URL = "sqlite:///./insider_tracker.db"


def get_engine(url: str | None = None) -> Engine:
    """Öppnar databasen och skapar tabellerna om de inte redan finns."""
    load_dotenv()
    engine = create_engine(url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL))
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def get_session(engine: Engine) -> Iterator[Session]:
    """Ger en session som sparar när allt gått bra och ångrar allt om något går fel."""
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
