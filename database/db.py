"""Database layer for AnchorIQ using SQLAlchemy.
Connects to Supabase PostgreSQL in production, and isolated in-memory SQLite under tests.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    create_engine,
)
from sqlalchemy.exc import ArgumentError, NoSuchModuleError, SQLAlchemyError
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import StaticPool

# Load local environment configuration if .env exists
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def get_database_url() -> str:
    """Return in-memory SQLite URL during tests, or the configured PostgreSQL connection string."""
    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING"):
        return "sqlite:///:memory:"

    database_url = (
        os.getenv("SUPABASE_DB_URL", "").strip()
        or os.getenv("SUPABASE_DATABASE_URL", "").strip()
        or os.getenv("DATABASE_URL", "").strip()
    )
    if not database_url:
        raise RuntimeError("Missing required SUPABASE_DB_URL environment variable for database connection.")

    try:
        from sqlalchemy.engine import make_url

        parsed = make_url(database_url)
        # Ensure psycopg driver is used for modern PostgreSQL connections
        if parsed.drivername.lower() in {"postgres", "postgresql"}:
            parsed = parsed.set(drivername="postgresql+psycopg")
        if parsed.drivername.startswith("postgresql") and not parsed.query.get("sslmode"):
            parsed = parsed.update_query_dict({"sslmode": "require"})
        return parsed.render_as_string(hide_password=False)
    except Exception as exc:
        raise ArgumentError(f"Invalid database URL: {exc}") from exc


Base = declarative_base()


class Run(Base):
    """Represents a single business analysis run, scoped to an authenticated user."""
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, nullable=True, index=True)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    target = Column(String, nullable=True)
    problem_type = Column(String, nullable=True)
    model_card = Column(JSON, default=dict, nullable=True)

    charts = relationship("Chart", back_populates="run", cascade="all, delete-orphan")
    insights = relationship("Insight", back_populates="run", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="run", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="run", cascade="all, delete-orphan")


class Chart(Base):
    """Represents a chart artifact associated with an analysis run."""
    __tablename__ = "charts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    title = Column(String, nullable=False)
    cloudinary_url = Column(String, nullable=True)
    cloudinary_public_id = Column(String, nullable=True)
    fact_card = Column(JSON, default=dict, nullable=True)

    run = relationship("Run", back_populates="charts")


class Insight(Base):
    """Represents an AI or rule-based business insight for a run."""
    __tablename__ = "insights"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    text = Column(String, nullable=False)
    source = Column(String, nullable=False, default="rules")

    run = relationship("Run", back_populates="insights")


class ChatMessage(Base):
    """Represents a conversation turn with AnchorIQ's business assistant."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    run = relationship("Run", back_populates="chat_messages")


class Prediction(Base):
    """Represents an ad-hoc prediction evaluation against a run's model."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    inputs = Column(JSON, default=dict, nullable=True)
    output = Column(JSON, default=dict, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    run = relationship("Run", back_populates="predictions")


def _create_engine(database_url: str):
    """Create a thread-safe SQLAlchemy engine configured for testing or pooling."""
    if database_url.startswith("sqlite:///:memory:"):
        return create_engine(
            database_url,
            future=True,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    return create_engine(database_url, future=True, pool_pre_ping=True, pool_recycle=300)


database_url = get_database_url()
_test_database = database_url.startswith("sqlite:///:memory:")

engine = _create_engine(database_url)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def with_database_fallback(function):
    """Decorator maintaining backward-compatible helper execution."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        return function(*args, **kwargs)
    return wrapped


def get_database_status() -> Dict[str, Any]:
    """Check whether the active database backend is connected and return its status."""
    backend = "sqlite-test" if _test_database else "supabase-postgres"
    connected = False
    try:
        with engine.connect() as conn:
            connected = True
    except Exception:
        connected = False
    return {
        "backend": backend,
        "supabase_configured": not _test_database,
        "connected": connected,
    }


def initialize_database() -> None:
    
    Base.metadata.create_all(bind=engine)

    
    if not _test_database:
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE runs ADD COLUMN IF NOT EXISTS user_id VARCHAR(64);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_runs_user_id ON runs(user_id);"))
                conn.commit()
        except Exception:
            pass


def build_fact_card(
    title: str,
    local_path: Optional[str] = None,
    numbers: Optional[List[str]] = None,
    finding: Optional[str] = None,
) -> Dict[str, Any]:
    """Helper to assemble a standardized fact card dict."""
    fact = {
        "title": title,
        "numbers": numbers or [],
        "finding": finding or "No summary available.",
    }
    if local_path:
        fact["local_path"] = local_path
    return fact


@with_database_fallback
def save_run_record(
    *,
    user_id: Optional[str] = None,
    filename: str,
    target: Optional[str] = None,
    problem_type: Optional[str] = None,
    model_card: Optional[Dict[str, Any]] = None,
    charts: Optional[List[Dict[str, Any]]] = None,
    insights: Optional[List[Dict[str, Any]]] = None,
    chat_messages: Optional[List[Dict[str, Any]]] = None,
    predictions: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Persist a completed analysis run and all related records with user ownership."""
    initialize_database()
    with SessionLocal() as session:
        run = Run(
            user_id=str(user_id) if user_id else None,
            filename=filename,
            target=target,
            problem_type=problem_type,
            model_card=model_card or {},
        )
        session.add(run)
        session.flush()

        for chart in charts or []:
            session.add(
                Chart(
                    run_id=run.id,
                    title=chart.get("title", "Chart"),
                    cloudinary_url=chart.get("cloudinary_url"),
                    cloudinary_public_id=chart.get("cloudinary_public_id"),
                    fact_card=chart.get("fact_card") or {},
                )
            )

        for insight in insights or []:
            session.add(
                Insight(
                    run_id=run.id,
                    text=insight.get("text", ""),
                    source=insight.get("source", "rules"),
                )
            )

        for message in chat_messages or []:
            session.add(
                ChatMessage(
                    run_id=run.id,
                    role=message.get("role", "user"),
                    content=message.get("content", ""),
                    created_at=message.get("created_at") or datetime.utcnow(),
                )
            )

        for prediction in predictions or []:
            session.add(
                Prediction(
                    run_id=run.id,
                    inputs=prediction.get("inputs"),
                    output=prediction.get("output"),
                    created_at=prediction.get("created_at") or datetime.utcnow(),
                )
            )

        session.commit()
        return int(run.id)


@with_database_fallback
def save_prediction_record(run_id: int, inputs: Dict[str, Any], output: Dict[str, Any]) -> int:
    """Persist a single prediction evaluation in the predictions table."""
    initialize_database()
    with SessionLocal() as session:
        pred = Prediction(
            run_id=run_id,
            inputs=inputs,
            output=output,
            created_at=datetime.utcnow(),
        )
        session.add(pred)
        session.commit()
        return int(pred.id)


def _chart_to_dict(chart: Chart) -> Dict[str, Any]:
    return {
        "id": chart.id,
        "title": chart.title,
        "cloudinary_url": chart.cloudinary_url,
        "cloudinary_public_id": chart.cloudinary_public_id,
        "fact_card": chart.fact_card or {},
    }


def _insight_to_dict(insight: Insight) -> Dict[str, Any]:
    return {
        "id": insight.id,
        "text": insight.text,
        "source": insight.source,
    }


def _chat_to_dict(message: ChatMessage) -> Dict[str, Any]:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


def _prediction_to_dict(prediction: Prediction) -> Dict[str, Any]:
    return {
        "id": prediction.id,
        "inputs": prediction.inputs or {},
        "output": prediction.output or {},
        "created_at": prediction.created_at.isoformat() if prediction.created_at else None,
    }


def _run_to_dict(run: Run) -> Dict[str, Any]:
    return {
        "id": run.id,
        "user_id": run.user_id,
        "filename": run.filename,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "target": run.target,
        "problem_type": run.problem_type,
        "model_card": run.model_card or {},
        "charts": [_chart_to_dict(chart) for chart in run.charts],
        "insights": [_insight_to_dict(insight) for insight in run.insights],
        "chat_messages": [_chat_to_dict(message) for message in run.chat_messages],
        "predictions": [_prediction_to_dict(prediction) for prediction in run.predictions],
    }


@with_database_fallback
def get_runs(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return analysis runs, optionally filtered by the owner's user_id."""
    initialize_database()
    with SessionLocal() as session:
        query = session.query(Run)
        if user_id is not None:
            query = query.filter(Run.user_id == str(user_id))
        runs = query.order_by(Run.created_at.desc()).all()
        return [_run_to_dict(run) for run in runs]


@with_database_fallback
def get_run(run_id: int, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch a run by ID; returns None if not found or if not owned by user_id."""
    initialize_database()
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        if not run:
            return None
        # Enforce strict user isolation: prevent cross-user access
        if user_id is not None and run.user_id and str(run.user_id) != str(user_id):
            return None
        return _run_to_dict(run)


@with_database_fallback
def delete_run_record(run_id: int, user_id: Optional[str] = None) -> bool:
    """Delete a run and associated Cloudinary assets; returns False if not found or unauthorized."""
    initialize_database()
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        if not run:
            return False
        if user_id is not None and run.user_id and str(run.user_id) != str(user_id):
            return False

        # Clean up Cloudinary chart images
        for chart in list(run.charts):
            public_id = chart.cloudinary_public_id
            if public_id:
                try:
                    from services.cloudinary import delete_cloudinary_image
                    delete_cloudinary_image(public_id)
                except Exception:
                    pass

        # Clean up Cloudinary private raw assets (models, datasets)
        model_card = run.model_card or {}
        for asset_key in ("model_asset", "dataset_asset", "report_asset"):
            asset = model_card.get(asset_key) or {}
            public_id = asset.get("public_id")
            if public_id:
                try:
                    from services.cloudinary import delete_private_file
                    delete_private_file(public_id)
                except Exception:
                    pass

        session.delete(run)
        session.commit()
        return True
