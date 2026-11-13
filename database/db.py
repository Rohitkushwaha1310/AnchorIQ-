import json
import os
import threading
from functools import wraps
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.exc import ArgumentError, NoSuchModuleError, SQLAlchemyError
from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import StaticPool
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def get_database_url() -> str:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return "sqlite:///:memory:"

    database_url = (
        os.getenv("SUPABASE_DB_URL", "").strip()
        or os.getenv("SUPABASE_DATABASE_URL", "").strip()
        or os.getenv("DATABASE_URL", "").strip()
    )
    if database_url:
        try:
            from sqlalchemy.engine import make_url

            parsed = make_url(database_url)
            if parsed.drivername.lower() in {"postgres", "postgresql"}:
                parsed = parsed.set(drivername="postgresql+psycopg")
            if parsed.drivername.startswith("postgresql") and not parsed.query.get("sslmode"):
                parsed = parsed.update_query_dict({"sslmode": "require"})
            database_url = parsed.render_as_string(hide_password=False)
            if parsed.drivername and parsed.drivername.split(":", 1)[0].lower() in {
                "sqlite",
                "postgresql",
                "postgresql+psycopg",
                "postgresql+psycopg2",
                "postgresql+pg8000",
                "postgresql+asyncpg",
                "mysql",
                "mysql+pymysql",
                "mariadb",
                "oracle",
                "mssql",
                "mssql+pyodbc",
                "duckdb",
            }:
                return database_url
        except Exception as exc:
            raise ArgumentError(f"Invalid database URL: {exc}") from exc

        raise ArgumentError("Unsupported database URL scheme")
    return "sqlite:///./anchoriq_local.db"


Base = declarative_base()


class Run(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
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
    __tablename__ = "charts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    title = Column(String, nullable=False)
    cloudinary_url = Column(String, nullable=True)
    cloudinary_public_id = Column(String, nullable=True)
    fact_card = Column(JSON, default=dict, nullable=True)

    run = relationship("Run", back_populates="charts")


class Insight(Base):
    __tablename__ = "insights"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    text = Column(String, nullable=False)
    source = Column(String, nullable=False, default="rules")

    run = relationship("Run", back_populates="insights")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    run = relationship("Run", back_populates="chat_messages")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    inputs = Column(JSON, default=dict, nullable=True)
    output = Column(JSON, default=dict, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    run = relationship("Run", back_populates="predictions")


def _create_engine(database_url: str):
    if database_url.startswith("sqlite:///:memory:"):
        return create_engine(
            database_url,
            future=True,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    if database_url.startswith("sqlite:"):
        return create_engine(
            database_url,
            future=True,
            connect_args={"check_same_thread": False, "timeout": 30},
            pool_pre_ping=True,
        )
    return create_engine(database_url, future=True, pool_pre_ping=True, pool_recycle=300)


def _emergency_database_url() -> str:
    path = Path(os.getenv("ANCHORIQ_EMERGENCY_DB_PATH", "./anchoriq_local.db")).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


database_url = get_database_url()
_test_database = database_url.startswith("sqlite:///:memory:")
_fallback_reason: Optional[str] = None
_engine_lock = threading.RLock()

try:
    engine = _create_engine(database_url)
    _primary_configured = not _test_database and engine.dialect.name == "postgresql"
except (NoSuchModuleError, ArgumentError) as exc:
    _fallback_reason = f"Primary database configuration is invalid: {exc}"
    engine = _create_engine("sqlite:///:memory:" if _test_database else _emergency_database_url())
    _primary_configured = False

if not _test_database and not _primary_configured and not _fallback_reason:
    _fallback_reason = "Supabase PostgreSQL is not configured; using emergency SQLite storage"

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def _activate_emergency_database(reason: str) -> None:
    """Switch this process to SQLite after a Supabase connection/operation failure."""
    global engine, _fallback_reason
    with _engine_lock:
        if engine.dialect.name == "sqlite":
            return
        emergency_engine = _create_engine(_emergency_database_url())
        Base.metadata.create_all(bind=emergency_engine)
        engine = emergency_engine
        SessionLocal.configure(bind=engine)
        _fallback_reason = reason[:500]
        print(f"[WARN] Supabase unavailable; using emergency SQLite storage: {_fallback_reason}")


def execute_with_fallback(operation):
    """Run a database operation; retry once on emergency SQLite after a remote DB error."""
    try:
        return operation()
    except SQLAlchemyError as exc:
        if engine.dialect.name == "sqlite" or not _primary_configured:
            raise
        _activate_emergency_database(f"{type(exc).__name__}; remote database operation failed")
        return operation()


def with_database_fallback(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        return execute_with_fallback(lambda: function(*args, **kwargs))
    return wrapped


def get_database_status() -> Dict[str, Any]:
    backend = "sqlite-emergency" if engine.dialect.name == "sqlite" and not _test_database else (
        "sqlite-test" if _test_database else "supabase-postgres"
    )
    return {
        "backend": backend,
        "supabase_configured": _primary_configured,
        "emergency_mode": backend == "sqlite-emergency",
        "fallback_reason": _fallback_reason,
    }


def initialize_database() -> None:
    execute_with_fallback(lambda: Base.metadata.create_all(bind=engine))


def build_fact_card(title: str, local_path: Optional[str] = None, numbers: Optional[List[str]] = None, finding: Optional[str] = None) -> Dict[str, Any]:
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
    filename: str,
    target: Optional[str] = None,
    problem_type: Optional[str] = None,
    model_card: Optional[Dict[str, Any]] = None,
    charts: Optional[List[Dict[str, Any]]] = None,
    insights: Optional[List[Dict[str, Any]]] = None,
    chat_messages: Optional[List[Dict[str, Any]]] = None,
    predictions: Optional[List[Dict[str, Any]]] = None,
) -> int:
    initialize_database()
    with SessionLocal() as session:
        run = Run(
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
def get_runs() -> List[Dict[str, Any]]:
    initialize_database()
    with SessionLocal() as session:
        runs = session.query(Run).order_by(Run.created_at.desc()).all()
        return [_run_to_dict(run) for run in runs]


@with_database_fallback
def get_run(run_id: int) -> Optional[Dict[str, Any]]:
    initialize_database()
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        if not run:
            return None
        return _run_to_dict(run)


@with_database_fallback
def delete_run_record(run_id: int) -> bool:
    initialize_database()
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        if not run:
            return False

        for chart in list(run.charts):
            public_id = chart.cloudinary_public_id
            if public_id:
                try:
                    from services.cloudinary import delete_cloudinary_image

                    delete_cloudinary_image(public_id)
                except Exception:
                    pass
            fact_card = chart.fact_card or {}
            local_path = fact_card.get("local_path")
            if local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass

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
        for local_key in ("dataset_local_path", "report_local_path"):
            local_path = model_card.get(local_key)
            if local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass

        session.delete(run)
        session.commit()
        return True
