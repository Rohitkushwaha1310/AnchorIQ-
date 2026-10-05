"""
services/loader.py
Universal file loader: bytes + filename -> (DataFrame, meta).
Supports CSV/TSV/TXT, Excel, JSON (nested), JSONL, Parquet, XML, SQLite, PDF tables.
"""
import io
import json
import os
import re
import tempfile
import warnings

import pandas as pd

try:
    import chardet
except Exception:  # optional
    chardet = None

SUPPORTED_EXTENSIONS = (
    ".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls", ".json", ".jsonl",
    ".ndjson", ".parquet", ".xml", ".db", ".sqlite", ".sqlite3", ".pdf",
)

MAX_BYTES = 200 * 1024 * 1024  # 200 MB safety cap


def load_any(filename: str, content: bytes):
    """Return (df, meta). Raises ValueError with a user-friendly message."""
    if not content or len(content.strip()) == 0:
        raise ValueError("The uploaded file is empty.")
    if len(content) > MAX_BYTES:
        raise ValueError("File is too large (limit 200 MB).")
    ext = os.path.splitext(filename.lower())[1]
    if not ext or ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    meta = {"source_format": ext.lstrip("."), "notes": []}

    try:
        if ext in (".csv", ".tsv", ".txt"):
            df = _read_delimited(content, ext, meta)
        elif ext in (".xlsx", ".xlsm", ".xls"):
            df = _read_excel(content, meta)
        elif ext == ".json":
            df = _read_json(content, meta)
        elif ext in (".jsonl", ".ndjson"):
            df = pd.read_json(io.BytesIO(content), lines=True)
        elif ext == ".parquet":
            df = pd.read_parquet(io.BytesIO(content))
        elif ext == ".xml":
            df = _read_xml(content, meta)
        elif ext in (".db", ".sqlite", ".sqlite3"):
            df = _read_sqlite(content, meta)
        else:  # .pdf
            df = _read_pdf_tables(content, meta)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to parse file '{filename}': {e}")

    df = _tidy(df, meta)
    return df, meta


# ── readers ────────────────────────────────────────────────────────────────
def _decode(content: bytes) -> str:
    enc = "utf-8"
    if chardet is not None:
        guess = chardet.detect(content[:200_000])
        if guess.get("encoding") and (guess.get("confidence") or 0) > 0.5:
            enc = guess["encoding"]
    try:
        return content.decode(enc)
    except Exception:
        return content.decode("utf-8", errors="replace")


def _read_delimited(content: bytes, ext: str, meta: dict) -> pd.DataFrame:
    text = _decode(content)
    if not text.strip():
        raise ValueError("The delimited file is empty.")
    sample = text[:50_000]
    sep = "\t" if ext == ".tsv" else None
    if sep is None:
        import csv
        try:
            sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except Exception:
            sep = ","
    meta["notes"].append(f"Detected delimiter: {sep!r}")
    try:
        return pd.read_csv(io.StringIO(text), sep=sep, low_memory=False,
                           on_bad_lines="skip")
    except Exception:
        try:
            return pd.read_csv(io.StringIO(text), sep=None, engine="python",
                               on_bad_lines="skip")
        except Exception as e:
            raise ValueError(f"Could not parse delimited file: {e}")


def _read_excel(content: bytes, meta: dict) -> pd.DataFrame:
    try:
        sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None)
    except Exception as e:
        raise ValueError(f"Could not read Excel file: {e}")
    if not sheets:
        raise ValueError("The Excel file has no sheets.")
    name, raw = max(sheets.items(), key=lambda kv: kv[1].size)
    meta["notes"].append(f"Excel: using sheet '{name}' ({len(sheets)} sheet(s) found)")
    if raw.empty or len(raw) < 2:
        raise ValueError(f"Sheet '{name}' contains insufficient tabular data.")
    # header detection: first row where >=50% of the cells are filled
    header_idx = 0
    for i in range(min(len(raw), 30)):
        if raw.iloc[i].notna().mean() >= 0.5:
            header_idx = i
            break
    header = [str(c).strip() if pd.notna(c) else f"col_{ci}" for ci, c in enumerate(raw.iloc[header_idx])]
    df = raw.iloc[header_idx + 1:].reset_index(drop=True)
    df.columns = header
    return df.infer_objects()


def _read_json(content: bytes, meta: dict) -> pd.DataFrame:
    try:
        obj = json.loads(_decode(content))
    except Exception as e:
        raise ValueError(f"Invalid JSON format: {e}")
    if isinstance(obj, list):
        if not obj:
            raise ValueError("The JSON list is empty.")
        return pd.json_normalize(obj)
    if isinstance(obj, dict):
        if not obj:
            raise ValueError("The JSON object is empty.")
        # dict of equal-length lists -> columns
        if all(isinstance(v, list) for v in obj.values()) and len(obj) > 0:
            lens = {len(v) for v in obj.values()}
            if len(lens) == 1:
                return pd.DataFrame(obj)
        # find the biggest list of records inside the object
        best, best_len = None, 0
        stack = [obj]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                for v in cur.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict) and len(v) > best_len:
                        best, best_len = v, len(v)
                    elif isinstance(v, dict):
                        stack.append(v)
        if best is not None:
            meta["notes"].append("JSON: flattened nested records list")
            return pd.json_normalize(best)
        return pd.json_normalize([obj])
    raise ValueError("JSON must contain a list of records or an object.")


def _read_xml(content: bytes, meta: dict) -> pd.DataFrame:
    try:
        return pd.read_xml(io.BytesIO(content))
    except Exception as e:
        raise ValueError(f"Could not parse XML data: {e}")


def _read_sqlite(content: bytes, meta: dict) -> pd.DataFrame:
    import sqlite3
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp.write(content)
        path = tmp.name
    try:
        con = sqlite3.connect(path)
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        if not tables:
            raise ValueError("The SQLite file has no tables.")
        counts = {}
        for t in tables:
            try:
                counts[t] = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            except Exception:
                counts[t] = 0
        if not counts or max(counts.values(), default=0) == 0:
            raise ValueError("All SQLite tables in this file are empty.")
        biggest = max(counts, key=counts.get)
        meta["notes"].append(f"SQLite: using table '{biggest}' ({len(tables)} table(s) found)")
        df = pd.read_sql_query(f'SELECT * FROM "{biggest}" LIMIT 500000', con)
        con.close()
        return df
    except Exception as e:
        raise ValueError(f"Failed to read SQLite file: {e}")
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _read_pdf_tables(content: bytes, meta: dict) -> pd.DataFrame:
    try:
        import pdfplumber
    except Exception:
        raise ValueError("PDF support needs: pip install pdfplumber")
    frames = []
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages[:200]:
                page_tables = page.extract_tables()
                if not page_tables:
                    page_tables = page.extract_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})
                for tbl in page_tables:
                    if tbl and len(tbl) > 1:
                        cleaned = [[re.sub(r"\s+", " ", str(c)).strip() if c is not None else "" for c in row] for row in tbl]
                        cleaned = [row for row in cleaned if any(cell for cell in row)]
                        if len(cleaned) > 1:
                            header = [c if c else f"col_{ci}" for ci, c in enumerate(cleaned[0])]
                            df = pd.DataFrame(cleaned[1:], columns=header)
                            frames.append(df)
    except Exception as e:
        raise ValueError(f"Failed to extract tables from PDF: {e}")
    if not frames:
        raise ValueError("No tables could be found in this PDF.")
    meta["notes"].append(f"PDF: extracted {len(frames)} table(s)")
    # merge tables that share the same header, otherwise use the largest
    same = [f for f in frames if list(f.columns) == list(frames[0].columns)]
    return pd.concat(same, ignore_index=True) if len(same) > 1 else max(frames, key=len)


# ── tidy ───────────────────────────────────────────────────────────────────
def _tidy(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("The file contains no usable tabular data or rows.")
    df = df.copy()
    cols, seen = [], {}
    for i, c in enumerate(df.columns):
        c = re.sub(r"\s+", " ", str(c)).strip() or f"column_{i}"
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        cols.append(c)
    df.columns = cols
    before = df.shape
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
    if df.shape[0] == 0 or df.shape[1] == 0 or df.notna().sum().sum() == 0:
        raise ValueError("The file contains 100% missing values or no usable data.")
    if df.shape[1] < 2:
        raise ValueError(f"The dataset must contain at least 2 columns for analysis (found {df.shape[1]}).")
    if df.shape[0] < 2:
        raise ValueError(f"The dataset must contain at least 2 rows of data (found {df.shape[0]}).")
    for c in df.columns:  # python date/datetime objects -> proper datetime64
        if df[c].dtype == object:
            nn = df[c].dropna()
            if len(nn) and nn.map(lambda v: hasattr(v, "year") and hasattr(v, "month")).all():
                df[c] = pd.to_datetime(df[c], errors="coerce")
    if df.shape != before:
        meta["notes"].append(f"Dropped empty rows/columns: {before} -> {df.shape}")
    return df
