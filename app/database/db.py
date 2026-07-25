# database/db.py
from sqlalchemy import create_engine
import pandas as pd

def get_engine():
    return create_engine(
        'postgresql://postgres:YOUR_PASSWORD@localhost:5432/superstore_db'
    )

def query(sql):
    engine = get_engine()
    return pd.read_sql_query(sql, engine)

def load_table(df, table_name, if_exists='replace'):
    engine = get_engine()
    df.to_sql(table_name, engine,
              if_exists=if_exists, index=False)
    print(f"✅ {len(df)} records loaded to '{table_name}'!")