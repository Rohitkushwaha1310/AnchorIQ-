import io
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

with open("telco_churn.csv", "rb") as f:
    content = f.read()

resp = client.post("/analyze", files={"file": ("telco_churn.csv", content, "text/csv")}, data={"analysis_type": "churn"})
print("Analyze status:", resp.status_code)
res = resp.json()
session_id = res["session_id"]
print("Session ID:", session_id)

resp_pdf = client.post("/generate_report", files={"file": ("telco_churn.csv", content, "text/csv")}, data={"analysis_type": "churn"})
print("Generate report status:", resp_pdf.status_code)
print("PDF content starts with:", resp_pdf.content[:20])

resp_dl = client.get(f"/download/{session_id}")
print("Download status:", resp_dl.status_code)
print("Download content length:", len(resp_dl.content))
print("Download content preview:", resp_dl.content[:150])

df_dl = pd.read_csv(io.BytesIO(resp_dl.content))
print("Parsed CSV shape:", df_dl.shape)
print("Columns:", list(df_dl.columns[:8]))
