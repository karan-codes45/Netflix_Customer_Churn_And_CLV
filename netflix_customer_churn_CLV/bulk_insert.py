"""
bulk_insert.py - loads the CSV data into MySQL.

Run once, after creating your .env file:   python bulk_insert.py

It fills 3 tables (see ../sql/01_schema.sql):
  customers        -> all 10,000 rows of the dataset, WITH the churn label (for SQL analysis)
  users            -> demo login accounts for the Flask app (password = DEMO_USER_PASSWORD in .env)
  customer_scores  -> churn probability / CLV / segment exported by the notebook (if the file exists)
"""
import os
import pandas as pd
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "netflix_customer_churn.csv")
SCORED_FILE = os.path.join(BASE_DIR, "..", "data", "netflix_scored_customers.csv")
SCHEMA_FILE = os.path.join(BASE_DIR, "..", "sql", "01_schema.sql")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
}
DB_NAME = os.getenv("DB_NAME", "netflix_churn_db")

PROFILE_COLS = ["age", "gender", "region", "subscription_type", "monthly_charges", "tenure_months",
                "number_of_profiles", "device", "payment_method", "genre_preference",
                "avg_watch_hours_per_week", "last_login_days", "support_tickets_raised",
                "num_devices_active", "has_kids_profile", "autopay_enabled", "discount_used"]


def run_schema(cursor):
    with open(SCHEMA_FILE, encoding="utf-8") as f:
        sql = f.read().replace("netflix_churn_db", DB_NAME)
    lines = [ln for ln in sql.splitlines() if not ln.strip().startswith("--")]
    for statement in "\n".join(lines).split(";"):
        if statement.strip():
            cursor.execute(statement)


def chunks(rows, size=500):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def native(value):
    """pandas/numpy numbers -> normal Python numbers (MySQL connector needs this)."""
    return value.item() if hasattr(value, "item") else value


def main(limit=None):
    demo_password = os.getenv("DEMO_USER_PASSWORD")
    if not demo_password:
        print("Please set DEMO_USER_PASSWORD in your .env file first (see .env.example).")
        return
    df = pd.read_csv(CSV_FILE)
    if limit:
        df = df.head(limit)

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor()
        run_schema(cur)
        cur.execute(f"USE {DB_NAME}")

        # ---- 1) customers (raw dataset with churn label)
        cols = ["customer_id"] + PROFILE_COLS + ["churn"]
        q = (f"INSERT INTO customers ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
             f"ON DUPLICATE KEY UPDATE churn = VALUES(churn)")
        rows = [tuple(native(v) for v in r) for r in df[cols].itertuples(index=False, name=None)]
        for part in chunks(rows):
            cur.executemany(q, part)
            conn.commit()
        print(f"customers        : {len(rows)} rows")

        # ---- 2) users (demo login accounts)
        # We hash the password ONCE and reuse it. Hashing 10,000 times would take very long.
        # (Real sign-ups through the website get their own salted hash.)
        hashed = generate_password_hash(demo_password)
        ucols = ["name", "email", "password"] + PROFILE_COLS
        q = (f"INSERT INTO users ({', '.join(ucols)}) VALUES ({', '.join(['%s'] * len(ucols))}) "
             f"ON DUPLICATE KEY UPDATE last_login_days = VALUES(last_login_days), "
             f"avg_watch_hours_per_week = VALUES(avg_watch_hours_per_week), "
             f"support_tickets_raised = VALUES(support_tickets_raised)")
        urows = []
        for r in df.itertuples(index=False):
            cid = str(r.customer_id).strip()
            urows.append((f"Subscriber {cid}", f"{cid.lower()}@netflixuser.com", hashed)
                         + tuple(native(getattr(r, c)) for c in PROFILE_COLS))
        for part in chunks(urows):
            cur.executemany(q, part)
            conn.commit()
        print(f"users            : {len(urows)} rows  (login: <customer_id>@netflixuser.com + your DEMO_USER_PASSWORD)")

        # ---- 3) customer_scores (from the notebook)
        if os.path.exists(SCORED_FILE):
            sc = pd.read_csv(SCORED_FILE)
            sc = sc[sc["customer_id"].isin(df["customer_id"])]
            scols = ["customer_id", "churn_prob", "annual_value", "clv_12m", "revenue_at_risk", "segment", "persona"]
            q = (f"INSERT INTO customer_scores ({', '.join(scols)}) VALUES ({', '.join(['%s'] * len(scols))}) "
                 f"ON DUPLICATE KEY UPDATE churn_prob = VALUES(churn_prob), clv_12m = VALUES(clv_12m), "
                 f"revenue_at_risk = VALUES(revenue_at_risk), segment = VALUES(segment)")
            srows = [tuple(native(v) for v in r) for r in sc[scols].itertuples(index=False, name=None)]
            for part in chunks(srows):
                cur.executemany(q, part)
                conn.commit()
            print(f"customer_scores  : {len(srows)} rows")
        else:
            print("customer_scores  : skipped (run the notebook first to create data/netflix_scored_customers.csv)")

        cur.close()
        conn.close()
        print("\nDone.")
    except Error as e:
        print(f"Database error: {e}")


if __name__ == "__main__":
    main(limit=None)   # set limit=500 if you only want a small test load
