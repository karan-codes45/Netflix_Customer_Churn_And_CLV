"""
ml_engine.py  -  all the machine-learning logic of the project in ONE place.

The Flask app (app.py) only calls the functions from this file.
The notebook (notebooks/netflix_churn_clv_analysis.ipynb) explains the same steps
one by one, so read the notebook first if you want to understand the logic.

What this file does
-------------------
1. Encode the categorical columns (LabelEncoder, one per column)
2. Train a Logistic Regression model      -> churn probability of a customer
3. Train a KMeans model (4 clusters)      -> behaviour persona of a customer
4. Calculate CLV (12 month customer value) and revenue at risk
5. Put every customer in a Value x Risk segment and pick a retention offer
"""
import os
import hashlib
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "netflix_customer_churn.csv")
BUNDLE_FILE = os.path.join(BASE_DIR, "models", "bundle.joblib")

# ---------------------------------------------------------------- columns
CATEGORICAL_COLS = ["gender", "region", "subscription_type", "device",
                    "payment_method", "genre_preference"]
NUMERIC_COLS = ["age", "monthly_charges", "tenure_months", "number_of_profiles",
                "avg_watch_hours_per_week", "last_login_days",
                "support_tickets_raised", "num_devices_active",
                "has_kids_profile", "autopay_enabled", "discount_used"]
# Same column order as the CSV (without customer_id and churn)
FEATURES = ["age", "gender", "region", "subscription_type", "monthly_charges",
            "tenure_months", "number_of_profiles", "device", "payment_method",
            "genre_preference", "avg_watch_hours_per_week", "last_login_days",
            "support_tickets_raised", "num_devices_active", "has_kids_profile",
            "autopay_enabled", "discount_used"]

# ---------------------------------------------------------------- business rules
# ASSUMPTION: the dataset has no time window for "churn", so we read the model output
# as "probability that the customer leaves within the next HORIZON_MONTHS months".
HORIZON_MONTHS = 12
HIGH_RISK_PROB = 0.50          # churn probability >= 50%  -> high risk
HIGH_VALUE_MIN_MONTHLY = 600   # pays >= Rs.600/month (Premium tier) -> high value

# Monthly price of each plan (the registration form no longer asks for the price)
PLAN_PRICES = {"Basic": 199.0, "Standard": 499.0, "Premium": 799.0}

SEGMENT_OFFERS = {
    "Rescue Now": {            # high value + high risk
        "discount": 25, "months": 3, "code": "RESCUE25",
        "perk": "25% off for 3 months + priority support call-back + 4K HDR preview weekend."},
    "Protect & Reward": {      # high value + low risk
        "discount": 0, "months": 0, "code": "VIPEARLY",
        "perk": "No discount needed. Give early access to new releases and offer an annual plan."},
    "Low-Cost Nudge": {        # lower value + high risk
        "discount": 10, "months": 2, "code": "NUDGE10",
        "perk": "10% off the next 2 bills + personalised watchlist alerts."},
    "Maintain": {              # lower value + low risk
        "discount": 0, "months": 0, "code": "-",
        "perk": "No spend. Keep in the normal newsletter flow."},
}
SEGMENT_ORDER = ["Rescue Now", "Protect & Reward", "Low-Cost Nudge", "Maintain"]
CAMPAIGN_SEGMENTS = ["Rescue Now", "Low-Cost Nudge"]   # segments that get an offer email


# ---------------------------------------------------------------- CLV
def clv_12m(monthly_charges, churn_prob, horizon=HORIZON_MONTHS):
    """
    Expected revenue from a customer over the next `horizon` months.

    If p = probability of leaving within the horizon, the (constant) monthly
    leaving chance is h = 1 - (1-p)^(1/horizon).
    Expected paying months = 1 + (1-h) + (1-h)^2 + ...  (horizon terms) = p / h
    CLV = monthly_charges * expected paying months
    """
    p = np.clip(np.asarray(churn_prob, dtype=float), 0.01, 0.99)
    h = 1 - (1 - p) ** (1 / horizon)
    expected_months = p / h
    return np.asarray(monthly_charges, dtype=float) * expected_months


def assign_segment(monthly_charges, churn_prob):
    high_value = np.asarray(monthly_charges, dtype=float) >= HIGH_VALUE_MIN_MONTHLY
    high_risk = np.asarray(churn_prob, dtype=float) >= HIGH_RISK_PROB
    return np.select(
        [high_value & high_risk, high_value & ~high_risk, ~high_value & high_risk],
        ["Rescue Now", "Protect & Reward", "Low-Cost Nudge"],
        default="Maintain")


# ---------------------------------------------------------------- cluster names
def name_clusters(profile):
    """
    KMeans gives numbers (0,1,2,3) with no meaning. We name them AFTER looking at the
    average numbers of each cluster, so the names always match the data.
    profile = DataFrame indexed by cluster id with the average of each column.
    """
    names, left = {}, set(profile.index)
    c = profile["monthly_charges"].idxmax(); names[c] = "Premium Loyalist"; left.discard(c)
    c = profile.loc[list(left), "monthly_charges"].idxmin(); names[c] = "Budget Streamer"; left.discard(c)
    rest = profile.loc[list(left), "last_login_days"].sort_values()
    for i, c in enumerate(rest.index):                   # least idle -> most idle
        names[c] = "Active Standard Viewer" if i == 0 else "Dormant Standard Viewer"
    return names


def describe_cluster(row):
    return (f"Avg Rs.{row['monthly_charges']:.0f}/month, {row['tenure_months']:.0f} months tenure, "
            f"{row['avg_watch_hours_per_week']:.1f} watch hrs/week, "
            f"{row['last_login_days']:.0f} days since last login.")


# ---------------------------------------------------------------- simulated usage (demo)
_USAGE_POOL = {}


def simulate_usage(seed_text, plan):
    """
    Demo stand-in for Netflix's activity logs (a real system would read them from log tables).

    Instead of pure random numbers (which gives odd combinations like 90 watch hours and 0 devices),
    we copy the usage of a REAL customer row from the dataset who is on the same plan.
      - the pick is seeded by the user's email -> every user gets different data,
        but the same user always gets the same data (refreshing the page changes nothing)
      - a brand-new account has 1 month of tenure, no past discount, and logged in within the last week
    """
    if not _USAGE_POOL:
        data = pd.read_csv(CSV_FILE)
        for name, grp in data.groupby("subscription_type"):
            _USAGE_POOL[name] = grp[["avg_watch_hours_per_week", "support_tickets_raised", "num_devices_active"]].to_dict("records")
    pool = _USAGE_POOL.get(plan) or next(iter(_USAGE_POOL.values()))
    rng = random.Random(int(hashlib.sha256(seed_text.strip().lower().encode()).hexdigest(), 16))
    row = rng.choice(pool)
    return {"tenure_months": 1,
            "last_login_days": rng.randint(0, 7),
            "avg_watch_hours_per_week": float(row["avg_watch_hours_per_week"]),
            "support_tickets_raised": int(row["support_tickets_raised"]),
            "num_devices_active": int(row["num_devices_active"]),
            "discount_used": 0}


# ---------------------------------------------------------------- training
def _encode(df, encoders):
    """
    Turn the text columns into numbers using the fitted encoders.

    The original churn model was trained on a limited set of genres.
    The recommendation system supports additional genres such as
    Sci-Fi, Animation, Fantasy, Horror, etc.

    For genre_preference only:
        - keep the real genre in MySQL
        - use a known fallback genre for the legacy churn model

    This avoids registration errors without changing the stored
    user preference used by the recommendation engine.
    """

    out = df[FEATURES].copy()

    for col in CATEGORICAL_COLS:

        values = out[col].astype(str)

        unknown = sorted(
            set(values) - set(encoders[col].classes_)
        )

        if unknown:

            if col == "genre_preference":

                # The movie recommendation system supports more genres
                # than the original churn-training dataset.

                # We keep the user's actual genre in MySQL.
                # Only the churn model receives a fallback value.

                fallback = (
                    "Action"
                    if "Action" in encoders[col].classes_
                    else encoders[col].classes_[0]
                )

                values = values.where(
                    values.isin(encoders[col].classes_),
                    fallback
                )

            else:

                raise ValueError(
                    f"Unknown value {unknown} for '{col}'. "
                    f"Allowed: {list(encoders[col].classes_)}"
                )

        out[col] = encoders[col].transform(values)

    for col in NUMERIC_COLS:
        out[col] = out[col].astype(float)

    return out[FEATURES]


def train(csv_path=CSV_FILE):
    df = pd.read_csv(csv_path).drop(columns="customer_id")

    encoders = {col: LabelEncoder().fit(df[col].astype(str)) for col in CATEGORICAL_COLS}
    X = _encode(df, encoders)
    y = df["churn"].astype(int)

    # 1) honest test score: train on 80%, test on 20%
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_tr), y_tr)
    proba = model.predict_proba(scaler.transform(X_te))[:, 1]
    metrics = {"accuracy": accuracy_score(y_te, proba >= 0.5),
               "roc_auc": roc_auc_score(y_te, proba),
               "rows": int(len(df))}

    # 2) final model for the app: train again on 100% of the data
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X), y)

    # 3) KMeans on all customers (same as the original notebook: 4 clusters)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
    profile = df.assign(cluster=kmeans.labels_).groupby("cluster")[
        ["monthly_charges", "tenure_months", "avg_watch_hours_per_week", "last_login_days"]].mean()
    names = name_clusters(profile)
    cluster_info = {int(c): {"name": f"Cluster {c}: {names[c]}", "desc": describe_cluster(profile.loc[c])}
                    for c in profile.index}

    return {"encoders": encoders, "scaler": scaler, "model": model, "kmeans": kmeans,
            "cluster_info": cluster_info, "metrics": metrics}


def load_or_train(force=False):
    """Load the saved models. If they don't exist yet, train them (takes ~2 seconds)."""
    if not force and os.path.exists(BUNDLE_FILE):
        return joblib.load(BUNDLE_FILE)
    bundle = train()
    os.makedirs(os.path.dirname(BUNDLE_FILE), exist_ok=True)
    joblib.dump(bundle, BUNDLE_FILE)
    return bundle


# ---------------------------------------------------------------- scoring
def score_users(bundle, users):
    """
    users: list of dicts (or a DataFrame) with the FEATURES columns.
    Returns a DataFrame with the original columns + prediction columns.
    All rows are scored together, so 10,000 users take milliseconds.
    """
    df = pd.DataFrame(users).copy()
    if df.empty:
        return df
    X = _encode(df, bundle["encoders"])
    prob = bundle["model"].predict_proba(bundle["scaler"].transform(X))[:, 1]
    cluster = bundle["kmeans"].predict(X)

    monthly = df["monthly_charges"].astype(float).values
    df["churn_prob"] = prob
    df["churn_rate"] = np.round(prob * 100, 1)
    df["cluster_id"] = cluster
    df["cluster_name"] = [bundle["cluster_info"][int(c)]["name"] for c in cluster]
    df["cluster_desc"] = [bundle["cluster_info"][int(c)]["desc"] for c in cluster]
    df["annual_value"] = np.round(monthly * HORIZON_MONTHS, 2)
    df["clv_12m"] = np.round(clv_12m(monthly, prob), 2)
    df["revenue_at_risk"] = np.round(df["annual_value"] - df["clv_12m"], 2)
    df["segment"] = assign_segment(monthly, prob)
    df["discount_percent"] = df["segment"].map(lambda s: SEGMENT_OFFERS[s]["discount"])
    df["discount_code"] = df["segment"].map(lambda s: SEGMENT_OFFERS[s]["code"])
    df["discount_perk"] = df["segment"].map(lambda s: SEGMENT_OFFERS[s]["perk"])
    df["savings"] = np.round(monthly * df["discount_percent"] / 100, 2)
    df["new_price"] = np.round(monthly - df["savings"], 2)
    df["offer_cost"] = np.round(df["savings"] * df["segment"].map(lambda s: SEGMENT_OFFERS[s]["months"]), 2)
    return df


def predict_one(bundle, user):
    """Score a single user dict and return a plain dict (used by /prediction)."""
    row = score_users(bundle, [user]).iloc[0]
    out = row.to_dict()
    for k, v in out.items():                      # numpy -> python types for Jinja
        if isinstance(v, np.generic):
            out[k] = v.item()
    return out


if __name__ == "__main__":
    b = load_or_train(force=True)
    print("Models trained and saved to", BUNDLE_FILE)
    print("Test metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in b["metrics"].items()})
    for cid, info in b["cluster_info"].items():
        print(f"  {info['name']}: {info['desc']}")
