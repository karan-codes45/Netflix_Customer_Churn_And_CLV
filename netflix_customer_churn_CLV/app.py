"""
Netflix Customer Churn & CLV - Flask app

Run:   python app.py
Needs: a .env file (copy .env.example) with your MySQL + admin details.

All ML logic lives in ml_engine.py. This file only does web stuff:
login / register / profile pages, the prediction page and the admin dashboard.
"""
import os
import json
import hmac
import secrets

import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

import ml_engine as ml

load_dotenv()

app = Flask(__name__)
def _load_secret_key():
    """SECRET_KEY from .env if set; otherwise create one ONCE and keep it in .secret_key,
    so logins are not lost every time the app restarts."""
    key = os.getenv("SECRET_KEY")
    if key:
        return key
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".secret_key")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    key = secrets.token_hex(32)
    with open(path, "w") as f:
        f.write(key)
    return key


# Secrets come from the .env file or are generated once - never hard-code them in the code.
app.secret_key = _load_secret_key()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "@Karanp3341"),
    "database": os.getenv("DB_NAME", "netflix_churn_db"),
}
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
ADMIN_AUTH_CODE = os.getenv("ADMIN_AUTH_CODE", "")

# Models are loaded ONCE when the app starts (trained automatically the first time).
BUNDLE = ml.load_or_train()


# ------------------------------------------------------------------ database
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def init_db():
    try:
        conn = mysql.connector.connect(host=DB_CONFIG["host"], user=DB_CONFIG["user"],
                                       password=DB_CONFIG["password"])
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        cur.close()
        conn.close()

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(120) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                age INT,
                gender VARCHAR(20),
                region VARCHAR(50),
                subscription_type VARCHAR(50),
                monthly_charges DECIMAL(8, 2),
                tenure_months INT,
                number_of_profiles INT,
                device VARCHAR(50),
                payment_method VARCHAR(50),
                genre_preference VARCHAR(50),
                avg_watch_hours_per_week DECIMAL(5, 2),
                last_login_days INT,
                support_tickets_raised INT,
                num_devices_active INT,
                has_kids_profile TINYINT(1),
                autopay_enabled TINYINT(1),
                discount_used TINYINT(1),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(120) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                code_hash VARCHAR(255) NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("OK: connected to MySQL and verified the 'users' and 'admins' tables.")
    except Error as e:
        print(f"Error connecting to MySQL: {e}")


def profile_values(form):
    """Read the 17 profile fields from a form (shared by register + edit_profile)."""
    values = _read_profile(form)
    # reject values the model has never seen (e.g. a made-up region) BEFORE saving anything
    names = ["age", "gender", "region", "subscription_type", "monthly_charges", "tenure_months",
             "number_of_profiles", "device", "payment_method", "genre_preference"]
    for name, value in zip(names, values):
        if name in ml.CATEGORICAL_COLS and value not in BUNDLE["encoders"][name].classes_:
            raise ValueError(f"'{value}' is not a valid {name.replace('_', ' ')}")
    return values


def _read_profile(form):
    """
    Fields the customer fills in come from the form.
    Fields the SYSTEM should know (usage, tenure, price) are simulated when the form
    does not have them (registration) and are only editable on the 'Simulate usage' page.
    """
    plan = form.get("subscription_type", "Standard")
    d = ml.simulate_usage(form.get("email") or session.get("user_name", "guest"), plan)   # used only if the form has no usage fields
    price = ml.PLAN_PRICES.get(plan, ml.PLAN_PRICES["Standard"])   # price always comes from the plan
    return (
        int(form.get("age", 25)), form.get("gender", "Female"), form.get("region", "West India"),
        plan, price,
        int(form.get("tenure_months", d["tenure_months"])), int(form.get("number_of_profiles", 2)),
        form.get("device", "Smart TV"), form.get("payment_method", "UPI"),
        form.get("genre_preference", "Action"), float(form.get("avg_watch_hours_per_week", d["avg_watch_hours_per_week"])),
        int(form.get("last_login_days", d["last_login_days"])), int(form.get("support_tickets_raised", d["support_tickets_raised"])),
        int(form.get("num_devices_active", d["num_devices_active"])), int(form.get("has_kids_profile", 0)),
        int(form.get("autopay_enabled", 1)), int(form.get("discount_used", d["discount_used"])),
    )


def fetch_all_users():
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM users ORDER BY id DESC")
    users = cur.fetchall()
    cur.close()
    conn.close()
    return users


# ------------------------------------------------------------------ auth pages
@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        try:
            conn = get_db_connection()
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            cur.close()
            conn.close()
            # passwords are stored as hashes, so we compare with check_password_hash
            if user and check_password_hash(user["password"], password):
                session["user_id"] = user["id"]
                session["user_name"] = user["name"]
                return redirect(url_for("prediction"))
            flash("Invalid email or password. Please check your credentials.", "error")
        except Error as e:
            flash(f"Database error during login: {e}", "error")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        form = request.form
        try:
            profile = profile_values(form)          # validate first, then touch the database
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO users (name, email, password, age, gender, region, subscription_type,
                    monthly_charges, tenure_months, number_of_profiles, device, payment_method,
                    genre_preference, avg_watch_hours_per_week, last_login_days,
                    support_tickets_raised, num_devices_active, has_kids_profile,
                    autopay_enabled, discount_used)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (form.get("name"), form.get("email"),
                  generate_password_hash(form.get("password", ""))) + profile)
            conn.commit()
            user_id = cur.lastrowid
            cur.close()
            conn.close()

            session["user_id"] = user_id
            session["user_name"] = form.get("name")
            flash("Registration successful! Here is your retention forecast.", "success")
            return redirect(url_for("prediction"))
        except mysql.connector.IntegrityError:
            flash("An account with this email address already exists. Please login.", "error")
        except (Error, ValueError) as e:
            flash(f"Could not register: {e}", "error")
    return render_template("register.html")


@app.route("/edit_profile", methods=["GET", "POST"])
def edit_profile():
    if "user_id" not in session:
        flash("Please log in first.", "error")
        return redirect(url_for("login"))

    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)

    if request.method == "POST":
        form = request.form
        try:
            cur.execute("""
                UPDATE users SET age=%s, gender=%s, region=%s, subscription_type=%s,
                    monthly_charges=%s, tenure_months=%s, number_of_profiles=%s, device=%s,
                    payment_method=%s, genre_preference=%s, avg_watch_hours_per_week=%s,
                    last_login_days=%s, support_tickets_raised=%s, num_devices_active=%s,
                    has_kids_profile=%s, autopay_enabled=%s, discount_used=%s, name=%s
                WHERE id=%s
            """, profile_values(form) + (form.get("name"), session["user_id"]))
            conn.commit()
            session["user_name"] = form.get("name")
            flash("Profile updated! Churn score, CLV and segment recalculated.", "success")
            cur.close()
            conn.close()
            return redirect(url_for("prediction"))
        except (Error, ValueError) as e:
            flash(f"Error updating profile: {e}", "error")

    cur.execute("SELECT * FROM users WHERE id = %s", (session["user_id"],))
    user = cur.fetchone()
    cur.close()
    conn.close()
    return render_template("edit_profile.html", user=user)


# ------------------------------------------------------------------ customer page
@app.route("/prediction")
def prediction():
    if "user_id" not in session:
        flash("Please log in first.", "error")
        return redirect(url_for("login"))
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE id = %s", (session["user_id"],))
        user = cur.fetchone()
        cur.close()
        conn.close()
        if not user:
            session.clear()
            return redirect(url_for("login"))
        result = ml.predict_one(BUNDLE, user)
        return render_template("prediction.html", user=user, result=result)
    except (Error, ValueError) as e:
        flash(f"Could not score this profile: {e}", "error")
        return redirect(url_for("login"))


# ------------------------------------------------------------------ admin
def _same(a, b):
    """Constant-time string compare (safer than ==)."""
    return hmac.compare_digest(a.encode(), b.encode())


def _admin_ok(email, password, code):
    """1) admin saved with create_admin.py (hashed, in MySQL)  2) fallback: ADMIN_* values in .env"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT password_hash, code_hash FROM admins WHERE email = %s", (email.lower(),))
        row = cur.fetchone()
        cur.close()
        conn.close()
    except Error:
        row = None          # table missing or database down -> try the .env fallback below
    if row:
        if not check_password_hash(row["password_hash"], password):
            return False
        return not row["code_hash"] or check_password_hash(row["code_hash"], code)

    if ADMIN_EMAIL and ADMIN_PASSWORD and _same(email, ADMIN_EMAIL) and _same(password, ADMIN_PASSWORD):
        return (not ADMIN_AUTH_CODE) or _same(code, ADMIN_AUTH_CODE)
    return False


@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        ok = _admin_ok(request.form.get("email", "").strip(),
                       request.form.get("password", ""),
                       request.form.get("auth_code", "").strip())
        if ok:
            session["is_admin"] = True
            return redirect(url_for("dashboard"))
        flash("Invalid administrator credentials.", "error")
    return render_template("admin_login.html")


def _slug(text):
    return text.lower().replace(" & ", "-").replace(" ", "-")


@app.route("/dashboard")
def dashboard():
    if not session.get("is_admin"):
        flash("Administrative access required.", "error")
        return redirect(url_for("admin_login"))
    try:
        users = fetch_all_users()
        df = ml.score_users(BUNDLE, users)          # scores ALL users in one go
        total = len(df)

        # ---- one card per Value x Risk segment
        segment_cards, members = [], {}
        for seg in ml.SEGMENT_ORDER:
            part = df[df["segment"] == seg]
            offer = ml.SEGMENT_OFFERS[seg]
            key = _slug(seg)
            segment_cards.append({
                "key": key, "name": seg,
                "desc": {"Rescue Now": "High-value customers who are likely to leave. Biggest money at stake.",
                         "Protect & Reward": "High-value customers who are happy. Keep them happy, no discount needed.",
                         "Low-Cost Nudge": "Lower-value customers likely to leave. Only a small, cheap offer makes sense.",
                         "Maintain": "Lower-value, low-risk customers. No extra spend."}[seg],
                "count": len(part),
                "pct_of_total": round(len(part) / max(total, 1) * 100, 1),
                "avg_churn": round(part["churn_rate"].mean(), 1) if len(part) else 0,
                "avg_monthly": round(part["monthly_charges"].astype(float).mean(), 2) if len(part) else 0,
                "avg_clv": round(part["clv_12m"].mean(), 0) if len(part) else 0,
                "risk_money": round(part["revenue_at_risk"].sum(), 0),
                "discount": offer["discount"], "discount_code": offer["code"], "perk": offer["perk"],
                "offer_cost": round(part["offer_cost"].sum(), 0),
            })
            # only the 200 biggest-money customers per segment go to the browser (keeps page fast)
            top = part.sort_values("revenue_at_risk", ascending=False).head(200)
            members[key] = [{
                "name": r["name"], "email": r["email"], "subscription_type": r["subscription_type"],
                "monthly_charges": float(r["monthly_charges"]), "tenure_months": int(r["tenure_months"]),
                "last_login_days": int(r["last_login_days"]), "churn_rate": float(r["churn_rate"]),
                "clv_12m": float(r["clv_12m"]),
            } for _, r in top.iterrows()]

        # ---- KMeans behaviour personas (small table)
        personas = []
        for cid, grp in df.groupby("cluster_id"):
            personas.append({"name": grp["cluster_name"].iloc[0], "desc": grp["cluster_desc"].iloc[0],
                             "count": len(grp), "avg_churn": round(grp["churn_rate"].mean(), 1)})

        campaign_count = int(df["segment"].isin(ml.CAMPAIGN_SEGMENTS).sum())
        return render_template(
            "dashboard.html",
            total_users=total,
            avg_churn=round(df["churn_rate"].mean(), 1) if total else 0,
            high_risk_count=int((df["churn_prob"] >= ml.HIGH_RISK_PROB).sum()),
            campaign_count=campaign_count,
            revenue_at_risk=int(df["revenue_at_risk"].sum()),
            total_clv=int(df["clv_12m"].sum()),
            segment_cards=segment_cards,
            personas=personas,
            members_json=json.dumps(members).replace("</", "<\\/"),  # stops "</script>" tricks
        )
    except (Error, ValueError) as e:
        flash(f"Dashboard error: {e}", "error")
        return redirect(url_for("admin_login"))


@app.route("/admin/send_campaign", methods=["POST"])
def send_campaign():
    """Simulated email campaign: only 'Rescue Now' + 'Low-Cost Nudge' customers get an offer."""
    if not session.get("is_admin"):
        flash("Administrative access required.", "error")
        return redirect(url_for("admin_login"))
    try:
        df = ml.score_users(BUNDLE, fetch_all_users())
        target = df[df["segment"].isin(ml.CAMPAIGN_SEGMENTS)]
        cost = target["offer_cost"].sum()
        saved = target["revenue_at_risk"].sum()

        print("\n=============== [EMAIL CAMPAIGN - SIMULATION] ===============")
        print(f"Customers targeted: {len(target)} | offer cost: Rs.{cost:,.0f} | revenue at risk: Rs.{saved:,.0f}")
        for _, r in target.head(10).iterrows():
            print(f"  -> {r['discount_code']} ({r['discount_percent']}% off) to {r['name']} <{r['email']}> "
                  f"| risk {r['churn_rate']}% | segment {r['segment']}")
        if len(target) > 10:
            print(f"  ... and {len(target) - 10} more")
        print("==============================================================\n")

        flash(f"Campaign simulated for {len(target)} customers (Rescue Now + Low-Cost Nudge). "
              f"Offer cost Rs.{cost:,.0f} vs revenue at risk Rs.{saved:,.0f}.", "success")
    except (Error, ValueError) as e:
        flash(f"Error during campaign: {e}", "error")
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


if __name__ == "__main__":
    init_db()
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1")