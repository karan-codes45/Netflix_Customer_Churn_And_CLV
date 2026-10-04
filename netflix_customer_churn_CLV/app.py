import os
import json
import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = Flask(__name__)
app.secret_key = "netflix_secret_key_prod_2026"

# ----------------- MYSQL DATABASE CONFIGURATION -----------------
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '@Karanp3341',  # Your verified MySQL password
    'database': 'netflix_churn_db'
}

CSV_FILE = "netflix_customer_churn.csv"

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def init_db():
    try:
        temp_conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        temp_cursor = temp_conn.cursor()
        temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']};")
        temp_cursor.close()
        temp_conn.close()

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
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
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        ''')
        conn.commit()
        cursor.close()
        conn.close()
        print("✓ Connected to MySQL database and verified 'users' table.")
    except Error as e:
        print(f"Error connecting to MySQL: {e}")

init_db()

# ----------------- MACHINE LEARNING ENGINE -----------------
scaler = StandardScaler()
model = LogisticRegression(max_iter=1000)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

NUM_COLS = [
    'age', 'monthly_charges', 'tenure_months', 'number_of_profiles',
    'avg_watch_hours_per_week', 'last_login_days', 'support_tickets_raised',
    'num_devices_active', 'has_kids_profile', 'autopay_enabled', 'discount_used'
]
CATEGORICAL_COLS = ['gender', 'region', 'subscription_type', 'device', 'payment_method', 'genre_preference']
MODEL_FEATURE_NAMES = []

CLUSTER_METADATA = {
    0: {
        "name": "Cluster 0: Budget Streamer",
        "desc": "Standard/Basic plan consumer with moderate multi-screen streaming.",
        "discount": 20,
        "code": "SAVE20NOW",
        "perk": "20% off 6-month bundle lock-in with zero ad interruptions."
    },
    1: {
        "name": "Cluster 1: Standard Active Household",
        "desc": "Regular engagement with balanced viewing hours across standard devices.",
        "discount": 15,
        "code": "STREAM15",
        "perk": "15% discount on upcoming renewal cycle + 1 bonus profile."
    },
    2: {
        "name": "Cluster 2: High-Spend Premium Loyalist",
        "desc": "Top-tier subscription with high monthly charge commitment.",
        "discount": 25,
        "code": "PREM25VIP",
        "perk": "25% discount for 3 months + free 4K HDR Atmos preview weekend."
    },
    3: {
        "name": "Cluster 3: Dormant / At-Risk Viewer",
        "desc": "Exhibits lower watch time or declining login activity; prime candidate for re-engagement.",
        "discount": 35,
        "code": "COMEBACK35",
        "perk": "35% bill reduction on next 3 billing cycles + personalized watchlist alerts."
    }
}

def train_models():
    global scaler, model, kmeans, MODEL_FEATURE_NAMES
    if not os.path.exists(CSV_FILE):
        print(f"Warning: {CSV_FILE} not found. Ensure dataset is in project root.")
        return

    df = pd.read_csv(CSV_FILE)
    X_num = df[NUM_COLS]
    X_cat = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1)
    MODEL_FEATURE_NAMES = X.columns.tolist()

    X_scaled = scaler.fit_transform(X)
    Y = df['churn']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=0.2, random_state=42
    )
    model.fit(X_train, Y_train)
    acc = model.score(X_test, Y_test)
    print(f"✓ Logistic Regression trained (Test Accuracy: {acc * 100:.2f}%)")

    kmeans.fit(X_scaled)
    print("✓ KMeans (4 clusters) trained on full scaled feature space.")

train_models()

def predict_customer(user_data):
    row_num = [float(user_data.get(col, 0)) for col in NUM_COLS]
    cat_df = pd.DataFrame([{col: user_data.get(col, '') for col in CATEGORICAL_COLS}])
    cat_encoded = pd.get_dummies(cat_df, drop_first=True)
    
    full_row = pd.DataFrame(0, index=[0], columns=MODEL_FEATURE_NAMES)
    for col in NUM_COLS:
        full_row[col] = float(user_data.get(col, 0))
    for col in cat_encoded.columns:
        if col in full_row.columns:
            full_row[col] = cat_encoded[col].iloc[0]

    scaled = scaler.transform(full_row)
    churn_prob = float(model.predict_proba(scaled)[0][1]) * 100
    cluster_idx = int(kmeans.predict(scaled)[0])

    meta = CLUSTER_METADATA.get(cluster_idx, CLUSTER_METADATA[0])
    discount = meta["discount"]
    monthly = float(user_data.get('monthly_charges', 499.0))
    savings = round((monthly * discount) / 100, 2)
    new_price = round(monthly - savings, 2)

    return {
        "churn_rate": round(churn_prob, 1),
        "cluster_id": cluster_idx,
        "cluster_name": meta["name"],
        "cluster_desc": meta["desc"],
        "discount_percent": discount,
        "discount_code": meta["code"],
        "discount_perk": meta["perk"],
        "savings": savings,
        "new_price": new_price
    }

# ----------------- ROUTES -----------------
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user:
                session['user_id'] = user['id']
                session['user_name'] = user['name']
                return redirect(url_for('prediction'))
            else:
                flash("Invalid email or password. Please check your credentials.", "error")
        except Error as e:
            flash(f"Database error: {e}", "error")

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        form = request.form
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            query = '''
                INSERT INTO users (
                    name, email, password, age, gender, region, subscription_type,
                    monthly_charges, tenure_months, number_of_profiles, device,
                    payment_method, genre_preference, avg_watch_hours_per_week,
                    last_login_days, support_tickets_raised, num_devices_active,
                    has_kids_profile, autopay_enabled, discount_used
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''
            values = (
                form.get('name'), form.get('email'), form.get('password'),
                int(form.get('age', 25)), form.get('gender', 'Female'), form.get('region', 'West India'),
                form.get('subscription_type', 'Standard'), float(form.get('monthly_charges', 499)),
                int(form.get('tenure_months', 12)), int(form.get('number_of_profiles', 2)),
                form.get('device', 'Smart TV'), form.get('payment_method', 'UPI'),
                form.get('genre_preference', 'Action'), float(form.get('avg_watch_hours_per_week', 10)),
                int(form.get('last_login_days', 5)), int(form.get('support_tickets_raised', 0)),
                int(form.get('num_devices_active', 2)), int(form.get('has_kids_profile', 0)),
                int(form.get('autopay_enabled', 1)), int(form.get('discount_used', 0))
            )
            cursor.execute(query, values)
            conn.commit()
            user_id = cursor.lastrowid
            cursor.close()
            conn.close()

            session['user_id'] = user_id
            session['user_name'] = form.get('name')
            flash("Registration successful! Here is your AI retention forecast.", "success")
            return redirect(url_for('prediction'))

        except mysql.connector.IntegrityError:
            flash("An account with this email address already exists. Please login.", "error")
        except Error as e:
            flash(f"Database error during registration: {e}", "error")

    return render_template('register.html')

@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user_id' not in session:
        flash("Please log in first.", "error")
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    if request.method == 'POST':
        form = request.form
        try:
            update_query = '''
                UPDATE users SET
                    name = %s, age = %s, gender = %s, region = %s,
                    subscription_type = %s, monthly_charges = %s, tenure_months = %s,
                    number_of_profiles = %s, device = %s, payment_method = %s,
                    genre_preference = %s, avg_watch_hours_per_week = %s,
                    last_login_days = %s, support_tickets_raised = %s,
                    num_devices_active = %s, has_kids_profile = %s,
                    autopay_enabled = %s, discount_used = %s
                WHERE id = %s
            '''
            values = (
                form.get('name'), int(form.get('age', 25)), form.get('gender', 'Female'),
                form.get('region', 'West India'), form.get('subscription_type', 'Standard'),
                float(form.get('monthly_charges', 499)), int(form.get('tenure_months', 12)),
                int(form.get('number_of_profiles', 2)), form.get('device', 'Smart TV'),
                form.get('payment_method', 'UPI'), form.get('genre_preference', 'Action'),
                float(form.get('avg_watch_hours_per_week', 10)), int(form.get('last_login_days', 5)),
                int(form.get('support_tickets_raised', 0)), int(form.get('num_devices_active', 2)),
                int(form.get('has_kids_profile', 0)), int(form.get('autopay_enabled', 1)),
                int(form.get('discount_used', 0)), session['user_id']
            )
            cursor.execute(update_query, values)
            conn.commit()
            session['user_name'] = form.get('name')
            flash("Profile updated! Churn score and cluster persona recalculated.", "success")
            cursor.close()
            conn.close()
            return redirect(url_for('prediction'))
        except Error as e:
            flash(f"Error updating profile: {e}", "error")

    cursor.execute("SELECT * FROM users WHERE id = %s", (session['user_id'],))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    return render_template('edit_profile.html', user=user)

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        auth_code = request.form.get('auth_code', '').strip()

        if email == "admin@netflix.com" and password == "Admin2026!" and (auth_code == "849201" or auth_code == ""):
            session['is_admin'] = True
            return redirect(url_for('dashboard'))
        else:
            flash("Admin authentication rejected. Invalid credentials or 2FA code.", "error")

    return render_template('admin_login.html')

@app.route('/prediction')
def prediction():
    if 'user_id' not in session:
        flash("Please log in to view your profile and personalized offers.", "error")
        return redirect(url_for('login'))

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE id = %s", (session['user_id'],))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if not user:
            flash("User not found.", "error")
            return redirect(url_for('login'))

        result = predict_customer(user)
        return render_template('prediction.html', user=user, result=result)
    except Error as e:
        flash(f"Database fetch error: {e}", "error")
        return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    """Cluster-Sorted Aggregation View (No raw user dump)"""
    if not session.get('is_admin'):
        flash("Administrative access required.", "error")
        return redirect(url_for('admin_login'))

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users ORDER BY id DESC")
        users = cursor.fetchall()
        cursor.close()
        conn.close()

        # Score every user and partition into cluster bins
        cluster_buckets = {0: [], 1: [], 2: [], 3: []}
        total_churn = 0
        critical_count = 0

        for u in users:
            pred = predict_customer(u)
            u_summary = {
                'id': u['id'],
                'name': u['name'],
                'email': u['email'],
                'subscription_type': u['subscription_type'],
                'monthly_charges': float(u['monthly_charges']),
                'tenure_months': u['tenure_months'],
                'last_login_days': u['last_login_days'],
                'churn_rate': pred['churn_rate'],
                'cluster_id': pred['cluster_id']
            }
            cluster_buckets[pred['cluster_id']].append(u_summary)
            total_churn += pred['churn_rate']
            if pred['churn_rate'] >= 60:
                critical_count += 1

        total_users = len(users)
        avg_churn = round(total_churn / max(total_users, 1), 1)

        # Build aggregated cluster summary cards
        cluster_summaries = []
        for c_id in range(4):
            members = cluster_buckets[c_id]
            c_meta = CLUSTER_METADATA[c_id]
            c_count = len(members)
            c_avg_churn = round(sum(m['churn_rate'] for m in members) / max(c_count, 1), 1)
            c_avg_monthly = round(sum(m['monthly_charges'] for m in members) / max(c_count, 1), 2)
            c_critical = sum(1 for m in members if m['churn_rate'] >= 60)
            pct_of_total = round((c_count / max(total_users, 1)) * 100, 1)

            cluster_summaries.append({
                "cluster_id": c_id,
                "name": c_meta["name"],
                "desc": c_meta["desc"],
                "count": c_count,
                "pct_of_total": pct_of_total,
                "avg_churn": c_avg_churn,
                "avg_monthly": c_avg_monthly,
                "discount": c_meta["discount"],
                "discount_code": c_meta["code"],
                "perk": c_meta["perk"],
                "critical_in_cluster": c_critical
            })

        # Sort clusters by highest average churn risk first
        cluster_summaries.sort(key=lambda x: x["avg_churn"], reverse=True)

        return render_template(
            'dashboard.html',
            total_users=total_users,
            avg_churn=avg_churn,
            high_risk_count=critical_count,
            cluster_summaries=cluster_summaries,
            cluster_members_json=json.dumps(cluster_buckets)
        )
    except Error as e:
        flash(f"Database fetch error: {e}", "error")
        return redirect(url_for('admin_login'))

@app.route('/admin/send_campaign', methods=['POST'])
def send_campaign():
    """Simulates targeted promotional email campaign dispatched to critical users (churn >= 60%)"""
    if not session.get('is_admin'):
        flash("Administrative access required.", "error")
        return redirect(url_for('admin_login'))

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        cursor.close()
        conn.close()

        sent_count = 0
        dispatched_recipients = []

        for u in users:
            pred = predict_customer(u)
            if pred['churn_rate'] >= 60.0:
                # Log simulated email dispatch
                dispatched_recipients.append({
                    "email": u['email'],
                    "name": u['name'],
                    "churn": pred['churn_rate'],
                    "code": pred['discount_code'],
                    "discount": pred['discount_percent']
                })
                sent_count += 1

        print(f"\n================ [EMAIL CAMPAIGN DISPATCH TRIGGERED] ================")
        print(f"Total Critical Subscribers Targeted: {sent_count}")
        for r in dispatched_recipients[:10]:  # Print first 10 to terminal console
            print(f"  ✉ Sent offer '{r['code']}' ({r['discount']}% OFF) to {r['name']} <{r['email']}> | Risk: {r['churn']}%")
        if sent_count > 10:
            print(f"  ... and {sent_count - 10} additional critical users emailed.")
        print(f"====================================================================\n")

        flash(f"Promotional campaign dispatched successfully! Sent targeted retention offers to {sent_count} critical subscribers (Churn >= 60%).", "success")
        return redirect(url_for('dashboard'))

    except Error as e:
        flash(f"Error during promotional dispatch: {e}", "error")
        return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    flash("You have logged out successfully.", "info")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)