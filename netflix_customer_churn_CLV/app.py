import os
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

# ----------------- YOUR JUPYTER NOTEBOOK ML PIPELINE -----------------
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

def train_models():
    """Exact pipeline from your Jupyter Notebook."""
    global scaler, model, kmeans, MODEL_FEATURE_NAMES
    if not os.path.exists(CSV_FILE):
        print(f"Warning: {CSV_FILE} not found. Ensure the dataset is in the project root.")
        return

    # 1. Load dataset
    df = pd.read_csv(CSV_FILE)

    # 2. Separate numerical and categorical columns with dummy encoding
    X_num = df[NUM_COLS]
    X_cat = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1)
    MODEL_FEATURE_NAMES = X.columns.tolist()

    # 3. Fit scaler on full feature set
    X_scaled = scaler.fit_transform(X)
    Y = df['churn']

    # 4. Train-test split (80/20) and fit Logistic Regression
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=0.2, random_state=42
    )
    model.fit(X_train, Y_train)
    acc = model.score(X_test, Y_test)
    print(f"✓ Logistic Regression trained (Test Accuracy: {acc * 100:.2f}%)")

    # 5. Fit KMeans clustering directly on X_scaled
    kmeans.fit(X_scaled)
    print("✓ KMeans (4 clusters) trained on full scaled feature space.")

train_models()

def predict_customer(user_data):
    """Encodes subscriber inputs and performs prediction via trained models."""
    row_num = [float(user_data.get(col, 0)) for col in NUM_COLS]
    cat_df = pd.DataFrame([{col: user_data.get(col, '') for col in CATEGORICAL_COLS}])
    cat_encoded = pd.get_dummies(cat_df, drop_first=True)
    
    # Align incoming single user vector to the exact 31 trained feature columns
    full_row = pd.DataFrame(0, index=[0], columns=MODEL_FEATURE_NAMES)
    for col in NUM_COLS:
        full_row[col] = float(user_data.get(col, 0))
    for col in cat_encoded.columns:
        if col in full_row.columns:
            full_row[col] = cat_encoded[col].iloc[0]

    # Scale using the notebook's fitted StandardScaler
    scaled = scaler.transform(full_row)

    # 1. Logistic Regression: Churn probability %
    churn_prob = float(model.predict_proba(scaled)[0][1]) * 100

    # 2. KMeans: Predict cluster index (0, 1, 2, or 3)
    cluster_idx = int(kmeans.predict(scaled)[0])

    # Cluster mapping & tailored retention discount
    cluster_metadata = {
        0: {
            "name": "Cluster 0: Budget Streamer",
            "desc": "Standard/Basic plan consumer with moderate multi-screen household streaming.",
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

    meta = cluster_metadata.get(cluster_idx, cluster_metadata[0])
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

        for u in users:
            pred = predict_customer(u)
            u['churn_rate'] = pred['churn_rate']
            u['cluster_name'] = pred['cluster_name']
            u['discount_percent'] = pred['discount_percent']

        total_users = len(users)
        avg_churn = round(sum(u['churn_rate'] for u in users) / max(total_users, 1), 1)
        high_risk_count = sum(1 for u in users if u['churn_rate'] >= 60)

        return render_template('dashboard.html', users=users, total_users=total_users, avg_churn=avg_churn, high_risk_count=high_risk_count)
    except Error as e:
        flash(f"Database fetch error: {e}", "error")
        return redirect(url_for('admin_login'))

@app.route('/logout')
def logout():
    session.clear()
    flash("You have logged out successfully.", "info")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)