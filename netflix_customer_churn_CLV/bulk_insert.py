import os
import pandas as pd
import mysql.connector
from mysql.connector import Error

# 1. Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',               # Change to your MySQL username
    'password': '@Karanp3341',   # Change to your MySQL password
    'database': 'netflix_churn_db'
}

CSV_FILE = "netflix_customer_churn.csv"

def bulk_insert_subscribers(csv_path, limit=None):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path} in the current directory.")
        return

    print("Reading CSV dataset...")
    df = pd.read_csv(csv_path)
    
    # If limit is specified (e.g. 500, 1000), slice the dataframe. Otherwise, import all.
    if limit:
        df = df.head(limit)

    print(f"Preparing {len(df)} records for MySQL insertion...")

    # Build tuples for batch insert
    records = []
    for _, row in df.iterrows():
        cust_id = str(row['customer_id']).strip()
        # Generate clean credentials
        name = f"Subscriber {cust_id}"
        email = f"{cust_id.lower()}@netflixuser.com"
        password = "User1234"

        record = (
            name,
            email,
            password,
            int(row['age']),
            str(row['gender']),
            str(row['region']),
            str(row['subscription_type']),
            float(row['monthly_charges']),
            int(row['tenure_months']),
            int(row['number_of_profiles']),
            str(row['device']),
            str(row['payment_method']),
            str(row['genre_preference']),
            float(row['avg_watch_hours_per_week']),
            int(row['last_login_days']),
            int(row['support_tickets_raised']),
            int(row['num_devices_active']),
            int(row['has_kids_profile']),
            int(row['autopay_enabled']),
            int(row['discount_used'])
        )
        records.append(record)

    insert_query = """
        INSERT INTO users (
            name, email, password, age, gender, region,
            subscription_type, monthly_charges, tenure_months, number_of_profiles,
            device, payment_method, genre_preference, avg_watch_hours_per_week,
            last_login_days, support_tickets_raised, num_devices_active,
            has_kids_profile, autopay_enabled, discount_used
        ) VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE 
            avg_watch_hours_per_week = VALUES(avg_watch_hours_per_week),
            last_login_days = VALUES(last_login_days),
            support_tickets_raised = VALUES(support_tickets_raised);
    """

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Batch insert in chunks of 500 for optimal memory & speed
        chunk_size = 500
        total_inserted = 0

        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            cursor.executemany(insert_query, chunk)
            conn.commit()
            total_inserted += len(chunk)
            print(f"  Inserted {total_inserted}/{len(records)} subscribers...")

        cursor.close()
        conn.close()
        print(f"\n✓ Successfully loaded {total_inserted} users into MySQL database!")

    except Error as e:
        print(f"Database error during bulk insert: {e}")

if __name__ == '__main__':
    # Change limit to None if you want to import all 10,000 users, or set a number (e.g. 500)
    bulk_insert_subscribers(CSV_FILE, limit=500)