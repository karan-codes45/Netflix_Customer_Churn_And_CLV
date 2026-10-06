"""
create_admin.py - run this ONCE to set your admin login. Run it again any time you want to change it.

    python create_admin.py

It asks for: admin email, password, and an optional 6-digit 2FA code.
They are saved in MySQL (table `admins`) as HASHES, so nobody can read the real password.
After this you do not need ADMIN_* values in the .env file at all.
"""
import os
import getpass
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
}
DB_NAME = os.getenv("DB_NAME", "netflix_churn_db")

ADMINS_TABLE = """
    CREATE TABLE IF NOT EXISTS admins (
        id INT AUTO_INCREMENT PRIMARY KEY,
        email VARCHAR(120) NOT NULL UNIQUE,
        password_hash VARCHAR(255) NOT NULL,
        code_hash VARCHAR(255) NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def save_admin(email, password, code=""):
    """Create the admin, or update it if the email already exists."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    cur.execute(f"USE {DB_NAME}")
    cur.execute(ADMINS_TABLE)
    cur.execute(
        "INSERT INTO admins (email, password_hash, code_hash) VALUES (%s, %s, %s) "
        "ON DUPLICATE KEY UPDATE password_hash = VALUES(password_hash), code_hash = VALUES(code_hash)",
        (email.strip().lower(), generate_password_hash(password), generate_password_hash(code) if code else None))
    conn.commit()
    cur.close()
    conn.close()


def main():
    print("=== Create / update the ADMIN login ===\n")
    email = input("Admin email: ").strip().lower()
    if "@" not in email:
        print("That does not look like an email. Run the script again.")
        return
    password = getpass.getpass("Admin password (typing is hidden): ")
    if len(password) < 8:
        print("Please use at least 8 characters. Run the script again.")
        return
    if password != getpass.getpass("Repeat password: "):
        print("The two passwords are different. Run the script again.")
        return
    code = input("2FA code, e.g. 6 digits (press Enter to skip): ").strip()

    try:
        save_admin(email, password, code)
    except Error as e:
        print(f"\nDatabase error: {e}\nCheck DB_PASSWORD in your .env file and that MySQL is running.")
        return
    print(f"\nDone. Admin '{email}' is saved. Log in at /admin_login"
          + (" with the email, password and 2FA code." if code else " with the email and password (leave the 2FA box empty)."))


if __name__ == "__main__":
    main()