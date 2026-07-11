import os
import time
import psycopg2
from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Passenger (cPanel/Plesk/CloudPanel) imports this file directly and does
# NOT read docker-compose's `env_file: .env` — that only applies when
# running via `docker compose up`. So we load .env explicitly here to make
# sure DB_USER/DB_PASS/DB_HOST/DB_NAME are actually populated no matter how
# the app is launched.
load_dotenv()

app = Flask(__name__)

# --- DATABASE CONFIGURATION VIA ENVIRONMENT VARIABLES ---
DB_USER = os.environ.get('DB_USER')
DB_PASS = os.environ.get('DB_PASS')
DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
DB_PORT = os.environ.get('DB_PORT', '5432')

print(f"DB config loaded -> host='{DB_HOST}' port={DB_PORT} db='{DB_NAME}' user='{DB_USER}' pass_set={'yes' if DB_PASS else 'NO'}")

# Port the FLASK app itself listens on (not the database port).
# Reads from env var PORT if set (useful for hosting platforms / containers),
# otherwise falls back to 8507. This means you never have to hunt for a
# hardcoded value again — set PORT=8507 in your environment if you want
# to force it, or just check the terminal log for whatever it picked.
APP_PORT = int(os.environ.get('PORT', 8507))


def get_db_connection():
    """Retry logic to prevent crash if DB is still booting.
    Uses a short connect_timeout per attempt so a bad host fails fast
    instead of hanging (a slow/hanging connect is a common cause of
    502s from an upstream proxy)."""
    retries = 5
    last_error = None
    while retries > 0:
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASS,
                host=DB_HOST,
                port=DB_PORT,
                connect_timeout=5,
            )
            return conn
        except psycopg2.OperationalError as e:
            last_error = e
            retries -= 1
            print(f"Waiting for database at host='{DB_HOST}' port={DB_PORT}... {retries} attempts left ({e})")
            time.sleep(2)
    raise Exception(
        f"Could not connect to PostgreSQL at host='{DB_HOST}' port={DB_PORT} "
        f"db='{DB_NAME}' user='{DB_USER}'. Last error: {last_error}"
    )


def init_db():
    """Create the feedback table if it doesn't already exist.
    This runs once at startup so you never get 'relation does not exist' errors."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS presentation_feedback (
            id SERIAL PRIMARY KEY,
            feedback_text TEXT NOT NULL,
            polarity REAL,
            subjectivity REAL,
            sentiment_label TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Database ready: presentation_feedback table checked/created.")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/admin')
def admin():
    return render_template('admin.html')


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json(silent=True) or {}
    text = data.get('feedback', '').strip()

    if not text:
        return jsonify({"error": "Empty feedback"}), 400

    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    subj = blob.sentiment.subjectivity

    label = "Neutral"
    if pol > 0.1:
        label = "Positive"
    elif pol < -0.1:
        label = "Negative"

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO presentation_feedback
               (feedback_text, polarity, subjectivity, sentiment_label)
               VALUES (%s, %s, %s, %s)""",
            (text, pol, subj, label)
        )
        conn.commit()
        cur.close()
        return jsonify({"status": "success", "polarity": round(pol, 2), "label": label})
    except Exception as e:
        print(f"DB INSERT ERROR: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            conn.close()


@app.route('/admin_stats')
def admin_stats():
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM presentation_feedback ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.execute("SELECT AVG(polarity) as avg_pol, COUNT(*) as total FROM presentation_feedback")
        stats = cur.fetchone()
        cur.close()
        return jsonify({"feedback": rows, "stats": stats or {"avg_pol": 0, "total": 0}})
    except Exception as e:
        print(f"DB READ ERROR: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            conn.close()


# Under Passenger, this file is imported (not executed as __main__), so
# app.run() below never fires — Passenger handles the actual serving/port.
# We still want the table to exist, so initialize it at import time too.
try:
    init_db()
except Exception as e:
    print(f"WARNING: could not initialize database at startup: {e}")

if __name__ == "__main__":
    # This block only runs if you launch with `python app.py` directly
    # (e.g. local dev, or the Docker path) — NOT under Passenger.
    print(f"Starting Flask app on port {APP_PORT} (set the PORT env var to override)")
    app.run(host="0.0.0.0", port=APP_PORT)
