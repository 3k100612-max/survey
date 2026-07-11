from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
import psycopg2
from psycopg2.extras import RealDictCursor
import os

app = Flask(__name__)

# --- DATABASE CONFIGURATION VIA ENVIRONMENT VARIABLES ---
DB_NAME = os.getenv("DB_NAME", "survey_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "password123")
DB_HOST = os.getenv("DB_HOST", "db")  # 'db' matches the docker-compose service name
DB_PORT = os.getenv("DB_PORT", "5432")

def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    text = data.get('feedback', '').strip()
    if not text:
        return jsonify({"error": "Empty feedback"}), 400

    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    subj = blob.sentiment.subjectivity
    label = "Neutral"
    if pol > 0.1: label = "Positive"
    elif pol < -0.1: label = "Negative"

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO presentation_feedback (feedback_text, polarity, subjectivity, sentiment_label) VALUES (%s, %s, %s, %s)",
            (text, pol, subj, label)
        )
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"status": "success", "polarity": round(pol, 2), "label": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/admin_stats')
def admin_stats():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM presentation_feedback ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.execute("SELECT AVG(polarity) as avg_pol, COUNT(*) as total FROM presentation_feedback")
        stats = cur.fetchone()
        cur.close()
        conn.close()
        return jsonify({"feedback": rows, "stats": stats or {"avg_pol": 0, "total": 0}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5432)
