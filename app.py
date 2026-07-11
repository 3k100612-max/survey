from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
import psycopg2
from psycopg2.extras import RealDictCursor
import os

app = Flask(__name__)

# --- DATABASE CONFIGURATION ---
# Replace these values with your actual Hostinger PostgreSQL credentials
DB_CONFIG = {
    "dbname": "u123456789_survey_db",
    "user": "u123456789_admin",
    "password": "YourSecurePassword123!",
    "host": "localhost",
    "port": "5432"
}

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    return psycopg2.connect(**DB_CONFIG)

# --- ROUTES ---

@app.route('/')
def index():
    """Renders the main student simulation and survey page."""
    return render_template('index.html')

@app.route('/admin')
def admin():
    """Renders the professional instructor dashboard."""
    return render_template('admin.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """
    Processes student feedback:
    1. Performs Sentiment Analysis (Polarity & Subjectivity).
    2. Persists the data into PostgreSQL.
    """
    data = request.json
    text = data.get('feedback', '').strip()
    
    if not text:
        return jsonify({"error": "Feedback text is required"}), 400

    # NLP Processing
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    subj = blob.sentiment.subjectivity
    
    # Determine Label based on Polarity score
    label = "Neutral"
    if pol > 0.1:
        label = "Positive"
    elif pol < -0.1:
        label = "Negative"

    # Database Insertion
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = """
            INSERT INTO presentation_feedback 
            (feedback_text, polarity, subjectivity, sentiment_label) 
            VALUES (%s, %s, %s, %s)
        """
        cur.execute(query, (text, pol, subj, label))
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({
            "status": "success", 
            "polarity": round(pol, 2), 
            "label": label
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/admin_stats')
def admin_stats():
    """
    Fetches aggregate statistics and raw logs for the Admin Dashboard.
    """
    try:
        conn = get_db_connection()
        # RealDictCursor allows us to return rows as dictionaries (JSON-ready)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Query 1: Fetch all feedback logs
        cur.execute("SELECT * FROM presentation_feedback ORDER BY created_at DESC")
        rows = cur.fetchall()
        
        # Query 2: Calculate aggregate statistics
        cur.execute("""
            SELECT 
                AVG(polarity) as avg_pol, 
                AVG(subjectivity) as avg_subj,
                COUNT(*) as total 
            FROM presentation_feedback
        """)
        stats = cur.fetchone()
        
        cur.close()
        conn.close()
        
        # Ensure stats are not None if table is empty
        if stats['total'] == 0:
            stats = {"avg_pol": 0, "avg_subj": 0, "total": 0}

        return jsonify({
            "feedback": rows, 
            "stats": stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- SERVER START ---
if __name__ == '__main__':
    # Hostinger uses passenger_wsgi, but this allows local testing
    app.run(debug=True, port=5000)