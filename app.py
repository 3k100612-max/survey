import os
import time
import re
import io
import csv
import psycopg2
from flask import Flask, render_template, request, jsonify, send_file, Response
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from transformers import pipeline

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
APP_PORT = int(os.environ.get('PORT', 8507))

# --- SENTIMENT ANALYSIS CONFIG ---
# Polarity is reconstructed from the Hugging Face model's per-class scores
# as (positive_score - negative_score), which lands in roughly -1..1 —
# same range TextBlob used — so these thresholds still apply unchanged.
SENTIMENT_THRESHOLDS = [
    (0.5, "Very Positive"),
    (0.1, "Positive"),
    (-0.1, "Neutral"),
    (-0.5, "Negative"),
    (-1.01, "Very Negative"),  # catches everything down to -1
]

MAX_FEEDBACK_LENGTH = 2000  # characters — prevents abuse / junk submissions

# Hugging Face model for sentiment. This is a 3-class (negative/neutral/
# positive) RoBERTa model fine-tuned on social-media-style text, which
# tends to generalize well to short freeform feedback.
HF_SENTIMENT_MODEL = os.environ.get(
    "HF_SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Basic stopword list for lightweight key-phrase extraction (no external
# corpus download required, unlike TextBlob's noun_phrases()).
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "for", "with", "at", "by",
    "from", "up", "about", "into", "over", "after", "this", "that", "these",
    "those", "it", "its", "i", "you", "we", "they", "he", "she", "them",
    "my", "your", "our", "their", "as", "so", "very", "just", "really",
    "not", "no", "do", "did", "does", "have", "has", "had", "will", "would",
    "could", "should", "can", "if", "than", "then", "there", "here", "was",
}


def classify_polarity(polarity):
    """Map a -1..1 polarity score to a human-readable label using
    SENTIMENT_THRESHOLDS, ordered from most positive to most negative."""
    for threshold, label in SENTIMENT_THRESHOLDS:
        if polarity >= threshold:
            return label
    return "Neutral"  # fallback, should never actually hit this


def classify_subjectivity(subjectivity):
    """Bucket the 0..1 subjectivity proxy for display. Unlike TextBlob's
    statistical subjectivity score, this is derived from how far the HF
    model's confidence sits from "neutral" — a proxy for how emotionally
    loaded/opinionated the text reads, not a linguistic subjectivity
    measure in the strict sense."""
    if subjectivity >= 0.6:
        return "Opinion"
    elif subjectivity >= 0.3:
        return "Mixed"
    else:
        return "Objective"


def extract_key_phrases(text, limit=5):
    """Lightweight, dependency-free key-phrase extraction: filters
    stopwords/punctuation and returns the most frequent remaining words,
    in order of first appearance. Not as linguistically rich as noun-phrase
    chunking, but needs no extra corpus downloads and is fast. Best-effort —
    returns an empty list rather than raising if extraction fails."""
    try:
        words = re.findall(r"[A-Za-z']+", text.lower())
        counts = {}
        order = []
        for w in words:
            if w in _STOPWORDS or len(w) < 3:
                continue
            if w not in counts:
                order.append(w)
            counts[w] = counts.get(w, 0) + 1
        ranked = sorted(order, key=lambda w: (-counts[w], order.index(w)))
        return ranked[:limit]
    except Exception as e:
        print(f"Key phrase extraction failed (non-fatal): {e}")
        return []


# --- HUGGING FACE SENTIMENT PIPELINE (loaded once at startup) ---
# Loading a transformer model takes a few seconds, so this must happen
# ONCE at process startup, not per-request, or every submission would pay
# that cost. If loading fails (e.g. no internet access to download weights
# on first run, or missing torch), the app still starts, but
# /submit_feedback will return a clear 500 telling you why.
_sentiment_pipeline = None


def load_sentiment_pipeline():
    global _sentiment_pipeline
    print(f"Loading Hugging Face sentiment model '{HF_SENTIMENT_MODEL}'...")
    _sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=HF_SENTIMENT_MODEL,
        top_k=None,  # return scores for ALL labels, not just the top one
        truncation=True,
        max_length=512,  # RoBERTa's max sequence length
    )
    print("Sentiment model loaded.")


# Phrases that signal "something is missing / should be improved" without
# using any explicitly negative words — e.g. "need more expertise",
# "could use more visuals", "lacking detail". General-purpose sentiment
# models (trained mostly on tweets/reviews with explicit emotional words)
# routinely score these as Positive, because there's no negative-sounding
# word for them to latch onto. This is a feedback-semantics gap, not
# something a sentiment model can fix on its own — so we catch it with a
# rule-based override instead of trusting the raw model here.
_CONSTRUCTIVE_REQUEST_PATTERNS = [
    r"\bneed(?:s|ed)?\s+more\b",
    r"\bcould\s+(?:use|have)\s+more\b",
    r"\bwould\s+(?:like|love|prefer)\s+more\b",
    r"\bwish(?:ed)?\s+(?:there\s+was|it\s+had)\b",
    r"\black(?:s|ed|ing)?\s+(?:of\s+)?\b",
    r"\bmissing\b",
    r"\bnot\s+enough\b",
    r"\btoo\s+(?:little|few|short|brief)\b",
    r"\bmore\s+\w+\s+(?:needed|required)\b",
]
_CONSTRUCTIVE_REQUEST_RE = re.compile(
    "|".join(_CONSTRUCTIVE_REQUEST_PATTERNS), re.IGNORECASE
)


def is_constructive_request(text):
    """True if the text reads as an implicit ask/gap ('need more X') rather
    than an explicit positive or negative statement."""
    return bool(_CONSTRUCTIVE_REQUEST_RE.search(text))


def analyze_sentiment(text):
    """Run the Hugging Face sentiment pipeline and return a dict shaped
    the same way the old TextBlob-based version did, so the rest of the
    app (DB schema, /admin_stats, CSV export) didn't need to change."""
    if _sentiment_pipeline is None:
        raise RuntimeError("Sentiment model is not loaded")

    raw = _sentiment_pipeline(text)[0]  # list of {"label": ..., "score": ...}
    scores = {r["label"].lower(): r["score"] for r in raw}
    pos = scores.get("positive", 0.0)
    neu = scores.get("neutral", 0.0)
    neg = scores.get("negative", 0.0)

    polarity = round(pos - neg, 4)
    subjectivity = round(1 - neu, 4)
    top_label = max(scores, key=scores.get)
    confidence = round(scores[top_label], 4)

    # Override: "need more X" / "lacking X" style feedback should never be
    # reported as Positive, even if the raw model leans that way — cap it
    # at Neutral so it doesn't misleadingly count toward positive stats.
    if is_constructive_request(text) and polarity > 0:
        polarity = min(polarity, 0.0)

    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "sentiment_label": classify_polarity(polarity),
        "subjectivity_label": classify_subjectivity(subjectivity),
        # Model's confidence in its top predicted class — a more faithful
        # "intensity" than abs(polarity) alone.
        "intensity": confidence,
        "key_phrases": extract_key_phrases(text),
        "word_count": len(text.split()),
    }


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
    """Create the feedback table if it doesn't already exist, and add any
    new columns to an existing table so upgrades don't require a manual
    migration step."""
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
    # New columns for the richer analysis — IF NOT EXISTS makes this safe
    # to run on every startup against an already-existing table.
    cur.execute("ALTER TABLE presentation_feedback ADD COLUMN IF NOT EXISTS subjectivity_label TEXT;")
    cur.execute("ALTER TABLE presentation_feedback ADD COLUMN IF NOT EXISTS intensity REAL;")
    cur.execute("ALTER TABLE presentation_feedback ADD COLUMN IF NOT EXISTS key_phrases TEXT;")
    cur.execute("ALTER TABLE presentation_feedback ADD COLUMN IF NOT EXISTS word_count INTEGER;")
    conn.commit()
    cur.close()
    conn.close()
    print("Database ready: presentation_feedback table checked/created/migrated.")


@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/admin')
def admin():
    return render_template('admin.html')


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json(silent=True) or {}
    text = data.get('feedback', '').strip()
    # Collapse repeated whitespace so word_count and analysis aren't
    # skewed by pasted text full of extra newlines/spaces.
    text = re.sub(r'\s+', ' ', text)

    if not text:
        return jsonify({"error": "Empty feedback"}), 400

    if len(text) > MAX_FEEDBACK_LENGTH:
        return jsonify({"error": f"Feedback too long (max {MAX_FEEDBACK_LENGTH} characters)"}), 400

    try:
        analysis = analyze_sentiment(text)
    except Exception as e:
        # Isolate analysis failures from DB failures — this tells you
        # immediately if the problem is the sentiment model vs. the database.
        print(f"SENTIMENT ANALYSIS ERROR: {e}")
        return jsonify({"error": f"Could not analyze feedback: {e}"}), 500

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO presentation_feedback
               (feedback_text, polarity, subjectivity, sentiment_label,
                subjectivity_label, intensity, key_phrases, word_count)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                text,
                analysis["polarity"],
                analysis["subjectivity"],
                analysis["sentiment_label"],
                analysis["subjectivity_label"],
                analysis["intensity"],
                ", ".join(analysis["key_phrases"]) if analysis["key_phrases"] else None,
                analysis["word_count"],
            )
        )
        conn.commit()
        cur.close()
        return jsonify({
            "status": "success",
            "polarity": analysis["polarity"],
            "label": analysis["sentiment_label"],
            "subjectivity_label": analysis["subjectivity_label"],
            "intensity": analysis["intensity"],
            "key_phrases": analysis["key_phrases"],
        })
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
        cur.execute("""
            SELECT
                AVG(polarity) as avg_pol,
                AVG(subjectivity) as avg_subj,
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE sentiment_label IN ('Positive', 'Very Positive')) as positive_count,
                COUNT(*) FILTER (WHERE sentiment_label = 'Neutral') as neutral_count,
                COUNT(*) FILTER (WHERE sentiment_label IN ('Negative', 'Very Negative')) as negative_count
            FROM presentation_feedback
        """)
        stats = cur.fetchone()
        cur.close()
        return jsonify({
            "feedback": rows,
            "stats": stats or {
                "avg_pol": 0, "avg_subj": 0, "total": 0,
                "positive_count": 0, "neutral_count": 0, "negative_count": 0
            }
        })
    except Exception as e:
        print(f"DB READ ERROR: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            conn.close()


@app.route('/admin_export')
def admin_export():
    """Export all feedback rows as a CSV file for download.
    Built entirely in memory (io.StringIO) so nothing is written to disk
    on the server — the file streams straight to the browser. CSV opens
    directly in Excel/Sheets/Numbers with no extra library needed."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM presentation_feedback ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close()
    except Exception as e:
        print(f"DB READ ERROR (export): {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            conn.close()

    headers = [
        "ID", "Feedback", "Sentiment", "Polarity", "Subjectivity Label",
        "Subjectivity Score", "Intensity", "Key Phrases", "Word Count", "Submitted At"
    ]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)

    for row in rows:
        created_at = row.get("created_at")
        writer.writerow([
            row.get("id"),
            row.get("feedback_text"),
            row.get("sentiment_label"),
            row.get("polarity"),
            row.get("subjectivity_label"),
            row.get("subjectivity"),
            row.get("intensity"),
            row.get("key_phrases"),
            row.get("word_count"),
            created_at.strftime("%Y-%m-%d %H:%M:%S") if created_at else None,
        ])

    csv_data = output.getvalue()
    output.close()

    filename = f"presentation_feedback_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


try:
    load_sentiment_pipeline()
except Exception as e:
    print(f"WARNING: could not load sentiment model at startup: {e}")

try:
    init_db()
except Exception as e:
    print(f"WARNING: could not initialize database at startup: {e}")

if __name__ == "__main__":
    print(f"Starting Flask app on port {APP_PORT} (set the PORT env var to override)")
    app.run(host="0.0.0.0", port=APP_PORT)
