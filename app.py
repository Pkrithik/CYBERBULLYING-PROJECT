import os
import pickle
import sqlite3
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from hybrid_engine import preprocess_text, hybrid_decision


app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})


# ============================================================
# LOAD ML MODEL AND VECTORIZER
# ============================================================

vectorizer = None
model = None

try:

    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("models/SGDClassifier_model.pkl", "rb") as f:
        model = pickle.load(f)

    print("Hybrid ML Model and Vectorizer loaded successfully.")

except Exception as e:

    print(f"!!! Error loading ML files: {str(e)}")


# ============================================================
# DATABASE
# ============================================================

def report_cyberbullying(content, user_ip, account_name="Anonymous"):

    """
    Logs a detected harmful message into the existing
    cyberbullying SQLite database.
    """

    db_path = "cyberbullying.db"

    try:

        conn = sqlite3.connect(db_path)

        c = conn.cursor()

        timestamp = datetime.now()

        c.execute(
            """
            INSERT INTO reports
            (content, user_ip, account_name, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (
                content,
                user_ip,
                account_name,
                timestamp
            )
        )

        conn.commit()

    except Exception as e:

        print(f"Database Error: {e}")

    finally:

        if "conn" in locals():
            conn.close()


# ============================================================
# VALIDATE COMMENT API
# ============================================================

@app.route("/validate-comment", methods=["POST"])
def validate_comment():

    # --------------------------------------------------------
    # Check ML model
    # --------------------------------------------------------

    if vectorizer is None or model is None:

        return jsonify({
            "error": "ML model is not properly loaded. Check server logs."
        }), 500


    # --------------------------------------------------------
    # Check JSON request
    # --------------------------------------------------------

    if not request.is_json:

        return jsonify({
            "error": "Request must be JSON"
        }), 400


    user_input = request.json.get("text", "")


    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if (
        not user_input
        or not isinstance(user_input, str)
        or not user_input.strip()
    ):

        return jsonify({
            "error": "Invalid or missing 'text' field"
        }), 400


    try:

        # ====================================================
        # 1. PREPROCESS TEXT
        # ====================================================

        cleaned_text = preprocess_text(user_input)


        # ====================================================
        # 2. ML PREDICTION
        # ====================================================

        if not cleaned_text:

            ml_pred = 0

        else:

            transformed_input = vectorizer.transform(
                [cleaned_text]
            )

            ml_pred = int(
                model.predict(transformed_input)[0]
            )


        # ====================================================
        # 3. HYBRID DECISION
        # ====================================================

        (
            pred_int,
            label_str,
            status_str,
            message_str,
            is_toxic,
            is_targeted,
            is_body_shaming
        ) = hybrid_decision(
            ml_pred,
            user_input
        )


        # ====================================================
        # 4. LOG HARMFUL CONTENT
        # ====================================================

        if pred_int == 1:

            user_ip = request.remote_addr or "Unknown IP"

            report_cyberbullying(
                user_input,
                user_ip
            )


        # ====================================================
        # 5. RETURN RESULT
        # ====================================================

        return jsonify({

            "prediction": pred_int,

            "label": label_str,

            "toxic_word_detected": is_toxic,

            "targeted_abuse": is_targeted,

            "body_shaming_detected": is_body_shaming,

            "status": status_str,

            "message": message_str

        })


    except Exception as e:

        print(
            f"!!! Prediction error: {str(e)}"
        )

        return jsonify({
            "error": "Failed to process comment"
        }), 500


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(debug=True)