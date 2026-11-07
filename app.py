
from flask import (
    Flask, render_template, request,
    redirect, session, url_for
)
import sqlite3
import joblib
import pandas as pd
from pathlib import Path
from functools import wraps
from werkzeug.security import check_password_hash
import os

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "development-only-change-this-secret"
)

BASE_DIR = Path(__file__).resolve().parent

# Load trained model
model = joblib.load(BASE_DIR / "vehicle_insurance_model.pkl")

FEATURES = [
    "driver_age",
    "past_accidents",
    "speeding_tickets",
    "annual_mileage"
]


# Connect to existing users database
def get_db_connection():
    conn = sqlite3.connect(BASE_DIR / "users.db")
    conn.row_factory = sqlite3.Row
    return conn


# Check login status
def login_required(view_function):
    @wraps(view_function)
    def wrapped(*args, **kwargs):
        if "username" not in session or "role" not in session:
            return redirect(url_for("login"))
        return view_function(*args, **kwargs)
    return wrapped


# Login page
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        role = request.form.get("role", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_db_connection()

        try:
            account = conn.execute(
                """
                SELECT username, password, role
                FROM users
                WHERE username = ? AND role = ?
                """,
                (username, role)
            ).fetchone()
        finally:
            conn.close()

        if account and check_password_hash(
            account["password"], password
        ):
            session.clear()
            session["username"] = account["username"]
            session["role"] = account["role"]

            return redirect(url_for("dashboard"))

        return render_template(
            "login.html",
            error="Invalid role, username, or password."
        ), 401

    if "username" in session:
        return redirect(url_for("dashboard"))

    return render_template("login.html")


# Dashboard
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template(
        "dashboard.html",
        role=session["role"],
        username=session["username"]
    )


# Admin User Management
@app.route("/users", methods=["GET", "POST"])
@login_required
def manage_users():
    # Only Admin can manage users
    if session.get("role") != "Admin":
        return "Access denied. Admin only.", 403

    message = None
    error = None

    if request.method == "POST":
        action = request.form.get("action", "")

        if action == "add":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            role = request.form.get("role", "").strip()

            if not username or not password:
                error = "Username and password are required."

            elif role not in ["Admin", "Analyst", "Underwriter"]:
                error = "Please select a valid role."

            elif len(password) < 8:
                error = "Password must contain at least 8 characters."

            else:
                from werkzeug.security import generate_password_hash

                conn = get_db_connection()

                try:
                    existing = conn.execute(
                        "SELECT id FROM users WHERE username = ?",
                        (username,)
                    ).fetchone()

                    if existing:
                        error = "This username already exists."
                    else:
                        conn.execute(
                            """
                            INSERT INTO users (username, password, role)
                            VALUES (?, ?, ?)
                            """,
                            (
                                username,
                                generate_password_hash(password),
                                role
                            )
                        )
                        conn.commit()
                        message = "User added successfully."

                except sqlite3.Error:
                    error = "Could not save the user. Please try again."

                finally:
                    conn.close()

        elif action == "delete":
            user_id = request.form.get("user_id", type=int)

            if user_id is None:
                error = "Invalid user selected."
            else:
                conn = get_db_connection()

                try:
                    target = conn.execute(
                        "SELECT id, username, role FROM users WHERE id = ?",
                        (user_id,)
                    ).fetchone()

                    if target is None:
                        error = "User not found."

                    elif target["username"] == session["username"]:
                        error = "You cannot delete your own account."

                    elif target["role"] == "Admin":
                        admin_count = conn.execute(
                            "SELECT COUNT(*) FROM users WHERE role = 'Admin'"
                        ).fetchone()[0]

                        if admin_count <= 1:
                            error = "At least one Admin account must remain."
                        else:
                            conn.execute(
                                "DELETE FROM users WHERE id = ?",
                                (user_id,)
                            )
                            conn.commit()
                            message = "User deleted successfully."
                    else:
                        conn.execute(
                            "DELETE FROM users WHERE id = ?",
                            (user_id,)
                        )
                        conn.commit()
                        message = "User deleted successfully."

                except sqlite3.Error:
                    error = "Could not delete the user."

                finally:
                    conn.close()

    conn = get_db_connection()

    try:
        users = conn.execute(
            "SELECT id, username, role FROM users ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    return render_template(
        "users.html",
        role=session["role"],
        username=session["username"],
        users=users,
        message=message,
        error=error
    )


@app.route("/model-performance")
@login_required
def model_performance():
    # Admin-only access
    if session.get("role") not in ["Admin", "Analyst"]:
     return "Access denied", 403

    metrics_path = BASE_DIR / "model_metrics.csv"
    matrix_path = BASE_DIR / "confusion_matrix.csv"
    report_path = BASE_DIR / "classification_report.csv"

    # Check required evaluation files
    if not metrics_path.exists():
        return "model_metrics.csv not found. Run model_evaluation.py first.", 404

    if not matrix_path.exists():
        return "confusion_matrix.csv not found. Run model_evaluation.py first.", 404

    if not report_path.exists():
        return "classification_report.csv not found. Run model_evaluation.py first.", 404

    # Read CSV files
    metrics_df = pd.read_csv(metrics_path)
    matrix_df = pd.read_csv(matrix_path, index_col=0)
    report_df = pd.read_csv(report_path, index_col=0)

    if metrics_df.empty:
        return "Model metrics file is empty.", 500

    row = metrics_df.iloc[0]

    def format_metric(value, percentage=False):
        if pd.isna(value):
            return "N/A"
        value = float(value)
        return f"{value:.2%}" if percentage else f"{value:.4f}"

    metrics = {
        "Accuracy": format_metric(row["accuracy"], True),
        "Precision": format_metric(row["precision"]),
        "Recall": format_metric(row["recall"]),
        "F1-Score": format_metric(row["f1_score"]),
        "ROC-AUC": format_metric(row["roc_auc"])
    }

    matrix = matrix_df.values.tolist()

    # Convert classification report to display-friendly rows
    report_columns = [
        column for column in
        ["precision", "recall", "f1-score", "support"]
        if column in report_df.columns
    ]

    report_rows = []

    for class_name, values in report_df.iterrows():
        report_rows.append({
            "class_name": str(class_name),
            "values": [
                (
                    f"{float(values[column]):.4f}"
                    if column != "support"
                    else f"{float(values[column]):.0f}"
                )
                if pd.notna(values[column])
                else "N/A"
                for column in report_columns
            ]
        })

    return render_template(
        "model_performance.html",
        metrics=metrics,
        matrix=matrix,
        report_rows=report_rows,
        report_columns=report_columns,
        role=session["role"]
    )

# Prediction form
@app.route("/prediction")
@login_required
def prediction():
    return render_template(
        "prediction.html",
        role=session["role"]
    )



# Predict claim probability
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        driver_age = int(request.form["driver_age"])
        past_accidents = int(request.form["past_accidents"])
        speeding_tickets = int(request.form["speeding_tickets"])
        annual_mileage = float(request.form["annual_mileage"])

        # Validate inputs
        if not 18 <= driver_age <= 100:
            return "Driver age must be between 18 and 100.", 400

        if (
            past_accidents < 0
            or speeding_tickets < 0
            or annual_mileage < 0
        ):
            return "Values cannot be negative.", 400

        # Prepare model input
        input_data = pd.DataFrame(
            [[
                driver_age,
                past_accidents,
                speeding_tickets,
                annual_mileage
            ]],
            columns=FEATURES
        )

        # Predict probability
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError("Claim class 1 is missing.")

        claim_probability = (
            float(probabilities[classes.index(1)]) * 100
        )

        probability = round(claim_probability, 2)
        prediction_value = int(model.predict(input_data)[0])

        # Categorize estimated risk
        if claim_probability <= 30:
            risk_label = "Low Claim Risk"
        elif claim_probability <= 60:
            risk_label = "Medium Claim Risk"
        else:
            risk_label = "High Claim Risk"

        # Generate understandable risk indicators
        risk_reasons = []

        if past_accidents >= 2:
            risk_reasons.append(
                f"The driver has {past_accidents} past accidents."
            )
        elif past_accidents == 1:
            risk_reasons.append(
                "The driver has a previous accident record."
            )

        if speeding_tickets >= 2:
            risk_reasons.append(
                f"The driver has {speeding_tickets} speeding tickets."
            )
        elif speeding_tickets == 1:
            risk_reasons.append(
                "The driver has a speeding violation record."
            )

        if annual_mileage >= 20000:
            risk_reasons.append(
                "High annual mileage indicates greater driving exposure."
            )

        if driver_age < 25:
            risk_reasons.append(
                "The driver belongs to a younger age group."
            )

        if not risk_reasons:
            risk_reasons.append(
                "No major risk indicators were identified from "
                "the simple checks above. The estimated risk reflects "
                "the combined input pattern considered by the model."
            )

        # Risk description
        risk_description = (
            "This is a model-estimated claim probability, "
            "not a guarantee of a future claim or an actual "
            "insurance decision."
        )

        # Save assessment history
        history_file = BASE_DIR / "assessment_history.csv"

        record = pd.DataFrame([{
            "driver_age": driver_age,
            "past_accidents": past_accidents,
            "speeding_tickets": speeding_tickets,
            "annual_mileage": annual_mileage,
            "prediction": prediction_value,
            "claim_probability": probability
        }])

        record.to_csv(
            history_file,
            mode="a",
            header=not history_file.exists()
                or history_file.stat().st_size == 0,
            index=False
        )

        # Display result page
        return render_template(
            "result.html",
            driver_age=driver_age,
            past_accidents=past_accidents,
            speeding_tickets=speeding_tickets,
            annual_mileage=annual_mileage,
            risk_label=risk_label,
            risk_description=risk_description,
            probability=probability,
            risk_reasons=risk_reasons,
            role=session["role"]
        )

    except (ValueError, KeyError):
        return "Please enter valid values in all fields.", 400




# Assessment History page
@app.route("/history")
@login_required
def history():
    history_file = BASE_DIR / "assessment_history.csv"

    if history_file.exists() and history_file.stat().st_size > 0:
        df = pd.read_csv(history_file).fillna("")
        records = df.to_dict(orient="records")
    else:
        records = []

    return render_template(
        "history.html",
        role=session["role"],
        username=session["username"],
        records=records
    )


# Logout
@app.route("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/save-underwriting-decision", methods=["POST"])
@login_required
def save_underwriting_decision():
    # Only Underwriters can save underwriting decisions.
    if session.get("role") != "Underwriter":
        return "Access denied", 403

    history_file = BASE_DIR / "assessment_history.csv"

    if not history_file.exists():
        return "Assessment history file not found.", 404

    decision = request.form.get("underwriting_decision", "").strip()

    allowed_decisions = {
        "Accept",
        "Reject",
        "Refer for Review"
    }

    if decision not in allowed_decisions:
        return "Invalid underwriting decision.", 400

    try:
        df = pd.read_csv(history_file)

        required_columns = [
            "driver_age",
            "past_accidents",
            "speeding_tickets",
            "annual_mileage"
        ]

        if not all(column in df.columns for column in required_columns):
            return "Assessment history columns do not match.", 500

        # Find the most recent matching assessment.
        matching_rows = pd.Series(True, index=df.index)

        for column in required_columns:
            submitted_value = float(request.form[column])
            column_values = pd.to_numeric(df[column], errors="coerce")
            matching_rows &= column_values.eq(submitted_value)

        matched_indices = df.index[matching_rows]

        if len(matched_indices) == 0:
            return (
                "Matching assessment not found. Please run the assessment again.",
                404
            )

        latest_index = matched_indices[-1]

        if "underwriting_decision" not in df.columns:
            df["underwriting_decision"] = ""

        if "reviewed_by" not in df.columns:
            df["reviewed_by"] = ""

        df.at[latest_index, "underwriting_decision"] = decision
        df.at[latest_index, "reviewed_by"] = session.get("username", "")

        df.to_csv(history_file, index=False)

        return redirect(url_for("history"))

    except (ValueError, KeyError):
        return "Invalid assessment details.", 400
    except Exception:
        app.logger.exception("Unable to save underwriting decision")
        return "Unable to save the decision. Check the application log.", 500

if __name__ == "__main__":
    import webbrowser
    from threading import Timer

    url = "http://127.0.0.1:5000"

    Timer(1.5, lambda: webbrowser.open(url)).start()

    app.run(debug=True, use_reloader=False)
