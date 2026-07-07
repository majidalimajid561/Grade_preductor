import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ─────────────────────────────────────────
# 1. PAGE  CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="centered"
)

# ─────────────────────────────────────────
# 2. TRAIN THE MODEL
#    We train once and cache it so the app
#    does not retrain on every button click.
# ─────────────────────────────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 200

    study_hours = np.random.uniform(1, 10, n)
    attendance  = np.random.uniform(40, 100, n)
    past_score  = np.random.uniform(30, 90, n)

    # Grade formula: study hours matter most
    grade = (study_hours * 3.5 +
             attendance  * 0.3 +
             past_score  * 0.4 +
             np.random.normal(0, 3, n))
    grade = np.clip(grade, 0, 100)

    df = pd.DataFrame({
        "study_hours": study_hours,
        "attendance":  attendance,
        "past_score":  past_score,
        "final_grade": grade
    })

    X = df[["study_hours", "attendance", "past_score"]]
    y = df["final_grade"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    return model, mae, r2

model, mae, r2 = train_model()

# ─────────────────────────────────────────
# 3. HEADER
# ─────────────────────────────────────────
st.title("🎓 Student Grade Predictor")
st.write("Enter your details below and find out your predicted final grade.")
st.divider()

# ─────────────────────────────────────────
# 4. USER INPUTS
# ─────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    study_hours = st.number_input(
        "📚 Daily Study Hours",
        min_value=0.0,
        max_value=12.0,
        value=4.0,
        step=0.5,
        help="How many hours do you study each day on average?"
    )

with col2:
    attendance = st.number_input(
        "🏫 Attendance (%)",
        min_value=0,
        max_value=100,
        value=75,
        step=1,
        help="Your class attendance percentage"
    )

with col3:
    past_score = st.number_input(
        "📝 Past Exam Score",
        min_value=0,
        max_value=100,
        value=60,
        step=1,
        help="Your score in the last exam (out of 100)"
    )

st.divider()

# ─────────────────────────────────────────
# 5. PREDICT BUTTON
# ─────────────────────────────────────────
if st.button("🔮  Predict My Grade", use_container_width=True, type="primary"):

    input_data = np.array([[study_hours, attendance, past_score]])
    prediction = round(float(model.predict(input_data)[0]), 1)

    # Clamp to 0–100 just in case
    prediction = max(0, min(100, prediction))

    # ── Result card ──────────────────────
    st.subheader("Your Predicted Grade")

    if prediction >= 80:
        st.info(f"🏆  **{prediction} / 100** With `Grade A+` — Excellent! Keep it up.")
        feedback = "You are on track for an A. Maintain your current habits."
    elif prediction >= 70:
        st.info(f"🏆  **{prediction} / 100** With `Grade A` — Excellent! Keep it up.")
    elif prediction >= 60:
        st.info(f"👍  **{prediction} / 100** wiht `Grade B` — Good standing.")
        feedback = "You are passing. Increasing daily study hours will push you higher."
    elif prediction >= 50:
        st.info(f"👍  **{prediction} / 100** wiht `Grade C` — Good standing.")
        feedback = "Try to improve your attendance and add 1–2 more study hours daily."
    elif prediction >= 40:
        st.warning(f"⚠️  **{prediction} / 100** `Grade D` — Needs improvement.")
        feedback = "Try to improve your attendance and add 1–2 more study hours daily."
    else:
        st.error(f"🚨  **{prediction} / 100** — At risk of failing.")
        feedback = "You need significant improvement in attendance and study hours."

    st.write(feedback)
    st.divider()

    # ── What-if tip ──────────────────────
    tip_hours  = round(float(model.predict([[study_hours + 1, attendance, past_score]])[0]), 1)
    tip_hours  = max(0, min(100, tip_hours))
    extra_gain = round(tip_hours - prediction, 1)

    if study_hours < 11 and extra_gain > 0:
        st.info(
            f"💡 **Quick tip:** Studying just **1 more hour per day** "
            f"would raise your grade by approximately **{extra_gain} points** "
            f"(to {tip_hours})."
        )

    # ── Feature impact ───────────────────
    st.subheader("What affects your grade the most?")

    features    = ["Study Hours", "Attendance", "Past Score"]
    coefs       = model.coef_
    impact_df   = pd.DataFrame({
        "Feature": features,
        "Impact per unit": [round(c, 3) for c in coefs]
    }).sort_values("Impact per unit", ascending=False)

    st.dataframe(impact_df, hide_index=True, use_container_width=True)
    st.caption(
        "Impact per unit = how much your grade changes for each 1-unit "
        "increase in that feature (1 extra study hour, 1% more attendance, etc.)"
    )

# ─────────────────────────────────────────
# 6. MODEL INFO (collapsible)
# ─────────────────────────────────────────
with st.expander("📊 Model performance info"):
    c1, c2 = st.columns(2)
    c1.metric("Mean Absolute Error", f"{mae:.2f} pts",
              help="On average the prediction is off by this many grade points")
    c2.metric("R² Score", f"{r2:.2f}",
              help="1.0 = perfect model. Above 0.85 is very good.")
    st.caption(
        "This model uses **Linear Regression** trained on 200 simulated "
        "student records. Replace the training data with real records "
        "to make predictions more accurate."
    )

# ─────────────────────────────────────────
# 7. FOOTER
# ─────────────────────────────────────────
st.divider()
st.caption("Built with Scikit-learn + Streamlit · Grade Predictor v1.0")
