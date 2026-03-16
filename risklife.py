import streamlit as st
import pandas as pd
import joblib
import kagglehub
import os
import sqlite3
import hashlib
import plotly.express as px

# ---------------- DATABASE ---------------- #

def init_db():
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users(
        username TEXT PRIMARY KEY,
        password TEXT
        )
    """)
    conn.commit()
    conn.close()


def make_hashes(password):
    return hashlib.sha256(password.encode()).hexdigest()


def add_userdata(username, password):
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()

    try:
        c.execute("INSERT INTO users(username,password) VALUES (?,?)",
                  (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def login_user(username, password):
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()

    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, password))

    data = c.fetchall()
    conn.close()

    return data


# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_assets():
    try:
        model = joblib.load("attrition_model.pkl")
        encoders = joblib.load("encoders.pkl")
        return model, encoders
    except Exception:
        st.warning("Model not found. Running in UI demo mode.")
        return None, None


# ---------------- APP CONFIG ---------------- #

st.set_page_config(
    page_title="Universal HR Predictor",
    layout="wide"
)

init_db()

if "auth_status" not in st.session_state:
    st.session_state.auth_status = False


# ---------------- LOGIN PAGE ---------------- #

if not st.session_state.auth_status:

    cols = st.columns([1, 2, 1])

    with cols[1]:

        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        # LOGIN
        with tab1:
            st.subheader("Login")

            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")

            if st.button("Login"):

                hashed_pswd = make_hashes(pwd)

                if login_user(user, hashed_pswd):
                    st.session_state.auth_status = True
                    st.session_state.username = user
                    st.rerun()

                else:
                    st.error("Invalid Credentials")

        # SIGNUP
        with tab2:

            st.subheader("Create Account")

            new_user = st.text_input("Username", key="signup_user")
            new_pwd = st.text_input("Password", type="password", key="signup_pwd")

            if st.button("Register"):

                if new_user and new_pwd:

                    if add_userdata(new_user, make_hashes(new_pwd)):
                        st.success("Account created. Login now.")

                    else:
                        st.error("Username already exists")

                else:
                    st.warning("Fill all fields")

    st.stop()


# ---------------- MAIN DASHBOARD ---------------- #

model, encoders = load_assets()

st.sidebar.title(f"Welcome {st.session_state.username}")

if st.sidebar.button("Logout"):
    st.session_state.auth_status = False
    st.rerun()

st.title("Universal Employee Risk Scanner")


# ---------------- DATA INPUT ---------------- #

user_input = st.text_input(
    "Enter Kaggle Dataset Slug OR CSV URL",
    placeholder="username/dataset-name OR https://file.csv"
)

if st.button("Analyze Data"):

    try:
        with st.spinner("Loading dataset..."):

            # URL CSV
            if user_input.startswith(("http://", "https://")):

                df = pd.read_csv(user_input)
                source = "Direct URL"

            # Kaggle dataset
            else:

                path = kagglehub.dataset_download(user_input)

                files = [f for f in os.listdir(path) if f.endswith(".csv")]

                df = pd.read_csv(os.path.join(path, files[0]))

                source = f"Kaggle ({files[0]})"


            # ---------------- PREPROCESS ---------------- #

            X = df.copy()

            cols_to_drop = [
                "EmployeeCount",
                "Over18",
                "StandardHours",
                "EmployeeNumber",
                "Attrition"
            ]

            X = X.drop(columns=[c for c in cols_to_drop if c in X.columns],
                       errors="ignore")


            # ---------------- PREDICTION ---------------- #

            if model and encoders:

                for col, le in encoders.items():

                    if col in X.columns:

                        X[col] = X[col].apply(
                            lambda x: le.transform([str(x)])[0]
                            if str(x) in le.classes_
                            else 0
                        )

                risk_probs = model.predict_proba(X)[:, 1]

                df["Risk_Score"] = risk_probs

            else:

                df["Risk_Score"] = 0.35


            df["Status"] = df["Risk_Score"].apply(
                lambda x:
                "High Risk" if x > 0.7
                else ("Warning" if x > 0.4 else "Stable")
            )


            st.session_state.df = df
            st.session_state.source = source

            st.success("Data Loaded")

    except Exception as e:

        st.error(e)


# ---------------- DASHBOARD ---------------- #

if "df" in st.session_state:

    df = st.session_state.df

    # Department Filter
    if "Department" in df.columns:

        all_depts = df["Department"].unique().tolist()

        selected = st.sidebar.multiselect(
            "Filter Department",
            all_depts,
            default=all_depts
        )

        df_display = df[df["Department"].isin(selected)]

    else:

        df_display = df


    # Metrics
    m1, m2, m3 = st.columns(3)

    m1.metric("Employees", len(df_display))
    m2.metric("Avg Risk", f"{df_display['Risk_Score'].mean():.1%}")
    m3.info(f"Source: {st.session_state.source}")


    # Charts
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Risk Distribution")

        fig = px.pie(
            df_display,
            names="Status",
            hole=0.4
        )

        st.plotly_chart(fig, use_container_width=True)


    with col2:

        if "Department" in df_display.columns:

            st.subheader("Department Risk")

            avg = df_display.groupby("Department")["Risk_Score"].mean().reset_index()

            fig2 = px.bar(
                avg,
                x="Department",
                y="Risk_Score",
                color="Risk_Score",
                color_continuous_scale="Reds"
            )

            st.plotly_chart(fig2, use_container_width=True)


    # Table
    st.subheader("Employee Report")

    st.dataframe(
        df_display.sort_values("Risk_Score", ascending=False),
        use_container_width=True
    )