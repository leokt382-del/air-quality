import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Air Quality & Health Impact",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Change these paths if your files are located somewhere else
DATA_PATH = "air_quality_health_impact_data.csv"

MODEL_PATH = "regression_model(1).pkl"

SCALER_PATH = "scaler.pkl"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.8;
        margin-bottom: 25px;
    }

    .card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        min-height: 130px;
    }

    .card-title {
        font-size: 16px;
        opacity: 0.75;
    }

    .card-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 10px;
    }

    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid rgba(128,128,128,0.3);
        margin-top: 20px;
    }

    .prediction-value {
        font-size: 42px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_PATH):
        return None

    return pd.read_csv(DATA_PATH)


df = load_data()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_scaler():

    if not os.path.exists(SCALER_PATH):
        return None

    return joblib.load(SCALER_PATH)


model = load_model()
scaler = load_scaler()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌍 Air Quality")

st.sidebar.markdown(
    """
    ### Navigation
    """
)

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Dashboard",
        "📊 Data Analysis",
        "🧹 Preprocessing",
        "📈 Model Performance",
        "🤖 Health Impact Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    """
    **Project**

    Air Quality & Health Impact Analysis

    **Target**

    HealthImpactScore

    **Model**

    Random Forest Regressor
    """
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">🌍 Air Quality & Health Impact Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore air quality data, health relationships, machine learning '
        'performance and health impact predictions.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DATA STATUS
    # --------------------------------------------------------

    if df is None:

        st.warning(
            "Dataset was not found. Check DATA_PATH at the top of app.py."
        )

    else:

        # ----------------------------------------------------
        # SUMMARY CARDS
        # ----------------------------------------------------

        rows = df.shape[0]
        columns = df.shape[1]

        missing_values = int(df.isnull().sum().sum())

        duplicates = int(df.duplicated().sum())

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">Dataset Records</div>
                    <div class="card-value">{rows:,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">Features / Columns</div>
                    <div class="card-value">{columns}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">Missing Values</div>
                    <div class="card-value">{missing_values:,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col4:

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">Duplicates</div>
                    <div class="card-value">{duplicates:,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        # ----------------------------------------------------
        # QUICK DATA PREVIEW
        # ----------------------------------------------------

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.divider()

        # ----------------------------------------------------
        # QUICK AIR QUALITY ANALYSIS
        # ----------------------------------------------------

        st.subheader("🌫️ Air Quality Overview")

        if "AQI" in df.columns:

            col1, col2 = st.columns(2)

            with col1:

                fig = px.histogram(
                    df,
                    x="AQI",
                    title="AQI Distribution",
                    marginal="box"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with col2:

                if "HealthImpactScore" in df.columns:

                    fig = px.scatter(
                        df,
                        x="AQI",
                        y="HealthImpactScore",
                        title="AQI vs Health Impact Score",
                        trendline="ols"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )


# ============================================================
# DATA ANALYSIS
# ============================================================

elif page == "📊 Data Analysis":

    st.title("📊 Data Analysis")

    if df is None:

        st.error(
            "Dataset not found. Check DATA_PATH."
        )

    else:

        st.subheader("Dataset Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Rows:**", df.shape[0])

        with col2:
            st.write("**Columns:**", df.shape[1])

        with col3:
            st.write("**Numerical Columns:**",
                     len(df.select_dtypes(include=np.number).columns))

        st.divider()

        # ----------------------------------------------------
        # COLUMN SELECTOR
        # ----------------------------------------------------

        numerical_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numerical_columns:

            selected_column = st.selectbox(
                "Select a numerical variable",
                numerical_columns
            )

            # Distribution
            st.subheader(
                f"📈 Distribution of {selected_column}"
            )

            fig = px.histogram(
                df,
                x=selected_column,
                marginal="box",
                title=f"{selected_column} Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        # ----------------------------------------------------
        # BOXPLOT
        # ----------------------------------------------------

        st.subheader("📦 Boxplot / Outlier Exploration")

        if numerical_columns:

            box_column = st.selectbox(
                "Choose a variable for boxplot",
                numerical_columns,
                key="boxplot_column"
            )

            fig = px.box(
                df,
                y=box_column,
                points="outliers",
                title=f"{box_column} Boxplot"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        # ----------------------------------------------------
        # BIVARIATE ANALYSIS
        # ----------------------------------------------------

        st.subheader("🔗 Bivariate Analysis")

        if len(numerical_columns) >= 2:

            col1, col2 = st.columns(2)

            with col1:

                x_column = st.selectbox(
                    "X-axis",
                    numerical_columns,
                    key="x_column"
                )

            with col2:

                y_column = st.selectbox(
                    "Y-axis",
                    numerical_columns,
                    key="y_column"
                )

            fig = px.scatter(
                df,
                x=x_column,
                y=y_column,
                title=f"{x_column} vs {y_column}",
                trendline="ols"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        # ----------------------------------------------------
        # CORRELATION
        # ----------------------------------------------------

        st.subheader("🔥 Correlation Analysis")

        correlation = df.select_dtypes(
            include=np.number
        ).corr()

        fig = px.imshow(
            correlation,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# PREPROCESSING
# ============================================================

elif page == "🧹 Preprocessing":

    st.title("🧹 Data Preprocessing")

    st.markdown(
        """
        The preprocessing stage prepares raw data for machine learning.
        """
    )

    if df is None:

        st.error("Dataset not found.")

    else:

        # ----------------------------------------------------
        # MISSING VALUES
        # ----------------------------------------------------

        st.subheader("1️⃣ Missing Values")

        missing = df.isnull().sum()

        missing_table = pd.DataFrame({
            "Column": missing.index,
            "Missing Values": missing.values
        })

        missing_table = missing_table[
            missing_table["Missing Values"] > 0
        ]

        if missing_table.empty:

            st.success(
                "No missing values found in the current dataset."
            )

        else:

            st.dataframe(
                missing_table,
                use_container_width=True
            )

        # ----------------------------------------------------
        # DUPLICATES
        # ----------------------------------------------------

        st.subheader("2️⃣ Duplicate Rows")

        duplicate_count = df.duplicated().sum()

        if duplicate_count == 0:

            st.success(
                "No duplicate rows found."
            )

        else:

            st.warning(
                f"{duplicate_count} duplicate rows found."
            )

        # ----------------------------------------------------
        # IQR EXPLANATION
        # ----------------------------------------------------

        st.subheader("3️⃣ Outlier Detection — IQR")

        st.markdown(
            """
            **IQR = Interquartile Range**

            The IQR method identifies unusually high or low observations.

            **Formula:**

            `IQR = Q3 - Q1`

            `Lower Bound = Q1 - 1.5 × IQR`

            `Upper Bound = Q3 + 1.5 × IQR`

            In the preprocessing workflow, extreme values can be
            **capped** rather than automatically deleted.
            """
        )

        # ----------------------------------------------------
        # OUTLIER TABLE
        # ----------------------------------------------------

        numerical_df = df.select_dtypes(
            include=np.number
        )

        outlier_results = []

        for column in numerical_df.columns:

            q1 = numerical_df[column].quantile(0.25)
            q3 = numerical_df[column].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            count = (
                (numerical_df[column] < lower) |
                (numerical_df[column] > upper)
            ).sum()

            outlier_results.append(
                {
                    "Column": column,
                    "Q1": q1,
                    "Q3": q3,
                    "IQR": iqr,
                    "Lower Bound": lower,
                    "Upper Bound": upper,
                    "Outlier Count": count
                }
            )

        outlier_table = pd.DataFrame(
            outlier_results
        )

        st.dataframe(
            outlier_table,
            use_container_width=True
        )

        # ----------------------------------------------------
        # SCALING
        # ----------------------------------------------------

        st.subheader("4️⃣ Feature Scaling")

        st.markdown(
            """
            ### StandardScaler

            StandardScaler transforms numerical features so that
            they are centered around a mean of 0 with a standard
            deviation of approximately 1.

            ### MinMaxScaler

            MinMaxScaler generally transforms values into the
            range 0 to 1.

            **Final ML workflow:** StandardScaler is used for
            the saved regression pipeline.
            """
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.title("📈 Machine Learning Model Performance")

    st.write(
        "This section displays the performance of the regression models."
    )

    if model is None:

        st.warning(
            "Saved regression model was not found."
        )

        st.info(
            "Expected file: " + MODEL_PATH
        )

    else:

        st.success(
            f"Loaded model: {type(model).__name__}"
        )

        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        st.subheader("🤖 Final Model")

        st.write(
            f"**Model:** {type(model).__name__}"
        )

        if hasattr(model, "n_features_in_"):

            st.write(
                f"**Number of input features:** "
                f"{model.n_features_in_}"
            )

        st.divider()

        # ----------------------------------------------------
        # METRICS FILE
        # ----------------------------------------------------

        metrics_path = "models/model_metrics.csv"

        if os.path.exists(metrics_path):

            metrics_df = pd.read_csv(
                metrics_path
            )

            st.subheader("📊 Model Comparison")

            st.dataframe(
                metrics_df,
                use_container_width=True
            )

            # Try to find R2 column
            r2_columns = [
                col for col in metrics_df.columns
                if col.lower() in ["r2", "r²", "r2_score"]
            ]

            if r2_columns:

                r2_column = r2_columns[0]

                fig = px.bar(
                    metrics_df,
                    x="Model",
                    y=r2_column,
                    title="R² Comparison"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        else:

            st.info(
                """
                No model_metrics.csv file has been saved yet.

                Once we save the actual regression metrics from Colab,
                this page can display the model comparison automatically.
                """
            )

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        if hasattr(model, "feature_importances_"):

            st.subheader("🎯 Feature Importance")

            feature_names = [
                "AQI",
                "PM10",
                "PM2_5",
                "NO2",
                "SO2",
                "O3",
                "Temperature",
                "Humidity",
                "WindSpeed",
                "RespiratoryCases",
                "CardiovascularCases",
                "HospitalAdmissions"
            ]

            importances = model.feature_importances_

            if len(feature_names) == len(importances):

                importance_df = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "Importance": importances
                    }
                ).sort_values(
                    "Importance",
                    ascending=False
                )

                fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Feature Importance"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.warning(
                    "Feature names do not match the number of model features. "
                    "We need to verify the exact training feature order."
                )


# ============================================================
# PREDICTION
# ============================================================

elif page == "🤖 Health Impact Prediction":

    st.title("🤖 Health Impact Prediction")

    st.markdown(
        """
        Enter the environmental and health-related values below.
        The trained machine learning model will estimate the
        **HealthImpactScore**.
        """
    )

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if model is None:

        st.error(
            "Regression model not found."
        )

        st.code(
            MODEL_PATH
        )

        st.stop()

    if scaler is None:

        st.error(
            "Scaler not found."
        )

        st.code(
            SCALER_PATH
        )

        st.stop()

    st.success(
        f"Model loaded successfully: {type(model).__name__}"
    )

    st.divider()

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    st.subheader("🌫️ Air Quality & Environmental Inputs")

    col1, col2, col3 = st.columns(3)

    with col1:

        AQI = st.number_input(
            "AQI",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        PM10 = st.number_input(
            "PM10",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

        PM2_5 = st.number_input(
            "PM2.5",
            min_value=0.0,
            value=25.0,
            step=1.0
        )

        NO2 = st.number_input(
            "NO2",
            min_value=0.0,
            value=30.0,
            step=1.0
        )

    with col2:

        SO2 = st.number_input(
            "SO2",
            min_value=0.0,
            value=10.0,
            step=1.0
        )

        O3 = st.number_input(
            "O3",
            min_value=0.0,
            value=40.0,
            step=1.0
        )

        Temperature = st.number_input(
            "Temperature",
            value=25.0,
            step=0.5
        )

        Humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0
        )

    with col3:

        WindSpeed = st.number_input(
            "Wind Speed",
            min_value=0.0,
            value=10.0,
            step=0.5
        )

        RespiratoryCases = st.number_input(
            "Respiratory Cases",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        CardiovascularCases = st.number_input(
            "Cardiovascular Cases",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

        HospitalAdmissions = st.number_input(
            "Hospital Admissions",
            min_value=0.0,
            value=20.0,
            step=1.0
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🔮 Predict Health Impact Score",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        try:

            # IMPORTANT:
            # This feature order must match the order used
            # when training the model.

            input_data = pd.DataFrame(
                [
                    [
                        AQI,
                        PM10,
                        PM2_5,
                        NO2,
                        SO2,
                        O3,
                        Temperature,
                        Humidity,
                        WindSpeed,
                        RespiratoryCases,
                        CardiovascularCases,
                        HospitalAdmissions
                    ]
                ],
                columns=[
                    "AQI",
                    "PM10",
                    "PM2_5",
                    "NO2",
                    "SO2",
                    "O3",
                    "Temperature",
                    "Humidity",
                    "WindSpeed",
                    "RespiratoryCases",
                    "CardiovascularCases",
                    "HospitalAdmissions"
                ]
            )

            # Scale input
            scaled_input = scaler.transform(
                input_data
            )

            # Prediction
            prediction = model.predict(
                scaled_input
            )[0]

            # Display prediction
            st.markdown(
                f"""
                <div class="prediction-box">
                    <div>Predicted Health Impact Score</div>
                    <div class="prediction-value">
                        {prediction:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "Prediction generated successfully."
            )

            # Show input data
            with st.expander("🔍 View Input Data"):

                st.dataframe(
                    input_data,
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)

            st.warning(
                """
                This usually means the feature order or number of
                features does not exactly match what the saved model
                expects. We will verify this against the Colab model
                before final deployment.
                """
            )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Air Quality & Health Impact Analysis • Machine Learning Project"
)
