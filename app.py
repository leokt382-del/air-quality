import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
# FILE PATHS
# ============================================================

DATA_PATH = "data/air_quality_health_impact_data.csv"

MODEL_PATH = "models/regression_model.pkl"

SCALER_PATH = "models/scaler.pkl"


# ============================================================
# MODEL FEATURES
# These MUST match the features used during model training
# ============================================================

FEATURES = [
    "AQI",
    "PM10",
    "PM2_5",
    "NO2",
    "SO2",
    "O3"
]

TARGET = "HealthImpactScore"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.3);
        text-align: center;
        min-height: 120px;
    }

    .card-title {
        font-size: 15px;
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
        border: 2px solid rgba(128,128,128,0.3);
        text-align: center;
        margin-top: 20px;
    }

    .prediction-label {
        font-size: 18px;
        opacity: 0.75;
    }

    .prediction-value {
        font-size: 48px;
        font-weight: 700;
        margin-top: 10px;
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


# ============================================================
# LOAD SCALER
# ============================================================

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

st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "🏠 Dashboard",
        "📊 Data Analysis",
        "🧹 Preprocessing",
        "📈 Model Performance",
        "🤖 Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    ### Machine Learning

    **Problem:** Regression

    **Target:** HealthImpactScore

    **Model:** Random Forest

    **Scaler:** StandardScaler
    """
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">'
        '🌍 Air Quality & Health Impact Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Interactive analysis and machine learning prediction '
        'of health impact from air quality measurements.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DATA CHECK
    # --------------------------------------------------------

    if df is None:

        st.warning(
            "Dataset not found."
        )

        st.info(
            f"Expected dataset location: {DATA_PATH}"
        )

        st.stop()

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    rows = df.shape[0]

    columns = df.shape[1]

    missing_values = int(
        df.isnull().sum().sum()
    )

    duplicates = int(
        df.duplicated().sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">
                    Dataset Records
                </div>
                <div class="card-value">
                    {rows:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">
                    Columns
                </div>
                <div class="card-value">
                    {columns}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">
                    Missing Values
                </div>
                <div class="card-value">
                    {missing_values:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">
                    Duplicate Rows
                </div>
                <div class="card-value">
                    {duplicates:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    if TARGET in df.columns:

        st.subheader(
            "🎯 Health Impact Score Distribution"
        )

        fig = px.histogram(
            df,
            x=TARGET,
            marginal="box",
            title="HealthImpactScore Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # AQI VS HEALTH IMPACT
    # --------------------------------------------------------

    if (
        "AQI" in df.columns
        and TARGET in df.columns
    ):

        st.subheader(
            "🌫️ AQI vs Health Impact"
        )

        fig = px.scatter(
            df,
            x="AQI",
            y=TARGET,
            trendline="ols",
            title="AQI vs HealthImpactScore"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# DATA ANALYSIS
# ============================================================

elif page == "📊 Data Analysis":

    st.title("📊 Exploratory Data Analysis")

    if df is None:

        st.error(
            "Dataset not found."
        )

        st.stop()

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    st.subheader("📋 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write(
            f"**Rows:** {df.shape[0]:,}"
        )

    with col2:

        st.write(
            f"**Columns:** {df.shape[1]}"
        )

    with col3:

        numerical_count = len(
            df.select_dtypes(
                include=np.number
            ).columns
        )

        st.write(
            f"**Numerical Columns:** {numerical_count}"
        )

    st.divider()

    # --------------------------------------------------------
    # COLUMN SELECTION
    # --------------------------------------------------------

    numerical_columns = (
        df
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    if len(numerical_columns) > 0:

        selected_column = st.selectbox(
            "Select a variable",
            numerical_columns
        )

        # ----------------------------------------------------
        # HISTOGRAM
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # BOX PLOT
        # ----------------------------------------------------

        st.subheader(
            f"📦 Boxplot of {selected_column}"
        )

        fig = px.box(
            df,
            y=selected_column,
            points="outliers",
            title=f"{selected_column} Boxplot"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # BIVARIATE ANALYSIS
    # --------------------------------------------------------

    st.subheader("🔗 Bivariate Analysis")

    if len(numerical_columns) >= 2:

        col1, col2 = st.columns(2)

        with col1:

            x_column = st.selectbox(
                "X-axis",
                numerical_columns,
                key="x_axis"
            )

        with col2:

            y_column = st.selectbox(
                "Y-axis",
                numerical_columns,
                key="y_axis"
            )

        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            trendline="ols",
            title=f"{x_column} vs {y_column}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    st.subheader("🔥 Correlation Heatmap")

    correlation = (
        df
        .select_dtypes(include=np.number)
        .corr()
    )

    fig = px.imshow(
        correlation,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix"
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

    if df is None:

        st.error(
            "Dataset not found."
        )

        st.stop()

    st.markdown(
        """
        Data preprocessing prepares the raw dataset before
        machine learning.
        """
    )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    st.subheader("1️⃣ Missing Values")

    missing = df.isnull().sum()

    missing_table = pd.DataFrame(
        {
            "Column": missing.index,
            "Missing Values": missing.values
        }
    )

    missing_table = missing_table[
        missing_table["Missing Values"] > 0
    ]

    if missing_table.empty:

        st.success(
            "No missing values found."
        )

    else:

        st.dataframe(
            missing_table,
            use_container_width=True
        )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    st.subheader("2️⃣ Duplicate Rows")

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count == 0:

        st.success(
            "No duplicate rows found."
        )

    else:

        st.warning(
            f"{duplicate_count} duplicate rows found."
        )

    # --------------------------------------------------------
    # OUTLIERS
    # --------------------------------------------------------

    st.subheader("3️⃣ Outlier Detection")

    st.markdown(
        """
        ### IQR Method

        **IQR = Interquartile Range**

        Formula:

        `IQR = Q3 - Q1`

        `Lower Bound = Q1 - 1.5 × IQR`

        `Upper Bound = Q3 + 1.5 × IQR`

        Values outside these bounds are considered potential
        outliers.
        """
    )

    numerical_df = df.select_dtypes(
        include=np.number
    )

    outlier_results = []

    for column in numerical_df.columns:

        q1 = numerical_df[column].quantile(0.25)

        q3 = numerical_df[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr

        upper_bound = q3 + 1.5 * iqr

        outlier_count = (
            (
                numerical_df[column] < lower_bound
            )
            |
            (
                numerical_df[column] > upper_bound
            )
        ).sum()

        outlier_results.append(
            {
                "Column": column,
                "Q1": round(q1, 2),
                "Q3": round(q3, 2),
                "IQR": round(iqr, 2),
                "Lower Bound": round(
                    lower_bound,
                    2
                ),
                "Upper Bound": round(
                    upper_bound,
                    2
                ),
                "Outlier Count": int(
                    outlier_count
                )
            }
        )

    outlier_table = pd.DataFrame(
        outlier_results
    )

    st.dataframe(
        outlier_table,
        use_container_width=True
    )

    # --------------------------------------------------------
    # SCALING
    # --------------------------------------------------------

    st.subheader("4️⃣ Feature Scaling")

    st.markdown(
        """
        ### StandardScaler

        StandardScaler transforms numerical variables so that
        they are centered around a mean of approximately 0
        with a standard deviation of approximately 1.

        ### MinMaxScaler

        MinMaxScaler generally transforms values into a
        0–1 range.

        ### Model Workflow

        The deployed regression model uses **StandardScaler**.
        """
    )

    # --------------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------------

    st.subheader("5️⃣ Feature Selection")

    st.write(
        "Features used by the regression model:"
    )

    st.code(
        "\n".join(FEATURES)
    )

    st.write(
        f"Target variable: **{TARGET}**"
    )

    st.info(
        "The target variable is not used as an input feature."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    # --------------------------------------------------------
    # MODEL CHECK
    # --------------------------------------------------------

    if model is None:

        st.error(
            "Regression model not found."
        )

        st.info(
            f"Expected: {MODEL_PATH}"
        )

        st.stop()

    st.success(
        f"Loaded model: {type(model).__name__}"
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.subheader("🤖 Model Information")

    st.write(
        f"**Model:** {type(model).__name__}"
    )

    st.write(
        f"**Target:** {TARGET}"
    )

    st.write(
        f"**Number of features:** {len(FEATURES)}"
    )

    st.write(
        "**Features:**"
    )

    st.code(
        ", ".join(FEATURES)
    )

    # --------------------------------------------------------
    # MODEL PARAMETERS
    # --------------------------------------------------------

    if hasattr(model, "get_params"):

        with st.expander(
            "⚙️ View Model Parameters"
        ):

            st.json(
                model.get_params()
            )

    st.divider()

    # --------------------------------------------------------
    # SAVED METRICS
    # --------------------------------------------------------

    metrics_path = (
        "models/model_metrics.csv"
    )

    if os.path.exists(metrics_path):

        st.subheader(
            "📊 Regression Model Comparison"
        )

        metrics_df = pd.read_csv(
            metrics_path
        )

        st.dataframe(
            metrics_df,
            use_container_width=True
        )

    else:

        st.info(
            """
            model_metrics.csv was not found.

            Once we save the actual evaluation metrics
            from your Colab regression section, they can
            be displayed here.
            """
        )

    st.divider()

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_importances_"
    ):

        st.subheader(
            "🎯 Feature Importance"
        )

        importances = (
            model.feature_importances_
        )

        if len(importances) == len(FEATURES):

            importance_df = pd.DataFrame(
                {
                    "Feature": FEATURES,
                    "Importance": importances
                }
            )

            importance_df = (
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
            )

            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Random Forest Feature Importance"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# PREDICTION
# ============================================================

elif page == "🤖 Prediction":

    st.title(
        "🤖 Health Impact Prediction"
    )

    st.markdown(
        """
        Enter the air quality measurements below.

        The trained Random Forest regression model will
        predict the **HealthImpactScore**.
        """
    )

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if model is None:

        st.error(
            "Regression model not found."
        )

        st.stop()

    if scaler is None:

        st.error(
            "Scaler not found."
        )

        st.stop()

    st.success(
        "Model and scaler loaded successfully."
    )

    st.divider()

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    st.subheader(
        "🌫️ Air Quality Measurements"
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with col1:

        AQI = st.number_input(
            "AQI",
            min_value=0.0,
            max_value=500.0,
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

    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with col2:

        NO2 = st.number_input(
            "NO2",
            min_value=0.0,
            value=30.0,
            step=1.0
        )

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

    st.divider()

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predict_button = st.button(
        "🔮 Predict Health Impact Score",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        try:

            # ------------------------------------------------
            # CREATE INPUT DATAFRAME
            #
            # IMPORTANT:
            # Feature order MUST match training
            # ------------------------------------------------

            input_data = pd.DataFrame(
                [[
                    AQI,
                    PM10,
                    PM2_5,
                    NO2,
                    SO2,
                    O3
                ]],
                columns=FEATURES
            )

            # ------------------------------------------------
            # SCALE INPUT
            # ------------------------------------------------

            scaled_input = scaler.transform(
                input_data
            )

            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            prediction = model.predict(
                scaled_input
            )[0]

            # ------------------------------------------------
            # DISPLAY RESULT
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="prediction-box">

                    <div class="prediction-label">
                        Predicted Health Impact Score
                    </div>

                    <div class="prediction-value">
                        {prediction:.2f}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "Prediction generated successfully!"
            )

            # ------------------------------------------------
            # INPUT SUMMARY
            # ------------------------------------------------

            with st.expander(
                "🔍 View Input Values"
            ):

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
                Check that the saved scaler and model were
                created from the same six features:

                AQI, PM10, PM2_5, NO2, SO2, O3
                """
            )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Air Quality & Health Impact Analysis"
)

st.sidebar.caption(
    "Machine Learning Regression Project"
)
