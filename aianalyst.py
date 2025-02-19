import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import json

from transformers import pipeline
from sklearn.linear_model import LinearRegression

# -------------------------------
# Custom CSS & SVG iconography
# -------------------------------
custom_css = """
<style>
/* General page style */
body {
    background-color: #f0f2f6;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Login container styling */
.login-container {
    max-width: 400px;
    margin: 100px auto;
    padding: 40px;
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}

/* Button style */
button[kind="primary"] {
    background-color: #4CAF50;
    border: none;
    color: white;
    padding: 10px 16px;
    border-radius: 4px;
    cursor: pointer;
}

button[kind="primary"]:hover {
    background-color: #45a049;
}

/* Header with SVG icon */
.header {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-bottom: 20px;
}

.header svg {
    width: 50px;
    height: 50px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -------------------------------
# Login Functionality
# -------------------------------
def login():
    st.markdown("<h2 style='text-align: center;'>Login</h2>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "username" and password == "password":
            st.session_state['logged_in'] = True
            st.experimental_rerun()
        else:
            st.error("Incorrect username or password.")

if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        login()
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# -------------------------------
# App Header
# -------------------------------
st.markdown("""
<div class="header">
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" stroke="currentColor" stroke-width="2" 
         stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
      <line x1="18" y1="20" x2="18" y2="10"></line>
      <line x1="12" y1="20" x2="12" y2="4"></line>
      <line x1="6" y1="20" x2="6" y2="14"></line>
    </svg>
    <h1>Data Discovery, Analysis & Visualization</h1>
</div>
""", unsafe_allow_html=True)
st.markdown("#### Powered by Hugging Face Transformers for NLP-based command parsing")

# -------------------------------
# File Upload & Data Processing
# -------------------------------
uploaded_file = st.file_uploader("Upload a CSV, Excel, or JSON file", type=["csv", "xlsx", "json"])
df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file type.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
    
    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # -------------------------------
        # Load Hugging Face Model for NLP
        # -------------------------------
        @st.cache_resource
        def load_nlp_model():
            # Using a lightweight text-to-text generation model.
            return pipeline("text2text-generation", model="google/flan-t5-small")
        
        nlp_pipeline = load_nlp_model()

        # -------------------------------
        # NLP Parsing via Hugging Face Transformers
        # -------------------------------
        def parse_instruction(instruction):
            # Refined prompt to encourage valid JSON output.
            prompt = (
                "You are an assistant that converts natural language instructions for data visualization and analysis into a valid JSON object. "
                "The JSON object must contain exactly one key 'action' with one of the following values: 'visualization' or 'analysis'.\n\n"
                "If 'action' is 'visualization', then the JSON object must also include a key 'chart_type' with one of these values: 'line', 'bar', 'histogram', 'scatter', 'pie', or 'box'. "
                "For charts that require axes, include keys 'x' and 'y' with column names from the dataset. For 'histogram' and 'pie', include a key 'column'.\n\n"
                "If 'action' is 'analysis', then include a key 'analysis_type' with one of these values: 'summary', 'correlation', 'regression', or 'missing_values'. "
                "For regression, include keys 'x' and 'y' for the independent and dependent variables.\n\n"
                "Return only a valid JSON string that can be parsed by json.loads, and do not include any extra text.\n\n"
                f"Instruction: {instruction}"
            )
            try:
                result = nlp_pipeline(prompt, max_length=250, do_sample=False)
                generated_text = result[0]['generated_text'].strip()
                # Debug: Show raw generated text if needed.
                st.write("Raw model output:", generated_text)
                parsed = json.loads(generated_text)
                # Verify that the parsed JSON has a valid action.
                if parsed.get("action") not in ["visualization", "analysis"]:
                    st.error("Parsed JSON does not have a valid 'action' key.")
                    parsed = {"action": "unknown"}
            except Exception as e:
                st.error("Error parsing model output to JSON. Please refine your instruction.")
                st.error(f"Parsing Exception: {e}")
                parsed = {"action": "unknown"}
            return parsed

        # -------------------------------
        # Visualization Generation Functions
        # -------------------------------
        def generate_visualization(command, df):
            chart_type = command.get("chart_type")
            if chart_type == "line":
                x = command.get("x")
                y = command.get("y")
                if not x or not y or x not in df.columns or y not in df.columns:
                    st.error("Required columns for a line chart not found.")
                    return
                try:
                    df[x] = pd.to_datetime(df[x])
                except Exception as e:
                    st.warning(f"Could not convert '{x}' to datetime: {e}")
                df_sorted = df.sort_values(by=x)
                fig1, ax = plt.subplots(figsize=(8, 4))
                ax.plot(df_sorted[x], df_sorted[y], marker='o', color='#4CAF50')
                ax.set_title(f"Line Chart of {y} over {x}")
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                st.pyplot(fig1)
                fig2 = px.line(df_sorted, x=x, y=y, title=f"Line Chart of {y} over {x}")
                st.plotly_chart(fig2)

            elif chart_type == "bar":
                x = command.get("x")
                y = command.get("y")
                if not x or not y or x not in df.columns or y not in df.columns:
                    st.error("Required columns for a bar chart not found.")
                    return
                fig1, ax = plt.subplots(figsize=(8, 4))
                ax.bar(df[x], df[y], color='#FF5733')
                ax.set_title(f"Bar Chart of {y} by {x}")
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                st.pyplot(fig1)
                fig2 = px.bar(df, x=x, y=y, title=f"Bar Chart of {y} by {x}")
                st.plotly_chart(fig2)

            elif chart_type == "histogram":
                column = command.get("column")
                if not column or column not in df.columns:
                    st.error("Required column for a histogram not found.")
                    return
                fig1, ax = plt.subplots(figsize=(8, 4))
                ax.hist(df[column], bins=20, color='#33FFCE')
                ax.set_title(f"Histogram of {column}")
                ax.set_xlabel(column)
                ax.set_ylabel("Frequency")
                st.pyplot(fig1)
                fig2 = px.histogram(df, x=column, nbins=20, title=f"Histogram of {column}")
                st.plotly_chart(fig2)

            elif chart_type == "scatter":
                x = command.get("x")
                y = command.get("y")
                if not x or not y or x not in df.columns or y not in df.columns:
                    st.error("Required columns for a scatter plot not found.")
                    return
                fig1, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(df[x], df[y], color='#9B59B6')
                ax.set_title(f"Scatter Plot of {y} vs {x}")
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                st.pyplot(fig1)
                fig2 = px.scatter(df, x=x, y=y, title=f"Scatter Plot of {y} vs {x}")
                st.plotly_chart(fig2)

            elif chart_type == "pie":
                column = command.get("column")
                if not column or column not in df.columns:
                    st.error("Required column for a pie chart not found.")
                    return
                counts = df[column].value_counts()
                fig1, ax = plt.subplots(figsize=(8, 4))
                ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
                ax.set_title(f"Pie Chart of {column}")
                st.pyplot(fig1)
                fig2 = px.pie(values=counts.values, names=counts.index, title=f"Pie Chart of {column}")
                st.plotly_chart(fig2)

            elif chart_type == "box":
                x = command.get("x")
                y = command.get("y")
                if not x or not y or x not in df.columns or y not in df.columns:
                    st.error("Required columns for a box plot not found.")
                    return
                fig1, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=df[x], y=df[y], palette="Set3", ax=ax)
                ax.set_title(f"Box Plot of {y} by {x}")
                st.pyplot(fig1)
                fig2 = px.box(df, x=x, y=y, title=f"Box Plot of {y} by {x}")
                st.plotly_chart(fig2)
            else:
                st.error("Unsupported visualization type.")

        # -------------------------------
        # Data Analysis Generation Functions
        # -------------------------------
        def generate_analysis(command, df):
            analysis_type = command.get("analysis_type")
            if analysis_type == "summary":
                st.subheader("Summary Statistics")
                st.dataframe(df.describe())
            elif analysis_type == "correlation":
                numeric_df = df.select_dtypes(include=['number'])
                if numeric_df.empty:
                    st.error("No numeric columns available for correlation analysis.")
                    return
                corr = numeric_df.corr()
                st.subheader("Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)
                fig2 = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                st.plotly_chart(fig2)
            elif analysis_type == "regression":
                x = command.get("x")
                y = command.get("y")
                if not x or not y or x not in df.columns or y not in df.columns:
                    st.error("Required columns for regression analysis not found.")
                    return
                try:
                    X = df[[x]].dropna()
                    Y = df[y].loc[X.index]
                    model = LinearRegression()
                    model.fit(X, Y)
                    predictions = model.predict(X)
                    st.subheader("Regression Analysis")
                    st.write(f"Coefficient (slope): {model.coef_[0]:.4f}")
                    st.write(f"Intercept: {model.intercept_:.4f}")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.scatter(X, Y, color='#9B59B6', label="Data")
                    ax.plot(X, predictions, color='red', label="Regression Line")
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    ax.set_title(f"Regression Analysis: {y} vs {x}")
                    ax.legend()
                    st.pyplot(fig)
                    fig2 = px.scatter(df, x=x, y=y, trendline="ols", title=f"Regression Analysis: {y} vs {x}")
                    st.plotly_chart(fig2)
                except Exception as e:
                    st.error(f"Error performing regression: {e}")
            elif analysis_type == "missing_values":
                st.subheader("Missing Value Analysis")
                missing = df.isnull().sum()
                st.dataframe(missing)
            else:
                st.error("Unsupported analysis type.")

        # -------------------------------
        # Handle the Parsed Command
        # -------------------------------
        st.markdown("### Enter Your Instruction")
        st.info(
            "Examples:\n"
            "- Visualization: Plot a line chart of revenue over time\n"
            "- Visualization: Plot a bar chart of sales by region\n"
            "- Analysis: Show summary statistics\n"
            "- Analysis: Display correlation heatmap\n"
            "- Analysis: Perform regression analysis of revenue on time\n"
            "- Analysis: Report missing values"
        )
        instruction = st.text_input("Enter your instruction for data visualization or analysis")
        if st.button("Process Instruction") and instruction:
            with st.spinner("Processing instruction with Hugging Face model..."):
                command = parse_instruction(instruction)
            st.markdown("### Parsed Command from NLP")
            st.write(command)
            action = command.get("action")
            if action == "visualization":
                generate_visualization(command, df)
            elif action == "analysis":
                generate_analysis(command, df)
            else:
                st.error("The parsed command did not specify a valid action.")
