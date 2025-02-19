import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import tempfile
import os

# LlamaIndex imports (using free/default models)
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext

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
    <h1>Data Discovery & Visualization</h1>
</div>
""", unsafe_allow_html=True)
st.markdown("#### Powered by LlamaIndex & Advanced NLP Parsing")
st.markdown("[LlamaIndex GitHub](https://github.com/jerryjliu/llama_index)")

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
        # Build LlamaIndex from DataFrame
        # -------------------------------
        csv_text = df.to_csv(index=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(csv_text.encode('utf-8'))
            tmp_path = tmp.name

        temp_dir = os.path.join(os.path.dirname(tmp_path), "temp_data")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, "data.csv")
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(csv_text)

        try:
            documents = SimpleDirectoryReader(temp_dir).load_data()
            service_context = ServiceContext.from_defaults()
            index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
        except Exception as e:
            st.error(f"Error building LlamaIndex: {e}")
            index = None

        # -------------------------------
        # NLP Parsing Functions
        # -------------------------------
        def extract_candidate_columns(instruction, df):
            instruction_lower = instruction.lower()
            candidates = []
            for col in df.columns:
                if col.lower() in instruction_lower:
                    candidates.append(col)
            return candidates

        def parse_line_chart(instruction, df):
            candidates = extract_candidate_columns(instruction, df)
            time_cols = [col for col in candidates if "date" in col.lower() or "time" in col.lower()]
            numeric_cols = [col for col in candidates if pd.api.types.is_numeric_dtype(df[col])]
            if time_cols:
                x = time_cols[0]
                y = numeric_cols[0] if numeric_cols else next((col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])), None)
            else:
                potential_datetime = None
                for col in df.columns:
                    try:
                        pd.to_datetime(df[col])
                        potential_datetime = col
                        break
                    except:
                        continue
                x = potential_datetime if potential_datetime else (candidates[0] if candidates else df.columns[0])
                y = numeric_cols[0] if numeric_cols else (df.columns[1] if len(df.columns)>1 else df.columns[0])
            return {"type": "line", "x": x, "y": y}

        def parse_bar_chart(instruction, df):
            candidates = extract_candidate_columns(instruction, df)
            categorical = [col for col in candidates if df[col].dtype == 'object']
            numeric = [col for col in candidates if pd.api.types.is_numeric_dtype(df[col])]
            if categorical:
                x = categorical[0]
                y = numeric[0] if numeric else next((col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])), None)
            else:
                x = df.columns[0]
                y = numeric[0] if numeric else (df.columns[1] if len(df.columns)>1 else df.columns[0])
            return {"type": "bar", "x": x, "y": y}

        def parse_histogram(instruction, df):
            candidates = extract_candidate_columns(instruction, df)
            numeric = [col for col in candidates if pd.api.types.is_numeric_dtype(df[col])]
            column = numeric[0] if numeric else next((col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])), df.columns[0])
            return {"type": "histogram", "column": column}

        def parse_scatter(instruction, df):
            candidates = extract_candidate_columns(instruction, df)
            numeric = [col for col in candidates if pd.api.types.is_numeric_dtype(df[col])]
            if len(numeric) >= 2:
                x, y = numeric[0], numeric[1]
            else:
                all_numeric = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                if len(all_numeric) >= 2:
                    x, y = all_numeric[0], all_numeric[1]
                else:
                    x = y = df.columns[0]
            return {"type": "scatter", "x": x, "y": y}

        def parse_pie_chart(instruction, df):
            candidates = extract_candidate_columns(instruction, df)
            categorical = [col for col in candidates if df[col].dtype == 'object']
            if categorical:
                column = categorical[0]
            else:
                for col in df.columns:
                    if df[col].nunique() < 10:
                        column = col
                        break
                else:
                    column = df.columns[0]
            return {"type": "pie", "column": column}

        def parse_summary(instruction, df):
            return {"type": "summary"}

        def parse_heatmap(instruction, df):
            return {"type": "heatmap"}

        def parse_box_plot(instruction, df):
            candidates = extract_candidate_columns(instruction, df)
            numeric = [col for col in candidates if pd.api.types.is_numeric_dtype(df[col])]
            categorical = [col for col in candidates if df[col].dtype == 'object']
            if numeric and categorical:
                x, y = categorical[0], numeric[0]
            else:
                x = next((col for col in df.columns if df[col].dtype == 'object'), df.columns[0])
                y = next((col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])), df.columns[0])
            return {"type": "box", "x": x, "y": y}

        def parse_instruction(instruction, df):
            instruction_lower = instruction.lower()
            if "line chart" in instruction_lower:
                return parse_line_chart(instruction, df)
            elif "bar chart" in instruction_lower:
                return parse_bar_chart(instruction, df)
            elif "histogram" in instruction_lower:
                return parse_histogram(instruction, df)
            elif "scatter" in instruction_lower:
                return parse_scatter(instruction, df)
            elif "pie chart" in instruction_lower:
                return parse_pie_chart(instruction, df)
            elif "summary" in instruction_lower or "describe" in instruction_lower:
                return parse_summary(instruction, df)
            elif "correlation heatmap" in instruction_lower or ("heatmap" in instruction_lower and "correlation" in instruction_lower):
                return parse_heatmap(instruction, df)
            elif "box plot" in instruction_lower:
                return parse_box_plot(instruction, df)
            else:
                return {"type": "unknown"}

        # -------------------------------
        # Visualization Generation
        # -------------------------------
        def generate_visualization(parsed, df):
            chart_type = parsed.get("type")
            if chart_type == "line":
                x = parsed.get("x")
                y = parsed.get("y")
                try:
                    df[x] = pd.to_datetime(df[x])
                except Exception as e:
                    st.warning(f"Could not convert {x} to datetime: {e}")
                df_sorted = df.sort_values(by=x)
                # Matplotlib line chart
                fig1, ax = plt.subplots(figsize=(8, 4))
                ax.plot(df_sorted[x], df_sorted[y], marker='o', color='#4CAF50')
                ax.set_title(f"Line Chart of {y} over {x}")
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                st.pyplot(fig1)
                # Plotly line chart
                fig2 = px.line(df_sorted, x=x, y=y, title=f"Line Chart of {y} over {x}")
                st.plotly_chart(fig2)
            elif chart_type == "bar":
                x = parsed.get("x")
                y = parsed.get("y")
                # Matplotlib bar chart
                fig1, ax = plt.subplots(figsize=(8, 4))
                ax.bar(df[x], df[y], color='#FF5733')
                ax.set_title(f"Bar Chart of {y} by {x}")
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                st.pyplot(fig1)
                # Plotly bar chart
                fig2 = px.bar(df, x=x, y=y, title=f"Bar Chart of {y} by {x}")
                st.plotly_chart(fig2)
            elif chart_type == "histogram":
                column = parsed.get("column")
                # Matplotlib histogram
                fig1, ax = plt.subplots(figsize=(8, 4))
                ax.hist(df[column], bins=20, color='#33FFCE')
                ax.set_title(f"Histogram of {column}")
                ax.set_xlabel(column)
                ax.set_ylabel("Frequency")
                st.pyplot(fig1)
                # Plotly histogram
                fig2 = px.histogram(df, x=column, nbins=20, title=f"Histogram of {column}")
                st.plotly_chart(fig2)
            elif chart_type == "scatter":
                x = parsed.get("x")
                y = parsed.get("y")
                # Matplotlib scatter plot
                fig1, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(df[x], df[y], color='#9B59B6')
                ax.set_title(f"Scatter Plot of {y} vs {x}")
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                st.pyplot(fig1)
                # Plotly scatter plot
                fig2 = px.scatter(df, x=x, y=y, title=f"Scatter Plot of {y} vs {x}")
                st.plotly_chart(fig2)
            elif chart_type == "pie":
                column = parsed.get("column")
                counts = df[column].value_counts()
                # Matplotlib pie chart
                fig1, ax = plt.subplots(figsize=(8, 4))
                ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
                ax.set_title(f"Pie Chart of {column}")
                st.pyplot(fig1)
                # Plotly pie chart
                fig2 = px.pie(values=counts.values, names=counts.index, title=f"Pie Chart of {column}")
                st.plotly_chart(fig2)
            elif chart_type == "summary":
                st.subheader("Summary Statistics")
                st.dataframe(df.describe())
            elif chart_type == "heatmap":
                numeric_df = df.select_dtypes(include=['number'])
                corr = numeric_df.corr()
                # Matplotlib heatmap
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)
                # Plotly heatmap
                fig2 = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                st.plotly_chart(fig2)
            elif chart_type == "box":
                x = parsed.get("x")
                y = parsed.get("y")
                # Matplotlib box plot using seaborn
                fig1, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=df[x], y=df[y], palette="Set3", ax=ax)
                ax.set_title(f"Box Plot of {y} by {x}")
                st.pyplot(fig1)
                # Plotly box plot
                fig2 = px.box(df, x=x, y=y, title=f"Box Plot of {y} by {x}")
                st.plotly_chart(fig2)
            else:
                st.error("Instruction not recognized or unsupported visualization type.")

        # -------------------------------
        # Instruction Input & Processing
        # -------------------------------
        st.markdown("### Enter your instruction")
        st.info("Examples: 'Plot a line chart of revenue over time', 'Show summary statistics', 'Plot a correlation heatmap', 'Plot a bar chart of sales by region', 'Plot a histogram of age', 'Plot a scatter plot of height vs weight', 'Plot a pie chart of department', 'Plot a box plot of salary by department'")
        instruction = st.text_input("Enter your instruction")
        if st.button("Generate Visualization") and instruction:
            # Use LlamaIndex to get additional NLP insight (displayed for context)
            if index is not None:
                with st.spinner("Querying LlamaIndex..."):
                    llama_response = index.query(instruction)
                st.markdown("### LlamaIndex NLP Response:")
                st.write(llama_response)
            # Parse instruction using our rule-based NLP parser
            parsed = parse_instruction(instruction, df)
            st.markdown("### Parsed Instruction Parameters:")
            st.write(parsed)
            # Generate visualization based on parsed instruction
            generate_visualization(parsed, df)
