import streamlit as st
import os
import psycopg2
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
import mysql.connector
import time

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Connect using the correct driver
def connect_to_db(db_type, host, port, db, user, password):
    if db_type == "PostgreSQL":
        return psycopg2.connect(
            host=host, port=port, database=db, user=user, password=password
        )
    elif db_type == "MySQL":
        return mysql.connector.connect(
            host=host, port=port, database=db, user=user, password=password
        )

# Fetch metadata
def fetch_db_metadata_dynamic(db_type, host, port, db, user, password):
    try:
        conn = connect_to_db(db_type, host, port, db, user, password)
        cur = conn.cursor()
        if db_type == "PostgreSQL":
            cur.execute(
                """
                SELECT table_schema, table_name, column_name
                FROM information_schema.columns
                WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                ORDER BY table_schema, table_name, ordinal_position;
            """
            )
        elif db_type == "MySQL":
            cur.execute(
                """
                SELECT table_schema, table_name, column_name
                FROM information_schema.columns
                WHERE table_schema NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
                ORDER BY table_schema, table_name, ordinal_position;
            """
            )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        return str(e)


def fetch_foreign_keys_dynamic(db_type, host, port, db, user, password):
    try:
        conn = connect_to_db(db_type, host, port, db, user, password)
        cur = conn.cursor()
        if db_type == "PostgreSQL":
            cur.execute(
                """
                SELECT
                    tc.table_schema AS source_schema,
                    tc.table_name AS source_table,
                    kcu.column_name AS source_column,
                    ccu.table_schema AS target_schema,
                    ccu.table_name AS target_table,
                    ccu.column_name AS target_column
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY';
            """
            )
        elif db_type == "MySQL":
            cur.execute(
                """
                SELECT
                    kcu.TABLE_SCHEMA AS source_schema,
                    kcu.TABLE_NAME AS source_table,
                    kcu.COLUMN_NAME AS source_column,
                    kcu.REFERENCED_TABLE_SCHEMA AS target_schema,
                    kcu.REFERENCED_TABLE_NAME AS target_table,
                    kcu.REFERENCED_COLUMN_NAME AS target_column
                FROM information_schema.KEY_COLUMN_USAGE AS kcu
                WHERE kcu.REFERENCED_TABLE_NAME IS NOT NULL;
            """
            )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception:
        return []


def read_sql_query_dynamic(db_type, sql, host, port, db, user, password):
    try:
        conn = connect_to_db(db_type, host, port, db, user, password)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()
        return colnames, rows
    except Exception as e:
        return ["Error"], [(str(e),)]


def build_dynamic_prompt(metadata_rows, foreign_keys):
    prompt = """
You are an expert at translating plain English into SQL queries.

The PostgreSQL database has the following tables and columns (case-sensitive):
"""
    tables = {}
    for schema, table, column in metadata_rows:
        key = f"{schema}.{table}"
        if key not in tables:
            tables[key] = []
        tables[key].append(column)

    for table_fullname, columns in tables.items():
        prompt += (
            f"- **{table_fullname}**\n  - "
            + ", ".join(f'"{col}"' for col in columns)
            + "\n"
        )

    prompt += "\nThese are the foreign key relationships between the tables:\n"
    if foreign_keys:
        for (
            src_schema,
            src_table,
            src_col,
            tgt_schema,
            tgt_table,
            tgt_col,
        ) in foreign_keys:
            prompt += f'- `{src_schema}.{src_table}`."{src_col}" → `{tgt_schema}.{tgt_table}`."{tgt_col}"\n'
    else:
        prompt += "(No foreign keys detected)\n"

    prompt += """
    
Your job:
- Translate English into **valid PostgreSQL SELECT queries**.
- Always use fully qualified table names (e.g., `public.my_table`) and wrap column names in **double quotes**.
- Use JOINs intelligently using the foreign key relationships above.
- Output only the final SQL query with no extra text or markdown.
"""
    return prompt


# Reasoning generation
def get_gemini_reasoning(question, sql_query):
    explanation_prompt = f"""
    You just generated the following SQL query for the user's question:

    Question:
    {question}

    SQL Query:
    {sql_query}

    Now, explain step-by-step how this SQL query works:
    - Mention which tables and columns are used.
    - Explain how tables are joined (if any).
    - Describe any WHERE conditions or filters.
    - Format clearly using markdown.
    - Use bullet points or emojis to visually break up logic.

    Keep the explanation clear and beginner-friendly.
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(explanation_prompt)
    return response.text.strip()


# Run query
def read_sql_query(sql, host, port, db, user, password):
    try:
        conn = psycopg2.connect(
            host=host, port=port, database=db, user=user, password=password
        )
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()
        return colnames, rows
    except Exception as e:
        return ["Error"], [(str(e),)]


# Generate Gemini response
def get_gemini_response(question, prompt):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([prompt, question])
    sql_text = response.text.strip().replace("```sql", "").replace("```", "").strip()
    return sql_text


def fetch_sample_values(host, port, db, user, password, metadata_rows):
    sample_data = {}
    try:
        conn = psycopg2.connect(
            host=host, port=port, database=db, user=user, password=password
        )
        cur = conn.cursor()
        for schema, table, column in metadata_rows:
            key = f"{schema}.{table}"
            try:
                cur.execute(
                    f"""
                    SELECT DISTINCT "{column}"
                    FROM "{schema}"."{table}"
                    WHERE "{column}" IS NOT NULL
                    LIMIT 3;
                """
                )
                values = cur.fetchall()
                sample_data[(key, column)] = [
                    str(v[0]) for v in values if v[0] is not None
                ]
            except Exception:
                continue
        cur.close()
        conn.close()
    except Exception as e:
        print("Sample fetch error:", e)
    return sample_data


def deduplicate_columns(columns):
    seen = {}
    new_columns = []
    for col in columns:
        if col not in seen:
            seen[col] = 1
            new_columns.append(col)
        else:
            new_columns.append(f"{col}_{seen[col]}")
            seen[col] += 1
    return new_columns


st.set_page_config(page_title="Gemini SQL Assistant (Dynamic DB)")
st.markdown(
    """
<style>
@keyframes blink {
  0% { opacity: 0.2; }
  50% { opacity: 1; }
  100% { opacity: 0.2; }
}

.blinking {
  animation: blink 1.5s infinite;
  font-weight: bold;
  font-size: 1.1rem;
  color: #4b6584;
  border: 2px dashed #a5b1c2;
  padding: 10px;
  border-radius: 10px;
  background-color: #f1f2f6;
  text-align: center;
  margin-top: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.header("🧠 Text2SQL Assistant (Any Datasource)")

# db_type = st.selectbox("Select Database Type", ["PostgreSQL", "MySQL"])

st.sidebar.title("🔌 Database Connections")

if "saved_connections" not in st.session_state:
    st.session_state.saved_connections = {}

# Create a form to input new DB connection
with st.sidebar.form("connection_form", clear_on_submit=False):
    db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL"], key="db_type")
    conn_name = st.text_input("Connection Name (e.g. prod-db)", key="conn_name")
    host = st.text_input("Host", key="host", value=os.getenv("DB_HOST", "localhost"))
    port = st.text_input(
        "Port",
        key="port",
        value="5432" if st.session_state.db_type == "PostgreSQL" else "3306",
    )
    dbname = st.text_input(
        "Database Name", key="dbname", value=os.getenv("DB_NAME", "your_db")
    )
    user = st.text_input(
        "Username", key="user", value=os.getenv("DB_USER", "your_user")
    )
    password = st.text_input(
        "Password", type="password", key="password", value=os.getenv("DB_PASSWORD", "")
    )

    if st.form_submit_button("💾 Save Connection"):
        if conn_name:
            st.session_state.saved_connections[conn_name] = {
                "db_type": db_type,
                "host": host,
                "port": port,
                "dbname": dbname,
                "user": user,
                "password": password,
            }
            st.success(f"✅ Saved connection: {conn_name}")
        else:
            st.error("❌ Please enter a connection name.")

# Select from saved connections
selected_conn_name = st.sidebar.selectbox(
    "🗂️ Choose a Connection", list(st.session_state.saved_connections.keys())
)

if selected_conn_name:
    conn_info = st.session_state.saved_connections[selected_conn_name]
    db_type = conn_info["db_type"]
    host = conn_info["host"]
    port = conn_info["port"]
    dbname = conn_info["dbname"]
    user = conn_info["user"]
    password = conn_info["password"]


# Schema loading
if st.button("🔄 Connect and Load Schema"):
    if not all([host, port, dbname, user, password]):
        st.error("❌ Please fill in all the credentials.")
    else:
        metadata = fetch_db_metadata_dynamic(
            db_type, host, port, dbname, user, password
        )
        fks = fetch_foreign_keys_dynamic(db_type, host, port, dbname, user, password)

        if isinstance(metadata, str):
            st.error(f"❌ Error: {metadata}")
        else:
            st.session_state["prompt"] = build_dynamic_prompt(metadata, fks)
            st.success("✅ Schema & relationships loaded successfully!")

# SQL generation and execution
question = st.text_input("📝 Ask your SQL question:")

def type_out_text(text, placeholder, delay=0.01, code_block=False):
    with placeholder:
        if code_block:
            st.code("", language="sql")
        else:
            st.markdown("")

        display_text = ""
        for char in text:
            display_text += char
            if code_block:
                placeholder.code(display_text, language="sql")
            else:
                placeholder.markdown(display_text)
            time.sleep(delay)


if st.button("🚀 Generate & Execute SQL"):
    if "prompt" not in st.session_state:
        st.warning("⚠️ Load the schema first!")
    elif not question.strip():
        st.error("❌ Please enter a valid question.")
    else:
        with st.spinner("Generating Gemini reasoning and SQL..."):
            prompt = st.session_state["prompt"]

            thinking_placeholder = st.empty()
            sql_query = get_gemini_response(question, prompt)
            reasoning = get_gemini_reasoning(question, sql_query)

            with st.expander("🧠 Gemini's Reasoning (click to expand)", expanded=False):
                reasoning_placeholder = st.empty()
                type_out_text(
                    reasoning, reasoning_placeholder, delay=0.001, code_block=False
                )

            st.divider()

            sql_placeholder = st.empty()
            st.markdown("🗒️ **Generated SQL Query:**")
            st.code(sql_query, language="sql")
            # type_out_text(sql_query, sql_placeholder, delay=0.03)

            st.divider()

            headers, results = read_sql_query_dynamic(
                db_type, sql_query, host, port, dbname, user, password
            )

            st.subheader("📊 Query Results:")
            if headers == ["Error"]:
                st.error(f"❌ Query Failed: {results[0][0]}")
            else:
                df = pd.DataFrame(results, columns=headers)
                df.columns = deduplicate_columns(df.columns)
                st.dataframe(df)
