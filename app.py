import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st


from langchain_core.pydantic_v1 import BaseModel, Field, validator
from cleaning_helper import plan_cleaning, apply_cleaning_plan, heuristic_clean


# Load environment variables
load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

# Configure Streamlit page
st.set_page_config(page_title="Data Science Dashboard", page_icon="📊", layout="wide")
st.title("📊 Data Science Dashboard")

# Sidebar with 3 sections
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Cleaning", "Analysis", "Model"])

# Model
if OPEN_AI_API_KEY:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPEN_AI_API_KEY)
else:
    st.sidebar.error("⚠️ OPEN_AI_API_KEY not found. Please set it in .env file.")




# ---------------- Cleaning Page ----------------
if menu == "Cleaning":
    st.header("🧹 Data Cleaning")

    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV to begin.")
    else:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Data")
        st.dataframe(df.head())

        col_missing = (df.isna().mean() * 100).round(2)
        with st.expander("Data profile (quick view)"):
            st.write(pd.DataFrame({"dtype": df.dtypes.astype(str), "%missing": col_missing}).T)

        if st.button("Run AI Cleaning"):
            with st.spinner("Planning cleaning steps with AI..."):
                try:
                    plan = plan_cleaning(df, llm)
                except Exception as e:
                    st.warning(f"AI plan failed, switching to safe heuristic cleaning. Details: {e}")
                    plan = None

            if plan:
                with st.expander("Cleaning plan (AI)"):
                    st.code(json.dumps(plan, indent=2), language="json")
                with st.spinner("Applying cleaning plan..."):
                    cleaned, report = apply_cleaning_plan(df, plan)
            else:
                cleaned, report = heuristic_clean(df)

            st.success(f"Cleaned dataset ready. Shape: {report.get('shape_after')} (was {report.get('shape_before')})")
            with st.expander("Cleaning report"):
                st.write(report.get("steps", []))

            # Download
            csv_bytes = cleaned.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Cleaned CSV",
                data=csv_bytes,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )

            # Show sample of cleaned data
            st.write("### Sample of Cleaned Data")
            st.dataframe(cleaned.head())




# ---------------- Analysis Page ----------------
elif menu == "Analysis":
    st.header("📊 AI-Powered Data Analysis & Visualization")

    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Data:")
        st.dataframe(df.head())
        dtypes_map = {c: str(t) for c, t in df.dtypes.items()}

        # Helpers
        def apply_filters(_df, filters):
            if not filters:
                return _df
            out = _df.copy()
            for f in filters:
                col = f.get("column")
                op = f.get("op")
                val = f.get("value")
                if col not in out.columns:
                    continue
                series = out[col]
                try:
                    if op == "==":
                        out = out[series == val]
                    elif op == "!=":
                        out = out[series != val]
                    elif op == ">":
                        out = out[series > val]
                    elif op == ">=":
                        out = out[series >= val]
                    elif op == "<":
                        out = out[series < val]
                    elif op == "<=":
                        out = out[series <= val]
                    elif op == "in":
                        vals = val if isinstance(val, list) else [val]
                        out = out[series.isin(vals)]
                    elif op == "not in":
                        vals = val if isinstance(val, list) else [val]
                        out = out[~series.isin(vals)]
                    elif op == "contains":
                        out = out[series.astype(str).str.contains(str(val), case=False, na=False)]
                except Exception:
                    # Skip invalid filter gracefully
                    continue
            return out

        def agg_series(s, agg):
            agg = (agg or "").lower()
            if agg in ["sum", "mean", "median", "min", "max", "count", "std", "nunique"]:
                return getattr(s, agg)()
            # default to count
            return s.count()

        def prepare_data_for_chart(_df, spec):
            # Supports optional groupby + agg or direct x/y plotting
            x = spec.get("x")
            y = spec.get("y")
            groupby = spec.get("groupby")
            agg = spec.get("agg")
            filters = spec.get("filters")
            dd = apply_filters(_df, filters)

            if groupby and y and agg:
                grouped = dd.groupby(groupby)[y].agg(agg)
                return grouped.reset_index(), groupby, y
            if x and y and agg and x in dd.columns and y in dd.columns:
                grouped = dd.groupby(x)[y].agg(agg)
                return grouped.reset_index(), x, y
            return dd, x, y

        def render_chart(_df, spec):
            ctype = (spec.get("type") or "").lower()
            title = spec.get("title") or ""
            bins = spec.get("bins")
            color = spec.get("color") or spec.get("hue")
            orientation = spec.get("orientation") or "v"

            data, x, y = prepare_data_for_chart(_df, spec)

            fig, ax = plt.subplots(figsize=(5, 3), dpi=80)

            try:
                if ctype in ["bar", "area", "line", "scatter"]:
                    kind = "bar" if ctype == "bar" else ("area" if ctype == "area" else ("line" if ctype == "line" else "scatter"))
                    if kind == "scatter":
                        if x and y:
                            data.plot(x=x, y=y, kind="scatter", c=None, ax=ax)
                        else:
                            raise ValueError("Scatter requires x and y.")
                    else:
                        if x and y:
                            data.plot(x=x, y=y, kind=kind, ax=ax)
                        elif x and not y:
                            data[x].value_counts().plot(kind=kind, ax=ax)
                        elif y and not x:
                            data[y].plot(kind=kind, ax=ax)
                elif ctype in ["hist", "histogram"]:
                    target = y or x
                    if target and target in data.columns:
                        data[target].plot(kind="hist", bins=bins or 20, ax=ax)
                    else:
                        num_cols = data.select_dtypes(include="number").columns.tolist()
                        if num_cols:
                            data[num_cols[0]].plot(kind="hist", bins=bins or 20, ax=ax)
                        else:
                            raise ValueError("No numeric column for histogram.")
                elif ctype == "box":
                    target = y or x
                    if target and target in data.columns:
                        data[[target]].plot(kind="box", ax=ax)
                    else:
                        num_cols = data.select_dtypes(include="number").columns.tolist()
                        if num_cols:
                            data[num_cols].plot(kind="box", ax=ax)
                        else:
                            raise ValueError("No numeric column for boxplot.")
                elif ctype == "pie":
                    target = y or x
                    if target and target in data.columns:
                        data[target].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
                    else:
                        raise ValueError("Pie chart needs a categorical column.")
                elif ctype == "heatmap":
                    # Simple correlation heatmap with matplotlib
                    corr = data.select_dtypes(include="number").corr()
                    im = ax.imshow(corr, cmap="viridis")
                    ax.set_xticks(range(len(corr.columns)))
                    ax.set_yticks(range(len(corr.columns)))
                    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                    ax.set_yticklabels(corr.columns)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    raise ValueError(f"Unsupported chart type: {ctype}")

                if title:
                    ax.set_title(title)
                # 👇 this prevents auto-expanding
                st.pyplot(fig, use_container_width=False)
                return True
            except Exception as e:
                st.warning(f"Could not render {ctype} chart: {e}")
                return False

        # User query for analysis
        user_query = st.text_area("Ask anything about your data (e.g. 'bar chart of sales by region', 'compare revenue and profit', 'filter region=EMEA then show average sales'):")        

        if st.button("Analyze & Visualize") and user_query:
            prompt = ChatPromptTemplate.from_template(
                """
                You are a data analysis planner. The dataframe columns and dtypes are: {schema}.
                The user's request: {query}.

                Return STRICT JSON only, no prose outside JSON. Plan using this schema:
                {{
                  "intent": "plot|stats|compare|filter|qa",
                  "analysis": "short explanation for the user",
                  "filters": [{{"column": "...", "op": "==|!=|>|>=|<|<=|in|not in|contains", "value": any}}],  // optional
                  "columns": ["colA","colB"], // optional for stats/compare
                  "aggs": ["sum","mean","count"], // optional for stats
                  "chart": {{
                    "type": "bar|line|scatter|hist|box|pie|heatmap|area",
                    "x": "col or null",
                    "y": "col or null",
                    "groupby": ["colA"] ,         // optional
                    "agg": "sum|mean|count|... ", // optional
                    "bins": 20,                   // optional for hist
                    "title": "optional title"
                  }},
                  "charts": [ ... same as chart ... ] // optional multiple charts
                }}

                Rules:
                - Use only existing column names exactly as in schema.
                - Choose appropriate chart types for the data types.
                - For 'compare', include a chart (scatter/line/bar) if sensible.
                - For 'stats', specify 'columns' and 'aggs' to compute.
                - For 'filter', include 'filters' and, if asked to plot, also include a 'chart'.
                - Always include 'analysis' summarizing the result.
                """
            )

            chain = prompt | llm
            response = chain.invoke({"schema": dtypes_map, "query": user_query})
            raw_response = (getattr(response, "content", None) or str(response)).strip()

            # Parse JSON
            try:
                import re
                match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                if not match:
                    raise ValueError("No JSON object found in model output.")
                plan = json.loads(match.group())
            except Exception as e:
                st.error(f"Could not parse AI plan: {e}")
                st.info(raw_response)
                plan = None

            if plan:
                # 1) Filters
                work_df = apply_filters(df, plan.get("filters"))

                # 2) Stats/Compare/QA computation
                intent = (plan.get("intent") or "").lower()
                if intent in ["stats", "qa", "compare"]:
                    cols = plan.get("columns") or []
                    aggs = plan.get("aggs") or []
                    # Basic stats/aggregations
                    try:
                        if intent == "compare" and len(cols) >= 2:
                            c1, c2 = cols[:2]
                            shown = False
                            # Correlation if numeric
                            if c1 in work_df.columns and c2 in work_df.columns:
                                if pd.api.types.is_numeric_dtype(work_df[c1]) and pd.api.types.is_numeric_dtype(work_df[c2]):
                                    corr = work_df[[c1, c2]].corr().iloc[0, 1]
                                    st.info(f"Correlation between {c1} and {c2}: {corr:.4f}")
                                    # Scatter plot helpful
                                    render_chart(work_df, {"type": "scatter", "x": c1, "y": c2, "title": f"{c1} vs {c2}"})
                                    shown = True
                            if not shown:
                                st.dataframe(work_df[cols].head())
                        elif cols and aggs:
                            out = {}
                            for c in cols:
                                if c in work_df.columns and pd.api.types.is_numeric_dtype(work_df[c]):
                                    out[c] = {a: agg_series(work_df[c], a) for a in aggs}
                            if out:
                                st.write(pd.DataFrame(out))
                            else:
                                st.info("No numeric columns to aggregate or invalid columns.")
                        elif cols:
                            st.write(work_df[cols].describe(include="all"))
                    except Exception as e:
                        st.warning(f"Stats/compare step failed: {e}")

                # 3) Charts
                charts = []
                if plan.get("chart"):
                    charts.append(plan["chart"])
                if isinstance(plan.get("charts"), list):
                    charts.extend(plan["charts"])

                for spec in charts:
                    render_chart(work_df, spec)

                # 4) Analysis text
                if plan.get("analysis"):
                    st.success(plan["analysis"])

# ---------------- Model Page ----------------
elif menu == "Model":
    st.header("🤖 Modeling")
    st.info("Feature coming soon 🚧")