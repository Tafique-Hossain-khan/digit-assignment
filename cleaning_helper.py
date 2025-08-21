import json
import math
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd


def _build_schema(df: pd.DataFrame) -> Dict[str, Any]:
	schema_cols = []
	for col in df.columns:
		s = df[col]
		missing = float(s.isna().mean())
		dtype = str(s.dtype)
		nunique = int(s.nunique(dropna=True))
		is_numeric = bool(pd.api.types.is_numeric_dtype(s))
		maybe_date = False
		if not is_numeric and nunique > 0:
			try:
				pd.to_datetime(s.dropna().head(200), errors="raise")
				maybe_date = True
			except Exception:
				maybe_date = False
		constant = bool(nunique == 1)
		schema_cols.append({
			"name": col,
			"dtype": dtype,
			"missing_ratio": round(missing, 4),
			"nunique": nunique,
			"is_numeric": is_numeric,
			"maybe_date": maybe_date,
			"is_constant": constant,
		})
	return {"n_rows": int(len(df)), "n_cols": int(df.shape[1]), "columns": schema_cols}


PLAN_SCHEMA_TEXT = """
Return STRICT JSON with this shape and nothing else:

{
  "drop_columns": ["colA", "colB"],                          // optional
  "drop_rows_with_missing_threshold": 0.6,                   // optional (0..1)
  "imputations": [                                           // optional
    {"column": "Age", "strategy": "median"},
    {"column": "Country", "strategy": "most_frequent"},
    {"column": "Price", "strategy": "mean"},
    {"column": "Date", "strategy": "ffill"},
    {"column": "SomeCol", "strategy": "constant", "value": 0}
  ],
  "duplicates": {"action": "drop"},                          // optional
  "dtypes": [                                                // optional
    {"column": "Date", "to": "datetime"},
    {"column": "IsActive", "to": "bool"},
    {"column": "Category", "to": "category"},
    {"column": "Id", "to": "int"},
    {"column": "Amount", "to": "float"},
    {"column": "Notes", "to": "string"}
  ],
  "outliers": [                                              // optional
    {"column": "Amount", "method": "cap", "params": {"lower_quantile": 0.01, "upper_quantile": 0.99}},
    {"column": "Score", "method": "remove", "params": {"zscore_threshold": 3.0}},
    {"column": "Salary", "method": "winsorize", "params": {"lower_quantile": 0.05, "upper_quantile": 0.95}},
    {"column": "Skewed", "method": "log", "params": {}}
  ],
  "notes": "Short reasoning of chosen steps."
}

Rules and selection guidance:
- Prefer dropping columns if missing_ratio > 0.8 or column is constant with little value.
- Use minimally destructive imputations: numeric -> median (robust), categorical -> most_frequent, date/time series -> ffill/bfill.
- Remove duplicates (action=drop) when many exact duplicates exist, otherwise omit.
- Fix obviously wrong dtypes (parse dates, numeric strings to numbers, categories).
- Handle outliers for numeric columns only. Prefer capping to [1%,99%] or IQR-based removal; log transform for heavy right skew.
- Only include steps that make sense for the provided schema. Omit others.
"""


def plan_cleaning(df: pd.DataFrame, llm) -> Dict[str, Any]:
	"""
	Build a compact schema and ask the LLM for a JSON cleaning plan.
	llm must support .invoke({"input": prompt}) or .invoke(dict) with ChatPromptTemplate upstream.
	"""
	schema = _build_schema(df)
	prompt = f"""
You are a data cleaning expert. Decide the best cleaning plan for this dataset.

Dataset schema (JSON):
{json.dumps(schema, ensure_ascii=False)}

{PLAN_SCHEMA_TEXT}
"""
	# llm can be a LangChain ChatModel; use direct .invoke on the prompt string
	resp = llm.invoke(prompt)
	raw = getattr(resp, "content", None) if resp is not None else None
	if not raw:
		raise ValueError("Empty response from model.")

	# Extract first {...} JSON object
	import re
	m = re.search(r"\{.*\}", raw, re.DOTALL)
	if not m:
		raise ValueError("No JSON object found in model output.")
	plan = json.loads(m.group())
	return plan


def _convert_dtype(s: pd.Series, to: str) -> pd.Series:
	to = (to or "").lower()
	try:
		if to == "datetime":
			return pd.to_datetime(s, errors="coerce")
		if to == "int":
			x = pd.to_numeric(s, errors="coerce")
			return x.round().astype("Int64")
		if to == "float":
			return pd.to_numeric(s, errors="coerce").astype(float)
		if to == "bool":
			if s.dtype == bool:
				return s
			return s.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}).astype("boolean")
		if to == "category":
			return s.astype("category")
		if to in ["str", "string"]:
			return s.astype("string")
	except Exception:
		return s
	return s


def _impute_series(s: pd.Series, strategy: str, value: Any = None) -> pd.Series:
	strategy = (strategy or "").lower()
	try:
		if strategy == "mean":
			return s.fillna(s.astype(float).mean())
		if strategy == "median":
			return s.fillna(s.astype(float).median())
		if strategy in ["mode", "most_frequent"]:
			if s.dropna().empty:
				return s.fillna(method="ffill").fillna(method="bfill")
			return s.fillna(s.mode(dropna=True).iloc[0])
		if strategy == "ffill":
			return s.fillna(method="ffill")
		if strategy == "bfill":
			return s.fillna(method="bfill")
		if strategy == "constant":
			return s.fillna(value)
	except Exception:
		return s
	return s


def _cap_outliers(s: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
	if not pd.api.types.is_numeric_dtype(s):
		return s
	lb = s.quantile(lower_q)
	ub = s.quantile(upper_q)
	return s.clip(lb, ub)


def _remove_outliers_z(s: pd.Series, z: float = 3.0) -> pd.Series:
	if not pd.api.types.is_numeric_dtype(s):
		return s
	mu = s.mean()
	sd = s.std(ddof=0)
	if not np.isfinite(sd) or sd == 0:
		return s
	mask = (np.abs((s - mu) / sd) <= z) | s.isna()
	return s.where(mask)


def _winsorize(s: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
	return _cap_outliers(s, lower_q, upper_q)


def _log_transform(s: pd.Series) -> pd.Series:
	if not pd.api.types.is_numeric_dtype(s):
		return s
	# Shift to be >=0
	minv = s.min(skipna=True)
	shift = 0.0 if (pd.isna(minv) or minv >= 0) else (0 - minv)
	return np.log1p(s + shift)


def apply_cleaning_plan(df: pd.DataFrame, plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
	report: Dict[str, Any] = {"steps": []}
	out = df.copy()

	# Drop columns
	for col in plan.get("drop_columns", []) or []:
		if col in out.columns:
			out = out.drop(columns=[col])
			report["steps"].append(f"dropped_column:{col}")

	# Drop rows with too many missing
	thr = plan.get("drop_rows_with_missing_threshold", None)
	if isinstance(thr, (int, float)) and 0 <= thr <= 1:
		row_miss = out.isna().mean(axis=1)
		before = len(out)
		out = out.loc[row_miss <= float(thr)]
		report["steps"].append(f"dropped_rows_missing>{thr}: {before - len(out)}")

	# Dtype fixes
	for d in plan.get("dtypes", []) or []:
		col = d.get("column")
		to = d.get("to")
		if col in out.columns and to:
			out[col] = _convert_dtype(out[col], to)
			report["steps"].append(f"convert_dtype:{col}->{to}")

	# Imputations
	for imp in plan.get("imputations", []) or []:
		col = imp.get("column")
		strategy = imp.get("strategy")
		val = imp.get("value")
		if col in out.columns and strategy:
			out[col] = _impute_series(out[col], strategy, val)
			report["steps"].append(f"impute:{col}:{strategy}")

	# Duplicates
	dup = plan.get("duplicates", {}) or {}
	if (dup.get("action") or "").lower() == "drop":
		before = len(out)
		out = out.drop_duplicates()
		report["steps"].append(f"drop_duplicates:{before - len(out)}")

	# Outliers
	for spec in plan.get("outliers", []) or []:
		col = spec.get("column")
		method = (spec.get("method") or "").lower()
		params = spec.get("params") or {}
		if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
			continue
		if method == "remove":
			z = float(params.get("zscore_threshold", 0) or 3.0)
			new_s = _remove_outliers_z(out[col], z)
			removed = int(out[col].notna().sum() - new_s.notna().sum())
			out[col] = new_s
			report["steps"].append(f"outliers_remove:{col}:z>{z}:{removed}")
		elif method in ["cap", "winsorize"]:
			lq = float(params.get("lower_quantile", 0.01) or 0.01)
			uq = float(params.get("upper_quantile", 0.99) or 0.99)
			out[col] = _cap_outliers(out[col], lq, uq) if method == "cap" else _winsorize(out[col], lq, uq)
			report["steps"].append(f"outliers_{method}:{col}:{lq}-{uq}")
		elif method == "log":
			out[col] = _log_transform(out[col])
			report["steps"].append(f"outliers_log:{col}")

	report["shape_before"] = df.shape
	report["shape_after"] = out.shape
	return out, report


def heuristic_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
	"""
	Safe fallback when LLM plan is unavailable: basic cleaning.
	"""
	rep: Dict[str, Any] = {"steps": []}
	out = df.copy()

	# Drop columns with >60% missing
	col_missing = out.isna().mean()
	to_drop = [c for c, r in col_missing.items() if r > 0.6]
	if to_drop:
		out = out.drop(columns=to_drop)
		rep["steps"].append(f"dropped_columns_missing>0.6:{to_drop}")

	# Fix common dtypes
	for c in out.columns:
		s = out[c]
		if s.dtype == "object":
			# try numeric
			num = pd.to_numeric(s, errors="coerce")
			if num.notna().sum() > 0.7 * len(s):
				out[c] = num
				rep["steps"].append(f"convert_dtype:{c}->float")
				continue
			# try datetime
			try:
				dt = pd.to_datetime(s, errors="raise")
				out[c] = dt
				rep["steps"].append(f"convert_dtype:{c}->datetime")
				continue
			except Exception:
				pass

	# Impute: numeric->median, categorical->mode
	for c in out.columns:
		s = out[c]
		if s.isna().any():
			if pd.api.types.is_numeric_dtype(s):
				out[c] = s.fillna(s.median())
				rep["steps"].append(f"impute:{c}:median")
			elif pd.api.types.is_datetime64_any_dtype(s):
				out[c] = s.fillna(method="ffill").fillna(method="bfill")
				rep["steps"].append(f"impute:{c}:ffill/bfill")
			else:
				mode_vals = s.mode(dropna=True)
				fillv = mode_vals.iloc[0] if not mode_vals.empty else ""
				out[c] = s.fillna(fillv)
				rep["steps"].append(f"impute:{c}:most_frequent")

	# Drop duplicates
	before = len(out)
	out = out.drop_duplicates()
	rep["steps"].append(f"drop_duplicates:{before - len(out)}")

	# Cap outliers for numeric columns to [1%,99%]
	for c in out.select_dtypes(include="number").columns:
		out[c] = _cap_outliers(out[c], 0.01, 0.99)
		rep["steps"].append(f"outliers_cap:{c}:0.01-0.99")

	rep["shape_before"] = df.shape
	rep["shape_after"] = out.shape
	return out, rep
