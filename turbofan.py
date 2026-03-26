"""
NASA CMAPSS Turbofan PHM — Streamlit application with GenAI (EPAM DIAL).

Architecture (high level)
-------------------------
- **Data layer:** KaggleHub or ``TURBOFAN_DATA_DIR`` → space-separated CMAPSS ``train_*.txt`` / ``test_*.txt``.
- **Visualization:** Plotly Express (histogram, pie, sensor time series).
- **GenAI:** OpenAI-compatible **DIAL** HTTP API for summaries and JSON extraction.
- **Evaluation:** Small gold set ``data/llm_extraction_gold.json`` — field-level accuracy of structured LLM outputs.

References: README.md and in-app "Resources" section.
"""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import kagglehub
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Constants — NASA CMAPSS column layout (26 numeric columns, whitespace-separated)
# ---------------------------------------------------------------------------

_TURBOFAN_COLS = (
    ["engine_no", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

KAGGLE_DATASET = "bishals098/nasa-turbofan-engine-degradation-simulation"

BASE_DIR = Path(__file__).resolve().parent
GOLD_EVAL_PATH = BASE_DIR / "data" / "llm_extraction_gold.json"

DIAL_API_URL = os.environ.get(
    "DIAL_API_URL",
    "https://ai-proxy.lab.epam.com/openai/deployments/gpt-4/chat/completions",
)

EXTRACTION_SYSTEM_PROMPT = (
    "You extract structured fields from turbofan PHM maintenance and operations text. "
    "Reply with a single JSON object only, keys: "
    "dataset_id (string, e.g. FD001), engine_no (integer), cycle (integer), fault_mode (string). "
    "No markdown fences, no extra commentary."
)


# ---------------------------------------------------------------------------
# Data loading — CMAPSS text logs
# ---------------------------------------------------------------------------


def try_load_turbofan_bytes(raw: bytes) -> Optional[pd.DataFrame]:
    """Parse NASA CMAPSS space-separated logs. Returns None if format does not match."""
    try:
        df = pd.read_csv(
            io.BytesIO(raw),
            sep=r"\s+",
            header=None,
            names=_TURBOFAN_COLS,
            engine="python",
        )
    except Exception:
        return None
    if df.empty or len(df.columns) != len(_TURBOFAN_COLS):
        return None
    return df


@st.cache_data(show_spinner="Loading dataset file…")
def load_turbofan_txt(file_path: str) -> pd.DataFrame:
    with open(file_path, "rb") as f:
        df = try_load_turbofan_bytes(f.read())
    if df is None:
        raise ValueError(f"Not a valid NASA CMAPSS file: {file_path}")
    return df


@st.cache_resource(show_spinner="Ensuring Kaggle dataset is available…")
def turbofan_dataset_dir() -> str:
    """Folder with train_*.txt / test_*.txt (Kaggle cache or ``TURBOFAN_DATA_DIR``)."""
    override = os.environ.get("TURBOFAN_DATA_DIR", "").strip()
    if override and os.path.isdir(override):
        return os.path.abspath(override)
    return kagglehub.dataset_download(KAGGLE_DATASET)


def list_turbofan_data_files(root: str) -> list[str]:
    return sorted(
        f
        for f in os.listdir(root)
        if f.endswith(".txt") and (f.startswith("train_") or f.startswith("test_"))
    )


# ---------------------------------------------------------------------------
# GenAI — DIAL (OpenAI-compatible chat completions)
# ---------------------------------------------------------------------------


def dial_chat_completion(
    system_text: str,
    user_text: str,
    api_key: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """Call EPAM DIAL chat completions endpoint."""
    headers = {"Content-Type": "application/json", "api-key": api_key}
    payload = {
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(DIAL_API_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        return f"HTTP {resp.status_code}: {resp.text}"
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def llm_complete(system_text: str, user_text: str, api_key: str) -> str:
    """Call DIAL chat completions (requires API key from the app UI)."""
    if not api_key.strip():
        return "Error: enter your DIAL API key in the sidebar."
    return dial_chat_completion(system_text, user_text, api_key.strip())


def extract_json_object(raw: str) -> Optional[dict[str, Any]]:
    """Parse first JSON object from model output (handles optional ```json fences)."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


def _norm_str(v: Any) -> str:
    return " ".join(str(v).lower().split())


def field_match(expected: Any, predicted: Any) -> bool:
    """Loose equality for numbers and fault strings."""
    if predicted is None:
        return False
    if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
        return int(expected) == int(float(predicted))
    return _norm_str(expected) == _norm_str(predicted) or _norm_str(expected) in _norm_str(predicted)


def score_extraction(expected: dict[str, Any], predicted: Optional[dict[str, Any]]) -> tuple[float, dict[str, bool]]:
    """Return mean accuracy over keys present in ``expected`` and per-key flags."""
    if not predicted:
        return 0.0, {k: False for k in expected}
    flags = {}
    for k, ev in expected.items():
        flags[k] = field_match(ev, predicted.get(k))
    acc = sum(flags.values()) / len(flags) if flags else 0.0
    return acc, flags


@st.cache_data
def load_gold_eval(path_str: str) -> dict[str, Any]:
    with open(path_str, encoding="utf-8") as f:
        return json.load(f)


def run_gold_evaluation(api_key: str) -> tuple[list[dict[str, Any]], float]:
    """Run all gold samples; return row dicts and overall mean field accuracy."""
    pack = load_gold_eval(str(GOLD_EVAL_PATH))
    samples = pack.get("samples", [])
    rows: list[dict[str, Any]] = []
    total_acc = 0.0
    n = 0
    for s in samples:
        text = s["text"]
        expected = s["expected"]
        user_prompt = f"Extract fields from this text:\n\n{text}\n\nRespond with JSON only."
        raw = llm_complete(EXTRACTION_SYSTEM_PROMPT, user_prompt, api_key)
        pred = extract_json_object(raw)
        acc, _ = score_extraction(expected, pred)
        total_acc += acc
        n += 1
        rows.append(
            {
                "id": s["id"],
                "field_accuracy": round(acc, 3),
                "expected": json.dumps(expected),
                "predicted": json.dumps(pred) if pred else raw[:200],
            }
        )
    mean_acc = total_acc / n if n else 0.0
    return rows, mean_acc


def dataframe_summary_block(df: pd.DataFrame, max_engines: int = 5) -> str:
    """Compact stats for LLM context (no raw table dump)."""
    eng = df["engine_no"].nunique() if "engine_no" in df.columns else 0
    lines = [
        f"rows={len(df)}, engines={eng}",
        f"cycle min/max: {df['cycle'].min()}-{df['cycle'].max()}" if "cycle" in df.columns else "",
    ]
    if "engine_no" in df.columns:
        vc = df["engine_no"].value_counts().head(max_engines)
        lines.append("top engines by row count: " + vc.to_string())
    return "\n".join(x for x in lines if x)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Turbofan PHM Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("NASA CMAPSS Turbofan — telemetry, charts & GenAI extraction")
st.markdown(
    """
**Dataset (auto-download via KaggleHub):**
[NASA Turbofan — bishals098](https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation) ·
[Alternate listing](https://www.kaggle.com/datasets/behradkamyab/turbofan-engine-degradation-simulation-data)

**Related tutorial:** [Two-stage retrieval RAG + rerank](https://www.kaggle.com/code/warcoder/two-stage-retrieval-rag-using-rerank-models)

This app uses **DIAL** (EPAM AI proxy, OpenAI-compatible) for summaries and structured JSON extraction.
See ``README.md`` for setup.
"""
)

# --- Sidebar: data + credentials ---
with st.sidebar:
    st.header("Data source")
    try:
        dataset_path = turbofan_dataset_dir()
    except Exception as e:
        st.error(f"Dataset unavailable: {e}")
        dataset_path = ""
        st.info("Set `TURBOFAN_DATA_DIR` or configure [Kaggle API](https://www.kaggle.com/docs/api).")
    if dataset_path:
        st.caption("Data directory")
        st.code(dataset_path, language="text")
        data_files = list_turbofan_data_files(dataset_path)
    else:
        data_files = []
    default_idx = data_files.index("train_FD001.txt") if "train_FD001.txt" in data_files else 0
    selected_name = (
        st.selectbox("CMAPSS file", data_files, index=default_idx) if data_files else None
    )
    uploaded_file = st.file_uploader("Or upload CSV / CMAPSS .txt / PDF", type=["csv", "pdf", "txt"])

    st.header("DIAL")
    dial_key = st.text_input(
        "DIAL API key",
        type="password",
        key="dial_api_key",
        help="Enter your EPAM DIAL API key here. It is not read from environment or secrets files—only this field.",
    )

# Resolve loaded dataframe
df: Optional[pd.DataFrame] = None
data_source: Optional[str] = None

if uploaded_file is not None:
    data_source = f"upload:{uploaded_file.name}"
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".txt"):
        df = try_load_turbofan_bytes(uploaded_file.getvalue())
    else:
        df = None
elif dataset_path and selected_name:
    data_source = f"kaggle:{selected_name}"
    df = load_turbofan_txt(os.path.join(dataset_path, selected_name))

api_key_resolved = (dial_key or "").strip()

tab_telemetry, tab_llm, tab_eval = st.tabs(
    ["Telemetry & charts", "LLM analysis", "Gold-set evaluation"]
)

# ----- Tab: telemetry -----
with tab_telemetry:
    if df is not None:
        st.success(f"**Source:** {data_source} — **{len(df):,}** rows")
        st.dataframe(df.head(12), use_container_width=True)

        if "engine_no" in df.columns and "cycle" in df.columns:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    px.histogram(df, x="cycle", color="engine_no", title="Cycle distribution by engine"),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    px.pie(df, names="engine_no", title="Row share by engine"),
                    use_container_width=True,
                )

            sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
            eng_list = sorted(df["engine_no"].unique().tolist())
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                pick_eng = st.selectbox("Engine for line chart", eng_list, index=0)
            with ec2:
                pick_sensor = st.selectbox("Sensor", sensor_cols, index=min(1, len(sensor_cols) - 1))
            with ec3:
                st.caption("Shows sensor vs cycle for one engine (degradation trends).")
            sub = df[df["engine_no"] == pick_eng]
            st.plotly_chart(
                px.line(
                    sub,
                    x="cycle",
                    y=pick_sensor,
                    title=f"{pick_sensor} vs cycle (engine {pick_eng})",
                    markers=True,
                ),
                use_container_width=True,
            )
        else:
            st.warning("Uploaded CSV must include `engine_no` and `cycle` for CMAPSS-style charts.")
    else:
        st.info("Choose a sidebar file or upload a CSV / NASA CMAPSS `.txt` (e.g. `test_FD001.txt`).")
        if uploaded_file is not None and uploaded_file.name.lower().endswith(".pdf"):
            st.caption("PDFs are handled in **LLM analysis** (text extraction path).")

# ----- Tab: LLM -----
with tab_llm:
    st.markdown(
        """
Use **DIAL** to (1) summarize loaded telemetry and (2) extract JSON from free text.
For PDFs, upload in the sidebar and use the document section below.
"""
    )
    if df is not None and st.button("Summarize loaded telemetry with LLM"):
        summary_in = dataframe_summary_block(df)
        sys_p = "You are a PHM analyst. Write 3-5 short bullet points on fleet health and what to watch next."
        out = llm_complete(sys_p, summary_in, api_key_resolved)
        st.markdown(out)

    st.subheader("Custom text — structured extraction")
    custom = st.text_area(
        "Paste maintenance / ops text",
        height=120,
        placeholder="e.g. FD001 engine 3 cycle 90 HPC degradation …",
    )
    if st.button("Extract JSON (dataset_id, engine_no, cycle, fault_mode)"):
        if not custom.strip():
            st.warning("Enter some text first.")
        else:
            u = f"Text:\n{custom.strip()}\n\nReturn JSON only."
            raw = llm_complete(EXTRACTION_SYSTEM_PROMPT, u, api_key_resolved)
            st.code(raw, language="json")
            parsed = extract_json_object(raw)
            if parsed:
                st.json(parsed)

    if uploaded_file is not None and df is None and uploaded_file.name.lower().endswith(".pdf"):
        st.subheader("Uploaded PDF")
        st.caption("Streamlit cannot parse PDF bytes here without extra libraries. Paste extracted text below or use a `.txt` export.")
        pdf_note = st.text_area("Paste PDF text content", height=200)
        if st.button("Extract JSON from pasted PDF text") and pdf_note.strip():
            raw = llm_complete(
                EXTRACTION_SYSTEM_PROMPT,
                f"Text:\n{pdf_note.strip()}\n\nReturn JSON only.",
                api_key_resolved,
            )
            st.code(raw, language="json")
            p = extract_json_object(raw)
            if p:
                st.json(p)

# ----- Tab: evaluation -----
with tab_eval:
    st.markdown(
        """
### Evaluation metric (assignment)

We measure **field-level accuracy** on a **small fixed gold set** (`data/llm_extraction_gold.json`):
for each synthetic snippet, the model must emit JSON with
`dataset_id`, `engine_no`, `cycle`, `fault_mode`.  
Per-sample score = fraction of matching fields; **overall** = mean across samples.

This quantifies **quality of generative structured extraction** (not retrieval RAG itself — see linked Kaggle notebook for RAG patterns).
"""
    )
    if not GOLD_EVAL_PATH.is_file():
        st.error(f"Gold file missing: {GOLD_EVAL_PATH}")
    elif st.button("Run gold-set evaluation (calls LLM per sample)"):
        if not api_key_resolved.strip():
            st.error("Enter your **DIAL API key** in the sidebar.")
        else:
            with st.spinner("Evaluating (one DIAL call per gold sample)…"):
                rows, mean_acc = run_gold_evaluation(api_key_resolved)
            st.metric("Mean field-level accuracy (across samples)", f"{mean_acc:.1%}")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.caption(f"Gold file path: `{GOLD_EVAL_PATH}`")

st.markdown("---")
st.markdown(
    """
**Resources**
- [Dataset — Kaggle (bishals098)](https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation)
- [RAG + rerank tutorial](https://www.kaggle.com/code/warcoder/two-stage-retrieval-rag-using-rerank-models)
"""
)
