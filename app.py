"""
Football Predictor — Over 4.5 Goals & BTTS First Half only
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from pathlib import Path

st.set_page_config(page_title="Football O4.5 & BTTS FH", page_icon="⚽", layout="centered")

st.markdown("""
<style>
.stMetric { background: rgba(28,131,225,0.08); border: 1px solid rgba(28,131,225,0.2);
            padding: 12px 16px; border-radius: 10px; }
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load():
    return joblib.load(Path(__file__).parent / "football_over45_btts_model.joblib")

data = load()
TEAMS = data["teams"]
form_hist = data["team_form"]
WINDOW = 8

def get_form(team):
    h = form_hist.get(team, [])[-WINDOW:]
    if len(h) < 3:
        return None
    return {k: float(np.mean([x[k] for x in h])) for k in h[0]}

def predict(home, away):
    hf, af = get_form(home), get_form(away)
    if not hf or not af:
        return None
    # Over 4.5 features
    row_o = {
        "home_gf": hf["gf"], "home_ga": hf["ga"],
        "away_gf": af["gf"], "away_ga": af["ga"],
        "exp_goals": hf["gf"]+af["gf"]+hf["ga"]+af["ga"],
        "home_over45": hf["over45"], "away_over45": af["over45"],
    }
    feats_o = data["over45_feats"]
    Xo = pd.DataFrame([row_o])[feats_o].astype(np.float32)
    do = xgb.DMatrix(Xo, feature_names=feats_o)
    p_over = float(data["over45_model"].predict(do, iteration_range=(0, data["over45_best_iter"]+1))[0])

    # BTTS FH features
    row_b = {
        "home_gf_fh": hf["gf_fh"], "home_ga_fh": hf["ga_fh"],
        "away_gf_fh": af["gf_fh"], "away_ga_fh": af["ga_fh"],
        "home_btts_fh": hf["btts_fh"], "away_btts_fh": af["btts_fh"],
    }
    feats_b = data["btts_fh_feats"]
    Xb = pd.DataFrame([row_b])[feats_b].astype(np.float32)
    db = xgb.DMatrix(Xb, feature_names=feats_b)
    p_btts = float(data["btts_fh_model"].predict(db, iteration_range=(0, data["btts_fh_best_iter"]+1))[0])

    return {
        "p_over45": p_over,
        "p_btts_fh": p_btts,
        "exp_goals": row_o["exp_goals"],
        "home_form_gf": hf["gf"], "away_form_gf": af["gf"],
    }

st.title("⚽ Over 4.5 & BTTS First Half")
st.caption("Major European leagues · Form-based XGBoost models")

c1, c2 = st.columns(2)
with c1:
    home = st.selectbox("Home team", TEAMS, index=TEAMS.index("Arsenal") if "Arsenal" in TEAMS else 0)
with c2:
    away = st.selectbox("Away team", TEAMS, index=TEAMS.index("Chelsea") if "Chelsea" in TEAMS else 1)

if st.button("Predict", type="primary", use_container_width=True):
    if home == away:
        st.warning("Select two different teams.")
    else:
        res = predict(home, away)
        if res is None:
            st.error("Not enough recent form data for one or both teams.")
        else:
            st.markdown("---")
            st.subheader("Predictions")

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Over 4.5 Goals", f"{res['p_over45']*100:.1f}%")
                st.caption(f"Implied decimal odds ≈ {1/max(res['p_over45'],0.02):.2f}")
            with m2:
                st.metric("BTTS First Half", f"{res['p_btts_fh']*100:.1f}%")
                st.caption(f"Implied decimal odds ≈ {1/max(res['p_btts_fh'],0.02):.2f}")

            st.progress(res["p_over45"], text=f"Over 4.5 → {res['p_over45']*100:.1f}%")
            st.progress(res["p_btts_fh"], text=f"BTTS FH → {res['p_btts_fh']*100:.1f}%")

            with st.expander("Form snapshot"):
                st.write({
                    "Home avg goals scored (last 8)": round(res["home_form_gf"], 2),
                    "Away avg goals scored (last 8)": round(res["away_form_gf"], 2),
                    "Rough expected total goals": round(res["exp_goals"], 2),
                })

st.markdown("---")
st.caption("Models trained on PL, Championship, La Liga, Serie A, Bundesliga (2017–2025). "
           "Over 4.5 base rate ≈ 14% · BTTS FH base rate ≈ 19%. Educational use only.")
