"""
Football Predictor
Over 2.5 · Over 4.5 · BTTS First Half · Total Corners
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from pathlib import Path
from scipy.stats import norm

st.set_page_config(page_title="Football Markets", page_icon="⚽", layout="centered")
st.markdown("""
<style>
.stMetric { background: rgba(28,131,225,0.08); border: 1px solid rgba(28,131,225,0.2);
            padding: 10px 14px; border-radius: 10px; }
div[data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load():
    p = Path(__file__).parent
    # prefer new combined model, fall back to old
    for name in ["football_model.joblib", "football_over45_btts_model.joblib"]:
        f = p / name
        if f.exists():
            return joblib.load(f)
    raise FileNotFoundError("No model file found")

data = load()
TEAMS = data["teams"]
form_hist = data["team_form"]
WINDOW = 8

def get_form(team):
    h = form_hist.get(team, [])[-WINDOW:]
    if len(h) < 3:
        return None
    return {k: float(np.nanmean([x[k] for x in h])) for k in h[0]}

def predict(home, away):
    hf, af = get_form(home), get_form(away)
    if not hf or not af:
        return None
    out = {}

    # shared goal features
    row_g = {
        "home_gf": hf.get("gf", 1.2), "home_ga": hf.get("ga", 1.2),
        "away_gf": af.get("gf", 1.2), "away_ga": af.get("ga", 1.2),
        "exp_goals": hf.get("gf",1.2)+af.get("gf",1.2)+hf.get("ga",1.2)+af.get("ga",1.2),
        "home_over25": hf.get("over25", 0.5), "away_over25": af.get("over25", 0.5),
        "home_over45": hf.get("over45", 0.14), "away_over45": af.get("over45", 0.14),
    }

    def pred_cls(model_key, feats_key, iter_key, row):
        feats = data[feats_key]
        X = pd.DataFrame([{k: row.get(k, 0) for k in feats}])[feats].astype(np.float32)
        d = xgb.DMatrix(X, feature_names=feats)
        return float(data[model_key].predict(d, iteration_range=(0, data[iter_key]+1))[0])

    if "over25_model" in data:
        out["p_over25"] = pred_cls("over25_model", "over25_feats", "over25_best_iter", row_g)
    out["p_over45"] = pred_cls("over45_model", "over45_feats", "over45_best_iter", row_g)

    row_b = {
        "home_gf_fh": hf.get("gf_fh", 0.5), "home_ga_fh": hf.get("ga_fh", 0.5),
        "away_gf_fh": af.get("gf_fh", 0.5), "away_ga_fh": af.get("ga_fh", 0.5),
        "home_btts_fh": hf.get("btts_fh", 0.19), "away_btts_fh": af.get("btts_fh", 0.19),
    }
    out["p_btts_fh"] = pred_cls("btts_fh_model", "btts_fh_feats", "btts_fh_best_iter", row_b)

    # corners
    if "corners_model" in data:
        row_c = {
            "home_corners": hf.get("corners", 5.0), "away_corners": af.get("corners", 4.5),
            "home_gf": hf.get("gf", 1.2), "away_gf": af.get("gf", 1.2),
            "exp_goals": row_g["exp_goals"],
        }
        feats = data["corners_feats"]
        X = pd.DataFrame([{k: row_c.get(k, 0) for k in feats}])[feats].astype(np.float32)
        d = xgb.DMatrix(X, feature_names=feats)
        exp_c = float(data["corners_model"].predict(d, iteration_range=(0, data["corners_best_iter"]+1))[0])
        std = data.get("corners_std", 3.0)
        out["exp_corners"] = exp_c
        out["corners_ou"] = {}
        for line in [8.5, 9.5, 10.5, 11.5]:
            p_over = 1 - norm.cdf(line, loc=exp_c, scale=std)
            out["corners_ou"][line] = {"over": p_over, "under": 1-p_over}

    out["exp_goals"] = row_g["exp_goals"]
    out["home_gf"] = hf.get("gf", 0)
    out["away_gf"] = af.get("gf", 0)
    return out

st.title("⚽ Football Markets")
st.caption("Over 2.5 · Over 4.5 · BTTS FH · Total Corners")

c1, c2 = st.columns(2)
with c1:
    home = st.selectbox("Home", TEAMS, index=min(TEAMS.index("Arsenal") if "Arsenal" in TEAMS else 0, len(TEAMS)-1))
with c2:
    away = st.selectbox("Away", TEAMS, index=min(TEAMS.index("Chelsea") if "Chelsea" in TEAMS else 1, len(TEAMS)-1))

if st.button("Predict", type="primary", use_container_width=True):
    if home == away:
        st.warning("Pick two different teams.")
    else:
        res = predict(home, away)
        if res is None:
            st.error("Not enough form data for one or both teams.")
        else:
            st.markdown("---")
            st.subheader("Goals")
            cols = st.columns(3 if "p_over25" in res else 2)
            i = 0
            if "p_over25" in res:
                with cols[i]:
                    st.metric("Over 2.5", f"{res['p_over25']*100:.1f}%")
                    st.caption(f"Odds ≈ {1/max(res['p_over25'],0.02):.2f}")
                i += 1
            with cols[i]:
                st.metric("Over 4.5", f"{res['p_over45']*100:.1f}%")
                st.caption(f"Odds ≈ {1/max(res['p_over45'],0.02):.2f}")
            with cols[i+1 if 'p_over25' in res else i]:
                st.metric("BTTS First Half", f"{res['p_btts_fh']*100:.1f}%")
                st.caption(f"Odds ≈ {1/max(res['p_btts_fh'],0.02):.2f}")

            if "p_over25" in res:
                st.progress(res["p_over25"], text=f"Over 2.5 → {res['p_over25']*100:.1f}%")
            st.progress(res["p_over45"], text=f"Over 4.5 → {res['p_over45']*100:.1f}%")
            st.progress(res["p_btts_fh"], text=f"BTTS FH → {res['p_btts_fh']*100:.1f}%")

            if "exp_corners" in res:
                st.markdown("---")
                st.subheader("Corners")
                st.metric("Expected total corners", f"{res['exp_corners']:.1f}")
                rows = [{"Line": line, "Over %": f"{v['over']*100:.0f}%", "Under %": f"{v['under']*100:.0f}%"}
                        for line, v in res["corners_ou"].items()]
                st.table(pd.DataFrame(rows))

            with st.expander("Form snapshot"):
                st.write({
                    "Home avg GF (last 8)": round(res["home_gf"], 2),
                    "Away avg GF (last 8)": round(res["away_gf"], 2),
                    "Rough expected goals": round(res["exp_goals"], 2),
                })

st.markdown("---")
st.caption("PL, Championship, La Liga, Serie A, Bundesliga · 2017–2025 · Educational use only")
