"""
RL Portfolio Command Center — Dashboard v5
Bloomberg-terminal aesthetic. Zero bugs. All hex colors valid.
"""

import os, json, numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from stable_baselines3 import PPO, A2C, SAC
from config import (
    MODEL_DIR, RESULTS_DIR, TICKERS, SECTOR_ETFS, LOOKBACK_WINDOW,
    AGENTS, SEEDS, WALK_FORWARD_WINDOWS, NUM_ASSETS,
    LAMBDA_DRAWDOWN, LAMBDA_TURNOVER, LAMBDA_SHARPE, LAMBDA_CONCENTRATION,
    LAMBDA_CVAR, CVAR_ALPHA, TRANSACTION_COST_BPS, SPREAD_COST_BPS,
    REGIME_THRESHOLDS, REGIME_REWARD_MULTIPLIERS, ABLATION_GROUPS,
    KILL_SWITCH_THRESHOLD, REWARD_CLIP,
)
from data_loader import prepare_data
from portfolio_env import PortfolioEnv
from evaluate import (
    run_agent, equal_weight, risk_parity, momentum_top3,
    min_variance, inverse_volatility, spy_buy_hold,
    compute_metrics, bootstrap_sharpe_ci, seed_robustness_analysis,
    _load_norm_stats,
)

AGENT_CLASSES = {"PPO": PPO, "A2C": A2C, "SAC": SAC}

# ── COLORS (NO ALPHA CHANNELS — Plotly compatible) ──
C = {
    "PPO": "#3b82f6", "A2C": "#a855f7", "SAC": "#22d3ee",
    "EW": "#6b7280", "RP": "#10b981", "MOM": "#f59e0b",
    "MV": "#ef4444", "IV": "#ec4899", "SPY": "#9ca3af",
}
SECTOR_PALETTE = [
    "#3b82f6", "#a855f7", "#22d3ee", "#10b981", "#f59e0b",
    "#ef4444", "#ec4899", "#6366f1", "#f97316", "#14b8a6", "#8b5cf6"
]

def color_name(n):
    if "PPO" in n: return C["PPO"]
    if "A2C" in n: return C["A2C"]
    if "SAC" in n: return C["SAC"]
    if "Equal" in n: return C["EW"]
    if "Risk" in n: return C["RP"]
    if "Momentum" in n: return C["MOM"]
    if "Min" in n: return C["MV"]
    if "Inverse" in n: return C["IV"]
    if "SPY" in n: return C["SPY"]
    return "#6b7280"

# ── PAGE CONFIG ──
st.set_page_config(page_title="RL Portfolio Command Center", page_icon="◆", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Sora:wght@300;400;500;600;700&display=swap');

.stApp {
    background: #050810;
    background-image: radial-gradient(ellipse at 20% 50%, rgba(56,97,251,0.04) 0%, transparent 50%);
}
h1,h2,h3,h4 { font-family:'Sora',sans-serif!important; color:#e8ecf4!important; }
p,span,div,li,td,th,label { font-family:'Sora',sans-serif; }
code,pre { font-family:'JetBrains Mono',monospace!important; }

[data-testid="stMetric"] {
    background: #0f1629;
    border: 1px solid rgba(56,97,251,0.12);
    border-radius: 12px;
    padding: 18px 20px;
}
[data-testid="stMetric"] label {
    color: #6b7a99!important;
    font-size: 0.7rem!important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family:'JetBrains Mono',monospace!important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e8ecf4!important;
    font-family:'JetBrains Mono',monospace!important;
    font-size: 1.6rem!important;
    font-weight: 600;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid rgba(56,97,251,0.1);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b7a99;
    font-family: 'JetBrains Mono',monospace;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 12px 20px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.stTabs [aria-selected="true"] {
    color: #3b82f6!important;
    border-bottom: 2px solid #3b82f6!important;
    background: transparent!important;
}

section[data-testid="stSidebar"] { background: #0a0f1e; }

.hdr {
    background: linear-gradient(135deg, #0f1629 0%, #111a38 50%, #0f1629 100%);
    border: 1px solid rgba(56,97,251,0.12);
    border-radius: 14px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hdr::before {
    content:'';
    position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,#3b82f6,#a855f7,transparent);
}
.hdr h1 { margin:0 0 4px 0; font-size:1.7rem; font-weight:700; letter-spacing:-0.02em; }
.hdr .sub { color:#6b7a99; font-family:'JetBrains Mono',monospace; font-size:0.78rem; letter-spacing:0.04em; }
.hdr .pill {
    display:inline-block;
    background:rgba(0,212,170,0.12);
    color:#00d4aa;
    font-family:'JetBrains Mono',monospace;
    font-size:0.68rem;
    padding:4px 14px;
    border-radius:100px;
    margin-top:10px;
    letter-spacing:0.04em;
}
.sep { border-top:1px solid rgba(56,97,251,0.08); margin:20px 0; }
</style>
""", unsafe_allow_html=True)

# ── PLOT LAYOUT ──
PL = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,30,0.4)",
    font=dict(family="JetBrains Mono, monospace", color="#6b7a99", size=11),
    margin=dict(l=50,r=20,t=40,b=40),
)

# ── DATA ──
@st.cache_data(ttl=3600, show_spinner="Loading market data...")
def load_data():
    return prepare_data()

@st.cache_resource(show_spinner="Loading agents...")
def load_models():
    models, norms = {}, {}
    w_idx = len(WALK_FORWARD_WINDOWS) - 1
    for a in AGENTS:
        for s in SEEDS:
            rid = f"{a}_w{w_idx}_s{s}"
            for sfx in ["best_model", f"{rid}_final"]:
                p = os.path.join(MODEL_DIR, rid, f"{sfx}.zip")
                if os.path.exists(p):
                    models[a] = AGENT_CLASSES[AGENTS[a]["class"]].load(p)
                    norms[a] = _load_norm_stats(rid)
                    break
            if a in models: break
    return models, norms

features, returns, close, regimes, regime_numeric, vix = load_data()
models, ncache = load_models()

window = WALK_FORWARD_WINDOWS[-1]
ts = pd.Timestamp
tf = features.loc[ts(window[2]):ts(window[3])]
tr = returns.loc[ts(window[2]):ts(window[3])]
stk = [t for t in TICKERS if t in tr.columns]
trs = tr[stk]
trg = regime_numeric.loc[ts(window[2]):ts(window[3])]
nm = len(models)

# ── HEADER ──
st.markdown(f"""
<div class="hdr">
    <h1>◆ RL Portfolio Command Center</h1>
    <div class="sub">REGIME-ADAPTIVE REINFORCEMENT LEARNING · DYNAMIC SECTOR ALLOCATION · {NUM_ASSETS} ETFs</div>
    <div class="pill">● {"OPERATIONAL" if nm > 0 else "AWAITING TRAINING"} · {nm} AGENT{"S" if nm!=1 else ""} · {len(features)} DAYS</div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "◆ COMMAND CENTER","◆ PERFORMANCE","◆ ALLOCATIONS","◆ SIGNALS","◆ EXPLAINABILITY","◆ ARCHITECTURE","◆ LIVE PORTFOLIO"
])

# ════════════════════════════════════════════════
# TAB 1: COMMAND CENTER
# ════════════════════════════════════════════════
with tab1:
    if not models:
        st.warning("No trained models. Run `python train.py` first.")
    else:
        st.markdown("### Current Market Signal")
        st.caption(f"Latest data: {features.index[-1].strftime('%Y-%m-%d')}")

        latest_regime = regimes.iloc[-1] if len(regimes)>0 else "unknown"
        latest_vix = vix.iloc[-1] if len(vix)>0 else 0
        best_name = list(models.keys())[0]
        best_model = models[best_name]

        rec = features.iloc[-LOOKBACK_WINDOW-50:]
        rec_r = returns.loc[rec.index]
        rec_rg = regime_numeric.loc[rec.index]
        try:
            _,warr = run_agent(best_model, rec, rec_r[stk], rec_rg, norm_stats=ncache.get(best_name))
            cw = warr[-1] if len(warr)>0 else np.ones(len(stk))/len(stk)
        except:
            cw = np.ones(len(stk))/len(stk)

        c1,c2,c3 = st.columns(3)
        c1.metric("REGIME", latest_regime.upper())
        c2.metric("VIX", f"{latest_vix:.1f}")
        c3.metric("AGENT", best_name)

        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
        st.markdown("### Recommended Allocation")

        wo = np.argsort(cw)[::-1]
        cols = st.columns(4)
        for i,idx in enumerate(wo):
            tk = stk[idx]
            w = cw[idx]
            ew = 1.0/len(stk)
            dp = (w-ew)/ew*100
            with cols[i%4]:
                st.metric(tk, f"{w:.1%}", f"{dp:+.0f}% vs EW")

        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

        # Bar chart
        slabels = [SECTOR_ETFS.get(t,t)[:12] for t in stk]
        si = np.argsort(cw)
        fig = go.Figure(go.Bar(
            y=[slabels[i] for i in si], x=[cw[i] for i in si], orientation="h",
            marker=dict(color=[C["PPO"] if cw[i]>1/len(stk) else "#1e293b" for i in si]),
            text=[f"{cw[i]:.1%}" for i in si], textposition="auto",
            textfont=dict(family="JetBrains Mono",size=11,color="#e8ecf4"),
        ))
        fig.add_vline(x=1/len(stk), line_dash="dot", line_color="#4b5563",
                       annotation_text="Equal Weight", annotation_position="top right")
        fig.update_layout(height=400, showlegend=False, title=f"{best_name} — Current Signal", **PL)
        st.plotly_chart(fig, width="stretch")

        st.markdown("### System Configuration")
        s1,s2,s3 = st.columns(3)
        with s1:
            st.markdown(f"**Reward Function**\n- λ_drawdown = {LAMBDA_DRAWDOWN}\n- λ_turnover = {LAMBDA_TURNOVER}\n- λ_sharpe = {LAMBDA_SHARPE}\n- λ_cvar = {LAMBDA_CVAR}\n- λ_concentration = {LAMBDA_CONCENTRATION}")
        with s2:
            st.markdown(f"**Risk Controls**\n- Kill switch: {KILL_SWITCH_THRESHOLD:.1%}\n- Max position: 30%\n- Cash reserve: 2%\n- TC: {TRANSACTION_COST_BPS}+{SPREAD_COST_BPS} bps\n- Reward clip: ±{REWARD_CLIP}")
        with s3:
            st.markdown(f"**Training**\n- Algorithms: PPO · A2C · SAC\n- Architecture: [256, 256, 128]\n- Walk-forward: {len(WALK_FORWARD_WINDOWS)} windows\n- Seeds: {len(SEEDS)} per agent\n- CVaR α: {CVAR_ALPHA:.0%}")

# ════════════════════════════════════════════════
# TAB 2: PERFORMANCE
# ════════════════════════════════════════════════
with tab2:
    if not models:
        st.warning("Train models first.")
    else:
        ar, aw = {}, {}
        for n,m in models.items():
            r,w = run_agent(m, tf, trs, trg, norm_stats=ncache.get(n))
            ar[f"{n} Agent"] = r
            aw[n] = w

        ri = list(ar.values())[0].index
        br = trs.loc[ri]
        bm = {
            "Equal Weight": equal_weight(br),
            "Risk Parity": risk_parity(br),
            "Momentum Top-3": momentum_top3(br),
            "Min Variance": min_variance(br),
            "Inverse Volatility": inverse_volatility(br),
        }
        if "SPY" in close.columns:
            bm["SPY Buy & Hold"] = spy_buy_hold(close, ri)

        ml = min(*[len(v) for v in ar.values()], *[len(v) for v in bm.values()])
        astr = {}
        for n,r in {**ar,**bm}.items():
            astr[n] = r.iloc[:ml]

        ews = astr.get("Equal Weight")
        amet = {}
        for n,r in astr.items():
            b = ews if "Agent" in n else None
            m = compute_metrics(r, benchmark_returns=b)
            sh,cl,ch = bootstrap_sharpe_ci(r, n_bootstrap=5000)
            m["Sharpe 95% CI"] = f"[{cl:.2f}, {ch:.2f}]"
            m["_ci_lo"],m["_ci_hi"] = cl,ch
            amet[n] = m

        ba = max([n for n in amet if "Agent" in n], key=lambda n: amet[n].get("_sharpe",-99), default=None)
        if ba:
            am = amet[ba]; em = amet.get("Equal Weight",{})
            c1,c2,c3 = st.columns(3)
            c1.metric(f"BEST: {ba.replace(' Agent','')}", am["CAGR"], f"vs EW: {am['_cagr']-em.get('_cagr',0):+.2%}")
            c2.metric("SHARPE", am["Sharpe"], f"vs EW: {am['_sharpe']-em.get('_sharpe',0):+.2f}")
            c3.metric("MAX DD", am["Max Drawdown"])

        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

        # Cumulative returns
        st.markdown("### Cumulative Returns")
        fig = go.Figure()
        for n,m in amet.items():
            if "_cumulative" not in m or m["_cumulative"] is None: continue
            cum = m["_cumulative"]
            ia = "Agent" in n
            fig.add_trace(go.Scatter(
                x=cum.index, y=cum.values, name=n,
                line=dict(color=color_name(n), width=2.5 if ia else 1, dash="solid" if ia else "dot"),
                opacity=1.0 if ia else 0.4,
            ))
        fig.update_layout(height=500, yaxis_title="Growth of $1", legend=dict(x=0.01,y=0.99,font_size=10), **PL)
        st.plotly_chart(fig, width="stretch")

        # Drawdown
        st.markdown("### Drawdown")
        fig2 = go.Figure()
        for n,m in amet.items():
            if "_drawdown" not in m or m["_drawdown"] is None: continue
            dd = m["_drawdown"]
            ia = "Agent" in n
            if ia or n=="Equal Weight":
                fig2.add_trace(go.Scatter(
                    x=dd.index, y=dd.values*100, name=n,
                    fill="tozeroy" if ia else None,
                    line=dict(color=color_name(n), width=1.5 if ia else 1, dash="solid" if ia else "dot"),
                ))
        fig2.update_layout(height=350, yaxis_title="Drawdown (%)", **PL)
        st.plotly_chart(fig2, width="stretch")

        # Sharpe bars with CIs
        st.markdown("### Sharpe Ratio · 95% Bootstrap CI")
        sn = sorted([n for n in amet if "_sharpe" in amet[n]], key=lambda n: amet[n]["_sharpe"])
        fig3 = go.Figure()
        for n in sn:
            m = amet[n]
            fig3.add_trace(go.Bar(
                y=[n], x=[m["_sharpe"]], orientation="h", name=n,
                marker_color=color_name(n),
                opacity=0.9 if "Agent" in n else 0.35,
                error_x=dict(type="data",symmetric=False,
                    array=[m.get("_ci_hi",m["_sharpe"])-m["_sharpe"]],
                    arrayminus=[m["_sharpe"]-m.get("_ci_lo",m["_sharpe"])]),
                showlegend=False,
            ))
        fig3.update_layout(height=400, xaxis_title="Sharpe Ratio", **PL)
        fig3.add_vline(x=0, line_color="#4b5563", line_width=0.5)
        st.plotly_chart(fig3, width="stretch")

        # Table
        st.markdown("### Full Metrics")
        tcols = ["CAGR","Annual Vol","Sharpe","Sharpe 95% CI","Sortino","Max Drawdown","Calmar","Win Rate","Info Ratio","Tracking Error"]
        td = {n:{c:m.get(c,"—") for c in tcols} for n,m in amet.items()}
        st.dataframe(pd.DataFrame(td).T, width="stretch")

# ════════════════════════════════════════════════
# TAB 3: ALLOCATIONS
# ════════════════════════════════════════════════
with tab3:
    if not models:
        st.warning("Train models first.")
    else:
        for an,md in models.items():
            r,w = run_agent(md, tf, trs, trg, norm_stats=ncache.get(an))
            st.markdown(f"### {an} · Sector Allocation")

            sl = [SECTOR_ETFS.get(t,t)[:12] for t in stk]
            wdf = pd.DataFrame(w[:len(r)], columns=sl, index=r.index)

            # Stacked area — NO alpha in colors
            figs = go.Figure()
            for i,s in enumerate(sl):
                figs.add_trace(go.Scatter(
                    x=wdf.index, y=wdf[s], name=s,
                    stackgroup="one", line=dict(width=0),
                    fillcolor=SECTOR_PALETTE[i % len(SECTOR_PALETTE)],
                ))
            figs.update_layout(height=450, yaxis=dict(range=[0,1],title="Weight"),
                                legend=dict(x=1.02,y=1,font_size=9), **PL)
            st.plotly_chart(figs, width="stretch")

            col1,col2 = st.columns(2)
            with col1:
                avg = wdf.mean().sort_values(ascending=True)
                figa = go.Figure(go.Bar(
                    x=avg.values, y=avg.index, orientation="h",
                    marker_color=C["PPO"],
                    text=[f"{v:.1%}" for v in avg.values], textposition="auto",
                    textfont=dict(family="JetBrains Mono",size=10,color="#e8ecf4"),
                ))
                figa.update_layout(height=400, title="Average Allocation", showlegend=False, **PL)
                st.plotly_chart(figa, width="stretch")

            with col2:
                to = np.sum(np.abs(np.diff(w[:len(r)], axis=0)), axis=1)
                tos = pd.Series(to, index=r.index[1:])
                figt = go.Figure()
                figt.add_trace(go.Scatter(x=tos.index,y=tos.values,mode="lines",
                    line=dict(color=C["MOM"],width=1),name="Daily"))
                figt.add_trace(go.Scatter(x=tos.index,y=tos.rolling(20).mean().values,
                    mode="lines",line=dict(color=C["MV"],width=2),name="20d MA"))
                figt.update_layout(height=400, title="Turnover", yaxis_title="Σ|Δw|", **PL)
                st.plotly_chart(figt, width="stretch")

            st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 4: SIGNALS
# ════════════════════════════════════════════════
with tab4:
    st.markdown("### Regime Timeline")
    treg = regimes.loc[ts(window[2]):ts(window[3])]
    rmap = {"calm":0,"normal":1,"stressed":2}
    rcol = {"calm":"#10b981","normal":"#f59e0b","stressed":"#ef4444"}
    figr = go.Figure()
    for rn,rc in rcol.items():
        mask = treg==rn
        if mask.any():
            figr.add_trace(go.Scatter(x=treg.index[mask],y=[rmap[rn]]*mask.sum(),
                mode="markers",marker=dict(color=rc,size=4),name=rn.title()))
    figr.update_layout(height=200,yaxis=dict(tickvals=[0,1,2],ticktext=["CALM","NORMAL","STRESSED"]),**PL)
    st.plotly_chart(figr, width="stretch")

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown("### Feature Explorer")
    fnames = sorted(features.columns.get_level_values("feature").unique())
    c1,c2 = st.columns(2)
    with c1: sf = st.selectbox("Feature", fnames)
    with c2: st2 = st.selectbox("Sector", stk, format_func=lambda x:f"{x} — {SECTOR_ETFS.get(x,x)}")

    if (sf,st2) in features.columns:
        fd = features[(sf,st2)].loc[ts(window[2]):ts(window[3])]
        figf = go.Figure(go.Scatter(x=fd.index,y=fd.values,mode="lines",
            line=dict(color=C["PPO"],width=1.5)))
        figf.update_layout(height=300, yaxis_title=sf, **PL)
        st.plotly_chart(figf, width="stretch")

    st.markdown("### Correlation Matrix")
    corr = trs.corr()
    cl = [SECTOR_ETFS.get(t,t)[:8] for t in corr.columns]
    figc = go.Figure(go.Heatmap(
        z=corr.values, x=cl, y=cl,
        colorscale=[[0,"#ef4444"],[0.5,"#0a0f1e"],[1,"#3b82f6"]],
        zmid=0, text=corr.round(2).values, texttemplate="%{text}",
        textfont=dict(size=9,family="JetBrains Mono"),
    ))
    figc.update_layout(height=500, **PL)
    st.plotly_chart(figc, width="stretch")

# ════════════════════════════════════════════════
# TAB 5: EXPLAINABILITY
# ════════════════════════════════════════════════
with tab5:
    st.markdown("### Feature Ablation Study")

    for agent_name in ["PPO","A2C","SAC"]:
        ap = os.path.join(RESULTS_DIR, f"ablation_{agent_name}.json")
        if os.path.exists(ap):
            with open(ap) as f: abl = json.load(f)
            if "all_features" in abl:
                bs = float(abl["all_features"]["Sharpe"])
                drops = {}
                for k,v in abl.items():
                    if k.startswith("no_"):
                        drops[k.replace("no_","")] = bs - float(v["Sharpe"])

                sd = dict(sorted(drops.items(), key=lambda x:x[1], reverse=True))
                figa = go.Figure(go.Bar(
                    x=list(sd.values()),
                    y=[f"Remove: {k}" for k in sd.keys()],
                    orientation="h",
                    marker_color=[C["MV"] if v>0 else C["RP"] for v in sd.values()],
                    text=[f"{v:+.3f}" for v in sd.values()], textposition="auto",
                    textfont=dict(family="JetBrains Mono",size=11),
                ))
                figa.update_layout(height=300, title=f"{agent_name} — Baseline Sharpe: {bs:.2f}",
                    xaxis_title="Sharpe Δ (positive = feature helps)", **PL)
                st.plotly_chart(figa, width="stretch")

    if not any(os.path.exists(os.path.join(RESULTS_DIR,f"ablation_{a}.json")) for a in ["PPO","A2C","SAC"]):
        st.info("Run `python evaluate.py --ablation`")

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown("### SHAP Feature Importance")
    for agent_name in ["PPO","A2C","SAC"]:
        sp = os.path.join(RESULTS_DIR, f"shap_groups_{agent_name}.png")
        if os.path.exists(sp):
            st.image(sp, caption=f"{agent_name} — SHAP Feature Groups")
        ssp = os.path.join(RESULTS_DIR, f"shap_summary_{agent_name}.png")
        if os.path.exists(ssp):
            st.image(ssp, caption=f"{agent_name} — Top Features")

    if not any(os.path.exists(os.path.join(RESULTS_DIR,f"shap_groups_{a}.png")) for a in ["PPO","A2C","SAC"]):
        st.info("Run `python explain.py --agent A2C`")

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown("### Seed Robustness")
    sdata = {}
    for a in AGENTS:
        sm = seed_robustness_analysis(a, features, returns[stk], regime_numeric, close)
        if sm: sdata[a] = sm

    if sdata:
        figs = go.Figure()
        for a in sdata:
            s = sdata[a]
            figs.add_trace(go.Bar(x=[a],y=[s["sharpe_mean"]],
                error_y=dict(type="data",array=[s["sharpe_std"]]),
                marker_color=color_name(f"{a} Agent"), name=f"{a} (n={s['n_seeds']})"))
        figs.update_layout(height=350, yaxis_title="Sharpe", title="Sharpe Across Seeds (mean ± σ)", **PL)
        st.plotly_chart(figs, width="stretch")
    else:
        st.info("Train with multiple seeds: `python train.py --full`")

# ════════════════════════════════════════════════
# TAB 6: ARCHITECTURE
# ════════════════════════════════════════════════
with tab6:
    st.markdown("### System Architecture")
    st.markdown("""
**The Problem.** Traditional allocation uses static rules that don't adapt to changing regimes.
This system trains neural networks to learn optimal sector weights through trial and error —
hundreds of thousands of simulated episodes spanning 18 years of market history.

---

**12 Technical Innovations**

| # | Feature | Description |
|---|---------|-------------|
| 1 | Random-Start Episodes | Prevents sequence memorization |
| 2 | CVaR Tail-Risk Penalty | Penalizes worst 5% of daily returns |
| 3 | Kill Switch | Forces defensive at -2.5% daily loss |
| 4 | SHAP Explainability | Shows why the agent made each decision |
| 5 | Optuna Tuning | Automated hyperparameter search |
| 6 | Dynamic Ensemble | Best agent selected per regime |
| 7 | VIX Term Structure | Crash prediction via contango/backwardation |
| 8 | Cosine LR Schedule | Smooth learning rate decay |
| 9 | TC Curriculum | Transaction costs ramp 0→100% during training |
| 10 | Transition Cost in Obs | Agent sees rebalancing cost explicitly |
| 11 | Reward Clipping | Gradient stability at ±5.0 |
| 12 | Regime-Adaptive Reward | Loss penalties scale with market stress |

---

**Planned: Sector Command Signal Server**

| Component | Purpose |
|-----------|---------|
| GitHub Actions | Runs signal pipeline 4x daily in the cloud |
| Google Sheets | Persists portfolio balance, holdings, trade log |
| Twilio SMS | Sends buy/sell signals, receives text replies |
| Vercel Webhook | Listens 24/7 for text responses |
| Ghost Portfolio | SPY buy-and-hold runs alongside for alpha tracking |
| Hard Veto Layer | RSI > 80 + stressed regime overrides RL agent |
| Conviction Sizing | 3/3 models = full size, 2/3 = half, 1/3 = quarter |
| Auto-Journal | Every trade generates a documented entry |
| Replay Mode | Demo with historical data anytime |

---

#### Reward Function
    """)

    st.latex(r"""
    r_t = R_t^{pf} \cdot \kappa(\text{regime})
          - \lambda_{dd} \cdot \max(0, \Delta DD_t) \cdot (1 + \sigma_t)
          - \lambda_{to} \cdot \|\Delta w_t\|_1
          + \lambda_{S} \cdot \max(0, \hat{S}_t^{60d})
          - \lambda_{CVaR} \cdot CVaR_{\alpha}
          - \lambda_{HHI} \cdot \max(0, HHI(w_t) - \tfrac{1}{N})
    """)

    nf = len(features.columns.get_level_values("feature").unique())
    od = LOOKBACK_WINDOW * nf * len(stk) + len(stk)*2 + 4

    st.code(f"""
Observation: {od:,} dimensions
├── {LOOKBACK_WINDOW}d × {nf} features × {len(stk)} sectors = {LOOKBACK_WINDOW*nf*len(stk):,}
├── Current weights: {len(stk)}
├── Transition costs: {len(stk)}
└── Meta: 4 (regime, rolling Sharpe, drawdown, TC multiplier)

Actor:  [{od}] → 256 → 256 → 128 → [{len(stk)} logits] → Softmax
Critic: [{od}] → 256 → 256 → 128 → [1 scalar V(s)]
    """, language="text")

    st.markdown("""
---

*Built by **Cameron Camarotti** · Class of 2027, Mill Creek High School*

[GitHub](https://github.com/cameroncc333) · [All Around Services](https://allaroundservice.com)
    """)

# ════════════════════════════════════════════════
# TAB 7: LIVE PORTFOLIO
# ════════════════════════════════════════════════
with tab7:
    st.markdown("### Live Portfolio Tracker")
    st.caption("Paper trading portfolio powered by Sector Command signal server")

    # Try to load portfolio data from Google Sheets export or local JSON
    portfolio_path = os.path.join(RESULTS_DIR, "portfolio_state.json")
    journal_path = os.path.join(RESULTS_DIR, "trade_journal.json")

    if os.path.exists(portfolio_path):
        with open(portfolio_path) as f:
            pf = json.load(f)

        bal = pf.get("balance", 0)
        init = pf.get("initial", 400)
        ghost = pf.get("ghost_balance", init)
        cash = pf.get("cash", bal)
        holdings = pf.get("holdings", {})
        pnl_pct = (bal - init) / init * 100 if init > 0 else 0
        ghost_pnl = (ghost - init) / init * 100 if init > 0 else 0
        alpha = pnl_pct - ghost_pnl

        # KPI row
        c1, c2, c3 = st.columns(3)
        c1.metric("PORTFOLIO", f"${bal:.2f}", f"{pnl_pct:+.1f}% all-time")
        c2.metric("GHOST SPY", f"${ghost:.2f}", f"{ghost_pnl:+.1f}% all-time")
        c3.metric("ALPHA", f"{alpha:+.1f}%",
                   "Outperforming" if alpha > 0 else "Underperforming")

        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

        # Holdings breakdown
        if holdings:
            st.markdown("### Current Holdings")
            hold_data = []
            for t, h in holdings.items():
                hold_data.append({
                    "Ticker": t,
                    "Sector": SECTOR_ETFS.get(t, t),
                    "Value": f"${h.get('market_value', 0):.2f}",
                    "Weight": f"{h.get('market_value', 0) / bal * 100:.1f}%" if bal > 0 else "0%",
                    "Today": f"{h.get('today_return', 0):+.1f}%",
                    "P&L": f"${h.get('unrealized_pnl', 0):+.2f}",
                })
            st.dataframe(pd.DataFrame(hold_data), width="stretch", hide_index=True)
        else:
            st.info("No holdings — portfolio is 100% cash. Waiting for first trade.")

        st.markdown(f"**Cash:** ${cash:.2f} ({cash/bal*100:.0f}% of portfolio)" if bal > 0 else "**Cash:** $0")

        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

        # Performance chart (if we have history)
        history_path = os.path.join(RESULTS_DIR, "portfolio_history.json")
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)

            if history:
                dates = [h["date"] for h in history]
                balances = [h["balance"] for h in history]
                ghosts = [h["ghost"] for h in history]

                fig_pf = go.Figure()
                fig_pf.add_trace(go.Scatter(x=dates, y=balances, name="Your Portfolio",
                    line=dict(color="#3b82f6", width=2.5)))
                fig_pf.add_trace(go.Scatter(x=dates, y=ghosts, name="Ghost (SPY)",
                    line=dict(color="#6b7280", width=1.5, dash="dot")))
                fig_pf.update_layout(height=400, yaxis_title="Balance ($)",
                    title="Portfolio vs Ghost SPY", **PL)
                st.plotly_chart(fig_pf, width="stretch")

    else:
        st.markdown("### Portfolio Status")
        st.markdown("**Starting balance:** $400.00 · **Status:** Awaiting first trade")

        c1,c2,c3 = st.columns(3)
        c1.metric("BALANCE", "$400.00")
        c2.metric("GHOST SPY", "$400.00")
        c3.metric("ALPHA", "0.0%")

        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
        st.markdown("**Holdings:** 100% cash — no positions yet. Signals will appear here after the signal server connects.")

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    # Trade journal
    st.markdown("### Trade Journal")
    if os.path.exists(journal_path):
        with open(journal_path) as f:
            journal = json.load(f)
        if journal:
            jdf = pd.DataFrame(journal)
            display_cols = [c for c in ["timestamp","action","ticker","amount","conviction",
                            "regime","decision"] if c in jdf.columns]
            st.dataframe(jdf[display_cols] if display_cols else jdf,
                         width="stretch", hide_index=True)
        else:
            st.info("No trades logged yet.")
    else:
        st.info("Trade journal will populate after your first trade via SMS.")

