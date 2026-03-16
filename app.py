# app.py
# ToxiScan — AI Drug Toxicity Prediction App
# Patient-safety focused — tuned to catch more toxic compounds.
#
# Run with:
#   python -m streamlit run app.py
#
# Tabs:
#   1. Single Molecule  — type a drug name with autocomplete to predict toxicity
#   2. Batch Screener   — upload a CSV and screen multiple drugs at once
#   3. AI Explainer     — ask Groq AI to explain any prediction in plain English

import os
import sys
import pickle
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pubchempy as pcp
from groq import Groq
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from streamlit_searchbox import st_searchbox

# add src/ to path so we can import our helper files
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from chembl_lookup import lookup_compound

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "ToxiScan",
    page_icon  = "🔬",
    layout     = "wide"
)

# ── api key ───────────────────────────────────────────────────────────────────
api_key = os.environ.get("GROQ_API_KEY")

TARGET_COLS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]


# ── load all 12 models once at startup ───────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for target in TARGET_COLS:
        path = f"models/{target}.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[target] = pickle.load(f)
    return models

models = load_models()


# ── helper functions ──────────────────────────────────────────────────────────

def search_pubchem(query):
    """
    calls PubChem autocomplete API to suggest drug names as user types.
    covers millions of compounds — no hardcoded list needed.
    """
    if not query or len(query) < 2:
        return []
    try:
        url      = (f"https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete"
                    f"/compound/{query}/JSON?limit=10")
        response = requests.get(url, timeout=3)
        data     = response.json()
        return data.get("dictionary_terms", {}).get("compound", [])
    except:
        return []


def name_to_smiles(drug_name):
    """looks up a drug name on PubChem and returns its SMILES string"""
    try:
        compounds = pcp.get_compounds(drug_name, "name")
        if compounds:
            return compounds[0].isomeric_smiles
        return None
    except:
        return None


def resolve_input(user_input):
    """
    detects if input is a SMILES string or drug name.
    SMILES contain special characters like = ( ) [ ] @ / \
    drug names are looked up on PubChem automatically.
    """
    if not user_input:
        return None
    smiles_chars = ["=", "#", "(", ")", "[", "]", "@", "/", "\\"]
    if any(c in user_input for c in smiles_chars):
        return user_input
    else:
        return name_to_smiles(user_input)


def get_features(smiles):
    """
    compute 136 molecular features from a SMILES string.
    - 8 standard descriptors (drug-likeness properties)
    - 128 Morgan fingerprint bits (structural patterns)
    Morgan fingerprints encode exact structural patterns — far more
    informative than descriptors alone for toxicity prediction.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = {
        "MolWeight"       : round(Descriptors.MolWt(mol), 3),
        "LogP"            : round(Descriptors.MolLogP(mol), 3),
        "TPSA"            : round(Descriptors.TPSA(mol), 3),
        "NumHDonors"      : Descriptors.NumHDonors(mol),
        "NumHAcceptors"   : Descriptors.NumHAcceptors(mol),
        "NumRotBonds"     : Descriptors.NumRotatableBonds(mol),
        "NumRings"        : Descriptors.RingCount(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
    }

    # 128 Morgan fingerprint bits — encodes structural patterns
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=128)
    for i, bit in enumerate(fp):
        features[f"fp_{i}"] = int(bit)

    return features


def predict(smiles):
    """
    run all 12 models on a SMILES string.
    models store their own safety threshold (0.1 for patient safety).
    returns feature dict and predictions dict (0-100% probability).
    """
    feat = get_features(smiles)
    if feat is None:
        return None, None

    preds = {}
    for t, model_data in models.items():
        if isinstance(model_data, dict):
            # new format — model stored with threshold and feature list
            m         = model_data["model"]
            feat_cols = model_data["features"]
            vals      = [feat.get(c, 0) for c in feat_cols]
        else:
            # old format fallback
            m    = model_data
            vals = list(feat.values())[:8]

        prob     = float(m.predict_proba([vals])[0][1])
        preds[t] = round(prob * 100, 1)

    return feat, preds


def risk_label(score):
    """
    patient-safety focused risk labels — 4 levels.
    we prefer to over-flag than under-flag.
    """
    if score < 25:  return "🟢 LOW RISK",      "success"
    if score < 50:  return "🟡 MEDIUM RISK",   "warning"
    if score < 75:  return "🔴 HIGH RISK",     "error"
    return               "🚨 VERY HIGH RISK", "error"


def safety_context(overall, preds):
    """
    returns a patient safety message based on prediction results.
    always shown so users understand the scope of the model.
    """
    high_targets = [t for t, v in preds.items() if v >= 50]

    if overall < 25 and len(high_targets) == 0:
        return (
            "ℹ️ This compound does not strongly activate the 12 Tox21 "
            "cellular pathways. Note: some dangerous compounds such as opioids "
            "and benzene act through mechanisms outside Tox21 scope — receptor "
            "binding, addiction, and metabolic activation are not captured here. "
            "Always combine computational predictions with laboratory testing."
        )
    elif len(high_targets) > 0:
        return (
            f"⚠️ High risk detected on: {', '.join(high_targets)}. "
            f"These pathways are linked to endocrine disruption, DNA damage, "
            f"or mitochondrial stress. Further laboratory testing is strongly "
            f"recommended before any clinical use."
        )
    else:
        return (
            "⚠️ Medium risk detected. This compound activates some Tox21 "
            "pathways at moderate levels. Proceed with caution and conduct "
            "in vitro validation before further development."
        )


def ask_groq(prompt):
    """send a prompt to Groq and return the text response"""
    client   = Groq(api_key)
    response = client.chat.completions.create(
        model    = "llama-3.3-70b-versatile",
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# ── header ────────────────────────────────────────────────────────────────────
st.title("🔬 ToxiScan")
st.caption("AI-powered drug toxicity prediction — CodeCure Biohackathon 2025")
st.caption("⚕️ Patient-safety mode: tuned to catch more toxic compounds")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "🔬 Single Molecule",
    "📦 Batch Screener",
    "🤖 AI Explainer"
])


# =============================================================================
# TAB 1 — Single Molecule
# =============================================================================
with tab1:
    st.subheader("Predict toxicity for one compound")

    user_input = st_searchbox(
        search_pubchem,
        placeholder = "start typing a drug name e.g. Aspirin, Morphine...",
        label       = "Enter drug name or SMILES string",
        key         = "tab1_search"
    )

    if not user_input:
        user_input = "Aspirin"

    st.caption("examples — Aspirin · Ibuprofen · Caffeine · Paracetamol · Morphine · Warfarin")

    with st.spinner(f"looking up {user_input}..."):
        smiles = resolve_input(user_input)

    if not smiles:
        st.error(f"could not find '{user_input}' — try a different name or paste SMILES directly")
    else:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            st.error("invalid SMILES — please try again")
        else:
            st.caption(f"SMILES: `{smiles}`")

            feat, preds = predict(smiles)
            overall     = round(np.mean(list(preds.values())), 1)
            label, lvl  = risk_label(overall)

            # show only 8 main descriptors in table — hide fingerprint bits
            display_feat = {k: v for k, v in feat.items()
                            if not k.startswith("fp_")}

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**Molecule structure**")
                img = Draw.MolToImage(mol, size=(260, 200))
                st.image(img, use_container_width=True)

                st.markdown("**Molecular properties**")
                prop_df = pd.DataFrame(display_feat.items(),
                                       columns=["Property", "Value"])
                st.dataframe(prop_df, hide_index=True,
                             use_container_width=True)

                st.markdown("**Lipinski Rule of Five**")
                st.caption("a drug must pass all 4 rules to be orally bioavailable")
                rules = {
                    "Mol. weight < 500 Da" : feat["MolWeight"]    < 500,
                    "LogP < 5"             : feat["LogP"]          < 5,
                    "H-donors ≤ 5"         : feat["NumHDonors"]    <= 5,
                    "H-acceptors ≤ 10"     : feat["NumHAcceptors"] <= 10,
                }
                passed = sum(rules.values())
                for rule, ok in rules.items():
                    if ok: st.success(f"✅ {rule}")
                    else:  st.error(f"❌ {rule}")
                st.caption(f"{passed}/4 rules passed")

                st.markdown("**ChEMBL Database**")
                try:
                    with st.spinner("checking ChEMBL..."):
                        chembl = lookup_compound(smiles)
                    if chembl:
                        st.info(
                            f"Found: **{chembl['name']}** "
                            f"({chembl['chembl_id']})\n\n"
                            f"Known assays: {chembl['num_assays']}  |  "
                            f"Max potency: {chembl['max_potency']}"
                        )
                    else:
                        st.caption("not found in ChEMBL database")
                except:
                    st.caption("ChEMBL unavailable — skipped")

            with col2:
                getattr(st, lvl)(
                    f"Overall toxicity risk: **{overall}%** — {label}"
                )
                st.progress(int(overall))

                # patient safety context — always shown
                st.info(safety_context(overall, preds))
                st.divider()

                st.markdown("**Risk score per toxicity target**")
                fig, ax = plt.subplots(figsize=(7, 4))
                bar_colors = [
                    "#e74c3c" if v >= 60 else
                    "#e67e22" if v >= 30 else
                    "#2ecc71"
                    for v in preds.values()
                ]
                ax.barh(list(preds.keys()), list(preds.values()),
                        color=bar_colors, edgecolor="white")
                ax.axvline(50, color="black", linestyle="--",
                           linewidth=0.8, alpha=0.4, label="50% threshold")
                ax.set_xlabel("Toxicity probability (%)")
                ax.set_xlim(0, 100)
                ax.legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                top3 = sorted(preds.items(),
                              key=lambda x: x[1], reverse=True)[:3]
                st.markdown("**Top 3 targets most at risk**")
                for target, score in top3:
                    lbl, _ = risk_label(score)
                    st.write(f"- **{target}** — {score}%  {lbl}")
                st.divider()
st.divider()
st.markdown("**🤖 AI Safety Summary**")
st.caption("based on all 12 toxicity target scores for this compound")

with st.spinner("AI is analyzing all 12 targets..."):
    try:
        summary_prompt = f"""You are a toxicology assistant helping 2nd year pharmacy students
interpret machine learning toxicity predictions.

IMPORTANT RULES:
- These are PREDICTED probabilities from a computational model — NOT confirmed lab results
- Always use cautious language: "may", "could", "potential risk of", "suggests possible"
- Never state toxicity as a confirmed fact
- AI can make mistakes — always recommend laboratory validation

Compound analyzed: {user_input}
SMILES: {smiles}
Overall risk score: {overall}% — {label}

All 12 toxicity target scores:
{chr(10).join([f"- {t}: {v}%" for t, v in preds.items()])}

Molecular properties:
{chr(10).join([f"- {k}: {v}" for k, v in display_feat.items()])}

Please provide a structured safety summary in these 4 sections:

**Overall Assessment**
1 sentence — what the overall pattern of scores suggests about this compound's
potential risk profile. Use cautious language.

**Potential Concerns** (only targets above 15%)
For each target above 15%, write 1 sentence:
- What this pathway does in the body
- What risk this score POTENTIALLY suggests for this compound
- Use "may", "could", "potential risk of" — never state as confirmed

**Targets Showing Low Predicted Risk** (below 15%)
1 sentence listing these targets and noting they show low computational
signal — but lab confirmation is still recommended.

**Recommended Next Steps**
2 sentences — what specific laboratory tests should be done to validate
these computational predictions before this compound could be considered
for further development.

**Disclaimer**
End with: "These predictions are generated by a machine learning model
and an AI language model. Both can make mistakes. These results must be
validated by laboratory testing before any clinical conclusion can be drawn."

Be specific to this compound and its scores.
Use simple language a 2nd year pharmacy student can understand."""

        summary = ask_groq(summary_prompt)
        st.write(summary)
        st.caption(
            "ℹ️ Generated by Groq AI (Llama 3.3) based on XGBoost model predictions. "
            "AI agents can make mistakes — always validate with laboratory testing."
        )

    except Exception as e:
        st.error(f"API error: {e}")


# =============================================================================
# TAB 2 — Batch Screener
# =============================================================================
with tab2:
    st.subheader("Screen multiple drugs at once")
    st.caption("upload a CSV with two columns: name and smiles")

    sample = pd.DataFrame({
        "name"  : ["Aspirin", "Ibuprofen", "Caffeine", "Morphine", "Warfarin"],
        "smiles": [
            "CC(=O)Oc1ccccc1C(=O)O",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
            "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5",
            "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2c(=O)o1"
        ]
    })
    with st.expander("see required CSV format"):
        st.dataframe(sample, hide_index=True)
        st.download_button(
            label     = "download sample CSV",
            data      = sample.to_csv(index=False),
            file_name = "sample_drugs.csv",
            mime      = "text/csv"
        )

    uploaded = st.file_uploader("upload your drugs CSV", type="csv")

    if uploaded:
        input_df = pd.read_csv(uploaded)

        if "name" not in input_df.columns or "smiles" not in input_df.columns:
            st.error("CSV must have columns named 'name' and 'smiles'")
        else:
            st.info(f"loaded {len(input_df)} molecules — screening now...")

            results  = []
            progress = st.progress(0)

            for i, row in input_df.iterrows():
                feat, preds = predict(str(row["smiles"]))
                if preds is None:
                    results.append({
                        "Drug"           : row["name"],
                        "Overall Risk %" : "N/A",
                        "Risk Level"     : "❓ Invalid SMILES"
                    })
                else:
                    overall = round(np.mean(list(preds.values())), 1)
                    lbl, _  = risk_label(overall)
                    results.append({
                        "Drug"           : row["name"],
                        "Overall Risk %" : overall,
                        "Risk Level"     : lbl
                    })
                progress.progress(int((i + 1) / len(input_df) * 100))

            results_df = pd.DataFrame(results)

            valid = results_df[results_df["Overall Risk %"] != "N/A"].copy()
            valid = valid.sort_values("Overall Risk %")
            valid.insert(0, "Safety Rank", range(1, len(valid) + 1))

            st.success(f"done! screened {len(valid)} compounds")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🟢 Low risk",
                      len(valid[valid["Risk Level"].str.contains("LOW",       na=False)]))
            c2.metric("🟡 Medium risk",
                      len(valid[valid["Risk Level"].str.contains("MEDIUM",    na=False)]))
            c3.metric("🔴 High risk",
                      len(valid[valid["Risk Level"].str.contains("HIGH",      na=False)]))
            c4.metric("🚨 Very high",
                      len(valid[valid["Risk Level"].str.contains("VERY HIGH", na=False)]))

            st.dataframe(valid, hide_index=True, use_container_width=True)

            st.download_button(
                label     = "⬇️ download full safety report (CSV)",
                data      = valid.to_csv(index=False),
                file_name = "toxiscan_safety_report.csv",
                mime      = "text/csv"
            )


# =============================================================================
# TAB 3 — AI Explainer (Groq)
# =============================================================================
with tab3:
    st.subheader("Ask the AI to explain any prediction")
    st.warning(
    "⚠️ **Disclaimer:** The AI explanations provided here are generated by a "
    "large language model and may contain errors or inaccuracies. All predictions "
    "are computational estimates based on molecular structure — they are not "
    "confirmed laboratory results. AI agents can and do make mistakes. "
    "Never use these results for clinical decisions without validation by a "
    "qualified toxicologist and appropriate laboratory testing."
)
    st.caption("type a drug name — Groq AI explains toxicity in plain English")

    ai_input = st_searchbox(
        search_pubchem,
        placeholder = "start typing a drug name e.g. Morphine, Warfarin...",
        label       = "enter drug name or SMILES",
        key         = "tab3_search"
    )

    if not ai_input:
        ai_input = "Aspirin"

    st.caption("examples — Morphine · Warfarin · Nicotine · Diazepam · Metformin")

    if st.button("analyze and explain ✨", type="primary"):
        with st.spinner(f"looking up {ai_input}..."):
            smiles_ai = resolve_input(ai_input)

        if not smiles_ai:
            st.error(f"could not find '{ai_input}' — try a different name")
        else:
            feat, preds = predict(smiles_ai)
            if preds is None:
                st.error("invalid molecule — please try again")
            else:
                overall  = round(np.mean(list(preds.values())), 1)
                lbl, _   = risk_label(overall)

                mol_name = ai_input if not any(
                    c in ai_input for c in ["=", "(", ")", "[", "]"]
                ) else "this compound"

                # only pass 8 main descriptors to AI — not fingerprint bits
                display_feat = {k: v for k, v in feat.items()
                                if not k.startswith("fp_")}

                try:
                    with st.spinner("checking ChEMBL..."):
                        chembl     = lookup_compound(smiles_ai)
                    chembl_txt = (
                        f"known as {chembl['name']} in ChEMBL, "
                        f"{chembl['num_assays']} known assays"
                        if chembl else "not found in ChEMBL"
                    )
                except:
                    chembl_txt = "ChEMBL unavailable"

                prompt = f"""You are a toxicology assistant helping 2nd year pharmacy students
understand drug safety predictions made by a machine learning model.
This model is tuned for patient safety — it uses a low decision threshold
to catch more potentially toxic compounds.

Molecule analyzed: {mol_name}
SMILES: {smiles_ai}

Molecular properties computed by RDKit:
{chr(10).join([f"- {k}: {v}" for k, v in display_feat.items()])}

Toxicity risk scores predicted by our XGBoost model (% probability):
{chr(10).join([f"- {k}: {v}%" for k, v in preds.items()])}

Overall risk score: {overall}% — {lbl}
ChEMBL database reference: {chembl_txt}

Please structure your response in exactly these 5 sections:

**What is {mol_name}?**
In 2 sentences — what this drug is, what it is used for medically, and its general reputation for safety or danger.

**Toxicity prediction results**
In 2 sentences — which targets scored highest, what biological pathway that affects, and whether this aligns with what is known about this drug.

**Key molecular property driving risk**
In 1 sentence — which of the 8 molecular properties most likely explains the risk score and why.

**Clinical implications**
In 2 sentences — what a doctor or pharmacist should know about this compound based on these results. Include any known real-world safety concerns.

**Recommendation**
In 1 sentence — should this compound be investigated further, avoided, or used with caution? Be specific.

Use simple language a 2nd year pharmacy student can understand. Avoid complex jargon."""

                with st.spinner("AI is analyzing the molecule..."):
                    try:
                        explanation = ask_groq(prompt)
                        st.caption(f"SMILES resolved: `{smiles_ai}`")
                        st.success("AI explanation")
                        st.write(explanation)
                        st.caption(
    "ℹ️ This explanation was generated by Groq AI (Llama 3.3). "
    "AI models can make mistakes — treat this as a starting point "
    "for investigation, not a final conclusion."
)
                        st.info(safety_context(overall, preds))

                        st.session_state["context"] = (
                            f"molecule: {mol_name}, "
                            f"properties: {display_feat}, "
                            f"predictions: {preds}, "
                            f"overall risk: {overall}%"
                        )

                    except Exception as e:
                        st.error(f"API error: {e}")

    # follow-up question box — only shows after first explanation
    if "context" in st.session_state:
        st.divider()
        followup = st.text_input(
            "ask a follow-up question",
            placeholder = "e.g. why does high LogP increase toxicity risk?"
        )
        if followup:
            with st.spinner("thinking..."):
                try:
                    followup_prompt = (
                        f"context about the molecule already analyzed:\n"
                        f"{st.session_state['context']}\n\n"
                        f"student question: {followup}\n\n"
                        f"answer in 3-4 sentences. include what the drug is "
                        f"used for if relevant, and explain the science simply. "
                        f"a 2nd year student should fully understand."
                    )
                    answer = ask_groq(followup_prompt)
                    st.info(answer)
                except Exception as e:
                    st.error(f"API error: {e}")
                st.divider()
st.caption(
    "🔬 ToxiScan — CodeCure Biohackathon 2025  |  "
    "Predictions are computational estimates only  |  "
    "Not for clinical use  |  "
    "AI explanations may contain errors — always validate with laboratory testing"
)