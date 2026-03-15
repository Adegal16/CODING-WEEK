"""
app.py - Heart Failure Risk Prediction
Interface Streamlit - Phase finale 
Modèle : models/best_model.pkl + models/scaler.pkl
SHAP   : models/X_train_scaled.csv + src/evaluate_model.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import pandas as pd
import random
import joblib
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
import base64

# PDF Generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ─────────────────────────────────────────────
# IMPORT SHAP DEPUIS src/evaluate_model.py
# ─────────────────────────────────────────────
SHAP_AVAILABLE = False
try:
    from src.evaluate_model import generate_shap_summary, generate_shap_individual
    SHAP_AVAILABLE = True
except ImportError:
    pass

# ─────────────────────────────────────────────
# CHARGEMENT DU MODÈLE, SCALER ET DONNÉES SHAP
# Fichiers générés par : python src/train_model.py
#   → models/best_model.pkl
#   → models/scaler.pkl
#   → models/X_train_scaled.csv
# ─────────────────────────────────────────────
MODEL_AVAILABLE = False
model           = None
scaler          = None
X_train_scaled  = None

try:
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    model          = joblib.load(os.path.join(BASE, 'models', 'best_model.pkl'))
    scaler         = joblib.load(os.path.join(BASE, 'models', 'scaler.pkl'))
    X_train_scaled = pd.read_csv(os.path.join(BASE, 'models', 'X_train_scaled.csv'))

    MODEL_AVAILABLE = True
except Exception:
    pass  # → mode placeholder automatique

# ─────────────────────────────────────────────
# FONCTION DE GÉNÉRATION PDF
# ─────────────────────────────────────────────
def generate_pdf_report(patient_data: dict, result: dict, shap_summary_fig=None, shap_individual_fig=None) -> BytesIO:
    """
    Génère un rapport PDF contenant les données patient et la prédiction de risque.
    
    Parameters:
    -----------
    patient_data : dict - Données cliniques du patient
    result : dict - Résultat de prédiction (probabilité, niveau de risque)
    shap_summary_fig : matplotlib.figure.Figure - Graphique de synthèse SHAP (optionnel)
    shap_individual_fig : matplotlib.figure.Figure - Graphique SHAP pour le patient (optionnel)
    
    Returns:
    --------
    BytesIO - Buffer PDF prêt pour téléchargement
    """
    if not PDF_AVAILABLE:
        st.error("La génération de PDF n'est pas disponible. Installez reportlab avec: pip install reportlab")
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f6feb'),
        spaceAfter=12,
        alignment=1  # Center
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#000000'),
        spaceAfter=10,
        spaceBefore=10,
        # no border or background to keep title simple
        #borderPadding=8,
        #backColor=colors.HexColor('#161b22'),
    )
    
    # En-tête du rapport
    story.append(Paragraph("CardioAI - Rapport de Prediction du Risque Cardiaque", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"<b>Date du rapport :</b> {datetime.now().strftime('%d/%m/%Y a %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Section 1 : Données du patient
    story.append(Paragraph("1. Informations Cliniques du Patient", heading_style))
    patient_table_data = [
        ["Parametre", "Valeur"],
        ["Age", f"{patient_data['age']:.0f} ans"],
        ["Sexe", "Homme" if patient_data['sex'] == 1 else "Femme"],
        ["Fraction d'ejection", f"{patient_data['ejection_fraction']:.1f}%"],
        ["Creatinine srique", f"{patient_data['serum_creatinine']:.2f} mg/dL"],
        ["Sodium srique", f"{patient_data['serum_sodium']:.0f} mEq/L"],
        ["CPK", f"{patient_data['creatinine_phosphokinase']:.0f} mcg/L"],
        ["Plaquettes", f"{patient_data['platelets']:,.0f} /mL"],
        ["Suivi", f"{patient_data['time']:.0f} jours"],
    ]
    patient_table = Table(patient_table_data)
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d3d3d3')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor('#f0f0f0')]),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Section 2 : Facteurs de risque
    story.append(Paragraph("2. Facteurs de Risque", heading_style))
    risk_factors = [
        ["Anemie", "OUI" if patient_data['anaemia'] else "NON"],
        ["Diabete", "OUI" if patient_data['diabetes'] else "NON"],
        ["Hypertension", "OUI" if patient_data['high_blood_pressure'] else "NON"],
        ["Tabagisme", "OUI" if patient_data['smoking'] else "NON"],
    ]
    risk_table = Table(risk_factors, colWidths=[2.5*inch, 2.5*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.whitesmoke, colors.HexColor('#f0f0f0')]),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Section 3 : Résultats de prédiction
    story.append(Paragraph("3. Résultats de la Prédiction", heading_style))
    
    risk_color_map = {
        "Élevé": "#f85149",
        "Modéré": "#d29922",
        "Faible": "#3fb950"
    }
    risk_color = risk_color_map.get(result['risk_level'], "#ffffff")
    
    result_text = f"""
    <b>Niveau de Risque :</b> <font color="{risk_color}"><b>{result['risk_emoji']} {result['risk_level']}</b></font><br/>
    <b>Probabilité de décès :</b> <font color="{risk_color}"><b>{result['probability']*100:.1f}%</b></font><br/>
    <b>Modèle :</b> {'Modèle ML réel' if result['is_real'] else 'Simulation (Placeholder)'}
    """
    story.append(Paragraph(result_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Recommandation clinique
    if result['risk_level'] == "Élevé":
        recommendation = "[RISQUE ELEVE] Une consultation cardiologique urgente est recommandée."
    elif result['risk_level'] == "Modéré":
        recommendation = "[RISQUE MODERE] Une surveillance renforcée et un bilan complémentaire sont conseillés."
    else:
        recommendation = "[RISQUE FAIBLE] Le maintien d'un suivi médical régulier est recommandé."
    
    story.append(Paragraph(f"<b>Recommandation Clinique :</b> {recommendation}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Section 4 : Graphiques SHAP (s'ils sont disponibles)
    if shap_summary_fig or shap_individual_fig:
        story.append(PageBreak())
        story.append(Paragraph("4. Analyse SHAP - Explications de la Prediction", heading_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Graphique de synthèse SHAP
        if shap_summary_fig:
            try:
                summary_buffer = BytesIO()
                shap_summary_fig.savefig(summary_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                summary_buffer.seek(0)
                story.append(Paragraph("<b>Graphique de Synthese SHAP (Vue Globale)</b>", styles['Heading3']))
                img_summary = Image(summary_buffer, width=6*inch, height=3*inch)
                story.append(img_summary)
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                story.append(Paragraph(f"<i>Erreur lors de l'inclusion du graphique SHAP: {e}</i>", styles['Normal']))
        
        # Graphique individuel SHAP
        if shap_individual_fig:
            try:
                individual_buffer = BytesIO()
                shap_individual_fig.savefig(individual_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                individual_buffer.seek(0)
                story.append(Paragraph("<b>Graphique SHAP pour ce Patient (Waterfall)</b>", styles['Heading3']))
                img_individual = Image(individual_buffer, width=6*inch, height=3*inch)
                story.append(img_individual)
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                story.append(Paragraph(f"<i>Erreur lors de l'inclusion du graphique SHAP individuel: {e}</i>", styles['Normal']))
    
    # Footer
    story.append(Spacer(1, 0.5*inch))
    footer_text = "CardioAI • Centrale Casablanca • Coding Week Mars 2026"
    story.append(Paragraph(f"<i style='font-size: 8pt; color: #8b949e;'>{footer_text}</i>", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# ─────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CardioAI — Heart Failure Risk",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #0f1117; }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem; color: #ffffff;
        line-height: 1.2; margin-bottom: 0.2rem;
    }
    .hero-sub { color: #8b949e; font-size: 1rem; font-weight: 300; margin-bottom: 2rem; }
    .card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
    }
    .metrics-card {
        background: linear-gradient(135deg, #0d2847, #1a5a8c);
        border: 1px solid #1f6feb; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
    }
    .card-title {
        color: #58a6ff; font-size: 0.75rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.8rem;
    }
    .metrics-card .card-title {
        font-size: 0.95rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #3d1a1a, #2d1515);
        border: 1px solid #f85149; border-radius: 12px;
        padding: 1.5rem; text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #1a3d2b, #152d1f);
        border: 1px solid #3fb950; border-radius: 12px;
        padding: 1.5rem; text-align: center;
    }
    .risk-medium {
        background: linear-gradient(135deg, #3d3417, #2d2710);
        border: 1px solid #d29922; border-radius: 12px;
        padding: 1.5rem; text-align: center;
    }
    .risk-score { font-family: 'DM Serif Display', serif; font-size: 3.5rem; line-height: 1; }
    .metric-value { font-family: 'DM Serif Display', serif; font-size: 1.8rem; color: #ffffff; }
    .metric-label { color: #8b949e; font-size: 0.8rem; }
    .patient-header {
        display: block; padding: 1rem 1.2rem; border-radius: 12px;
        font-size: 1.2rem; font-weight: 600; background: linear-gradient(135deg, #d97706, #f97316);
        color: #ffffff; border: 2px solid #ea580c; margin-bottom: 2rem;
        margin-top: -0.5rem; text-align: center; letter-spacing: 0.05em;
    }
    .badge {
        display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px;
        font-size: 0.75rem; font-weight: 500; background: #1f3a5f;
        color: #58a6ff; border: 1px solid #1f6feb; margin-bottom: 1rem;
    }
    .placeholder-note {
        background: #1c2128; border-left: 3px solid #d29922;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        color: #d29922; font-size: 0.85rem; margin-bottom: 1.5rem;
    }
    .success-note {
        background: #1a3d2b; border-left: 3px solid #3fb950;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        color: #3fb950; font-size: 0.85rem; margin-bottom: 1.5rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white; border: none; border-radius: 8px; padding: 0.6rem 2rem;
        font-family: 'DM Sans', sans-serif; font-weight: 500;
        font-size: 1rem; width: 100%; transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #388bfd, #58a6ff);
        transform: translateY(-1px);
    }
    /* Style des onglets */
    [data-baseweb="tab-list"] {
        gap: 8px;
    }
    [data-baseweb="tab"] {
        padding: 12px 24px !important;
        font-size: 1.2rem !important;
        font-weight: 500 !important;
    }
    hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ORDRE DES COLONNES — identique à train_model.py
# ─────────────────────────────────────────────
FEATURE_ORDER = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
]


# ─────────────────────────────────────────────
# FONCTIONS DE PRÉDICTION
# ─────────────────────────────────────────────
def predict_real(patient_data: dict) -> dict:
    """
    Prédiction avec le vrai modèle.
    Pipeline identique à train_model.py :
      1. Mise en ordre des colonnes
      2. Application du scaler
      3. predict_proba → probabilité de décès
    """
    features_df     = pd.DataFrame([patient_data])[FEATURE_ORDER]
    features_scaled = scaler.transform(features_df)
    proba           = model.predict_proba(features_scaled)[0][1]
    proba           = round(float(proba), 3)

    if proba >= 0.6:
        risk_level, risk_class = "Élevé",  "risk-high"
        risk_color, risk_emoji = "#f85149", "🔴"
    elif proba >= 0.35:
        risk_level, risk_class = "Modéré",  "risk-medium"
        risk_color, risk_emoji = "#d29922", "🟡"
    else:
        risk_level, risk_class = "Faible",  "risk-low"
        risk_color, risk_emoji = "#3fb950", "🟢"

    return {
        "probability": proba, "risk_level": risk_level,
        "risk_class":  risk_class, "risk_color": risk_color,
        "risk_emoji":  risk_emoji, "is_real": True,
    }


def predict_placeholder(patient_data: dict) -> dict:
    """
    Prédiction aléatoire PLACEHOLDER.
    Active automatiquement si models/best_model.pkl est absent.
    """
    random.seed(int(sum(patient_data.values())))
    proba = round(random.uniform(0.05, 0.95), 3)

    if proba >= 0.6:
        risk_level, risk_class = "Élevé",  "risk-high"
        risk_color, risk_emoji = "#f85149", "🔴"
    elif proba >= 0.35:
        risk_level, risk_class = "Modéré",  "risk-medium"
        risk_color, risk_emoji = "#d29922", "🟡"
    else:
        risk_level, risk_class = "Faible",  "risk-low"
        risk_color, risk_emoji = "#3fb950", "🟢"

    shap_values = [round(random.uniform(-0.3, 0.3), 3) for _ in FEATURE_ORDER]

    return {
        "probability":   proba, "risk_level": risk_level,
        "risk_class":    risk_class, "risk_color": risk_color,
        "risk_emoji":    risk_emoji, "shap_features": FEATURE_ORDER,
        "shap_values":   shap_values, "is_real": False,
    }


# ─────────────────────────────────────────────
# SIDEBAR — SAISIE PATIENT
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="patient-header">Données Patient</div>', unsafe_allow_html=True)
    st.markdown("### Informations cliniques")

    age                      = st.number_input("Âge (ans)", value=60, step=1)
    ejection_fraction        = st.number_input("Fraction d'éjection (%)", 10, 80, 38)
    serum_creatinine         = st.number_input("Créatinine sérique (mg/dL)", 0.1, 10.0, 1.2, step=0.1)
    serum_sodium             = st.number_input("Sodium sérique (mEq/L)", 100, 150, 137)
    creatinine_phosphokinase = st.number_input("CPK (mcg/L)", 10, 8000, 250)
    platelets                = st.number_input("Plaquettes (kiloplatelets/mL)", 10000, 900000, 262000, step=1000)

    st.markdown("---")
    st.markdown("### Facteurs de risque")
    anaemia      = st.toggle("Anémie",       value=False)
    diabetes     = st.toggle("Diabète",      value=False)
    hypertension = st.toggle("Hypertension", value=False)
    smoking      = st.toggle("Tabagisme",    value=False)
    sex          = st.radio("Sexe", ["Homme", "Femme"], horizontal=True)
    time         = st.number_input("Période de suivi (jours)", value=100, step=1)

    st.markdown("---")
    predict_btn = st.button("Analyser le Risque")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🫀 CardioAI</h1>', unsafe_allow_html=True)
st.markdown(
    "<p class='hero-sub'>Outil d'aide à la décision clinique — "
    "Prédiction du risque d'insuffisance cardiaque</p>",
    unsafe_allow_html=True
)

# Bannière dynamique — affiche uniquement en mode démonstration
if not MODEL_AVAILABLE:
    st.markdown("""
    <div class="placeholder-note">
         <strong>Mode Démonstration</strong> — Prédictions aléatoires (placeholder).<br>
        Lancez <code>python src/train_model.py</code> pour activer le vrai modèle.
    </div>
    """, unsafe_allow_html=True)

# ─── TABS ───
tab1, tab2, tab3 = st.tabs([
    "Résultat & Prédiction",
    "Explication SHAP",
    "Récapitulatif Patient"
])

# Données patient dans le bon ordre
patient_data = {
    "age":                      float(age),
    "anaemia":                  int(anaemia),
    "creatinine_phosphokinase": float(creatinine_phosphokinase),
    "diabetes":                 int(diabetes),
    "ejection_fraction":        float(ejection_fraction),
    "high_blood_pressure":      int(hypertension),
    "platelets":                float(platelets),
    "serum_creatinine":         float(serum_creatinine),
    "serum_sodium":             float(serum_sodium),
    "sex":                      1 if sex == "Homme" else 0,
    "smoking":                  int(smoking),
    "time":                     float(time),
}

if predict_btn:
    result = predict_real(patient_data) if MODEL_AVAILABLE else predict_placeholder(patient_data)

    # ────────────────────────────────────────
    # TAB 1 : RÉSULTAT
    # ────────────────────────────────────────
    with tab1:
        col1, col2, col3 = st.columns([1.2, 1, 1])

        with col1:
            st.markdown(f"""
            <div class="{result['risk_class']}">
                <div style="color:{result['risk_color']}; font-size:0.85rem; font-weight:500;
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">
                    Niveau de Risque
                </div>
                <div class="risk-score" style="color:{result['risk_color']};">
                    {result['risk_emoji']} {result['risk_level']}
                </div>
                <div style="color:#8b949e; font-size:0.9rem; margin-top:0.5rem;">
                    Probabilité de décès :
                    <strong style="color:{result['risk_color']};">
                        {result['probability']*100:.1f}%
                    </strong>
                </div>
                <div style="color:#8b949e; font-size:0.75rem; margin-top:0.4rem;">
                    {' Modèle ML réel' if result['is_real'] else ' Simulation (placeholder)'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['probability'] * 100,
                number={'suffix': '%', 'font': {'color': result['risk_color'], 'size': 28}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#8b949e'},
                    'bar':  {'color': result['risk_color']},
                    'bgcolor': '#161b22',
                    'steps': [
                        {'range': [0,  35], 'color': '#1a3d2b'},
                        {'range': [35, 60], 'color': '#3d3417'},
                        {'range': [60,100], 'color': '#3d1a1a'},
                    ],
                    'threshold': {
                        'line': {'color': 'white', 'width': 2},
                        'thickness': 0.75,
                        'value': result['probability'] * 100
                    }
                },
                title={'text': "Score de Risque", 'font': {'color': '#8b949e', 'size': 13}}
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=200, margin=dict(t=30, b=10, l=20, r=20),
                font={'color': '#ffffff'}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col3:
            st.markdown('<div class="metrics-card"><div class="card-title">Métriques Clés</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{age}</div><div class="metric-label">Âge du patient</div><br>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{ejection_fraction}%</div><div class="metric-label">Fraction d\'éjection</div><br>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{serum_creatinine}</div><div class="metric-label">Créatinine sérique</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Recommandation clinique")
        if result['risk_level'] == "Élevé":
            st.error(" Risque élevé détecté. Consultation cardiologique urgente recommandée.")
        elif result['risk_level'] == "Modéré":
            st.warning(" Risque modéré. Surveillance renforcée et bilan complémentaire conseillés.")
        else:
            st.success("Risque faible. Maintien du suivi médical régulier recommandé.")

    # ────────────────────────────────────────
    # TAB 2 : SHAP
    # ────────────────────────────────────────
    with tab2:
        st.markdown("#### Explication SHAP — Pourquoi cette prédiction ?")

        if MODEL_AVAILABLE and SHAP_AVAILABLE:
            # ── VRAI SHAP ──
            # X_train_scaled déjà chargé depuis models/X_train_scaled.csv
            # Patient scalé avec le même scaler
            patient_df     = pd.DataFrame([patient_data])[FEATURE_ORDER]
            patient_scaled = pd.DataFrame(
                scaler.transform(patient_df), columns=FEATURE_ORDER
            )

            col_s1, col_s2 = st.columns(2)

            with col_s1:
                st.markdown("** Summary Plot — Vue globale**")
                st.caption("Impact de chaque feature sur l'ensemble des patients")
                try:
                    fig_summary = generate_shap_summary(model, X_train_scaled)
                    st.pyplot(fig_summary)
                except Exception as e:
                    st.error(f"Erreur SHAP Summary : {e}")

            with col_s2:
                st.markdown("**🔍 Waterfall Plot — Ce patient**")
                st.caption("Ce qui pousse le risque vers le haut ou le bas")
                try:
                    fig_individual = generate_shap_individual(model, X_train_scaled, patient_scaled)
                    st.pyplot(fig_individual)
                except Exception as e:
                    st.error(f"Erreur SHAP Individual : {e}")

            st.info(" Rouge = augmente le risque  |   Bleu = diminue le risque")

        else:
            # ── SHAP simulé ──
            st.markdown(
                '<div class="placeholder-note">⚠️ SHAP simulé — '
                'Sera remplacé après <code>python src/train_model.py</code>.</div>',
                unsafe_allow_html=True
            )
            shap_df = pd.DataFrame({
                "Feature":    result['shap_features'],
                "SHAP Value": result['shap_values']
            }).sort_values("SHAP Value", key=abs, ascending=True)

            colors   = ["#f85149" if v > 0 else "#3fb950" for v in shap_df["SHAP Value"]]
            fig_shap = go.Figure(go.Bar(
                x=shap_df["SHAP Value"], y=shap_df["Feature"],
                orientation='h', marker_color=colors,
                text=[f"{v:+.3f}" for v in shap_df["SHAP Value"]],
                textposition='outside',
            ))
            fig_shap.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                xaxis=dict(title="Valeur SHAP", color='#8b949e',
                           gridcolor='#30363d', zerolinecolor='#8b949e'),
                yaxis=dict(color='#ffffff', gridcolor='#30363d'),
                font=dict(color='#ffffff'), margin=dict(l=10, r=60, t=20, b=40),
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.info(" Rouge = augmente le risque  |  Bleu = diminue le risque")

    # ────────────────────────────────────────
    # TAB 3 : RÉCAPITULATIF
    # ────────────────────────────────────────
    with tab3:
        st.markdown("#### Fiche Patient")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="card"><div class="card-title">Données Numériques</div>', unsafe_allow_html=True)
            for k, v in {
                "Âge":                 f"{age} ans",
                "Fraction d'éjection": f"{ejection_fraction}%",
                "Créatinine sérique":  f"{serum_creatinine} mg/dL",
                "Sodium sérique":      f"{serum_sodium} mEq/L",
                "CPK":                 f"{creatinine_phosphokinase} mcg/L",
                "Plaquettes":          f"{platelets:,.0f} /mL",
                "Suivi":               f"{time} jours",
            }.items():
                st.markdown(f"**{k}** : {v}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="card"><div class="card-title">Facteurs de Risque</div>', unsafe_allow_html=True)
            for k, v in {
                "Anémie": anaemia, "Diabète": diabetes,
                "Hypertension": hypertension, "Tabagisme": smoking
            }.items():
                st.markdown(f"{'✅' if v else '❌'} **{k}**")
            st.markdown(f"**Sexe** : {sex}")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── BOUTON TÉLÉCHARGEMENT PDF ──
        st.markdown("---")
        
        # Fonction de callback pour générer le PDF
        def generate_and_download_pdf():
            """Callback pour générer le PDF lors du téléchargement"""
            shap_summary_fig = None
            shap_individual_fig = None

            if MODEL_AVAILABLE and SHAP_AVAILABLE:
                try:
                    patient_df = pd.DataFrame([patient_data])[FEATURE_ORDER]
                    patient_scaled = pd.DataFrame(
                        scaler.transform(patient_df), columns=FEATURE_ORDER
                    )
                    shap_summary_fig = generate_shap_summary(model, X_train_scaled)
                    shap_individual_fig = generate_shap_individual(model, X_train_scaled, patient_scaled)
                except Exception as e:
                    st.warning(f"⚠️ Impossible d'inclure les graphiques SHAP : {e}")

            # Générer le PDF
            pdf_buffer = generate_pdf_report(
                patient_data, result, shap_summary_fig, shap_individual_fig
            )
            return pdf_buffer
        
        # Bouton de téléchargement direct
        try:
            pdf_data = generate_and_download_pdf()
            
            if pdf_data is not None:
                st.download_button(
                    label="📥 Télécharger le Rapport PDF",
                    data=pdf_data,
                    file_name=f"Rapport_CardioAI_Patient_{age}ans_{result['risk_level']}.pdf",
                    mime="application/pdf",
                    key="pdf_download_btn"
                )
            else:
                st.error("❌ Erreur : Impossible de générer le PDF. Vérifiez que reportlab est installé.")
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération du PDF : {str(e)}")


else:
    # ── État initial ──
    with tab1:
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem; color:#8b949e;">
            <div style="font-size:4rem; margin-bottom:1rem;">🫀</div>
            <h3 style="color:#ffffff; font-family:'DM Serif Display', serif;">
                Prêt pour l'analyse
            </h3>
            <p>Renseignez les données cliniques dans le panneau gauche,<br>
            puis cliquez sur <strong style="color:#58a6ff;">Analyser le Risque</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
    with tab2:
        st.info("Les explications SHAP apparaîtront après la prédiction.")
    with tab3:
        st.info("Le récapitulatif patient apparaîtra après la prédiction.")

# ─── FOOTER ───
st.markdown("---")
mode_label = f"Modèle {type(model).__name__} ✅" if MODEL_AVAILABLE else "Placeholder actif ⚠️"
st.markdown(
    f'<p style="text-align:center; color:#8b949e; font-size:0.8rem;">'
    f'CardioAI • Centrale Casablanca • Coding Week Mars 2026 • {mode_label}</p>',
    unsafe_allow_html=True
)
