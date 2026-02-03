import io
import re
import numpy as np
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Painel de Gastos (Google Drive)", layout="wide")

GDRIVE_ID = "1s-lIrHxMZMRnCOayQeQ5ML0LpLbVTRNy"
GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_ID}"

def money_brl(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "R$ 0,00"
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def brl_to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace("R$", "", regex=False).str.replace("\u00a0", " ", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df

def find_col(df: pd.DataFrame, *candidates: str):
    cols_norm = {c: re.sub(r"\s+", " ", str(c)).strip().lower() for c in df.columns}
    for cand in candidates:
        cand = cand.lower().strip()
        for real, norm in cols_norm.items():
            if cand in norm:
                return real
    return None

@st.cache_data(ttl=60*30, show_spinner=True)  # 30 min (ajuste como quiser)
def baixar_drive(url: str) -> bytes:
    # Google Drive às vezes exige "confirm" (anti-virus). Trata isso.
    s = requests.Session()
    r = s.get(url, timeout=120, allow_redirects=True)
    r.raise_for_status()

    # Se vier HTML pedindo confirmação, pega o token e baixa de novo
    if "text/html" in r.headers.get("Content-Type", "") and "confirm=" in r.text:
        import re as _re
        m = _re.search(r'confirm=([0-9A-Za-z_]+)', r.text)
        if m:
            confirm = m.group(1)
            r = s.get(url + f"&confirm={confirm}", timeout=120, allow_redirects=True)
            r.raise_for_status()

    return r.content

def ler_excel_tratado(data: bytes) -> pd.DataFrame:
    # regra: remover 1ª linha; 2ª vira cabeçalho; remover 2 últimas linhas
    df = pd.read_excel(io.BytesIO(data), skiprows=1, dtype=str)
    if len(df) >= 2:
        df = df.iloc[:-2].copy()
    return df

st.title("📊 Painel de Gastos — Google Drive (atualização automática)")

if st.button("🔄 Atualizar agora"):
    st.cache_data.clear()

try:
    data = baixar_drive(GDRIVE_URL)
    df = ler_excel_tratado(data)
    df = normalize_cols(df)
except Exception as e:
    st.error(f"Erro ao baixar/ler do Google Drive: {e}")
    st.stop()

st.success(f"Arquivo carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")

# ===== Painel por UG =====
COL_UG   = find_col(df, "ug executora", "ug execut", "ug", "unidade gestora")
COL_DOT  = find_col(df, "dotacao atualizada", "dotação atualizada", "dotacao", "dotação")
COL_CRED = find_col(df, "credito disponivel", "crédito disponível", "credito disponível")
COL_ALIQ = find_col(df, "empenhos a liquidar", "a liquidar")
COL_LIQP = find_col(df, "empenhos liquidados a pagar", "liquidados a pagar")
COL_PAGO = find_col(df, "empenhos pagos", "pagos")

missing = [n for n,c in [
    ("UG Executora", COL_UG),
    ("Dotação atualizada", COL_DOT),
    ("Crédito disponível", COL_CRED),
    ("Empenhos a liquidar", COL_ALIQ),
    ("Empenhos liquidados a pagar", COL_LIQP),
    ("Empenhos pagos", COL_PAGO),
] if c is None]

if missing:
    st.error("Não encontrei estas colunas no arquivo: " + ", ".join(missing))
    st.write("Colunas detectadas:", list(df.columns))
    st.stop()

for c in [COL_DOT, COL_CRED, COL_ALIQ, COL_LIQP, COL_PAGO]:
    df[c] = brl_to_float(df[c])

with st.sidebar:
    st.header("Filtro obrigatório")
    ugs = sorted(df[COL_UG].dropna().astype(str).unique().tolist())
    ug_sel = st.selectbox("UG Executora", options=ugs, index=None, placeholder="Selecione a UG...")

if not ug_sel:
    st.warning("Selecione uma **UG Executora** para gerar o painel.")
    st.stop()

df_ug = df[df[COL_UG].astype(str) == str(ug_sel)].copy()

dotacao_loa = df_ug[COL_DOT].sum(skipna=True)
creditos_recebidos = (
    df_ug[COL_CRED].sum(skipna=True)
    + df_ug[COL_ALIQ].sum(skipna=True)
    + df_ug[COL_LIQP].sum(skipna=True)
    + df_ug[COL_PAGO].sum(skipna=True)
)
empenhos_pagos = df_ug[COL_PAGO].sum(skipna=True)
saldo = creditos_recebidos - empenhos_pagos

st.subheader(f"📌 Painel — UG Executora: {ug_sel}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dotação Atualizada (LOA)", money_brl(dotacao_loa))
c2.metric("Créditos Recebidos (soma)", money_brl(creditos_recebidos))
c3.metric("Empenhos pagos", money_brl(empenhos_pagos))
c4.metric("Saldo (Recebidos - Pagos)", money_brl(saldo))

st.divider()

# Resumo por ND/GND/Elemento (se existir)
COL_ND   = find_col(df_ug, "nd", "natureza da despesa", "natureza despesa")
COL_GND  = find_col(df_ug, "gnd", "grupo natureza")
COL_ELEM = find_col(df_ug, "elemento", "elemento despesa")

group_cols = [c for c in [COL_ND, COL_GND, COL_ELEM] if c is not None]
if group_cols:
    resumo = (
        df_ug.groupby(group_cols, dropna=False)[[COL_DOT, COL_CRED, COL_ALIQ, COL_LIQP, COL_PAGO]]
        .sum()
        .reset_index()
    )
    resumo["Créditos recebidos"] = resumo[COL_CRED] + resumo[COL_ALIQ] + resumo[COL_LIQP] + resumo[COL_PAGO]
    st.subheader("Resumo por classificação")
    st.dataframe(resumo, use_container_width=True, height=520)
else:
    st.subheader("Dados filtrados da UG")
    st.dataframe(df_ug, use_container_width=True, height=520)
