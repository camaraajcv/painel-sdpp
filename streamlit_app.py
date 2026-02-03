import io
import re
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Painel de Gastos — Google Drive", layout="wide")

FILE_ID = "1s-lIrHxMZMRnCOayQeQ5ML0LpLbVTRNy"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# =========================
# Helpers
# =========================
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

# =========================
# Leitura do CSV UTF-16 do Google Drive
# =========================
@st.cache_data(ttl=60*15, show_spinner=True)
def carregar_df():
    r = requests.get(URL, timeout=120, allow_redirects=True)
    r.raise_for_status()
    data = r.content

    df = pd.read_csv(
        io.BytesIO(data),
        encoding="utf-16",
        sep=",",
        quotechar='"',
        engine="python",
        skiprows=1,       # remove a linha "xxxxxxx"
        skipfooter=2,     # remove as 2 últimas linhas
        header=0,
        dtype=str,
        skipinitialspace=True
    )

    # ✅ SUA REGRA: excluir as 2 primeiras linhas do DF gerado
    df = df.iloc[2:].reset_index(drop=True)

    # limpar colunas vazias/Unnamed
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed", case=False)]
    df = df.dropna(axis=1, how="all")

    df = normalize_cols(df)

    return df

# =========================
# UI
# =========================
st.title("📊 Painel de Gastos — Google Drive")

if st.button("🔄 Atualizar agora"):
    st.cache_data.clear()

try:
    df = carregar_df()
except Exception as e:
    st.error(f"Erro ao ler arquivo do Google Drive: {e}")
    st.stop()

st.success(f"Arquivo carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")

# =========================
# Localizar colunas importantes
# =========================
COL_UG   = find_col(df, "ug executora", "ug execut", "ug", "unidade gestora")
COL_DOT  = find_col(df, "dotacao atualizada", "dotação atualizada")

COL_CRED = find_col(df, "credito disponivel", "crédito disponível")
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

# Converter moeda
for c in [COL_DOT, COL_CRED, COL_ALIQ, COL_LIQP, COL_PAGO]:
    df[c] = brl_to_float(df[c])

# =========================
# Filtro obrigatório UG
# =========================
with st.sidebar:
    st.header("Filtro obrigatório")
    ugs = sorted(df[COL_UG].dropna().astype(str).unique().tolist())
    ug_sel = st.selectbox("UG Executora", options=ugs, index=None, placeholder="Selecione a UG...")

if not ug_sel:
    st.warning("Selecione uma **UG Executora** para gerar o painel.")
    st.stop()

df_ug = df[df[COL_UG].astype(str) == str(ug_sel)].copy()

# =========================
# Métricas
# =========================
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
c2.metric("Créditos Recebidos", money_brl(creditos_recebidos))
c3.metric("Empenhos pagos", money_brl(empenhos_pagos))
c4.metric("Saldo", money_brl(saldo))

st.divider()
st.subheader("Dados filtrados")
st.dataframe(df_ug, use_container_width=True, height=500)
