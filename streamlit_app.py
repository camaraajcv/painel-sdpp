import io
import re
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Painel de Gastos — Google Drive", layout="wide")

FILE_ID = "1s-lIrHxMZMRnCOayQeQ5ML0LpLbVTRNy"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

UGS_LOA = {"120002", "121002"}

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
# Leitura CSV UTF-16
# =========================
@st.cache_data(ttl=60 * 15, show_spinner=True)
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
        skiprows=1,
        skipfooter=2,
        header=0,
        dtype=str,
        skipinitialspace=True,
    )

    df = df.iloc[2:].reset_index(drop=True)

    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed", case=False)]
    df = df.dropna(axis=1, how="all")

    df = normalize_cols(df)

    df = df.rename(columns={
        "13": "DOTACAO_ATUALIZADA",
        "19": "CREDITO_DISPONIVEL",
        "30": "EMPENHADAS_A_LIQUIDAR",
        "32": "LIQUIDADAS_A_PAGAR",
        "34": "PAGAS",
    })

    return df

# =========================
# UI
# =========================
st.title("📊 Painel de Gastos — Google Drive")

if st.button("🔄 Atualizar agora"):
    st.cache_data.clear()

df = carregar_df()

# =========================
# Colunas
# =========================
COL_UG_EXEC = find_col(df, "ug executora")
COL_UGR     = find_col(df, "ug responsável", "ug responsavel")
COL_DOT  = find_col(df, "dotacao_atualizada")
COL_CRED = find_col(df, "credito_disponivel")
COL_ALIQ = find_col(df, "empenhadas_a_liquidar")
COL_LIQP = find_col(df, "liquidadas_a_pagar")
COL_PAGO = find_col(df, "pagas")

for c in [COL_DOT, COL_CRED, COL_ALIQ, COL_LIQP, COL_PAGO]:
    df[c] = brl_to_float(df[c])

# =========================
# Filtro UG
# =========================
with st.sidebar:
    ugs = sorted(df[COL_UG_EXEC].dropna().astype(str).unique())
    ug_sel = st.selectbox("UG Executora", options=ugs)

ug_sel_str = str(ug_sel).strip()
df_ug = df[df[COL_UG_EXEC].astype(str).str.strip() == ug_sel_str].copy()

# =========================
# UGRs vinculadas à UG selecionada
# =========================
ugrs_vinculadas = (
    df_ug[COL_UGR]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

# =========================
# Cálculo da LOA correto
# =========================
df_loa_pool = df[df[COL_UG_EXEC].astype(str).str.strip().isin(UGS_LOA)].copy()
df_loa = df_loa_pool[df_loa_pool[COL_UGR].astype(str).str.strip().isin(ugrs_vinculadas)]

dotacao_loa = df_loa[COL_DOT].sum(skipna=True)

# =========================
# Créditos Recebidos
# =========================
creditos_recebidos = (
    df_ug[COL_CRED].sum()
    + df_ug[COL_ALIQ].sum()
    + df_ug[COL_LIQP].sum()
    + df_ug[COL_PAGO].sum()
)

empenhos_pagos = df_ug[COL_PAGO].sum()
saldo = creditos_recebidos - empenhos_pagos

# =========================
# Painel
# =========================
st.subheader(f"UG Executora: {ug_sel_str}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dotação Atualizada (LOA)", money_brl(dotacao_loa))
c2.metric("Créditos Recebidos", money_brl(creditos_recebidos))
c3.metric("Despesas Pagas", money_brl(empenhos_pagos))
c4.metric("Saldo", money_brl(saldo))

st.divider()
st.dataframe(df_ug, use_container_width=True)
