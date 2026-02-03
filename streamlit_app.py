import io
import re
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Painel de Gastos — Google Drive", layout="wide")

FILE_ID = "1s-lIrHxMZMRnCOayQeQ5ML0LpLbVTRNy"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# UGs onde a LOA/Dotação Atualizada está registrada (pool LOA)
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
# Leitura do CSV UTF-16 do Google Drive
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
        skiprows=1,       # remove a linha "xxxxxxx"
        skipfooter=2,     # remove as 2 últimas linhas
        header=0,
        dtype=str,
        skipinitialspace=True,
    )

    # ✅ excluir as 2 primeiras linhas do DF gerado (células mescladas)
    df = df.iloc[2:].reset_index(drop=True)

    # limpar colunas vazias/Unnamed
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed", case=False)]
    df = df.dropna(axis=1, how="all")

    df = normalize_cols(df)

    # ✅ renomear colunas numéricas
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

try:
    df = carregar_df()
except Exception as e:
    st.error(f"Erro ao ler arquivo do Google Drive: {e}")
    st.stop()

st.success(f"Arquivo carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")

# =========================
# Localizar colunas importantes
# =========================
COL_UG_EXEC = find_col(df, "ug executora", "ug execut", "ug", "unidade gestora")
COL_UG_RESP = find_col(df, "ug responsável", "ug responsavel")
COL_NECCOR  = find_col(df, "ne ccor", "neccor")

COL_DOT  = find_col(df, "dotacao_atualizada", "dotação atualizada", "dotacao atualizada")
COL_CRED = find_col(df, "credito_disponivel", "crédito disponível", "credito disponivel")
COL_ALIQ = find_col(df, "empenhadas_a_liquidar", "empenhos a liquidar", "a liquidar")
COL_LIQP = find_col(df, "liquidadas_a_pagar", "liquidados a pagar", "liquidadas a pagar")
COL_PAGO = find_col(df, "pagas", "empenhos pagos", "pagos")

missing = [n for n,c in [
    ("UG Executora", COL_UG_EXEC),
    ("Dotação atualizada", COL_DOT),
    ("Crédito disponível", COL_CRED),
    ("Empenhadas a liquidar", COL_ALIQ),
    ("Liquidadas a pagar", COL_LIQP),
    ("Pagas", COL_PAGO),
] if c is None]

if missing:
    st.error("Não encontrei estas colunas no arquivo: " + ", ".join(missing))
    st.write("Colunas detectadas:", list(df.columns))
    st.stop()

# Converter moeda (somente as colunas de valores)
for c in [COL_DOT, COL_CRED, COL_ALIQ, COL_LIQP, COL_PAGO]:
    df[c] = brl_to_float(df[c])

# =========================
# Filtro obrigatório UG (Executora)
# =========================
with st.sidebar:
    st.header("Filtro obrigatório")
    ugs = sorted(df[COL_UG_EXEC].dropna().astype(str).str.strip().unique().tolist())
    ug_sel = st.selectbox("UG Executora", options=ugs, index=None, placeholder="Selecione a UG...")

if not ug_sel:
    st.warning("Selecione uma **UG Executora** para gerar o painel.")
    st.stop()

ug_sel_str = str(ug_sel).strip()

df_ug = df[df[COL_UG_EXEC].astype(str).str.strip() == ug_sel_str].copy()

# =========================
# Dotação Atualizada (LOA) com regra especial:
# - valores só existem nas UGs 120002 e 121002
# - usuário seleciona UG Executora, mas LOA deve ser buscada como se a UG selecionada fosse a "responsável"
#   (tentando casar por UG Responsável; se não existir, tenta por NE CCor)
# =========================
df_loa_pool = df[df[COL_UG_EXEC].astype(str).str.strip().isin(UGS_LOA)].copy()

dotacao_loa = 0.0

# tenta por UG Responsável
if COL_UG_RESP:
    dotacao_loa = df_loa_pool[df_loa_pool[COL_UG_RESP].astype(str).str.strip() == ug_sel_str][COL_DOT].sum(skipna=True)

# fallback por NE CCor (se existir e se por UG Responsável não achou nada)
if (dotacao_loa == 0 or np.isnan(dotacao_loa)) and COL_NECCOR:
    dotacao_loa = df_loa_pool[df_loa_pool[COL_NECCOR].astype(str).str.strip() == ug_sel_str][COL_DOT].sum(skipna=True)

# =========================
# Métricas (regra que você passou)
# Créditos recebidos = Crédito disponível + A liquidar + Liquidadas a pagar + Pagas
# =========================
creditos_recebidos = (
    df_ug[COL_CRED].sum(skipna=True)
    + df_ug[COL_ALIQ].sum(skipna=True)
    + df_ug[COL_LIQP].sum(skipna=True)
    + df_ug[COL_PAGO].sum(skipna=True)
)

empenhos_pagos = df_ug[COL_PAGO].sum(skipna=True)
saldo = creditos_recebidos - empenhos_pagos

st.subheader(f"📌 Painel — UG Executora: {ug_sel_str}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dotação Atualizada (LOA — pool 120002/121002)", money_brl(dotacao_loa))
c2.metric("Créditos Recebidos", money_brl(creditos_recebidos))
c3.metric("Despesas pagas (controle empenho)", money_brl(empenhos_pagos))
c4.metric("Saldo (Recebidos - Pagos)", money_brl(saldo))

st.divider()
st.subheader("Dados filtrados (UG executora selecionada)")
st.dataframe(df_ug, use_container_width=True, height=520)
