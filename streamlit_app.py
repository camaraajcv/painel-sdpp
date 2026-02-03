import io
import re
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Painel de Gastos — Google Drive", layout="wide")

FILE_ID = "1s-lIrHxMZMRnCOayQeQ5ML0LpLbVTRNy"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

UGS_LOA = {"120002", "121002"}  # pool LOA

# =========================
# Helpers
# =========================
def money_brl(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "R$ 0,00"
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def brl_to_float(series: pd.Series) -> pd.Series:
    # tolerante: remove R$, espaços, separadores BR, etc.
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
        skiprows=1,       # remove "xxxxxxx"
        skipfooter=2,     # remove 2 últimas linhas
        header=0,
        dtype=str,
        skipinitialspace=True,
    )

    # ✅ remover apenas as 2 primeiras linhas do DF gerado
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

colA, colB = st.columns([1, 1])
with colA:
    if st.button("🔄 Atualizar agora"):
        st.cache_data.clear()
with colB:
    debug = st.checkbox("Mostrar diagnóstico", value=False)

try:
    df = carregar_df()
except Exception as e:
    st.error(f"Erro ao ler arquivo do Google Drive: {e}")
    st.stop()

st.success(f"Arquivo carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")

# =========================
# Colunas (fixas do seu CSV)
# =========================
COL_UG_EXEC = find_col(df, "ug executora")
COL_UGR     = find_col(df, "ug responsável", "ug responsavel", "ugr")  # << UGR (chave da LOA)
COL_DOT     = find_col(df, "dotacao_atualizada")
COL_CRED    = find_col(df, "credito_disponivel")
COL_ALIQ    = find_col(df, "empenhadas_a_liquidar")
COL_LIQP    = find_col(df, "liquidadas_a_pagar")
COL_PAGO    = find_col(df, "pagas")

missing = [n for n,c in [
    ("UG Executora", COL_UG_EXEC),
    ("UGR (UG Responsável)", COL_UGR),
    ("DOTACAO_ATUALIZADA", COL_DOT),
    ("CREDITO_DISPONIVEL", COL_CRED),
    ("EMPENHADAS_A_LIQUIDAR", COL_ALIQ),
    ("LIQUIDADAS_A_PAGAR", COL_LIQP),
    ("PAGAS", COL_PAGO),
] if c is None]

if missing:
    st.error("Não encontrei estas colunas no arquivo: " + ", ".join(missing))
    st.write("Colunas detectadas:", list(df.columns))
    st.stop()

# Converter valores (moeda)
for c in [COL_DOT, COL_CRED, COL_ALIQ, COL_LIQP, COL_PAGO]:
    df[c] = brl_to_float(df[c])

# =========================
# Filtro obrigatório UG Executora
# =========================
with st.sidebar:
    st.header("Filtro obrigatório")
    ugs = sorted(df[COL_UG_EXEC].dropna().astype(str).str.strip().unique().tolist())
    ug_sel = st.selectbox("UG Executora", options=ugs, index=None, placeholder="Selecione a UG...")

if not ug_sel:
    st.warning("Selecione uma **UG Executora** para gerar o painel.")
    st.stop()

ug_sel_str = str(ug_sel).strip()

# Dados da UG selecionada (para créditos recebidos etc.)
df_ug = df[df[COL_UG_EXEC].astype(str).str.strip() == ug_sel_str].copy()

# =========================
# REGRA CERTA DA DOTAÇÃO (LOA)
# pool LOA = UG Exec 120002/121002
# chave = UGR == UG selecionada
# =========================
df_loa_pool = df[df[COL_UG_EXEC].astype(str).str.strip().isin(UGS_LOA)].copy()
df_loa_ug = df_loa_pool[df_loa_pool[COL_UGR].astype(str).str.strip() == ug_sel_str].copy()
dotacao_loa = df_loa_ug[COL_DOT].sum(skipna=True)

# =========================
# REGRA CERTA DOS CRÉDITOS RECEBIDOS (UG selecionada)
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
c1.metric("Dotação Atualizada (LOA — pool 120002/121002, por UGR)", money_brl(dotacao_loa))
c2.metric("Créditos Recebidos (CD + ALIQ + LAP + PAGAS)", money_brl(creditos_recebidos))
c3.metric("Despesas pagas", money_brl(empenhos_pagos))
c4.metric("Saldo (Recebidos - Pagos)", money_brl(saldo))

if debug:
    st.divider()
    st.subheader("Diagnóstico")
    st.write("Linhas da UG selecionada:", len(df_ug))
    st.write("Linhas no pool LOA (120002/121002):", len(df_loa_pool))
    st.write("Linhas do pool LOA filtradas por UGR==UG selecionada:", len(df_loa_ug))
    st.write("Somatório componentes (UG selecionada):")
    st.write({
        "CREDITO_DISPONIVEL": float(df_ug[COL_CRED].sum(skipna=True)),
        "EMPENHADAS_A_LIQUIDAR": float(df_ug[COL_ALIQ].sum(skipna=True)),
        "LIQUIDADAS_A_PAGAR": float(df_ug[COL_LIQP].sum(skipna=True)),
        "PAGAS": float(df_ug[COL_PAGO].sum(skipna=True)),
    })
    st.write("Dotação (pool LOA por UGR):", float(dotacao_loa))
    st.dataframe(df_loa_ug[[COL_UG_EXEC, COL_UGR, COL_DOT]].head(50), use_container_width=True)

st.divider()
st.subheader("Dados filtrados (UG executora selecionada)")
st.dataframe(df_ug, use_container_width=True, height=520)

# Se ainda ocorrer LOA < Recebidos, mostre alerta (para facilitar depuração)
if dotacao_loa < creditos_recebidos:
    st.warning(
        "⚠️ Atenção: Dotação (LOA) ficou menor que Créditos Recebidos. "
        "Isso normalmente indica que a chave UGR não está batendo com a UG selecionada "
        "(ex.: espaços, zeros à esquerda, ou UGR diferente do que você considera). "
        "Marque 'Mostrar diagnóstico' para ver as linhas usadas na LOA."
    )
