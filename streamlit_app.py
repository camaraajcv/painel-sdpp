import io
import re
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Painel de Gastos — Google Drive", layout="wide")

# =========================
# Fonte (Google Drive)
# =========================
FILE_ID = "1s-lIrHxMZMRnCOayQeQ5ML0LpLbVTRNy"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# UGs onde a LOA/Dotação Atualizada está registrada (pool LOA)
UGS_LOA = {"120002", "121002"}

# =========================
# Helpers
# =========================
def br_compact(x: float) -> str:
    """Formato curto: R$ 1,23 bi / mi / mil"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "R$ 0"
    absx = abs(x)
    if absx >= 1_000_000_000:
        return "R$ " + f"{x/1_000_000_000:.2f}".replace(".", ",") + " bi"
    if absx >= 1_000_000:
        return "R$ " + f"{x/1_000_000:.2f}".replace(".", ",") + " mi"
    if absx >= 1_000:
        return "R$ " + f"{x/1_000:.2f}".replace(".", ",") + " mil"
    return "R$ " + f"{x:.0f}".replace(".", ",")

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
        skipinitialspace=True
    )

    # excluir as 2 primeiras linhas do DF gerado (células mescladas)
    df = df.iloc[2:].reset_index(drop=True)

    # limpar colunas vazias/Unnamed
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed", case=False)]
    df = df.dropna(axis=1, how="all")

    df = normalize_cols(df)

    # renomear colunas numéricas
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
# Colunas principais
# =========================
COL_UG_EXEC = find_col(df, "ug executora")
COL_UGR     = find_col(df, "ug responsável", "ug responsavel", "ugr")

COL_ACAO = find_col(df, "ação governo", "acao governo", "acao")
COL_ND   = find_col(df, "natureza despesa", "natureza da despesa", "nd")
COL_PO   = find_col(df, "plano orçamentário", "plano orcamentario", "plano orçamentario")

COL_DOT  = find_col(df, "dotacao_atualizada")
COL_CRED = find_col(df, "credito_disponivel")
COL_ALIQ = find_col(df, "empenhadas_a_liquidar")
COL_LIQP = find_col(df, "liquidadas_a_pagar")
COL_PAGO = find_col(df, "pagas")

missing = [n for n,c in [
    ("UG Executora", COL_UG_EXEC),
    ("UG Responsável (UGR)", COL_UGR),
    ("Ação Governo", COL_ACAO),
    ("Natureza Despesa", COL_ND),
    ("Plano Orçamentário", COL_PO),
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

# Converter valores
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

# Base da UG selecionada (para colunas fixas)
df_ug = df[df[COL_UG_EXEC].astype(str).str.strip() == ug_sel_str].copy()

# UGRs vinculadas à UG selecionada
ugrs_vinculadas = (
    df_ug[COL_UGR]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

# Pool LOA (120002/121002) filtrado pelas UGRs vinculadas
df_loa_pool = df[df[COL_UG_EXEC].astype(str).str.strip().isin(UGS_LOA)].copy()
df_loa = df_loa_pool[df_loa_pool[COL_UGR].astype(str).str.strip().isin(ugrs_vinculadas)].copy()

# =========================
# KPIs
# =========================
dotacao_loa = df_loa[COL_DOT].sum(skipna=True)

creditos_recebidos = (
    df_ug[COL_CRED].sum(skipna=True)
    + df_ug[COL_ALIQ].sum(skipna=True)
    + df_ug[COL_LIQP].sum(skipna=True)
    + df_ug[COL_PAGO].sum(skipna=True)
)

despesas_pagas = df_ug[COL_PAGO].sum(skipna=True)
saldo = creditos_recebidos - despesas_pagas

st.subheader(f"📌 Painel — UG Executora: {ug_sel_str}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Dotação Atualizada (LOA)", br_compact(dotacao_loa))
k2.metric("Créditos Recebidos", br_compact(creditos_recebidos))
k3.metric("Despesas Pagas", br_compact(despesas_pagas))
k4.metric("Saldo (Recebidos - Pagos)", br_compact(saldo))

# =========================
# Tabela profissional (UGR vinculada × Ação × ND × Plano Orçamentário)
# LOA vem do pool; demais valores vêm da UG executora filtrada
# =========================
st.divider()
st.subheader("📌 Detalhamento — UGR vinculada × Ação × Natureza Despesa × Plano Orçamentário")

group_cols = [COL_UGR, COL_ACAO, COL_ND, COL_PO]

# LOA (pool LOA)
loa_grp = (
    df_loa.groupby(group_cols, dropna=False)[COL_DOT]
    .sum(min_count=1)
    .reset_index()
    .rename(columns={COL_DOT: "DOTACAO_ATUALIZADA"})
)

# Execução (UG selecionada)
exec_grp = (
    df_ug.groupby(group_cols, dropna=False)[[COL_CRED, COL_ALIQ, COL_LIQP, COL_PAGO]]
    .sum(min_count=1)
    .reset_index()
    .rename(columns={
        COL_CRED: "CREDITO_DISPONIVEL",
        COL_ALIQ: "EMPENHADAS_A_LIQUIDAR",
        COL_LIQP: "LIQUIDADAS_A_PAGAR",
        COL_PAGO: "PAGAS",
    })
)

# Junta
tabela = loa_grp.merge(exec_grp, on=group_cols, how="outer")

# Normaliza NaNs
for c in ["DOTACAO_ATUALIZADA", "CREDITO_DISPONIVEL", "EMPENHADAS_A_LIQUIDAR", "LIQUIDADAS_A_PAGAR", "PAGAS"]:
    tabela[c] = pd.to_numeric(tabela[c], errors="coerce").fillna(0.0)

tabela["CREDITOS_RECEBIDOS"] = (
    tabela["CREDITO_DISPONIVEL"]
    + tabela["EMPENHADAS_A_LIQUIDAR"]
    + tabela["LIQUIDADAS_A_PAGAR"]
    + tabela["PAGAS"]
)

# Ordenação
tabela = tabela.sort_values(["DOTACAO_ATUALIZADA", "CREDITOS_RECEBIDOS"], ascending=[False, False])

# Filtros rápidos
with st.expander("Filtros rápidos", expanded=False):
    ugr_opts = sorted(tabela[COL_UGR].dropna().astype(str).unique().tolist())
    acao_opts = sorted(tabela[COL_ACAO].dropna().astype(str).unique().tolist())
    nd_opts = sorted(tabela[COL_ND].dropna().astype(str).unique().tolist())
    po_opts = sorted(tabela[COL_PO].dropna().astype(str).unique().tolist())

    f_ugr = st.multiselect("UGR", options=ugr_opts, default=[])
    f_acao = st.multiselect("Ação Governo", options=acao_opts, default=[])
    f_nd = st.multiselect("Natureza Despesa", options=nd_opts, default=[])
    f_po = st.multiselect("Plano Orçamentário", options=po_opts, default=[])

tab_f = tabela.copy()
if f_ugr: tab_f = tab_f[tab_f[COL_UGR].astype(str).isin(f_ugr)]
if f_acao: tab_f = tab_f[tab_f[COL_ACAO].astype(str).isin(f_acao)]
if f_nd: tab_f = tab_f[tab_f[COL_ND].astype(str).isin(f_nd)]
if f_po: tab_f = tab_f[tab_f[COL_PO].astype(str).isin(f_po)]

# Formatação BRL para exibir
tab_show = tab_f.copy()
for c in ["DOTACAO_ATUALIZADA", "CREDITO_DISPONIVEL", "EMPENHADAS_A_LIQUIDAR", "LIQUIDADAS_A_PAGAR", "PAGAS", "CREDITOS_RECEBIDOS"]:
    tab_show[c] = tab_show[c].map(money_brl)

st.dataframe(tab_show, use_container_width=True, height=560)

# =========================
# Diagnóstico opcional
# =========================
if debug:
    st.divider()
    st.subheader("Diagnóstico")
    st.write("UGR(s) vinculadas:", uগ্রs_vinculadas if False else ugs_vinculadas)  # evita typo acidental
    st.write("Linhas pool LOA:", len(df_loa_pool))
    st.write("Linhas LOA usadas (pool filtrado por UGR vinculada):", len(df_loa))
    st.write("Linhas UG filtrada:", len(df_ug))
    st.dataframe(df_loa[[COL_UG_EXEC, COL_UGR, COL_ACAO, COL_ND, COL_PO, COL_DOT]].head(50), use_container_width=True)
    st.dataframe(df_ug[[COL_UG_EXEC, COL_UGR, COL_ACAO, COL_ND, COL_PO, COL_CRED, COL_ALIQ, COL_LIQP, COL_PAGO]].head(50), use_container_width=True)
