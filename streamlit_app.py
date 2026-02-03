import io
import re
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

st.set_page_config(page_title="Painel de Gastos — Google Drive", layout="wide")

# =========================
# Fonte (Google Drive)
# =========================
FILE_ID = "1s-lIrHxMZMRnCOayQeQ5ML0LpLbVTRNy"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# Pool LOA (UGs onde a dotação aparece)
UGS_LOA = {"120002", "121002"}

# =========================
# Estilo (KPIs em cards)
# =========================
st.markdown("""
<style>
.kpi-grid{
  display:grid;
  grid-template-columns:repeat(4, minmax(0, 1fr));
  gap:12px;
  margin-top:8px;
  margin-bottom:8px;
}
.kpi-card{
  border-radius:16px;
  padding:14px 14px 12px;
  border:1px solid rgba(255,255,255,.25);
  box-shadow:0 10px 24px rgba(0,0,0,.10);
  color:#0b1220;
}
.kpi-title{
  font-size:0.85rem;
  opacity:.85;
  margin-bottom:8px;
}
.kpi-value{
  font-size:1.60rem;
  font-weight:850;
  letter-spacing:-0.02em;
  line-height:1.1;
}
.kpi-sub{
  font-size:0.80rem;
  opacity:.78;
  margin-top:6px;
}
.kpi-1{ background:linear-gradient(135deg, #dbeafe 0%, #eff6ff 55%, #ffffff 100%); }
.kpi-2{ background:linear-gradient(135deg, #dcfce7 0%, #f0fdf4 55%, #ffffff 100%); }
.kpi-3{ background:linear-gradient(135deg, #ffedd5 0%, #fff7ed 55%, #ffffff 100%); }
.kpi-4{ background:linear-gradient(135deg, #fae8ff 0%, #fdf4ff 55%, #ffffff 100%); }
.small-note{font-size:.78rem; opacity:.75}
</style>
""", unsafe_allow_html=True)

def kpi_card(title: str, value: str, subtitle: str, cls: str):
    st.markdown(
        f"""
        <div class="kpi-card {cls}">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Helpers
# =========================
def _to_float(x) -> float:
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x) if not np.isnan(x) else 0.0
        return float(x)
    except Exception:
        return 0.0

def br_compact(x: float) -> str:
    """
    Formato curto BR:
      R$ 1,23 bi | R$ 450,80 mi | R$ 12,34 mil | R$ 980
    """
    x = _to_float(x)
    absx = abs(x)
    if absx >= 1_000_000_000:
        v = x / 1_000_000_000
        return f"R$ {str(f'{v:.2f}').replace('.', ',')} bi"
    if absx >= 1_000_000:
        v = x / 1_000_000
        return f"R$ {str(f'{v:.2f}').replace('.', ',')} mi"
    if absx >= 1_000:
        v = x / 1_000
        return f"R$ {str(f'{v:.2f}').replace('.', ',')} mil"
    s = f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"

def money_brl(x: float) -> str:
    x = _to_float(x)
    return "R$ " + f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

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
        skiprows=1,       # remove linha "xxxxxxx"
        skipfooter=2,     # remove 2 últimas linhas
        header=0,
        dtype=str,
        skipinitialspace=True,
    )

    # remove 2 primeiras linhas do DF gerado (células mescladas)
    df = df.iloc[2:].reset_index(drop=True)

    # remove colunas vazias/Unnamed
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed", case=False)]
    df = df.dropna(axis=1, how="all")

    df = normalize_cols(df)

    # renomear colunas numéricas (nomes "13","19","30","32","34")
    df = df.rename(columns={
        "13": "DOTACAO_ATUALIZADA",
        "19": "CREDITO_DISPONIVEL",
        "30": "EMPENHADAS_A_LIQUIDAR",
        "32": "LIQUIDADAS_A_PAGAR",
        "34": "PAGAS",
    })

    return df

# =========================
# UI topo
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
# Colunas necessárias
# =========================
COL_UG_EXEC = find_col(df, "ug executora")
COL_UGR     = find_col(df, "ug responsável", "ug responsavel", "ugr")
COL_ACAO    = find_col(df, "ação governo", "acao governo", "acao")
COL_ND      = find_col(df, "natureza despesa", "natureza da despesa", "nd")
COL_PO      = find_col(df, "plano orçamentário", "plano orcamentario", "plano orçamentario", "plano orcamentário")

COL_DOT  = find_col(df, "dotacao_atualizada")
COL_CRED = find_col(df, "credito_disponivel")
COL_ALIQ = find_col(df, "empenhadas_a_liquidar")
COL_LIQP = find_col(df, "liquidadas_a_pagar")
COL_PAGO = find_col(df, "pagas")

missing = [n for n, c in [
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

# Converte valores BRL -> float
for c in [COL_DOT, COL_CRED, COL_ALIQ, COL_LIQP, COL_PAGO]:
    df[c] = brl_to_float(df[c])

# =========================
# Sidebar (UG obrigatória)
# =========================
with st.sidebar:
    st.header("Filtro obrigatório")
    ugs = sorted(df[COL_UG_EXEC].dropna().astype(str).str.strip().unique().tolist())
    ug_sel = st.selectbox("UG Executora", options=ugs, index=None, placeholder="Selecione a UG...")

if not ug_sel:
    st.warning("Selecione uma **UG Executora** para gerar o painel.")
    st.stop()

ug_sel_str = str(ug_sel).strip()

# Base da UG selecionada (demais valores fixos nela)
df_ug = df[df[COL_UG_EXEC].astype(str).str.strip() == ug_sel_str].copy()

# UGRs vinculadas à UG selecionada
ugrs_vinculadas = (
    df_ug[COL_UGR].dropna().astype(str).str.strip().unique().tolist()
)

# Pool LOA (120002/121002) filtrado por UGRs vinculadas
df_loa_pool = df[df[COL_UG_EXEC].astype(str).str.strip().isin(UGS_LOA)].copy()
df_loa = df_loa_pool[df_loa_pool[COL_UGR].astype(str).str.strip().isin(ugrs_vinculadas)].copy()

# =========================
# KPIs
# =========================
dotacao_loa = float(df_loa[COL_DOT].sum(skipna=True))

creditos_recebidos = float(
    df_ug[COL_CRED].sum(skipna=True)
    + df_ug[COL_ALIQ].sum(skipna=True)
    + df_ug[COL_LIQP].sum(skipna=True)
    + df_ug[COL_PAGO].sum(skipna=True)
)

despesas_pagas = float(df_ug[COL_PAGO].sum(skipna=True))
saldo = creditos_recebidos - despesas_pagas

st.subheader(f"📌 Painel — UG Executora: {ug_sel_str}")

st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
kpi_card("Dotação Atualizada (LOA)", br_compact(dotacao_loa), "Pool 120002/121002 por UGRs vinculadas", "kpi-1")
kpi_card("Créditos Recebidos", br_compact(creditos_recebidos), "CD + ALIQ + LAP + Pagas (UG selecionada)", "kpi-2")
kpi_card("Despesas Pagas", br_compact(despesas_pagas), "Controle empenho (UG selecionada)", "kpi-3")
kpi_card("Saldo", br_compact(saldo), "Recebidos − Pagos", "kpi-4")
st.markdown('</div>', unsafe_allow_html=True)

st.caption("Regras: **Dotação (LOA)** vem do pool 120002/121002 filtrado por **UGRs vinculadas**; demais valores são da **UG Executora selecionada**.")

# =========================
# Tabela executiva com ND (UGR × Ação × ND × Plano)
# =========================
st.divider()
st.subheader("📌 Detalhamento — UGR × Ação Governo × Natureza Despesa × Plano Orçamentário")

group_cols = [COL_UGR, COL_ACAO, COL_ND, COL_PO]

# LOA (pool) por dimensões
loa_grp = (
    df_loa.groupby(group_cols, dropna=False)[COL_DOT]
    .sum(min_count=1)
    .reset_index()
    .rename(columns={COL_DOT: "DOTACAO_ATUALIZADA"})
)

# Execução (UG selecionada) por dimensões
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

tab_exec = loa_grp.merge(exec_grp, on=group_cols, how="outer")

for c in ["DOTACAO_ATUALIZADA", "CREDITO_DISPONIVEL", "EMPENHADAS_A_LIQUIDAR", "LIQUIDADAS_A_PAGAR", "PAGAS"]:
    tab_exec[c] = pd.to_numeric(tab_exec[c], errors="coerce").fillna(0.0)

tab_exec["CREDITOS_RECEBIDOS"] = (
    tab_exec["CREDITO_DISPONIVEL"]
    + tab_exec["EMPENHADAS_A_LIQUIDAR"]
    + tab_exec["LIQUIDADAS_A_PAGAR"]
    + tab_exec["PAGAS"]
)

tab_exec = tab_exec.sort_values(["DOTACAO_ATUALIZADA", "CREDITOS_RECEBIDOS"], ascending=[False, False])

# Filtros rápidos (mais “modernos” em colunas)
with st.expander("Filtros rápidos", expanded=False):
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])

    ugr_opts = sorted(tab_exec[COL_UGR].dropna().astype(str).unique().tolist())
    acao_opts = sorted(tab_exec[COL_ACAO].dropna().astype(str).unique().tolist())
    nd_opts = sorted(tab_exec[COL_ND].dropna().astype(str).unique().tolist())
    po_opts = sorted(tab_exec[COL_PO].dropna().astype(str).unique().tolist())

    with f1:
        f_ugr = st.multiselect("UGR", options=ugr_opts, default=[], placeholder="Selecione...")
    with f2:
        f_acao = st.multiselect("Ação", options=acao_opts, default=[], placeholder="Selecione...")
    with f3:
        f_nd = st.multiselect("Natureza", options=nd_opts, default=[], placeholder="Selecione...")
    with f4:
        f_po = st.multiselect("Plano Orç.", options=po_opts, default=[], placeholder="Selecione...")

tab_f = tab_exec.copy()
if f_ugr:
    tab_f = tab_f[tab_f[COL_UGR].astype(str).isin(f_ugr)]
if f_acao:
    tab_f = tab_f[tab_f[COL_ACAO].astype(str).isin(f_acao)]
if f_nd:
    tab_f = tab_f[tab_f[COL_ND].astype(str).isin(f_nd)]
if f_po:
    tab_f = tab_f[tab_f[COL_PO].astype(str).isin(f_po)]

tab_show = tab_f.copy()
for c in ["DOTACAO_ATUALIZADA", "CREDITO_DISPONIVEL", "EMPENHADAS_A_LIQUIDAR", "LIQUIDADAS_A_PAGAR", "PAGAS", "CREDITOS_RECEBIDOS"]:
    tab_show[c] = tab_show[c].map(money_brl)

st.dataframe(tab_show, use_container_width=True, height=560)

# =========================
# Gráficos por Ação (UG executora selecionada)
# =========================
st.divider()
st.subheader("📈 Gráficos — por Ação Governo (valores da UG executora selecionada)")

acao_sum = (
    df_ug.groupby(COL_ACAO, dropna=False)[[COL_CRED, COL_ALIQ, COL_LIQP, COL_PAGO]]
    .sum(min_count=1)
    .reset_index()
    .rename(columns={
        COL_ACAO: "Ação Governo",
        COL_CRED: "Crédito disponível",
        COL_ALIQ: "Empenhadas a liquidar",
        COL_LIQP: "Liquidadas a pagar",
        COL_PAGO: "Pagas",
    })
)

acao_sum["Créditos recebidos"] = (
    acao_sum["Crédito disponível"]
    + acao_sum["Empenhadas a liquidar"]
    + acao_sum["Liquidadas a pagar"]
    + acao_sum["Pagas"]
)

# Ordena para os gráficos ficarem mais legíveis
acao_sum = acao_sum.sort_values("Créditos recebidos", ascending=False)

# Gráfico 1: Créditos recebidos por Ação (barra)
chart1 = alt.Chart(acao_sum).mark_bar().encode(
    x=alt.X("Ação Governo:N", sort="-y", title="Ação Governo"),
    y=alt.Y("Créditos recebidos:Q", title="R$"),
    tooltip=["Ação Governo", "Créditos recebidos", "Crédito disponível", "Empenhadas a liquidar", "Liquidadas a pagar", "Pagas"]
).properties(height=340)

st.altair_chart(chart1, use_container_width=True)

# Gráfico 2: Comparação (stack) dos componentes — sem dotação
stack = acao_sum.melt(
    id_vars=["Ação Governo"],
    value_vars=["Crédito disponível", "Empenhadas a liquidar", "Liquidadas a pagar", "Pagas"],
    var_name="Componente",
    value_name="Valor"
)

chart2 = alt.Chart(stack).mark_bar().encode(
    x=alt.X("Ação Governo:N", sort="-y", title="Ação Governo"),
    y=alt.Y("Valor:Q", stack=True, title="R$"),
    color=alt.Color("Componente:N", legend=alt.Legend(title="Componente")),
    tooltip=["Ação Governo", "Componente", "Valor"]
).properties(height=380)

st.altair_chart(chart2, use_container_width=True)

# (Opcional) Gráfico 3: Pagas vs Recebidos (% por ação)
acao_sum["% Pagas/Recebidos"] = np.where(
    acao_sum["Créditos recebidos"] > 0,
    (acao_sum["Pagas"] / acao_sum["Créditos recebidos"]) * 100.0,
    0.0
)

chart3 = alt.Chart(acao_sum).mark_bar().encode(
    x=alt.X("Ação Governo:N", sort="-y", title="Ação Governo"),
    y=alt.Y("% Pagas/Recebidos:Q", title="%"),
    tooltip=["Ação Governo", "% Pagas/Recebidos", "Pagas", "Créditos recebidos"]
).properties(height=300)

st.subheader("📊 Percentual — Pagas / Créditos Recebidos (por Ação)")
st.altair_chart(chart3, use_container_width=True)

# =========================
# Diagnóstico
# =========================
if debug:
    st.divider()
    st.subheader("Diagnóstico")

    st.write("UG Executora selecionada:", ug_sel_str)
    st.write("UGR(s) vinculadas:", ugrs_vinculadas)

    st.write("Linhas pool LOA:", len(df_loa_pool))
    st.write("Linhas LOA usadas:", len(df_loa))
    st.write("Linhas UG filtrada:", len(df_ug))

    st.write("Dotação (float):", dotacao_loa)
    st.write("Créditos recebidos (float):", creditos_recebidos)
    st.write("Pagas (float):", despesas_pagas)
    st.write("Saldo (float):", saldo)
