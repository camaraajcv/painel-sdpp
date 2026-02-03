# app.py
import io
import re
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Painel de Gastos (CSV OneDrive)", layout="wide")

# =========================
# Helpers
# =========================
def make_download_url(url: str) -> str:
    """
    Para links do OneDrive/Share (1drv.ms), normalmente funciona acrescentar download=1.
    """
    url = (url or "").strip()
    if not url:
        return url
    if "download=1" in url:
        return url
    joiner = "&" if "?" in url else "?"
    return url + f"{joiner}download=1"


def bytes_to_text(data: bytes) -> str:
    # tenta decodificações comuns de CSV BR
    for enc in ("utf-8-sig", "utf-8", "latin1", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    # último recurso
    return data.decode("utf-8", errors="replace")


def preprocess_csv_text(csv_text: str) -> str:
    """
    Regras:
    - remover a primeira linha
    - remover as duas últimas linhas
    - a segunda linha (do arquivo original) vira cabeçalho => após remover a 1ª, ela vira a 1ª
    """
    lines = csv_text.splitlines()
    if len(lines) < 4:
        raise ValueError("CSV muito curto — não dá pra remover 1ª e 2 últimas linhas com segurança.")

    kept = lines[1:-2]  # remove 1ª e 2 últimas
    if not kept:
        raise ValueError("Após remover linhas, não sobrou conteúdo no CSV.")
    return "\n".join(kept)


def read_table_from_bytes(data: bytes, source_name: str = "arquivo") -> pd.DataFrame:
    """
    Tenta interpretar bytes como CSV (com o pré-tratamento).
    Se não parecer CSV, tenta Excel.
    """
    # 1) tenta CSV
    text = bytes_to_text(data)

    # Heurística simples: se tiver muitos separadores comuns, trata como CSV
    looks_like_csv = any(sep in text[:5000] for sep in [",", ";", "\t"])

    if looks_like_csv:
        treated = preprocess_csv_text(text)
        # tenta separadores comuns (BR geralmente ';')
        for sep in (";", ",", "\t"):
            try:
                df = pd.read_csv(io.StringIO(treated), sep=sep, header=0, dtype=str)
                # se veio 1 coluna só, talvez separador errado
                if df.shape[1] == 1 and sep != "\t":
                    continue
                return df
            except Exception:
                continue
        raise ValueError(f"Não consegui ler o CSV de {source_name}. Verifique separador/estrutura.")

    # 2) tenta Excel
    try:
        df = pd.read_excel(io.BytesIO(data), dtype=str)
        return df
    except Exception as e:
        raise ValueError(f"Não consegui interpretar {source_name} como CSV nem como Excel. Detalhe: {e}")


import requests
import re

def baixar_onedrive(url: str) -> bytes:
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    # Abre a página do compartilhamento
    r = session.get(url, headers=headers, timeout=240)
    r.raise_for_status()

    # Procura o link real de download dentro do HTML
    m = re.search(r'"downloadUrl":"(https:[^"]+)"', r.text)
    if not m:
        raise RuntimeError(
            "Não consegui extrair o downloadUrl. "
            "Verifique se o link está como 'Qualquer pessoa com o link'."
        )

    download_url = m.group(1).encode("utf-8").decode("unicode_escape")

    # Baixa o arquivo real
    file = session.get(download_url, headers=headers, timeout=240)
    file.raise_for_status()

    return file.content
def ler_excel_tratado(data: bytes) -> pd.DataFrame:
    # pula a 1ª linha -> a 2ª vira cabeçalho
    df = pd.read_excel(io.BytesIO(data), skiprows=1, dtype=str)

    # remove as 2 últimas linhas
    if len(df) >= 2:
        df = df.iloc[:-2].copy()

    return df
# =========================
# UI
# =========================
st.title("📊 Painel de Gastos — Leitura diária do OneDrive")
url = "https://onedrive.live.com/edit?cid=79e2ea2aaf97ea6b&id=79E2EA2AAF97EA6B!s69abd6e991b14e84af28cb23a5f2569c&resid=79E2EA2AAF97EA6B!s69abd6e991b14e84af28cb23a5f2569c&ithint=file%2Cxlsx&embed=1&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3gvYy83OWUyZWEyYWFmOTdlYTZiL0lRVHAxcXRwc1pHRVRxOG95eU9sOGxhY0Fma3dhZTlBM3IxZEZIeVk3Nm1FLTZN&wdo=2"

try:
    data = baixar_onedrive(url)
    df = ler_excel_tratado(data)

except Exception as e:
    st.error(f"Erro ao ler arquivo do OneDrive: {e}")
    st.stop()

st.success(f"Arquivo carregado: {df.shape[0]} linhas")

with st.sidebar:
    st.header("Fonte de dados")

    fonte = st.radio("Escolha a fonte", ["OneDrive (URL)", "Arquivo local (caminho)", "Upload (manual)"], index=0)

    url_default = "https://onedrive.live.com/edit?cid=79e2ea2aaf97ea6b&id=79E2EA2AAF97EA6B!s69abd6e991b14e84af28cb23a5f2569c&resid=79E2EA2AAF97EA6B!s69abd6e991b14e84af28cb23a5f2569c&ithint=file%2Cxlsx&embed=1&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3gvYy83OWUyZWEyYWFmOTdlYTZiL0lRVHAxcXRwc1pHRVRxOG95eU9sOGxhY0Fma3dhZTlBM3IxZEZIeVk3Nm1FLTZN&wdo=2"
    url = st.text_input("Link do OneDrive", value=url_default) if fonte == "OneDrive (URL)" else ""

    path = st.text_input(
        "Caminho do arquivo (no servidor/PC onde o Streamlit roda)",
        value=r"/mnt/data/CONTROLE COMPLETO EXECUÇAO ORÇAMENTÁRIA (1).csv",
    ) if fonte == "Arquivo local (caminho)" else ""

    uploaded = st.file_uploader("Envie o arquivo", type=["csv", "xlsx", "xls"]) if fonte == "Upload (manual)" else None

    col1, col2 = st.columns(2)
    with col1:
        btn_carregar = st.button("🔄 Carregar agora", use_container_width=True)
    with col2:
        mostrar_tipos = st.checkbox("Mostrar tipos/diagnóstico", value=False)

st.caption("Regras aplicadas no CSV: remove 1ª linha e as 2 últimas; usa a 2ª linha como cabeçalho.")

# Carrega sempre que apertar o botão (sem cache, pra garantir atualização diária ao abrir)
df = None
erro = None

if btn_carregar:
    try:
        if fonte == "OneDrive (URL)":
            if not url.strip():
                raise ValueError("Informe a URL do OneDrive.")
            data = download_onedrive_file(url)
            df = read_table_from_bytes(data, source_name="OneDrive")

        elif fonte == "Arquivo local (caminho)":
            if not path.strip():
                raise ValueError("Informe o caminho do arquivo.")
            with open(path, "rb") as f:
                data = f.read()
            df = read_table_from_bytes(data, source_name="arquivo local")

        else:  # Upload
            if uploaded is None:
                raise ValueError("Envie um arquivo para continuar.")
            df = read_table_from_bytes(uploaded.getvalue(), source_name="upload")

    except Exception as e:
        erro = str(e)

if erro:
    st.error(erro)

if df is not None:
    st.success(f"Dados carregados: {df.shape[0]} linhas × {df.shape[1]} colunas")

    # limpeza básica: remove colunas vazias (às vezes aparecem do CSV)
    df = df.dropna(axis=1, how="all")

    # Diagnóstico opcional
    if mostrar_tipos:
        st.write("Prévia das colunas:", list(df.columns))
        st.write(df.head(10))

    # Exemplo de “DataFrame pronto”: padroniza nomes (opcional)
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]

    # Área do painel (placeholder até você passar as regras)
    st.subheader("Visão geral (placeholder)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas", f"{len(df):,}".replace(",", "."))
    c2.metric("Colunas", f"{df.shape[1]}")
    c3.metric("Nulos (total)", f"{int(df.isna().sum().sum()):,}".replace(",", "."))
    c4.metric("Duplicadas", f"{int(df.duplicated().sum()):,}".replace(",", "."))

    st.divider()
    st.subheader("Tabela (base)")
    st.dataframe(df, use_container_width=True, height=520)

    st.info("Agora me passe a **regra do painel de gastos** (quais colunas usar, filtros, agregações e KPIs) que eu já encaixo aqui.")
else:
    st.warning("Clique em **Carregar agora** para buscar o arquivo e montar o DataFrame.")
    import numpy as np

def find_col(df: pd.DataFrame, *candidates: str):
    cols_norm = {c: re.sub(r"\s+", " ", str(c)).strip().lower() for c in df.columns}
    for cand in candidates:
        cand = cand.lower().strip()
        # match por "contém"
        for real, norm in cols_norm.items():
            if cand in norm:
                return real
    return None

def brl_to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    # remove "R$" e espaços
    s = s.str.replace("R$", "", regex=False).str.replace("\u00a0", " ", regex=False).str.replace(" ", "", regex=False)
    # troca separadores BR -> padrão
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

# =========================
# Após df carregado e df.columns normalizado
# =========================
COL_UG = find_col(df, "ug executora", "ug", "unidade gestora")
COL_DOT = find_col(df, "dotacao atualizada", "dotação atualizada", "dotacao", "dotação")

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
    st.stop()

# converter moeda
for c in [COL_DOT, COL_CRED, COL_ALIQ, COL_LIQP, COL_PAGO]:
    df[c] = brl_to_float(df[c])

# Sidebar: UG obrigatória
with st.sidebar:
    st.header("Filtro obrigatório")
    ugs = sorted(df[COL_UG].dropna().astype(str).unique().tolist())
    ug_sel = st.selectbox("UG Executora", options=ugs, index=None, placeholder="Selecione a UG...")

if not ug_sel:
    st.info("Selecione uma **UG Executora** para gerar o painel.")
    st.stop()

df_ug = df[df[COL_UG].astype(str) == str(ug_sel)].copy()

# métricas principais
dotacao_loa = df_ug[COL_DOT].sum(skipna=True)
creditos_recebidos = (
    df_ug[COL_CRED].sum(skipna=True)
    + df_ug[COL_ALIQ].sum(skipna=True)
    + df_ug[COL_LIQP].sum(skipna=True)
    + df_ug[COL_PAGO].sum(skipna=True)
)

# (opcional) execução: quanto já virou pagamento
empenhos_pagos = df_ug[COL_PAGO].sum(skipna=True)

st.subheader(f"📌 Painel — UG Executora: {ug_sel}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dotação Atualizada (LOA)", f"R$ {dotacao_loa:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
c2.metric("Créditos Recebidos (soma)", f"R$ {creditos_recebidos:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
c3.metric("Empenhos pagos", f"R$ {empenhos_pagos:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
# saldo simples (se fizer sentido pra você)
saldo = creditos_recebidos - empenhos_pagos
c4.metric("Saldo (Recebidos - Pagos)", f"R$ {saldo:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

st.divider()

# Resumo por alguma dimensão (se existir)
COL_ND  = find_col(df_ug, "nd", "natureza da despesa", "natureza despesa")
COL_GND = find_col(df_ug, "gnd", "grupo natureza")
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

