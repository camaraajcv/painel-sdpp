# app.py
import io
import re
import numpy as np
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Painel de Gastos — OneDrive", layout="wide")

# =========================
# Utils
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
# OneDrive download (public only)
# =========================
def baixar_onedrive_publico(url: str) -> bytes:
    """
    Tenta baixar via link público. Se o OneDrive exigir login/sessão no Cloud, vai dar 403.
    """
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    r = session.get(url, headers=headers, timeout=240, allow_redirects=True)
    r.raise_for_status()

    # tenta extrair downloadUrl do HTML (quando disponível)
    m = re.search(r'"downloadUrl":"(https:[^"]+)"', r.text)
    if not m:
        raise RuntimeError(
            "Não consegui extrair downloadUrl do OneDrive. "
            "Mesmo com link 'qualquer pessoa', alguns links exigem autenticação e dão 403 no Streamlit Cloud."
        )

    download_url = m.group(1).encode("utf-8").decode("unicode_escape")
    f = session.get(download_url, headers=headers, timeout=240, allow_redirects=True)
    f.raise_for_status()
    return f.content

# =========================
# Read Excel/CSV with your rules
# =========================
def preprocess_csv_text(csv_text: str) -> str:
    lines = csv_text.splitlines()
    if len(lines) < 4:
        raise ValueError("CSV muito curto — não dá pra remover 1ª e 2 últimas linhas com segurança.")
    kept = lines[1:-2]  # remove 1ª e 2 últimas
    if not kept:
        raise ValueError("Após remover linhas, não sobrou conteúdo no CSV.")
    return "\n".join(kept)

def bytes_to_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin1", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")

def ler_excel_tratado(data: bytes) -> pd.DataFrame:
    # regra: remove 1ª linha; 2ª vira cabeçalho; remove 2 últimas linhas
    df = pd.read_excel(io.BytesIO(data), skiprows=1, dtype=str)
    if len(df) >= 2:
        df = df.iloc[:-2].copy()
    return df

def read_table_from_bytes(data: bytes) -> pd.DataFrame:
    # tenta detectar CSV rápido
    text = bytes_to_text(data)
    looks_like_csv = any(sep in text[:5000] for sep in [",", ";", "\t"])

    if looks_like_csv:
        treated = preprocess_csv_text(text)
        for sep in (";", ",", "\t"):
            try:
                df = pd.read_csv(io.StringIO(treated), sep=sep, header=0, dtype=str)
                if df.shape[1] == 1 and sep != "\t":
                    continue
                return df
            except Exception:
                continue
        raise ValueError("Não consegui ler o CSV (separador/estrutura).")

    # caso contrário, Excel
    return ler_excel_tratado(data)

# =========================
# UI
# =========================
st.title("📊 Painel de Gastos — Leitura diária (Streamlit Cloud)")

with st.sidebar:
    st.header("Fonte de dados")
    fonte = st.radio("Escolha a fonte", ["OneDrive (URL)", "Upload (manual)", "Arquivo local (dev)"], index=0)

    url_default = "https://1drv.ms/x/c/79e2ea2aaf97ea6b/IQDp1qtpsZGETq8oyyOl8lacAYtMfzw3hr9M5F-mEYGVKv4?e=jICJ9i"
    url = st.text_input("Link do OneDrive", value=url_default) if fonte == "OneDrive (URL)" else ""

    uploaded = st.file_uploader("Envie o arquivo", type=["csv", "xlsx", "xls"]) if fonte == "Upload (manual)" else None

    path = st.text_input(
        "Caminho local (somente se rodar localmente)",
        value="",
    ) if fonte == "Arquivo local (dev)" else ""

    btn_carregar = st.button("🔄 Carregar agora", use_container_width=True)
    mostrar_diag = st.checkbox("Mostrar diagnóstico", value=False)

st.caption("Regras: remover 1ª linha e as 2 últimas; a 2ª linha vira cabeçalho (para Excel: skiprows=1 e depois corta 2 últimas).")

df = None

if btn_carregar:
    try:
        if fonte == "OneDrive (URL)":
            if not url.strip():
                raise ValueError("Informe a URL do OneDrive.")
            data = baixar_onedrive_publico(url)
            df = read_table_from_bytes(data)

        elif fonte == "Upload (manual)":
            if uploaded is None:
                raise ValueError("Envie um arquivo para continuar.")
            df = read_table_from_bytes(uploaded.getvalue())

        else:  # Arquivo local (dev)
            if not path.strip():
                raise ValueError("Informe o caminho do arquivo.")
            with open(path, "rb") as f:
                data = f.read()
            df = read_table_from_bytes(data)

        df = normalize_cols(df)

    except requests.HTTPError as e:
        st.error(f"Erro HTTP ao acessar OneDrive: {e}\n\n"
                 f"➡️ Se for 403 no Streamlit Cloud, o link ainda exige autenticação. "
                 f"Nesse caso, só resolve com Microsoft Graph (OAuth) ou hospedando o arquivo em um link realmente público.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar/ler o arquivo: {e}")
        st.stop()

if df is None:
    st.info("Clique em **Carregar agora** para buscar o arquivo e montar o painel.")
    st.stop()

st.success(f"Arquivo carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")

if mostrar_diag:
    st.write("Colunas encontradas:", list(df.columns))
    st.dataframe(df.head(20), use_container_width=True)

# =========================
# Painel por UG
# =========================
COL_UG   = find_col(df, "ug executora", "ug execut", "ug", "unidade gestora")
COL_DOT  = find_col(df, "dotacao atualizada", "dotação atualizada", "dotacao", "dotação")

COL_CRED = find_col(df, "credito disponivel", "crédito disponível", "credito disponível")
COL_ALIQ = find_col(df, "empenhos a liquidar", "a liquidar")
COL_LIQP = find_col(df, "empenhos liquidados a pagar", "liquidados a pagar")
COL_PAGO = find_col(df, "empenhos pagos", "pagos")

missing = [n for n, c in [
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

# converte moeda
for c in [COL_DOT, COL_CRED, COL_ALIQ, COL_LIQP, COL_PAGO]:
    df[c] = brl_to_float(df[c])

# seletor obrigatório de UG
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

# resumo por ND/GND/Elemento (se existir)
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
    st.subheader("Resumo por classificação (ND/GND/Elemento)")
    st.dataframe(resumo, use_container_width=True, height=520)
else:
    st.subheader("Dados filtrados da UG (sem colunas ND/GND/Elemento detectadas)")
    st.dataframe(df_ug, use_container_width=True, height=520)
