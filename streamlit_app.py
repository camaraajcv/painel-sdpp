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


def download_onedrive_file(url: str) -> bytes:
    dl = make_download_url(url)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*",
    }
    r = requests.get(dl, headers=headers, timeout=240, allow_redirects=True)
    r.raise_for_status()
    return r.content


# =========================
# UI
# =========================
st.title("📊 Painel de Gastos — Leitura diária do OneDrive")

with st.sidebar:
    st.header("Fonte de dados")

    fonte = st.radio("Escolha a fonte", ["OneDrive (URL)", "Arquivo local (caminho)", "Upload (manual)"], index=0)

    url_default = "https://1drv.ms/x/c/79e2ea2aaf97ea6b/IQDp1qtpsZGETq8oyyOl8lacAYtMfzw3hr9M5F-mEYGVKv4?e=li4aJ2"
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
