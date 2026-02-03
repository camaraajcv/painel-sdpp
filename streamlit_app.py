import io
import re
import requests
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
st.title("TESTE — Leitura CSV Google Drive")

FILE_ID = "1s-lIrHxMZMRnCOayQeQ5ML0LpLbVTRNy"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

def bytes_to_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin1", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")

def preprocess_csv_text(csv_text: str) -> str:
    lines = csv_text.splitlines()
    if len(lines) < 4:
        raise ValueError(f"Poucas linhas ({len(lines)}). Não dá pra remover 1ª e 2 últimas.")
    kept = lines[1:-2]  # remove 1ª e 2 últimas
    return "\n".join(kept)

def baixar_drive(url: str) -> tuple[bytes, str]:
    s = requests.Session()
    r = s.get(url, timeout=120, allow_redirects=True)
    r.raise_for_status()

    ctype = (r.headers.get("Content-Type") or "")
    data = r.content

    # Se veio HTML, tenta token confirm=
    if ("text/html" in ctype.lower()) or data[:20].lstrip().startswith(b"<"):
        html = r.text
        m = re.search(r'confirm=([0-9A-Za-z_]+)', html)
        if m:
            confirm = m.group(1)
            r2 = s.get(url + f"&confirm={confirm}", timeout=120, allow_redirects=True)
            r2.raise_for_status()
            ctype = (r2.headers.get("Content-Type") or "")
            data = r2.content

    return data, ctype

if st.button("Baixar e testar leitura"):
    data, ctype = baixar_drive(URL)

    st.write("Content-Type recebido:", ctype)
    st.write("Tamanho (bytes):", len(data))

    preview = data[:300]
    st.code(preview.decode("utf-8", errors="replace"))

    # Se ainda for HTML, para aqui com mensagem clara
    if preview.lstrip().startswith(b"<"):
        st.error("⚠️ Veio HTML do Google Drive (não veio o CSV). Ajuste permissão: 'Qualquer pessoa com o link' (Visualizador).")
        st.stop()

    text = bytes_to_text(data)
    treated = preprocess_csv_text(text)

    # tenta separadores
    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(io.StringIO(treated), sep=sep, header=0, dtype=str)
            if df.shape[1] == 1 and sep != "\t":
                continue
            st.success(f"✅ Leu CSV com separador '{sep}' — {df.shape[0]} linhas × {df.shape[1]} colunas")
            st.write("Colunas:", list(df.columns))
            st.dataframe(df.head(20), use_container_width=True)
            st.stop()
        except Exception as e:
            last_err = e

    st.error(f"Não consegui ler como CSV. Último erro: {last_err}")
