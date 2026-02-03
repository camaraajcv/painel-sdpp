import io
import re
import requests
import pandas as pd
import streamlit as st

st.title("TESTE — Leitura CSV Google Drive (robusta)")

FILE_ID = "1s-lIrHxMZMRnCOayQeQ5ML0LpLbVTRNy"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

def baixar_drive(url: str) -> bytes:
    s = requests.Session()
    r = s.get(url, timeout=120, allow_redirects=True)
    r.raise_for_status()
    return r.content

def decode_cp1252(data: bytes) -> str:
    return data.decode("cp1252", errors="replace")

def tratar_texto(text: str) -> str:
    lines = text.splitlines()

    # remove linhas iniciais "lixo" até achar uma linha que comece com aspas (cabeçalho)
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith('"'):
            start = i
            break
    lines = lines[start:]

    # aplica sua regra: remove 1ª e as 2 últimas
    if len(lines) < 4:
        raise ValueError("Poucas linhas para aplicar o corte.")
    lines = lines[1:-2]
    return "\n".join(lines)

if st.button("Baixar e ler"):
    data = baixar_drive(URL)
    text = decode_cp1252(data)
    treated = tratar_texto(text)

    # leitura robusta: pula linhas ruins
    df = pd.read_csv(
        io.StringIO(treated),
        sep=",",
        quotechar='"',
        engine="python",
        dtype=str,
        on_bad_lines="skip",
        skipinitialspace=True,
    )

    # remove colunas vazias/unnamed
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed", case=False)]
    df = df.dropna(axis=1, how="all")

    st.success(f"OK: {df.shape[0]} linhas × {df.shape[1]} colunas")
    st.write("Colunas:", list(df.columns))
    st.dataframe(df.head(30), use_container_width=True)

    # diagnóstico: quantas linhas foram puladas (aprox)
    st.caption("Obs.: linhas 'quebradas' foram ignoradas (on_bad_lines='skip').")
