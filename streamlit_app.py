import io
import re
import requests
import pandas as pd
import streamlit as st

st.title("TESTE — Leitura CSV (corrigido)")

FILE_ID = "1s-lIrHxMZMRnCOayQeQ5ML0LpLbVTRNy"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

def baixar_drive(url: str) -> bytes:
    s = requests.Session()
    r = s.get(url, timeout=120, allow_redirects=True)
    r.raise_for_status()
    return r.content

def decode_cp1252(data: bytes) -> str:
    return data.decode("cp1252", errors="replace")

def limpar_linhas_iniciais(text: str) -> str:
    # remove quaisquer linhas iniciais que não pareçam parte do CSV
    lines = text.splitlines()
    # acha a primeira linha "de verdade" (normalmente começa com aspas)
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith('"'):
            start = i
            break
    lines = lines[start:]
    # aplica sua regra: remove 1ª linha e 2 últimas
    if len(lines) < 4:
        raise ValueError(f"Poucas linhas após limpeza ({len(lines)})")
    lines = lines[1:-2]
    return "\n".join(lines)

if st.button("Baixar e ler"):
    data = baixar_drive(URL)
    st.write("Bytes:", len(data))
    st.code(data[:200].decode("cp1252", errors="replace"))

    text = decode_cp1252(data)
    treated = limpar_linhas_iniciais(text)

    # leitura robusta de CSV com vírgula e aspas
    df = pd.read_csv(
        io.StringIO(treated),
        sep=",",
        header=0,
        dtype=str,
        engine="python",
        quotechar='"',
        skipinitialspace=True,
    )

    st.success(f"OK: {df.shape[0]} linhas × {df.shape[1]} colunas")
    st.write("Colunas:", list(df.columns))
    st.dataframe(df.head(20), use_container_width=True)
