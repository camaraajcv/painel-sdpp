import io
import pandas as pd
import requests
import streamlit as st

st.title("TESTE — Ler CSV do Drive (UTF-16)")

FILE_ID = "1s-lIrHxMZMRnCOayQeQ5ML0LpLbVTRNy"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

if st.button("Baixar e ler"):
    r = requests.get(URL, timeout=120, allow_redirects=True)
    r.raise_for_status()
    data = r.content

    st.write("Bytes:", len(data))
    st.code(data[:120].decode("utf-16", errors="replace"))

    # ✅ leitura correta: UTF-16 + pula 1ª linha "xxxxxxx" + remove 2 últimas linhas
    df = pd.read_csv(
        io.BytesIO(data),
        encoding="utf-16",
        sep=",",
        quotechar='"',
        engine="python",     # necessário para skipfooter
        skiprows=1,          # remove 1ª linha (xxxxxxx)
        skipfooter=2,        # remove 2 últimas linhas
        header=0,            # a linha seguinte vira cabeçalho
        dtype=str,
        skipinitialspace=True
    )

    # remove colunas vazias ("Unnamed") e colunas totalmente vazias
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed", case=False)]
    df = df.dropna(axis=1, how="all")

    st.success(f"OK: {df.shape[0]} linhas × {df.shape[1]} colunas")
    st.write("Colunas:", list(df.columns))
    st.dataframe(df.head(20), use_container_width=True)
