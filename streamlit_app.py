import io
import requests
import pandas as pd
import streamlit as st

st.title("TESTE — Ler CSV do Drive (corrigido)")

FILE_ID = "1s-lIrHxMZMRnCOayQeQ5ML0LpLbVTRNy"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

def baixar() -> bytes:
    r = requests.get(URL, timeout=120, allow_redirects=True)
    r.raise_for_status()
    return r.content

def ler_csv_corrigido(data: bytes) -> pd.DataFrame:
    # seu arquivo está com cara de cp1252 (Respons�vel)
    text = data.decode("cp1252", errors="replace")

    # 1) remove tudo antes do primeiro caractere " (onde começa o cabeçalho real)
    first_quote = text.find('"')
    if first_quote == -1:
        raise ValueError("Não achei aspas duplas no arquivo (não parece CSV).")
    text = text[first_quote:]

    # 2) remove as 2 últimas linhas (regra sua)
    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError("Poucas linhas após limpeza.")
    text = "\n".join(lines[:-2])

    # 3) lê com vírgula + aspas
    df = pd.read_csv(
        io.StringIO(text),
        sep=",",
        quotechar='"',
        engine="python",
        dtype=str,
        skipinitialspace=True,
    )

    # 4) remove colunas vazias (vindas de ",,")
    df = df.loc[:, [str(c).strip() != "" for c in df.columns]]
    df = df.dropna(axis=1, how="all")

    return df

if st.button("Baixar e ler"):
    data = baixar()
    st.write("Bytes:", len(data))
    st.code(data[:200].decode("cp1252", errors="replace"))

    df = ler_csv_corrigido(data)

    st.success(f"OK: {df.shape[0]} linhas × {df.shape[1]} colunas")
    st.write("Colunas:", list(df.columns))
    st.dataframe(df.head(20), use_container_width=True)
