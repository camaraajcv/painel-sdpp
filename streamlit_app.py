import io
import csv
import requests
import pandas as pd
import streamlit as st

st.title("TESTE — Leitura CSV Google Drive (tolerante)")

FILE_ID = "1s-lIrHxMZMRnCOayQeQ5ML0LpLbVTRNy"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

def baixar_drive(url: str) -> bytes:
    s = requests.Session()
    r = s.get(url, timeout=120, allow_redirects=True)
    r.raise_for_status()
    return r.content

def decode_cp1252(data: bytes) -> str:
    # pelo seu preview (Respons�vel), isso é o mais provável
    return data.decode("cp1252", errors="replace")

def preprocess_lines(text: str) -> list[str]:
    lines = text.splitlines()
    if len(lines) < 4:
        raise ValueError(f"Arquivo muito curto: {len(lines)} linhas.")
    # sua regra: remove 1ª linha e as 2 últimas; a 2ª vira cabeçalho
    return lines[1:-2]

def parse_csv_tolerante(lines: list[str]) -> pd.DataFrame:
    reader = csv.reader(lines, delimiter=",", quotechar='"', skipinitialspace=True)
    rows = list(reader)
    if not rows:
        raise ValueError("Nenhuma linha após pré-processamento.")

    header = rows[0]
    n = len(header)

    fixed = []
    for r in rows[1:]:
        if len(r) < n:
            r = r + [""] * (n - len(r))           # completa faltando
        elif len(r) > n:
            # junta excedentes no último campo para não perder dado
            r = r[:n-1] + [",".join(r[n-1:])]
        fixed.append(r)

    df = pd.DataFrame(fixed, columns=header)

    # remove colunas vazias / sem nome (vindas de ",,")
    df = df.loc[:, [c.strip() != "" for c in df.columns]]
    return df

if st.button("Baixar e ler"):
    data = baixar_drive(URL)
    st.write("Bytes:", len(data))

    text = decode_cp1252(data)
    lines = preprocess_lines(text)

    # diagnóstico: mostra cabeçalho bruto
    st.code(lines[0][:300])

    df = parse_csv_tolerante(lines)

    st.success(f"OK: {df.shape[0]} linhas × {df.shape[1]} colunas")
    st.write("Colunas:", list(df.columns))
    st.dataframe(df.head(30), use_container_width=True)
