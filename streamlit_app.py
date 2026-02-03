import io
import numpy as np
import pandas as pd
import requests
import streamlit as st

FILE_ID = "1s-lIrHxMZMRnCOayQeQ5ML0LpLbVTRNy"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

def brl_to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace("R$", "", regex=False).str.replace("\u00a0", " ", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

@st.cache_data(ttl=60*15, show_spinner=True)
def carregar_df():
    r = requests.get(URL, timeout=120, allow_redirects=True)
    r.raise_for_status()
    data = r.content

    # CSV UTF-16 (tem BOM ÿþ) + remover 2 primeiras linhas e 2 últimas
    df = pd.read_csv(
        io.BytesIO(data),
        encoding="utf-16",
        sep=",",
        quotechar='"',
        engine="python",
        skiprows=2,      # ✅ remove as 2 primeiras linhas
        skipfooter=2,    # ✅ remove as 2 últimas linhas
        header=0,
        dtype=str,
        skipinitialspace=True
    )

    # remove colunas vazias/Unnamed
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed", case=False)]
    df = df.dropna(axis=1, how="all")

    # remove "Item Informação" se existir
    for col in df.columns:
        if str(col).strip().lower().replace("ç","c") in ["item informacao", "item informação"]:
            df = df.drop(columns=[col])
            break

    # renomear colunas por posição (Excel 1-based -> pandas 0-based)
    rename_by_pos = {
        12: "DOTACAO_ATUALIZADA",      # coluna 13
        18: "CREDITO_DISPONIVEL",      # coluna 19
        29: "EMPENHADAS_A_LIQUIDAR",   # coluna 30
        31: "LIQUIDADAS_A_PAGAR",      # coluna 32
        33: "PAGAS",                   # coluna 34
    }

    cols = list(df.columns)
    for pos, new_name in rename_by_pos.items():
        if pos < len(cols):
            df = df.rename(columns={cols[pos]: new_name})

    # converter essas colunas pra número
    for c in ["DOTACAO_ATUALIZADA","CREDITO_DISPONIVEL","EMPENHADAS_A_LIQUIDAR","LIQUIDADAS_A_PAGAR","PAGAS"]:
        if c in df.columns:
            df[c] = brl_to_float(df[c])

    return df

# ==== teste rápido (pra você validar) ====
df = carregar_df()
st.success(f"OK: {df.shape[0]} linhas × {df.shape[1]} colunas")
st.write("Colunas:", list(df.columns))
st.dataframe(df.head(20), use_container_width=True)
