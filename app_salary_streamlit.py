import json
import requests
import streamlit as st

# Titulo
st.title("Modelo de Predição de Salário")

# Inputs
st.write("Tempo do profissional na empresa")
tempo_na_empresa = st.slider("Meses", min_value=1, max_value=120, value=60, step=1)

st.write("Nível do profissional na empresa")
nivel_na_empresa = st.slider("Nível", min_value=1, max_value=10, value=5, step=1)

# Dados para a API
input_features = {
    'tempo_na_empresa': tempo_na_empresa,
    'nivel_na_empresa': nivel_na_empresa
}

# Botão para disparar api
if st.button('Estimar Salário'):
    res = requests.post(url="http://127.0.0.1:8000/predict", data=json.dumps(input_features))
    res_json = json.loads(res.text)
    salario_em_reais = round(res_json['salario_em_reais'][0], 2)
    st.subheader(f"O salário estimado é de R$ {salario_em_reais}")
