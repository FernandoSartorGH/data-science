from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib
import pandas as pd

# Instâncisdo FastApi
app = FastAPI()


# Class request body com tipos esperados
class RequestBody(BaseModel):
    tempo_na_empresa: int
    nivel_na_empresa: int


# Carrega o modelo
poly_model = joblib.load('./modelo_salary.pickle')
@app.post('/predict')
def predict(data : RequestBody):
    input_data = {
        'tempo_na_empresa': data.tempo_na_empresa,
        'nivel_na_empresa': data.nivel_na_empresa
    }
    sample_data = pd.DataFrame(input_data, index=[1])

    # Predict
    y_pred = poly_model.predict(sample_data)[0].astype(float)

    return {'salario_em_reais': list(y_pred)}
