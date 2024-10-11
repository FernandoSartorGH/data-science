from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd

# Criar instância do FastAPI
app = FastAPI()


# Criar classe com dados do requesty body para a API
class RequestBody(BaseModel):
    grupo_sanguineo: str
    fumante: str
    nivel_de_atividade: str
    idade: int
    peso: int
    altura: int


# Carregar o modelo
modelo = joblib.load('./modelo_cholesterol.pickle')


# Prediction
@app.post('/predict')
def predict(data: RequestBody):
    input_feature = {
        'grupo_sanguineo': data.grupo_sanguineo,
        'fumante': data.fumante,
        'nivel_de_atividade': data.nivel_de_atividade,
        'idade': data.idade,
        'peso': data.peso,
        'altura': data.altura
    }

    sample_data = pd.DataFrame(input_feature, index=[1])
    y_pred = modelo.predict(sample_data)[0].astype(float)

    return {'cholesterol': list(y_pred)}
