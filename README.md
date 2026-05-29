Projeto de detecção de fraude:

API Flask/FastAPI
Modelo de ML
MongoDB

O sistema deve ser capaz de receber transações, extrair atributos, calcular risco de fraude, 
salvar no banco e retornar “Fraude” ou “Legítima”.

Passos para execução do projeto:
    1.pip install -r requirements.txt

    2.python merge_csv.py

    3.python train.py

    4.python -m uvicorn api:app --reload

    5.Teste no Swagger:
        http://127.0.0.1:8000/docs