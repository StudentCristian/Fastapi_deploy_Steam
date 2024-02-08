from fastapi import FastAPI
from funciones import developer, user_data, User_For_Genre, best_developer_year, desarrollador_reviews_analysis, recomendacion_juego, recomendacion_usuario
import pandas as pd
import numpy as np
from fastapi.encoders import jsonable_encoder
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split, GridSearchCV
import pickle 
 

app = FastAPI()


# fUNCION 1
@app.get("/get_developer/{developer}")
def get_developer(desarrollador: str):
    funcion_uno = developer(desarrollador)
    return funcion_uno

# fUNCION 2
@app.get("/userdata/{user_id}")
def userdata(user_id: str):
    funcion_dos = user_data(user_id)
    return funcion_dos

# fUNCION 3
@app.get("/usergenre/{genres}")
def usergenre(genero: str):
    funcion_tres = User_For_Genre(genero)
    return funcion_tres

# fUNCION 4
@app.get("/bestdeveloper/{year}")
def bestdeveloper(year: int):
    funcion_cuatro = best_developer_year(year)
    return funcion_cuatro

    # fUNCION 5
@app.get("/developerreviews/{developer}")
def developerreviews(desarrolladora: str):
    funcion_cinco = desarrollador_reviews_analysis(desarrolladora)
    return funcion_cinco

# Sistemas de recomendación item-item recomendacion_juego
# Ruta para obtener recomendaciones de juegos: 
@app.get("/recomendacion_juego/{item_id}")
def obtener_recomendacion_juego(item_id: int):
    # Llama a la función lógica de recomendación de juego
    recommended_games = recomendacion_juego(item_id)
    return recommended_games


# Sistemas de recomendación user-user recomendacion_usuario
# Ruta para obtener recomendaciones de usuario:
@app.get("/recomendacion_usuario/{user_id}")
def obtener_recomendacion_usuario(user_id: str):
    # Llama a la función lógica de recomendación de usuario
    recommended_games_user = recomendacion_usuario(user_id)
    return recommended_games_user




















# app.mount("/ui", StaticFiles(directory=static_path), name="ui")

# @app.get('/')
# def root():
#     html_path = join(static_path, "index.html")
#     return FileResponse(html_path)

# current_dir = dirname(abspath(__file__))
# static_path = join(current_dir, "static")




# class Body(BaseModel):
#     length: Union[int, None] = 20



# @app.post('/generate')
# def generate(body: Body):
#     """
#     Generate a pseudo-random token ID of twenty characters by default. Example POST request body:

#     {
#         "length": 20
#     }
#     """
#     string = base64.b64encode(os.urandom(64))[:body.length].decode('utf-8')
#     return {'token': string}