import pandas as pd
from fastapi import FastAPI
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split, GridSearchCV
import pickle 
import numpy as np

# Cargar el modelo del archivo .pkl
with open('/workspaces/codespaces-project-template-py/data/user_item_model.pkl', 'rb') as archivo:
    modelo = pickle.load(archivo)


# FUNCIÓN 1 - DESARROLLADORES CON JUEGOS GRATUITOS 
def developer(desarrollador):
    """
    Esta función toma como entrada el nombre de un desarrollador y devuelve un diccionario que contiene información sobre los juegos lanzados por ese desarrollador. 
    La información incluye el número total de juegos lanzados por año, el número de juegos gratuitos lanzados por año y el porcentaje de juegos gratuitos lanzados por año.
    
    Parámetros:
    desarrollador (str): El nombre del desarrollador.
    
    Devuelve:
    dict_developer (dict): Un diccionario que contiene información sobre los juegos lanzados por el desarrollador.

    Ejemplo:
    >>> developer("Nombre del Desarrollador")
    [{'Año': 2010, 'Cantidad de artículos': 5, 'Contenidos Gratis': 2, 'Porcentaje Gratis': '40%'},
     {'Año': 2011, 'Cantidad de artículos': 10, 'Contenidos Gratis': 5, 'Porcentaje Gratis': '50%'},
     {'Año': 'Desconocido', 'Cantidad de artículos': 1, 'Contenidos Gratis': 0, 'Porcentaje Gratis': '0%'}]

    En caso de que el desarrollador no exista, devuelve un diccionario con un mensaje de error.

    Ejemplo:
    >>> developer("Desarrollador Inexistente")
    {
        "detail": [
            {
                "loc": ["string", 0],
                "msg": "El desarrollador no existe",
                "type": "value_error"
            }
        ]
    }
    """
    # Abrimos el archivo parquet en un dataframe
    funcion_1=pd.read_parquet("/workspaces/codespaces-project-template-py/data/games.parquet")
    # Filtrar el dataframe por el desarrollador dado
    df_dev = funcion_1[funcion_1['developer'] == desarrollador]

    # Si el dataframe está vacío, lanzar una excepción
    if df_dev.empty:
        return {
            "detail": [
                {
                    "loc": ["string", 0],
                    "msg": "El desarrollador no existe",
                    "type": "value_error"
                }
            ]
        }
    
    # Agrupar por año y contar el número total de juegos y el número de juegos gratuitos
    juegos_por_año = df_dev.groupby('release_date').size()
    juegos_gratis_por_año = df_dev[df_dev['price'] == 0.0].groupby('release_date').size()
    
    # Asegurarse de que juegos_gratis_por_año tenga la misma longitud que juegos_por_año
    juegos_gratis_por_año = juegos_gratis_por_año.reindex(juegos_por_año.index, fill_value=0)
    
    # Calcular el porcentaje de juegos gratis y redondearlo al entero más cercano
    porcentaje_gratis = ((juegos_gratis_por_año / juegos_por_año) * 100).round().astype(int)
    
    # Asignar nombres a las series
    juegos_por_año.name = "Cantidad de artículos"
    juegos_gratis_por_año.name = "Contenidos Gratis"
    porcentaje_gratis.name = "Porcentaje Gratis"
    
    # Unir las series en un DataFrame
    tabla = pd.concat([juegos_por_año, juegos_gratis_por_año, porcentaje_gratis], axis=1).reset_index()
    tabla.columns = ['Año', 'Cantidad de artículos', 'Contenidos Gratis', 'Porcentaje Gratis']
    
    # Reemplazar 0 con 'Desconocido' en la columna 'Año'
    tabla['Año'] = tabla['Año'].replace(0, 'Desconocido')
    
    # Agregar el signo de porcentaje a 'Porcentaje Gratis'
    tabla['Porcentaje Gratis'] = tabla['Porcentaje Gratis'].apply(lambda x: f"{x}%")
    
    # Convertir el DataFrame a un diccionario
    dict_developer = tabla.to_dict(orient='records')
    
    return dict_developer


# FUNCIÓN 2 - CANTIDAD DE DINERO GASTADO POR USUARIO (modificar el porcentaje que aproxime)
def user_data(user_id):
    """
    Esta función devuelve la cantidad de dinero gastado por el usuario, el porcentaje de recomendación
    basado en las revisiones (recommend) y la cantidad de artículos.

    Args:
    - User_id (str): El ID del usuario para el cual se desea obtener la información.

    Returns:
    - dict: Un diccionario que contiene la información del usuario, incluyendo:
        - "Usuario": El ID del usuario.
        - "Dinero gastado": La cantidad de dinero gastado por el usuario en USD.
        - "% de recomendación": El porcentaje de recomendación basado en las revisiones.
        - "cantidad de artículos": La cantidad de artículos únicos que el usuario ha revisado.

    Ejemplo:
    >>> user_data("ID_Usuario")
    {
        'Usuario': 'ID_Usuario',
        'Dinero gastado': '35.0 USD',
        '% de recomendación': '60%',
        'cantidad de artículos': 5
    }

    En caso de que el usuario no exista, devuelve un diccionario con un mensaje de error.

    Ejemplo:
    >>> user_data("Usuario_Inexistente")
    {
        "detail": [
            {
                "loc": ["string", 0],
                "msg": "El usuario Usuario_Inexistente no existe.",
                "type": "value_error"
            }
        ]
    }
    """
    # Abrimos el archivo parquet en un dataframe
    funcion_2 = pd.read_parquet('/workspaces/codespaces-project-template-py/data/games_reviews.parquet')

    # Filtrar el dataframe 'games_reviews' por el 'User_id' dado
    df_filtered = funcion_2[funcion_2['user_id'] == user_id]

    # Verificar si el dataframe filtrado está vacío
    if df_filtered.empty:
        return {
            "detail": [
                {
                    "loc": ["string", 0],
                    "msg": f"El usuario {user_id} no existe.",
                    "type": "value_error"
                }
            ]
        }

    # Calcular la cantidad de dinero gastado y redondearlo a dos decimales
    total_spent = round(df_filtered['price'].sum(), 2)

    # Calcular el porcentaje de recomendación
    total_reviews = df_filtered['recommend'].count()
    recommended_reviews = df_filtered[df_filtered['recommend'] == True]['recommend'].count()
    recommendation_percentage = round((recommended_reviews / total_reviews) * 100) if total_reviews > 0 else 0

    # Calcular la cantidad de items
    total_items = df_filtered['item_name'].nunique()

    # Crear el diccionario de resultados
    dict_userdata = {
        "Usuario": user_id,
        "Dinero gastado": f"{total_spent} USD",
        "% de recomendación": f"{recommendation_percentage}%",
        "cantidad de artículos": total_items
    }

    return dict_userdata



# FUNCIÓN 3 - Usuario que acumula horas más jugadas para el genero dado
def User_For_Genre(genero):
    """
    Devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas
    por año de lanzamiento.

    Args:
    - genero (str): El género para el cual se desea encontrar el usuario con más horas jugadas.

    Returns:
    - dict: Un diccionario que contiene el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación
      de horas jugadas por año de lanzamiento.

    Ejemplo:
    >>> User_For_Genre("RPG")
    {
        'Usuario con más horas jugadas para el género RPG': 'ID_Usuario',
        'Horas jugadas': [{'Año': 2019, 'Horas': 350},
                          {'Año': 2020, 'Horas': 500},
                          {'Año': 'desconocido', 'Horas': 50}]
    }

    En caso de que el género no exista, devuelve un diccionario con un mensaje de error.

    Ejemplo:
    >>> User_For_Genre("Género_Inexistente")
    {
        "detail": [
            {
                "loc": ["string", 0],
                "msg": "El género no existe",
                "type": "value_error"
            }
        ]
    }
    """
    # Abrimos el archivo parquet en un dataframe
    funcion_3 = pd.read_parquet('/workspaces/codespaces-project-template-py/data/games_items.parquet')

    # Filtrar el DataFrame por el género dado
    df_genre = funcion_3[funcion_3['genres'] == genero]

    # si el dataframe esta vacio, lanzar una excepcion 
    if df_genre.empty:
        return {
            "detail": [
                {
                    "loc": ["string", 0],
                    "msg": "El género no existe",
                    "type": "value_error"
                }
            ]
        }
    
    # Obtener el usuario con más horas jugadas
    user_most_played_id = df_genre.groupby('user_id')['playtime_forever'].sum().idxmax()

    # Obtener la acumulación de horas jugadas por año de lanzamiento
    hours_per_year = get_hours_per_year(df_genre)

    # Devolver los resultados en el formato especificado
    return {"Usuario con más horas jugadas para el género {}".format(genero) : user_most_played_id, "Horas jugadas": hours_per_year}

def get_hours_per_year(df_genre):
    hours_per_year = df_genre.groupby('release_date')['playtime_forever'].sum().astype(int).reset_index()
    hours_per_year.columns = ['Año', 'Horas']

    # Cambiar el valor del año a 'desconocido' cuando es igual a 0
    hours_per_year['Año'] = hours_per_year['Año'].apply(lambda x: 'desconocido' if x == 0 else x)

    # Crear el diccionario de resultados
    hours_per_year = hours_per_year.to_dict('records')

    return hours_per_year



# FUNCIÓN 4 - DESARROLLADOR CON MÁS JUEGOS RECOMENDADOS Y CON COMENTARIOS POSITIVOS
def best_developer_year(year):
    """
    Esta función devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado.
    Se consideran solo los juegos con reviews donde 'recommend' es True y 'sentiment_analysis' es positivo (2).

    Args:
    - año (int): El año para el cual se desea obtener el top de desarrolladores.

    Returns:
    - list: Una lista de diccionarios que contiene el top 3 de desarrolladores con juegos MÁS recomendados por usuarios
      para el año dado. Cada diccionario en la lista tiene el formato {"Puesto X": "Nombre del desarrollador"}.

    Ejemplo:
    >>> best_developer_year(2023)
    [{'Puesto 1': 'Desarrollador A'}, {'Puesto 2': 'Desarrollador B'}, {'Puesto 3': 'Desarrollador C'}]

    En caso de que el año no exista, devuelve un diccionario con un mensaje de error.

    Ejemplo:
    >>> best_developer_year(2020)
    {
        "detail": [
            {
                "loc": ["string", 0],
                "msg": "El año no existe",
                "type": "value_error"
            }
        ]
    }
    """
    # Abrimos el archivo parquet en un dataframe
    funcion_4 = pd.read_parquet('/workspaces/codespaces-project-template-py/data/games_reviews.parquet')

    # Filtrar el dataframe 'games_reviews' por año, recomendaciones y comentarios positivos
    df_filtered = funcion_4[(funcion_4['release_date'] == year) & (funcion_4['recommend'] == True) & (funcion_4['sentiment_analysis'] == 2)]

    # cuando el año no existe, lanzar una excepcion
    if year not in funcion_4['release_date'].unique():
        return {
            "detail": [
                {
                    "loc": ["string", 0],
                    "msg": "El año no existe",
                    "type": "value_error"
                }
            ]
        }

    # Agrupar por desarrollador y contar las reseñas
    developer_counts = df_filtered['developer'].value_counts()

    # Ordenar los resultados y seleccionar los tres primeros
    top_developers = developer_counts.nlargest(3)

    # Crear la lista de resultados
    list_best_developer = []
    for i, (developer, count) in enumerate(top_developers.items(), start=1):
        list_best_developer.append({f"Puesto {i}": developer})

    return list_best_developer


# FUNCIÓN 5 - ANÁLISIS DE SENTIMIENTO DE RESEÑAS PARA UN DESARROLLADOR
def desarrollador_reviews_analysis(desarrolladora):
    """
    Según el desarrollador, devuelve un diccionario con el nombre del desarrollador como llave
    y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren
    categorizados con un análisis de sentimiento como valor positivo o negativo.

    Args:
    - desarrolladora (str): El nombre del desarrollador para el cual se desea realizar el análisis.

    Returns:
    - dict: Un diccionario que contiene el nombre del desarrollador como llave y una lista con la cantidad
      total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento
      como valor positivo o negativo. Ejemplo de retorno:
      {'Valve': [{'Negativo': 182, 'Positivo': 278}]}

    En caso de que no haya datos para el desarrollador especificado, devuelve un diccionario con un mensaje de error.

    Ejemplo:
    >>> desarrollador_reviews_analysis("Valve")
    {'Valve': [{'Negativo': 182, 'Positivo': 278}]}
    """
    # Abrimos el archivo parquet en un dataframe
    funcion_5 = pd.read_parquet('/workspaces/codespaces-project-template-py/data/games_reviews.parquet')

    # Filtrar el dataframe por el desarrollador
    df_filtered = funcion_5[funcion_5['developer'] == desarrolladora]

    # Verificar si el DataFrame filtrado está vacío y dar una excepción
    if df_filtered.empty:
        return {
            "detail": [
                {
                    "loc": ["string", 0],
                    "msg": "No hay datos para el desarrollador {}".format(desarrolladora),
                    "type": "value_error"
                }
            ]
        }

    # Contar el número de reseñas con análisis de sentimiento positivo y negativo
    sentiment_counts = df_filtered['sentiment_analysis'].value_counts()

    # Crear el diccionario de resultados
    list_reviews_analysis = {desarrolladora: [{'Negativo': int(sentiment_counts.get(0, 0)), 'Positivo': int(sentiment_counts.get(2, 0))}]}

    # Asegurarse de que el resultado es un diccionario
    if not isinstance(list_reviews_analysis, dict):
        list_reviews_analysis = list_reviews_analysis.to_dict()

    return list_reviews_analysis


# SISTEMA DE RECOMENDACIÓN - ITEM-ITEM
def recomendacion_juego(item_id):
    """
    Sistema de recomendación item-item.

    Ingresando el ID de un producto, se devuelve una lista con 5 juegos recomendados similares al ingresado.

    Args:
    - item_id: El ID del producto para el cual se desean obtener recomendaciones.

    Returns:
    - dict: Un diccionario que contiene una lista de 5 juegos recomendados similares al ingresado.
      El diccionario tiene el formato {"Juegos similares al ID ingresado": {ID1: 'Nombre1', ID2: 'Nombre2', ...}}.

    Raises:
    - Si el DataFrame está vacío, devuelve un mensaje de error.
    - Si el item_id no existe en el DataFrame, devuelve un mensaje de error.
    - Si ocurre un error al cargar el archivo de similitud, intenta calcular la similitud del coseno
      entre los productos y proporciona recomendaciones basadas en eso.
    """
    # Leer el archivo parquet en un DataFrame
    data = pd.read_parquet('/workspaces/codespaces-project-template-py/data/item_item.parquet')
    
    try:
        # Cargar la matriz de similitud de items_similarity
        with open('/workspaces/codespaces-project-template-py/data/item_similarity.pkl', 'rb') as f:
            item_similarity_df = pickle.load(f)

        # mensaje de test para verificar que se cargó la matriz precalculada
        print("Se cargó la matriz precalculada.")

    except Exception as e:
        print(f"Ocurrió un error al cargar el archivo: {e}")

        

        # Si el dataframe está vacío, devolver un mensaje de error
        if data.empty:
            return {
                "detail": [
                    {
                        "loc": ["string", 0],
                        "msg": "El DataFrame está vacío",
                        "type": "value_error"
                    }
                ]
            }

        # Si el item_id no existe en el DataFrame, devolver un mensaje de error
        if item_id not in data['item_id'].values:
            return {
                "detail": [
                    {
                        "loc": ["string", 0],
                        "msg": "El item_id proporcionado no existe",
                        "type": "value_error"
                    }
                ]
            }

        # Convertir la columna 'recommend' a valores numéricos
        data['recommend'] = data['recommend'].apply(lambda x: 1 if x else 0)

        # Crear la matriz de utilidad
        matrix = data.pivot_table(index='item_id', values=['recommend', 'sentiment_analysis'])

        # Rellenar los valores faltantes con 0
        matrix.fillna(0, inplace=True)

        # Calcular la similitud del coseno
        item_similarity = cosine_similarity(matrix)

        # Convertir los resultados en un DataFrame
        item_similarity_df = pd.DataFrame(item_similarity, index=matrix.index, columns=matrix.index)

        # mensaje de test para verificar que se calculó la matriz de similitud
        print("Se calculó la matriz de similitud.")

    if 'item_similarity_df' not in locals():
        # Obtener los juegos similares al juego dado
        similar_games = item_similarity_df[item_id].sort_values(ascending=False)

        # Obtener los 5 juegos más similares
        similar_games_ids = similar_games.head(6).index.tolist()[1:]  # Excluimos el primer juego porque es el mismo juego

        # Crear un diccionario con los ids y los nombres de los juegos
        recommended_games = {game_id: data[data['item_id'] == game_id]['item_name'].iloc[0] for game_id in similar_games_ids}

        return {"Juegos similares al id ingresado": recommended_games}
    else:
        # Obtener los juegos similares al juego dado
        similar_games = item_similarity_df[item_id].sort_values(ascending=False)

        # Obtener los 5 juegos más similares
        similar_games_ids = similar_games.head(6).index.tolist()[1:]  # Excluimos el primer juego porque es el mismo juego

        # Crear un diccionario con los ids y los nombres de los juegos
        recommended_games = {game_id: data[data['item_id'] == game_id]['item_name'].iloc[0] for game_id in similar_games_ids}

        return {"Juegos similares al id ingresado": recommended_games}

  

# SISTEMA DE RECOMENDACIÓN - USER-USER 
def recomendacion_usuario(user_id):
    """
    Sistema de recomendación user-item.

    Ingresando el ID de un usuario, se devuelve una lista con 5 juegos recomendados para dicho usuario.

    Args:
    - user_id (str): El ID del usuario para el cual se desean obtener recomendaciones.

    Returns:
    - dict: Un diccionario que contiene una lista de 5 juegos recomendados para el usuario ingresado.
      El diccionario tiene el formato {"Juegos recomendados para el usuario": user_id,
      'Juego 1': 'Nombre1', 'Juego 2': 'Nombre2', 'Juego 3': 'Nombre3', 'Juego 4': 'Nombre4', 'Juego 5': 'Nombre5'}.

    Raises:
    - Si el DataFrame está vacío, devuelve un mensaje de error.
    - Si el user_id no existe en el DataFrame, devuelve un mensaje de error.
    """
    # Cargar los datos del archivo .parquet
    new_df = pd.read_parquet('/workspaces/codespaces-project-template-py/data/user_item_model.parquet')

    # # Si el dataframe está vacío, devolver un mensaje de error
    if new_df.empty:
        return {
        "detail": [
            {
                "loc": ["string", 0],
                "msg": "El DataFrame está vacío",
                "type": "value_error"
            }
        ]
    }

    if user_id not in new_df['user_id'].unique():
        return {
        "detail": [
            {
                "loc": ["string", 0],
                "msg": f"El usuario {user_id} no existe.",
                "type": "value_error"
            }
        ]
    }

    juegos_valorados = new_df[new_df['user_id'] == user_id]['item_name'].unique()
    todos_los_juegos = new_df['item_name'].unique()
    juegos_no_valorados = list(set(todos_los_juegos) - set(juegos_valorados))
    predicciones = [modelo.predict(user_id, juego) for juego in juegos_no_valorados]
    recomendaciones = sorted(predicciones, key=lambda x: x.est, reverse=True)[:5]
    juegos_recomendados = [recomendacion.iid for recomendacion in recomendaciones]
    recomendaciones_dict = {"Juegos recomendados para el usuario": user_id, 'Juego 1': juegos_recomendados[0], 'Juego 2': juegos_recomendados[1], 'Juego 3': juegos_recomendados[2], 'Juego 4': juegos_recomendados[3], 'Juego 5': juegos_recomendados[4]}
    return recomendaciones_dict