import requests
import socket
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Obtenemos la localización del usuario
def get_location(api_key):

    response = requests.get(f"https://ipinfo.io?{api_key}")
    print("IP Detectada...", response.json()["ip"])
    return response


# Obtenemos los datos del clima de la región donde se encuentra el usuario
def get_weather(lat, lon, api_key):

    response = requests.get(
        f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&cnt=3&appid={api_key}&units=metric"
    )
    print("Obteniendo datos climáticos...")
    return response.json()


# Analizamos la data climática por medio de OpenAI o plataformas de inferencia
# compatibles con la OpenAI API
def analyze_weather(data, api, ai_prompt):

    api_base = "https://openrouter.ai/api/v1"

    client = OpenAI(api_key=api, base_url=api_base)

    print("Analizando los datos climáticos...")

    send_data = json.dumps(data)

    chat_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": ai_prompt + send_data,
            },
        ],
    )

    return chat_response


# Función principal del programa aqui cargamos todo lo necesario para iniciar
def main():

    # DotENV
    load_dotenv()

    # Leemos las API_KEYS para OpenWeather, ipinfo, OpenAI y el system prompt
    openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
    if not openweather_api_key:
        print("Error: No se encontró la clave de API de OpenWeather.")
        return

    ip_api_key = os.getenv("IP_API_KEY")
    if not ip_api_key:
        print("Error: No se encontró la clave de API de IP API.")
        return

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: No se encontró la clave de API de OpenAI o compatible.")
        return

    ai_prompt = os.getenv("AI_PROMPT")
    if not ai_prompt:
        print("Error: No se encontró el prompt para el análisis.")
        return

    # Obtenemos la dirección IP y nuestra ubicación aproximada
    location = get_location(ip_api_key)

    # Extraemos nuestra ubicación para obtener el clima
    coordinates = location.json()["loc"]
    latitude_str, longitude_str = coordinates.split(",")

    # Obtenemos el clima de la región usando Open Weather
    weather_data = get_weather(latitude_str, longitude_str, openweather_api_key)

    # Analizamos el clima usando Mistral
    weather_analysis = analyze_weather(weather_data, openai_api_key, ai_prompt)

    # Imprimimos análsis en pantalla
    print(weather_analysis.choices[0].message.content)


if __name__ == "__main__":
    main()
