import os
os.environ["TRANSFORMERS_NO_TORCH_IMPORT"] = "1"
import pandas as pd
import numpy as np
import streamlit as st 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import requests
from datetime import datetime
from transformers import pipeline
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración inicial de la página
st.set_page_config(page_title="RugbyIA Analytics", page_icon="🏉", layout="wide")

# Cargar datos de ejemplo (en un caso real usarías una API o base de datos)
@st.cache_data
def load_data():
    # Datos ficticios para demostración
    data = {
        'Equipo': ['All Blacks', 'Springboks', 'Wallabies', 'Pumas', 'Les Bleus', 'Red Roses'],
        'Puntos a favor': [120, 95, 80, 75, 110, 85],
        'Puntos en contra': [65, 70, 90, 85, 75, 60],
        'Victorias': [4, 3, 2, 2, 3, 3],
        'Derrotas': [1, 2, 3, 3, 2, 2],
        'Tries': [15, 12, 8, 7, 14, 10]
    }
    return pd.DataFrame(data)

# Cargar modelo de predicción (simulado)
@st.cache_resource
def load_model():
    # En una aplicación real, cargarías un modelo entrenado
    model = RandomForestClassifier()
    return model

# Cargar pipeline de NLP para resúmenes
load_dotenv()  

@st.cache_resource
def load_nlp_pipeline():
    try:
        return pipeline(
            "summarization",
            model="t5-small",  # Modelo ligero y sin requisitos de autenticación
            framework='tf',
            token=os.getenv("HF_TOKEN") if os.getenv("HF_TOKEN") else None
        )
    except Exception as e:
        return None

# Función para generar predicción
def predict_match(team1, team2, data):
    # Simulación de predicción - en un caso real usarías el modelo cargado
    team1_stats = data[data['Equipo'] == team1].iloc[0]
    team2_stats = data[data['Equipo'] == team2].iloc[0]
    
    diff = (team1_stats['Puntos a favor'] - team1_stats['Puntos en contra']) - \
           (team2_stats['Puntos a favor'] - team2_stats['Puntos en contra'])
    
    if diff > 10:
        return f"{team1} tiene alta probabilidad de ganar"
    elif diff < -10:
        return f"{team2} tiene alta probabilidad de ganar"
    else:
        return "Partido equilibrado, resultado incierto"

# Función para generar resumen de análisis
def generate_analysis_summary(team, data, nlp_pipeline):
    team_data = data[data['Equipo'] == team].iloc[0]
    text = f"""
    El equipo {team} ha demostrado un rendimiento sólido en la temporada actual. 
    Con {team_data['Victorias']} victorias y {team_data['Derrotas']} derrotas, 
    han anotado un total de {team_data['Puntos a favor']} puntos y concedido {team_data['Puntos en contra']}. 
    Su ataque ha sido efectivo con {team_data['Tries']} tries marcados.
    """
    
    if nlp_pipeline:
        try:
            summary = nlp_pipeline(text, max_length=130, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except:
            return text
    return text

# Función para crear gráficos interactivos con Plotly
def plot_team_stats(team_data):
    # Gráfico de estadísticas principales
    stats_df = team_data.melt(id_vars=['Equipo'], 
                             value_vars=['Puntos a favor', 'Puntos en contra', 'Tries'])
    
    fig1 = px.bar(stats_df, 
                 x='variable', 
                 y='value', 
                 color='variable',
                 title=f'Estadísticas de {team_data["Equipo"].values[0]}',
                 labels={'value': 'Cantidad', 'variable': 'Métrica'},
                 text='value',
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    
    fig1.update_traces(textposition='outside')
    fig1.update_layout(showlegend=False, hovermode="x")
    
    # Gráfico de victorias/derrotas
    results_df = team_data.melt(id_vars=['Equipo'], 
                              value_vars=['Victorias', 'Derrotas'])
    
    fig2 = px.pie(results_df, 
                 values='value', 
                 names='variable',
                 title='Balance de Partidos',
                 color='variable',
                 color_discrete_map={'Victorias': '#4CAF50', 'Derrotas': '#F44336'})
    
    return fig1, fig2

# Interfaz de la aplicación
def main():
    st.title("🏉 RugbyIA Analytics")
    st.markdown("""
    **Aplicación de análisis de rugby con inteligencia artificial**  
    Obtén predicciones, estadísticas y análisis generados por IA para partidos de rugby.
    """)
    
    # Cargar datos y modelos
    df = load_data()
    model = load_model()
    nlp_pipeline = load_nlp_pipeline()
    
    # Sidebar con opciones
    st.sidebar.header("Opciones")
    app_mode = st.sidebar.selectbox("Selecciona el modo", 
                                   ["Análisis de Equipos", "Predicción de Partidos", "Resumen IA"])
    
    if app_mode == "Análisis de Equipos":
        st.header("📊 Análisis Estadístico de Equipos")
        
        selected_team = st.selectbox("Selecciona un equipo", df['Equipo'].unique())
        team_data = df[df['Equipo'] == selected_team]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Estadísticas de {selected_team}")
            st.dataframe(team_data.set_index('Equipo').T.style.background_gradient(cmap='Blues'))
        
        with col2:
            st.subheader("Análisis Visual")
            fig1, fig2 = plot_team_stats(team_data)
            st.plotly_chart(fig1, use_container_width=True)
            
            if st.checkbox("Mostrar balance de partidos"):
                st.plotly_chart(fig2, use_container_width=True)
    
    elif app_mode == "Predicción de Partidos":
        st.header("🔮 Predicción de Partidos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            team1 = st.selectbox("Equipo Local", df['Equipo'].unique())
        
        with col2:
            team2 = st.selectbox("Equipo Visitante", df['Equipo'].unique())
        
        if st.button("Predecir Resultado"):
            with st.spinner("Analizando datos y generando predicción..."):
                prediction = predict_match(team1, team2, df)
                st.success("Predicción:")
                st.markdown(f"## 🏆 {prediction}")
                
                # Mostrar comparativa
                st.subheader("Comparativa de Equipos")
                comparison = df[df['Equipo'].isin([team1, team2])].set_index('Equipo')
                st.dataframe(comparison.style.background_gradient(cmap='YlOrBr'))
                
                # Gráfico comparativo interactivo
                fig = px.bar(comparison.reset_index(), 
                            x='Equipo',
                            y=['Puntos a favor', 'Puntos en contra'],
                            barmode='group',
                            title="Comparativa de Puntos",
                            labels={'value': 'Puntos', 'variable': 'Tipo'},
                            color_discrete_sequence=px.colors.qualitative.Pastel)
                
                st.plotly_chart(fig, use_container_width=True)
    
    elif app_mode == "Resumen IA":
        st.header("🤖 Resumen Generado por IA")
        st.info("Esta función analiza los últimos partidos y genera un resumen con insights clave.")
        
        selected_team = st.selectbox("Selecciona un equipo para analizar", df['Equipo'].unique())
        
        if st.button("Generar Resumen de Temporada"):
            with st.spinner("Procesando con IA..."):
                summary = generate_analysis_summary(selected_team, df, nlp_pipeline)
                st.success("Resumen generado:")
                st.markdown(f"**Resumen de {selected_team}**")
                st.write(summary)
                
                # Datos adicionales con métricas visuales
                cols = st.columns(3)
                team_data = df[df['Equipo'] == selected_team].iloc[0]
                
                with cols[0]:
                    st.metric("Victorias", team_data['Victorias'], delta=f"+{team_data['Victorias']}", delta_color="normal")
                
                with cols[1]:
                    st.metric("Derrotas", team_data['Derrotas'], delta=f"-{team_data['Derrotas']}", delta_color="inverse")
                
                with cols[2]:
                    st.metric("Tries", team_data['Tries'], delta=f"+{team_data['Tries']}", delta_color="normal")
    
    # Sección "Cómo funciona"
    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ Cómo funciona RugbyIA"):
        st.markdown("""
        1. **Análisis de Equipos**: Visualiza estadísticas detalladas de cada equipo.
        2. **Predicción de Partidos**: Compara dos equipos y obtén una predicción basada en IA.
        3. **Resumen IA**: Genera un análisis textual automático del rendimiento del equipo.
        
        **Características técnicas**:
        - Gráficos interactivos con Plotly
        - Modelos de Machine Learning
        - Procesamiento de Lenguaje Natural
        - Diseño responsive
        """)

if __name__ == "__main__":
    main()
