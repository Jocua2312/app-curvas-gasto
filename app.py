# -*- coding: utf-8 -*-
"""
APLICACIÓN DE CURVA DE GASTO - DASHBOARD INTERACTIVO (VERSIÓN STREAMLIT)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d
import pickle
import datetime
import warnings
import io
import logging
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración de página a pantalla completa
st.set_page_config(page_title="Curva de Gasto", layout="wide")

# --- CSS  ---
st.markdown(
    """
    <style>
    div[data-testid="stDateInput"] input {
                min-width: 150px;
            }
    .st-h7 {
        min-width: 40rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 1. FUNCIONES GEOMÉTRICAS (VECTORIZADAS CON NUMPY)
# ============================================================

import numpy as np

def _preparar_vectores(abscisas, cotas, cota_cero, nivel):
    """Función auxiliar para calcular intersecciones y máscaras de forma vectorizada"""
    NA = cota_cero + nivel
    
    # Filtrar NaNs automáticamente
    mask_valid = ~(np.isnan(abscisas) | np.isnan(cotas))
    x, y = abscisas[mask_valid], cotas[mask_valid]
    
    if len(x) < 2:
        return None
        
    x0, x1 = x[:-1], x[1:]
    y0, y1 = y[:-1], y[1:]
    dx = x1 - x0
    dy = y1 - y0
    
    # Profundidad de cada punto respecto al nivel del agua
    h0 = NA - y0
    h1 = NA - y1
    
    # Evitar división por cero en terrenos totalmente planos
    dy_safe = np.where(dy == 0, 1e-10, dy)
    
    # Fracción de distancia donde el agua toca el talud (Teorema de Tales)
    frac = (NA - y0) / dy_safe
    
    # Máscaras booleanas para los 3 casos posibles
    m_ambos = (h0 >= 0) & (h1 >= 0)  # Totalmente sumergido
    m_baja = (h0 < 0) & (h1 > 0)     # Orilla izquierda (el nivel toca el suelo bajando)
    m_sube = (h0 > 0) & (h1 < 0)     # Orilla derecha (el nivel choca con el suelo subiendo)
    
    return x0, x1, y0, y1, dx, dy, h0, h1, frac, m_ambos, m_baja, m_sube

def area_mojada(abscisas, cotas, cota_cero, nivel, modo_muro="sin_friccion"):
    datos = _preparar_vectores(abscisas, cotas, cota_cero, nivel)
    if not datos: return 0.0
    x0, x1, y0, y1, dx, dy, h0, h1, frac, m_ambos, m_baja, m_sube = datos
    
    area = np.zeros_like(dx)
    # Trapecio completo (Funciona como muro vertical nativo si desborda)
    area[m_ambos] = (h0[m_ambos] + h1[m_ambos]) * dx[m_ambos] / 2.0
    # Triángulo izquierdo
    area[m_baja] = h1[m_baja] * dx[m_baja] * (1 - frac[m_baja]) / 2.0
    # Triángulo derecho
    area[m_sube] = h0[m_sube] * dx[m_sube] * frac[m_sube] / 2.0
    
    return float(np.sum(area))

def ancho_superficial(abscisas, cotas, cota_cero, nivel, modo_muro="sin_friccion"):
    datos = _preparar_vectores(abscisas, cotas, cota_cero, nivel)
    if not datos: return 0.0
    x0, x1, y0, y1, dx, dy, h0, h1, frac, m_ambos, m_baja, m_sube = datos
    
    ancho = np.zeros_like(dx)
    # Ancho completo (se tranca en el ancho máximo si desborda, actuando como muro)
    ancho[m_ambos] = dx[m_ambos]
    # Ancho parcial izquierdo
    ancho[m_baja] = dx[m_baja] * (1 - frac[m_baja])
    # Ancho parcial derecho
    ancho[m_sube] = dx[m_sube] * frac[m_sube]
    
    return float(np.sum(ancho))

def perimetro_mojado(abscisas, cotas, cota_cero, nivel, modo_muro="sin_friccion"):
    datos = _preparar_vectores(abscisas, cotas, cota_cero, nivel)
    if not datos: return 0.0
    x0, x1, y0, y1, dx, dy, h0, h1, frac, m_ambos, m_baja, m_sube = datos
    
    perimetro = np.zeros_like(dx)
    # Hipotenusa completa
    perimetro[m_ambos] = np.hypot(dx[m_ambos], dy[m_ambos])
    # Hipotenusa parcial izquierda
    perimetro[m_baja] = np.hypot(dx[m_baja] * (1 - frac[m_baja]), h1[m_baja])
    # Hipotenusa parcial derecha
    perimetro[m_sube] = np.hypot(dx[m_sube] * frac[m_sube], h0[m_sube])
    
    p_total = float(np.sum(perimetro))
    
    # --- LÓGICA DE MUROS DE EXTRAPOLACIÓN ---
    # Si el usuario eligió "con_friccion", castigamos al río sumando los muros al perímetro
    if modo_muro == "con_friccion":
        # h0[0] es la altura del agua sobre la primera abscisa. Si es > 0, se desbordó por la izquierda.
        if h0[0] > 0:
            p_total += h0[0]
            
        # h1[-1] es la altura del agua sobre la última abscisa. Si es > 0, se desbordó por la derecha.
        if h1[-1] > 0:
            p_total += h1[-1]
            
    # Si es "sin_friccion" o "ninguno", simplemente devolvemos p_total intacto 
    # (el perímetro queda congelado en la topografía máxima, justo como recomienda Ven Te Chow)
    return p_total

def calcular_mape(q_obs, q_calc):
    """
    Calcula el Error Porcentual Absoluto Medio (MAPE).
    Equivalente a =PROMEDIO(ABS(Qcalc - Qobs)/Qobs) * 100
    """
    q_obs = np.array(q_obs)
    q_calc = np.array(q_calc)
    
    # Máscara de seguridad para evitar divisiones por cero por si acaso
    mask = q_obs != 0 
    
    # Cálculo vectorizado del error absoluto porcentual
    error_porcentual = np.abs((q_calc[mask] - q_obs[mask]) / q_obs[mask])
    
    # Retorna el promedio multiplicado por 100 para tenerlo en %
    return np.mean(error_porcentual) * 100

def calcular_error_procedimiento(Q_obs, Q_est, K=2):
    """
    Calcula el error de procedimiento según la fórmula estadística.
    Q_obs: Arreglo de caudales aforados reales.
    Q_est: Arreglo de caudales estimados por la curva.
    K: Grados de libertad (2 para modelos lineal, exp, log, potencial).
    """
    # Evitar divisiones por cero y valores no finitos
    mask = (Q_est != 0) & np.isfinite(Q_est) & (Q_obs != 0)
    Q_o = np.array(Q_obs)[mask]
    Q_e = np.array(Q_est)[mask]
    
    N = len(Q_o)
    
    if N <= K:
        return np.nan # No hay suficientes puntos para calcular el error con esos grados de libertad
        
    # Aplicar la fórmula
    suma = np.sum(((Q_o - Q_e) / Q_e)**2)
    error_sigma = np.sqrt(suma / (N - K))
    
    return error_sigma * 100 # Lo multiplicamos por 100 para mostrarlo en %

def crear_figura_k(titulo, H_act, Y_act, inactivos, H_smooth, funciones_adicionales, Y_sel, color_activos, color_inactivos, color_ajuste, color_adicional='cyan', xlabel="Nivel H (m)", ylabel="K"):
    """
    Crea una figura de Plotly para gráficas de K vs H (o V vs H).
    
    Parámetros:
    - titulo: str, título de la gráfica.
    - H_act: array, niveles de puntos activos.
    - Y_act: array, valores de K (o V) de puntos activos.
    - inactivos: DataFrame con columnas H y K (o V) para puntos inactivos.
    - H_smooth: array, niveles para la curva suave.
    - funciones_adicionales: dict con claves 'lineal', 'exp', 'log', 'pot' y valores (función, color) opcionales.
    - Y_sel: array, valores de la curva del modelo seleccionado (para H_smooth).
    - color_activos, color_inactivos, color_ajuste: strings con colores.
    - color_adicional: color base para los modelos adicionales (se generan variaciones).
    - xlabel, ylabel: etiquetas de ejes.
    """
    fig = go.Figure()
    
    # Puntos activos
    fig.add_trace(go.Scatter(x=H_act, y=Y_act, mode='markers', name='Activos',
                              marker=dict(color=color_activos, size=9, opacity=0.9, line=dict(color='white', width=0.5))))
    # Puntos inactivos
    if inactivos is not None and not inactivos.empty:
        fig.add_trace(go.Scatter(x=inactivos["H"], y=inactivos["Y"], mode='markers', name='Ignorados',
                                  marker=dict(color=color_inactivos, symbol='x', size=8, line=dict(color='white', width=0.5))))
    
    # Modelos adicionales (si se proporcionan)
    colores_adicionales = {'lineal': 'cyan', 'exp': 'magenta', 'log': 'yellow', 'pot': 'lime'}
    for key, (func, mostrar) in funciones_adicionales.items():
        if mostrar and func is not None:
            Y_adicional = func(H_smooth)
            fig.add_trace(go.Scatter(x=H_smooth, y=Y_adicional, mode='lines', name=key.capitalize(),
                                      line=dict(color=colores_adicionales[key], width=2, dash='dash')))
    
    # Curva del modelo seleccionado
    fig.add_trace(go.Scatter(x=H_smooth, y=Y_sel, mode='lines', name='Modelo seleccionado',
                              line=dict(color=color_ajuste, width=3)))
    
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        title=titulo,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        hovermode="closest",
        margin=dict(l=0, r=0, t=80, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def crear_figura_curva(titulo, H_fino, Q_suave, Q_act, H_act, inactivos=None, 
                       color_curva='#ffaa00', color_activos='#00ccff', 
                       color_inactivos='#ff5555', color_banda='rgba(200, 200, 200, 0.15)',
                       banda_pct=15.0): # <--- NUEVO PARÁMETRO
    
    import plotly.graph_objects as go
    fig = go.Figure()

    # Calcular bandas usando el porcentaje dinámico
    factor = banda_pct / 100.0
    Q_upper = Q_suave * (1 + factor)
    Q_lower = Q_suave * (1 - factor)

    # Banda de incertidumbre usando las variables calculadas
    fig.add_trace(go.Scatter(x=Q_upper, y=H_fino, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=Q_lower, y=H_fino, mode='lines', fill='tonexty',
                              fillcolor=color_banda, line=dict(width=0), name=f'Banda ±{banda_pct}%'))
    
    # Curva principal
    fig.add_trace(go.Scatter(x=Q_suave, y=H_fino, mode='lines', name='Curva',
                              line=dict(color=color_curva, width=3)))
    
    # Puntos activos
    fig.add_trace(go.Scatter(x=Q_act, y=H_act, mode='markers', name='Activos',
                              marker=dict(color=color_activos, size=9, opacity=0.9, line=dict(color='white', width=0.5))))
    
    # Puntos inactivos
    if inactivos is not None and not inactivos.empty:
        fig.add_trace(go.Scatter(x=inactivos["Q"], y=inactivos["H"], mode='markers', name='Ignorados',
                                  marker=dict(color=color_inactivos, symbol='x', size=8, line=dict(color='white', width=0.5))))
    
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        title=titulo,
        xaxis_title="Caudal Q (m³/s)",
        yaxis_title="Nivel H (m)",
        hovermode="closest",
        margin=dict(l=0, r=0, t=80, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

# ============================================================
# 2. LECTURA DE ARCHIVOS
# ============================================================

@st.cache_data
def cargar_excel_seguro(file_buffer):
    if file_buffer.name.endswith('.xls'):
        return pd.read_excel(file_buffer, header=None, engine='xlrd')
    return pd.read_excel(file_buffer, header=None)

@st.cache_data
def leer_perfil_transversal_completo(archivo_perfil):
    df_raw = cargar_excel_seguro(archivo_perfil)
    estacion, codigo, fecha_perfil = "Desconocida", "Desconocido", "No especificada"
    cota_cero = None

    # --- BÚSQUEDA DE COTA CERO ---
    for i in range(min(30, len(df_raw))):
        fila_vals = df_raw.iloc[i].astype(str).tolist()
        for j, celda in enumerate(fila_vals):
            texto_celda = celda.strip().upper()
            if "COTA CERO LM" in texto_celda:
                if ":" in texto_celda:
                    val_potencial = texto_celda.split(":")[1].strip()
                    if val_potencial and val_potencial != "NAN":
                        try:
                            cota_cero = float(val_potencial.replace(',', '.'))
                            break
                        except ValueError:
                            pass
                if cota_cero is None:
                    for k in range(j + 1, len(fila_vals)):
                        val_derecha = str(df_raw.iloc[i, k]).strip()
                        if val_derecha and val_derecha.upper() != "NAN":
                            try:
                                cota_cero = float(val_derecha.replace(',', '.'))
                                break
                            except ValueError:
                                pass
            if cota_cero is not None:
                break
        if cota_cero is not None:
            break

    # --- EXTRACCIÓN DE METADATOS (MEJORADA) ---
    for i in range(min(30, len(df_raw))):
        row_vals = df_raw.iloc[i].dropna().astype(str).tolist()
        for j, val in enumerate(row_vals):
            v_up = val.upper()
            if "CÓDIGO" in v_up or "CODIGO" in v_up:
                if len(row_vals) > j + 1:
                    codigo = row_vals[j+1].strip()
            elif "FECHA" in v_up:
                if len(row_vals) > j + 1:
                    fecha_perfil = row_vals[j+1].strip()
            # Intento de extraer estación (puede estar en una celda fija, pero también buscamos por palabras)
            # Primero intentamos la posición fila 10, col 6 (como antes)
            try:
                val_est = df_raw.iloc[10, 6]
                if pd.notna(val_est):
                    estacion_str = str(val_est).strip()
                    # Buscar formato "Nombre [Código]"
                    import re
                    match = re.search(r'^(.*?)\s*\[(.*?)\]$', estacion_str)
                    if match:
                        estacion = match.group(1).strip()
                        codigo = match.group(2).strip()
                    else:
                        estacion = estacion_str
            except:
                pass

    # Si no se encontró código en la estación, usar el obtenido de "CÓDIGO"
    # (ya está en codigo)
    # Asegurar que estacion no sea "Desconocida" si ya tenemos algo
    if estacion == "Desconocida" and 'val_est' in locals():
        estacion = str(val_est).strip()

    if cota_cero is None:
        raise ValueError("No se pudo encontrar el valor después de 'COTA CERO LM(m):'. Revisa el Excel.")

    # --- Listas separadas para Cartera y Sondeo ---
    puntos_cartera = []
    puntos_sondeo = []

    def to_num(val):
        if pd.isna(val) or str(val).strip() == "": return None
        try: return float(str(val).replace(',', '.'))
        except ValueError: return None

    def extraer_puntos_tabla(palabra_clave, lista_destino):
        idx_abscisa, idx_y, idx_desc, fila_inicio_datos = None, None, None, None
        fila_encabezados_idx = None
        for i, row in df_raw.iterrows():
            row_texto = " ".join(row.dropna().astype(str)).upper()
            if palabra_clave in row_texto:
                for j in range(i, min(i+10, len(df_raw))):
                    fila_encabezados = df_raw.iloc[j]
                    for col_num, celda in enumerate(fila_encabezados):
                        if pd.isna(celda): continue
                        texto = str(celda).upper()
                        if "ABSCISADO" in texto:
                            idx_abscisa = col_num
                        elif "COTA" in texto and "CERO" not in texto:
                            idx_y = col_num
                        if "DESCRIPCIÓN" in texto:
                            idx_desc = col_num
                    if idx_abscisa is not None and idx_y is not None:
                        fila_encabezados_idx = j
                        break
                if fila_encabezados_idx is not None:
                    fila_inicio_datos = fila_encabezados_idx + 1
                break

        if fila_inicio_datos is not None:
            vacios_consecutivos = 0
            for k in range(fila_inicio_datos, len(df_raw)):
                row_actual = df_raw.iloc[k]
                if idx_abscisa >= len(row_actual) or idx_y >= len(row_actual): break
                val_x = to_num(row_actual.iloc[idx_abscisa])
                val_y = to_num(row_actual.iloc[idx_y])

                desc = None
                if idx_desc is not None and idx_desc < len(row_actual):
                    d = row_actual.iloc[idx_desc]
                    if pd.notna(d):
                        desc = str(d).strip()

                if val_x is None and val_y is None:
                    vacios_consecutivos += 1
                    if vacios_consecutivos > 3: break
                    continue
                if val_x is None or val_y is None:
                    continue

                if not (0 <= val_x <= 10000 and -10000 <= val_y <= 10000):
                    continue

                # --- FILTRO AUTOMÁTICO POR DESCRIPCIÓN (solo para CARTERA) ---
                if palabra_clave == "CARTERA" and desc is not None:
                    if "S/MAXIMETRO" in desc.upper() or "S/LM" in desc.upper():
                        continue  # no agregar este punto

                lista_destino.append({
                    'abscisa': val_x,
                    'cota': val_y,
                    'descripcion': desc,
                    'tabla': palabra_clave,
                    'orden': k
                })
                vacios_consecutivos = 0

    extraer_puntos_tabla("CARTERA", puntos_cartera)
    extraer_puntos_tabla("SONDEO", puntos_sondeo)

    if not puntos_cartera and not puntos_sondeo:
        raise ValueError("No se extrajeron puntos válidos de Cartera ni Sondeo.")

    # --- Fusionar todos los puntos (sin prioridad) ---
    puntos_finales = puntos_cartera + puntos_sondeo

    # --- Eliminar duplicados exactos (misma abscisa y misma cota) ---
    puntos_unicos = {}
    for p in puntos_finales:
        clave = (p['abscisa'], p['cota'])
        if clave not in puntos_unicos:
            puntos_unicos[clave] = p
    puntos_finales = list(puntos_unicos.values())

    # --- Ordenar por abscisa, y para igual abscisa, por orden de aparición ---
    puntos_finales.sort(key=lambda p: (p['abscisa'], p['orden']))

    # Extraer listas finales
    abscisas_ord = [p['abscisa'] for p in puntos_finales]
    cotas_ord = [p['cota'] for p in puntos_finales]
    descs_ord = [p['descripcion'] for p in puntos_finales]
    tabs_ord = [p['tabla'] for p in puntos_finales]

    return cota_cero, np.array(abscisas_ord), np.array(cotas_ord), estacion, codigo, fecha_perfil, descs_ord, tabs_ord

# ================== FUNCIÓN EN CACHÉ ==================
@st.cache_data(show_spinner="Leyendo el archivo Excel histórico. Esto puede tardar unos segundos...")
def cargar_historico_excel(file_buffer):
    import pandas as pd
    df = pd.read_excel(file_buffer)
    df.columns = df.columns.str.strip()
    return df


def buscar_columna(columnas, palabras_clave):
    """Busca una columna ignorando mayúsculas, tildes y espacios extras."""
    for col in columnas:
        col_limpia = str(col).upper().replace('Á','A').replace('É','E').replace('Í','I').replace('Ó','O').replace('Ú','U').strip()
        # Verifica si TODAS las palabras clave están en el nombre de la columna
        if all(clave in col_limpia for clave in palabras_clave):
            return col
    return None

# ============================================================
# 3. INTERFAZ Y LÓGICA DE STREAMLIT 
# ============================================================

if 'df_aforos' not in st.session_state: st.session_state.df_aforos = None
if 'df_geo' not in st.session_state: st.session_state.df_geo = None
if 'perfil_data' not in st.session_state: st.session_state.perfil_data = None
if 'manning_data' not in st.session_state: st.session_state.manning_data = None

# Inicialización de curvas y errores para comparación
if 'manning_curve' not in st.session_state: st.session_state.manning_curve = None
if 'stevens_curve' not in st.session_state: st.session_state.stevens_curve = None
if 'av_curve' not in st.session_state: st.session_state.av_curve = None
if 'manning_error' not in st.session_state: st.session_state.manning_error = None
if 'stevens_error' not in st.session_state: st.session_state.stevens_error = None
if 'av_error' not in st.session_state: st.session_state.av_error = None
# Inicializar banda de error global (por defecto Estable 10%)
if 'banda_error_global' not in st.session_state:
    st.session_state.banda_error_global = 10.0

if 'opts_modelos_man' not in st.session_state: 
    st.session_state.opts_modelos_man = {"lineal": True, "exp": True, "log": True, "pot": True}

st.title("🌊 Curva de Gasto")

# --- CARGA DE ARCHIVOS --- 
with st.sidebar:
    st.header("Archivos Base")
    file_aforos = st.file_uploader("1. Consolidado Aforos (.xls/xlsx)", type=["xlsx", "xls", "xlsm"])
    file_perfil = st.file_uploader("2. Perfil Transversal (.xls/xlsx)", type=["xlsx", "xls", "xlsm"])
    
    st.markdown("---")
    if st.button("🔄 Nuevo análisis (cambiar estación)", use_container_width=True):
        # Lista de claves a reiniciar (todas las relacionadas con la estación actual)
        keys_to_reset = [
            'df_aforos', 'perfil_data', 'df_geo', 
            'manning_data', 'stevens_data', 'av_data',
            'manning_curve', 'stevens_curve', 'av_curve',
            'manning_error', 'stevens_error', 'av_error',
            'manning_error_sigma', 'stevens_error_sigma', 'av_error_sigma',
            'manning_edited_df', 'stevens_edited_df', 'av_edited_df',
            'temp_aforos_activos', 'temp_perfil_activos', 'perfil_puntos_activos',
            'df_aforos_activos', 'h0_seleccionados', 'metodo_definitivo',"Curva_selec", "usar_auto"
        ]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ==========================================
# SECCIÓN EN LA BARRA LATERAL: GUARDAR/CARGAR
# ==========================================
with st.sidebar.expander("💾 Guardar / Cargar Proyecto", expanded=False):
    st.markdown("Guarda tu progreso actual o carga un análisis previo.")
    
    # --- 1. BOTÓN PARA GUARDAR (EXPORTAR) ---
    # AMPLIAMOS ESTA LISTA para incluir TODO tu progreso (Manning, Stevens, temporales, etc.)
    claves_a_guardar = [
        'df_aforos', 'df_aforos_activos', 'temp_aforos_activos', 'last_df_size',
        'perfil_data', 'perfil_puntos_activos', 'temp_perfil_activos',
        'df_geo', 'codigo_estacion',
        'manning_data', 'manning_curve', 'manning_error', 'manning_error_sigma', 'manning_edited_df', 'opts_modelos_man',
        'stevens_data', 'stevens_curve', 'stevens_error', 'stevens_error_sigma', 'stevens_edited_df',
        'av_data', 'av_curve', 'av_error', 'av_error_sigma', 'av_edited_df',
        'h0_seleccionados', 'metodo_definitivo',
        'fecha_inicio', 'fecha_fin', 'fuente_k_radio','fuente_k_stevens','fuente_v_radio',
        'metodo_select_manning','metodo_select_stevens', 'metodo_select_av',
        'tipo_paso', 'paso_fijo','paso_fino','paso_grueso',
        'usar_auto','H_max_manual','banda_error_global', "Curva_selec", 
        'manning_h_quiebre', 'manning_modelo_inf', 'manning_modelo_sup',
        'stevens_h_quiebre', 'stevens_modelo_inf', 'stevens_modelo_sup',
        'av_h_quiebre', 'av_modelo_inf', 'av_modelo_sup', 'nivel_interes_1', 'nivel_interes_2',
        # Añade también las claves de los radios de H0 si decides mantenerlas
    ]
    
    datos_sesion = {k: st.session_state[k] for k in claves_a_guardar if k in st.session_state}
    
    if datos_sesion:
        archivo_pkl = pickle.dumps(datos_sesion)
        
        # --- GENERAR NOMBRE DINÁMICO ---
        nombre_estacion = "Estacion_Desconocida"
        if 'perfil_data' in st.session_state and st.session_state.perfil_data is not None:
            # Rescatamos el nombre y reemplazamos espacios por guiones bajos
            nombre_estacion = str(st.session_state.perfil_data.get('estacion', 'Estacion')).replace(" ", "_")
            
        fecha_hoy = datetime.datetime.now().strftime("%Y%m%d")
        nombre_archivo_dinamico = f"Proyecto_{nombre_estacion}_{fecha_hoy}.pkl"
        
        st.download_button(
            label="⬇️ Descargar Análisis (.pkl)",
            data=archivo_pkl,
            file_name=nombre_archivo_dinamico,
            mime="application/octet-stream",
            use_container_width=True,
            type="primary"
        )
    else:
        st.info("Aún no hay datos cargados para guardar.")

    st.markdown("---")

    # --- 2. BOTÓN PARA CARGAR (IMPORTAR) ---
    archivo_subido = st.file_uploader("📂 Cargar Análisis (.pkl)", type=["pkl"], key="uploader_proyecto")
    
    if archivo_subido is not None:
        if st.button("🔄 Restaurar Sesión", use_container_width=True):
            try:
                datos_cargados = pickle.loads(archivo_subido.getvalue())
                for clave, valor in datos_cargados.items():
                    st.session_state[clave] = valor
                
                st.success(f"¡Proyecto restaurado con éxito!")
                st.rerun() 
            except Exception as e:
                st.error(f"Error al cargar el archivo: {e}")

# --- PROCESAMIENTO AFOROS ---
if file_aforos is not None and st.session_state.df_aforos is None:
    try:
        df_raw_af = cargar_excel_seguro(file_aforos)
        fila_enc = None
        for i, row in df_raw_af.iterrows():
            if row.astype(str).str.contains(r"NIVEL MEDIO \(cms\)").any():
                fila_enc = i
                break
                
        if fila_enc is not None:
            df_aforos = df_raw_af.iloc[fila_enc+1:].reset_index(drop=True)
            nombres_crudas = df_raw_af.iloc[fila_enc].tolist()
            nombres_limpios = []
            
            for idx, col in enumerate(nombres_crudas):
                if pd.isna(col) or str(col).strip().lower() == 'nan':
                    nombres_limpios.append(f"Col_Vacia_{idx}")
                else:
                    nombres_limpios.append(str(col).strip())
            
            nombres_finales = []
            vistos = {}
            for col in nombres_limpios:
                if col in vistos:
                    vistos[col] += 1
                    nombres_finales.append(f"{col}_{vistos[col]}")
                else:
                    vistos[col] = 0
                    nombres_finales.append(col)
            
            df_aforos.columns = nombres_finales
            
            # Convertir a numérico
            df_aforos["H_m"] = pd.to_numeric(df_aforos["NIVEL MEDIO (cms)"], errors='coerce') / 100.0
            df_aforos["CAUDAL TOTAL (m3/s)"] = pd.to_numeric(df_aforos["CAUDAL TOTAL (m3/s)"], errors='coerce')
            
            # Eliminar filas con nivel o caudal nulos
            df_aforos = df_aforos.dropna(subset=["H_m", "CAUDAL TOTAL (m3/s)"])
            
            # Filtrar caudales positivos (descartar ceros o negativos)
            antes = len(df_aforos)
            df_aforos = df_aforos[df_aforos["CAUDAL TOTAL (m3/s)"] > 0]
            despues = len(df_aforos)
            if antes - despues > 0:
                st.sidebar.info(f"Se eliminaron {antes - despues} aforos con caudal no positivo.")
            
            # Limpiar columnas vacías
            columnas_limpias = [col for col in df_aforos.columns if not col.startswith("Col_Vacia")]
            st.session_state.df_aforos = df_aforos[columnas_limpias]
            
    except Exception as e:
        st.sidebar.error(f"Error procesando Aforos: {e}")

# --- PROCESAMIENTO DEL PERFIL ---
if file_perfil is not None and st.session_state.perfil_data is None:
    try:
        cota, x, y, est, cod, fecha, descs, tabs = leer_perfil_transversal_completo(file_perfil)
        st.session_state.perfil_data = {
            'cota_cero': cota, 'abscisas': x, 'cotas': y,
            'estacion': est, 'codigo': cod, 'fecha': fecha,
            'descripciones': descs, 'tablas': tabs
        }
        st.session_state.codigo_estacion = cod  
    except Exception as e:
        st.sidebar.error(f"Error procesando el Perfil: {e}")

# --- PESTAÑAS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8= st.tabs(["📊 Resumen Aforos", "📐 Geometría", "📉 Método Manning", "📉 Método Stevens", "📉 Método Área-Velocidad", "📉 Minimos", "📊 Comparación", "🔍 Historico"])

# ================== PESTAÑA 1: RESUMEN AFOROS ==================
with tab1:
    if st.session_state.df_aforos is not None:
        st.subheader("Datos de Aforos Extraídos")

        # 1. Asegurar que existe la columna 'Activo' en el DataFrame global
        if 'Activo' not in st.session_state.df_aforos.columns:
            st.session_state.df_aforos.insert(0, 'Activo', True)

        df = st.session_state.df_aforos.copy()

        # 2. Inicializar estado temporal o RESETEARLO si se subió un archivo nuevo
        if 'temp_aforos_activos' not in st.session_state or 'last_df_size' not in st.session_state or st.session_state.last_df_size != len(df):
            st.session_state.temp_aforos_activos = df.copy()
            st.session_state.last_df_size = len(df)

        # 3. Buscar la columna de fecha para habilitar el filtro
        col_fecha = next((col for col in df.columns if "FECHA" in str(col).upper()), None)

        if col_fecha:
            df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce', dayfirst=True)
            st.session_state.temp_aforos_activos[col_fecha] = pd.to_datetime(
                st.session_state.temp_aforos_activos[col_fecha], errors='coerce', dayfirst=True
            )
            fechas_validas = df[col_fecha].dropna()
            
            if not fechas_validas.empty:
                min_date = fechas_validas.min().date()
                max_date = fechas_validas.max().date()

                st.markdown("**📅 Filtro Temporal de Aforos:**")
                
                with st.container():
                    col_f1, col_f2 = st.columns(2)
                    if "fecha_inicio" not in st.session_state:
                        st.session_state.fecha_inicio = min_date
                    if "fecha_fin" not in st.session_state:
                        st.session_state.fecha_fin = max_date

                    with col_f1:
                        f_inicio = st.date_input("Fecha Inicio:", min_value=min_date, max_value=max_date, key="fecha_inicio")
                    with col_f2:
                        f_fin = st.date_input("Fecha Fin:", min_value=min_date, max_value=max_date, key="fecha_fin")

                if f_inicio > f_fin:
                    st.error("⚠️ La fecha de inicio no puede ser mayor a la fecha de fin.")
                    df_mostrar_temp = st.session_state.temp_aforos_activos
                else:
                    mask_fecha = (st.session_state.temp_aforos_activos[col_fecha].dt.date >= f_inicio) & \
                                 (st.session_state.temp_aforos_activos[col_fecha].dt.date <= f_fin)
                    df_mostrar_temp = st.session_state.temp_aforos_activos[mask_fecha]
            else:
                df_mostrar_temp = st.session_state.temp_aforos_activos
        else:
            st.warning("No se detectó automáticamente una columna de 'FECHA'. Filtrado por calendario deshabilitado.")
            df_mostrar_temp = st.session_state.temp_aforos_activos

        st.caption("Selecciona o desmarca en la columna 'Activo' los aforos que deseas usar en los cálculos de los modelos.")

        # --- AQUÍ EMPIEZA LA MAGIA DEL FORMULARIO PARA AFOROS ---
        columnas_protegidas = [col for col in df_mostrar_temp.columns if col != "Activo"]
        
        with st.form("form_edicion_aforos"):
            # 4. Editor interactivo dentro del formulario (no recarga al hacer clic)
            edited_temp = st.data_editor(
                df_mostrar_temp,
                column_config={
                    "Activo": st.column_config.CheckboxColumn("Activo", default=True),
                    col_fecha: st.column_config.DateColumn("Fecha", format="DD/MM/YYYY") if col_fecha else None
                },
                disabled=columnas_protegidas,
                hide_index=True,
                use_container_width=True,
                key="editor_aforos_temp"
            )

            # 5. Botones de acción dentro del formulario
            col_apply, col_cancel = st.columns(2)
            with col_apply:
                # Al presionar este botón, Streamlit procesa todos los clics acumulados
                aplicar_aforos = st.form_submit_button("✅ Aplicar cambios en aforos", use_container_width=True)
            
            if aplicar_aforos:
                st.session_state.temp_aforos_activos.update(edited_temp)
                aforos_para_modelos = edited_temp[edited_temp['Activo'] == True].copy()
                st.session_state.df_aforos_activos = aforos_para_modelos
                st.session_state.df_aforos.update(aforos_para_modelos[['Activo']])
                st.session_state.flag_actualizar_modelos = True
                
                st.success("Cambios aplicados. Solo se usarán los aforos seleccionados.")

        # Resumen Visual
        activos_visibles = len(edited_temp[edited_temp['Activo'] == True])
        totales_visibles = len(edited_temp)
        totales_absolutos = len(st.session_state.df_aforos)
        
        st.info(f"📊 **Aforos listos para cálculo en este periodo:** {activos_visibles} (de {totales_visibles} en el rango | {totales_absolutos} en total histórico).")

    else:
        st.info("Sube el archivo de Consolidado de Aforos en el panel lateral.")

    st.markdown("---")
    st.subheader("Puntos del Perfil Transversal")

    if st.session_state.perfil_data is not None:
        p_data = st.session_state.perfil_data
        df_perfil_puntos = pd.DataFrame({
            'Abscisa (m)': p_data['abscisas'],
            'Cota (m)': p_data['cotas'],
            'Descripción': p_data['descripciones'],
            'Tabla': p_data['tablas']
        })
        
        if 'temp_perfil_activos' not in st.session_state:
            st.session_state.temp_perfil_activos = [True] * len(df_perfil_puntos)

        df_perfil_edit = df_perfil_puntos.copy()
        df_perfil_edit['Activo'] = st.session_state.temp_perfil_activos

        # --- AQUÍ APLICAMOS EL MISMO FORMULARIO PARA EL PERFIL ---
        with st.form("form_edicion_perfil"):
            edited_perfil = st.data_editor(
                df_perfil_edit,
                column_config={
                    "Activo": st.column_config.CheckboxColumn("Incluir", default=True),
                    "Abscisa (m)": st.column_config.NumberColumn("Abscisa", disabled=True, format="%.3f"),
                    "Cota (m)": st.column_config.NumberColumn("Cota", disabled=True, format="%.3f"),
                    "Descripción": st.column_config.TextColumn("Descripción", disabled=True),
                    "Tabla": st.column_config.TextColumn("Origen", disabled=True)
                },
                disabled=False,
                hide_index=True,
                use_container_width=True,
                key="perfil_puntos_editor_temp"
            )

            col_apply_perf, col_cancel_perf = st.columns(2)
            with col_apply_perf:
                aplicar_perfil = st.form_submit_button("✅ Aplicar cambios en perfil", use_container_width=True)
            
            if aplicar_perfil:
                st.session_state.temp_perfil_activos = edited_perfil['Activo'].tolist()
                st.session_state.perfil_puntos_activos = st.session_state.temp_perfil_activos.copy()
                st.success("Cambios aplicados al perfil.")

        st.caption("Selecciona los puntos que deseas incluir en la geometría (los desactivados no se usarán en cálculos).")
    else:
        st.info("Una vez que cargues el Perfil Transversal, aquí aparecerán los puntos para que puedas seleccionarlos.")

# ================== PESTAÑA 2: GEOMETRÍA ==================
with tab2:
    if st.session_state.perfil_data is not None:
        p_data = st.session_state.perfil_data
        
        # Obtener puntos activos de la selección en pestaña 1
        if 'perfil_puntos_activos' in st.session_state and len(st.session_state.perfil_puntos_activos) == len(p_data['abscisas']):
            mask = st.session_state.perfil_puntos_activos
            abscisas_filt = p_data['abscisas'][mask]
            cotas_filt = p_data['cotas'][mask]
            desc_filt = np.array(p_data['descripciones'])[mask]  # para NAOI
            st.info(f"Usando {np.sum(mask)} de {len(mask)} puntos del perfil (según selección en 'Resumen Aforos').")
        else:
            abscisas_filt = p_data['abscisas']
            cotas_filt = p_data['cotas']
            desc_filt = p_data['descripciones']
            st.warning("No hay selección de puntos activos. Usando todos los puntos del perfil.")
        
        # Verificar que haya puntos activos
        if len(abscisas_filt) == 0:
            st.error("No hay puntos activos en el perfil. Activa al menos un punto en la pestaña 'Resumen Aforos'.")
            st.stop()
        
        # Métricas de cabecera
        col1, col2, col3 = st.columns(3)
        col1.metric("Estación", f"{p_data['estacion']} ({p_data['codigo']})")
        col2.metric("Fecha Perfil", p_data['fecha'])
        col3.metric("Cota Cero", f"{p_data['cota_cero']:.3f} m")
        
        # --- Obtener puntos NAOI ---
        puntos_naoi = []
        for i, desc in enumerate(desc_filt):
            if desc and "NAOI" in desc.upper():
                puntos_naoi.append({'abscisa': abscisas_filt[i], 'cota': cotas_filt[i]})
        df_naoi = pd.DataFrame(puntos_naoi) if puntos_naoi else None

        # --- Calcular nivel de agua a partir del primer punto NAOI (o promedio) ---
        nivel_agua = None
        if df_naoi is not None and not df_naoi.empty:
            # Usar la cota del primer punto NAOI como referencia
            cota_naoi = df_naoi['cota'].iloc[0]
            nivel_agua = cota_naoi - p_data['cota_cero']
            st.info(f"Nivel de agua NAOI detectado: {nivel_agua:.2f} m (cota {cota_naoi:.2f} m)")
        
        # --- Determinar nivel máximo para el gráfico inicial (persistencia) ---
        nivel_max_inicial = None
        if st.session_state.get('df_geo') is not None and not st.session_state.df_geo.empty:
            nivel_max_inicial = st.session_state.df_geo['H (m)'].max()

        # --- Función para crear el gráfico del perfil ---
        def crear_grafico_perfil(abscisas, cotas, cota_cero, nivel_max=None, nivel_agua=None, puntos_naoi=None):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=abscisas, 
                y=cotas, 
                mode='lines+markers',
                name='Terreno',
                line=dict(color='#2c3e50', width=3),
                marker=dict(color='#2c3e50', size=5, symbol='circle'),
                fill='tozeroy',
                fillcolor='rgba(44, 62, 80, 0.2)'
            ))
            fig.add_hline(
                y=cota_cero, 
                line_dash="dash", 
                line_color="#e74c3c", 
                line_width=2.5,
                annotation_text=f"Cota Cero = {cota_cero:.3f} m",
                annotation_position="bottom right",
                annotation_font_size=13,
                annotation_font_color="#e74c3c",
                annotation_font_family="Arial Black"
            )
            if nivel_max is not None:
                cota_max = cota_cero + nivel_max
                fig.add_hline(
                    y=cota_max,
                    line_dash="dot",
                    line_color="#2ecc71",
                    line_width=2.5,
                    annotation_text=f"Nivel máx. calculado: {nivel_max:.2f} m",
                    annotation_position="top right",
                    annotation_font_size=12,
                    annotation_font_color="#2ecc71",
                    annotation_font_family="Arial"
                )
            if nivel_agua is not None:
                cota_agua = cota_cero + nivel_agua
                fig.add_hline(
                    y=cota_agua,
                    line_dash="solid",
                    line_color="#3498db",
                    line_width=2,
                    annotation_text=f"Nivel agua: {nivel_agua:.2f} m",
                    annotation_position="bottom left",
                    annotation_font_size=12,
                    annotation_font_color="#3498db"
                )
            if puntos_naoi is not None and not puntos_naoi.empty:
                fig.add_trace(go.Scatter(
                    x=puntos_naoi['abscisa'],
                    y=puntos_naoi['cota'],
                    mode='markers',
                    name='NAOI',
                    marker=dict(color='#f1c40f', size=12, symbol='star', line=dict(color='black', width=1))
                ))
            fig.update_layout(
                title="Sección Transversal - Perfil del Terreno",
                xaxis_title="Abscisa (m)",
                yaxis_title="Cota (m)",
                hovermode="x unified",
                margin=dict(l=60, r=40, t=80, b=60),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=12),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.15)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.15)'),
                showlegend=True,
                legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.5)')
            )
            y_min = min(cotas.min(), cota_cero) - 1
            y_max = max(cotas.max(), cota_cero) + 1
            if nivel_max is not None:
                y_max = max(y_max, cota_cero + nivel_max + 1)
            if nivel_agua is not None:
                y_max = max(y_max, cota_agua + 1)
            fig.update_yaxes(range=[y_min, y_max])
            return fig
        
        # Mostrar gráfico con nivel máximo persistente y nivel de agua NAOI (si existe)
        fig_perfil = crear_grafico_perfil(
            abscisas_filt, cotas_filt, p_data['cota_cero'],
            nivel_max=nivel_max_inicial,
            nivel_agua=nivel_agua,
            puntos_naoi=df_naoi
        )
        placeholder_perfil = st.plotly_chart(fig_perfil, use_container_width=True, key="perfil_plot")
        
        st.markdown("---")
        
        # --- Cálculo de niveles mínimos y máximos ---
        cota_min_lecho = float(cotas_filt.min())
        distancia_lecho = float(cota_min_lecho - p_data['cota_cero']) 
        H_min_calc = 0.0 

        porcentaje_borde = 15
        n_puntos = len(abscisas_filt)
        n_borde = max(1, int(n_puntos * porcentaje_borde / 100))

        indices_orden = np.argsort(abscisas_filt)
        abscisas_ordenadas = abscisas_filt[indices_orden]
        cotas_ordenadas = cotas_filt[indices_orden]

        cotas_izq = cotas_ordenadas[:n_borde]
        cota_izq_max = float(np.max(cotas_izq))

        cotas_der = cotas_ordenadas[-n_borde:]
        cota_der_max = float(np.max(cotas_der))

        cota_desborde = float(min(cota_izq_max, cota_der_max))
        H_max_auto = float(cota_desborde - p_data['cota_cero'])
        
        if not (np.isfinite(distancia_lecho) and np.isfinite(H_max_auto)):
            st.error("No se pudieron calcular niveles válidos. Verifica los puntos activos del perfil.")
            st.stop()
        
        st.info(f"**Distancia de Cota Cero al lecho:** {distancia_lecho:.2f} m | **Nivel máximo por desbordamiento (H):** {H_max_auto:.2f} m\n\n"
                f"Cota máxima en el {porcentaje_borde}% izquierdo: {cota_izq_max:.2f} m, derecho: {cota_der_max:.2f} m")

        # --- Opciones de Desbordamiento y Extrapolación ---
        st.markdown("### 🌊 Manejo de Desbordamientos")
        tipo_extrapolacion = st.radio(
            "¿Cómo manejar el cálculo si el nivel supera la topografía medida?",
            options=[
                "Cortar en el desbordamiento (Sin muros)", 
                "Muros físicos (Suma perímetro/fricción)", 
                "Área virtual (Muros sin fricción)"
            ],
            index=0, 
            help="Define cómo se comporta la geometría cuando el agua supera la cota máxima del terreno.",
            key="tipo_extrapolacion_geo"
        )

        if tipo_extrapolacion == "Cortar en el desbordamiento (Sin muros)":
            modo_muro = "ninguno"
        elif tipo_extrapolacion == "Muros físicos (Suma perímetro/fricción)":
            modo_muro = "con_friccion"
        else:
            modo_muro = "sin_friccion"

        # --- Opciones de rango ---
        usar_auto = st.checkbox("Usar nivel máximo automático (cota mínima de desborde)", value=(modo_muro=="ninguno"), key="usar_auto_geo")
        
        if usar_auto:
            H_max = H_max_auto
            st.caption(f"Se usará H_max = {H_max:.2f} m")
        else:
            min_val = H_min_calc 
            cota_max_top = float(np.nanmax(cotas_filt))
            max_val_calc = float(cota_max_top - p_data['cota_cero'])
            
            limite_slider = max_val_calc + 50.0 if modo_muro != "ninguno" else max_val_calc

            if not np.isfinite(max_val_calc):
                max_val = H_max_auto
                st.warning("El valor máximo calculado no es finito. Se usará el nivel de desbordamiento como límite.")
            else:
                max_val = limite_slider
                
            default_val = H_max_auto
            if default_val < min_val: default_val = min_val
            elif default_val > max_val: default_val = max_val
                
            H_max_manual = st.number_input(
                "Nivel máximo deseado (m):",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=0.1,
                format="%.2f", key="H_max_manual_geo" 
            )
            H_max = H_max_manual
            
            if H_max > H_max_auto and modo_muro == "ninguno":
                st.warning(f"⚠️ El nivel ({H_max:.2f} m) supera el desbordamiento. Para niveles tan altos, se recomienda activar los muros de extrapolación arriba.")
            elif H_max > H_max_auto and modo_muro != "ninguno":
                st.success(f"✅ Calculando niveles extremos usando: {tipo_extrapolacion}")

        # --- Opciones de paso ---
        tipo_paso = st.radio("Tipo de paso:", options=["Fijo", "Progresivo"], index=0, horizontal=True, key="tipo_paso_geo") 
        
        if tipo_paso == "Fijo":
            paso_h = st.number_input("Paso de Nivel (H) en metros:", value=0.1, min_value=0.05, step=0.05, key="paso_fijo_geo") 
        else:
            col_paso1, col_paso2 = st.columns(2)
            with col_paso1:
                paso_fino = st.number_input("Paso fino (dentro de aforos):", value=0.1, min_value=0.01, step=0.05, key="paso_fino_geo") 
            with col_paso2:
                paso_grueso = st.number_input("Paso grueso (fuera de aforos):", value=0.5, min_value=0.1, step=0.1, key="paso_grueso_geo") 
            if 'df_aforos_activos' in st.session_state and st.session_state.df_aforos_activos is not None and not st.session_state.df_aforos_activos.empty:
                H_aforos = st.session_state.df_aforos_activos["H_m"].values
                H_min_aforos = H_aforos.min()
                H_max_aforos = H_aforos.max()
                st.caption(f"Rango de aforos activos: {H_min_aforos:.2f} - {H_max_aforos:.2f} m")
            else:
                H_min_aforos = H_min_calc
                H_max_aforos = H_max_auto
                st.warning("No hay aforos activos, se usará el rango completo con paso fino.")
        
        # --- BOTÓN CORREGIDO CON KEY ÚNICA ---
        if st.button("Generar Tabla de Geometría", key="btn_generar_tabla_geo"):
            if tipo_paso == "Fijo":
                H_vals = np.arange(H_min_calc, H_max + paso_h, paso_h)
            else:
                H_grueso = np.arange(H_min_calc, H_max + paso_grueso, paso_grueso)
                inicio_fino = H_min_calc 
                fin_fino = np.ceil(H_max_aforos / paso_fino) * paso_fino
                H_fino = np.arange(inicio_fino, fin_fino + paso_fino, paso_fino)
                H_vals = np.concatenate([H_grueso, H_fino])
            
            H_vals = np.unique(np.round(H_vals, 3))
            H_vals = H_vals[H_vals <= H_max]

            # --- BLOQUE DE CÁLCULO (SIN DUPLICADOS) ---
            if len(H_vals) == 0:
                st.error("No hay niveles en el rango seleccionado. Ajusta el paso o el nivel máximo.")
            else:
                data_geo = []
                for H in H_vals:
                    # Aplicamos modo_muro
                    Am = area_mojada(abscisas_filt, cotas_filt, p_data['cota_cero'], H, modo_muro)
                    if Am <= 0: continue 
                    
                    Wh = ancho_superficial(abscisas_filt, cotas_filt, p_data['cota_cero'], H, modo_muro)
                    Pm = perimetro_mojado(abscisas_filt, cotas_filt, p_data['cota_cero'], H, modo_muro)
                    
                    D = Am / Wh if Wh > 0 else 0
                    R = Am / Pm if Pm > 0 else 0
                    relacion_RD = R / D if D > 0 else np.nan
                    
                    data_geo.append({
                        "H (m)": H, 
                        "Wh (m)": Wh, 
                        "Am (m2)": Am, 
                        "D (m)": D, 
                        "Pm (m)": Pm, 
                        "R (m)": R, 
                        "R2/3": R ** (2/3),
                        "R/D": relacion_RD
                    })
                
                st.session_state.df_geo = pd.DataFrame(data_geo)
                nivel_max_generado = H_vals[-1]
                st.success(f"Tabla generada correctamente. Primer nivel: {H_vals[0]:.2f} m, último nivel: {nivel_max_generado:.2f} m")
                
                fig_perfil_actualizado = crear_grafico_perfil(
                    abscisas_filt, cotas_filt, p_data['cota_cero'],
                    nivel_max=nivel_max_generado,
                    nivel_agua=nivel_agua,
                    puntos_naoi=df_naoi
                )
                placeholder_perfil.plotly_chart(fig_perfil_actualizado, use_container_width=True)
        
        # --- Mostrar tabla de geometría y validación R/D ---
        if 'df_geo' in st.session_state and st.session_state.df_geo is not None:
            df_geo = st.session_state.df_geo
            if "R/D" in df_geo.columns:
                valores_rd = df_geo["R/D"].dropna()
                if len(valores_rd) > 0:
                    min_rd = valores_rd.min()
                    max_rd = valores_rd.max()
                    mean_rd = valores_rd.mean()
                    
                    col_rd1, col_rd2, col_rd3, col_rd4 = st.columns(4)
                    with col_rd1:
                        st.metric("Relación R/D mín.", f"{min_rd:.3f}")
                    with col_rd2:
                        st.metric("Relación R/D máx.", f"{max_rd:.3f}")
                    with col_rd3:
                        st.metric("Relación R/D prom.", f"{mean_rd:.3f}")
                    
                    if min_rd < 0.8 or max_rd > 1.2:
                        st.warning("⚠️ La relación R/D se aleja significativamente de 1. El método Stevens (que asume R ≈ D) podría no ser adecuado para esta sección.")
                    elif min_rd < 0.9 or max_rd > 1.1:
                        st.info("ℹ️ La relación R/D está moderadamente cerca de 1. El método Stevens puede ser aceptable, pero verifique los resultados.")
                    else:
                        st.success("✅ La relación R/D es cercana a 1 en todo el rango. El método Stevens es aplicable.")
            
            st.dataframe(st.session_state.df_geo, use_container_width=True)
            if len(st.session_state.df_geo) > 0:
                nivel_max_generado_final = st.session_state.df_geo["H (m)"].max()
                st.metric("Nivel máximo generado", f"{nivel_max_generado_final:.2f} m")

# ================== PESTAÑA 3: MÉTODO MANNING ==================
with tab3:
    if st.session_state.df_aforos is None or st.session_state.df_geo is None:
        st.warning("Procesa los Aforos y genera la Tabla de Geometría primero.")
    else:
        # --- ENCABEZADO INFORMATIVO DEL MÉTODO ---
        with st.expander("📘 **Fundamento y consideraciones del método Manning**", expanded=False):
            st.markdown("""
            **Ecuación de Manning:** V = (1/n) * R^(2/3) * S^(1/2)  
            - V: velocidad media (m/s)  
            - n: coeficiente de rugosidad  
            - R: radio hidráulico (m)  
            - S: pendiente hidráulica  

            Para niveles altos se asume que S^(1/2)/n es constante. Definiendo K = S^(1/2)/n, la ecuación se simplifica a V = K * R^(2/3).

            En esta herramienta se ajusta K en función del nivel H usando los aforos activos, y luego se extrapola aplicando la geometría calculada.

            > *Limitación:* La validez de la extrapolación depende de que la relación S^(1/2)/n se mantenga realmente constante en el rango de niveles altos. Se recomienda revisar los resultados con los aforos disponibles.
            """)
        
        # --- AQUÍ ESTÁ LA CORRECCIÓN: Escuchamos la alarma de la Pestaña 1 ---
        necesita_actualizar = (
            st.session_state.get('manning_data') is None 
            or st.session_state.get('flag_actualizar_modelos', False)
        )

        if necesita_actualizar:
            # 1. RECIBIR AFOROS FILTRADOS (Ya vienen limpios de la Pestaña 1)
            if 'df_aforos_activos' in st.session_state:
                df_a = st.session_state.df_aforos_activos.copy()
            else:
                df_a = st.session_state.df_aforos.copy()
                
            p_data = st.session_state.perfil_data

            col_v = buscar_columna(df_a.columns, ["VELOC", "MEDIA"])
            col_rh = buscar_columna(df_a.columns, ["RH"])
            
            if not col_v: st.sidebar.error("No se encontró la columna de Velocidad Media en Aforos.")
            if not col_rh: st.sidebar.error("No se encontró la columna de RH en Aforos.")
            
            # 2. FILTRAR PERFIL TOPOGRÁFICO ACTIVO
            if 'perfil_puntos_activos' in st.session_state:
                mascara_perfil = np.array(st.session_state.perfil_puntos_activos, dtype=bool)
                abscisas_activas = np.array(p_data['abscisas'])[mascara_perfil]
                cotas_activas = np.array(p_data['cotas'])[mascara_perfil]
            else:
                abscisas_activas = np.array(p_data['abscisas'])
                cotas_activas = np.array(p_data['cotas'])

            datos_manning = []
            for i, row in df_a.iterrows():
                id_aforo = row.get("NO.", i+1)
                H = row["H_m"]
                Q = row["CAUDAL TOTAL (m3/s)"]
                
                A_per = area_mojada(abscisas_activas, cotas_activas, p_data['cota_cero'], H)
                P_per = perimetro_mojado(abscisas_activas, cotas_activas, p_data['cota_cero'], H)
                
                K_per = 0
                if P_per > 0 and A_per > 0:
                    R23_per = (A_per / P_per) ** (2/3)
                    K_per = (Q / A_per) / R23_per if R23_per > 0 else 0
                
                v_af = row.get(col_v, np.nan)
                rh = row.get(col_rh, np.nan)
                
                K_af = 0
                r23_af = np.nan
                if pd.notna(v_af) and pd.notna(rh) and rh > 0:
                    r23_af = rh ** (2/3)
                    K_af = v_af / r23_af
                else:
                    st.sidebar.warning(f"Aforo {id_aforo}: faltan V_af o RH, K_af = 0")
                
                if K_per > 0 or K_af > 0:
                    datos_manning.append({
                        "Incluir": True, 
                        "ID": id_aforo, 
                        "H": H, 
                        "Q": Q, 
                        "V_af": v_af,
                        "RH": rh,
                        "R23_af": r23_af,
                        "K_af": K_af,
                        "K_per": K_per,
                    })
            
            st.session_state.manning_data = pd.DataFrame(datos_manning)
            
            # --- APAGAMOS LA ALARMA Y RESETEAMOS LA TABLA VISUAL ---
            st.session_state.flag_actualizar_modelos = False
            if 'manning_edited_df' in st.session_state:
                del st.session_state['manning_edited_df']

        idx_fuente = 0
        if "fuente_k_radio" in st.session_state:
            if st.session_state.fuente_k_radio == "K del perfil":
                idx_fuente = 1

        fuente_k = st.radio(
            "Fuente de K:",
            options=["K de aforos", "K del perfil"],
            index=idx_fuente, 
            horizontal=True,
            key="fuente_k_radio"
        )

        df_edit = st.session_state.manning_data.copy()
        
        if fuente_k == "K de aforos":
            df_edit["K"] = df_edit["K_af"]
        else:
            df_edit["K"] = df_edit["K_per"]

        # --- PERSISTENCIA: Inicializar DataFrame editado y opciones de modelos ---
        if 'manning_edited_df' not in st.session_state:
            default_df = df_edit.copy()
            default_df['Incluir'] = True
            st.session_state.manning_edited_df = default_df

        if 'opts_modelos_man' not in st.session_state:
            st.session_state.opts_modelos_man = {"lineal": False, "exp": False, "log": False, "pot": False}

        # --- Inicializar banda de error global ---
        if 'banda_error_global' not in st.session_state:
            st.session_state.banda_error_global = 15.0

        # --- Botón con popover (esquina superior izquierda) ---
        col_btn, _ = st.columns([0.15, 0.85])
        with col_btn:
            with st.popover("⚙️ Controles"):
                with st.form(key="manning_form"):
                    st.caption("Filtro de Aforos")
                    edited = st.data_editor(
                        st.session_state.manning_edited_df[["Incluir", "ID", "H", "Q"]],
                        column_config={
                            "Incluir": st.column_config.CheckboxColumn("Activo", default=True),
                            "ID": st.column_config.NumberColumn("ID", disabled=True),
                            "H": st.column_config.NumberColumn("H (m)", disabled=True, format="%.2f"),
                            "Q": st.column_config.NumberColumn("Q (m³/s)", disabled=True, format="%.2f"),
                        },
                        disabled=False,
                        hide_index=True,
                        use_container_width=True,
                        key="manning_editor"
                    )

                    st.caption("Modelos de ajuste")
                    lineal = st.checkbox("Lineal", value=st.session_state.opts_modelos_man["lineal"])
                    exp = st.checkbox("Exponencial", value=st.session_state.opts_modelos_man["exp"])
                    log = st.checkbox("Logarítmica", value=st.session_state.opts_modelos_man["log"])
                    pot = st.checkbox("Potencial", value=st.session_state.opts_modelos_man["pot"])

                    st.caption("Configuración General")
                    opciones_banda = ["Estable (10%)", "Inestable (15%)"]
                    indice_actual = 0 if st.session_state.banda_error_global == 10 else 1
                    tipo_seccion = st.radio(
                        "Tipo de sección:",
                        options=opciones_banda,
                        index=indice_actual,
                        horizontal=True
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        submitted = st.form_submit_button("Aplicar")
                    with col2:
                        cancelled = st.form_submit_button("Cancelar")

                    if submitted:
                        st.session_state.manning_edited_df = edited
                        st.session_state.opts_modelos_man["lineal"] = lineal
                        st.session_state.opts_modelos_man["exp"] = exp
                        st.session_state.opts_modelos_man["log"] = log
                        st.session_state.opts_modelos_man["pot"] = pot
                        if tipo_seccion == "Estable (10%)":
                            st.session_state.banda_error_global = 10.0
                        else:
                            st.session_state.banda_error_global = 15.0
                        st.rerun()
                    elif cancelled:
                        st.rerun()

        # --- Gráficas (siempre ancho completo) ---
        col_k, col_q = st.columns(2)
        with col_k:
            st.subheader("K vs Nivel (H)")
            placeholder_k = st.empty()
        with col_q:
            st.subheader("Curva de Gasto")
            placeholder_q = st.empty()

        # --- Leer estado actual para las gráficas ---
        edited_manning = st.session_state.manning_edited_df
        mostrar_lineal = st.session_state.opts_modelos_man["lineal"]
        mostrar_exp = st.session_state.opts_modelos_man["exp"]
        mostrar_log = st.session_state.opts_modelos_man["log"]
        mostrar_pot = st.session_state.opts_modelos_man["pot"]

        # Obtener DataFrame original completo
        df_original = st.session_state.manning_data.copy()

        # Obtener DataFrame editado (solo Incluir, ID, H, Q)
        df_edit = st.session_state.manning_edited_df.copy()

        # IDs activos e inactivos según el editor
        ids_activos = df_edit[df_edit["Incluir"] == True]["ID"].values
        ids_inactivos = df_edit[df_edit["Incluir"] == False]["ID"].values

        # Filtrar el original
        activos = df_original[df_original["ID"].isin(ids_activos)].copy()
        inactivos = df_original[df_original["ID"].isin(ids_inactivos)].copy()

        # Asignar K según la fuente seleccionada
        if fuente_k == "K de aforos":
            activos["K"] = activos["K_af"]
            inactivos["K"] = inactivos["K_af"]
        else:
            activos["K"] = activos["K_per"]
            inactivos["K"] = inactivos["K_per"]

        # Filtrar valores válidos (K > 0)
        activos = activos[activos["K"] > 0].dropna(subset=["K"])
        inactivos = inactivos[inactivos["K"] > 0].dropna(subset=["K"])

        if len(activos) < 2:
            st.error("Necesitas al menos 2 puntos activos con K válido para calcular la curva.")
        else:
            H_act = activos["H"].values
            K_act = activos["K"].values
            Q_act = activos["Q"].values
            
            X_val = np.array(H_act)
            Y_val = np.array(K_act)
            
            # --- Interpoladores de geometría para simulación de Q ---
            df_g = st.session_state.df_geo.copy()
            H_min_geo = df_g["H (m)"].min()
            H_max_geo = df_g["H (m)"].max()
            
            interp_R23 = interp1d(df_g["H (m)"], df_g["R2/3"], kind='linear', fill_value='extrapolate')
            interp_Am = interp1d(df_g["H (m)"], df_g["Am (m2)"], kind='linear', fill_value='extrapolate')
            R23_act = interp_R23(X_val)
            Am_act = interp_Am(X_val)
            
            # Ajustes de modelos
            ajustes = {}
            mask_log = (X_val > 0) & (Y_val > 0)
            X_log, Y_log = X_val[mask_log], Y_val[mask_log]

            # Lineal
            p_lin = np.polyfit(X_val, Y_val, 1)
            Y_lin = p_lin[0]*X_val + p_lin[1]
            r_lin = np.corrcoef(Y_val, Y_lin)[0, 1]
            ajustes["Lineal"] = {
                "r": r_lin, "R2": r_lin**2,
                "func": lambda x: p_lin[0]*x + p_lin[1],
                "eq": f"K = {p_lin[0]:.3f}H + {p_lin[1]:.3f}"
            }

            if len(X_log) > 1:
                p_exp = np.polyfit(X_val, np.log(Y_val), 1)
                Y_exp = np.exp(p_exp[1]) * np.exp(p_exp[0]*X_val)
                r_exp = np.corrcoef(Y_val, Y_exp)[0, 1]
                ajustes["Exponencial"] = {
                    "r": r_exp, "R2": r_exp**2,
                    "func": lambda x: np.exp(p_exp[1]) * np.exp(p_exp[0]*x),
                    "eq": f"K = {np.exp(p_exp[1]):.3f} e^({p_exp[0]:.3f}H)"
                }

                p_log = np.polyfit(np.log(X_log), Y_log, 1)
                Y_log_fit = p_log[0]*np.log(X_log) + p_log[1]
                r_log = np.corrcoef(Y_log, Y_log_fit)[0, 1]
                ajustes["Logarítmica"] = {
                    "r": r_log, "R2": r_log**2,
                    "func": lambda x: np.where(x > 0, p_log[0]*np.log(x) + p_log[1], np.nan),
                    "eq": f"K = {p_log[0]:.3f} ln(H) + {p_log[1]:.3f}"
                }

                p_pot = np.polyfit(np.log(X_log), np.log(Y_log), 1)
                Y_pot = np.exp(p_pot[1]) * (X_log ** p_pot[0])
                r_pot = np.corrcoef(Y_log, Y_pot)[0, 1]
                ajustes["Potencial"] = {
                    "r": r_pot, "R2": r_pot**2,
                    "func": lambda x: np.where(x > 0, np.exp(p_pot[1]) * (x ** p_pot[0]), np.nan),
                    "eq": f"K = {np.exp(p_pot[1]):.3f} H^({p_pot[0]:.3f})"
                }

            # --- Calcular MAPE y σq para cada modelo ---
            for k, v in ajustes.items():
                K_est = v["func"](X_val)
                Q_est = K_est * R23_act * Am_act
                mask_err = (Q_act != 0) & np.isfinite(Q_est)
                if np.any(mask_err):
                    v["MAPE"] = np.mean(np.abs((Q_est[mask_err] - Q_act[mask_err]) / Q_act[mask_err])) * 100
                    v["Sigma_q"] = calcular_error_procedimiento(Q_act[mask_err], Q_est[mask_err], K=2)
                else:
                    v["MAPE"] = np.nan
                    v["Sigma_q"] = np.nan

            # --- Selector de método y tabla de modelos ---
            # 1. Añadimos la opción de Curva Compuesta al menú
            opciones_metodo = ["Automático (Menor MAPE)", "Potencial", "Logarítmica", "Exponencial", "Lineal", "Compuesta (Por tramos)"]
            idx_metodo = 0
            if "metodo_select_manning" in st.session_state:
                if st.session_state.metodo_select_manning in opciones_metodo:
                    idx_metodo = opciones_metodo.index(st.session_state.metodo_select_manning)

            st.markdown("---")
            metodo_seleccionado = st.selectbox(
                "Método Matemático Principal:",
                opciones_metodo,
                index=idx_metodo,
                key="metodo_select_manning"
            )

            # 2. Lógica UI si selecciona la Compuesta
            if metodo_seleccionado == "Compuesta (Por tramos)":
                st.markdown("#### 🔀 Configuración de Curva Compuesta")
                col_c1, col_c2, col_c3 = st.columns(3)

                h_min_safe = float(H_act.min()) if len(H_act) > 0 else 0.0
                h_max_safe = float(H_max_geo)
                h_med_safe = float(np.median(H_act)) if len(H_act) > 0 else 1.0

                with col_c1:
                    h_quiebre = st.number_input(
                        "Nivel de Quiebre (H en m)",
                        min_value=h_min_safe,
                        max_value=h_max_safe,
                        value=st.session_state.get('manning_h_quiebre', h_med_safe),
                        step=0.1,
                        key='manning_h_quiebre'
                    )
                with col_c2:
                    modelo_inf = st.selectbox(
                        "Modelo Inferior (H < Quiebre)",
                        ["Potencial", "Logarítmica", "Exponencial", "Lineal"],
                        key='manning_modelo_inf'
                    )
                with col_c3:
                    modelo_sup = st.selectbox(
                        "Modelo Superior (H ≥ Quiebre)",
                        ["Exponencial", "Lineal", "Potencial", "Logarítmica"],
                        key='manning_modelo_sup'
                    )

                # Definir la función compuesta usando los valores actuales de session_state
                def func_compuesta(x):
                    h_q = st.session_state.manning_h_quiebre
                    m_inf = st.session_state.manning_modelo_inf
                    m_sup = st.session_state.manning_modelo_sup
                    f_inf = ajustes[m_inf]["func"] if m_inf in ajustes else ajustes.get("Lineal")["func"]
                    f_sup = ajustes[m_sup]["func"] if m_sup in ajustes else ajustes.get("Lineal")["func"]
                    return np.where(x < h_q, f_inf(x), f_sup(x))
                
                # Calculamos el error de esta nueva curva inventada
                K_est_comp = func_compuesta(X_val)
                Q_est_comp = K_est_comp * R23_act * Am_act
                mask_err_comp = (Q_act != 0) & np.isfinite(Q_est_comp)
                
                if np.any(mask_err_comp):
                    mape_comp = np.mean(np.abs((Q_est_comp[mask_err_comp] - Q_act[mask_err_comp]) / Q_act[mask_err_comp])) * 100
                    sigma_comp = calcular_error_procedimiento(Q_act[mask_err_comp], Q_est_comp[mask_err_comp], K=2)
                else:
                    mape_comp, sigma_comp = np.nan, np.nan
                
                # Inyectamos nuestro nuevo modelo compuesto en el diccionario de 'ajustes'
                ajustes["Compuesta (Por tramos)"] = {
                    "func": func_compuesta,
                    "eq": f"{modelo_inf} (<{h_quiebre:.2f}m) y {modelo_sup} (≥{h_quiebre:.2f}m)",
                    "MAPE": mape_comp,
                    "Sigma_q": sigma_comp,
                    "r": np.nan, "R2": np.nan # R2 estadístico tradicional no aplica directo a piecewise
                }

            # 3. Mostrar la tabla comparativa de los modelos
            st.subheader("Comparación de modelos")
            df_modelos = pd.DataFrame([
                {
                    "Modelo": k, 
                    "r": f"{v['r']:.4f}" if pd.notna(v['r']) else "-", 
                    "R²": f"{v['R2']:.4f}" if pd.notna(v['R2']) else "-", 
                    "Error Absoluto (%)": f"{v['MAPE']:.2f}" if pd.notna(v['MAPE']) else "N/A",
                    "σq (%)": f"{v['Sigma_q']:.2f}" if pd.notna(v['Sigma_q']) else "N/A",
                    "Ecuación": v['eq']
                }
                for k, v in ajustes.items()
            ])
            df_modelos['MAPE_num'] = pd.to_numeric(df_modelos['Error Absoluto (%)'], errors='coerce')
            df_modelos = df_modelos.sort_values('MAPE_num', na_position='last').drop(columns=['MAPE_num'])
            st.dataframe(df_modelos, use_container_width=True, hide_index=True)

            # 4. Seleccionar modelo definitivo
            if metodo_seleccionado == "Automático (Menor MAPE)":
                mejor_modelo = min(
                    (k for k in ajustes if pd.notna(ajustes[k]["MAPE"]) and k != "Compuesta (Por tramos)"),
                    key=lambda k: ajustes[k]["MAPE"],
                    default=list(ajustes.keys())[0]
                )
            else:
                mejor_modelo = metodo_seleccionado
            
            funcion_optima = ajustes[mejor_modelo]["func"]
            
            # Formateo seguro para R2 en el mensaje (la compuesta tiene np.nan)
            r2_display = f" (R² = {ajustes[mejor_modelo]['R2']:.4f})" if pd.notna(ajustes[mejor_modelo]['R2']) else ""
            st.info(f"Modelo seleccionado (Línea principal): **{mejor_modelo}** con Error Absoluto = {ajustes[mejor_modelo]['MAPE']:.2f}%{r2_display}")

            # --- Aplicar a geometría (extrapolación) ---
            paso_fino = 0.2
            H_fino = np.arange(H_min_geo, H_max_geo + paso_fino, paso_fino)
            
            K_fino = funcion_optima(H_fino)
            R23_fino = interp_R23(H_fino)
            Am_fino = interp_Am(H_fino)
            
            V_fino = K_fino * R23_fino
            Q_fino = V_fino * Am_fino

            # --- Filtrar valores no finitos y negativos ---
            mask_finite = np.isfinite(Q_fino)
            if not np.all(mask_finite):
                st.warning(f"Se encontraron {np.sum(~mask_finite)} valores no finitos en la curva. Se eliminarán.")
                H_fino = H_fino[mask_finite]
                Q_fino = Q_fino[mask_finite]

            Q_fino = np.maximum(Q_fino, 0)

            # Suavizado con interpolación lineal
            mask_dentro = (H_fino >= H_min_geo) & (H_fino <= H_max_geo)
            H_dentro = H_fino[mask_dentro]
            Q_dentro = Q_fino[mask_dentro]

            if len(H_dentro) > 1:
                f_lin = interp1d(H_dentro, Q_dentro, kind='linear', fill_value='extrapolate')
                Q_suave = f_lin(H_fino)
            else:
                Q_suave = Q_fino

            # Colores base
            color_activos = '#00ccff'
            color_inactivos = '#ff5555'
            color_ajuste = '#55ff55'
            color_curva = '#ffaa00'
            color_banda = 'rgba(200, 200, 200, 0.15)'

            # --- Preparar funciones adicionales para la gráfica K vs H ---
            funcs_adic = {}
            if 'Lineal' in ajustes: funcs_adic['lineal'] = (ajustes['Lineal']['func'], mostrar_lineal)
            if 'Exponencial' in ajustes: funcs_adic['exp'] = (ajustes['Exponencial']['func'], mostrar_exp)
            if 'Logarítmica' in ajustes: funcs_adic['log'] = (ajustes['Logarítmica']['func'], mostrar_log)
            if 'Potencial' in ajustes: funcs_adic['pot'] = (ajustes['Potencial']['func'], mostrar_pot)

            H_smooth = np.linspace(min(X_val)*0.5, max(X_val)*1.5, 200)
            K_sel = funcion_optima(H_smooth)

            inactivos_k = inactivos[['H', 'K']].rename(columns={'K': 'Y'}) if not inactivos.empty else None

            fig_k = crear_figura_k(
                titulo="K vs Nivel (H) - Manning",
                H_act=H_act,
                Y_act=K_act,
                inactivos=inactivos_k,
                H_smooth=H_smooth,
                funciones_adicionales=funcs_adic,
                Y_sel=K_sel,
                color_activos=color_activos,
                color_inactivos=color_inactivos,
                color_ajuste=color_ajuste,
                xlabel="Nivel H (m)",
                ylabel="K"
            )
            placeholder_k.plotly_chart(fig_k, use_container_width=True)

            # --- Gráfica Curva de Gasto con banda global Y TODOS LOS MODELOS ACTIVOS ---
            fig_q = crear_figura_curva(
                titulo=f"Curva de Gasto - usando {fuente_k}",
                H_fino=H_fino,
                Q_suave=Q_suave,
                Q_act=Q_act,
                H_act=H_act,
                inactivos=inactivos[['H', 'Q']] if not inactivos.empty else None,
                color_curva=color_curva,
                color_activos=color_activos,
                color_inactivos=color_inactivos,
                color_banda=color_banda,
                banda_pct=st.session_state.banda_error_global
            )
            
            # AGREGAR MODELOS ADICIONALES A LA CURVA DE GASTO
            colores_extras = {'Lineal': 'cyan', 'Exponencial': 'magenta', 'Logarítmica': 'yellow', 'Potencial': 'lime'}
            mapa_nombres = {'Lineal': mostrar_lineal, 'Exponencial': mostrar_exp, 'Logarítmica': mostrar_log, 'Potencial': mostrar_pot}
            
            for nombre_mod, dict_mod in ajustes.items():
                # Si el usuario lo tiene chuleado en controles y NO es el modelo principal
                if mapa_nombres.get(nombre_mod, False) and nombre_mod != mejor_modelo:
                    K_extra = dict_mod["func"](H_fino)
                    Q_extra = K_extra * interp_R23(H_fino) * interp_Am(H_fino)
                    Q_extra = np.maximum(Q_extra, 0)
                    
                    fig_q.add_trace(go.Scatter(
                        x=Q_extra, y=H_fino, mode='lines', name=f'Q ({nombre_mod})',
                        line=dict(color=colores_extras[nombre_mod], width=2, dash='dot')
                    ))

            placeholder_q.plotly_chart(fig_q, use_container_width=True)

            # --- NUEVA TABLA DE ERRORES CON TODOS LOS MODELOS ---
            errores_data = []
            for _, row in activos.iterrows():
                h_val = row["H"]
                q_obs = row["Q"]
                r23_val = float(interp_R23(h_val))
                am_val = float(interp_Am(h_val))
                
                fila_error = {
                    "H (m)": h_val, 
                    "Q Aforado (m³/s)": q_obs
                }
                
                # Calcular el error para cada modelo matemático
                for nombre_mod, dict_mod in ajustes.items():
                    k_est = dict_mod["func"](h_val)
                    q_est = k_est * r23_val * am_val
                    err_pct = abs(q_est - q_obs) / q_obs * 100 if q_obs != 0 else np.nan
                    fila_error[f"Error {nombre_mod} (%)"] = err_pct
                    
                errores_data.append(fila_error)

            # --- Obtener errores globales del mejor modelo ---
            prom_mape = ajustes[mejor_modelo]["MAPE"]
            prom_sigma = ajustes[mejor_modelo]["Sigma_q"]

            # --- Fila de guardado y métricas ---
            col_guardar, col_mape, col_sigma = st.columns([1, 1, 1])
            with col_guardar:
                # --- AUTO-GUARDADO SILENCIOSO ---
                # (Se eliminó el if st.button para que guarde automáticamente)
                if len(H_fino) > 0 and len(Q_suave) > 0 and np.isfinite(Q_suave).any():
                    st.session_state.manning_curve = pd.DataFrame({"H": H_fino, "Q": Q_suave})
                    st.session_state.manning_error = prom_mape
                    st.session_state.manning_error_sigma = prom_sigma
                    st.success("✅ Auto-guardado: Curva en memoria")
                else:
                    st.error("No hay curva válida para guardar")
                
                # --- LA MAGIA DE LA RESTAURACIÓN ---
                if "manning_curve" in st.session_state and st.session_state.manning_curve is not None:
                    st.success("✅ Curva definitiva cargada en memoria")
                    with st.expander("Ver tabla guardada"):
                        st.dataframe(st.session_state.manning_curve, height=150, use_container_width=True)
            with col_mape:
                st.metric("Error Absoluto Promedio", f"{prom_mape:.1f}%" if pd.notna(prom_mape) else "N/A")
            with col_sigma:
                st.metric("Error Procedimiento (σq)", f"{prom_sigma:.1f}%" if pd.notna(prom_sigma) else "N/A")

            # --- Tabla de errores (SEMÁFORO) ---
            st.markdown("---")
            st.subheader("Errores en aforos activos (Todos los modelos)")
            
            df_errores = pd.DataFrame(errores_data)
            
            # --- ORDENAR LA TABLA POR H DE MENOR A MAYOR ---
            df_errores = df_errores.sort_values(by="H (m)", ascending=True).reset_index(drop=True)
            
            # Identificar qué columnas contienen la palabra "Error"
            cols_error = [col for col in df_errores.columns if "Error" in col]
            
            # Función para aplicar la regla de color semáforo
            def color_semaforo(val):
                if pd.isna(val):
                    return ''
                try:
                    v = float(val)
                    if v <= 10:
                        return 'background-color: rgba(39, 174, 96, 0.4); color: white;' # Verde
                    elif v <= 50:
                        return 'background-color: rgba(243, 156, 18, 0.4); color: white;' # Amarillo
                    else:
                        return 'background-color: rgba(192, 57, 43, 0.4); color: white;' # Rojo
                except:
                    return ''

            # Diccionario para redondear correctamente
            format_dict = {"H (m)": "{:.2f}", "Q Aforado (m³/s)": "{:.2f}"}
            for c in cols_error:
                format_dict[c] = "{:.1f}"
                
            # Aplicar formato numérico y aplicar colores solo a las columnas de error
            styled_df = df_errores.style.format(format_dict).map(color_semaforo, subset=cols_error)
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

# ================== PESTAÑA 4: MÉTODO STEVENS ==================
with tab4:
    if st.session_state.df_aforos is None or st.session_state.df_geo is None:
        st.warning("Procesa los Aforos y genera la Tabla de Geometría primero.")
    else:
        # --- ENCABEZADO INFORMATIVO DEL MÉTODO ---
        with st.expander("📘 **Fundamento y consideraciones del método Stevens**", expanded=False):
            st.markdown("""
            **Fundamento:** Este método es adecuado para ríos considerablemente anchos y poco profundos, donde la profundidad media (D) se puede considerar aproximadamente igual al radio hidráulico (R).  
            Esto ocurre cuando la relación ancho de la sección / perímetro mojado es próxima a 1, es decir, R ≈ D = A / W.

            **Ecuación base (Chezy):** Q = C √(R S).  
            Reemplazando R por D: Q = C √S · A √D.

            Para niveles altos, el término C√S tiende a ser constante.  
            Experimentalmente se ha encontrado que usar el exponente 2/3 en lugar de 1/2 proporciona mejores resultados, por lo que la variable independiente utilizada es **A·D^(2/3)**.

            En esta herramienta se define K = Q / (A·D^(2/3)), se ajusta K en función del nivel H y luego se extrapola usando la geometría calculada.

            > *Limitación:* La aproximación R ≈ D es válida solo en secciones muy anchas. Si el río no cumple esta condición, el método puede introducir errores sistemáticos.
            """)

        # --- CORRECCIÓN: Verificar si los aforos activos han cambiado ---
        df_a_current = st.session_state.get('df_aforos_activos', st.session_state.get('df_aforos'))
        ids_current = df_a_current['NO.'].tolist() if (df_a_current is not None and not df_a_current.empty) else []
        ids_stored = st.session_state.stevens_data['ID'].tolist() if (st.session_state.get('stevens_data') is not None and not st.session_state.stevens_data.empty) else []

        necesita_actualizar_s = (
            st.session_state.get('stevens_data') is None 
            or st.session_state.get('flag_actualizar_modelos', False)
            or ids_current != ids_stored
        )

        if necesita_actualizar_s:
            # ... (resto del código de actualización igual)
            # Asegúrate de que al final se haga:
            st.session_state.flag_actualizar_modelos = False
            if 'stevens_edited_df' in st.session_state:
                del st.session_state['stevens_edited_df']

        # --- INDICADOR DE APLICABILIDAD DEL MÉTODO STEVENS (basado en geometría) ---
        if st.session_state.df_geo is not None and "R/D" in st.session_state.df_geo.columns:
            df_geo = st.session_state.df_geo
            valores_rd = df_geo["R/D"].dropna()
            if len(valores_rd) > 0:
                min_rd = valores_rd.min()
                max_rd = valores_rd.max()
                mean_rd = valores_rd.mean()
                
                if min_rd < 0.8 or max_rd > 1.2:
                    st.warning("⚠️ **La relación R/D en la geometría se aleja significativamente de 1.** El método Stevens (que asume R ≈ D) podría no ser adecuado para esta sección. Revise la pestaña Geometría para más detalles.")
                elif min_rd < 0.9 or max_rd > 1.1:
                    st.info("ℹ️ **La relación R/D está moderadamente cerca de 1.** El método Stevens puede ser aceptable, pero verifique los resultados cuidadosamente.")
                else:
                    st.success("✅ **La relación R/D es cercana a 1 en todo el rango.** El método Stevens es aplicable.")
        else:
            st.caption("*(Genere la tabla de geometría en la pestaña correspondiente para evaluar la aplicabilidad del método Stevens)*")
        
        # --- AQUÍ ESTÁ LA CORRECCIÓN: Escuchamos la alarma de la Pestaña 1 ---
        necesita_actualizar_s = (
            st.session_state.get('stevens_data') is None 
            or st.session_state.get('flag_actualizar_modelos', False)
        )

        if necesita_actualizar_s:
            # 1. RECIBIR AFOROS FILTRADOS (Ya vienen limpios de la Pestaña 1)
            if 'df_aforos_activos' in st.session_state:
                df_a = st.session_state.df_aforos_activos.copy()
            else:
                df_a = st.session_state.df_aforos.copy()
                
            p_data = st.session_state.perfil_data

            col_area_af = buscar_columna(df_a.columns, ["AREA", "SEC"])
            col_d_af = buscar_columna(df_a.columns, ["PROF", "MEDIA"])
            
            if not col_area_af: st.sidebar.error("No se encontró la columna de Área de Sección en Aforos.")
            if not col_d_af: st.sidebar.error("No se encontró la columna de Profundidad Media en Aforos.")
            
            # 2. FILTRAR PERFIL TOPOGRÁFICO ACTIVO
            if 'perfil_puntos_activos' in st.session_state:
                mascara_perfil = np.array(st.session_state.perfil_puntos_activos, dtype=bool)
                abscisas_activas = np.array(p_data['abscisas'])[mascara_perfil]
                cotas_activas = np.array(p_data['cotas'])[mascara_perfil]
            else:
                abscisas_activas = np.array(p_data['abscisas'])
                cotas_activas = np.array(p_data['cotas'])

            datos_stevens = []
            for i, row in df_a.iterrows():
                id_aforo = row.get("NO.", i+1)
                H = row["H_m"]
                Q = row["CAUDAL TOTAL (m3/s)"]
                
                # --- Cálculo de K del perfil (Stevens) ---
                A_per = area_mojada(abscisas_activas, cotas_activas, p_data['cota_cero'], H)
                Wh_per = ancho_superficial(abscisas_activas, cotas_activas, p_data['cota_cero'], H)
                
                K_per = 0
                X_per = 0
                if Wh_per > 0 and A_per > 0:
                    D_per = A_per / Wh_per
                    X_per = A_per * (D_per ** (2/3))
                    K_per = Q / X_per if X_per > 0 else 0
                
                # --- Cálculo de K de aforos (Stevens) ---
                A_af = row.get(col_area_af, np.nan)
                D_af = row.get(col_d_af, np.nan)
                
                K_af = 0
                X_af = np.nan
                if pd.notna(A_af) and pd.notna(D_af) and A_af > 0 and D_af > 0:
                    X_af = A_af * (D_af ** (2/3))
                    K_af = Q / X_af if X_af > 0 else 0
                else:
                    st.sidebar.warning(f"Aforo {id_aforo}: faltan A_af o D_af, K_af = 0")
                
                if K_per > 0 or K_af > 0:
                    datos_stevens.append({
                        "Incluir": True, 
                        "ID": id_aforo, 
                        "H": H, 
                        "Q": Q, 
                        "A_af": A_af,
                        "D_af": D_af,
                        "X_af": X_af,
                        "K_af": K_af,
                        "K_per": K_per,
                    })
            
            st.session_state.stevens_data = pd.DataFrame(datos_stevens)
            
            # --- APAGAMOS LA ALARMA Y RESETEAMOS LA TABLA VISUAL ---
            st.session_state.flag_actualizar_modelos = False
            if 'stevens_edited_df' in st.session_state:
                del st.session_state['stevens_edited_df']

        idx_fuente_s = 0
        if "fuente_k_radio_s" in st.session_state:
            if st.session_state.fuente_k_radio_s == "K del perfil":
                idx_fuente_s = 1

        fuente_k_s = st.radio(
            "Fuente de K:",
            options=["K de aforos", "K del perfil"],
            index=idx_fuente_s, 
            horizontal=True,
            key="fuente_k_radio_s"
        )

        df_edit_s = st.session_state.stevens_data.copy()
        
        if fuente_k_s == "K de aforos":
            df_edit_s["K"] = df_edit_s["K_af"]
        else:
            df_edit_s["K"] = df_edit_s["K_per"]

        # --- PERSISTENCIA: Inicializar DataFrame editado y opciones de modelos ---
        if 'stevens_edited_df' not in st.session_state:
            default_df_s = df_edit_s.copy()
            default_df_s['Incluir'] = True
            st.session_state.stevens_edited_df = default_df_s

        if 'opts_modelos_stevens' not in st.session_state:
            st.session_state.opts_modelos_stevens = {"lineal": True, "exp": True, "log": True, "pot": True}

        if 'banda_error_global' not in st.session_state:
            st.session_state.banda_error_global = 15.0

        # --- Botón con popover (esquina superior izquierda) ---
        col_btn, _ = st.columns([0.15, 0.85])
        with col_btn:
            with st.popover("⚙️ Controles"):
                with st.form(key="stevens_form"):
                    st.caption("Filtro de Aforos")
                    edited_s = st.data_editor(
                        st.session_state.stevens_edited_df[["Incluir", "ID", "H", "Q"]],
                        column_config={
                            "Incluir": st.column_config.CheckboxColumn("Activo", default=True),
                            "ID": st.column_config.NumberColumn("ID", disabled=True),
                            "H": st.column_config.NumberColumn("H (m)", disabled=True, format="%.2f"),
                            "Q": st.column_config.NumberColumn("Q (m³/s)", disabled=True, format="%.2f"),
                        },
                        disabled=False,
                        hide_index=True,
                        use_container_width=True,
                        key="stevens_editor"
                    )

                    st.caption("Modelos de ajuste")
                    lineal_s = st.checkbox("Lineal", value=st.session_state.opts_modelos_stevens["lineal"])
                    exp_s = st.checkbox("Exponencial", value=st.session_state.opts_modelos_stevens["exp"])
                    log_s = st.checkbox("Logarítmica", value=st.session_state.opts_modelos_stevens["log"])
                    pot_s = st.checkbox("Potencial", value=st.session_state.opts_modelos_stevens["pot"])

                    st.caption("Configuración General")
                    opciones_banda = ["Estable (10%)", "Inestable (15%)"]
                    indice_actual = 0 if st.session_state.banda_error_global == 10 else 1
                    tipo_seccion = st.radio(
                        "Tipo de sección:",
                        options=opciones_banda,
                        index=indice_actual,
                        horizontal=True
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        submitted = st.form_submit_button("Aplicar")
                    with col2:
                        cancelled = st.form_submit_button("Cancelar")

                    if submitted:
                        st.session_state.stevens_edited_df = edited_s
                        st.session_state.opts_modelos_stevens["lineal"] = lineal_s
                        st.session_state.opts_modelos_stevens["exp"] = exp_s
                        st.session_state.opts_modelos_stevens["log"] = log_s
                        st.session_state.opts_modelos_stevens["pot"] = pot_s
                        if tipo_seccion == "Estable (10%)":
                            st.session_state.banda_error_global = 10.0
                        else:
                            st.session_state.banda_error_global = 15.0
                        st.rerun()
                    elif cancelled:
                        st.rerun()

        # --- Gráficas (siempre ancho completo) ---
        col_k_s, col_q_s = st.columns(2)
        with col_k_s:
            st.subheader("K vs Nivel (H)")
            placeholder_k_s = st.empty()
        with col_q_s:
            st.subheader("Curva de Gasto")
            placeholder_q_s = st.empty()

        # --- Leer estado actual para las gráficas ---
        mostrar_lineal_s = st.session_state.opts_modelos_stevens["lineal"]
        mostrar_exp_s = st.session_state.opts_modelos_stevens["exp"]
        mostrar_log_s = st.session_state.opts_modelos_stevens["log"]
        mostrar_pot_s = st.session_state.opts_modelos_stevens["pot"]

        # Obtener DataFrames
        df_original_s = st.session_state.stevens_data.copy()
        df_edit_s = st.session_state.stevens_edited_df.copy()

        ids_activos_s = df_edit_s[df_edit_s["Incluir"] == True]["ID"].values
        ids_inactivos_s = df_edit_s[df_edit_s["Incluir"] == False]["ID"].values

        activos_s = df_original_s[df_original_s["ID"].isin(ids_activos_s)].copy()
        inactivos_s = df_original_s[df_original_s["ID"].isin(ids_inactivos_s)].copy()

        if fuente_k_s == "K de aforos":
            activos_s["K"] = activos_s["K_af"]
            inactivos_s["K"] = inactivos_s["K_af"]
        else:
            activos_s["K"] = activos_s["K_per"]
            inactivos_s["K"] = inactivos_s["K_per"]

        activos_s = activos_s[activos_s["K"] > 0].dropna(subset=["K"])
        inactivos_s = inactivos_s[inactivos_s["K"] > 0].dropna(subset=["K"])

        if len(activos_s) < 2:
            st.error("Necesitas al menos 2 puntos activos con K válido para calcular la curva.")
        else:
            H_act_s = activos_s["H"].values
            K_act_s = activos_s["K"].values
            Q_act_s = activos_s["Q"].values
            
            X_val_s = np.array(H_act_s)
            Y_val_s = np.array(K_act_s)
            
            # --- Interpoladores de geometría para simulación de Q ---
            df_g = st.session_state.df_geo.copy()
            H_min_geo = df_g["H (m)"].min()
            H_max_geo = df_g["H (m)"].max()
            
            from scipy.interpolate import interp1d
            interp_A_s = interp1d(df_g["H (m)"], df_g["Am (m2)"], kind='linear', fill_value='extrapolate')
            interp_D_s = interp1d(df_g["H (m)"], df_g["D (m)"], kind='linear', fill_value='extrapolate')
            
            A_act_s = interp_A_s(X_val_s)
            D_act_s = interp_D_s(X_val_s)
            X_factor_act = A_act_s * (D_act_s ** (2/3))
            
            # Ajustes de modelos
            ajustes_s = {}
            mask_log_s = (X_val_s > 0) & (Y_val_s > 0)
            X_log_s, Y_log_s = X_val_s[mask_log_s], Y_val_s[mask_log_s]

            # Lineal
            p_lin_s = np.polyfit(X_val_s, Y_val_s, 1)
            Y_lin_s = p_lin_s[0]*X_val_s + p_lin_s[1]
            r_lin_s = np.corrcoef(Y_val_s, Y_lin_s)[0, 1]
            ajustes_s["Lineal"] = {
                "r": r_lin_s, "R2": r_lin_s**2,
                "func": lambda x: p_lin_s[0]*x + p_lin_s[1],
                "eq": f"K = {p_lin_s[0]:.3f}H + {p_lin_s[1]:.3f}"
            }

            if len(X_log_s) > 1:
                p_exp_s = np.polyfit(X_val_s, np.log(Y_val_s), 1)
                Y_exp_s = np.exp(p_exp_s[1]) * np.exp(p_exp_s[0]*X_val_s)
                r_exp_s = np.corrcoef(Y_val_s, Y_exp_s)[0, 1]
                ajustes_s["Exponencial"] = {
                    "r": r_exp_s, "R2": r_exp_s**2,
                    "func": lambda x: np.exp(p_exp_s[1]) * np.exp(p_exp_s[0]*x),
                    "eq": f"K = {np.exp(p_exp_s[1]):.3f} e^({p_exp_s[0]:.3f}H)"
                }

                p_log_s = np.polyfit(np.log(X_log_s), Y_log_s, 1)
                Y_log_fit_s = p_log_s[0]*np.log(X_log_s) + p_log_s[1]
                r_log_s = np.corrcoef(Y_log_s, Y_log_fit_s)[0, 1]
                ajustes_s["Logarítmica"] = {
                    "r": r_log_s, "R2": r_log_s**2,
                    "func": lambda x: np.where(x > 0, p_log_s[0]*np.log(x) + p_log_s[1], np.nan),
                    "eq": f"K = {p_log_s[0]:.3f} ln(H) + {p_log_s[1]:.3f}"
                }

                p_pot_s = np.polyfit(np.log(X_log_s), np.log(Y_log_s), 1)
                Y_pot_s = np.exp(p_pot_s[1]) * (X_log_s ** p_pot_s[0])
                r_pot_s = np.corrcoef(Y_log_s, Y_pot_s)[0, 1]
                ajustes_s["Potencial"] = {
                    "r": r_pot_s, "R2": r_pot_s**2,
                    "func": lambda x: np.where(x > 0, np.exp(p_pot_s[1]) * (x ** p_pot_s[0]), np.nan),
                    "eq": f"K = {np.exp(p_pot_s[1]):.3f} H^({p_pot_s[0]:.3f})"
                }

            # --- Calcular MAPE y σq para cada modelo ---
            for k, v in ajustes_s.items():
                K_est_s = v["func"](X_val_s)
                Q_est_s = K_est_s * X_factor_act
                mask_err_s = (Q_act_s != 0) & np.isfinite(Q_est_s)
                if np.any(mask_err_s):
                    v["MAPE"] = np.mean(np.abs((Q_est_s[mask_err_s] - Q_act_s[mask_err_s]) / Q_act_s[mask_err_s])) * 100
                    v["Sigma_q"] = calcular_error_procedimiento(Q_act_s[mask_err_s], Q_est_s[mask_err_s], K=2)
                else:
                    v["MAPE"] = np.nan
                    v["Sigma_q"] = np.nan

            # --- Selector de método y tabla de modelos ---
            opciones_metodo_s = ["Automático (Menor MAPE)", "Potencial", "Logarítmica", "Exponencial", "Lineal", "Compuesta (Por tramos)"]
            idx_metodo_s = 0
            if "metodo_select_stevens" in st.session_state:
                if st.session_state.metodo_select_stevens in opciones_metodo_s:
                    idx_metodo_s = opciones_metodo_s.index(st.session_state.metodo_select_stevens)

            st.markdown("---")
            metodo_seleccionado_s = st.selectbox(
                "Método Matemático Principal:",
                opciones_metodo_s,
                index=idx_metodo_s,
                key="metodo_select_stevens"
            )

            # 2. Lógica UI si selecciona la Compuesta
            if metodo_seleccionado_s == "Compuesta (Por tramos)":
                st.markdown("#### 🔀 Configuración de Curva Compuesta")
                col_c1, col_c2, col_c3 = st.columns(3)

                h_min_safe = float(H_act_s.min()) if len(H_act_s) > 0 else 0.0
                h_max_safe = float(H_max_geo)
                h_med_safe = float(np.median(H_act_s)) if len(H_act_s) > 0 else 1.0

                with col_c1:
                    h_quiebre_s = st.number_input(
                        "Nivel de Quiebre (H en m) - Stevens",
                        min_value=h_min_safe,
                        max_value=h_max_safe,
                        value=st.session_state.get('stevens_h_quiebre', h_med_safe),
                        step=0.1,
                        key='stevens_h_quiebre'
                    )
                with col_c2:
                    modelo_inf_s = st.selectbox(
                        "Modelo Inferior (H < Quiebre)",
                        ["Potencial", "Logarítmica", "Exponencial", "Lineal"],
                        index=0,
                        key='stevens_modelo_inf'
                    )
                with col_c3:
                    modelo_sup_s = st.selectbox(
                        "Modelo Superior (H ≥ Quiebre)",
                        ["Exponencial", "Lineal", "Potencial", "Logarítmica"],
                        index=1,
                        key='stevens_modelo_sup'
                    )

                def func_compuesta_s(x):
                    h_q = st.session_state.stevens_h_quiebre
                    m_inf = st.session_state.stevens_modelo_inf
                    m_sup = st.session_state.stevens_modelo_sup
                    f_inf = ajustes_s[m_inf]["func"] if m_inf in ajustes_s else ajustes_s.get("Lineal")["func"]
                    f_sup = ajustes_s[m_sup]["func"] if m_sup in ajustes_s else ajustes_s.get("Lineal")["func"]
                    return np.where(x < h_q, f_inf(x), f_sup(x))

                # Calcular errores de la curva compuesta
                K_est_comp_s = func_compuesta_s(X_val_s)
                Q_est_comp_s = K_est_comp_s * X_factor_act
                mask_err_comp_s = (Q_act_s != 0) & np.isfinite(Q_est_comp_s)

                if np.any(mask_err_comp_s):
                    mape_comp_s = np.mean(np.abs((Q_est_comp_s[mask_err_comp_s] - Q_act_s[mask_err_comp_s]) / Q_act_s[mask_err_comp_s])) * 100
                    sigma_comp_s = calcular_error_procedimiento(Q_act_s[mask_err_comp_s], Q_est_comp_s[mask_err_comp_s], K=2)
                else:
                    mape_comp_s, sigma_comp_s = np.nan, np.nan

                ajustes_s["Compuesta (Por tramos)"] = {
                    "func": func_compuesta_s,
                    "eq": f"{modelo_inf_s} (<{h_quiebre_s:.2f}m) y {modelo_sup_s} (≥{h_quiebre_s:.2f}m)",
                    "MAPE": mape_comp_s,
                    "Sigma_q": sigma_comp_s,
                    "r": np.nan, "R2": np.nan
                }

            # 3. Mostrar la tabla comparativa de los modelos
            st.subheader("Comparación de modelos")
            df_modelos_s = pd.DataFrame([
                {
                    "Modelo": k, 
                    "r": f"{v['r']:.4f}" if pd.notna(v['r']) else "-", 
                    "R²": f"{v['R2']:.4f}" if pd.notna(v['R2']) else "-", 
                    "Error Absoluto (%)": f"{v['MAPE']:.2f}" if pd.notna(v['MAPE']) else "N/A",
                    "σq (%)": f"{v['Sigma_q']:.2f}" if pd.notna(v['Sigma_q']) else "N/A",
                    "Ecuación": v['eq']
                }
                for k, v in ajustes_s.items()
            ])
            df_modelos_s['MAPE_num'] = pd.to_numeric(df_modelos_s['Error Absoluto (%)'], errors='coerce')
            df_modelos_s = df_modelos_s.sort_values('MAPE_num', na_position='last').drop(columns=['MAPE_num'])
            st.dataframe(df_modelos_s, use_container_width=True, hide_index=True)

            # 4. Seleccionar modelo definitivo
            if metodo_seleccionado_s == "Automático (Menor MAPE)":
                mejor_modelo_s = min(
                    (k for k in ajustes_s if pd.notna(ajustes_s[k]["MAPE"]) and k != "Compuesta (Por tramos)"),
                    key=lambda k: ajustes_s[k]["MAPE"],
                    default=list(ajustes_s.keys())[0]
                )
            else:
                mejor_modelo_s = metodo_seleccionado_s
            
            funcion_optima_s = ajustes_s[mejor_modelo_s]["func"]
            
            r2_display_s = f" (R² = {ajustes_s[mejor_modelo_s]['R2']:.4f})" if pd.notna(ajustes_s[mejor_modelo_s]['R2']) else ""
            st.info(f"Modelo seleccionado (Línea principal): **{mejor_modelo_s}** con Error Absoluto = {ajustes_s[mejor_modelo_s]['MAPE']:.2f}%{r2_display_s}")

            # --- Aplicar a geometría (extrapolación) ---
            paso_fino = 0.2
            H_fino_s = np.arange(H_min_geo, H_max_geo + paso_fino, paso_fino)
            
            K_fino_s = funcion_optima_s(H_fino_s)
            A_fino_s = interp_A_s(H_fino_s)
            D_fino_s = interp_D_s(H_fino_s)
            
            X_fino_s = A_fino_s * (D_fino_s ** (2/3))
            Q_fino_s = K_fino_s * X_fino_s

            # --- Filtrar valores no finitos y negativos ---
            mask_finite_s = np.isfinite(Q_fino_s)
            if not np.all(mask_finite_s):
                st.warning(f"Se encontraron {np.sum(~mask_finite_s)} valores no finitos en la curva. Se eliminarán.")
                H_fino_s = H_fino_s[mask_finite_s]
                Q_fino_s = Q_fino_s[mask_finite_s]

            Q_fino_s = np.maximum(Q_fino_s, 0)

            # Suavizado con interpolación lineal
            mask_dentro_s = (H_fino_s >= H_min_geo) & (H_fino_s <= H_max_geo)
            H_dentro_s = H_fino_s[mask_dentro_s]
            Q_dentro_s = Q_fino_s[mask_dentro_s]

            if len(H_dentro_s) > 1:
                f_lin_s = interp1d(H_dentro_s, Q_dentro_s, kind='linear', fill_value='extrapolate')
                Q_suave_s = f_lin_s(H_fino_s)
            else:
                Q_suave_s = Q_fino_s

            # Colores base
            color_activos = '#00ccff'
            color_inactivos = '#ff5555'
            color_ajuste = '#55ff55'
            color_curva = '#ffaa00'
            color_banda = 'rgba(200, 200, 200, 0.15)'

            # --- Preparar funciones adicionales para la gráfica K vs H ---
            funcs_adic_s = {}
            if 'Lineal' in ajustes_s: funcs_adic_s['lineal'] = (ajustes_s['Lineal']['func'], mostrar_lineal_s)
            if 'Exponencial' in ajustes_s: funcs_adic_s['exp'] = (ajustes_s['Exponencial']['func'], mostrar_exp_s)
            if 'Logarítmica' in ajustes_s: funcs_adic_s['log'] = (ajustes_s['Logarítmica']['func'], mostrar_log_s)
            if 'Potencial' in ajustes_s: funcs_adic_s['pot'] = (ajustes_s['Potencial']['func'], mostrar_pot_s)

            H_smooth_s = np.linspace(min(X_val_s)*0.5, max(X_val_s)*1.5, 200)
            K_sel_s = funcion_optima_s(H_smooth_s)

            inactivos_k_s = inactivos_s[['H', 'K']].rename(columns={'K': 'Y'}) if not inactivos_s.empty else None

            fig_k_s = crear_figura_k(
                titulo="K vs Nivel (H) - Stevens",
                H_act=H_act_s,
                Y_act=K_act_s,
                inactivos=inactivos_k_s,
                H_smooth=H_smooth_s,
                funciones_adicionales=funcs_adic_s,
                Y_sel=K_sel_s,
                color_activos=color_activos,
                color_inactivos=color_inactivos,
                color_ajuste=color_ajuste,
                xlabel="Nivel H (m)",
                ylabel="K"
            )
            placeholder_k_s.plotly_chart(fig_k_s, use_container_width=True)

            # --- Gráfica Curva de Gasto con banda global Y TODOS LOS MODELOS ACTIVOS ---
            fig_q_s = crear_figura_curva(
                titulo=f"Curva de Gasto - usando {fuente_k_s}",
                H_fino=H_fino_s,
                Q_suave=Q_suave_s,
                Q_act=Q_act_s,
                H_act=H_act_s,
                inactivos=inactivos_s[['H', 'Q']] if not inactivos_s.empty else None,
                color_curva=color_curva,
                color_activos=color_activos,
                color_inactivos=color_inactivos,
                color_banda=color_banda,
                banda_pct=st.session_state.banda_error_global
            )
            
            # AGREGAR MODELOS ADICIONALES A LA CURVA DE GASTO
            colores_extras = {'Lineal': 'cyan', 'Exponencial': 'magenta', 'Logarítmica': 'yellow', 'Potencial': 'lime'}
            mapa_nombres_s = {'Lineal': mostrar_lineal_s, 'Exponencial': mostrar_exp_s, 'Logarítmica': mostrar_log_s, 'Potencial': mostrar_pot_s}
            
            for nombre_mod, dict_mod in ajustes_s.items():
                if mapa_nombres_s.get(nombre_mod, False) and nombre_mod != mejor_modelo_s:
                    K_extra_s = dict_mod["func"](H_fino_s)
                    Q_extra_s = K_extra_s * interp_A_s(H_fino_s) * (interp_D_s(H_fino_s) ** (2/3))
                    Q_extra_s = np.maximum(Q_extra_s, 0)
                    
                    fig_q_s.add_trace(go.Scatter(
                        x=Q_extra_s, y=H_fino_s, mode='lines', name=f'Q ({nombre_mod})',
                        line=dict(color=colores_extras[nombre_mod], width=2, dash='dot')
                    ))

            placeholder_q_s.plotly_chart(fig_q_s, use_container_width=True)

            # --- NUEVA TABLA DE ERRORES CON TODOS LOS MODELOS ---
            errores_data_s = []
            for _, row in activos_s.iterrows():
                h_val_s = row["H"]
                q_obs_s = row["Q"]
                a_val_s = float(interp_A_s(h_val_s))
                d_val_s = float(interp_D_s(h_val_s))
                x_val_factor = a_val_s * (d_val_s ** (2/3))
                
                fila_error_s = {
                    "H (m)": h_val_s, 
                    "Q Aforado (m³/s)": q_obs_s
                }
                
                # Calcular el error para cada modelo matemático
                for nombre_mod, dict_mod in ajustes_s.items():
                    k_est_s = dict_mod["func"](h_val_s)
                    q_est_s = k_est_s * x_val_factor
                    err_pct_s = abs(q_est_s - q_obs_s) / q_obs_s * 100 if q_obs_s != 0 else np.nan
                    fila_error_s[f"Error {nombre_mod} (%)"] = err_pct_s
                    
                errores_data_s.append(fila_error_s)

            # --- Obtener errores globales del mejor modelo ---
            prom_mape_s = ajustes_s[mejor_modelo_s]["MAPE"]
            prom_sigma_s = ajustes_s[mejor_modelo_s]["Sigma_q"]

            # --- Fila de guardado y métricas ---
            col_guardar_s, col_mape_s, col_sigma_s = st.columns([1, 1, 1])
                
            with col_guardar_s:
                # --- AUTO-GUARDADO SILENCIOSO ---
                if len(H_fino_s) > 0 and len(Q_suave_s) > 0 and np.isfinite(Q_suave_s).any():
                    st.session_state.stevens_curve = pd.DataFrame({"H": H_fino_s, "Q": Q_suave_s})
                    st.session_state.stevens_error = prom_mape_s
                    st.session_state.stevens_error_sigma = prom_sigma_s
                    st.success("✅ Auto-guardado: Curva en memoria")
                else:
                    st.error("No hay curva válida para guardar")
                
                # --- LA MAGIA DE LA RESTAURACIÓN ---
                if "stevens_curve" in st.session_state and st.session_state.stevens_curve is not None:
                    st.success("✅ Curva definitiva cargada en memoria")
                    with st.expander("Ver tabla guardada"):
                        st.dataframe(st.session_state.stevens_curve, height=150, use_container_width=True)
            with col_mape_s:
                st.metric("Error Absoluto Promedio", f"{prom_mape_s:.1f}%" if pd.notna(prom_mape_s) else "N/A")
            with col_sigma_s:
                st.metric("Error Procedimiento (σq)", f"{prom_sigma_s:.1f}%" if pd.notna(prom_sigma_s) else "N/A")

            # --- Tabla de errores (SEMÁFORO) ---
            st.markdown("---")
            st.subheader("Errores en aforos activos (Todos los modelos)")
            
            df_errores_s = pd.DataFrame(errores_data_s)
            
            # --- ORDENAR LA TABLA POR H DE MENOR A MAYOR ---
            df_errores_s = df_errores_s.sort_values(by="H (m)", ascending=True).reset_index(drop=True)
            
            # Identificar qué columnas contienen la palabra "Error"
            cols_error_s = [col for col in df_errores_s.columns if "Error" in col]
            
            # Función para aplicar la regla de color semáforo
            def color_semaforo_s(val):
                if pd.isna(val):
                    return ''
                try:
                    v = float(val)
                    if v <= 10:
                        return 'background-color: rgba(39, 174, 96, 0.4); color: white;' # Verde
                    elif v <= 50:
                        return 'background-color: rgba(243, 156, 18, 0.4); color: white;' # Amarillo
                    else:
                        return 'background-color: rgba(192, 57, 43, 0.4); color: white;' # Rojo
                except:
                    return ''

            # Diccionario para redondear correctamente
            format_dict_s = {"H (m)": "{:.2f}", "Q Aforado (m³/s)": "{:.2f}"}
            for c in cols_error_s:
                format_dict_s[c] = "{:.1f}"
                
            # Aplicar formato numérico y aplicar colores solo a las columnas de error
            styled_df_s = df_errores_s.style.format(format_dict_s).map(color_semaforo_s, subset=cols_error_s)
            
            st.dataframe(styled_df_s, use_container_width=True, hide_index=True)

# ================== PESTAÑA 5: MÉTODO ÁREA-VELOCIDAD ==================
with tab5:
    if st.session_state.get('df_aforos') is None or st.session_state.get('df_geo') is None:
        st.warning("Procesa los Aforos y genera la Tabla de Geometría primero.")
    else:
        # --- ENCABEZADO INFORMATIVO DEL MÉTODO ---
        with st.expander("📘 **Fundamento y consideraciones del método Área-Velocidad**", expanded=False):
            st.markdown("""
            **Fundamento:** Este método se basa directamente en la ecuación de continuidad:  
            \( Q = V * A \)  
            donde:
            - \( Q \): caudal (m³/s)
            - \( V \): velocidad media (m/s)
            - \( A \): área mojada (m²)

            Su aplicación requiere un buen conocimiento del comportamiento hidráulico y las características geométricas del río, así como contar con aforos que cubran los estados medios y altos.

            A diferencia de los métodos de Manning y Stevens, que buscan constantes de rugosidad, este método separa los componentes del caudal:
            - El **área** se obtiene exclusivamente de la topografía (geometría de la sección).
            - La **velocidad** se extrapola a partir de los aforos disponibles, ajustando modelos matemáticos (lineal, exponencial, logarítmico o potencial) en función del nivel \( H \).

            > *Limitación:* La calidad de la extrapolación depende fuertemente de la precisión de los aforos y de que la sección transversal sea estable (sin cambios morfológicos significativos). Se recomienda verificar que los aforos utilizados representen adecuadamente el rango de niveles a extrapolar.
            """)
        
        # --- 🚨 SISTEMA DE ACTUALIZACIÓN INTELIGENTE (Alarma + Verificación de IDs) ---
        df_a_current = st.session_state.get('df_aforos_activos', st.session_state.get('df_aforos'))
        ids_current = df_a_current['NO.'].tolist() if (df_a_current is not None and not df_a_current.empty) else []
        ids_stored = st.session_state.av_data['ID'].tolist() if (st.session_state.get('av_data') is not None and not st.session_state.av_data.empty) else []

        necesita_actualizar_av = (
            st.session_state.get('av_data') is None 
            or st.session_state.get('flag_actualizar_modelos', False)
            or ids_current != ids_stored
        )

        if necesita_actualizar_av:
            df_a = df_a_current.copy()
            p_data = st.session_state.perfil_data
            df_g = st.session_state.df_geo

            col_v_af = buscar_columna(df_a.columns, ["VELOC", "MEDIA"])
            
            if not col_v_af:
                st.sidebar.error("No se encontró la columna de Velocidad Media en Aforos.")
            
            datos_av = []
            for i, row in df_a.iterrows():
                id_aforo = row.get("NO.", i+1)
                H = row["H_m"]
                Q = row["CAUDAL TOTAL (m3/s)"]
                
                v_af = row.get(col_v_af, np.nan)
                
                idx_cercano = np.argmin(np.abs(df_g["H (m)"] - H))
                H_cercano = df_g.iloc[idx_cercano]["H (m)"]
                if abs(H_cercano - H) > 0.2:
                    st.sidebar.warning(f"Aforo {id_aforo}: H={H:.2f} no cercano a geometría (H_geo={H_cercano:.2f})")
                
                A_per = df_g.iloc[idx_cercano]["Am (m2)"]
                v_per = Q / A_per if A_per > 0 else 0
                
                if pd.notna(v_af) or v_per > 0:
                    datos_av.append({
                        "Incluir": True,
                        "ID": id_aforo,
                        "H": H,
                        "Q": Q,
                        "V_af": v_af if pd.notna(v_af) else 0,
                        "V_per": v_per,
                    })
            
            st.session_state.av_data = pd.DataFrame(datos_av)
            
            # Limpiar editor local
            if 'av_edited_df' in st.session_state:
                del st.session_state['av_edited_df']

        # --- Opciones Principales ---
        idx_fuente_v = 0
        if "fuente_v_radio" in st.session_state and st.session_state.fuente_v_radio == "Velocidad estimada del perfil":
            idx_fuente_v = 1

        fuente_v = st.radio(
            "Fuente de Velocidad:",
            options=["Velocidad de aforos", "Velocidad estimada del perfil"],
            index=idx_fuente_v,
            horizontal=True,
            key="fuente_v_radio"
        )

        df_edit_av = st.session_state.av_data.copy()
        if fuente_v == "Velocidad de aforos":
            df_edit_av["V"] = df_edit_av["V_af"]
        else:
            df_edit_av["V"] = df_edit_av["V_per"]

        # --- Persistencia ---
        if 'av_edited_df' not in st.session_state:
            default_df_av = df_edit_av.copy()
            default_df_av['Incluir'] = True
            st.session_state.av_edited_df = default_df_av

        if 'opts_modelos_av' not in st.session_state:
            st.session_state.opts_modelos_av = {"lineal": True, "exp": True, "log": True, "pot": True}

        if 'banda_error_global' not in st.session_state:
            st.session_state.banda_error_global = 15.0

        # --- Controles ---
        col_btn, _ = st.columns([0.15, 0.85])
        with col_btn:
            with st.popover("⚙️ Controles"):
                with st.form(key="av_form"):
                    st.caption("Filtro de Aforos")
                    edited_av = st.data_editor(
                        st.session_state.av_edited_df[["Incluir", "ID", "H", "Q"]],
                        column_config={
                            "Incluir": st.column_config.CheckboxColumn("Activo", default=True),
                            "ID": st.column_config.NumberColumn("ID", disabled=True),
                            "H": st.column_config.NumberColumn("H (m)", disabled=True, format="%.2f"),
                            "Q": st.column_config.NumberColumn("Q (m³/s)", disabled=True, format="%.2f"),
                        },
                        disabled=False, hide_index=True, use_container_width=True, key="av_editor"
                    )

                    st.caption("Modelos de ajuste")
                    lineal_av = st.checkbox("Lineal", value=st.session_state.opts_modelos_av["lineal"])
                    exp_av = st.checkbox("Exponencial", value=st.session_state.opts_modelos_av["exp"])
                    log_av = st.checkbox("Logarítmica", value=st.session_state.opts_modelos_av["log"])
                    pot_av = st.checkbox("Potencial", value=st.session_state.opts_modelos_av["pot"])

                    st.caption("Configuración General")
                    opciones_banda = ["Estable (10%)", "Inestable (15%)"]
                    indice_actual = 0 if st.session_state.banda_error_global == 10 else 1
                    tipo_seccion = st.radio("Tipo de banda de error:", options=opciones_banda, index=indice_actual, horizontal=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        submitted = st.form_submit_button("Aplicar")
                    with col2:
                        cancelled = st.form_submit_button("Cancelar")

                    if submitted:
                        st.session_state.av_edited_df = edited_av
                        st.session_state.opts_modelos_av["lineal"] = lineal_av
                        st.session_state.opts_modelos_av["exp"] = exp_av
                        st.session_state.opts_modelos_av["log"] = log_av
                        st.session_state.opts_modelos_av["pot"] = pot_av
                        st.session_state.banda_error_global = 10.0 if tipo_seccion == "Estable (10%)" else 15.0
                        st.rerun()
                    elif cancelled:
                        st.rerun()

        # --- Gráficas Layout ---
        col_v_av, col_q_av = st.columns(2)
        with col_v_av:
            st.subheader("Velocidad vs Nivel (H)")
            placeholder_v_av = st.empty()
        with col_q_av:
            st.subheader("Curva de Gasto")
            placeholder_q_av = st.empty()

        # --- Leer estado ---
        mostrar_lineal_av = st.session_state.opts_modelos_av["lineal"]
        mostrar_exp_av = st.session_state.opts_modelos_av["exp"]
        mostrar_log_av = st.session_state.opts_modelos_av["log"]
        mostrar_pot_av = st.session_state.opts_modelos_av["pot"]

        df_original_av = st.session_state.av_data.copy()
        df_edit = st.session_state.av_edited_df.copy()

        ids_activos = df_edit[df_edit["Incluir"] == True]["ID"].values
        ids_inactivos = df_edit[df_edit["Incluir"] == False]["ID"].values

        activos_av = df_original_av[df_original_av["ID"].isin(ids_activos)].copy()
        inactivos_av = df_original_av[df_original_av["ID"].isin(ids_inactivos)].copy()

        if fuente_v == "Velocidad de aforos":
            activos_av["V"] = activos_av["V_af"]
            inactivos_av["V"] = inactivos_av["V_af"]
        else:
            activos_av["V"] = activos_av["V_per"]
            inactivos_av["V"] = inactivos_av["V_per"]

        activos_av = activos_av[activos_av["V"] > 0].dropna(subset=["V"])
        inactivos_av = inactivos_av[inactivos_av["V"] > 0].dropna(subset=["V"])

        if len(activos_av) < 2:
            st.error("Necesitas al menos 2 puntos activos con V válido para calcular la curva.")
        else:
            H_act_av = activos_av["H"].values
            V_act_av = activos_av["V"].values
            Q_act_av = activos_av["Q"].values
            
            X_val_av = np.array(H_act_av)
            Y_val_av = np.array(V_act_av)
            
            # --- Interpoladores de geometría para simulación de Q ---
            df_g = st.session_state.df_geo.copy()
            H_min_geo = df_g["H (m)"].min()
            H_max_geo = df_g["H (m)"].max()
            
            from scipy.interpolate import interp1d
            interp_A_av = interp1d(df_g["H (m)"], df_g["Am (m2)"], kind='linear', fill_value='extrapolate')
            A_act_av = interp_A_av(X_val_av)
            
            # --- Modelos Matemáticos ---
            ajustes_av = {}
            mask_log_av = (X_val_av > 0) & (Y_val_av > 0)
            X_log_av, Y_log_av = X_val_av[mask_log_av], Y_val_av[mask_log_av]

            # Lineal
            p_lin_av = np.polyfit(X_val_av, Y_val_av, 1)
            Y_lin_av = p_lin_av[0]*X_val_av + p_lin_av[1]
            r_lin_av = np.corrcoef(Y_val_av, Y_lin_av)[0, 1]
            ajustes_av["Lineal"] = {
                "r": r_lin_av, "R2": r_lin_av**2,
                "func": lambda x: p_lin_av[0]*x + p_lin_av[1],
                "eq": f"V = {p_lin_av[0]:.3f}H + {p_lin_av[1]:.3f}"
            }

            if len(X_log_av) > 1:
                p_exp_av = np.polyfit(X_val_av, np.log(Y_val_av), 1)
                Y_exp_av = np.exp(p_exp_av[1]) * np.exp(p_exp_av[0]*X_val_av)
                r_exp_av = np.corrcoef(Y_val_av, Y_exp_av)[0, 1]
                ajustes_av["Exponencial"] = {
                    "r": r_exp_av, "R2": r_exp_av**2,
                    "func": lambda x: np.exp(p_exp_av[1]) * np.exp(p_exp_av[0]*x),
                    "eq": f"V = {np.exp(p_exp_av[1]):.3f} e^({p_exp_av[0]:.3f}H)"
                }

                p_log_av = np.polyfit(np.log(X_log_av), Y_log_av, 1)
                Y_log_fit_av = p_log_av[0]*np.log(X_log_av) + p_log_av[1]
                r_log_av = np.corrcoef(Y_log_av, Y_log_fit_av)[0, 1]
                ajustes_av["Logarítmica"] = {
                    "r": r_log_av, "R2": r_log_av**2,
                    "func": lambda x: np.where(x > 0, p_log_av[0]*np.log(x) + p_log_av[1], np.nan),
                    "eq": f"V = {p_log_av[0]:.3f} ln(H) + {p_log_av[1]:.3f}"
                }

                p_pot_av = np.polyfit(np.log(X_log_av), np.log(Y_log_av), 1)
                Y_pot_av = np.exp(p_pot_av[1]) * (X_log_av ** p_pot_av[0])
                r_pot_av = np.corrcoef(Y_log_av, Y_pot_av)[0, 1]
                ajustes_av["Potencial"] = {
                    "r": r_pot_av, "R2": r_pot_av**2,
                    "func": lambda x: np.where(x > 0, np.exp(p_pot_av[1]) * (x ** p_pot_av[0]), np.nan),
                    "eq": f"V = {np.exp(p_pot_av[1]):.3f} H^({p_pot_av[0]:.3f})"
                }

            # --- Calcular MAPE y σq para cada modelo base ---
            for k, v in ajustes_av.items():
                V_est_av = v["func"](X_val_av)
                Q_est_av = V_est_av * A_act_av
                mask_err_av = (Q_act_av != 0) & np.isfinite(Q_est_av)
                if np.any(mask_err_av):
                    v["MAPE"] = np.mean(np.abs((Q_est_av[mask_err_av] - Q_act_av[mask_err_av]) / Q_act_av[mask_err_av])) * 100
                    v["Sigma_q"] = calcular_error_procedimiento(Q_act_av[mask_err_av], Q_est_av[mask_err_av], K=2)
                else:
                    v["MAPE"] = np.nan
                    v["Sigma_q"] = np.nan

            # --- Selector y UI Curva Compuesta ---
            opciones_metodo_av = ["Automático (Menor MAPE)", "Potencial", "Logarítmica", "Exponencial", "Lineal", "Compuesta (Por tramos)"]
            idx_metodo_av = 0
            if "metodo_select_av" in st.session_state and st.session_state.metodo_select_av in opciones_metodo_av:
                idx_metodo_av = opciones_metodo_av.index(st.session_state.metodo_select_av)

            st.markdown("---")
            metodo_seleccionado_av = st.selectbox(
                "Método Matemático Principal:",
                opciones_metodo_av,
                index=idx_metodo_av,
                key="metodo_select_av"
            )

            # Lógica Compuesta
            if metodo_seleccionado_av == "Compuesta (Por tramos)":
                st.markdown("#### 🔀 Configuración de Curva Compuesta")
                col_c1, col_c2, col_c3 = st.columns(3)
                
                h_min_safe = float(H_act_av.min()) if len(H_act_av) > 0 else 0.0
                h_max_safe = float(H_max_geo)
                h_med_safe = float(np.median(H_act_av)) if len(H_act_av) > 0 else 1.0

                with col_c1:
                    h_quiebre_av = st.number_input("Nivel de Quiebre (H en m)", min_value=h_min_safe, max_value=h_max_safe, value=h_med_safe, step=0.1, key="av_quiebre")
                with col_c2:
                    modelo_inf_av = st.selectbox("Modelo Inferior (H < Quiebre)", ["Potencial", "Logarítmica", "Exponencial", "Lineal"], index=0, key="av_inf")
                with col_c3:
                    modelo_sup_av = st.selectbox("Modelo Superior (H ≥ Quiebre)", ["Exponencial", "Lineal", "Potencial", "Logarítmica"], index=1, key="av_sup")
                
                def func_compuesta_av(x, h_q=h_quiebre_av, m_inf=modelo_inf_av, m_sup=modelo_sup_av):
                    f_inf = ajustes_av[m_inf]["func"] if m_inf in ajustes_av else ajustes_av.get("Lineal")["func"]
                    f_sup = ajustes_av[m_sup]["func"] if m_sup in ajustes_av else ajustes_av.get("Lineal")["func"]
                    return np.where(x < h_q, f_inf(x), f_sup(x))
                
                V_est_comp_av = func_compuesta_av(X_val_av)
                Q_est_comp_av = V_est_comp_av * A_act_av
                mask_err_comp_av = (Q_act_av != 0) & np.isfinite(Q_est_comp_av)
                
                if np.any(mask_err_comp_av):
                    mape_comp_av = np.mean(np.abs((Q_est_comp_av[mask_err_comp_av] - Q_act_av[mask_err_comp_av]) / Q_act_av[mask_err_comp_av])) * 100
                    sigma_comp_av = calcular_error_procedimiento(Q_act_av[mask_err_comp_av], Q_est_comp_av[mask_err_comp_av], K=2)
                else:
                    mape_comp_av, sigma_comp_av = np.nan, np.nan
                
                ajustes_av["Compuesta (Por tramos)"] = {
                    "func": func_compuesta_av,
                    "eq": f"{modelo_inf_av} (<{h_quiebre_av:.2f}m) y {modelo_sup_av} (≥{h_quiebre_av:.2f}m)",
                    "MAPE": mape_comp_av,
                    "Sigma_q": sigma_comp_av,
                    "r": np.nan, "R2": np.nan
                }

            # 3. Mostrar la tabla comparativa de los modelos
            st.subheader("Comparación de modelos")
            df_modelos_av = pd.DataFrame([
                {
                    "Modelo": k, 
                    "r": f"{v['r']:.4f}" if pd.notna(v['r']) else "-", 
                    "R²": f"{v['R2']:.4f}" if pd.notna(v['R2']) else "-", 
                    "MAPE (%)": f"{v['MAPE']:.2f}" if pd.notna(v['MAPE']) else "N/A",
                    "σq (%)": f"{v['Sigma_q']:.2f}" if pd.notna(v['Sigma_q']) else "N/A",
                    "Ecuación": v['eq']
                }
                for k, v in ajustes_av.items()
            ])
            df_modelos_av['MAPE_num'] = pd.to_numeric(df_modelos_av['MAPE (%)'], errors='coerce')
            df_modelos_av = df_modelos_av.sort_values('MAPE_num', na_position='last').drop(columns=['MAPE_num'])
            st.dataframe(df_modelos_av, use_container_width=True, hide_index=True)

            # 4. Seleccionar modelo definitivo
            if metodo_seleccionado_av == "Automático (Menor MAPE)":
                mejor_modelo_av = min(
                    (k for k in ajustes_av if pd.notna(ajustes_av[k]["MAPE"]) and k != "Compuesta (Por tramos)"),
                    key=lambda k: ajustes_av[k]["MAPE"],
                    default=list(ajustes_av.keys())[0]
                )
            else:
                mejor_modelo_av = metodo_seleccionado_av
            
            funcion_optima_av = ajustes_av[mejor_modelo_av]["func"]
            
            r2_display_av = f" (R² = {ajustes_av[mejor_modelo_av]['R2']:.4f})" if pd.notna(ajustes_av[mejor_modelo_av]['R2']) else ""
            st.info(f"Modelo seleccionado (Línea principal): **{mejor_modelo_av}** con MAPE = {ajustes_av[mejor_modelo_av]['MAPE']:.2f}%{r2_display_av}")

            # --- Aplicar a geometría (extrapolación) ---
            paso_fino = 0.2
            H_fino_av = np.arange(H_min_geo, H_max_geo + paso_fino, paso_fino)
            
            V_fino_av = funcion_optima_av(H_fino_av)
            A_fino_av = interp_A_av(H_fino_av)
            
            Q_fino_av = V_fino_av * A_fino_av

            # --- Filtrar valores no finitos y negativos ---
            mask_finite_av = np.isfinite(Q_fino_av)
            if not np.all(mask_finite_av):
                st.warning(f"Se encontraron {np.sum(~mask_finite_av)} valores no finitos en la curva. Se eliminarán.")
                H_fino_av = H_fino_av[mask_finite_av]
                Q_fino_av = Q_fino_av[mask_finite_av]

            Q_fino_av = np.maximum(Q_fino_av, 0)

            # Suavizado con interpolación lineal
            mask_dentro_av = (H_fino_av >= H_min_geo) & (H_fino_av <= H_max_geo)
            H_dentro_av = H_fino_av[mask_dentro_av]
            Q_dentro_av = Q_fino_av[mask_dentro_av]

            if len(H_dentro_av) > 1:
                f_lin_av = interp1d(H_dentro_av, Q_dentro_av, kind='linear', fill_value='extrapolate')
                Q_suave_av = f_lin_av(H_fino_av)
            else:
                Q_suave_av = Q_fino_av

            # Colores base
            color_activos = '#00ccff'
            color_inactivos = '#ff5555'
            color_ajuste = '#55ff55'
            color_curva = '#ffaa00'
            color_banda = 'rgba(200, 200, 200, 0.15)'

            # --- Preparar funciones adicionales para la gráfica V vs H ---
            funcs_adic_av = {}
            if 'Lineal' in ajustes_av: funcs_adic_av['lineal'] = (ajustes_av['Lineal']['func'], mostrar_lineal_av)
            if 'Exponencial' in ajustes_av: funcs_adic_av['exp'] = (ajustes_av['Exponencial']['func'], mostrar_exp_av)
            if 'Logarítmica' in ajustes_av: funcs_adic_av['log'] = (ajustes_av['Logarítmica']['func'], mostrar_log_av)
            if 'Potencial' in ajustes_av: funcs_adic_av['pot'] = (ajustes_av['Potencial']['func'], mostrar_pot_av)

            H_smooth_av = np.linspace(min(X_val_av)*0.5, max(X_val_av)*1.5, 200)
            V_sel_av = funcion_optima_av(H_smooth_av)

            inactivos_v_av = inactivos_av[['H', 'V']].rename(columns={'V': 'Y'}) if not inactivos_av.empty else None

            fig_v_av = crear_figura_k(
                titulo="Velocidad vs Nivel (H) - Área-Velocidad",
                H_act=H_act_av,
                Y_act=V_act_av,
                inactivos=inactivos_v_av,
                H_smooth=H_smooth_av,
                funciones_adicionales=funcs_adic_av,
                Y_sel=V_sel_av,
                color_activos=color_activos,
                color_inactivos=color_inactivos,
                color_ajuste=color_ajuste,
                xlabel="Nivel H (m)",
                ylabel="V (m/s)"
            )
            placeholder_v_av.plotly_chart(fig_v_av, use_container_width=True)

            # --- Gráfica Curva de Gasto con banda global Y TODOS LOS MODELOS ACTIVOS ---
            fig_q_av = crear_figura_curva(
                titulo=f"Curva de Gasto - usando {fuente_v}",
                H_fino=H_fino_av,
                Q_suave=Q_suave_av,
                Q_act=Q_act_av,
                H_act=H_act_av,
                inactivos=inactivos_av[['H', 'Q']] if not inactivos_av.empty else None,
                color_curva=color_curva,
                color_activos=color_activos,
                color_inactivos=color_inactivos,
                color_banda=color_banda,
                banda_pct=st.session_state.banda_error_global
            )
            
            # AGREGAR MODELOS ADICIONALES A LA CURVA DE GASTO
            colores_extras = {'Lineal': 'cyan', 'Exponencial': 'magenta', 'Logarítmica': 'yellow', 'Potencial': 'lime'}
            mapa_nombres_av = {'Lineal': mostrar_lineal_av, 'Exponencial': mostrar_exp_av, 'Logarítmica': mostrar_log_av, 'Potencial': mostrar_pot_av}
            
            for nombre_mod, dict_mod in ajustes_av.items():
                if mapa_nombres_av.get(nombre_mod, False) and nombre_mod != mejor_modelo_av:
                    V_extra_av = dict_mod["func"](H_fino_av)
                    Q_extra_av = V_extra_av * interp_A_av(H_fino_av)
                    Q_extra_av = np.maximum(Q_extra_av, 0)
                    
                    fig_q_av.add_trace(go.Scatter(
                        x=Q_extra_av, y=H_fino_av, mode='lines', name=f'Q ({nombre_mod})',
                        line=dict(color=colores_extras[nombre_mod], width=2, dash='dot')
                    ))

            placeholder_q_av.plotly_chart(fig_q_av, use_container_width=True)

            # --- NUEVA TABLA DE ERRORES CON TODOS LOS MODELOS ---
            errores_data_av = []
            for _, row in activos_av.iterrows():
                h_val_av = row["H"]
                q_obs_av = row["Q"]
                a_val_av = float(interp_A_av(h_val_av))
                
                fila_error_av = {
                    "H (m)": h_val_av, 
                    "Q Aforado (m³/s)": q_obs_av
                }
                
                # Calcular el error para cada modelo matemático
                for nombre_mod, dict_mod in ajustes_av.items():
                    v_est_av = dict_mod["func"](h_val_av)
                    q_est_av = v_est_av * a_val_av
                    err_pct_av = abs(q_est_av - q_obs_av) / q_obs_av * 100 if q_obs_av != 0 else np.nan
                    fila_error_av[f"Error {nombre_mod} (%)"] = err_pct_av
                    
                errores_data_av.append(fila_error_av)

            # --- Obtener errores globales del mejor modelo ---
            prom_mape_av = ajustes_av[mejor_modelo_av]["MAPE"]
            prom_sigma_av = ajustes_av[mejor_modelo_av]["Sigma_q"]

            # --- Fila de guardado y métricas ---
            col_guardar_av, col_mape_av, col_sigma_av = st.columns([1, 1, 1])

            with col_guardar_av:
                # --- AUTO-GUARDADO SILENCIOSO ---
                if len(H_fino_av) > 0 and len(Q_suave_av) > 0 and np.isfinite(Q_suave_av).any():
                    st.session_state.av_curve = pd.DataFrame({"H": H_fino_av, "Q": Q_suave_av})
                    st.session_state.av_error = prom_mape_av
                    st.session_state.av_error_sigma = prom_sigma_av
                    st.success("✅ Auto-guardado: Curva en memoria")
                else:
                    st.error("No hay curva válida para guardar")
                
                if "av_curve" in st.session_state and st.session_state.av_curve is not None:
                    st.success("✅ Curva definitiva cargada en memoria")
                    with st.expander("Ver tabla guardada"):
                        st.dataframe(st.session_state.av_curve, height=150, use_container_width=True)
            with col_mape_av:
                st.metric("MAPE Promedio", f"{prom_mape_av:.1f}%" if pd.notna(prom_mape_av) else "N/A")
            with col_sigma_av:
                st.metric("Error Procedimiento (σq)", f"{prom_sigma_av:.1f}%" if pd.notna(prom_sigma_av) else "N/A")

            # --- Tabla de errores (SEMÁFORO) ---
            st.markdown("---")
            st.subheader("Errores en aforos activos (Todos los modelos)")
            
            df_errores_av = pd.DataFrame(errores_data_av)
            
            # --- ORDENAR LA TABLA POR H DE MENOR A MAYOR ---
            df_errores_av = df_errores_av.sort_values(by="H (m)", ascending=True).reset_index(drop=True)
            
            # Identificar qué columnas contienen la palabra "Error"
            cols_error_av = [col for col in df_errores_av.columns if "Error" in col]
            
            # Función para aplicar la regla de color semáforo
            def color_semaforo_av(val):
                if pd.isna(val):
                    return ''
                try:
                    v = float(val)
                    if v <= 10:
                        return 'background-color: rgba(39, 174, 96, 0.4); color: white;' # Verde
                    elif v <= 50:
                        return 'background-color: rgba(243, 156, 18, 0.4); color: white;' # Amarillo
                    else:
                        return 'background-color: rgba(192, 57, 43, 0.4); color: white;' # Rojo
                except:
                    return ''

            format_dict_av = {"H (m)": "{:.2f}", "Q Aforado (m³/s)": "{:.2f}"}
            for c in cols_error_av:
                format_dict_av[c] = "{:.1f}"
                
            # Aplicar formato numérico y aplicar colores solo a las columnas de error
            styled_df_av = df_errores_av.style.format(format_dict_av).map(color_semaforo_av, subset=cols_error_av)
            
            st.dataframe(styled_df_av, use_container_width=True, hide_index=True)

# ================== PESTAÑA 6: MÍNIMOS Y H0 ==================
with tab6:
    st.header("Cálculo de H0 (Nivel de Caudal Nulo)")

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info("""
        **Método de Running:**
        1. Toma los extremos teóricos de la curva generada (Q-MIN y Q-MAX).
        2. Q-INT es la progresión geométrica de estos extremos.
        3. H0 se calcula por la intersección analítica de rectas de la envolvente.
        """)
    with col_info2:
        st.success("""
        **Método de Johnson:**
        1. Toma como límites los caudales reales aforados (Q1 = Mínimo, Q2 = Máximo).
        2. Q3 es la progresión geométrica (Q3 = √(Q1 * Q2)).
        3. Evalúa H1, H2 y H3 en la curva y aplica la fórmula directa de H0.
        """)

    llaves = {
        "Manning": "manning_curve",
        "Stevens": "stevens_curve",
        "Área-Velocidad": "av_curve"
    }

    resultados_running = {}
    resultados_johnson = {}
    resumen_h0 = []

    from scipy.interpolate import interp1d
    import numpy as np
    import pandas as pd

    df_aforos = st.session_state.df_aforos
    hay_aforos = df_aforos is not None and not df_aforos.empty
    if hay_aforos:
        Q1_j = df_aforos["CAUDAL TOTAL (m3/s)"].min()
        Q2_j = df_aforos["CAUDAL TOTAL (m3/s)"].max()
        Q3_j = np.sqrt(Q1_j * Q2_j)
    else:
        Q1_j = Q2_j = Q3_j = 0

    for nombre, llave in llaves.items():
        if llave in st.session_state and st.session_state[llave] is not None:
            df = st.session_state[llave].copy()
            df_positivos = df[df["Q"] > 0.001].sort_values(by="H").drop_duplicates(subset=["H"])

            if len(df_positivos) > 3:
                df_sorted_q = df_positivos.sort_values(by="Q").drop_duplicates(subset=["Q"])
                f_H_inversa = interp1d(df_sorted_q["Q"], df_sorted_q["H"], kind='linear', bounds_error=False, fill_value='extrapolate')

                # --- RUNNING ---
                H_max = df_positivos["H"].max()
                H_min = df_positivos["H"].min()
                Q_max = df_positivos.loc[df_positivos["H"] == H_max, "Q"].values[0]
                Q_min = df_positivos.loc[df_positivos["H"] == H_min, "Q"].values[0]
                Q_int = np.sqrt(Q_max * Q_min)

                H_max_calc = float(f_H_inversa(Q_max))
                H_int = float(f_H_inversa(Q_int))
                H_min_calc = float(f_H_inversa(Q_min))

                denominador = (Q_min - Q_int) if (Q_min - Q_int) != 0 else 0.0001
                m1 = (H_int - H_max_calc) / denominador
                b1 = H_max_calc - m1 * Q_int
                m2 = (H_min_calc - H_int) / denominador
                b2 = H_int - m2 * Q_int
                m_diff = (m2 - m1) if (m2 - m1) != 0 else 0.0001
                Q_prima = (b1 - b2) / m_diff
                Ho_run = b2 + m2 * Q_prima

                resultados_running[nombre] = {
                    "Q-MAX": Q_max, "Q-INT": Q_int, "Q-MIN": Q_min,
                    "H-MAX": H_max_calc, "H-INT": H_int, "H-MIN": H_min_calc,
                    "m1 (%)": m1 * 100, "b1": b1, "m2 (%)": m2 * 100, "b2": b2,
                    "Q'": Q_prima, "Ho": Ho_run
                }

                # --- JOHNSON ---
                Ho_johnson = np.nan
                if hay_aforos:
                    H1_j = float(f_H_inversa(Q1_j))
                    H2_j = float(f_H_inversa(Q2_j))
                    H3_j = float(f_H_inversa(Q3_j))
                    denom_j = (H1_j + H2_j - 2 * H3_j)
                    if denom_j != 0:
                        Ho_johnson = (H1_j * H2_j - H3_j**2) / denom_j
                    resultados_johnson[nombre] = {
                        "Q-1 (Mín Aforo)": Q1_j, "Q-2 (Máx Aforo)": Q2_j, "Q-3 (Intermedio)": Q3_j,
                        "H-1": H1_j, "H-2": H2_j, "H-3": H3_j,
                        "Ho": Ho_johnson
                    }

                # Guardar para el resumen
                resumen_h0.append({
                    "Método": nombre,
                    "H0 (Running)": f"{Ho_run:.3f} m",
                    "H0 (Johnson)": f"{Ho_johnson:.3f} m" if pd.notna(Ho_johnson) else "N/A",
                    "_val_run": Ho_run,
                    "_val_john": Ho_johnson,
                    "_run_negativo": Ho_run < 0,
                    "_john_negativo": pd.notna(Ho_johnson) and Ho_johnson < 0
                })

    # --- RENDERIZADO VISUAL ---
    if resumen_h0:
        st.markdown("---")
        st.subheader("🏆 Comparativa de Resultados H0")

        df_resumen = pd.DataFrame(resumen_h0)
        # Marcar valores negativos en la tabla
        def formatear_con_advertencia(val, es_negativo):
            if pd.isna(val) or val == "N/A":
                return "N/A"
            if es_negativo:
                return f"⚠️ {val} (negativo)"
            return val

        df_mostrar = df_resumen.copy()
        df_mostrar["H0 (Running)"] = df_mostrar.apply(
            lambda row: formatear_con_advertencia(row["H0 (Running)"], row["_run_negativo"]), axis=1
        )
        df_mostrar["H0 (Johnson)"] = df_mostrar.apply(
            lambda row: formatear_con_advertencia(row["H0 (Johnson)"], row["_john_negativo"]), axis=1
        )
        df_mostrar = df_mostrar[["Método", "H0 (Running)", "H0 (Johnson)"]]

        col_tabla, col_botones = st.columns([1.5, 1])

        with col_tabla:
            st.dataframe(df_mostrar, use_container_width=True, hide_index=True)

        with col_botones:
            st.write("**Selección Oficial de H0**")
            st.caption("Elige qué H0 se usará como cota base para cada método.")

            if 'h0_seleccionados' not in st.session_state:
                st.session_state.h0_seleccionados = {}
            if 'h0_fuentes' not in st.session_state: # NUEVO: Guardaremos la etiqueta
                st.session_state.h0_fuentes = {}

            for idx, row in df_resumen.iterrows():
                metodo = row["Método"]
                run_val = row["_val_run"]
                john_val = row["_val_john"] if pd.notna(row["_val_john"]) else None

                opciones = []
                valores_opciones = []
                if not row["_run_negativo"]:
                    opciones.append("Running")
                    valores_opciones.append(run_val)
                if not row["_john_negativo"] and john_val is not None:
                    opciones.append("Johnson")
                    valores_opciones.append(john_val)
                if len(opciones) == 0:
                    st.warning(f"⚠️ Para {metodo}, ambos dan H0 negativo. No se forzará la curva.")
                    st.session_state.h0_seleccionados[metodo] = np.nan
                    st.session_state.h0_fuentes[metodo] = "No aplicado" 
                    st.caption(f"**{metodo}:** Curva original (sin H0)")
                else:
                    if len(opciones) == 1:
                        st.session_state.h0_seleccionados[metodo] = valores_opciones[0]
                        st.session_state.h0_fuentes[metodo] = opciones[0] # NUEVO
                        st.caption(f"**{metodo}:** seleccionado automáticamente **{opciones[0]}** (H0 = {valores_opciones[0]:.3f} m)")
                    else:
                        valor_guardado = st.session_state.h0_seleccionados.get(metodo, None)
                        if valor_guardado is not None and valor_guardado in valores_opciones:
                            indice_default = valores_opciones.index(valor_guardado)
                        else:
                            indice_default = 0

                        eleccion = st.radio(
                            f"Para {metodo}:",
                            options=opciones,
                            index=indice_default,
                            horizontal=True,
                            key=f"radio_h0_{metodo}"
                        )
                        if eleccion == "Running":
                            st.session_state.h0_seleccionados[metodo] = run_val
                            st.session_state.h0_fuentes[metodo] = "Running" # NUEVO
                        else:
                            st.session_state.h0_seleccionados[metodo] = john_val
                            st.session_state.h0_fuentes[metodo] = "Johnson" # NUEVO

        st.success("✅ Selecciones guardadas en memoria para aplicar a los cálculos finales.")

        # --- TABLAS DETALLADAS OCULTAS ---
        st.markdown("---")
        st.subheader("Desglose de Cálculos")

        with st.expander("🔍 Ver parámetros detallados del Método de Running"):
            if resultados_running:
                df_res_run = pd.DataFrame(resultados_running)
                st.dataframe(df_res_run.style.format("{:.3f}"), use_container_width=True)

        with st.expander("🔍 Ver parámetros detallados del Método de Johnson"):
            if resultados_johnson:
                df_res_john = pd.DataFrame(resultados_johnson)
                st.dataframe(df_res_john.style.format("{:.3f}"), use_container_width=True)

        with st.expander("📐 Ver validación matemática formal (Ambos métodos)"):
            st.write("""
            Aunque Running y Johnson tienen enfoques distintos para seleccionar la terna de caudales evaluados, **ambos convergen en la misma resolución parabólica para estimar el nivel de caudal nulo ($H_0$)**:
            """)
            st.latex(r"H_0 = \frac{H_1 \cdot H_2 - H_3^2}{H_1 + H_2 - 2H_3}")
            st.write("""
            * **En Running:** $H_1$ y $H_2$ son los extremos de la **curva generada**.
            * **En Johnson:** $H_1$ y $H_2$ son estrictamente los niveles asociados a los caudales **mínimo y máximo aforados en campo**.
            * En ambos, $H_3$ es el nivel evaluado en la media geométrica de los dos caudales extremos elegidos.
            """)
    else:
        st.warning("Aún no hay curvas guardadas. Ve a las pestañas de Manning, Stevens o Área-Velocidad y presiona el botón 'Guardar curva' para que aparezcan aquí.")

# ================== PESTAÑA 7: COMPARACIÓN DE MÉTODOS ==================
with tab7:
    st.header("Comparación de Métodos de Extrapolación")

    if (st.session_state.get("manning_curve") is None or 
        st.session_state.get("stevens_curve") is None or 
        st.session_state.get("av_curve") is None):
        st.warning("Primero debes calcular las curvas en cada método (pestañas 3, 4 y 5) con al menos 2 puntos activos.")
    else:
        if st.session_state.get("df_geo") is None:
            st.error("No hay datos de geometría. Genera la tabla en la pestaña Geometría.")
        else:
            H_niveles_base = np.sort(st.session_state.df_geo["H (m)"].unique())
            
            from scipy.interpolate import interp1d
            import numpy as np
            import plotly.graph_objects as go
            
            # Obtener los H0 y sus fuentes
            h0_dict = st.session_state.get('h0_seleccionados', {})
            h0_fuentes = st.session_state.get('h0_fuentes', {}) 

            h0_man = h0_dict.get('Manning', None)
            h0_ste = h0_dict.get('Stevens', None)
            h0_av = h0_dict.get('Área-Velocidad', None)
            
            fuente_man = h0_fuentes.get('Manning', '') 
            fuente_ste = h0_fuentes.get('Stevens', '') 
            fuente_av = h0_fuentes.get('Área-Velocidad', '') 

            if 'df_aforos_activos' in st.session_state:
                df_aforos_comp = st.session_state.df_aforos_activos
            else:
                df_aforos_comp = st.session_state.df_aforos

            def preparar_curva_grafico(df_curve, h0_val):
                if df_curve is None or df_curve.empty:
                    return [], []
                
                H_vals = df_curve["H"].values
                Q_vals = df_curve["Q"].values
                
                mask_validos = Q_vals >= 0
                if pd.notna(h0_val):
                    mask_validos = mask_validos & (H_vals > h0_val)
                
                H_clean = H_vals[mask_validos]
                Q_clean = Q_vals[mask_validos]
                
                if pd.notna(h0_val):
                    H_clean = np.insert(H_clean, 0, h0_val)
                    Q_clean = np.insert(Q_clean, 0, 0.0)
                    
                return H_clean, Q_clean

            H_man_plot, Q_man_plot = preparar_curva_grafico(st.session_state.manning_curve, h0_man)
            H_ste_plot, Q_ste_plot = preparar_curva_grafico(st.session_state.stevens_curve, h0_ste)
            H_av_plot, Q_av_plot = preparar_curva_grafico(st.session_state.av_curve, h0_av)

            f_man = interp1d(st.session_state.manning_curve["H"], st.session_state.manning_curve["Q"], 
                             kind='linear', fill_value='extrapolate', bounds_error=False)
            f_ste = interp1d(st.session_state.stevens_curve["H"], st.session_state.stevens_curve["Q"], 
                             kind='linear', fill_value='extrapolate', bounds_error=False)
            f_av = interp1d(st.session_state.av_curve["H"], st.session_state.av_curve["Q"], 
                            kind='linear', fill_value='extrapolate', bounds_error=False)

            def calcular_q_tabla(f_interp, H_arr, h0_val):
                Q_arr = f_interp(H_arr)
                if pd.notna(h0_val):
                    Q_arr = np.where(H_arr < h0_val, np.nan, Q_arr)
                Q_arr = np.where(Q_arr < 0, np.nan, Q_arr)
                return Q_arr

            st.markdown("### 🔍 Análisis Visual por Niveles de Interés")
            st.write("Ajusta estos niveles (ej. percentiles P25 y P75) para trazar líneas de referencia en la gráfica y añadirlos a la tabla.")
            
            h_min_sug = df_aforos_comp["H_m"].quantile(0.25) if not df_aforos_comp.empty else H_niveles_base[0]
            h_max_sug = df_aforos_comp["H_m"].quantile(0.75) if not df_aforos_comp.empty else H_niveles_base[-1]

            # Inicializar en la sesión solo si no existen previamente (así respeta lo que cargues del .pkl)
            if "nivel_interes_1" not in st.session_state:
                st.session_state.nivel_interes_1 = float(h_min_sug)
            if "nivel_interes_2" not in st.session_state:
                st.session_state.nivel_interes_2 = float(h_max_sug)

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                # Quitamos el 'value' y ponemos el 'key' para que Streamlit lo maneje automáticamente
                nivel_interes_1 = st.number_input("Nivel 1 (Aguas bajas/P25):", step=0.1, format="%.3f", key="nivel_interes_1")
            with col_p2:
                nivel_interes_2 = st.number_input("Nivel 2 (Aguas altas/P75):", step=0.1, format="%.3f", key="nivel_interes_2")

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Error Manning", f"{st.session_state.manning_error:.1f}%" if st.session_state.get("manning_error") else "N/A")
            with col2:
                st.metric("Error Stevens", f"{st.session_state.stevens_error:.1f}%" if st.session_state.get("stevens_error") else "N/A")
            with col3:
                st.metric("Error Área-Velocidad", f"{st.session_state.av_error:.1f}%" if st.session_state.get("av_error") else "N/A")

            fig_comp = go.Figure()
            
            if len(H_man_plot) > 0:
                fig_comp.add_trace(go.Scatter(x=Q_man_plot, y=H_man_plot, mode='lines', name='Manning', line=dict(color='#00ccff', width=3)))
            if len(H_ste_plot) > 0:
                fig_comp.add_trace(go.Scatter(x=Q_ste_plot, y=H_ste_plot, mode='lines', name='Stevens', line=dict(color='#ffaa00', width=3)))
            if len(H_av_plot) > 0:
                fig_comp.add_trace(go.Scatter(x=Q_av_plot, y=H_av_plot, mode='lines', name='Área-Velocidad', line=dict(color='#55ff55', width=3)))
            
            if pd.notna(h0_man):
                fig_comp.add_trace(go.Scatter(x=[0], y=[h0_man], mode='markers', name='H0 Manning', marker=dict(color='#00ccff', symbol='star', size=12, line=dict(color='white', width=1))))
            if pd.notna(h0_ste):
                fig_comp.add_trace(go.Scatter(x=[0], y=[h0_ste], mode='markers', name='H0 Stevens', marker=dict(color='#ffaa00', symbol='star', size=12, line=dict(color='white', width=1))))
            if pd.notna(h0_av):
                fig_comp.add_trace(go.Scatter(x=[0], y=[h0_av], mode='markers', name='H0 Á-V', marker=dict(color='#55ff55', symbol='star', size=12, line=dict(color='white', width=1))))

            if df_aforos_comp is not None and not df_aforos_comp.empty:
                fig_comp.add_trace(go.Scatter(x=df_aforos_comp["CAUDAL TOTAL (m3/s)"], y=df_aforos_comp["H_m"], mode='markers', name='Aforos', marker=dict(color='white', size=8, line=dict(color='black', width=1))))
            
            fig_comp.add_hline(
                y=nivel_interes_1, line_dash="dash", line_color="#e74c3c", line_width=2,
                annotation_text=f"Nivel 1: {nivel_interes_1:.2f} m", annotation_position="bottom right",
                annotation_font_color="#e74c3c"
            )
            fig_comp.add_hline(
                y=nivel_interes_2, line_dash="dash", line_color="#9b59b6", line_width=2,
                annotation_text=f"Nivel 2: {nivel_interes_2:.2f} m", annotation_position="top right",
                annotation_font_color="#9b59b6"
            )

            fig_comp.update_layout(title="Curvas de Gasto Comparativas", xaxis_title="Caudal Q (m³/s)", yaxis_title="Nivel H (m)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_comp, use_container_width=True)

            st.markdown("---")

            H_niveles_extendidos = np.append(H_niveles_base, [nivel_interes_1, nivel_interes_2])
            H_niveles = np.unique(np.sort(H_niveles_extendidos))

            df_comp = pd.DataFrame({
                "H (m)": H_niveles,
                "Q_man (m³/s)": calcular_q_tabla(f_man, H_niveles, h0_man),
                "Q_ste (m³/s)": calcular_q_tabla(f_ste, H_niveles, h0_ste),
                "Q_av (m³/s)": calcular_q_tabla(f_av, H_niveles, h0_av)
            })

            df_comp["Nota"] = ""
            df_comp.loc[df_comp["H (m)"] == nivel_interes_1, "Nota"] = "🔴 Nivel Int. 1"
            df_comp.loc[df_comp["H (m)"] == nivel_interes_2, "Nota"] = "🟣 Nivel Int. 2"

            st.subheader("Tabla de Caudales por Nivel")
            
            # --- NUEVO: AÑADIENDO LA FUENTE AL TEXTO ---
            cols_h0 = st.columns(3)
            txt_man = f"**H0 Manning:** {h0_man:.3f} m ({fuente_man})" if pd.notna(h0_man) else "**H0 Manning:** N/A"
            txt_ste = f"**H0 Stevens:** {h0_ste:.3f} m ({fuente_ste})" if pd.notna(h0_ste) else "**H0 Stevens:** N/A"
            txt_av = f"**H0 Á-V:** {h0_av:.3f} m ({fuente_av})" if pd.notna(h0_av) else "**H0 Á-V:** N/A"

            cols_h0[0].info(txt_man)
            cols_h0[1].warning(txt_ste)
            cols_h0[2].success(txt_av)
            
            formato_caudal = lambda x: f"{x:.2f}" if pd.notna(x) else "-"
            
            st.dataframe(df_comp.style.format({
                "H (m)": "{:.3f}", "Q_man (m³/s)": formato_caudal, "Q_ste (m³/s)": formato_caudal, "Q_av (m³/s)": formato_caudal
            }), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("🏆 Selección de Curva Definitiva")
            
            col_sel, col_desc = st.columns([1.5, 1])
            with col_sel:
                metodo_definitivo = st.radio(
                    "Con base en el error visual y numérico, selecciona el método que conformará tu Curva de Gasto Oficial:",
                    options=["Manning", "Stevens", "Área-Velocidad"],
                    horizontal=True, key="Curva_selec"
                )
                st.session_state.metodo_definitivo = metodo_definitivo
            
            with col_desc:
                col_map = {
                    "Manning": "Q_man (m³/s)",
                    "Stevens": "Q_ste (m³/s)",
                    "Área-Velocidad": "Q_av (m³/s)"
                }
                
                # 1. Dejar solo la columna de H y la del método seleccionado
                df_export = df_comp[["H (m)", col_map[metodo_definitivo]]].copy()
                
                # 2. Renombrar la columna de caudal para que quede limpia y general
                df_export = df_export.rename(columns={col_map[metodo_definitivo]: "Q (m³/s)"})
                
                # 3. Redondear a 3 decimales
                df_export = df_export.round({"H (m)": 3, "Q (m³/s)": 3})
                csv_export = df_export.to_csv(index=False).encode('utf-8')
                
                # 4. Obtener el código/nombre de la estación y la fecha para el nombre del archivo
                import datetime
                estacion_codigo = "Estacion_Desconocida"
                if st.session_state.get('perfil_data'):
                    # Intenta buscar 'codigo', si no lo halla, usa 'estacion'
                    estacion_codigo = str(st.session_state.perfil_data.get('codigo', st.session_state.perfil_data.get('estacion', 'Estacion'))).replace(" ", "_")
                
                fecha_hoy_csv = datetime.datetime.now().strftime("%Y%m%d")
                nombre_archivo = f"Curva_Definitiva_{estacion_codigo}_{fecha_hoy_csv}.csv"
                
                st.write("") 
                st.download_button(
                    label=f"💾 Exportar Curva {metodo_definitivo} (CSV)",
                    data=csv_export,
                    file_name=nombre_archivo, 
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
            
            st.caption(f"El archivo exportado contiene únicamente las columnas de Nivel (H) y Caudal (Q) basadas en el método de {metodo_definitivo}.")do_definitivo}, manteniendo las demás como referencia.")

            with st.expander("Ver aforos utilizados"):
                if df_aforos_comp is not None and not df_aforos_comp.empty:
                    cols_aforo = ["NO.", "FECHA", "H_m", "CAUDAL TOTAL (m3/s)", "ÁREA SEC. (m2)", "VELOC. MEDIA (m/s)"]
                    cols_existentes = [c for c in cols_aforo if c in df_aforos_comp.columns]
                    df_af_mostrar = df_aforos_comp[cols_existentes].copy()
                    df_af_mostrar = df_af_mostrar.rename(columns={"H_m": "H (m)", "CAUDAL TOTAL (m3/s)": "Q (m³/s)"})
                    st.dataframe(df_af_mostrar, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("💾 Exportar Resultados Completos")
    st.markdown("Descarga un archivo Excel con todas las tablas calculadas: geometría, variables de los aforos y las curvas de gasto extrapoladas de los 3 métodos.")

    # Función para empaquetar todo en un Excel en memoria
    def generar_excel_exportacion():
        output = io.BytesIO()
        # Usamos xlsxwriter como motor para crear el archivo Excel
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            
            # 1. Exportar Geometría
            if st.session_state.get('df_geo') is not None:
                st.session_state.df_geo.to_excel(writer, sheet_name='Geometria', index=False)
            
            # 2. Exportar Manning
            if st.session_state.get('manning_data') is not None:
                st.session_state.manning_data.to_excel(writer, sheet_name='Manning_Aforos', index=False)
            if st.session_state.get('manning_curve') is not None:
                st.session_state.manning_curve.to_excel(writer, sheet_name='Manning_Curva', index=False)
            
            # 3. Exportar Stevens
            if st.session_state.get('stevens_data') is not None:
                st.session_state.stevens_data.to_excel(writer, sheet_name='Stevens_Aforos', index=False)
            if st.session_state.get('stevens_curve') is not None:
                st.session_state.stevens_curve.to_excel(writer, sheet_name='Stevens_Curva', index=False)
            
            # 4. Exportar Área-Velocidad
            if st.session_state.get('av_data') is not None:
                st.session_state.av_data.to_excel(writer, sheet_name='AreaVelocidad_Aforos', index=False)
            if st.session_state.get('av_curve') is not None:
                st.session_state.av_curve.to_excel(writer, sheet_name='AreaVelocidad_Curva', index=False)
            
            # 5. Hoja Resumen de Errores
            resumen_errores = pd.DataFrame({
                "Método": ["Manning", "Stevens", "Área-Velocidad"],
                "MAPE Promedio (%)": [
                    st.session_state.get('manning_error'),
                    st.session_state.get('stevens_error'),
                    st.session_state.get('av_error')
                ],
                "Error de Procedimiento σq (%)": [
                    st.session_state.get('manning_error_sigma'),
                    st.session_state.get('stevens_error_sigma'),
                    st.session_state.get('av_error_sigma')
                ]
            })
            resumen_errores.to_excel(writer, sheet_name='Resumen_Errores', index=False)

        return output.getvalue()

    # Mostrar el botón solo si al menos hay datos procesados (ej. Geometría)
    if st.session_state.get('df_geo') is not None:
        excel_data = generar_excel_exportacion()
        
        # Generar nombre dinámico para el archivo Excel
        estacion_nombre = "Estacion_Desconocida"
        if st.session_state.get('perfil_data'):
            estacion_nombre = str(st.session_state.perfil_data.get('estacion', 'Estacion')).replace(" ", "_")
        fecha_hoy = datetime.datetime.now().strftime("%Y%m%d")
        
        st.download_button(
            label="📥 Descargar todos los cálculos a Excel (.xlsx)",
            data=excel_data,
            file_name=f"Calculos_Curva_{estacion_nombre}_{fecha_hoy}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )
    else:
        st.info("💡 Procesa los aforos y genera la geometría para habilitar la exportación de cálculos.")

# ================== PESTAÑA 8: HISTÓRICO DE CURVAS ==================
with tab8:
    st.header("Comparación con el Histórico de Curvas")
    st.markdown("Sube el archivo institucional con el histórico para comparar la nueva curva calculada. **(Recomendado: usar formato .parquet o .pkl para mayor velocidad)**")

    # 1. Cargar el archivo (AHORA ACEPTA MULTIPLES FORMATOS)
    archivo_historico = st.file_uploader(
        "Sube el archivo histórico (.xlsx, .xls, .parquet, .pkl, .csv)", 
        type=["xlsx", "xls", "parquet", "pkl", "csv"], 
        key="file_hist"
    )
    
    if archivo_historico is not None:
            # Detectar el tipo de archivo y cargarlo
            nombre_archivo = archivo_historico.name.lower()
            
            try:
                if nombre_archivo.endswith('.parquet'):
                    df_hist_full = pd.read_parquet(archivo_historico)
                elif nombre_archivo.endswith('.pkl'):
                    df_hist_full = pd.read_pickle(archivo_historico)
                elif nombre_archivo.endswith('.csv'):
                    df_hist_full = pd.read_csv(archivo_historico, sep=None, engine='python', decimal=',', encoding='utf-8-sig')
                else:
                    df_hist_full = cargar_historico_excel(archivo_historico)
                    
                # Limpiar espacios extra
                df_hist_full.columns = df_hist_full.columns.str.strip()
                
                # 🛑 RADAR DE DEBUG (Si no encuentra la columna, nos mostrará por qué)
                if 'Etiqueta_Estacion' not in df_hist_full.columns:
                    st.error("❌ No se encontró la columna 'Etiqueta_Estacion'.")
                    st.warning("Pandas detectó que tu archivo tiene ESTAS columnas:")
                    st.write(df_hist_full.columns.tolist())
                    st.stop() # Detiene la app aquí, evitando el "KeyError" rojo y feo
                    
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")
                st.stop()
            
            col_est, col_empty = st.columns([1, 2])
            with col_est:
                # --- NUEVO: Extraer código de la estación automáticamente ---
                codigo_default = ""
                if st.session_state.get('perfil_data') and st.session_state.perfil_data.get('codigo'):
                    codigo_default = str(st.session_state.perfil_data['codigo']).strip()
                    
                # Se usa codigo_default si no hay nada en session_state
                valor_input = st.session_state.get('codigo_estacion', codigo_default)
                codigo_estacion = st.text_input("Etiqueta_Estacion a filtrar:", value=valor_input)
            
            if codigo_estacion:
                # Asegurar que ambos lados sean texto para una comparación exacta
                df_hist_full['Etiqueta_Estacion'] = df_hist_full['Etiqueta_Estacion'].astype(str)
                df_estacion = df_hist_full[df_hist_full['Etiqueta_Estacion'] == codigo_estacion].copy()
                
                if df_estacion.empty:
                    st.warning(f"No se encontraron datos históricos para la estación {codigo_estacion}.")
                else:
                    st.success(f"✅ Se encontraron {len(df_estacion)} registros para la estación {codigo_estacion}.")
                    
                    # 3. Conversión de unidades (Nivel de cm a metros)
                    if 'Nivel' in df_estacion.columns:
                        df_estacion['Nivel_m'] = df_estacion['Nivel'] / 100.0
                    else:
                        st.error("El archivo no tiene la columna 'Nivel'. Verifica la estructura de los datos.")
                        st.stop()
                        
                    # 4. Gráfica Comparativa
                    fig_hist = go.Figure()
                    
                    # Agrupar por Curva_id para trazar cada curva histórica como una línea distinta
                    if 'Curva_id' in df_estacion.columns:
                        curvas_historicas = df_estacion.groupby('Curva_id')
                        
                        for curva_id, datos_curva in curvas_historicas:
                            # Ordenar por nivel para que la línea no se cruce
                            datos_curva = datos_curva.sort_values(by='Nivel_m')
                            
                            # Extraer el año de inicio para la leyenda (si existe)
                            fecha_inicio = datos_curva['Fecha_Inicio'].iloc[0] if 'Fecha_Inicio' in datos_curva.columns else None
                            anio_str = f" ({str(fecha_inicio)[:4]})" if pd.notna(fecha_inicio) else ""
                            
                            fig_hist.add_trace(go.Scatter(
                                x=datos_curva['Caudal'],
                                y=datos_curva['Nivel_m'],
                                mode='lines+markers',
                                name=f'Histórica ID: {curva_id}{anio_str}',
                                line=dict(dash='dash', width=2), # Líneas punteadas para el histórico
                                marker=dict(size=5)
                            ))
                    
                    # 5. Agregar la Curva Definitiva Actual
                    metodo_def = st.session_state.get('metodo_definitivo', None)
                    
                    if metodo_def is not None:
                        # Mapear la selección al dataframe guardado en la Pestaña 7
                        curvas_map = {
                            "Manning": st.session_state.get("manning_curve"),
                            "Stevens": st.session_state.get("stevens_curve"),
                            "Área-Velocidad": st.session_state.get("av_curve")
                        }
                        df_curva_act = curvas_map.get(metodo_def)
                        
                        if df_curva_act is not None and not df_curva_act.empty:
                            # Obtener H0 si existe para aplicar el límite visual
                            h0_dict = st.session_state.get('h0_seleccionados', {})
                            h0_val = h0_dict.get(metodo_def, None)
                            
                            H_act = df_curva_act['H'].values
                            Q_act = df_curva_act['Q'].values
                            
                            # Filtrar datos válidos de la nueva curva
                            mask_validos = Q_act >= 0
                            if pd.notna(h0_val):
                                mask_validos = mask_validos & (H_act > h0_val)
                                H_act = np.insert(H_act[mask_validos], 0, h0_val)
                                Q_act = np.insert(Q_act[mask_validos], 0, 0.0)
                            else:
                                H_act = H_act[mask_validos]
                                Q_act = Q_act[mask_validos]
                                
                            # --- NUEVO: BANDA DEL 80% ---
                            factor_banda = st.session_state.banda_error_global / 100.0
                            Q_sup = Q_act * (1 + factor_banda)
                            Q_inf = np.maximum(Q_act * (1 - factor_banda), 0)
                            
                            # Traza del límite inferior (Invisible)
                            fig_hist.add_trace(go.Scatter(
                                x=Q_inf, y=H_act,
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                            
                            # Traza del límite superior (Rellena horizontalmente hasta el inferior)
                            fig_hist.add_trace(go.Scatter(
                                x=Q_sup, y=H_act,
                                mode='lines',
                                fill='tonextx', 
                                fillcolor='rgba(255, 51, 51, 0.15)', # Rojo muy suave y transparente
                                line=dict(width=0),
                                name=f'Banda ±{st.session_state.banda_error_global}%',
                                hoverinfo='skip'
                            ))
                            
                            # Trazar por último la línea maestra para que destaque encima del color
                            fig_hist.add_trace(go.Scatter(
                                x=Q_act,
                                y=H_act,
                                mode='lines',
                                name=f'⭐ NUEVA ({metodo_def})',
                                line=dict(color='#ff3333', width=5) # Línea gruesa roja
                            ))
                    else:
                        st.info("💡 Ve a la pestaña 'Comparación' y selecciona tu método definitivo para que aparezca en esta gráfica.")

                    # Formato de la gráfica
                    fig_hist.update_layout(
                        title=f"Evolución Histórica de Curvas de Gasto - Estación {codigo_estacion}",
                        xaxis_title="Caudal Q (m³/s)",
                        yaxis_title="Nivel H (m)",
                        hovermode="y unified", 
                        legend=dict(title="Curvas", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # 6. Tabla de datos visual para referencia
                    with st.expander("📊 Ver matriz de datos históricos (Filtrada)"):
                        columnas_mostrar = [c for c in ['Curva_id', 'Fecha_Inicio', 'Fecha_Final', 'Nivel', 'Nivel_m', 'Caudal'] if c in df_estacion.columns]
                        st.dataframe(df_estacion[columnas_mostrar].reset_index(drop=True), use_container_width=True)  
