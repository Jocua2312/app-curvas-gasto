# -*- coding: utf-8 -*-
"""
APLICACIÓN DE CURVA DE GASTO - DASHBOARD INTERACTIVO (VERSIÓN STREAMLIT)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

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

def area_mojada(abscisas, cotas, cota_cero, nivel):
    datos = _preparar_vectores(abscisas, cotas, cota_cero, nivel)
    if not datos: return 0.0
    x0, x1, y0, y1, dx, dy, h0, h1, frac, m_ambos, m_baja, m_sube = datos
    
    area = np.zeros_like(dx)
    # Trapecio completo
    area[m_ambos] = (h0[m_ambos] + h1[m_ambos]) * dx[m_ambos] / 2.0
    # Triángulo izquierdo
    area[m_baja] = h1[m_baja] * dx[m_baja] * (1 - frac[m_baja]) / 2.0
    # Triángulo derecho
    area[m_sube] = h0[m_sube] * dx[m_sube] * frac[m_sube] / 2.0
    
    return float(np.sum(area))

def perimetro_mojado(abscisas, cotas, cota_cero, nivel):
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
    
    return float(np.sum(perimetro))

def ancho_superficial(abscisas, cotas, cota_cero, nivel):
    datos = _preparar_vectores(abscisas, cotas, cota_cero, nivel)
    if not datos: return 0.0
    x0, x1, y0, y1, dx, dy, h0, h1, frac, m_ambos, m_baja, m_sube = datos
    
    ancho = np.zeros_like(dx)
    # Ancho completo
    ancho[m_ambos] = dx[m_ambos]
    # Ancho parcial izquierdo
    ancho[m_baja] = dx[m_baja] * (1 - frac[m_baja])
    # Ancho parcial derecho
    ancho[m_sube] = dx[m_sube] * frac[m_sube]
    
    return float(np.sum(ancho))

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
            'df_aforos_activos', 'h0_seleccionados', 'metodo_definitivo'
        ]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

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

        # 2. Inicializar estado temporal para aforos (si no existe)
        if 'temp_aforos_activos' not in st.session_state:
            st.session_state.temp_aforos_activos = df.copy()

        # 3. Buscar la columna de fecha para habilitar el filtro
        col_fecha = next((col for col in df.columns if "FECHA" in str(col).upper()), None)

        if col_fecha:
            # Asegurar formato fecha
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
                    with col_f1:
                        f_inicio = st.date_input("Fecha Inicio:", min_date, min_value=min_date, max_value=max_date, key="cal_inicio")
                    with col_f2:
                        f_fin = st.date_input("Fecha Fin:", max_date, min_value=min_date, max_value=max_date, key="cal_fin")

                # Aplicar filtro de fechas al DataFrame temporal (para visualización)
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

        # 4. Editor interactivo de Aforos (sobre el DataFrame temporal)
        columnas_protegidas = [col for col in df_mostrar_temp.columns if col != "Activo"]
        
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

        # Actualizar el DataFrame temporal con las ediciones (sin rerun aún)
        st.session_state.temp_aforos_activos.update(edited_temp)

        # 5. Botones de acción
        col_apply, col_cancel = st.columns(2)
        with col_apply:
            if st.button("✅ Aplicar cambios en aforos", use_container_width=True):
                # Actualizar el DataFrame global con las selecciones temporales
                st.session_state.df_aforos.update(st.session_state.temp_aforos_activos[['Activo']])
                # Recalcular aforos activos
                aforos_para_modelos = st.session_state.temp_aforos_activos[
                    st.session_state.temp_aforos_activos['Activo'] == True
                ].copy()
                st.session_state.df_aforos_activos = aforos_para_modelos
                st.success("Cambios aplicados a los aforos.")
                st.rerun()
        with col_cancel:
            if st.button("❌ Cancelar cambios", use_container_width=True):
                # Restaurar el temporal desde el global
                st.session_state.temp_aforos_activos = st.session_state.df_aforos.copy()
                st.rerun()

        # Resumen visual
        activos_visibles = len(st.session_state.temp_aforos_activos[st.session_state.temp_aforos_activos['Activo'] == True])
        totales_visibles = len(st.session_state.temp_aforos_activos)
        totales_absolutos = len(st.session_state.df_aforos)
        
        st.info(f"📊 **Aforos listos para cálculo en este periodo:** {activos_visibles} (de {totales_visibles} en el rango | {totales_absolutos} en total histórico).")

    else:
        st.info("Sube el archivo de Consolidado de Aforos en el panel lateral.")

    st.markdown("---")
    st.subheader("Puntos del Perfil Transversal")

    if st.session_state.perfil_data is not None:
        p_data = st.session_state.perfil_data
        # Crear DataFrame con los puntos del perfil
        df_perfil_puntos = pd.DataFrame({
            'Abscisa (m)': p_data['abscisas'],
            'Cota (m)': p_data['cotas'],
            'Descripción': p_data['descripciones'],
            'Tabla': p_data['tablas']
        })
        
        # Inicializar estado de selección temporal
        if 'temp_perfil_activos' not in st.session_state:
            st.session_state.temp_perfil_activos = [True] * len(df_perfil_puntos)

        # Mostrar editor con checkboxes (usando el temporal)
        df_perfil_edit = df_perfil_puntos.copy()
        df_perfil_edit['Activo'] = st.session_state.temp_perfil_activos

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

        # Actualizar temporal
        st.session_state.temp_perfil_activos = edited_perfil['Activo'].tolist()

        # Botones para perfil
        col_apply_perf, col_cancel_perf = st.columns(2)
        with col_apply_perf:
            if st.button("✅ Aplicar cambios en perfil", use_container_width=True):
                st.session_state.perfil_puntos_activos = st.session_state.temp_perfil_activos.copy()
                st.success("Cambios aplicados al perfil.")
                st.rerun()
        with col_cancel_perf:
            if st.button("❌ Cancelar cambios en perfil", use_container_width=True):
                # Restaurar temporal desde el global
                st.session_state.temp_perfil_activos = st.session_state.perfil_puntos_activos.copy()
                st.rerun()

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
            st.info(f"Usando {np.sum(mask)} de {len(mask)} puntos del perfil (según selección en 'Resumen Aforos').")
        else:
            abscisas_filt = p_data['abscisas']
            cotas_filt = p_data['cotas']
            st.warning("No hay selección de puntos activos. Usando todos los puntos del perfil.")
        
        # Verificar que haya puntos activos
        if len(abscisas_filt) == 0:
            st.error("No hay puntos activos en el perfil. Activa al menos un punto en la pestaña 'Resumen Aforos'.")
            st.stop()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Estación", f"{p_data['estacion']} ({p_data['codigo']})")
        col2.metric("Fecha Perfil", p_data['fecha'])
        col3.metric("Cota Cero", f"{p_data['cota_cero']:.3f} m")
        
        # --- Función para crear el gráfico del perfil ---
        def crear_grafico_perfil(abscisas, cotas, cota_cero, nivel_max=None):
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
            fig.update_yaxes(range=[y_min, y_max])
            return fig
        
        # Mostrar gráfico inicial con puntos filtrados
        fig_perfil = crear_grafico_perfil(abscisas_filt, cotas_filt, p_data['cota_cero'], nivel_max=None)
        placeholder_perfil = st.plotly_chart(fig_perfil, use_container_width=True, key="perfil_plot")
        
        st.markdown("---")
        
        # --- Cálculo de niveles mínimos y máximos (solo una vez) ---
        cota_min_lecho = float(cotas_filt.min())
        H_min_auto = float(max(0, cota_min_lecho - p_data['cota_cero']))
        
        # Parámetro: porcentaje de puntos a considerar en cada borde (puedes cambiarlo a un input si deseas)
        porcentaje_borde = 10  # 10% de los puntos en cada extremo
        n_puntos = len(abscisas_filt)
        n_borde = max(1, int(n_puntos * porcentaje_borde / 100))

        # Ordenar puntos por abscisa (por si acaso no lo están)
        indices_orden = np.argsort(abscisas_filt)
        abscisas_ordenadas = abscisas_filt[indices_orden]
        cotas_ordenadas = cotas_filt[indices_orden]

        # Lado izquierdo: primeros n_borde puntos
        cotas_izq = cotas_ordenadas[:n_borde]
        cota_izq_max = float(np.max(cotas_izq))

        # Lado derecho: últimos n_borde puntos
        cotas_der = cotas_ordenadas[-n_borde:]
        cota_der_max = float(np.max(cotas_der))

        cota_desborde = float(min(cota_izq_max, cota_der_max))
        H_max_auto = float(cota_desborde - p_data['cota_cero'])
        
        # Validar que los valores sean finitos
        if not (np.isfinite(H_min_auto) and np.isfinite(H_max_auto)):
            st.error("No se pudieron calcular niveles válidos. Verifica los puntos activos del perfil.")
            st.stop()
        
        st.info(f"**Nivel mínimo (lecho):** {H_min_auto:.2f} m | **Nivel máximo por desbordamiento:** {H_max_auto:.2f} m\n\n"
                f"Cota máxima en el {porcentaje_borde}% izquierdo: {cota_izq_max:.2f} m, derecho: {cota_der_max:.2f} m")

        # --- Opciones de rango ---
        usar_auto = st.checkbox("Usar nivel máximo automático (desbordamiento)", value=True)
        if usar_auto:
            H_max = H_max_auto
            st.caption(f"Se usará H_max = {H_max:.2f} m")
        else:
            # Calcular valores asegurando que sean finitos
            min_val = H_min_auto
            # Calcular cota máxima topográfica
            cota_max_top = float(np.nanmax(cotas_filt))
            max_val_calc = float(cota_max_top - p_data['cota_cero'])
            
            # Si max_val_calc no es finito, usar H_max_auto como límite
            if not np.isfinite(max_val_calc):
                max_val = H_max_auto
                st.warning("El valor máximo calculado no es finito. Se usará el nivel de desbordamiento como límite.")
            else:
                max_val = max_val_calc
            
            # Asegurar que default_val esté dentro del rango y sea finito
            default_val = H_max_auto
            if default_val < min_val:
                default_val = min_val
            elif default_val > max_val:
                default_val = max_val
            
            # Convertir todo a float de Python estándar
            min_val = float(min_val)
            max_val = float(max_val)
            default_val = float(default_val)
            step_val = 0.1
            
            H_max_manual = st.number_input(
                "Nivel máximo deseado (m):",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=step_val,
                format="%.2f"
            )
            H_max = H_max_manual
            if H_max > H_max_auto:
                st.warning(f"El nivel ingresado ({H_max:.2f} m) supera el nivel de desbordamiento. La sección ya no confinaría el flujo.")

        # --- Opciones de paso ---
        tipo_paso = st.radio(
            "Tipo de paso:",
            options=["Fijo", "Progresivo"],
            index=0,
            horizontal=True,
            key="tipo_paso"
        )

        if tipo_paso == "Fijo":
            paso_h = st.number_input("Paso de Nivel (H) en metros:", value=0.2, min_value=0.05, step=0.05, key="paso_fijo")
        else:
            col_paso1, col_paso2 = st.columns(2)
            with col_paso1:
                paso_fino = st.number_input("Paso fino (dentro de aforos):", value=0.1, min_value=0.01, step=0.05, key="paso_fino")
            with col_paso2:
                paso_grueso = st.number_input("Paso grueso (fuera de aforos):", value=0.5, min_value=0.1, step=0.1, key="paso_grueso")
            
            # Obtener rango de aforos activos (si existen)
            if 'df_aforos_activos' in st.session_state and st.session_state.df_aforos_activos is not None and not st.session_state.df_aforos_activos.empty:
                H_aforos = st.session_state.df_aforos_activos["H_m"].values
                H_min_aforos = H_aforos.min()
                H_max_aforos = H_aforos.max()
                st.caption(f"Rango de aforos activos: {H_min_aforos:.2f} - {H_max_aforos:.2f} m")
            else:
                H_min_aforos = H_min_auto
                H_max_aforos = H_max_auto
                st.warning("No hay aforos activos, se usará el rango completo con paso fino.")
        
        if st.button("Generar Tabla de Geometría"):
            def generar_rango_redondeado(inicio, fin, paso):
                """
                Genera un array de valores asegurando que sean múltiplos exactos del paso
                y que el último valor (fin) siempre esté incluido.
                """
                if inicio >= fin:
                    return np.array([fin])
                
                # Encontrar el primer múltiplo exacto del paso que sea >= inicio
                primer_multiplo = np.ceil(inicio / paso) * paso
                
                # Usar linspace en lugar de arange para evitar errores de precisión decimal
                num_pasos = int(np.floor(round((fin - primer_multiplo) / paso, 5)))
                
                if num_pasos < 0:
                    return np.array([fin])
                    
                # Generar secuencia
                secuencia = [primer_multiplo + i * paso for i in range(num_pasos + 1)]
                
                # Asegurar que el último valor 'fin' esté en el array
                if abs(secuencia[-1] - fin) > 1e-5:
                    secuencia.append(fin)
                    
                return np.round(secuencia, 3)

            if tipo_paso == "Fijo":
                # Cálculo con paso fijo usando la función robusta
                H_vals = generar_rango_redondeado(H_min_auto, H_max, paso_h)
                
            else:
                # Paso progresivo
                H_min_aforos = max(H_min_auto, H_min_aforos)
                H_max_aforos = min(H_max, H_max_aforos)
                
                # --- Segmento inferior (desde H_min_auto hasta H_min_aforos) ---
                H_inf = generar_rango_redondeado(H_min_auto, H_min_aforos, paso_grueso)
                
                # --- Segmento medio (aforos) con paso fino ---
                H_med = generar_rango_redondeado(H_min_aforos, H_max_aforos, paso_fino)
                
                # --- Segmento superior (desde H_max_aforos hasta H_max) ---
                H_sup = generar_rango_redondeado(H_max_aforos, H_max, paso_grueso)
                
                # Concatenar y eliminar duplicados manteniendo el orden
                H_vals = np.concatenate([H_inf, H_med, H_sup])
                H_vals = np.unique(np.round(H_vals, 3))
            
            # Asegurarse de que no haya valores mayores que H_max debido a redondeos
            H_vals = H_vals[H_vals <= H_max]

            if len(H_vals) == 0:
                st.error("No hay niveles en el rango seleccionado. Ajusta el paso o el nivel máximo.")
            else:
                data_geo = []
                for H in H_vals:
                    Am = area_mojada(abscisas_filt, cotas_filt, p_data['cota_cero'], H)
                    if Am <= 0: continue 
                    Wh = ancho_superficial(abscisas_filt, cotas_filt, p_data['cota_cero'], H)
                    Pm = perimetro_mojado(abscisas_filt, cotas_filt, p_data['cota_cero'], H)
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
                
                # Actualizar gráfico con nivel máximo
                fig_perfil_actualizado = crear_grafico_perfil(abscisas_filt, cotas_filt, p_data['cota_cero'], nivel_max=nivel_max_generado)
                placeholder_perfil.plotly_chart(fig_perfil_actualizado, use_container_width=True)
        
        if 'df_geo' in st.session_state and st.session_state.df_geo is not None:
            # --- INDICADOR DE RELACIÓN R/D (para validar método Stevens) ---
            df_geo = st.session_state.df_geo
            if "R/D" in df_geo.columns:
                valores_rd = df_geo["R/D"].dropna()
                if len(valores_rd) > 0:
                    min_rd = valores_rd.min()
                    max_rd = valores_rd.max()
                    mean_rd = valores_rd.mean()
                    
                    # Crear un contenedor de métricas
                    col_rd1, col_rd2, col_rd3, col_rd4 = st.columns(4)
                    with col_rd1:
                        st.metric("Relación R/D mín.", f"{min_rd:.3f}")
                    with col_rd2:
                        st.metric("Relación R/D máx.", f"{max_rd:.3f}")
                    with col_rd3:
                        st.metric("Relación R/D prom.", f"{mean_rd:.3f}")
                    
                    # Advertencia si la relación se aleja de 1
                    if min_rd < 0.8 or max_rd > 1.2:
                        st.warning("⚠️ La relación R/D se aleja significativamente de 1. El método Stevens (que asume R ≈ D) podría no ser adecuado para esta sección.")
                    elif min_rd < 0.9 or max_rd > 1.1:
                        st.info("ℹ️ La relación R/D está moderadamente cerca de 1. El método Stevens puede ser aceptable, pero verifique los resultados.")
                    else:
                        st.success("✅ La relación R/D es cercana a 1 en todo el rango. El método Stevens es aplicable.")
            
            # Mostrar la tabla de geometría (ahora con la columna R/D)
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
        
        if st.session_state.manning_data is None:
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

        fuente_k = st.radio(
            "Fuente de K:",
            options=["K de aforos", "K del perfil"],
            index=0,
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
        col_btn, _ = st.columns([0.1, 0.9])
        with col_btn:
            with st.popover("⚙️ Controles"):
                st.caption("Filtro de Aforos")
                with st.form(key="manning_form"):
                    edited = st.data_editor(
                        st.session_state.manning_edited_df,
                        column_config={
                            "Incluir": st.column_config.CheckboxColumn("Activo", default=True, width="small"),
                            "ID": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                            "H": st.column_config.NumberColumn("H", disabled=True, format="%.2f", width="small"),
                            "Q": st.column_config.NumberColumn("Q", disabled=True, format="%.2f", width="small"),
                            "V_af": None, "RH": None, "R23_af": None, "K_af": None, "K_per": None, "K": None
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
                    banda_input = st.number_input("Banda de Validación (%)", 
                                                  min_value=1.0, max_value=50.0, 
                                                  value=float(st.session_state.banda_error_global), 
                                                  step=1.0)

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
                        st.session_state.banda_error_global = banda_input
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

        # --- Separar activos e inactivos ---
        activos = edited_manning[edited_manning["Incluir"] == True].copy()
        inactivos = edited_manning[edited_manning["Incluir"] == False].copy()

        if fuente_k == "K de aforos":
            activos["K"] = activos["K_af"]
            inactivos["K"] = inactivos["K_af"]
        else:
            activos["K"] = activos["K_per"]
            inactivos["K"] = inactivos["K_per"]

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
            col_metodo, col_modelos = st.columns([1, 2])
            with col_metodo:
                metodo_seleccionado = st.selectbox(
                    "Método Matemático:",
                    ["Automático (Menor MAPE)", "Potencial", "Logarítmica", "Exponencial", "Lineal"],
                    key="metodo_select_manning"
                )
            with col_modelos:
                st.subheader("Comparación de modelos")
                df_modelos = pd.DataFrame([
                    {
                        "Modelo": k, 
                        "r": f"{v['r']:.4f}", 
                        "R²": f"{v['R2']:.4f}", 
                        "MAPE (%)": f"{v['MAPE']:.2f}" if pd.notna(v['MAPE']) else "N/A",
                        "σq (%)": f"{v['Sigma_q']:.2f}" if pd.notna(v['Sigma_q']) else "N/A",
                        "Ecuación": v['eq']
                    }
                    for k, v in ajustes.items()
                ])
                # Ordenar por MAPE (menor primero)
                df_modelos['MAPE_num'] = pd.to_numeric(df_modelos['MAPE (%)'], errors='coerce')
                df_modelos = df_modelos.sort_values('MAPE_num', na_position='last').drop(columns=['MAPE_num'])
                st.dataframe(df_modelos, use_container_width=True, hide_index=True)

            # Seleccionar modelo
            if metodo_seleccionado == "Automático (Menor MAPE)":
                mejor_modelo = min(
                    (k for k in ajustes if pd.notna(ajustes[k]["MAPE"])),
                    key=lambda k: ajustes[k]["MAPE"],
                    default=list(ajustes.keys())[0]
                )
            else:
                mejor_modelo = metodo_seleccionado
            
            funcion_optima = ajustes[mejor_modelo]["func"]
            st.info(f"Modelo seleccionado: **{mejor_modelo}** con MAPE = {ajustes[mejor_modelo]['MAPE']:.2f}% (R² = {ajustes[mejor_modelo]['R2']:.4f})")

            # --- Aplicar a geometría (extrapolación) ---
            df_g = st.session_state.df_geo.copy()
            H_min_geo = df_g["H (m)"].min()
            H_max_geo = df_g["H (m)"].max()
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
            if 'Lineal' in ajustes:
                funcs_adic['lineal'] = (ajustes['Lineal']['func'], mostrar_lineal)
            if 'Exponencial' in ajustes:
                funcs_adic['exp'] = (ajustes['Exponencial']['func'], mostrar_exp)
            if 'Logarítmica' in ajustes:
                funcs_adic['log'] = (ajustes['Logarítmica']['func'], mostrar_log)
            if 'Potencial' in ajustes:
                funcs_adic['pot'] = (ajustes['Potencial']['func'], mostrar_pot)

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

            # --- Gráfica Curva de Gasto con banda global ---
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
            placeholder_q.plotly_chart(fig_q, use_container_width=True)

            # --- Calcular errores individuales usando la función pura del modelo ---
            errores_data = []
            for _, row in activos.iterrows():
                h_val = row["H"]
                q_obs = row["Q"]
                k_est = funcion_optima(h_val)
                r23_val = float(interp_R23(h_val))
                am_val = float(interp_Am(h_val))
                q_est = k_est * r23_val * am_val
                err_pct = abs(q_est - q_obs) / q_obs * 100 if q_obs != 0 else np.nan
                errores_data.append({
                    "H (m)": h_val, 
                    "Q Estimado (m³/s)": q_est,
                    "Q Aforado (m³/s)": q_obs, 
                    "Error %": err_pct
                })

            # --- Obtener errores del mejor modelo ---
            prom_mape = ajustes[mejor_modelo]["MAPE"]
            prom_sigma = ajustes[mejor_modelo]["Sigma_q"]

            # --- Fila de guardado y métricas ---
            col_guardar, col_mape, col_sigma = st.columns([1, 1, 1])
            with col_guardar:
                if st.button("💾 Guardar curva", key="guardar_manning"):
                    if len(H_fino) > 0 and len(Q_suave) > 0 and np.isfinite(Q_suave).any():
                        st.session_state.manning_curve = pd.DataFrame({"H": H_fino, "Q": Q_suave})
                        st.session_state.manning_error = prom_mape
                        st.session_state.manning_error_sigma = prom_sigma
                        st.success("Curva guardada")
                    else:
                        st.error("No hay curva válida")
            with col_mape:
                st.metric("MAPE Promedio", f"{prom_mape:.1f}%" if pd.notna(prom_mape) else "N/A")
            with col_sigma:
                st.metric("Error Procedimiento (σq)", f"{prom_sigma:.1f}%" if pd.notna(prom_sigma) else "N/A")

            # --- Tabla de errores ---
            st.markdown("---")
            st.subheader("Errores en aforos activos")
            df_errores = pd.DataFrame(errores_data)
            st.dataframe(df_errores.style.format({"H (m)": "{:.2f}", "Q Estimado (m³/s)": "{:.2f}",
                                                   "Q Aforado (m³/s)": "{:.2f}", "Error %": "{:.1f}"}),
                         use_container_width=True, hide_index=True)

# ================== PESTAÑA 4: MÉTODO STEVENS ==================
with tab4:
    if st.session_state.df_aforos is None or st.session_state.df_geo is None:
        st.warning("Procesa los Aforos y genera la Tabla de Geometría primero.")
    else:
        # --- ENCABEZADO INFORMATIVO DEL MÉTODO ---
        with st.expander("📘 **Fundamento y consideraciones del método Stevens**", expanded=False):
            st.markdown("""
            **Fundamento:**  
            Este método es adecuado para ríos considerablemente anchos y poco profundos, donde la profundidad media (D) se puede considerar aproximadamente igual al radio hidráulico (R).  
            Esto ocurre cuando la relación ancho de la sección / perímetro mojado es próxima a 1, es decir, R ≈ D = A / W.

            **Ecuación base (Chezy):** Q = C √(R S).  
            Reemplazando R por D: Q = C √S · A √D.

            Para niveles altos, el término C√S tiende a ser constante.  
            Experimentalmente se ha encontrado que usar el exponente 2/3 en lugar de 1/2 proporciona mejores resultados, por lo que la variable independiente utilizada es **A·D^(2/3)**.

            En esta herramienta se define K = Q / (A·D^(2/3)), se ajusta K en función del nivel H y luego se extrapola usando la geometría calculada.

            > *Limitación:* La aproximación R ≈ D es válida solo en secciones muy anchas. Si el río no cumple esta condición, el método puede introducir errores sistemáticos.
            """)

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

        # --- Inicialización de datos (igual) ---
        if 'stevens_data' not in st.session_state:
            st.session_state.stevens_data = None

        if st.session_state.stevens_data is None:
            # 1. RECIBIR AFOROS FILTRADOS (Ya vienen limpios de la Pestaña 1)
            if 'df_aforos_activos' in st.session_state:
                df_a = st.session_state.df_aforos_activos.copy()
            else:
                df_a = st.session_state.df_aforos.copy()
            
            p_data = st.session_state.perfil_data
            df_g = st.session_state.df_geo

            col_area_af = buscar_columna(df_a.columns, ["AREA", "SEC"])
            col_d_af = buscar_columna(df_a.columns, ["PROF", "MEDIA"])
            
            if not col_area_af: st.sidebar.error("No se encontró la columna de Área de Sección en Aforos.")
            if not col_d_af: st.sidebar.error("No se encontró la columna de Profundidad Media en Aforos.")
            
            datos_stevens = []
            for i, row in df_a.iterrows():
                id_aforo = row.get("NO.", i+1)
                H = row["H_m"]
                Q = row["CAUDAL TOTAL (m3/s)"]
                
                A_af = row.get(col_area_af, np.nan)
                D_af = row.get(col_d_af, np.nan)
                
                X_af = np.nan
                if pd.notna(A_af) and pd.notna(D_af) and A_af > 0 and D_af > 0:
                    X_af = A_af * (D_af ** (2/3))
                
                K_af = 0
                if pd.notna(X_af) and X_af > 0:
                    K_af = Q / X_af
                else:
                    st.sidebar.warning(f"Aforo {id_aforo}: faltan A_af o D_af, K_af = 0")
                
                idx_cercano = np.argmin(np.abs(df_g["H (m)"] - H))
                H_cercano = df_g.iloc[idx_cercano]["H (m)"]
                if abs(H_cercano - H) > 0.2:
                    st.sidebar.warning(f"Aforo {id_aforo}: H={H:.2f} no cercano a geometría (H_geo={H_cercano:.2f})")
                
                A_per = df_g.iloc[idx_cercano]["Am (m2)"]
                D_per = df_g.iloc[idx_cercano]["D (m)"]
                
                X_per = A_per * (D_per ** (2/3)) if A_per > 0 and D_per > 0 else 0
                K_per = Q / X_per if X_per > 0 else 0
                
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
                        "A_per": A_per,
                        "D_per": D_per,
                        "X_per": X_per,
                        "K_per": K_per,
                    })
            
            st.session_state.stevens_data = pd.DataFrame(datos_stevens)

        fuente_k_stevens = st.radio(
            "Fuente de K:",
            options=["K de aforos", "K del perfil"],
            index=0,
            horizontal=True,
            key="fuente_k_stevens"
        )

        df_edit_stevens = st.session_state.stevens_data.copy()
        if fuente_k_stevens == "K de aforos":
            df_edit_stevens["K"] = df_edit_stevens["K_af"]
        else:
            df_edit_stevens["K"] = df_edit_stevens["K_per"]

        # --- Persistencia: DataFrames editados y opciones de modelos ---
        if 'stevens_edited_df' not in st.session_state:
            default_df = df_edit_stevens.copy()
            default_df['Incluir'] = True
            st.session_state.stevens_edited_df = default_df

        if 'opts_modelos_stevens' not in st.session_state:
            st.session_state.opts_modelos_stevens = {"lineal": True, "exp": True, "log": True, "pot": True}

        # --- Inicializar banda de error global (opcional, si se quiere usar) ---
        if 'banda_error_global' not in st.session_state:
            st.session_state.banda_error_global = 15.0

        # --- Botón con popover (esquina superior izquierda) ---
        col_btn, _ = st.columns([0.1, 0.9])
        with col_btn:
            with st.popover("⚙️ Controles"):
                with st.form(key="stevens_form"):
                    st.caption("Filtro de Aforos")
                    edited = st.data_editor(
                        st.session_state.stevens_edited_df,
                        column_config={
                            "Incluir": st.column_config.CheckboxColumn("Activo", default=True, width="small"),
                            "ID": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                            "H": st.column_config.NumberColumn("H", disabled=True, format="%.2f", width="small"),
                            "Q": st.column_config.NumberColumn("Q", disabled=True, format="%.2f", width="small"),
                            "A_af": None, "D_af": None, "X_af": None, "K_af": None,
                            "A_per": None, "D_per": None, "X_per": None, "K_per": None, "K": None
                        },
                        disabled=False,
                        hide_index=True,
                        use_container_width=True,
                        key="stevens_editor"
                    )

                    st.caption("Modelos de ajuste")
                    lineal = st.checkbox("Lineal", value=st.session_state.opts_modelos_stevens["lineal"])
                    exp = st.checkbox("Exponencial", value=st.session_state.opts_modelos_stevens["exp"])
                    log = st.checkbox("Logarítmica", value=st.session_state.opts_modelos_stevens["log"])
                    pot = st.checkbox("Potencial", value=st.session_state.opts_modelos_stevens["pot"])

                    # --- Input para la banda (si se desea) ---
                    st.caption("Configuración General")
                    banda_input = st.number_input("Banda de Validación (%)", 
                                                  min_value=1.0, max_value=50.0, 
                                                  value=float(st.session_state.banda_error_global), 
                                                  step=1.0)

                    col1, col2 = st.columns(2)
                    with col1:
                        submitted = st.form_submit_button("Aplicar")
                    with col2:
                        cancelled = st.form_submit_button("Cancelar")

                    if submitted:
                        st.session_state.stevens_edited_df = edited
                        st.session_state.opts_modelos_stevens["lineal"] = lineal
                        st.session_state.opts_modelos_stevens["exp"] = exp
                        st.session_state.opts_modelos_stevens["log"] = log
                        st.session_state.opts_modelos_stevens["pot"] = pot
                        st.session_state.banda_error_global = banda_input
                        st.rerun()
                    elif cancelled:
                        st.rerun()

        # --- Gráficas ---
        col_k_s, col_q_s = st.columns(2)
        with col_k_s:
            st.subheader("K vs Nivel (H) - Stevens")
            placeholder_k_s = st.empty()
        with col_q_s:
            st.subheader("Curva de Gasto - Stevens")
            placeholder_q_s = st.empty()

        # --- Leer estado ---
        edited_stevens = st.session_state.stevens_edited_df
        mostrar_lineal_s = st.session_state.opts_modelos_stevens["lineal"]
        mostrar_exp_s = st.session_state.opts_modelos_stevens["exp"]
        mostrar_log_s = st.session_state.opts_modelos_stevens["log"]
        mostrar_pot_s = st.session_state.opts_modelos_stevens["pot"]

        # --- Separar activos e inactivos ---
        activos_s = edited_stevens[edited_stevens["Incluir"] == True].copy()
        inactivos_s = edited_stevens[edited_stevens["Incluir"] == False].copy()

        if fuente_k_stevens == "K de aforos":
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
            
            # --- Ajustes de modelos (igual) ---
            ajustes_s = {}
            mask_log_s = (X_val_s > 0) & (Y_val_s > 0)
            X_log_s, Y_log_s = X_val_s[mask_log_s], Y_val_s[mask_log_s]

            # Lineal
            p_lin_s = np.polyfit(X_val_s, Y_val_s, 1)
            Y_lin_s = p_lin_s[0]*X_val_s + p_lin_s[1]
            r_lin_s = np.corrcoef(Y_val_s, Y_lin_s)[0, 1]
            r2_lin_s = r_lin_s**2
            ajustes_s["Lineal"] = {
                "r": r_lin_s,
                "R2": r2_lin_s,
                "func": lambda x: p_lin_s[0]*x + p_lin_s[1],
                "eq": f"K = {p_lin_s[0]:.3f}H + {p_lin_s[1]:.3f}"
            }

            if len(X_log_s) > 1:
                # Exponencial
                p_exp_s = np.polyfit(X_val_s, np.log(Y_val_s), 1)
                Y_exp_s = np.exp(p_exp_s[1]) * np.exp(p_exp_s[0]*X_val_s)
                r_exp_s = np.corrcoef(Y_val_s, Y_exp_s)[0, 1]
                r2_exp_s = r_exp_s**2
                ajustes_s["Exponencial"] = {
                    "r": r_exp_s,
                    "R2": r2_exp_s,
                    "func": lambda x: np.exp(p_exp_s[1]) * np.exp(p_exp_s[0]*x),
                    "eq": f"K = {np.exp(p_exp_s[1]):.3f} e^({p_exp_s[0]:.3f}H)"
                }

                # Logarítmica
                p_log_s = np.polyfit(np.log(X_log_s), Y_log_s, 1)
                Y_log_fit_s = p_log_s[0]*np.log(X_log_s) + p_log_s[1]
                r_log_s = np.corrcoef(Y_log_s, Y_log_fit_s)[0, 1]
                r2_log_s = r_log_s**2
                ajustes_s["Logarítmica"] = {
                    "r": r_log_s,
                    "R2": r2_log_s,
                    "func": lambda x: np.where(x > 0, p_log_s[0]*np.log(x) + p_log_s[1], np.nan),
                    "eq": f"K = {p_log_s[0]:.3f} ln(H) + {p_log_s[1]:.3f}"
                }

                # Potencial
                p_pot_s = np.polyfit(np.log(X_log_s), np.log(Y_log_s), 1)
                Y_pot_s = np.exp(p_pot_s[1]) * (X_log_s ** p_pot_s[0])
                r_pot_s = np.corrcoef(Y_log_s, Y_pot_s)[0, 1]
                r2_pot_s = r_pot_s**2
                ajustes_s["Potencial"] = {
                    "r": r_pot_s,
                    "R2": r2_pot_s,
                    "func": lambda x: np.where(x > 0, np.exp(p_pot_s[1]) * (x ** p_pot_s[0]), np.nan),
                    "eq": f"K = {np.exp(p_pot_s[1]):.3f} H^({p_pot_s[0]:.3f})"
                }

            # --- Interpoladores para calcular X = A * D^(2/3) exacto ---
            df_g = st.session_state.df_geo.copy()
            from scipy.interpolate import interp1d
            interp_A_s = interp1d(df_g["H (m)"], df_g["Am (m2)"], kind='linear', fill_value='extrapolate')
            interp_D_s = interp1d(df_g["H (m)"], df_g["D (m)"], kind='linear', fill_value='extrapolate')
            
            # X = A * D^(2/3)
            A_act_s = interp_A_s(X_val_s)
            D_act_s = interp_D_s(X_val_s)
            X_factor_act = A_act_s * (D_act_s ** (2/3))

            # Calcular MAPE y Sigma para cada modelo
            for k, v in ajustes_s.items():
                K_est_s = v["func"](X_val_s)
                Q_est_s = K_est_s * X_factor_act
                
                mask_err_s = (Q_act_s != 0) & np.isfinite(Q_est_s)
                if np.any(mask_err_s):
                    error_mape_s = np.mean(np.abs((Q_est_s[mask_err_s] - Q_act_s[mask_err_s]) / Q_act_s[mask_err_s])) * 100
                    error_sigma_s = calcular_error_procedimiento(Q_act_s[mask_err_s], Q_est_s[mask_err_s], K=2)
                else:
                    error_mape_s = np.nan
                    error_sigma_s = np.nan
                v["MAPE"] = error_mape_s
                v["Sigma_q"] = error_sigma_s

            # --- Selector de método y tabla de modelos ---
            col_metodo_s, col_modelos_s = st.columns([1, 2])
            with col_metodo_s:
                metodo_seleccionado_s = st.selectbox(
                    "Método Matemático:",
                    ["Automático (Menor MAPE)", "Potencial", "Logarítmica", "Exponencial", "Lineal"],
                    key="metodo_select_stevens"
                )
            with col_modelos_s:
                st.subheader("Comparación de modelos")
                # Crear DataFrame con todos los modelos y sus métricas
                df_modelos_s = pd.DataFrame([
                    {
                        "Modelo": k, 
                        "r": f"{v['r']:.4f}", 
                        "R²": f"{v['R2']:.4f}", 
                        "MAPE (%)": f"{v['MAPE']:.2f}" if pd.notna(v['MAPE']) else "N/A",
                        "σq (%)": f"{v['Sigma_q']:.2f}" if pd.notna(v['Sigma_q']) else "N/A",
                        "Ecuación": v['eq']
                    }
                    for k, v in ajustes_s.items()
                ])
                # Convertir MAPE a número para ordenar (los N/A se colocan al final)
                df_modelos_s['MAPE_num'] = pd.to_numeric(df_modelos_s['MAPE (%)'], errors='coerce')
                df_modelos_s = df_modelos_s.sort_values('MAPE_num', na_position='last').drop(columns=['MAPE_num'])
                
                st.dataframe(df_modelos_s, use_container_width=True, hide_index=True)

            # Seleccionar modelo
            if metodo_seleccionado_s == "Automático (Menor MAPE)":
                # Elegir el modelo con el MAPE más pequeño (ignorando NaN)
                mejor_modelo_s = min(
                    (k for k in ajustes_s if pd.notna(ajustes_s[k]["MAPE"])),
                    key=lambda k: ajustes_s[k]["MAPE"],
                    default=list(ajustes_s.keys())[0]  # fallback por si todos son NaN
                )
            else:
                mejor_modelo_s = metodo_seleccionado_s
            
            funcion_optima_s = ajustes_s[mejor_modelo_s]["func"]
            st.info(f"Modelo seleccionado: **{mejor_modelo_s}** con MAPE = {ajustes_s[mejor_modelo_s]['MAPE']:.2f}% (R² = {ajustes_s[mejor_modelo_s]['R2']:.4f})")

            # --- Aplicar a geometría (extrapolación) ---
            df_g = st.session_state.df_geo.copy()
            H_min_geo = df_g["H (m)"].min()
            H_max_geo = df_g["H (m)"].max()
            paso_fino = 0.2  # Se puede ajustar (originalmente 0.01, pero 0.2 es más rápido)
            H_fino_s = np.arange(H_min_geo, H_max_geo + paso_fino, paso_fino)
            
            K_fino_s = funcion_optima_s(H_fino_s)
            
            interp_A = interp1d(df_g["H (m)"], df_g["Am (m2)"], kind='linear', fill_value='extrapolate')
            interp_D = interp1d(df_g["H (m)"], df_g["D (m)"], kind='linear', fill_value='extrapolate')
            
            A_fino_s = interp_A(H_fino_s)
            D_fino_s = interp_D(H_fino_s)
            
            X_fino_s = A_fino_s * (D_fino_s ** (2/3))
            Q_fino_s = K_fino_s * X_fino_s

            # --- Filtrar valores no finitos y negativos ---
            mask_finite_s = np.isfinite(Q_fino_s)
            if not np.all(mask_finite_s):
                st.warning(f"Se encontraron {np.sum(~mask_finite_s)} valores no finitos en la curva Stevens. Se eliminarán.")
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

            # --- Preparar funciones adicionales para K vs H ---
            funcs_adic_s = {}
            if 'Lineal' in ajustes_s:
                funcs_adic_s['lineal'] = (ajustes_s['Lineal']['func'], mostrar_lineal_s)
            if 'Exponencial' in ajustes_s:
                funcs_adic_s['exp'] = (ajustes_s['Exponencial']['func'], mostrar_exp_s)
            if 'Logarítmica' in ajustes_s:
                funcs_adic_s['log'] = (ajustes_s['Logarítmica']['func'], mostrar_log_s)
            if 'Potencial' in ajustes_s:
                funcs_adic_s['pot'] = (ajustes_s['Potencial']['func'], mostrar_pot_s)

            # --- Crear figura K vs H ---
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

            # --- Crear figura Curva de Gasto (usando banda global) ---
            fig_q_s = crear_figura_curva(
                titulo=f"Curva de Gasto - usando {fuente_k_stevens}",
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
            placeholder_q_s.plotly_chart(fig_q_s, use_container_width=True)

            # --- Calcular errores para la tabla individual usando la función pura ---
            errores_s = []
            for _, row in activos_s.iterrows():
                h_val_s = row["H"]
                q_obs_s = row["Q"]
                
                k_est_s = funcion_optima_s(h_val_s)
                a_val_s = float(interp_A_s(h_val_s))
                d_val_s = float(interp_D_s(h_val_s))
                x_val_factor = a_val_s * (d_val_s ** (2/3))
                q_est_puro_s = k_est_s * x_val_factor
                
                err_pct_s = abs(q_est_puro_s - q_obs_s) / q_obs_s * 100 if q_obs_s != 0 else np.nan
                errores_s.append({
                    "H (m)": h_val_s, 
                    "Q Estimado (m³/s)": q_est_puro_s,
                    "Q Aforado (m³/s)": q_obs_s, 
                    "Error %": err_pct_s
                })

            # --- Obtener errores del mejor modelo ---
            prom_mape_s = ajustes_s[mejor_modelo_s]["MAPE"]
            prom_sigma_s = ajustes_s[mejor_modelo_s]["Sigma_q"]

            # --- Fila de guardado y métricas ---
            col_guardar_s, col_mape_s, col_sigma_s = st.columns([1, 1, 1])
            with col_guardar_s:
                if st.button("💾 Guardar curva", key="guardar_stevens"):
                    if len(H_fino_s) > 0 and len(Q_suave_s) > 0 and np.isfinite(Q_suave_s).any():
                        st.session_state.stevens_curve = pd.DataFrame({"H": H_fino_s, "Q": Q_suave_s})
                        st.session_state.stevens_error = prom_mape_s  # retrocompatibilidad
                        st.session_state.stevens_error_sigma = prom_sigma_s
                        st.success("Curva guardada")
                    else:
                        st.error("No hay curva válida")
            with col_mape_s:
                st.metric("MAPE Promedio", f"{prom_mape_s:.1f}%" if pd.notna(prom_mape_s) else "N/A")
            with col_sigma_s:
                st.metric("Error Procedimiento (σq)", f"{prom_sigma_s:.1f}%" if pd.notna(prom_sigma_s) else "N/A")

            # --- Tabla de errores ---
            st.markdown("---")
            st.subheader("Errores en aforos activos")
            df_err_s = pd.DataFrame(errores_s)
            st.dataframe(df_err_s.style.format({"H (m)": "{:.2f}", "Q Estimado (m³/s)": "{:.2f}",
                                                 "Q Aforado (m³/s)": "{:.2f}", "Error %": "{:.1f}"}),
                         use_container_width=True, hide_index=True)

# ================== PESTAÑA 5: MÉTODO ÁREA-VELOCIDAD ==================
with tab5:
    if st.session_state.df_aforos is None or st.session_state.df_geo is None:
        st.warning("Procesa los Aforos y genera la Tabla de Geometría primero.")
    else:

        # --- ENCABEZADO INFORMATIVO DEL MÉTODO ---
        with st.expander("📘 **Fundamento y consideraciones del método Área-Velocidad**", expanded=False):
            st.markdown("""
            **Fundamento:**  
            Este método se basa directamente en la ecuación de continuidad:  
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
        
        # --- Inicialización de datos (código completo) ---
        if 'av_data' not in st.session_state:
            st.session_state.av_data = None

        if st.session_state.av_data is None:
            # 1. RECIBIR AFOROS FILTRADOS (Ya vienen limpios de la Pestaña 1)
            if 'df_aforos_activos' in st.session_state:
                df_a = st.session_state.df_aforos_activos.copy()
            else:
                df_a = st.session_state.df_aforos.copy()
            
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

        fuente_v = st.radio(
            "Fuente de Velocidad:",
            options=["Velocidad de aforos", "Velocidad estimada del perfil"],
            index=0,
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
            default_df = df_edit_av.copy()
            default_df['Incluir'] = True
            st.session_state.av_edited_df = default_df

        if 'opts_modelos_av' not in st.session_state:
            st.session_state.opts_modelos_av = {"lineal": True, "exp": True, "log": True, "pot": True}

        # --- Inicializar banda de error global ---
        if 'banda_error_global' not in st.session_state:
            st.session_state.banda_error_global = 15.0 # Por defecto 15%

        # --- Botón con popover (con formulario) ---
        col_btn, _ = st.columns([0.1, 0.9])
        with col_btn:
            with st.popover("⚙️ Controles"):
                with st.form(key="av_form"):
                    st.caption("Filtro de Aforos")
                    edited = st.data_editor(
                        st.session_state.av_edited_df,
                        column_config={
                            "Incluir": st.column_config.CheckboxColumn("Activo", default=True, width="small"),
                            "ID": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                            "H": st.column_config.NumberColumn("H", disabled=True, format="%.2f", width="small"),
                            "Q": st.column_config.NumberColumn("Q", disabled=True, format="%.2f", width="small"),
                            "V_af": None, "V_per": None, "V": None
                        },
                        disabled=False,
                        hide_index=True,
                        use_container_width=True,
                        key="av_editor"
                    )

                    st.caption("Modelos de ajuste")
                    lineal = st.checkbox("Lineal", value=st.session_state.opts_modelos_av["lineal"])
                    exp = st.checkbox("Exponencial", value=st.session_state.opts_modelos_av["exp"])
                    log = st.checkbox("Logarítmica", value=st.session_state.opts_modelos_av["log"])
                    pot = st.checkbox("Potencial", value=st.session_state.opts_modelos_av["pot"])

                    # --- Input para la banda ---
                    st.caption("Configuración General (Aplica a todos los métodos)")
                    banda_input = st.number_input("Banda de Validación (%)", 
                                                  min_value=1.0, max_value=50.0, 
                                                  value=float(st.session_state.banda_error_global), 
                                                  step=1.0)

                    col1, col2 = st.columns(2)
                    with col1:
                        submitted = st.form_submit_button("Aplicar")
                    with col2:
                        cancelled = st.form_submit_button("Cancelar")

                    if submitted:
                        st.session_state.av_edited_df = edited
                        st.session_state.opts_modelos_av["lineal"] = lineal
                        st.session_state.opts_modelos_av["exp"] = exp
                        st.session_state.opts_modelos_av["log"] = log
                        st.session_state.opts_modelos_av["pot"] = pot
                        # --- Guardar banda ---
                        st.session_state.banda_error_global = banda_input
                        st.rerun()
                    elif cancelled:
                        st.rerun()

        # --- Gráficas ---
        col_v, col_q_av = st.columns(2)
        with col_v:
            st.subheader("Velocidad vs Nivel (H)")
            placeholder_v = st.empty()
        with col_q_av:
            st.subheader("Curva de Gasto")
            placeholder_q_av = st.empty()

        # --- Leer estado ---
        edited_av = st.session_state.av_edited_df
        mostrar_lineal_av = st.session_state.opts_modelos_av["lineal"]
        mostrar_exp_av = st.session_state.opts_modelos_av["exp"]
        mostrar_log_av = st.session_state.opts_modelos_av["log"]
        mostrar_pot_av = st.session_state.opts_modelos_av["pot"]

        # --- Separar activos e inactivos ---
        activos_av = edited_av[edited_av["Incluir"] == True].copy()
        inactivos_av = edited_av[edited_av["Incluir"] == False].copy()

        if fuente_v == "Velocidad de aforos (V_af)":
            activos_av["V"] = activos_av["V_af"]
            inactivos_av["V"] = inactivos_av["V_af"]
        else:
            activos_av["V"] = activos_av["V_per"]
            inactivos_av["V"] = inactivos_av["V_per"]

        activos_av = activos_av[activos_av["V"] > 0].dropna(subset=["V"])
        inactivos_av = inactivos_av[inactivos_av["V"] > 0].dropna(subset=["V"])

        if len(activos_av) < 2:
            st.error("Necesitas al menos 2 puntos activos con velocidad válida para calcular la curva.")
        else:
            H_act_av = activos_av["H"].values
            V_act_av = activos_av["V"].values
            Q_act_av = activos_av["Q"].values
            
            X_val_av = np.array(H_act_av)
            Y_val_av = np.array(V_act_av)
            
            # --- Ajustes de modelos (para V vs H) ---
            ajustes_av = {}
            mask_log_av = (X_val_av > 0) & (Y_val_av > 0)
            X_log_av, Y_log_av = X_val_av[mask_log_av], Y_val_av[mask_log_av]

            # Lineal
            p_lin_av = np.polyfit(X_val_av, Y_val_av, 1)
            Y_lin_av = p_lin_av[0]*X_val_av + p_lin_av[1]
            r_lin_av = np.corrcoef(Y_val_av, Y_lin_av)[0, 1]
            r2_lin_av = r_lin_av**2
            ajustes_av["Lineal"] = {
                "r": r_lin_av,
                "R2": r2_lin_av,
                "func": lambda x: p_lin_av[0]*x + p_lin_av[1],
                "eq": f"V = {p_lin_av[0]:.3f}H + {p_lin_av[1]:.3f}"
            }

            if len(X_log_av) > 1:
                # Exponencial
                p_exp_av = np.polyfit(X_val_av, np.log(Y_val_av), 1)
                Y_exp_av = np.exp(p_exp_av[1]) * np.exp(p_exp_av[0]*X_val_av)
                r_exp_av = np.corrcoef(Y_val_av, Y_exp_av)[0, 1]
                r2_exp_av = r_exp_av**2
                ajustes_av["Exponencial"] = {
                    "r": r_exp_av,
                    "R2": r2_exp_av,
                    "func": lambda x: np.exp(p_exp_av[1]) * np.exp(p_exp_av[0]*x),
                    "eq": f"V = {np.exp(p_exp_av[1]):.3f} e^({p_exp_av[0]:.3f}H)"
                }

                # Logarítmica
                p_log_av = np.polyfit(np.log(X_log_av), Y_log_av, 1)
                Y_log_fit_av = p_log_av[0]*np.log(X_log_av) + p_log_av[1]
                r_log_av = np.corrcoef(Y_log_av, Y_log_fit_av)[0, 1]
                r2_log_av = r_log_av**2
                ajustes_av["Logarítmica"] = {
                    "r": r_log_av,
                    "R2": r2_log_av,
                    "func": lambda x: np.where(x > 0, p_log_av[0]*np.log(x) + p_log_av[1], np.nan),
                    "eq": f"V = {p_log_av[0]:.3f} ln(H) + {p_log_av[1]:.3f}"
                }

                # Potencial
                p_pot_av = np.polyfit(np.log(X_log_av), np.log(Y_log_av), 1)
                Y_pot_av = np.exp(p_pot_av[1]) * (X_log_av ** p_pot_av[0])
                r_pot_av = np.corrcoef(Y_log_av, Y_pot_av)[0, 1]
                r2_pot_av = r_pot_av**2
                ajustes_av["Potencial"] = {
                    "r": r_pot_av,
                    "R2": r2_pot_av,
                    "func": lambda x: np.where(x > 0, np.exp(p_pot_av[1]) * (x ** p_pot_av[0]), np.nan),
                    "eq": f"V = {np.exp(p_pot_av[1]):.3f} H^({p_pot_av[0]:.3f})"
                }

            # --- Interpoladores para calcular A exacto y luego el MAPE ---
            df_g = st.session_state.df_geo.copy()
            interp_A_av = interp1d(df_g["H (m)"], df_g["Am (m2)"], kind='linear', fill_value='extrapolate')
            
            A_act_av = interp_A_av(X_val_av)

            # Calcular MAPE y Error de Procedimiento para cada modelo
            for k, v in ajustes_av.items():
                V_est_av = v["func"](X_val_av)
                # Fórmula de Área-Velocidad: Q = V * A
                Q_est_av = V_est_av * A_act_av
                
                # 1. Calcular MAPE original
                mask_err_av = (Q_act_av != 0) & np.isfinite(Q_est_av)
                if np.any(mask_err_av):
                    error_mape_av = np.mean(np.abs((Q_est_av[mask_err_av] - Q_act_av[mask_err_av]) / Q_act_av[mask_err_av])) * 100
                else:
                    error_mape_av = np.nan
                v["MAPE"] = error_mape_av
                
                # 2. Calcular el NUEVO Error de Procedimiento (Sigma)
                # Usamos K=2 porque los modelos (Lineal, Exp, Log, Pot) tienen 2 parámetros (ej. a y b)
                error_sigma_av = calcular_error_procedimiento(Q_act_av, Q_est_av, K=2)
                v["Sigma_q"] = error_sigma_av

            # --- Selector de método y tabla de modelos ---
            col_metodo_av, col_modelos_av = st.columns([1, 2])
            with col_metodo_av:
                metodo_seleccionado_av = st.selectbox(
                    "Método Matemático:",
                    ["Automático (Menor MAPE)", "Potencial", "Logarítmica", "Exponencial", "Lineal"],
                    key="metodo_select_av"
                )
            with col_modelos_av:
                st.subheader("Comparación de modelos")
                # ACTUALIZADO: Añadimos ambas columnas de error a la tabla
                df_modelos_av = pd.DataFrame([
                    {
                        "Modelo": k, 
                        "r": f"{v['r']:.4f}", 
                        "R²": f"{v['R2']:.4f}", 
                        "MAPE (%)": f"{v['MAPE']:.2f}", 
                        "Error σq (%)": f"{v['Sigma_q']:.2f}" if pd.notna(v['Sigma_q']) else "N/A", 
                        "Ecuación": v['eq']
                    }
                    for k, v in ajustes_av.items()
                ])
                # Ordenar la tabla del menor error σq al mayor
                df_modelos_av['Error_num'] = pd.to_numeric(df_modelos_av['Error σq (%)'], errors='coerce')
                df_modelos_av = df_modelos_av.sort_values('Error_num').drop(columns=['Error_num'])
                
                st.dataframe(df_modelos_av, use_container_width=True, hide_index=True)

           # Seleccionar modelo
            if metodo_seleccionado_av == "Automático (Menor MAPE)":
                # Usamos min() porque queremos el error más bajo
                mejor_modelo_av = min(ajustes_av, key=lambda k: ajustes_av[k]["MAPE"])
            else:
                mejor_modelo_av = metodo_seleccionado_av
            
            funcion_optima_av = ajustes_av[mejor_modelo_av]["func"]
            st.info(f"Modelo seleccionado: **{mejor_modelo_av}** con MAPE = {ajustes_av[mejor_modelo_av]['MAPE']:.2f}% (R² = {ajustes_av[mejor_modelo_av]['R2']:.4f})")

            # --- Aplicar a geometría (extrapolación) ---
            df_g = st.session_state.df_geo.copy()
            H_min_geo = df_g["H (m)"].min()
            H_max_geo = df_g["H (m)"].max()
            paso_fino = 0.2
            H_fino_av = np.arange(H_min_geo, H_max_geo + paso_fino, paso_fino)

            interp_A = interp1d(df_g["H (m)"], df_g["Am (m2)"], kind='linear', fill_value='extrapolate')
            # --- Calcular Q_fino_av ---
            V_fino_av = funcion_optima_av(H_fino_av)
            A_fino_av = interp_A(H_fino_av)
            Q_fino_av = V_fino_av * A_fino_av

            # --- Filtrar valores no finitos y negativos ---
            mask_finite_av = np.isfinite(Q_fino_av)
            if not np.all(mask_finite_av):
                st.warning(f"Se encontraron {np.sum(~mask_finite_av)} valores no finitos en la curva Área-Velocidad. Se eliminarán.")
                H_fino_av = H_fino_av[mask_finite_av]
                Q_fino_av = Q_fino_av[mask_finite_av]

            # Asegurar que los caudales no sean negativos
            Q_fino_av = np.maximum(Q_fino_av, 0)

            # Suavizado con interpolación lineal
            from scipy.interpolate import interp1d
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

            # --- Preparar funciones adicionales para V vs H ---
            funcs_adic_av = {}
            if 'Lineal' in ajustes_av:
                funcs_adic_av['lineal'] = (ajustes_av['Lineal']['func'], mostrar_lineal_av)
            if 'Exponencial' in ajustes_av:
                funcs_adic_av['exp'] = (ajustes_av['Exponencial']['func'], mostrar_exp_av)
            if 'Logarítmica' in ajustes_av:
                funcs_adic_av['log'] = (ajustes_av['Logarítmica']['func'], mostrar_log_av)
            if 'Potencial' in ajustes_av:
                funcs_adic_av['pot'] = (ajustes_av['Potencial']['func'], mostrar_pot_av)

            # --- Crear figura V vs H ---
            H_smooth_av = np.linspace(min(X_val_av)*0.5, max(X_val_av)*1.5, 200)
            V_sel_av = funcion_optima_av(H_smooth_av)
            inactivos_v_av = inactivos_av[['H', 'V']].rename(columns={'V': 'Y'}) if not inactivos_av.empty else None

            fig_v = crear_figura_k(
                titulo="Velocidad vs Nivel - Área-Velocidad",
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
            placeholder_v.plotly_chart(fig_v, use_container_width=True)

            # --- Crear figura Curva de Gasto ---
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
            placeholder_q_av.plotly_chart(fig_q_av, use_container_width=True)

            # --- Calcular error promedio usando la función matemática pura de Área-Velocidad ---
            prom_err_av = ajustes_av[mejor_modelo_av]["MAPE"]
            
            errores_av = []
            for _, row in activos_av.iterrows():
                h_val_av = row["H"]
                q_obs_av = row["Q"]
                
                # Calculamos V
                v_est_av = funcion_optima_av(h_val_av)
                
                # Extraemos A exacto para este H
                a_val_av = float(interp_A_av(h_val_av))
                
                # Q simulado puro
                q_est_puro_av = v_est_av * a_val_av
                
                err_pct_av = abs(q_est_puro_av - q_obs_av) / q_obs_av * 100 if q_obs_av != 0 else np.nan
                
                errores_av.append({
                    "H (m)": h_val_av, 
                    "Q Estimado (m³/s)": q_est_puro_av,
                    "Q Aforado (m³/s)": q_obs_av, 
                    "Error %": err_pct_av
                })

            # --- Fila de guardado y error actual (justo después de gráficas) ---
            # --- Extraer errores del mejor modelo ---
            prom_mape_av = ajustes_av[mejor_modelo_av]["MAPE"]
            prom_sigma_av = ajustes_av[mejor_modelo_av]["Sigma_q"]
            
            # ... (Aquí dejas el código intacto donde calculas la lista 'errores_av' para la tabla individual) ...

            # --- Fila de guardado y métricas actuales ---
            col_guardar_av, col_mape_av, col_sigma_av = st.columns([1, 1, 1])
            with col_guardar_av:
                if st.button("💾 Guardar curva", key="guardar_av"):
                    if len(H_fino_av) > 0 and len(Q_suave_av) > 0 and np.isfinite(Q_suave_av).any():
                        st.session_state.av_curve = pd.DataFrame({"H": H_fino_av, "Q": Q_suave_av})
                        st.session_state.av_error = prom_mape_av # Guardamos MAPE para retrocompatibilidad
                        st.session_state.av_error_sigma = prom_sigma_av # Guardamos el nuevo error
                        st.success("Curva guardada")
                    else:
                        st.error("No hay curva válida")
            with col_mape_av:
                st.metric("MAPE Promedio", f"{prom_mape_av:.1f}%")
            with col_sigma_av:
                st.metric("Error Procedimiento (σq)", f"{prom_sigma_av:.1f}%" if pd.notna(prom_sigma_av) else "N/A")

            # --- Tabla de errores (sin métrica adicional) ---
            st.markdown("---")
            st.subheader("Errores en aforos activos")
            df_err_av = pd.DataFrame(errores_av)
            st.dataframe(df_err_av.style.format({"H (m)": "{:.2f}", "Q Estimado (m³/s)": "{:.2f}",
                                                  "Q Aforado (m³/s)": "{:.2f}", "Error %": "{:.1f}"}),
                         use_container_width=True, hide_index=True)

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
    
    # Para la tabla resumen
    resumen_h0 = []

    from scipy.interpolate import interp1d

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
                if hay_aforos:
                    H1_j = float(f_H_inversa(Q1_j))
                    H2_j = float(f_H_inversa(Q2_j))
                    H3_j = float(f_H_inversa(Q3_j))
                    
                    denom_j = (H1_j + H2_j - 2 * H3_j)
                    if denom_j != 0:
                        Ho_johnson = (H1_j * H2_j - H3_j**2) / denom_j
                    else:
                        Ho_johnson = np.nan

                    resultados_johnson[nombre] = {
                        "Q-1 (Mín Aforo)": Q1_j, "Q-2 (Máx Aforo)": Q2_j, "Q-3 (Intermedio)": Q3_j,
                        "H-1": H1_j, "H-2": H2_j, "H-3": H3_j,
                        "Ho": Ho_johnson
                    }
                else:
                    Ho_johnson = np.nan
                    
                # Guardar para el resumen
                resumen_h0.append({
                    "Método": nombre,
                    "H0 (Running)": f"{Ho_run:.3f} m",
                    "H0 (Johnson)": f"{Ho_johnson:.3f} m" if pd.notna(Ho_johnson) else "N/A",
                    "_val_run": Ho_run,
                    "_val_john": Ho_johnson
                })

    # --- RENDERIZADO VISUAL ---
    if resumen_h0:
        st.markdown("---")
        st.subheader("🏆 Comparativa de Resultados H0")
        
        # DataFrame Resumen
        df_resumen = pd.DataFrame(resumen_h0)
        df_mostrar = df_resumen[["Método", "H0 (Running)", "H0 (Johnson)"]]
        
        col_tabla, col_botones = st.columns([1.5, 1])
        
        with col_tabla:
            st.dataframe(df_mostrar, use_container_width=True, hide_index=True)
            
        with col_botones:
            st.write("**Selección Oficial de H0**")
            st.caption("Elige qué H0 se usará como cota base para cada método.")
            
            # Inicializar variables en session_state si no existen
            if 'h0_seleccionados' not in st.session_state:
                st.session_state.h0_seleccionados = {}
                
            for idx, row in df_resumen.iterrows():
                metodo = row["Método"]
                
                # Definir opciones disponibles
                opciones = ["Running"]
                if pd.notna(row["_val_john"]):
                    opciones.append("Johnson")
                    
                # Selector
                eleccion = st.radio(
                    f"Para {metodo}:", 
                    options=opciones,
                    horizontal=True,
                    key=f"radio_h0_{metodo}"
                )
                
                # Guardar en memoria el valor numérico elegido
                if eleccion == "Running":
                    st.session_state.h0_seleccionados[metodo] = row["_val_run"]
                else:
                    st.session_state.h0_seleccionados[metodo] = row["_val_john"]
                    
        # Confirmación de selección
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

    # Verificar que existan las curvas de los métodos
    if (st.session_state.get("manning_curve") is None or 
        st.session_state.get("stevens_curve") is None or 
        st.session_state.get("av_curve") is None):
        st.warning("Primero debes calcular las curvas en cada método (pestañas 3, 4 y 5) con al menos 2 puntos activos.")
    else:
        # Obtener niveles de comparación desde la geometría
        if st.session_state.get("df_geo") is None:
            st.error("No hay datos de geometría. Genera la tabla en la pestaña Geometría.")
        else:
            # 1. NIVELES UNIFORMES PARA LA TABLA
            H_niveles = np.sort(st.session_state.df_geo["H (m)"].unique())
            
            from scipy.interpolate import interp1d
            import numpy as np
            
            # Obtener los H0 seleccionados en la Pestaña 6
            h0_dict = st.session_state.get('h0_seleccionados', {})
            h0_man = h0_dict.get('Manning', None)
            h0_ste = h0_dict.get('Stevens', None)
            h0_av = h0_dict.get('Área-Velocidad', None)

            # Capturar aforos filtrados
            if 'df_aforos_activos' in st.session_state:
                df_aforos_comp = st.session_state.df_aforos_activos
            else:
                df_aforos_comp = st.session_state.df_aforos

            # --- FUNCIONES PARA PREPARAR CURVAS GRÁFICAS ---
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

            # --- FUNCIONES PARA LA TABLA ---
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

            df_comp = pd.DataFrame({
                "H (m)": H_niveles,
                "Q_man (m³/s)": calcular_q_tabla(f_man, H_niveles, h0_man),
                "Q_ste (m³/s)": calcular_q_tabla(f_ste, H_niveles, h0_ste),
                "Q_av (m³/s)": calcular_q_tabla(f_av, H_niveles, h0_av)
            })

            # --- MÉTRICAS ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Error Manning", f"{st.session_state.manning_error:.1f}%" if st.session_state.get("manning_error") else "N/A")
            with col2:
                st.metric("Error Stevens", f"{st.session_state.stevens_error:.1f}%" if st.session_state.get("stevens_error") else "N/A")
            with col3:
                st.metric("Error Área-Velocidad", f"{st.session_state.av_error:.1f}%" if st.session_state.get("av_error") else "N/A")

            # --- GRÁFICA COMPARATIVA ---
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
            
            fig_comp.update_layout(title="Curvas de Gasto Comparativas", xaxis_title="Caudal Q (m³/s)", yaxis_title="Nivel H (m)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_comp, use_container_width=True)

            # --- TABLA COMPARATIVA ---
            st.subheader("Tabla de Caudales por Nivel (Geometría Base)")
            
            cols_h0 = st.columns(3)
            cols_h0[0].info(f"**H0 Manning:** {h0_man:.3f} m" if pd.notna(h0_man) else "**H0 Manning:** N/A")
            cols_h0[1].warning(f"**H0 Stevens:** {h0_ste:.3f} m" if pd.notna(h0_ste) else "**H0 Stevens:** N/A")
            cols_h0[2].success(f"**H0 Á-V:** {h0_av:.3f} m" if pd.notna(h0_av) else "**H0 Á-V:** N/A")
            
            formato_caudal = lambda x: f"{x:.2f}" if pd.notna(x) else "-"
            
            st.dataframe(df_comp.style.format({
                "H (m)": "{:.3f}", "Q_man (m³/s)": formato_caudal, "Q_ste (m³/s)": formato_caudal, "Q_av (m³/s)": formato_caudal
            }), use_container_width=True, hide_index=True)

            # --- NUEVO: SELECCIÓN OFICIAL Y EXPORTACIÓN ---
            st.markdown("---")
            st.subheader("🏆 Selección de Curva Definitiva")
            
            col_sel, col_desc = st.columns([1.5, 1])
            with col_sel:
                metodo_definitivo = st.radio(
                    "Con base en el error visual y numérico, selecciona el método que conformará tu Curva de Gasto Oficial:",
                    options=["Manning", "Stevens", "Área-Velocidad"],
                    horizontal=True
                )
                # Guardar selección en sesión
                st.session_state.metodo_definitivo = metodo_definitivo
            
            with col_desc:
                # Mapeo de columnas según selección
                col_map = {
                    "Manning": "Q_man (m³/s)",
                    "Stevens": "Q_ste (m³/s)",
                    "Área-Velocidad": "Q_av (m³/s)"
                }
                
                # Preparar DataFrame final
                df_export = df_comp.copy()
                df_export.insert(1, "Q_Definitivo (m³/s)", df_export[col_map[metodo_definitivo]])
                
                # Redondear valores para un CSV limpio
                df_export = df_export.round({"H (m)": 3, "Q_Definitivo (m³/s)": 3, "Q_man (m³/s)": 3, "Q_ste (m³/s)": 3, "Q_av (m³/s)": 3})
                csv_export = df_export.to_csv(index=False).encode('utf-8')
                
                st.write("") # Espaciador
                st.download_button(
                    label=f"💾 Exportar Curva {metodo_definitivo} (CSV)",
                    data=csv_export,
                    file_name=f"Curva_Gasto_Definitiva_{metodo_definitivo}.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
            st.caption(f"El archivo exportado incluirá una columna maestra llamada **Q_Definitivo** basada en el método de {metodo_definitivo}, manteniendo las demás como referencia.")

            # Expander con tabla de aforos
            with st.expander("Ver aforos utilizados"):
                if df_aforos_comp is not None and not df_aforos_comp.empty:
                    cols_aforo = ["NO.", "FECHA", "H_m", "CAUDAL TOTAL (m3/s)", "ÁREA SEC. (m2)", "VELOC. MEDIA (m/s)"]
                    cols_existentes = [c for c in cols_aforo if c in df_aforos_comp.columns]
                    df_af_mostrar = df_aforos_comp[cols_existentes].copy()
                    df_af_mostrar = df_af_mostrar.rename(columns={"H_m": "H (m)", "CAUDAL TOTAL (m3/s)": "Q (m³/s)"})
                    st.dataframe(df_af_mostrar, use_container_width=True, hide_index=True)

# ================== PESTAÑA 8: HISTÓRICO DE CURVAS ==================
with tab8:
    st.header("Comparación con el Histórico de Curvas")
    st.markdown("Sube el archivo institucional con el histórico para comparar la nueva curva calculada.")

    # 1. Cargar el archivo Excel
    archivo_historico = st.file_uploader("Sube el archivo Excel (.xlsx, .xls)", type=["xlsx", "xls"], key="file_hist")
    
    if archivo_historico is not None:
        # Cargar usando la función en caché
        df_hist_full = cargar_historico_excel(archivo_historico)
        
        col_est, col_empty = st.columns([1, 2])
        with col_est:
            # --- NUEVO: Extraer código de la estación automáticamente ---
            codigo_default = ""
            if st.session_state.get('perfil_data') and st.session_state.perfil_data.get('codigo'):
                codigo_default = str(st.session_state.perfil_data['codigo']).strip()
                
            codigo_estacion = st.text_input("Etiqueta_Estacion a filtrar:", value=codigo_default)
        
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
                    st.error("El archivo no tiene la columna 'Nivel'. Verifica la estructura del Excel.")
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
                        factor_banda = 0.80 # 80% de amplitud
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
                            name='Banda ±80%',
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

                # Formato de la gráfica (Cambié hovermode a 'y unified' para que sea más fácil comparar a la misma altura)
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
                    
