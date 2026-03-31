# -*- coding: utf-8 -*-
"""
MÓDULO DE FUNCIONES GEOMÉTRICAS VECTORIZADAS
============================================
Contiene todas las funciones para cálculos hidráulicos de:
- Área mojada
- Perímetro mojado
- Ancho superficial
- Errores estadísticos (MAPE, error de procedimiento)

Optimizado con NumPy para máximo rendimiento.
"""

import numpy as np


def _preparar_vectores(abscisas, cotas, cota_cero, nivel):
    """
    Función auxiliar para calcular intersecciones y máscaras de forma vectorizada.
    
    Prepara datos para cálculos geométricos de sección transversal con un nivel 
    específico de agua. Convierte arrays en formato vectorizado para operaciones 
    eficientes con NumPy.
    
    Args:
        abscisas (np.ndarray): Coordenadas horizontales del perfil (m)
        cotas (np.ndarray): Elevaciones del lecho (m)
        cota_cero (float): Elevación del cero de la regla (m)
        nivel (float): Altura del agua sobre el cero (m)
    
    Returns:
        tuple or None: (x0, x1, y0, y1, dx, dy, h0, h1, frac, m_ambos, m_baja, m_sube)
                      where:
                      - x0, x1: abscisas de pares de puntos consecutivos
                      - y0, y1: cotas de cada par
                      - dx, dy: diferencias horizontales y verticales
                      - h0, h1: profundidades en cada punto
                      - frac: fracción donde agua toca talud (Tales theorem)
                      - m_*: máscaras booleanas para 3 casos geométricos
                      
                      None si no hay suficientes puntos válidos
    
    Notes:
        - Filtra automáticamente valores NaN
        - Maneja divisiones por cero en terrenos planos
        - Usa Tales theorem para intersecciones
    """
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
    m_baja = (h0 < 0) & (h1 > 0)     # Orilla izquierda
    m_sube = (h0 > 0) & (h1 < 0)     # Orilla derecha
    
    return x0, x1, y0, y1, dx, dy, h0, h1, frac, m_ambos, m_baja, m_sube


def area_mojada(abscisas, cotas, cota_cero, nivel, modo_muro="sin_friccion"):
    """
    Calcula el área mojada para un nivel de agua dado.
    
    Integra numéricamente usando trapezoides para la sección transversal
    bajo el nivel especificado.
    
    Args:
        abscisas (np.ndarray): Coordenadas horizontales (m)
        cotas (np.ndarray): Elevaciones del lecho (m)
        cota_cero (float): Elevación del cero de la regla (m)
        nivel (float): Altura del agua sobre el cero (m)
        modo_muro (str): "sin_friccion" o "con_friccion"
                         Modo de tratamiento de desbordamientos.
    
    Returns:
        float: Área mojada en m²
        
    Notes:
        - Uso: cantidad mínima de agua requerida
        - Fórmula: Integración trapezoidal en sección no regularizada
        - Devuelve 0 si menos de 2 puntos válidos
    """
    datos = _preparar_vectores(abscisas, cotas, cota_cero, nivel)
    if not datos: 
        return 0.0
    
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
    """
    Calcula el ancho superficial (espejo de agua) para un nivel dado.
    
    Proyecta el nivel horizontal sobre la topografía para obtener el ancho
    en contacto con la atmósfera.
    
    Args:
        abscisas (np.ndarray): Coordenadas horizontales (m)
        cotas (np.ndarray): Elevaciones del lecho (m)
        cota_cero (float): Elevación del cero de la regla (m)
        nivel (float): Altura del agua sobre el cero (m)
        modo_muro (str): No usado para ancho (devuelve siempre ancho máximo)
    
    Returns:
        float: Ancho superficial en m
        
    Notes:
        - Se congela en ancho máximo si desborda (actúa como muro)
        - Devuelve 0 si menos de 2 puntos válidos
    """
    datos = _preparar_vectores(abscisas, cotas, cota_cero, nivel)
    if not datos: 
        return 0.0
    
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
    """
    Calcula el perímetro mojado para un nivel de agua dado.
    
    Suma las longitudes de lecho en contacto con agua, usado en 
    cálculos de flujo según Manning.
    
    Args:
        abscisas (np.ndarray): Coordenadas horizontales (m)
        cotas (np.ndarray): Elevaciones del lecho (m)
        cota_cero (float): Elevación del cero de la regla (m)
        nivel (float): Altura del agua sobre el cero (m)
        modo_muro (str): "sin_friccion" → perímetro congelado en topografía máxima
                         "con_friccion" → suma muros verticales si desborda
    
    Returns:
        float: Perímetro mojado en m
        
    Notes:
        - Modo "sin_friccion": Ven Te Chow recommendation (mejor para extrapolación)
        - Modo "con_friccion": Penaliza desbordamientos laterales
        - Devuelve 0 si menos de 2 puntos válidos
    """
    datos = _preparar_vectores(abscisas, cotas, cota_cero, nivel)
    if not datos: 
        return 0.0
    
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
        if len(h0) > 0 and h0[0] > 0:
            p_total += h0[0]
            
        # h1[-1] es la altura del agua sobre la última abscisa. Si es > 0, se desbordó por la derecha.
        if len(h1) > 0 and h1[-1] > 0:
            p_total += h1[-1]
            
    # Si es "sin_friccion" o "ninguno", simplemente devolvemos p_total intacto 
    # (el perímetro queda congelado en la topografía máxima, justo como recomienda Ven Te Chow)
    return p_total


def calcular_mape(q_obs, q_calc):
    """
    Calcula el Error Porcentual Absoluto Medio (MAPE).
    
    Métrica de error relativo promedio usado para validar modelos
    de estimación de caudal.
    
    Args:
        q_obs (array-like): Caudales observados (m³/s)
        q_calc (array-like): Caudales calculados (m³/s)
    
    Returns:
        float: MAPE en porcentaje (%)
        
    Notes:
        Equivalente a: =PROMEDIO(ABS(Qcalc - Qobs)/Qobs) * 100
        Excelente para comparar modelos diferentes.
        Se ignoran observaciones con Q = 0.
    
    Formula:
        MAPE = mean(|Q_calc - Q_obs| / |Q_obs|) * 100
    """
    q_obs = np.array(q_obs)
    q_calc = np.array(q_calc)
    
    # Máscara de seguridad para evitar divisiones por cero
    mask = q_obs != 0 
    
    # Cálculo vectorizado del error absoluto porcentual
    error_porcentual = np.abs((q_calc[mask] - q_obs[mask]) / q_obs[mask])
    
    # Retorna el promedio multiplicado por 100 para tenerlo en %
    return float(np.mean(error_porcentual) * 100) if len(error_porcentual) > 0 else np.nan


def calcular_error_procedimiento(Q_obs, Q_est, K=2):
    """
    Calcula el error de procedimiento según la fórmula estadística.
    
    Usado para estimar incertidumbre de ajuste en modelos hidrodinámicos.
    Se basa en residuales normalizados.
    
    Args:
        Q_obs (array-like): Caudales aforados (observados) (m³/s)
        Q_est (array-like): Caudales estimados por modelo (m³/s)
        K (int): Grados de libertad (típicamente 2 para modelos: lineal, exp, log, potencial)
    
    Returns:
        float: Error de procedimiento en porcentaje (%)
                Returns np.nan si N ≤ K (muestras insuficientes)
    
    Notes:
        - Fórmula: error_sigma = sqrt(sum((Qo-Qe)²/Qe²) / (N-K))
        - Convierte a porcentaje multiplicando × 100
        - Rechaza muestras con Qe ≤ 0 o valores no finitos
        
    References:
        Base teórica en análisis residuales de regresión.
    """
    # Evitar divisiones por cero y valores no finitos
    mask = (Q_est != 0) & np.isfinite(Q_est) & (Q_obs != 0)
    Q_o = np.array(Q_obs)[mask]
    Q_e = np.array(Q_est)[mask]
    
    N = len(Q_o)
    
    if N <= K:
        return np.nan  # No hay suficientes puntos para calcular el error con esos grados de libertad
        
    # Aplicar la fórmula
    suma = np.sum(((Q_o - Q_e) / Q_e)**2)
    error_sigma = np.sqrt(suma / (N - K))
    
    return float(error_sigma * 100)  # Multiplicamos por 100 para mostrarlo en %
