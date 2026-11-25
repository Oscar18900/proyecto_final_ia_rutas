"""Módulo metaheurístico: GWO, PSO, TLBO y GA con claves aleatorias en [0, 1]."""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np

from .rutas import (
    calcular_costo_ruta,
    construir_ruta_completa,
    decodificar_claves_aleatorias,
)


PARAMETROS_METAHEURISTICAS: list[dict[str, Any]] = [
    {
        "Algoritmo": "Comunes",
        "Parámetro": "Población / Dimensión / Presupuesto / Semillas",
        "Valor": "30 individuos; D = número de centros (8); 3,000 evaluaciones de L(π); semillas 0–19",
        "Descripción": "Representación por claves aleatorias x ∈ [0, 1]^D decodificadas por orden ascendente.",
    },
    {
        "Algoritmo": "GWO",
        "Parámetro": "Líderes y parámetro a",
        "Valor": "α, β, δ (top-3); a lineal de 2.0 → 0.0; límites [0, 1]",
        "Descripción": "Actualización guiada por los tres mejores lobos con vectores aleatorios A y C.",
    },
    {
        "Algoritmo": "PSO",
        "Parámetro": "Inercia y aceleración",
        "Valor": "w: 0.9 → 0.4; c1 = 1.5; c2 = 1.5; v_max = 0.2; límites [0, 1]",
        "Descripción": "Velocidad acotada en [-v_max, v_max] con memoria personal (pbest) y global (gbest).",
    },
    {
        "Algoritmo": "TLBO",
        "Parámetro": "Factor de enseñanza y fases",
        "Valor": "TF ∈ {1, 2}; fase profesor + fase estudiante; límites [0, 1]",
        "Descripción": "Sin parámetros de ajuste específicos adicionales; aceptación codiciosa de mejoras.",
    },
    {
        "Algoritmo": "GA",
        "Parámetro": "Selección, cruce, mutación y élite",
        "Valor": "Torneo k=3; cruce aritmético pc=0.85; mutación gaussiana pm=0.2, σ=0.1; elitismo=2",
        "Descripción": "Preserva los 2 mejores sin re-evaluar y aplica recorte a [0, 1] tras mutación.",
    },
]


class EvaluadorRutas:
    """Evaluador central que contabiliza exactamente las evaluaciones de L(π)."""

    def __init__(
        self,
        matriz_distancias: Sequence[Sequence[float]],
        max_evaluaciones: int = 3000,
    ) -> None:
        self.matriz_distancias = matriz_distancias
        self.dim = len(matriz_distancias) - 1
        self.max_evaluaciones = max_evaluaciones
        self.evaluaciones = 0
        self.mejor_distancia = float("inf")
        self.mejor_permutacion: tuple[int, ...] = ()
        self.mejor_claves: np.ndarray | None = None
        self.historial = np.empty(max_evaluaciones, dtype=float)

    def puede_evaluar(self) -> bool:
        """Indica si aún queda presupuesto de evaluaciones."""
        return self.evaluaciones < self.max_evaluaciones

    def evaluar(self, claves: np.ndarray) -> float:
        """Decodifica las claves aleatorias, calcula L(π) y registra la convergencia."""
        if not self.puede_evaluar():
            raise RuntimeError(
                f"Se superó el presupuesto máximo de {self.max_evaluaciones} evaluaciones."
            )

        permutacion = decodificar_claves_aleatorias(claves)
        distancia = calcular_costo_ruta(
            permutacion, self.matriz_distancias, validar=False
        )

        if distancia < self.mejor_distancia:
            self.mejor_distancia = distancia
            self.mejor_permutacion = permutacion
            self.mejor_claves = claves.copy()

        self.historial[self.evaluaciones] = self.mejor_distancia
        self.evaluaciones += 1
        return distancia


def _inicializar_poblacion(
    rng: np.random.Generator,
    tam_poblacion: int,
    dim: int,
    evaluador: EvaluadorRutas,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera y evalúa la población inicial en [0, 1]^dim."""
    poblacion = rng.uniform(0.0, 1.0, size=(tam_poblacion, dim))
    aptitudes = np.full(tam_poblacion, float("inf"), dtype=float)
    for i in range(tam_poblacion):
        if not evaluador.puede_evaluar():
            break
        aptitudes[i] = evaluador.evaluar(poblacion[i])
    return poblacion, aptitudes


def _empaquetar_resultado(
    algoritmo: str,
    semilla: int,
    evaluador: EvaluadorRutas,
    tiempo_s: float,
) -> dict[str, Any]:
    """Empaqueta los resultados de una corrida metaheurística."""
    historial_usado = evaluador.historial[: evaluador.evaluaciones].copy()
    return {
        "algoritmo": algoritmo,
        "semilla": semilla,
        "mejor_distancia": float(evaluador.mejor_distancia),
        "mejor_permutacion": evaluador.mejor_permutacion,
        "mejor_ruta_completa": construir_ruta_completa(evaluador.mejor_permutacion),
        "evaluaciones": evaluador.evaluaciones,
        "tiempo_s": float(tiempo_s),
        "historial": historial_usado,
    }


def ejecutar_gwo(
    matriz_distancias: Sequence[Sequence[float]],
    semilla: int,
    tam_poblacion: int = 30,
    max_evaluaciones: int = 3000,
) -> dict[str, Any]:
    """Optimizador de Lobos Grises (GWO) sobre claves aleatorias en [0, 1]."""
    inicio_t = time.perf_counter()
    rng = np.random.default_rng(semilla)
    evaluador = EvaluadorRutas(matriz_distancias, max_evaluaciones)
    dim = evaluador.dim

    poblacion, aptitudes = _inicializar_poblacion(rng, tam_poblacion, dim, evaluador)

    # Identificar líderes alfa, beta y delta iniciales
    orden = np.argsort(aptitudes)
    alfa_pos = poblacion[orden[0]].copy()
    alfa_fit = float(aptitudes[orden[0]])
    beta_pos = poblacion[orden[1]].copy()
    beta_fit = float(aptitudes[orden[1]])
    delta_pos = poblacion[orden[2]].copy()
    delta_fit = float(aptitudes[orden[2]])

    max_iter = max(1, (max_evaluaciones - tam_poblacion) // tam_poblacion)
    iteracion = 0

    while evaluador.puede_evaluar():
        a = max(0.0, 2.0 * (1.0 - (iteracion / max_iter)))

        for i in range(tam_poblacion):
            if not evaluador.puede_evaluar():
                break

            r1 = rng.random(dim)
            r2 = rng.random(dim)
            a1 = 2.0 * a * r1 - a
            c1 = 2.0 * r2
            d_alfa = np.abs(c1 * alfa_pos - poblacion[i])
            x1 = alfa_pos - a1 * d_alfa

            r1 = rng.random(dim)
            r2 = rng.random(dim)
            a2 = 2.0 * a * r1 - a
            c2 = 2.0 * r2
            d_beta = np.abs(c2 * beta_pos - poblacion[i])
            x2 = beta_pos - a2 * d_beta

            r1 = rng.random(dim)
            r2 = rng.random(dim)
            a3 = 2.0 * a * r1 - a
            c3 = 2.0 * r2
            d_delta = np.abs(c3 * delta_pos - poblacion[i])
            x3 = delta_pos - a3 * d_delta

            nueva_pos = np.clip((x1 + x2 + x3) / 3.0, 0.0, 1.0)
            poblacion[i] = nueva_pos
            aptitudes[i] = evaluador.evaluar(nueva_pos)

            # Actualizar jerarquía alfa, beta, delta conforme mejora la manada
            fit = float(aptitudes[i])
            if fit < alfa_fit:
                delta_fit, delta_pos = beta_fit, beta_pos.copy()
                beta_fit, beta_pos = alfa_fit, alfa_pos.copy()
                alfa_fit, alfa_pos = fit, nueva_pos.copy()
            elif fit < beta_fit:
                delta_fit, delta_pos = beta_fit, beta_pos.copy()
                beta_fit, beta_pos = fit, nueva_pos.copy()
            elif fit < delta_fit:
                delta_fit, delta_pos = fit, nueva_pos.copy()

        iteracion += 1

    tiempo_s = time.perf_counter() - inicio_t
    return _empaquetar_resultado("GWO", semilla, evaluador, tiempo_s)


def ejecutar_pso(
    matriz_distancias: Sequence[Sequence[float]],
    semilla: int,
    tam_poblacion: int = 30,
    max_evaluaciones: int = 3000,
    w_max: float = 0.9,
    w_min: float = 0.4,
    c1: float = 1.5,
    c2: float = 1.5,
    v_max: float = 0.2,
) -> dict[str, Any]:
    """Optimización por Enjambre de Partículas (PSO) sobre claves aleatorias en [0, 1]."""
    inicio_t = time.perf_counter()
    rng = np.random.default_rng(semilla)
    evaluador = EvaluadorRutas(matriz_distancias, max_evaluaciones)
    dim = evaluador.dim

    posiciones, aptitudes = _inicializar_poblacion(rng, tam_poblacion, dim, evaluador)
    velocidades = rng.uniform(-v_max, v_max, size=(tam_poblacion, dim))

    pbest_pos = posiciones.copy()
    pbest_fit = aptitudes.copy()

    idx_gbest = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[idx_gbest].copy()
    gbest_fit = float(pbest_fit[idx_gbest])

    while evaluador.puede_evaluar():
        progreso = evaluador.evaluaciones / max_evaluaciones
        w = w_max - (w_max - w_min) * progreso

        for i in range(tam_poblacion):
            if not evaluador.puede_evaluar():
                break

            r1 = rng.random(dim)
            r2 = rng.random(dim)
            velocidades[i] = (
                w * velocidades[i]
                + c1 * r1 * (pbest_pos[i] - posiciones[i])
                + c2 * r2 * (gbest_pos - posiciones[i])
            )
            velocidades[i] = np.clip(velocidades[i], -v_max, v_max)
            posiciones[i] = np.clip(posiciones[i] + velocidades[i], 0.0, 1.0)

            fit = evaluador.evaluar(posiciones[i])
            aptitudes[i] = fit

            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_pos[i] = posiciones[i].copy()
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest_pos = posiciones[i].copy()

    tiempo_s = time.perf_counter() - inicio_t
    return _empaquetar_resultado("PSO", semilla, evaluador, tiempo_s)


def ejecutar_tlbo(
    matriz_distancias: Sequence[Sequence[float]],
    semilla: int,
    tam_poblacion: int = 30,
    max_evaluaciones: int = 3000,
) -> dict[str, Any]:
    """Optimización Basada en Enseñanza-Aprendizaje (TLBO) sobre claves aleatorias."""
    inicio_t = time.perf_counter()
    rng = np.random.default_rng(semilla)
    evaluador = EvaluadorRutas(matriz_distancias, max_evaluaciones)
    dim = evaluador.dim

    poblacion, aptitudes = _inicializar_poblacion(rng, tam_poblacion, dim, evaluador)

    while evaluador.puede_evaluar():
        # Fase del profesor
        idx_profesor = int(np.argmin(aptitudes))
        profesor = poblacion[idx_profesor].copy()
        media = np.mean(poblacion, axis=0)

        for i in range(tam_poblacion):
            if not evaluador.puede_evaluar():
                break

            tf = int(rng.integers(1, 3))  # Factor de enseñanza 1 o 2
            r = rng.random(dim)
            candidato = np.clip(poblacion[i] + r * (profesor - tf * media), 0.0, 1.0)
            fit_candidato = evaluador.evaluar(candidato)

            if fit_candidato <= aptitudes[i]:
                poblacion[i] = candidato
                aptitudes[i] = fit_candidato

        # Fase del estudiante
        for i in range(tam_poblacion):
            if not evaluador.puede_evaluar():
                break

            j = int(rng.integers(0, tam_poblacion - 1))
            if j >= i:
                j += 1

            r = rng.random(dim)
            if aptitudes[i] < aptitudes[j]:
                diferencia = poblacion[i] - poblacion[j]
            else:
                diferencia = poblacion[j] - poblacion[i]

            candidato = np.clip(poblacion[i] + r * diferencia, 0.0, 1.0)
            fit_candidato = evaluador.evaluar(candidato)

            if fit_candidato <= aptitudes[i]:
                poblacion[i] = candidato
                aptitudes[i] = fit_candidato

    tiempo_s = time.perf_counter() - inicio_t
    return _empaquetar_resultado("TLBO", semilla, evaluador, tiempo_s)


def _seleccion_torneo(
    rng: np.random.Generator,
    aptitudes: np.ndarray,
    tam_torneo: int,
) -> int:
    """Selecciona el índice del mejor individuo entre tam_torneo candidatos aleatorios."""
    tam_pob = len(aptitudes)
    k = min(tam_torneo, tam_pob)
    candidatos = rng.choice(tam_pob, size=k, replace=False)
    mejor_idx = int(candidatos[int(np.argmin(aptitudes[candidatos]))])
    return mejor_idx


def ejecutar_ga(
    matriz_distancias: Sequence[Sequence[float]],
    semilla: int,
    tam_poblacion: int = 30,
    max_evaluaciones: int = 3000,
    tam_torneo: int = 3,
    prob_cruce: float = 0.85,
    prob_mutacion: float = 0.2,
    sigma_mutacion: float = 0.1,
    elitismo: int = 2,
) -> dict[str, Any]:
    """Algoritmo Genético (GA): torneo, cruzamiento aritmético, mutación gaussiana y elitismo."""
    inicio_t = time.perf_counter()
    rng = np.random.default_rng(semilla)
    evaluador = EvaluadorRutas(matriz_distancias, max_evaluaciones)
    dim = evaluador.dim

    poblacion, aptitudes = _inicializar_poblacion(rng, tam_poblacion, dim, evaluador)
    elitismo_efectivo = min(elitismo, tam_poblacion)

    while evaluador.puede_evaluar():
        orden = np.argsort(aptitudes)
        nueva_poblacion = [poblacion[idx].copy() for idx in orden[:elitismo_efectivo]]
        nuevas_aptitudes = [float(aptitudes[idx]) for idx in orden[:elitismo_efectivo]]

        while len(nueva_poblacion) < tam_poblacion and evaluador.puede_evaluar():
            idx_p1 = _seleccion_torneo(rng, aptitudes, tam_torneo)
            idx_p2 = _seleccion_torneo(rng, aptitudes, tam_torneo)
            padre1 = poblacion[idx_p1]
            padre2 = poblacion[idx_p2]

            if rng.random() < prob_cruce:
                alpha = rng.random(dim)
                hijo1 = alpha * padre1 + (1.0 - alpha) * padre2
                hijo2 = (1.0 - alpha) * padre1 + alpha * padre2
            else:
                hijo1 = padre1.copy()
                hijo2 = padre2.copy()

            for hijo in (hijo1, hijo2):
                if len(nueva_poblacion) >= tam_poblacion or not evaluador.puede_evaluar():
                    break
                mascara_mut = rng.random(dim) < prob_mutacion
                if np.any(mascara_mut):
                    ruido = rng.normal(0.0, sigma_mutacion, size=dim)
                    hijo = np.where(mascara_mut, hijo + ruido, hijo)
                hijo = np.clip(hijo, 0.0, 1.0)

                fit_hijo = evaluador.evaluar(hijo)
                nueva_poblacion.append(hijo)
                nuevas_aptitudes.append(fit_hijo)

        if len(nueva_poblacion) == tam_poblacion:
            poblacion = np.array(nueva_poblacion, dtype=float)
            aptitudes = np.array(nuevas_aptitudes, dtype=float)
        else:
            # Si el presupuesto terminó a mitad de generación, conservar los ya evaluados
            n_nuevos = len(nueva_poblacion)
            poblacion[:n_nuevos] = np.array(nueva_poblacion, dtype=float)
            aptitudes[:n_nuevos] = np.array(nuevas_aptitudes, dtype=float)

    tiempo_s = time.perf_counter() - inicio_t
    return _empaquetar_resultado("GA", semilla, evaluador, tiempo_s)


ALGORITMOS_METAHEURISTICOS: dict[str, Callable[..., dict[str, Any]]] = {
    "GWO": ejecutar_gwo,
    "PSO": ejecutar_pso,
    "TLBO": ejecutar_tlbo,
    "GA": ejecutar_ga,
}
