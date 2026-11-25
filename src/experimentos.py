"""Módulo de experimentos: ejecución comparativa, métricas estadísticas y gráficas."""

from __future__ import annotations

import math
import os
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .busqueda import ejecutar_astar, ejecutar_ucs
from .datos import calcular_matriz_distancias, cargar_puntos_csv
from .metaheuristicas import (
    ALGORITMOS_METAHEURISTICOS,
    PARAMETROS_METAHEURISTICAS,
)
from .rutas import (
    formatear_ruta,
    graficar_comparacion_rutas,
    graficar_ruta,
    validar_ruta_completa,
)


TOLERANCIA_OPTIMO = 1e-6


def calcular_brecha_optimalidad(
    distancia: float, f_estrella: float | None
) -> float | None:
    """Calcula la brecha de optimalidad (%) = [(L(π) − f*) / f*] × 100."""
    if f_estrella is None or not math.isfinite(f_estrella) or f_estrella <= 0:
        return None
    if abs(distancia - f_estrella) <= 1e-9:
        return 0.0
    return ((distancia - f_estrella) / f_estrella) * 100.0


def _construir_tablas(
    puntos: list[dict[str, Any]],
    matriz_distancias: list[list[float]],
    res_ucs: dict[str, Any],
    res_astar: dict[str, Any],
    f_estrella: float | None,
    resumen_meta: dict[str, dict[str, Any]],
    corridas_detalladas: list[dict[str, Any]],
    max_evaluaciones: int,
) -> dict[str, pd.DataFrame]:
    """Construye los DataFrames estructurados de todas las tablas del proyecto."""
    # Tabla 1: Comparación descriptiva principal (Sección 8 del documento)
    filas_principal: list[dict[str, Any]] = []
    for res_ex in (res_ucs, res_astar):
        if res_ex["encontrado"]:
            mejor_str = f"{res_ex['costo']:.4f}"
            esfuerzo_str = f"{res_ex['estados_expandidos']} estados expandidos"
        else:
            estado_txt = "Inconclusa" if res_ex["inconclusa"] else "Sin solución"
            mejor_str = estado_txt
            esfuerzo_str = f"{res_ex['estados_expandidos']} estados expandidos ({estado_txt})"

        filas_principal.append(
            {
                "Algoritmo": res_ex["algoritmo"],
                "Mejor distancia (km)": mejor_str,
                "Promedio (km)": "—",
                "Tiempo (s)": f"{res_ex['tiempo_s']:.6f}",
                "Esfuerzo": esfuerzo_str,
            }
        )

    for nombre in ("GWO", "PSO", "TLBO", "GA"):
        est = resumen_meta[nombre]
        filas_principal.append(
            {
                "Algoritmo": nombre,
                "Mejor distancia (km)": f"{est['mejor_distancia']:.4f}",
                "Promedio (km)": f"{est['distancia_promedio']:.4f}",
                "Tiempo (s)": f"{est['tiempo_promedio_s']:.6f}",
                "Esfuerzo": f"{max_evaluaciones} evaluaciones",
            }
        )
    df_principal = pd.DataFrame(filas_principal)

    # Tabla 2: Estadísticas detalladas de las 20 ejecuciones de las metaheurísticas
    filas_meta: list[dict[str, Any]] = []
    for nombre in ("GWO", "PSO", "TLBO", "GA"):
        est = resumen_meta[nombre]
        filas_meta.append(
            {
                "Algoritmo": nombre,
                "Mejor distancia (km)": round(est["mejor_distancia"], 4),
                "Peor distancia (km)": round(est["peor_distancia"], 4),
                "Promedio (km)": round(est["distancia_promedio"], 4),
                "Desv. estándar (km)": round(est["desv_estandar"], 4),
                "Tiempo promedio (s)": round(est["tiempo_promedio_s"], 6),
                "% alcanza f*": (
                    round(est["porcentaje_optimo"], 2)
                    if est["porcentaje_optimo"] is not None
                    else "—"
                ),
                "Brecha mejor (%)": (
                    round(est["brecha_mejor_pct"], 4)
                    if est["brecha_mejor_pct"] is not None
                    else "—"
                ),
                "Brecha promedio (%)": (
                    round(est["brecha_promedio_pct"], 4)
                    if est["brecha_promedio_pct"] is not None
                    else "—"
                ),
                "Mejor ruta": formatear_ruta(est["mejor_ruta_completa"]),
            }
        )
    df_meta = pd.DataFrame(filas_meta)

    # Tabla 3: Detalle comparativo de búsqueda exacta (UCS vs A*)
    exp_ucs = res_ucs["estados_expandidos"]
    filas_exacta: list[dict[str, Any]] = []
    for res_ex in (res_ucs, res_astar):
        if res_ex["encontrado"]:
            estado_txt = "Óptimo encontrado"
            costo_val: Any = round(res_ex["costo"], 4)
            ruta_txt = formatear_ruta(res_ex["ruta_completa"])
        else:
            estado_txt = "Inconclusa" if res_ex["inconclusa"] else "Sin solución"
            costo_val = "—"
            ruta_txt = "—"

        if res_ex["algoritmo"] == "UCS":
            reduccion: Any = 0.0
        elif exp_ucs > 0:
            reduccion = round(
                ((exp_ucs - res_ex["estados_expandidos"]) / exp_ucs) * 100.0, 2
            )
        else:
            reduccion = 0.0

        filas_exacta.append(
            {
                "Algoritmo": res_ex["algoritmo"],
                "Estado": estado_txt,
                "f* (km)": costo_val,
                "Ruta óptima": ruta_txt,
                "Estados expandidos": res_ex["estados_expandidos"],
                "Tiempo (s)": round(res_ex["tiempo_s"], 6),
                "Tiempo (ms)": round(res_ex["tiempo_ms"], 4),
                "Reducción de estados vs UCS (%)": reduccion,
            }
        )
    df_exacta = pd.DataFrame(filas_exacta)

    # Tabla 4: Ejecuciones detalladas (80 corridas)
    df_detalladas = pd.DataFrame(corridas_detalladas)

    # Tabla 5: Parámetros documentados de las metaheurísticas
    df_parametros = pd.DataFrame(PARAMETROS_METAHEURISTICAS)

    # Tabla 6: Coordenadas de entrada
    df_coordenadas = pd.DataFrame(
        [
            {
                "Lugar": p["lugar"],
                "Nodo": p["nodo"],
                "X (km)": p["x"],
                "Y (km)": p["y"],
            }
            for p in puntos
        ]
    )

    # Tabla 7: Matriz de distancias N×N
    filas_matriz: list[dict[str, Any]] = []
    for i, fila in enumerate(matriz_distancias):
        registro: dict[str, Any] = {"Nodo": i}
        for j, dist in enumerate(fila):
            registro[str(j)] = round(dist, 4)
        filas_matriz.append(registro)
    df_matriz = pd.DataFrame(filas_matriz)

    return {
        "tabla_1_comparacion_principal": df_principal,
        "tabla_2_estadisticas_metaheuristicas": df_meta,
        "tabla_3_busqueda_exacta": df_exacta,
        "tabla_4_ejecuciones_detalladas": df_detalladas,
        "tabla_5_parametros_metaheuristicas": df_parametros,
        "tabla_6_coordenadas": df_coordenadas,
        "tabla_7_matriz_distancias": df_matriz,
    }


def ejecutar_experimento_completo(
    ruta_csv: str | None = None,
    semillas: Iterable[int] = range(20),
    tam_poblacion: int = 30,
    max_evaluaciones: int = 3000,
    tiempo_limite_exacto: float | None = None,
) -> dict[str, Any]:
    """Ejecuta UCS, A* y las 4 metaheurísticas sobre las semillas indicadas."""
    lista_semillas = list(semillas)
    puntos = cargar_puntos_csv(ruta_csv)
    matriz_distancias = calcular_matriz_distancias(puntos)
    num_centros = len(puntos) - 1

    # 1. Búsqueda exacta (ejecución determinista única)
    res_ucs = ejecutar_ucs(matriz_distancias, tiempo_limite=tiempo_limite_exacto)
    res_astar = ejecutar_astar(matriz_distancias, tiempo_limite=tiempo_limite_exacto)

    if res_ucs["encontrado"]:
        es_valida, msg = validar_ruta_completa(res_ucs["ruta_completa"], num_centros)
        if not es_valida:
            raise RuntimeError(f"Ruta inválida producida por UCS: {msg}")

    if res_astar["encontrado"]:
        es_valida, msg = validar_ruta_completa(res_astar["ruta_completa"], num_centros)
        if not es_valida:
            raise RuntimeError(f"Ruta inválida producida por A*: {msg}")

    if res_ucs["encontrado"] and res_astar["encontrado"]:
        diferencia = abs(res_ucs["costo"] - res_astar["costo"])
        if diferencia > TOLERANCIA_OPTIMO:
            raise RuntimeError(
                f"Inconsistencia entre UCS ({res_ucs['costo']:.8f}) y A* "
                f"({res_astar['costo']:.8f}); diferencia = {diferencia:.8e}"
            )
        f_estrella: float | None = float(res_astar["costo"])
        ruta_optima: tuple[int, ...] | None = res_astar["permutacion"]
    elif res_astar["encontrado"]:
        f_estrella = float(res_astar["costo"])
        ruta_optima = res_astar["permutacion"]
    elif res_ucs["encontrado"]:
        f_estrella = float(res_ucs["costo"])
        ruta_optima = res_ucs["permutacion"]
    else:
        f_estrella = None
        ruta_optima = None

    # 2. Cuatro metaheurísticas (20 ejecuciones con semillas 0..19)
    resumen_meta: dict[str, dict[str, Any]] = {}
    corridas_detalladas: list[dict[str, Any]] = []

    for nombre, funcion_meta in ALGORITMOS_METAHEURISTICOS.items():
        corridas: list[dict[str, Any]] = []
        for semilla in lista_semillas:
            res = funcion_meta(
                matriz_distancias,
                semilla=semilla,
                tam_poblacion=tam_poblacion,
                max_evaluaciones=max_evaluaciones,
            )
            es_valida, msg = validar_ruta_completa(
                res["mejor_ruta_completa"], num_centros
            )
            if not es_valida:
                raise RuntimeError(
                    f"Ruta inválida en {nombre} (semilla {semilla}): {msg}"
                )
            corridas.append(res)

            brecha_corrida = calcular_brecha_optimalidad(
                res["mejor_distancia"], f_estrella
            )
            alcanza = (
                abs(res["mejor_distancia"] - f_estrella) <= TOLERANCIA_OPTIMO
                if f_estrella is not None
                else False
            )
            corridas_detalladas.append(
                {
                    "Algoritmo": nombre,
                    "Semilla": semilla,
                    "Mejor distancia (km)": round(res["mejor_distancia"], 4),
                    "Brecha (%)": (
                        round(brecha_corrida, 4) if brecha_corrida is not None else "—"
                    ),
                    "Alcanza f*": "Sí" if alcanza else "No",
                    "Evaluaciones": res["evaluaciones"],
                    "Tiempo (s)": round(res["tiempo_s"], 6),
                    "Ruta": formatear_ruta(res["mejor_ruta_completa"]),
                }
            )

        distancias = np.array([c["mejor_distancia"] for c in corridas], dtype=float)
        tiempos = np.array([c["tiempo_s"] for c in corridas], dtype=float)
        historiales = np.vstack([c["historial"] for c in corridas])

        idx_mejor = int(np.argmin(distancias))
        mejor_corrida = corridas[idx_mejor]
        mejor_dist = float(distancias[idx_mejor])
        peor_dist = float(np.max(distancias))
        promedio_dist = float(np.mean(distancias))
        desv_std = float(np.std(distancias, ddof=1)) if len(distancias) > 1 else 0.0
        tiempo_prom = float(np.mean(tiempos))

        if f_estrella is not None:
            aciertos = int(np.sum(np.abs(distancias - f_estrella) <= TOLERANCIA_OPTIMO))
            porcentaje_optimo: float | None = (aciertos / len(corridas)) * 100.0
        else:
            porcentaje_optimo = None

        resumen_meta[nombre] = {
            "algoritmo": nombre,
            "corridas": corridas,
            "mejor_distancia": mejor_dist,
            "peor_distancia": peor_dist,
            "distancia_promedio": promedio_dist,
            "desv_estandar": desv_std,
            "tiempo_promedio_s": tiempo_prom,
            "tiempo_total_s": float(np.sum(tiempos)),
            "porcentaje_optimo": porcentaje_optimo,
            "brecha_mejor_pct": calcular_brecha_optimalidad(mejor_dist, f_estrella),
            "brecha_promedio_pct": calcular_brecha_optimalidad(
                promedio_dist, f_estrella
            ),
            "mejor_permutacion": mejor_corrida["mejor_permutacion"],
            "mejor_ruta_completa": mejor_corrida["mejor_ruta_completa"],
            "mejor_semilla": mejor_corrida["semilla"],
            "curva_promedio": np.mean(historiales, axis=0),
            "curva_std": (
                np.std(historiales, axis=0, ddof=1)
                if len(corridas) > 1
                else np.zeros(max_evaluaciones, dtype=float)
            ),
            "curva_mejor": np.min(historiales, axis=0),
        }

    tablas = _construir_tablas(
        puntos=puntos,
        matriz_distancias=matriz_distancias,
        res_ucs=res_ucs,
        res_astar=res_astar,
        f_estrella=f_estrella,
        resumen_meta=resumen_meta,
        corridas_detalladas=corridas_detalladas,
        max_evaluaciones=max_evaluaciones,
    )

    return {
        "puntos": puntos,
        "matriz_distancias": matriz_distancias,
        "ucs": res_ucs,
        "astar": res_astar,
        "f_estrella": f_estrella,
        "ruta_optima": ruta_optima,
        "metaheuristicas": resumen_meta,
        "corridas_detalladas": corridas_detalladas,
        "tablas": tablas,
        "max_evaluaciones": max_evaluaciones,
        "tam_poblacion": tam_poblacion,
        "semillas": lista_semillas,
    }


def graficar_curvas_convergencia(
    resultados: dict[str, Any],
    ruta_salida: str | None = None,
    mostrar: bool = False,
) -> None:
    """Grafica mejor distancia frente a número de evaluaciones con referencia en f*."""
    colores = {
        "GWO": "#0284c7",
        "PSO": "#d97706",
        "TLBO": "#7c3aed",
        "GA": "#059669",
    }
    max_eval = resultados["max_evaluaciones"]
    eje_x = np.arange(1, max_eval + 1)
    f_estrella = resultados["f_estrella"]

    fig, ax = plt.subplots(figsize=(10.2, 6.2), dpi=180)

    for nombre in ("GWO", "PSO", "TLBO", "GA"):
        est = resultados["metaheuristicas"][nombre]
        media = est["curva_promedio"]
        std = est["curva_std"]
        color = colores[nombre]
        ax.plot(
            eje_x,
            media,
            label=f"{nombre} (prom. final: {media[-1]:.4f} km)",
            color=color,
            linewidth=2.0,
        )
        ax.fill_between(
            eje_x,
            np.maximum(media - std, f_estrella if f_estrella is not None else 0.0),
            media + std,
            color=color,
            alpha=0.12,
        )

    if f_estrella is not None:
        ax.axhline(
            y=f_estrella,
            color="#dc2626",
            linestyle="--",
            linewidth=1.8,
            label=f"Óptimo global f* = {f_estrella:.4f} km",
        )

    ax.set_title(
        "Curvas de convergencia: Mejor distancia promedio vs. número de evaluaciones",
        fontsize=12.5,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Número de evaluaciones de L(π)", fontsize=10.5)
    ax.set_ylabel("Mejor distancia encontrada (km)", fontsize=10.5)
    ax.set_xlim(1, max_eval)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    fig.tight_layout()

    if ruta_salida:
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        fig.savefig(ruta_salida, dpi=180, bbox_inches="tight")
    if mostrar:
        plt.show()
    plt.close(fig)


def graficar_comparacion_busqueda_exacta(
    resultados: dict[str, Any],
    ruta_salida: str | None = None,
    mostrar: bool = False,
) -> None:
    """Compara exclusivamente los estados expandidos por UCS y A* (métrica de búsqueda exacta)."""
    ucs_exp = resultados["ucs"]["estados_expandidos"]
    astar_exp = resultados["astar"]["estados_expandidos"]
    reduccion = ((ucs_exp - astar_exp) / ucs_exp * 100.0) if ucs_exp > 0 else 0.0

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)
    etiquetas = ["UCS\n(Costo Uniforme)", "A*\n(Heurística MST Prim)"]
    valores = [ucs_exp, astar_exp]
    barras = ax.bar(
        etiquetas,
        valores,
        color=["#64748b", "#059669"],
        width=0.5,
        edgecolor="#0f172a",
        linewidth=0.8,
    )

    max_val = max(valores) if valores else 1
    ax.set_ylim(0, max_val * 1.2)

    for idx, barra in enumerate(barras):
        altura = barra.get_height()
        texto = f"{int(altura):,} estados"
        if idx == 1:
            texto += f"\n(-{reduccion:.2f}%)"
        ax.annotate(
            texto,
            xy=(barra.get_x() + barra.get_width() / 2, altura),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    f_estrella = resultados["f_estrella"]
    subtitulo = (
        f"Ambos garantizan f* = {f_estrella:.4f} km"
        if f_estrella is not None
        else "Comparación de esfuerzo en espacio de estados"
    )
    ax.set_title(
        f"Esfuerzo de búsqueda exacta: Estados expandidos\n({subtitulo})",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax.set_ylabel("Estados expandidos", fontsize=10.5)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()

    if ruta_salida:
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        fig.savefig(ruta_salida, dpi=180, bbox_inches="tight")
    if mostrar:
        plt.show()
    plt.close(fig)


def graficar_boxplot_metaheuristicas(
    resultados: dict[str, Any],
    ruta_salida: str | None = None,
    mostrar: bool = False,
) -> None:
    """Diagrama de caja con la distribución de distancias en las 20 ejecuciones."""
    orden = ["GWO", "PSO", "TLBO", "GA"]
    datos = [
        [c["mejor_distancia"] for c in resultados["metaheuristicas"][alg]["corridas"]]
        for alg in orden
    ]
    colores = ["#0284c7", "#d97706", "#7c3aed", "#059669"]
    f_estrella = resultados["f_estrella"]

    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=180)
    bp = ax.boxplot(
        datos,
        tick_labels=orden,
        patch_artist=True,
        medianprops={"color": "#0f172a", "linewidth": 1.8},
    )
    for patch, color in zip(bp["boxes"], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    # Puntos individuales con leve desplazamiento determinista
    rng_vis = np.random.default_rng(42)
    for idx, vals in enumerate(datos, start=1):
        xs = rng_vis.normal(idx, 0.04, size=len(vals))
        ax.scatter(xs, vals, color="#1e293b", alpha=0.65, s=22, zorder=3)

    if f_estrella is not None:
        ax.axhline(
            y=f_estrella,
            color="#dc2626",
            linestyle="--",
            linewidth=1.8,
            label=f"Óptimo global f* = {f_estrella:.4f} km",
        )
        ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    ax.set_title(
        f"Distribución de distancias obtenidas ({len(resultados['semillas'])} semillas por algoritmo)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.set_ylabel("Mejor distancia por corrida (km)", fontsize=10.5)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()

    if ruta_salida:
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        fig.savefig(ruta_salida, dpi=180, bbox_inches="tight")
    if mostrar:
        plt.show()
    plt.close(fig)


def generar_todas_las_graficas(
    resultados: dict[str, Any],
    carpeta_graficas: str,
    mostrar: bool = False,
) -> list[str]:
    """Genera y guarda todas las gráficas requeridas en la carpeta indicada."""
    os.makedirs(carpeta_graficas, exist_ok=True)
    rutas_generadas: list[str] = []

    if resultados["ruta_optima"] is not None and resultados["f_estrella"] is not None:
        ruta_opt = os.path.join(carpeta_graficas, "mapa_ruta_optima.png")
        graficar_ruta(
            resultados["puntos"],
            resultados["ruta_optima"],
            resultados["f_estrella"],
            titulo="Ruta óptima global (UCS y A* con MST)",
            ruta_salida=ruta_opt,
            mostrar=mostrar,
        )
        rutas_generadas.append(ruta_opt)

        ruta_meta_mapas = os.path.join(
            carpeta_graficas, "mapa_mejores_rutas_metaheuristicas.png"
        )
        graficar_comparacion_rutas(
            resultados["puntos"],
            resultados["f_estrella"],
            resultados["ruta_optima"],
            resultados["metaheuristicas"],
            ruta_salida=ruta_meta_mapas,
            mostrar=mostrar,
        )
        rutas_generadas.append(ruta_meta_mapas)

    ruta_conv = os.path.join(carpeta_graficas, "curvas_convergencia.png")
    graficar_curvas_convergencia(resultados, ruta_salida=ruta_conv, mostrar=mostrar)
    rutas_generadas.append(ruta_conv)

    ruta_exacta = os.path.join(carpeta_graficas, "comparacion_busqueda_exacta.png")
    graficar_comparacion_busqueda_exacta(
        resultados, ruta_salida=ruta_exacta, mostrar=mostrar
    )
    rutas_generadas.append(ruta_exacta)

    ruta_box = os.path.join(carpeta_graficas, "boxplot_metaheuristicas.png")
    graficar_boxplot_metaheuristicas(resultados, ruta_salida=ruta_box, mostrar=mostrar)
    rutas_generadas.append(ruta_box)

    return rutas_generadas
