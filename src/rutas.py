"""Módulo de rutas: validación, decodificación, cálculo de L(π) y gráficas de rutas."""

from __future__ import annotations

import os
from typing import Any, Sequence

import matplotlib.pyplot as plt


def validar_permutacion(
    permutacion: Sequence[int], num_centros: int
) -> tuple[bool, str]:
    """Verifica que una permutación visite los centros 1..num_centros exactamente una vez."""
    if len(permutacion) != num_centros:
        return (
            False,
            f"La permutación debe tener {num_centros} centros, tiene {len(permutacion)}.",
        )

    vistos: set[int] = set()
    for elemento in permutacion:
        if isinstance(elemento, bool) or not isinstance(elemento, int):
            return False, f"Elemento inválido en la ruta: {elemento!r}."
        if elemento < 1 or elemento > num_centros:
            return (
                False,
                f"El nodo {elemento} está fuera del rango válido 1..{num_centros}.",
            )
        if elemento in vistos:
            return False, f"El centro {elemento} aparece repetido en la ruta."
        vistos.add(elemento)

    faltantes = set(range(1, num_centros + 1)) - vistos
    if faltantes:
        return False, f"Faltan centros por visitar: {sorted(faltantes)}."

    return True, "Permutación válida."


def validar_ruta_completa(
    ruta_completa: Sequence[int], num_centros: int
) -> tuple[bool, str]:
    """Verifica que una ruta completa inicie y termine en el almacén (0) y visite todos los centros."""
    if len(ruta_completa) != num_centros + 2:
        return (
            False,
            f"La ruta completa debe tener {num_centros + 2} nodos (0 + centros + 0), "
            f"tiene {len(ruta_completa)}.",
        )
    if ruta_completa[0] != 0 or ruta_completa[-1] != 0:
        return False, "La ruta completa debe iniciar y terminar en el almacén (nodo 0)."

    return validar_permutacion(ruta_completa[1:-1], num_centros)


def construir_ruta_completa(permutacion: Sequence[int]) -> list[int]:
    """Convierte una permutación de centros (π1..πn) en la ruta completa [0, π1..πn, 0]."""
    return [0, *permutacion, 0]


def calcular_costo_ruta(
    permutacion: Sequence[int],
    matriz_distancias: Sequence[Sequence[float]],
    validar: bool = True,
) -> float:
    """Calcula L(π) = d(0, π1) + Σ d(πk, πk+1) + d(πn, 0)."""
    num_centros = len(matriz_distancias) - 1
    if validar:
        es_valida, mensaje = validar_permutacion(permutacion, num_centros)
        if not es_valida:
            raise ValueError(mensaje)

    if num_centros == 0:
        return 0.0

    costo = matriz_distancias[0][permutacion[0]]
    for k in range(num_centros - 1):
        costo += matriz_distancias[permutacion[k]][permutacion[k + 1]]
    costo += matriz_distancias[permutacion[-1]][0]
    return float(costo)


def decodificar_claves_aleatorias(claves: Sequence[float]) -> tuple[int, ...]:
    """Decodifica un vector de claves aleatorias x=(x1..xn) ordenando índices de menor a mayor.

    En caso de empate exacto, desempata por el índice más pequeño.
    """
    indices_ordenados = sorted(range(len(claves)), key=lambda i: (float(claves[i]), i))
    return tuple(i + 1 for i in indices_ordenados)


def formatear_ruta(ruta_completa: Sequence[int]) -> str:
    """Formatea una ruta como '0 -> c1 -> ... -> 0'."""
    return " -> ".join(str(nodo) for nodo in ruta_completa)


def _dibujar_ruta_en_eje(
    ax: plt.Axes,
    puntos: list[dict[str, Any]],
    permutacion: Sequence[int],
    color_ruta: str,
    titulo: str,
    ruta_referencia: Sequence[int] | None = None,
) -> None:
    """Dibuja un mapa cartesiano 2D con nodos y flechas dirigidas de la ruta."""
    coords = {p["nodo"]: (p["x"], p["y"]) for p in puntos}

    # Dibujar ruta de referencia óptima en el fondo si se proporciona
    if ruta_referencia is not None:
        ref_completa = construir_ruta_completa(ruta_referencia)
        rx = [coords[n][0] for n in ref_completa]
        ry = [coords[n][1] for n in ref_completa]
        ax.plot(
            rx,
            ry,
            linestyle="--",
            linewidth=1.3,
            color="#94a3b8",
            alpha=0.8,
            label="Referencia óptima f*",
            zorder=1,
        )

    ruta_completa = construir_ruta_completa(permutacion)
    for idx in range(len(ruta_completa) - 1):
        origen = ruta_completa[idx]
        destino = ruta_completa[idx + 1]
        x1, y1 = coords[origen]
        x2, y2 = coords[destino]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "-|>,head_length=0.55,head_width=0.32",
                "color": color_ruta,
                "lw": 2.0,
                "shrinkA": 7,
                "shrinkB": 8,
            },
            zorder=2,
        )

    # Dibujar almacén y centros
    almacen = puntos[0]
    ax.scatter(
        [almacen["x"]],
        [almacen["y"]],
        marker="s",
        s=150,
        color="#dc2626",
        edgecolors="#7f1d1d",
        linewidths=1.4,
        label="Almacén (0)",
        zorder=4,
    )
    ax.annotate(
        f"0 ({almacen['x']:g},{almacen['y']:g})",
        (almacen["x"], almacen["y"]),
        textcoords="offset points",
        xytext=(7, -12),
        fontsize=8.5,
        fontweight="bold",
        color="#7f1d1d",
        zorder=5,
    )

    centros_x = [p["x"] for p in puntos[1:]]
    centros_y = [p["y"] for p in puntos[1:]]
    ax.scatter(
        centros_x,
        centros_y,
        marker="o",
        s=95,
        color="#1d4ed8",
        edgecolors="#1e3a8a",
        linewidths=1.1,
        label="Centros (1–8)",
        zorder=3,
    )
    for p in puntos[1:]:
        ax.annotate(
            f"{p['nodo']} ({p['x']:g},{p['y']:g})",
            (p["x"], p["y"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8.5,
            color="#0f172a",
            zorder=5,
        )

    ax.set_title(titulo, fontsize=10.5, fontweight="bold", pad=8)
    ax.set_xlabel("X (km)", fontsize=9)
    ax.set_ylabel("Y (km)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.55)
    xs = [p["x"] for p in puntos]
    ys = [p["y"] for p in puntos]
    ax.set_xlim(min(xs) - 1.5, max(xs) + 2.2)
    ax.set_ylim(min(ys) - 1.5, max(ys) + 1.8)
    ax.set_aspect("equal", adjustable="box")


def graficar_ruta(
    puntos: list[dict[str, Any]],
    permutacion: Sequence[int],
    costo: float,
    titulo: str = "Ruta óptima de distribución",
    ruta_salida: str | None = None,
    mostrar: bool = False,
) -> None:
    """Genera y opcionalmente guarda la gráfica cartesiana de una ruta."""
    fig, ax = plt.subplots(figsize=(8.2, 6.4), dpi=180)
    ruta_txt = formatear_ruta(construir_ruta_completa(permutacion))
    titulo_completo = f"{titulo}\nL(π) = {costo:.4f} km | Ruta: {ruta_txt}"
    _dibujar_ruta_en_eje(ax, puntos, permutacion, "#059669", titulo_completo)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    fig.tight_layout()

    if ruta_salida:
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        fig.savefig(ruta_salida, dpi=180, bbox_inches="tight")
    if mostrar:
        plt.show()
    plt.close(fig)


def graficar_comparacion_rutas(
    puntos: list[dict[str, Any]],
    f_estrella: float,
    ruta_optima: Sequence[int],
    mejores_rutas_meta: dict[str, dict[str, Any]],
    ruta_salida: str | None = None,
    mostrar: bool = False,
) -> None:
    """Genera una figura 2×2 con las mejores rutas de GWO, PSO, TLBO y GA."""
    colores = {
        "GWO": "#0284c7",
        "PSO": "#d97706",
        "TLBO": "#7c3aed",
        "GA": "#059669",
    }
    orden_algoritmos = ["GWO", "PSO", "TLBO", "GA"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), dpi=180)
    axes_planos = axes.flatten()

    for idx, nombre in enumerate(orden_algoritmos):
        ax = axes_planos[idx]
        info = mejores_rutas_meta[nombre]
        perm = info["mejor_permutacion"]
        dist = info["mejor_distancia"]
        brecha = info["brecha_mejor_pct"]
        ruta_txt = formatear_ruta(construir_ruta_completa(perm))
        titulo = (
            f"{nombre}: L = {dist:.4f} km (brecha: {brecha:.4f}%)\n"
            f"Ruta: {ruta_txt}"
        )
        _dibujar_ruta_en_eje(
            ax,
            puntos,
            perm,
            colores.get(nombre, "#2563eb"),
            titulo,
            ruta_referencia=ruta_optima,
        )
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    fig.suptitle(
        f"Mejores rutas obtenidas por cada metaheurística vs óptimo global (f* = {f_estrella:.4f} km)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if ruta_salida:
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        fig.savefig(ruta_salida, dpi=180, bbox_inches="tight")
    if mostrar:
        plt.show()
    plt.close(fig)
