"""Módulo de datos: lectura del archivo CSV y cálculo de la matriz de distancias."""

from __future__ import annotations

import csv
import math
import os
from typing import Any


COLUMNAS_REQUERIDAS = ("Lugar", "Nodo", "X", "Y")


def obtener_ruta_csv_por_defecto() -> str:
    """Devuelve la ruta por defecto al archivo CSV de los nueve puntos."""
    raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(raiz_proyecto, "data", "puntos_distribucion.csv")


def cargar_puntos_csv(ruta_csv: str | None = None) -> list[dict[str, Any]]:
    """Lee y valida el archivo CSV con los puntos de distribución.

    Cada fila debe contener las columnas: Lugar, Nodo, X, Y.
    Los nodos deben formar la secuencia contigua 0..N-1, siendo 0 el almacén.
    """
    ruta = ruta_csv or obtener_ruta_csv_por_defecto()
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No se encontró el archivo CSV: {ruta}")

    puntos: list[dict[str, Any]] = []
    with open(ruta, mode="r", encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)
        if lector.fieldnames is None:
            raise ValueError("El archivo CSV está vacío o no tiene encabezados.")

        encabezados = [col.strip() for col in lector.fieldnames]
        faltantes = [col for col in COLUMNAS_REQUERIDAS if col not in encabezados]
        if faltantes:
            raise ValueError(
                f"Faltan columnas requeridas en el CSV: {', '.join(faltantes)}"
            )

        for num_fila, fila in enumerate(lector, start=2):
            fila_limpia = {
                (k.strip() if k is not None else ""): (
                    v.strip() if v is not None else ""
                )
                for k, v in fila.items()
            }
            if not any(fila_limpia.values()):
                continue

            lugar = fila_limpia["Lugar"]
            if not lugar:
                raise ValueError(f"Fila {num_fila}: la columna 'Lugar' está vacía.")

            try:
                nodo = int(fila_limpia["Nodo"])
                x = float(fila_limpia["X"])
                y = float(fila_limpia["Y"])
            except ValueError as exc:
                raise ValueError(
                    f"Fila {num_fila}: valores numéricos inválidos en Nodo, X o Y."
                ) from exc

            puntos.append({"lugar": lugar, "nodo": nodo, "x": x, "y": y})

    if len(puntos) < 2:
        raise ValueError(
            "El archivo CSV debe incluir al menos el almacén (Nodo 0) y un centro."
        )

    puntos.sort(key=lambda p: p["nodo"])
    nodos_esperados = list(range(len(puntos)))
    nodos_leidos = [p["nodo"] for p in puntos]
    if nodos_leidos != nodos_esperados:
        raise ValueError(
            f"Los nodos deben ser consecutivos 0..{len(puntos) - 1}. "
            f"Se encontraron: {nodos_leidos}"
        )

    return puntos


def calcular_matriz_distancias(puntos: list[dict[str, Any]]) -> list[list[float]]:
    """Construye la matriz N×N de distancias euclidianas d(i, j)."""
    n = len(puntos)
    matriz = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = puntos[i]["x"], puntos[i]["y"]
        for j in range(i + 1, n):
            xj, yj = puntos[j]["x"], puntos[j]["y"]
            distancia = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            matriz[i][j] = distancia
            matriz[j][i] = distancia
    return matriz
