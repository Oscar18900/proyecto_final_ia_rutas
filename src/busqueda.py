"""Módulo de búsqueda exacta: UCS y A* con heurística basada en MST (Prim).

Implementado tomando como base directa las prácticas de laboratorio de UCS y A*,
conservando la estructura de colas de prioridad, reconstrucción de camino por
predecesores, listas abierta/cerrada y orden de visita.
"""

from __future__ import annotations

import heapq
import time
from typing import Any, Callable, Sequence


Estado = tuple[int, int]  # (nodo_actual, mascara_visitados)


def reconstruir_camino(
    padres: dict[Estado, Estado], inicio: Estado, meta: Estado
) -> list[Estado]:
    """Reconstruye la secuencia de estados desde inicio hasta meta (idéntica al laboratorio)."""
    camino = [meta]
    actual = meta
    while actual != inicio:
        actual = padres[actual]
        camino.append(actual)
    camino.reverse()
    return camino


def obtener_sucesores(
    estado: Estado,
    num_centros: int,
    matriz_distancias: Sequence[Sequence[float]],
) -> list[tuple[Estado, float]]:
    """Genera los sucesores deterministas de (actual, visitados).

    - Si quedan centros sin visitar, genera transiciones hacia cada centro no visitado
      en orden ascendente (1..num_centros).
    - Cuando todos los centros han sido visitados, genera el retorno al almacén (0).
    """
    actual, visitados = estado
    mascara_completa = (1 << num_centros) - 1

    if visitados == mascara_completa:
        if actual == 0:
            return []
        meta: Estado = (0, mascara_completa)
        return [(meta, float(matriz_distancias[actual][0]))]

    sucesores: list[tuple[Estado, float]] = []
    for siguiente in range(1, num_centros + 1):
        bit = 1 << (siguiente - 1)
        if not (visitados & bit):
            nuevo_estado: Estado = (siguiente, visitados | bit)
            costo_paso = float(matriz_distancias[actual][siguiente])
            sucesores.append((nuevo_estado, costo_paso))
    return sucesores


def mst_prim(
    nodos_pendientes: Sequence[int],
    matriz_distancias: Sequence[Sequence[float]],
) -> float:
    """Calcula el peso total del Árbol de Expansión Mínima (MST) mediante Prim.

    Si |U| <= 1, retorna 0.0.
    """
    k = len(nodos_pendientes)
    if k <= 1:
        return 0.0

    en_mst = [False] * k
    dist_min = [float("inf")] * k
    dist_min[0] = 0.0
    peso_total = 0.0

    for _ in range(k):
        idx_u = -1
        mejor_d = float("inf")
        for idx in range(k):
            if not en_mst[idx] and dist_min[idx] < mejor_d:
                mejor_d = dist_min[idx]
                idx_u = idx

        en_mst[idx_u] = True
        peso_total += mejor_d
        u = nodos_pendientes[idx_u]

        for idx_v in range(k):
            if not en_mst[idx_v]:
                v = nodos_pendientes[idx_v]
                d_uv = float(matriz_distancias[u][v])
                if d_uv < dist_min[idx_v]:
                    dist_min[idx_v] = d_uv

    return peso_total


def heuristica_mst(
    estado: Estado,
    matriz_distancias: Sequence[Sequence[float]],
    num_centros: int,
    cache_mst: dict[int, float] | None = None,
) -> float:
    """Heurística admisible basada en MST:

    - Si U está vacío: h(s) = d(v, 0).
    - Si U no está vacío: h(s) = min[u∈U] d(v, u) + MST(U) + min[u∈U] d(u, 0).
    """
    v, visitados = estado
    mascara_completa = (1 << num_centros) - 1
    mascara_pendientes = mascara_completa ^ visitados

    if mascara_pendientes == 0:
        return float(matriz_distancias[v][0])

    pendientes = [
        c for c in range(1, num_centros + 1) if not (visitados & (1 << (c - 1)))
    ]

    min_v_u = min(float(matriz_distancias[v][u]) for u in pendientes)
    min_u_0 = min(float(matriz_distancias[u][0]) for u in pendientes)

    if cache_mst is not None and mascara_pendientes in cache_mst:
        peso_mst = cache_mst[mascara_pendientes]
    else:
        peso_mst = mst_prim(pendientes, matriz_distancias)
        if cache_mst is not None:
            cache_mst[mascara_pendientes] = peso_mst

    return min_v_u + peso_mst + min_u_0


def extraer_ruta_desde_estados(
    camino_estados: Sequence[Estado],
) -> tuple[list[int], tuple[int, ...]]:
    """Extrae la ruta completa [0, ..., 0] y la permutación (π1..πn) desde estados."""
    if not camino_estados:
        return [], ()
    ruta_completa = [estado[0] for estado in camino_estados]
    permutacion = tuple(ruta_completa[1:-1])
    return ruta_completa, permutacion


def ucs(
    matriz_distancias: Sequence[Sequence[float]],
    inicio: Estado | None = None,
    meta: Estado | None = None,
    tiempo_limite: float | None = None,
) -> tuple[bool, list[Estado], list[Estado], float, dict[Estado, Estado], bool]:
    """Búsqueda de Costo Uniforme (UCS) siguiendo el patrón del laboratorio.

    Retorna:
        (encontrado, camino, orden_visita, costo_total, padres, inconclusa)
    """
    num_centros = len(matriz_distancias) - 1
    mascara_completa = (1 << num_centros) - 1
    if inicio is None:
        inicio = (0, 0)
    if meta is None:
        meta = (0, mascara_completa)

    inicio_reloj = time.perf_counter()
    contador = 0
    cola: list[tuple[float, int, Estado]] = [(0.0, contador, inicio)]

    visitados: set[Estado] = set()
    mejor_costo: dict[Estado, float] = {inicio: 0.0}
    padres: dict[Estado, Estado] = {}
    orden_visita: list[Estado] = []

    while cola:
        if tiempo_limite is not None and (time.perf_counter() - inicio_reloj) > tiempo_limite:
            return False, [], orden_visita, float("inf"), padres, True

        costo_actual, _cnt, estado_actual = heapq.heappop(cola)

        if estado_actual in visitados:
            continue
        if costo_actual > mejor_costo.get(estado_actual, float("inf")):
            continue

        visitados.add(estado_actual)
        orden_visita.append(estado_actual)

        if estado_actual == meta:
            camino = reconstruir_camino(padres, inicio, meta)
            return True, camino, orden_visita, float(costo_actual), padres, False

        for vecino, costo_paso in obtener_sucesores(
            estado_actual, num_centros, matriz_distancias
        ):
            if vecino in visitados:
                continue

            nuevo_costo = costo_actual + costo_paso
            if nuevo_costo < mejor_costo.get(vecino, float("inf")):
                mejor_costo[vecino] = nuevo_costo
                padres[vecino] = estado_actual
                contador += 1
                heapq.heappush(cola, (nuevo_costo, contador, vecino))

    return False, [], orden_visita, float("inf"), padres, False


def Aestrella(
    matriz_distancias: Sequence[Sequence[float]],
    inicio: Estado | None = None,
    meta: Estado | None = None,
    funcion_heuristica: Callable[..., float] = heuristica_mst,
    tiempo_limite: float | None = None,
) -> tuple[
    bool,
    list[Estado],
    list[Estado],
    dict[Estado, float],
    dict[Estado, float],
    dict[Estado, float],
    dict[Estado, Estado],
    bool,
]:
    """Búsqueda A* fiel a la estructura del laboratorio (ListaAbierta, ListaCerrada, g/h/f).

    Retorna:
        (encontrado, camino, orden_visita, g_valores, h_valores, f_valores, padres, inconclusa)
    """
    num_centros = len(matriz_distancias) - 1
    mascara_completa = (1 << num_centros) - 1
    if inicio is None:
        inicio = (0, 0)
    if meta is None:
        meta = (0, mascara_completa)

    inicio_reloj = time.perf_counter()
    cache_mst: dict[int, float] = {}

    contador = 0
    ListaAbierta: list[tuple[float, float, int, Estado]] = []
    en_abierta: dict[Estado, float] = {}
    ListaCerrada: set[Estado] = set()
    padres: dict[Estado, Estado] = {}

    g_valores: dict[Estado, float] = {inicio: 0.0}
    h_valores: dict[Estado, float] = {
        inicio: funcion_heuristica(inicio, matriz_distancias, num_centros, cache_mst)
    }
    f_valores: dict[Estado, float] = {inicio: g_valores[inicio] + h_valores[inicio]}

    heapq.heappush(
        ListaAbierta, (f_valores[inicio], h_valores[inicio], contador, inicio)
    )
    en_abierta[inicio] = g_valores[inicio]

    orden_visita: list[Estado] = []

    while ListaAbierta:
        if tiempo_limite is not None and (time.perf_counter() - inicio_reloj) > tiempo_limite:
            return (
                False,
                [],
                orden_visita,
                g_valores,
                h_valores,
                f_valores,
                padres,
                True,
            )

        f_pop, h_pop, contador_pop, NodoActual = heapq.heappop(ListaAbierta)

        if NodoActual in ListaCerrada:
            continue

        en_abierta.pop(NodoActual, None)
        ListaCerrada.add(NodoActual)
        orden_visita.append(NodoActual)

        if NodoActual == meta:
            camino = reconstruir_camino(padres, inicio, meta)
            return (
                True,
                camino,
                orden_visita,
                g_valores,
                h_valores,
                f_valores,
                padres,
                False,
            )

        for vecino, costo_paso in obtener_sucesores(
            NodoActual, num_centros, matriz_distancias
        ):
            if vecino in ListaCerrada:
                continue

            g_tentativo = g_valores[NodoActual] + costo_paso

            if vecino in en_abierta and g_tentativo >= en_abierta[vecino]:
                continue

            padres[vecino] = NodoActual
            g_valores[vecino] = g_tentativo
            h_valores[vecino] = funcion_heuristica(
                vecino, matriz_distancias, num_centros, cache_mst
            )
            f_valores[vecino] = g_valores[vecino] + h_valores[vecino]

            contador += 1
            heapq.heappush(
                ListaAbierta,
                (f_valores[vecino], h_valores[vecino], contador, vecino),
            )
            en_abierta[vecino] = g_valores[vecino]

    return False, [], orden_visita, g_valores, h_valores, f_valores, padres, False


def ejecutar_ucs(
    matriz_distancias: Sequence[Sequence[float]],
    tiempo_limite: float | None = None,
) -> dict[str, Any]:
    """Ejecuta UCS y devuelve métricas completas y ruta reconstruida."""
    inicio_t = time.perf_counter()
    encontrado, camino, orden_visita, costo_total, padres, inconclusa = ucs(
        matriz_distancias, tiempo_limite=tiempo_limite
    )
    tiempo_s = time.perf_counter() - inicio_t
    ruta_completa, permutacion = extraer_ruta_desde_estados(camino)

    return {
        "algoritmo": "UCS",
        "encontrado": encontrado,
        "inconclusa": inconclusa,
        "costo": costo_total if encontrado else float("inf"),
        "camino_estados": camino,
        "ruta_completa": ruta_completa,
        "permutacion": permutacion,
        "orden_visita": orden_visita,
        "estados_expandidos": len(orden_visita),
        "tiempo_s": tiempo_s,
        "tiempo_ms": tiempo_s * 1000.0,
        "padres": padres,
    }


def ejecutar_astar(
    matriz_distancias: Sequence[Sequence[float]],
    tiempo_limite: float | None = None,
) -> dict[str, Any]:
    """Ejecuta A* con heurística MST y devuelve métricas completas y ruta reconstruida."""
    inicio_t = time.perf_counter()
    (
        encontrado,
        camino,
        orden_visita,
        g_valores,
        h_valores,
        f_valores,
        padres,
        inconclusa,
    ) = Aestrella(matriz_distancias, tiempo_limite=tiempo_limite)
    tiempo_s = time.perf_counter() - inicio_t
    ruta_completa, permutacion = extraer_ruta_desde_estados(camino)

    meta_estado = camino[-1] if camino else None
    costo_total = (
        float(g_valores[meta_estado]) if encontrado and meta_estado is not None else float("inf")
    )

    return {
        "algoritmo": "A*",
        "encontrado": encontrado,
        "inconclusa": inconclusa,
        "costo": costo_total,
        "camino_estados": camino,
        "ruta_completa": ruta_completa,
        "permutacion": permutacion,
        "orden_visita": orden_visita,
        "estados_expandidos": len(orden_visita),
        "tiempo_s": tiempo_s,
        "tiempo_ms": tiempo_s * 1000.0,
        "g_valores": g_valores,
        "h_valores": h_valores,
        "f_valores": f_valores,
        "padres": padres,
    }
