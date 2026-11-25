"""Interfaz principal CLI y menú interactivo del sistema de optimización de rutas."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from tabulate import tabulate

from src.busqueda import ejecutar_astar, ejecutar_ucs
from src.datos import (
    calcular_matriz_distancias,
    cargar_puntos_csv,
    obtener_ruta_csv_por_defecto,
)
from src.experimentos import (
    ejecutar_experimento_completo,
    generar_todas_las_graficas,
)
from src.exportar import FORMATOS_VALIDOS, exportar_tablas
from src.rutas import (
    calcular_costo_ruta,
    construir_ruta_completa,
    formatear_ruta,
    validar_permutacion,
)


def _configurar_consola() -> None:
    """Asegura salida UTF-8 en terminales de Windows/PowerShell."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _carpeta_resultados_base() -> str:
    raiz = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(raiz, "resultados")


def _imprimir_tabla(titulo: str, df: Any) -> None:
    print(f"\n=== {titulo} ===")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))


def _imprimir_resumen_consola(resultados: dict[str, Any]) -> None:
    f_estrella = resultados["f_estrella"]
    ruta_optima = resultados["ruta_optima"]

    print("\n" + "=" * 78)
    print("RESUMEN GLOBAL DE RESULTADOS")
    print("=" * 78)
    if f_estrella is not None and ruta_optima is not None:
        print(f"Distancia óptima global f* : {f_estrella:.6f} km")
        print(
            f"Ruta óptima global         : {formatear_ruta(construir_ruta_completa(ruta_optima))}"
        )
    else:
        print("f* no disponible (búsqueda exacta inconclusa dentro del límite).")

    tablas = resultados["tablas"]
    _imprimir_tabla(
        "Tabla 1. Comparación descriptiva (Sección 8)",
        tablas["tabla_1_comparacion_principal"],
    )
    _imprimir_tabla(
        "Tabla 2. Estadísticas de las metaheurísticas",
        tablas["tabla_2_estadisticas_metaheuristicas"],
    )
    _imprimir_tabla(
        "Tabla 3. Detalle de búsqueda exacta (UCS vs. A*)",
        tablas["tabla_3_busqueda_exacta"],
    )
    _imprimir_tabla(
        "Tabla 5. Parámetros de las metaheurísticas",
        tablas["tabla_5_parametros_metaheuristicas"],
    )


def _ejecutar_todo(args: argparse.Namespace) -> None:
    semillas = range(args.semillas)
    print(f"Cargando datos desde: {args.csv or obtener_ruta_csv_por_defecto()}")
    print(
        f"Ejecutando UCS, A* y 4 metaheurísticas "
        f"({args.semillas} semillas, {args.evaluaciones} evaluaciones/corrida)..."
    )

    resultados = ejecutar_experimento_completo(
        ruta_csv=args.csv,
        semillas=semillas,
        max_evaluaciones=args.evaluaciones,
        tiempo_limite_exacto=args.tiempo_exacto,
    )
    _imprimir_resumen_consola(resultados)

    carpeta_base = _carpeta_resultados_base()

    if not args.sin_graficas:
        carpeta_graficas = os.path.join(carpeta_base, "graficas")
        graficas = generar_todas_las_graficas(
            resultados, carpeta_graficas, mostrar=False
        )
        print("\n=== Gráficas generadas ===")
        for g in graficas:
            print(f"  - {g}")

    carpeta_tablas = os.path.join(carpeta_base, "tablas")
    archivos_tablas = exportar_tablas(
        resultados, carpeta_tablas, formatos=args.exportar
    )
    print(f"\n=== Tablas exportadas ({', '.join(args.exportar)}) ===")
    for ruta in archivos_tablas:
        print(f"  - {ruta}")


def _mostrar_datos_y_matriz(
    puntos: list[dict[str, Any]], matriz: list[list[float]]
) -> None:
    print("\n=== Puntos cargados del CSV ===")
    filas_puntos = [
        [p["lugar"], p["nodo"], p["x"], p["y"]] for p in puntos
    ]
    print(
        tabulate(
            filas_puntos,
            headers=["Lugar", "Nodo", "X (km)", "Y (km)"],
            tablefmt="github",
        )
    )

    print("\n=== Matriz de distancias euclidianas (km) ===")
    encabezados = ["Nodo"] + [str(i) for i in range(len(puntos))]
    filas_matriz = [
        [i] + [f"{d:.3f}" for d in fila] for i, fila in enumerate(matriz)
    ]
    print(tabulate(filas_matriz, headers=encabezados, tablefmt="github"))


def _validar_ruta_usuario(matriz: list[list[float]]) -> None:
    num_centros = len(matriz) - 1
    ejemplo = " ".join(str(i) for i in range(1, num_centros + 1))
    entrada = input(
        f"\nIngrese la permutación de centros 1..{num_centros} "
        f"(separados por espacio o coma, ej. '{ejemplo}'): "
    ).strip()

    if not entrada:
        print("Operación cancelada.")
        return

    tokens = entrada.replace(",", " ").split()
    try:
        permutacion = [int(t) for t in tokens]
    except ValueError:
        print("Error: ingrese únicamente números enteros.")
        return

    es_valida, mensaje = validar_permutacion(permutacion, num_centros)
    if not es_valida:
        print(f"Ruta INVÁLIDA: {mensaje}")
        return

    costo = calcular_costo_ruta(permutacion, matriz, validar=False)
    ruta_comp = construir_ruta_completa(permutacion)
    print(f"Ruta VÁLIDA: {formatear_ruta(ruta_comp)}")
    print(f"Distancia total L(π): {costo:.6f} km")


def _ejecutar_busqueda_exacta_detalle(matriz: list[list[float]]) -> None:
    print("\n--- RESULTADOS UCS (Búsqueda de Costo Uniforme) ---")
    res_ucs = ejecutar_ucs(matriz)
    if res_ucs["encontrado"]:
        print(f"Costo óptimo (f*)          : {res_ucs['costo']:.6f} km")
        print(f"Ruta óptima                : {formatear_ruta(res_ucs['ruta_completa'])}")
        print(f"Estados expandidos         : {res_ucs['estados_expandidos']}")
        print(f"Tiempo de ejecución        : {res_ucs['tiempo_ms']:.4f} ms")
        print(f"Primeros 5 estados         : {res_ucs['orden_visita'][:5]}")
    else:
        print("UCS no encontró solución.")

    print("\n--- RESULTADOS A* (Heurística basada en MST de Prim) ---")
    res_astar = ejecutar_astar(matriz)
    if res_astar["encontrado"]:
        print(f"Costo óptimo (f*)          : {res_astar['costo']:.6f} km")
        print(f"Ruta óptima                : {formatear_ruta(res_astar['ruta_completa'])}")
        print(f"Estados expandidos         : {res_astar['estados_expandidos']}")
        print(f"Tiempo de ejecución        : {res_astar['tiempo_ms']:.4f} ms")
        print(f"Primeros 5 estados         : {res_astar['orden_visita'][:5]}")
    else:
        print("A* no encontró solución.")

    if res_ucs["encontrado"] and res_astar["encontrado"]:
        dif = abs(res_ucs["costo"] - res_astar["costo"])
        red = (
            (res_ucs["estados_expandidos"] - res_astar["estados_expandidos"])
            / res_ucs["estados_expandidos"]
            * 100.0
        )
        print(
            f"\nVerificación f* UCS vs A*: diferencia = {dif:.2e} km | "
            f"Reducción de estados con MST = {red:.2f}%"
        )


def _menu_interactivo(ruta_csv_inicial: str | None) -> None:
    ruta_csv = ruta_csv_inicial or obtener_ruta_csv_por_defecto()
    puntos = cargar_puntos_csv(ruta_csv)
    matriz = calcular_matriz_distancias(puntos)
    resultados_cache: dict[str, Any] | None = None
    carpeta_base = _carpeta_resultados_base()

    while True:
        print("\n" + "-" * 65)
        print("PROYECTO FINAL IA — OPTIMIZACIÓN DE RUTAS DE DISTRIBUCIÓN")
        print(f"Archivo activo: {ruta_csv} ({len(puntos)} nodos)")
        print("-" * 65)
        print("1. Ver puntos cargados del CSV y matriz de distancias")
        print("2. Validar y evaluar una ruta personalizada")
        print("3. Ejecutar Búsqueda Exacta (UCS y A* con MST)")
        print("4. Ejecutar Experimento Completo (UCS, A*, GWO, PSO, TLBO, GA × 20 semillas)")
        print("5. Exportar tablas del documento (CSV, Markdown, LaTeX, HTML)")
        print("6. Generar y guardar gráficas comparativas")
        print("7. Cargar otro archivo CSV")
        print("0. Salir")

        opcion = input("\nSeleccione una opción: ").strip()

        if opcion == "1":
            _mostrar_datos_y_matriz(puntos, matriz)
        elif opcion == "2":
            _validar_ruta_usuario(matriz)
        elif opcion == "3":
            _ejecutar_busqueda_exacta_detalle(matriz)
        elif opcion == "4":
            print("\nEjecutando experimento completo (20 semillas, 3,000 eval/corrida)...")
            resultados_cache = ejecutar_experimento_completo(ruta_csv=ruta_csv)
            _imprimir_resumen_consola(resultados_cache)
        elif opcion == "5":
            if resultados_cache is None:
                print("\nPrimero ejecutando el experimento completo para calcular las tablas...")
                resultados_cache = ejecutar_experimento_completo(ruta_csv=ruta_csv)

            print("\nFormatos disponibles: csv, md, tex, html, todos")
            fmt_raw = input(
                "Ingrese formatos separados por espacio o coma [todos]: "
            ).strip()
            formatos = (
                [x for x in fmt_raw.replace(",", " ").split() if x]
                if fmt_raw
                else ["todos"]
            )
            try:
                rutas = exportar_tablas(
                    resultados_cache,
                    os.path.join(carpeta_base, "tablas"),
                    formatos=formatos,
                )
                print(f"\nSe exportaron {len(rutas)} archivos:")
                for r in rutas:
                    print(f"  - {r}")
            except ValueError as err:
                print(f"Error: {err}")
        elif opcion == "6":
            if resultados_cache is None:
                print("\nPrimero ejecutando el experimento completo para generar las gráficas...")
                resultados_cache = ejecutar_experimento_completo(ruta_csv=ruta_csv)
            rutas_g = generar_todas_las_graficas(
                resultados_cache,
                os.path.join(carpeta_base, "graficas"),
                mostrar=False,
            )
            print("\nGráficas guardadas en:")
            for r in rutas_g:
                print(f"  - {r}")
        elif opcion == "7":
            nueva_ruta = input("Ingrese la ruta al nuevo archivo CSV: ").strip().strip('"')
            try:
                nuevos_puntos = cargar_puntos_csv(nueva_ruta)
                nueva_matriz = calcular_matriz_distancias(nuevos_puntos)
                ruta_csv = nueva_ruta
                puntos = nuevos_puntos
                matriz = nueva_matriz
                resultados_cache = None
                print(f"Archivo cargado correctamente ({len(puntos)} nodos).")
            except Exception as exc:
                print(f"No se pudo cargar el CSV: {exc}")
        elif opcion == "0":
            print("Saliendo del sistema.")
            break
        else:
            print("Opción no válida. Intente de nuevo.")


def _parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimización inteligente de rutas de distribución (UCS, A*, GWO, PSO, TLBO, GA)."
    )
    parser.add_argument(
        "--todo",
        action="store_true",
        help="Ejecuta el experimento completo, imprime resumen, genera gráficas y exporta tablas.",
    )
    parser.add_argument(
        "--exportar",
        nargs="+",
        choices=[*FORMATOS_VALIDOS, "todos"],
        default=["todos"],
        help="Formatos de exportación de tablas (csv, md, tex, html, todos).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Ruta a un archivo CSV personalizado con columnas Lugar,Nodo,X,Y.",
    )
    parser.add_argument(
        "--sin-graficas",
        action="store_true",
        help="No generar archivos de gráficas en modo --todo.",
    )
    parser.add_argument(
        "--semillas",
        type=int,
        default=20,
        help="Cantidad de ejecuciones por metaheurística (semillas 0..N-1). Por defecto: 20.",
    )
    parser.add_argument(
        "--evaluaciones",
        type=int,
        default=3000,
        help="Presupuesto máximo de evaluaciones de L(π) por corrida. Por defecto: 3000.",
    )
    parser.add_argument(
        "--tiempo-exacto",
        type=float,
        default=None,
        help="Tiempo límite en segundos para UCS y A* (útil en pruebas de escalabilidad).",
    )
    return parser.parse_args()


def main() -> None:
    _configurar_consola()
    args = _parsear_argumentos()
    if args.todo:
        _ejecutar_todo(args)
    else:
        _menu_interactivo(args.csv)


if __name__ == "__main__":
    main()
