"""Módulo de exportación de tablas en formatos CSV, Markdown, LaTeX y HTML."""

from __future__ import annotations

import html
import os
from typing import Any, Sequence

import pandas as pd
from tabulate import tabulate


FORMATOS_VALIDOS = ("csv", "md", "tex", "html")

TITULOS_TABLAS: dict[str, str] = {
    "tabla_1_comparacion_principal": "Tabla 1. Comparación descriptiva de algoritmos (Sección 8)",
    "tabla_2_estadisticas_metaheuristicas": "Tabla 2. Estadísticas de las 20 ejecuciones por metaheurística",
    "tabla_3_busqueda_exacta": "Tabla 3. Comparación de búsqueda exacta (UCS vs. A* con MST)",
    "tabla_4_ejecuciones_detalladas": "Tabla 4. Registro detallado de las ejecuciones (semillas 0–19)",
    "tabla_5_parametros_metaheuristicas": "Tabla 5. Parámetros de las metaheurísticas",
    "tabla_6_coordenadas": "Tabla 6. Coordenadas cartesianas de almacén y centros",
    "tabla_7_matriz_distancias": "Tabla 7. Matriz 9×9 de distancias euclidianas (km)",
}


def _escapar_latex(texto: Any) -> str:
    """Escapa caracteres especiales de LaTeX en valores de celda y encabezados."""
    cadena = str(texto)
    reemplazos = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
        ("->", r"$\rightarrow$"),
        ("—", "---"),
    )
    for origen, destino in reemplazos:
        cadena = cadena.replace(origen, destino)
    return cadena


def _dataframe_a_latex(df: pd.DataFrame, titulo: str, etiqueta: str) -> str:
    """Convierte un DataFrame a una tabla LaTeX profesional con booktabs."""
    cols = list(df.columns)
    alineacion = "l" * len(cols)
    encabezado = " & ".join(_escapar_latex(c) for c in cols) + r" \\"

    lineas_filas: list[str] = []
    for _, fila in df.iterrows():
        valores = [_escapar_latex(fila[c]) for c in cols]
        lineas_filas.append("    " + " & ".join(valores) + r" \\")

    cuerpo = "\n".join(lineas_filas)
    etiqueta_limpia = etiqueta.replace("_", "-")
    return (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        f"  \\caption{{{_escapar_latex(titulo)}}}\n"
        f"  \\label{{tab:{etiqueta_limpia}}}\n"
        f"  \\begin{{tabular}}{{{alineacion}}}\n"
        "    \\toprule\n"
        f"    {encabezado}\n"
        "    \\midrule\n"
        f"{cuerpo}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )


def _dataframe_a_html(df: pd.DataFrame, titulo: str) -> str:
    """Convierte un DataFrame a un bloque HTML estilizado."""
    cols = list(df.columns)
    ths = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    trs: list[str] = []
    for _, fila in df.iterrows():
        tds = "".join(f"<td>{html.escape(str(fila[c]))}</td>" for c in cols)
        trs.append(f"<tr>{tds}</tr>")

    return (
        "<section class=\"tabla-bloque\">\n"
        f"  <h2>{html.escape(titulo)}</h2>\n"
        "  <div class=\"tabla-scroll\">\n"
        "    <table>\n"
        f"      <thead><tr>{ths}</tr></thead>\n"
        f"      <tbody>\n        {''.join(trs)}\n      </tbody>\n"
        "    </table>\n"
        "  </div>\n"
        "</section>\n"
    )


def _envolver_documento_html(cuerpo_html: str, titulo_doc: str) -> str:
    """Envuelve fragmentos de tablas en un documento HTML5 completo."""
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"es\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        f"  <title>{html.escape(titulo_doc)}</title>\n"
        "  <style>\n"
        "    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 28px; color: #0f172a; background: #f8fafc; }\n"
        "    h1 { font-size: 1.5rem; margin-bottom: 18px; color: #1e293b; }\n"
        "    h2 { font-size: 1.1rem; margin: 22px 0 10px; color: #334155; }\n"
        "    .tabla-bloque { background: #ffffff; padding: 16px 20px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 22px; }\n"
        "    .tabla-scroll { overflow-x: auto; }\n"
        "    table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }\n"
        "    th, td { border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; white-space: nowrap; }\n"
        "    thead th { background: #1e3a8a; color: #ffffff; font-weight: 600; }\n"
        "    tbody tr:nth-child(even) { background: #f1f5f9; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"  <h1>{html.escape(titulo_doc)}</h1>\n"
        f"{cuerpo_html}"
        "</body>\n"
        "</html>\n"
    )


def exportar_tablas(
    resultados: dict[str, Any],
    carpeta_salida: str,
    formatos: Sequence[str] = FORMATOS_VALIDOS,
) -> list[str]:
    """Exporta todas las tablas generadas en los formatos solicitados (csv, md, tex, html)."""
    formatos_normalizados = [f.lower().strip() for f in formatos]
    if "todos" in formatos_normalizados:
        formatos_normalizados = list(FORMATOS_VALIDOS)

    invalidos = [f for f in formatos_normalizados if f not in FORMATOS_VALIDOS]
    if invalidos:
        raise ValueError(
            f"Formatos no soportados: {', '.join(invalidos)}. "
            f"Use: {', '.join(FORMATOS_VALIDOS)} o 'todos'."
        )

    tablas: dict[str, pd.DataFrame] = resultados["tablas"]
    archivos_generados: list[str] = []

    subcarpetas = {
        "csv": os.path.join(carpeta_salida, "csv"),
        "md": os.path.join(carpeta_salida, "markdown"),
        "tex": os.path.join(carpeta_salida, "latex"),
        "html": os.path.join(carpeta_salida, "html"),
    }
    for fmt in formatos_normalizados:
        os.makedirs(subcarpetas[fmt], exist_ok=True)

    secciones_md: list[str] = [
        "# Tablas de resultados — Optimización inteligente de rutas de distribución\n"
    ]
    secciones_tex: list[str] = [
        "% Requiere \\usepackage{booktabs} en el preámbulo de LaTeX\n"
    ]
    secciones_html: list[str] = []

    for clave, df in tablas.items():
        titulo = TITULOS_TABLAS.get(clave, clave)

        if "csv" in formatos_normalizados:
            ruta_csv = os.path.join(subcarpetas["csv"], f"{clave}.csv")
            df.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
            archivos_generados.append(ruta_csv)

        if "md" in formatos_normalizados:
            tabla_md = tabulate(
                df, headers="keys", tablefmt="github", showindex=False
            )
            contenido_md = f"## {titulo}\n\n{tabla_md}\n"
            ruta_md = os.path.join(subcarpetas["md"], f"{clave}.md")
            with open(ruta_md, "w", encoding="utf-8") as f:
                f.write(contenido_md)
            archivos_generados.append(ruta_md)
            secciones_md.append(contenido_md)

        if "tex" in formatos_normalizados:
            tabla_tex = _dataframe_a_latex(df, titulo, clave)
            ruta_tex = os.path.join(subcarpetas["tex"], f"{clave}.tex")
            with open(ruta_tex, "w", encoding="utf-8") as f:
                f.write("% Requiere \\usepackage{booktabs}\n" + tabla_tex)
            archivos_generados.append(ruta_tex)
            secciones_tex.append(tabla_tex)

        if "html" in formatos_normalizados:
            bloque_html = _dataframe_a_html(df, titulo)
            ruta_html = os.path.join(subcarpetas["html"], f"{clave}.html")
            with open(ruta_html, "w", encoding="utf-8") as f:
                f.write(_envolver_documento_html(bloque_html, titulo))
            archivos_generados.append(ruta_html)
            secciones_html.append(bloque_html)

    # Archivos consolidados para copiar/pegar todas las tablas juntas
    if "md" in formatos_normalizados:
        ruta_all_md = os.path.join(subcarpetas["md"], "todas_las_tablas.md")
        with open(ruta_all_md, "w", encoding="utf-8") as f:
            f.write("\n".join(secciones_md))
        archivos_generados.append(ruta_all_md)

    if "tex" in formatos_normalizados:
        ruta_all_tex = os.path.join(subcarpetas["tex"], "todas_las_tablas.tex")
        with open(ruta_all_tex, "w", encoding="utf-8") as f:
            f.write("\n".join(secciones_tex))
        archivos_generados.append(ruta_all_tex)

    if "html" in formatos_normalizados:
        ruta_all_html = os.path.join(subcarpetas["html"], "todas_las_tablas.html")
        with open(ruta_all_html, "w", encoding="utf-8") as f:
            f.write(
                _envolver_documento_html(
                    "\n".join(secciones_html),
                    "Tablas comparativas — Proyecto Final de Inteligencia Artificial",
                )
            )
        archivos_generados.append(ruta_all_html)

    return archivos_generados
