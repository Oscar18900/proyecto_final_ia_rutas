# Proyecto Final de Inteligencia Artificial: Optimización Inteligente de Rutas de Distribución

Sistema en Python para optimizar la ruta de distribución desde un almacén (`Nodo 0`) hacia ocho centros comunitarios (`Nodos 1..8`) y retorno al almacén, comparando búsqueda exacta (**UCS** y **A\*** con heurística basada en **MST de Prim**) y cuatro metaheurísticas basadas en claves aleatorias (**GWO**, **PSO**, **TLBO** y **GA**).

## Estructura del proyecto

```text
proyecto_final_ia_rutas/
├── data/
│   └── puntos_distribucion.csv   # Coordenadas de los 9 nodos (Almacén + Centros 1..8)
├── src/
│   ├── __init__.py
│   ├── datos.py                  # Lectura de CSV y matriz 9x9 de distancias euclidianas
│   ├── rutas.py                  # Validación de rutas, cálculo de L(π) y mapas de rutas
│   ├── busqueda.py               # UCS y A* con MST de Prim (basados en prácticas de lab)
│   ├── metaheuristicas.py        # GWO, PSO, TLBO y GA con claves aleatorias en [0,1]
│   ├── experimentos.py           # Experimento completo (20 semillas), métricas y gráficas
│   └── exportar.py               # Exportación de tablas en CSV, Markdown, LaTeX y HTML
├── tests/
│   └── test_sistema.py           # Pruebas unitarias y verificación exacta vs fuerza bruta
├── resultados/                   # Generado al ejecutar (tablas y gráficas)
├── main.py                       # Interfaz CLI y menú interactivo
└── requirements.txt
```

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

### 1. Menú interactivo
```bash
python main.py
```

### 2. Ejecución automática completa (UCS, A*, 20 corridas por metaheurística, gráficas y exportación)
```bash
python main.py --todo --exportar todos
```

Opciones adicionales de línea de comandos:
- `--csv RUTA`: usar otro archivo CSV con columnas `Lugar,Nodo,X,Y`.
- `--exportar csv md tex html`: elegir uno o más formatos de exportación (o `todos`).
- `--sin-graficas`: omitir la generación de archivos PNG en `resultados/graficas/`.
- `--semillas 20`: número de semillas a ejecutar (`0..N-1`, por defecto `20`).
- `--evaluaciones 3000`: presupuesto máximo de evaluaciones de `L(π)` por corrida (por defecto `3000`).

### 3. Pruebas unitarias
```bash
python -m unittest discover -s tests -v
```
