"""Pruebas unitarias y verificación exacta del sistema de optimización de rutas."""

from __future__ import annotations

import itertools
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.busqueda import ejecutar_astar, ejecutar_ucs, heuristica_mst, mst_prim
from src.datos import calcular_matriz_distancias, cargar_puntos_csv
from src.experimentos import calcular_brecha_optimalidad, ejecutar_experimento_completo
from src.exportar import exportar_tablas
from src.metaheuristicas import ALGORITMOS_METAHEURISTICOS
from src.rutas import (
    calcular_costo_ruta,
    decodificar_claves_aleatorias,
    validar_permutacion,
    validar_ruta_completa,
)


class TestSistemaRutas(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.puntos = cargar_puntos_csv()
        cls.matriz = calcular_matriz_distancias(cls.puntos)
        cls.num_centros = len(cls.puntos) - 1

        # Óptimo global exacto por enumeración exhaustiva (8! = 40,320)
        mejor_dist = float("inf")
        for perm in itertools.permutations(range(1, cls.num_centros + 1)):
            dist = calcular_costo_ruta(perm, cls.matriz, validar=False)
            if dist < mejor_dist:
                mejor_dist = dist
        cls.optimo_fuerza_bruta = mejor_dist

    def test_carga_csv_y_matriz_9x9(self) -> None:
        self.assertEqual(len(self.puntos), 9)
        self.assertEqual(self.puntos[0]["lugar"], "Almacén")
        self.assertEqual((self.puntos[0]["x"], self.puntos[0]["y"]), (5.0, 5.0))
        self.assertEqual((self.puntos[8]["x"], self.puntos[8]["y"]), (13.0, 10.0))

        self.assertEqual(len(self.matriz), 9)
        for i in range(9):
            self.assertEqual(len(self.matriz[i]), 9)
            self.assertEqual(self.matriz[i][i], 0.0)
            for j in range(9):
                self.assertAlmostEqual(self.matriz[i][j], self.matriz[j][i], places=12)

    def test_validacion_rutas_y_decodificacion(self) -> None:
        ok, _ = validar_permutacion([1, 2, 3, 4, 5, 6, 7, 8], 8)
        self.assertTrue(ok)
        ok_comp, _ = validar_ruta_completa([0, 8, 7, 6, 5, 4, 3, 2, 1, 0], 8)
        self.assertTrue(ok_comp)

        # Casos inválidos
        self.assertFalse(validar_permutacion([1, 1, 3, 4, 5, 6, 7, 8], 8)[0])
        self.assertFalse(validar_permutacion([1, 2, 3, 4, 5, 6, 7], 8)[0])
        self.assertFalse(validar_permutacion([0, 1, 2, 3, 4, 5, 6, 7], 8)[0])
        self.assertFalse(validar_ruta_completa([1, 2, 3, 4, 5, 6, 7, 8, 0], 8)[0])

        # Empates en claves aleatorias deben desempatar por índice ascendente
        claves_empate = [0.5, 0.2, 0.5, 0.2]
        self.assertEqual(decodificar_claves_aleatorias(claves_empate), (2, 4, 1, 3))

    def test_ucs_y_astar_coinciden_con_fuerza_bruta(self) -> None:
        res_ucs = ejecutar_ucs(self.matriz)
        res_astar = ejecutar_astar(self.matriz)

        self.assertTrue(res_ucs["encontrado"])
        self.assertTrue(res_astar["encontrado"])
        self.assertFalse(res_ucs["inconclusa"])
        self.assertFalse(res_astar["inconclusa"])

        self.assertAlmostEqual(
            res_ucs["costo"], self.optimo_fuerza_bruta, places=9
        )
        self.assertAlmostEqual(
            res_astar["costo"], self.optimo_fuerza_bruta, places=9
        )
        self.assertAlmostEqual(res_ucs["costo"], res_astar["costo"], places=9)

        self.assertTrue(
            validar_ruta_completa(res_ucs["ruta_completa"], self.num_centros)[0]
        )
        self.assertTrue(
            validar_ruta_completa(res_astar["ruta_completa"], self.num_centros)[0]
        )
        self.assertLess(
            res_astar["estados_expandidos"], res_ucs["estados_expandidos"]
        )

    def test_admisibilidad_mst_y_propiedades(self) -> None:
        self.assertEqual(mst_prim([], self.matriz), 0.0)
        self.assertEqual(mst_prim([3], self.matriz), 0.0)
        # Desde el inicio (0, 0), h((0, 0)) debe ser <= f*
        h_inicial = heuristica_mst((0, 0), self.matriz, self.num_centros)
        self.assertLessEqual(h_inicial, self.optimo_fuerza_bruta + 1e-9)
        # Cuando U está vacío en el nodo 4, h((4, 255)) == d(4, 0)
        self.assertAlmostEqual(
            heuristica_mst((4, (1 << 8) - 1), self.matriz, self.num_centros),
            self.matriz[4][0],
            places=12,
        )

    def test_metaheuristicas_presupuesto_y_determinismo(self) -> None:
        for nombre, fn in ALGORITMOS_METAHEURISTICOS.items():
            res1 = fn(self.matriz, semilla=7, tam_poblacion=30, max_evaluaciones=3000)
            res2 = fn(self.matriz, semilla=7, tam_poblacion=30, max_evaluaciones=3000)

            self.assertEqual(res1["evaluaciones"], 3000, msg=nombre)
            self.assertEqual(len(res1["historial"]), 3000, msg=nombre)
            self.assertTrue(
                validar_ruta_completa(res1["mejor_ruta_completa"], self.num_centros)[0],
                msg=nombre,
            )
            self.assertGreaterEqual(
                res1["mejor_distancia"], self.optimo_fuerza_bruta - 1e-9, msg=nombre
            )
            self.assertAlmostEqual(
                res1["mejor_distancia"], res2["mejor_distancia"], places=12, msg=nombre
            )
            self.assertEqual(
                res1["mejor_permutacion"], res2["mejor_permutacion"], msg=nombre
            )

    def test_brecha_y_exportacion_tablas(self) -> None:
        self.assertEqual(
            calcular_brecha_optimalidad(self.optimo_fuerza_bruta, self.optimo_fuerza_bruta),
            0.0,
        )
        self.assertAlmostEqual(calcular_brecha_optimalidad(110.0, 100.0), 10.0, places=9)

        resultados = ejecutar_experimento_completo(
            semillas=range(2), tam_poblacion=30, max_evaluaciones=300
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            archivos = exportar_tablas(
                resultados, tmpdir, formatos=("csv", "md", "tex", "html")
            )
            # 7 tablas * 4 formatos + 3 consolidados (md, tex, html) = 31 archivos
            self.assertEqual(len(archivos), 31)
            for ruta in archivos:
                self.assertTrue(os.path.isfile(ruta))
                self.assertGreater(os.path.getsize(ruta), 0)


if __name__ == "__main__":
    unittest.main()
