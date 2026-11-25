## Tabla 3. Comparación de búsqueda exacta (UCS vs. A* con MST)

| Algoritmo   | Estado            |   f* (km) | Ruta óptima                                    |   Estados expandidos |   Tiempo (s) |   Tiempo (ms) |   Reducción de estados vs UCS (%) |
|-------------|-------------------|-----------|------------------------------------------------|----------------------|--------------|---------------|-----------------------------------|
| UCS         | Óptimo encontrado |   39.5893 | 0 -> 5 -> 4 -> 3 -> 8 -> 2 -> 7 -> 1 -> 6 -> 0 |                 1026 |     0.004523 |        4.5226 |                              0    |
| A*          | Óptimo encontrado |   39.5893 | 0 -> 5 -> 4 -> 3 -> 8 -> 2 -> 7 -> 1 -> 6 -> 0 |                   17 |     0.000822 |        0.8224 |                             98.34 |
