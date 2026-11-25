## Tabla 5. Parámetros de las metaheurísticas

| Algoritmo   | Parámetro                                      | Valor                                                                               | Descripción                                                                           |
|-------------|------------------------------------------------|-------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| Comunes     | Población / Dimensión / Presupuesto / Semillas | 30 individuos; D = número de centros (8); 3,000 evaluaciones de L(π); semillas 0–19 | Representación por claves aleatorias x ∈ [0, 1]^D decodificadas por orden ascendente. |
| GWO         | Líderes y parámetro a                          | α, β, δ (top-3); a lineal de 2.0 → 0.0; límites [0, 1]                              | Actualización guiada por los tres mejores lobos con vectores aleatorios A y C.        |
| PSO         | Inercia y aceleración                          | w: 0.9 → 0.4; c1 = 1.5; c2 = 1.5; v_max = 0.2; límites [0, 1]                       | Velocidad acotada en [-v_max, v_max] con memoria personal (pbest) y global (gbest).   |
| TLBO        | Factor de enseñanza y fases                    | TF ∈ {1, 2}; fase profesor + fase estudiante; límites [0, 1]                        | Sin parámetros de ajuste específicos adicionales; aceptación codiciosa de mejoras.    |
| GA          | Selección, cruce, mutación y élite             | Torneo k=3; cruce aritmético pc=0.85; mutación gaussiana pm=0.2, σ=0.1; elitismo=2  | Preserva los 2 mejores sin re-evaluar y aplica recorte a [0, 1] tras mutación.        |
