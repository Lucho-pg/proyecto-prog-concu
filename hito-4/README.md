\# Hito 4 — Medición y optimización del sistema



\## Objetivo



Medir el rendimiento de un modelo de lenguaje local y analizar cómo cambian los resultados al modificar algunas configuraciones de inferencia.



\## Ruta elegida



\*\*Ruta B — Medición y optimización del sistema.\*\*



Se utiliza `llama.cpp` para ejecutar el modelo Qwen2.5-0.5B-Instruct en una GPU NVIDIA Tesla T4.



\## Experimentos



\- \*\*Configuración base:\*\* cuantización Q4\_K\_M, contexto de 2048 y batch de 512.

\- \*\*Cambio de cuantización:\*\* Q8\_0, manteniendo el contexto y el batch.

\- \*\*Cambio de contexto:\*\* contexto de 4096, manteniendo Q4\_K\_M y batch de 512.



Cada configuración se ejecutó tres veces para comparar los resultados.



\## Archivos



\- `medir\_optimizaciones.py`: script de medición.

\- `resultados.csv`: resultados de las ejecuciones.

\- `grafico\_resultados.png`: comparación gráfica de las medianas.

\- `bitacora.md`: registro de los experimentos.

\- `perfil\_base.nsys-rep`: perfil de Nsight Systems, si está disponible.



\## Estado



Carpeta creada. Pendiente incorporar los archivos generados en Google Colab, revisar los resultados y completar el análisis final.

