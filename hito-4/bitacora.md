# Hito 4 — Ruta B: medición y optimización del sistema

## Entorno
- Google Colab con GPU NVIDIA Tesla T4.
- Modelo Qwen2.5-0.5B-Instruct en formato GGUF.
- Ejecución mediante llama.cpp con descarga de capas a GPU (`-ngl 99`).
- Batch: 512. Generación: 128 tokens en las mediciones.
- Perfilado con Nsight Systems; el perfil guardado corresponde a una ejecución separada.

## Resultados

| Experimento | Repeticiones válidas | Mediana (tokens/s) |
|---|---:|---:|
| base | 3 | 225.00 |
| cuantizacion_Q8 | 3 | 232.30 |
| contexto_4096 | 3 | 219.40 |

## Interpretación preliminar
- La configuración base usa Q4_K_M con contexto 2048.
- La prueba Q8_0 cambia la cuantización y permite observar velocidad y uso de VRAM.
- La prueba contexto_4096 cambia el contexto a 4096 manteniendo Q4_K_M.
- Las mediciones varían entre repeticiones; se deben conservar y analizar todos los valores.
- El uso máximo de GPU y la VRAM son máximos muestreados periódicamente por el script, no picos garantizados.
- El perfilado puede alterar los tiempos; no se deben mezclar sus velocidades con las mediciones normales.

## Archivos
- `medir_optimizaciones.py`: script de medición.
- `resultados.csv`: observaciones de las nueve ejecuciones.
- `grafico_resultados.png`: gráfico de medianas.
- `perfil_base.nsys-rep`: perfil de Nsight Systems, si estuvo disponible.

## Reproducibilidad
El script original utiliza rutas específicas de Google Colab y requiere el ejecutable llama.cpp, las bibliotecas CUDA y los dos modelos GGUF. Ajustar las rutas si se ejecuta en otro entorno.

## Pendiente
- Incorporar hardware y resultados al README general del repositorio.
- Revisar la variabilidad del experimento contexto_4096.
- Integrar este directorio con los Hitos 1, 2 y 3 y completar el informe consolidado.
