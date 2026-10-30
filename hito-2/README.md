
# Hito 2 - Kernels CUDA optimizados

## Objetivo

En este hito se realiza el port del problema de suma de un vector del Hito 1, originalmente implementado sobre CPU, hacia una implementación utilizando CUDA sobre GPU.

Se desarrollan y comparan dos versiones:

* **Kernel ingenuo:** utiliza `atomicAdd` sobre un único acumulador global.
* **Kernel optimizado:** utiliza una reducción paralela mediante memoria compartida (`shared memory`).

El objetivo es analizar el impacto de una técnica de optimización CUDA sobre el rendimiento de la suma de un vector de gran tamaño.

---

## Problema

El problema consiste en calcular la suma de un vector de `N` elementos de tipo `float`, donde todos los elementos tienen valor `1.0`.

Para las pruebas se utilizó:

```text
N = 16.777.216
```

El resultado esperado es:

```text
16.777.216
```

La misma entrada se utilizó para comparar la implementación de CPU del Hito 1 con las dos implementaciones CUDA.

---

## Implementación ingenua

El archivo `kernel_ingenuo.cu` utiliza un único acumulador ubicado en memoria global.

Cada hilo procesa un elemento del vector y realiza:

```cpp
atomicAdd(out, x[i]);
```

De esta manera, todos los hilos intentan actualizar la misma posición de memoria.

La implementación es correcta, pero las operaciones atómicas generan una fuerte serialización, ya que los hilos deben competir por el acceso al acumulador global.

Por este motivo se utiliza como **baseline** para comparar posteriormente la versión optimizada.

### Correctitud

Para `N = 16.777.216` se obtuvo:

```text
ref = 16.777.216,0
gpu = 16.777.216,0
rel = 0,000e+00
ok
```

---

## Implementación optimizada

El archivo `kernel_optimizado.cu` implementa una reducción paralela utilizando memoria compartida.

Cada bloque carga los elementos correspondientes del vector en `shared memory`:

```cpp
s[tid] = (i < n) ? x[i] : 0.f;
```

Luego los hilos del bloque realizan una reducción en forma de árbol:

```cpp
for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
        s[tid] += s[tid + stride];
    }

    __syncthreads();
}
```

Finalmente, el hilo `0` de cada bloque guarda el resultado parcial:

```cpp
if (tid == 0) {
    parcial[blockIdx.x] = s[0];
}
```

Los resultados parciales obtenidos por cada bloque son sumados posteriormente por la CPU.

De esta manera, la implementación realiza una **reducción paralela real**, evitando que todos los hilos actualicen directamente un único acumulador global.

### Técnica de optimización utilizada

La técnica principal utilizada es:

* **Memoria compartida (`shared memory`)**
* **Reducción paralela en árbol**
* Acceso consecutivo a los elementos del vector, favoreciendo lecturas coalescentes.

El uso de memoria compartida permite realizar las operaciones intermedias dentro de cada bloque sin depender continuamente de operaciones atómicas sobre memoria global.

### Correctitud

Para `N = 16.777.216` se obtuvo:

```text
ref = 16.777.216,0
gpu = 16.777.216,0
rel = 0,000e+00
ok
```

---

## Metodología

Para comparar las implementaciones se realizaron **5 ejecuciones** de cada versión y se utilizó la **mediana** de los tiempos obtenidos.

La medición de los kernels CUDA se realizó mediante `cudaEvent`, midiendo únicamente el tiempo de ejecución del kernel.

Para el profiling se utilizó **NVIDIA Nsight Compute**.

---

## Hardware y entorno

Las pruebas CUDA fueron realizadas utilizando Google Colab con una GPU NVIDIA Tesla T4.

| Característica     | Valor           |
| ------------------ | --------------- |
| GPU                | NVIDIA Tesla T4 |
| Compute Capability | 7.5             |
| Memoria GPU        | 15 GB           |
| CUDA Toolkit       | 12.8            |
| Nsight Compute     | 2025.1.1        |
| Elementos          | 16.777.216      |
| Threads por bloque | 256             |

La implementación de CPU utilizada como referencia corresponde al Hito 1.

---

## Resultados

### Comparación CPU vs GPU

| Implementación   | Tiempo mediano |
| ---------------- | -------------: |
| CPU - Hito 1     |     284,463 ms |
| GPU - Ingenua    |      50,946 ms |
| GPU - Optimizada |       0,847 ms |

A partir de estos resultados:

* La GPU ingenua fue aproximadamente **5,58×** más rápida que la implementación secuencial de CPU.
* La GPU optimizada fue aproximadamente **335,45×** más rápida que la implementación secuencial de CPU.
* La versión optimizada fue aproximadamente **60,15×** más rápida que la versión ingenua.

El speedup entre las dos implementaciones CUDA se calculó como:

```text
50,946 / 0,847 ≈ 60,15×
```

---

## Profiling con Nsight Compute

Ambos kernels fueron perfilados utilizando NVIDIA Nsight Compute sobre la NVIDIA Tesla T4.

Los principales indicadores obtenidos fueron:

| Métrica                |   Ingenua | Optimizada |
| ---------------------- | --------: | ---------: |
| Memory Throughput      |    2,06 % |    76,27 % |
| Compute Throughput     |    1,22 % |    76,27 % |
| Memoria                | 1,62 GB/s | 58,28 GB/s |
| Achieved Occupancy     |   87,71 % |    90,38 % |
| Theoretical Occupancy  |     100 % |      100 % |
| Shared Memory dinámica |      0 KB |    1,02 KB |

Los resultados completos utilizados para esta comparación se encuentran en:

```text
profiling.csv
```

---

## Análisis del kernel ingenuo

El kernel ingenuo presenta una ocupación relativamente alta, con un **87,71 %** de ocupación alcanzada.

Sin embargo, el aprovechamiento de los recursos de la GPU es bajo:

```text
Memory Throughput = 2,06 %
Compute Throughput = 1,22 %
```

Además, Nsight Compute mostró aproximadamente un **99,89 % de ciclos sin warps elegibles**.

Esto es consistente con el uso de `atomicAdd` sobre un único acumulador global. Aunque existen muchos hilos ejecutándose, las actualizaciones sobre la misma posición de memoria generan una fuerte serialización y provocan esperas.

Por lo tanto, una ocupación elevada no implica necesariamente un buen aprovechamiento de los recursos de la GPU.

---

## Análisis del kernel optimizado

La versión optimizada alcanza una ocupación de:

```text
90,38 %
```

pero, a diferencia de la versión ingenua, presenta un aprovechamiento mucho mayor de los recursos:

```text
Memory Throughput = 76,27 %
Compute Throughput = 76,27 %
Memoria = 58,28 GB/s
```

La reducción se realiza dentro de cada bloque utilizando memoria compartida.

Esto permite que los hilos colaboren para obtener una suma parcial antes de escribir el resultado a memoria global.

De esta forma se reduce considerablemente la cantidad de operaciones atómicas y se aprovecha mejor la capacidad de procesamiento de la GPU.

---

## Comparación

Los resultados muestran una diferencia significativa entre ambas implementaciones CUDA.

La versión ingenua utiliza un único acumulador global, provocando una fuerte competencia entre los hilos.

En cambio, la versión optimizada divide el problema en reducciones parciales por bloque y utiliza memoria compartida para realizar las operaciones intermedias.

Esto se refleja tanto en las métricas obtenidas mediante Nsight Compute como en los tiempos de ejecución.

La diferencia de rendimiento medida fue:

```text
GPU ingenua       = 50,946 ms
GPU optimizada    = 0,847 ms

Speedup ≈ 60,15×
```

---

## Consideración sobre los tiempos de Nsight Compute

Los tiempos mostrados durante el profiling de Nsight Compute no fueron utilizados como benchmark principal.

Nsight Compute realiza múltiples pasadas sobre los kernels para obtener diferentes métricas. Esto introduce un overhead considerable.

Por este motivo, para la comparación de rendimiento se utilizaron las mediciones realizadas mediante `cudaEvent`.

El profiling se utilizó para analizar el comportamiento interno de los kernels y no para reemplazar las mediciones normales de ejecución.

---

## Conclusiones

En este hito se realizó la migración del problema de suma de vectores desde CPU hacia GPU utilizando CUDA.

La implementación ingenua permitió comprobar el funcionamiento correcto del algoritmo utilizando `atomicAdd`, pero presentó una fuerte serialización debido al uso de un único acumulador global.

La implementación optimizada utilizó memoria compartida y una reducción paralela en árbol, permitiendo distribuir el trabajo entre los hilos de cada bloque y reducir la dependencia de operaciones atómicas.

Para `N = 16.777.216`, la implementación optimizada obtuvo un tiempo mediano de **0,847 ms**, frente a **50,946 ms** de la versión ingenua.

Esto representa una mejora aproximada de **60,15×** entre ambas implementaciones CUDA.

Al compararla con la implementación secuencial de CPU del Hito 1, la versión optimizada obtuvo una mejora aproximada de **335,45×**.

El profiling realizado con Nsight Compute también mostró un aumento considerable en el aprovechamiento de los recursos de memoria y cómputo de la GPU.

Este hito permitió comprobar cómo una estrategia de reducción paralela junto con el uso de memoria compartida puede mejorar significativamente el rendimiento de un algoritmo de suma sobre GPU.
