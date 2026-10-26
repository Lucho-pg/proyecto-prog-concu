import csv
import os
import re
import subprocess
import time
import statistics
from datetime import datetime

LLAMA_CLI = "/content/llama-cuda/cuda-12.8/llama-cli"
DIR_LIBS = (
    "/usr/local/lib/python3.13/dist-packages/nvidia/cuda_runtime/lib:"
    "/usr/local/lib/python3.13/dist-packages/nvidia/cublas/lib:"
    "/usr/lib64-nvidia:/content/llama-cuda/cuda-12.8"
)
MODELOS = {
    "Q4_K_M": "/content/modelos/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "Q8_0": "/content/modelos/qwen2.5-0.5b-instruct-q8_0.gguf",
}

CONFIGURACIONES = [
    {"experimento": "base", "cuantizacion": "Q4_K_M", "ctx": 2048, "batch": 512},
    {"experimento": "cuantizacion_Q8", "cuantizacion": "Q8_0", "ctx": 2048, "batch": 512},
    {"experimento": "contexto_4096", "cuantizacion": "Q4_K_M", "ctx": 4096, "batch": 512},
]

REPETICIONES = 3
TOKENS = 128
PROMPT = (
    "Explicá la ley de Amdahl en programación concurrente, "
    "indicando qué ocurre cuando aumenta la cantidad de procesadores."
)
SALIDA = "/content/hito4_resultados.csv"

env = os.environ.copy()
env["LD_LIBRARY_PATH"] = DIR_LIBS

def medir_gpu():
    try:
        resultado = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        valores = resultado.stdout.strip().split(",")
        if len(valores) >= 2:
            return int(valores[0].strip()), int(valores[1].strip())
    except Exception:
        pass
    return None, None

def ejecutar(config, repeticion):
    modelo = MODELOS[config["cuantizacion"]]
    comando = [
        LLAMA_CLI,
        "-m", modelo,
        "-ngl", "99",
        "-c", str(config["ctx"]),
        "-b", str(config["batch"]),
        "-n", str(TOKENS),
        "-p", PROMPT,
        "-st",
    ]

    inicio = time.perf_counter()
    proceso = subprocess.Popen(
        comando, env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    max_vram = 0
    max_gpu = 0
    while proceso.poll() is None:
        vram, uso_gpu = medir_gpu()
        if vram is not None:
            max_vram = max(max_vram, vram)
            max_gpu = max(max_gpu, uso_gpu)
        time.sleep(0.25)

    salida, _ = proceso.communicate()
    duracion = time.perf_counter() - inicio

    coincidencias = re.findall(
        r"Generation:\s*([\d.]+)\s*t/s", salida
    )
    tokens_s = float(coincidencias[-1]) if coincidencias else None

    fila = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "experimento": config["experimento"],
        "cuantizacion": config["cuantizacion"],
        "ctx": config["ctx"],
        "batch": config["batch"],
        "repeticion": repeticion,
        "tokens_s": tokens_s,
        "vram_max_mib": max_vram,
        "uso_gpu_max_pct": max_gpu,
        "duracion_total_s": round(duracion, 2),
        "codigo_salida": proceso.returncode,
    }

    if proceso.returncode != 0 or tokens_s is None:
        print("\nError o falta de estadísticas. Última salida:")
        print(salida[-2500:])
    print(fila)
    return fila

def main():
    if not os.path.isfile(LLAMA_CLI):
        raise FileNotFoundError(LLAMA_CLI)
    for modelo in MODELOS.values():
        if not os.path.isfile(modelo):
            raise FileNotFoundError(modelo)

    filas = []
    for config in CONFIGURACIONES:
        print(f"\n=== {config['experimento']} | {config['cuantizacion']} "
              f"| ctx={config['ctx']} | batch={config['batch']} ===")
        for repeticion in range(1, REPETICIONES + 1):
            filas.append(ejecutar(config, repeticion))

    campos = list(filas[0].keys())
    with open(SALIDA, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)

    print("\n=== RESUMEN: MEDIANA DE TOKENS/S ===")
    for config in CONFIGURACIONES:
        valores = [
            f["tokens_s"] for f in filas
            if f["experimento"] == config["experimento"]
            and f["tokens_s"] is not None
        ]
        if valores:
            print(f"{config['experimento']}: "
                  f"{statistics.median(valores):.2f} tokens/s")
    print(f"\nCSV guardado en: {SALIDA}")

if __name__ == "__main__":
    main()
