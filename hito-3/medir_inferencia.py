from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import threading
from typing import Any


LLAMA_CLI = "/content/llama-cuda/cuda-12.8/llama-cli"
LD_LIBRARY_PATH = "/usr/lib64-nvidia:/content/llama-cuda/cuda-12.8"

MODELOS = {
    "q4": "/content/modelos/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "q8": "/content/modelos/qwen2.5-0.5b-instruct-q8_0.gguf",
}


def medir_vram(pid: int, resultado: dict[str, Any]) -> None:
    """Consulta periódicamente la VRAM utilizada por el proceso."""
    maxima_vram = 0

    while True:
        try:
            salida = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )

            for linea in salida.strip().splitlines():
                if not linea.strip():
                    continue

                partes = [p.strip() for p in linea.split(",")]

                if len(partes) >= 2:
                    try:
                        proceso = int(partes[0])
                    except ValueError:
                        continue

                    if proceso == pid:
                        memoria = partes[1]

                        match = re.search(r"([0-9]+)", memoria)

                        if match:
                            maxima_vram = max(
                                maxima_vram,
                                int(match.group(1)),
                            )

            resultado["vram_mib"] = maxima_vram

        except Exception:
            pass

        if resultado.get("terminado"):
            break


def medir_llama(
    model: str,
    prompt: str,
    cuantizacion: str,
    batch_size: int,
    ctx: int,
    max_tokens: int,
) -> dict[str, Any]:

    cmd = [
        LLAMA_CLI,
        "-m", model,
        "-ngl", "99",
        "-c", str(ctx),
        "-b", str(batch_size),
        "-n", str(max_tokens),
        "-p", prompt,
        "-st",
    ]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH

    resultado_vram: dict[str, Any] = {
        "vram_mib": 0,
        "terminado": False,
    }

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    hilo_vram = threading.Thread(
        target=medir_vram,
        args=(proc.pid, resultado_vram),
        daemon=True,
    )

    hilo_vram.start()

    stdout, stderr = proc.communicate(timeout=600)

    resultado_vram["terminado"] = True
    hilo_vram.join(timeout=1)

    output = (stdout or "") + "\n" + (stderr or "")

    match = re.search(
        r"Generation:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s",
        output,
    )

    tokens_per_s = float(match.group(1)) if match else None

    return {
        "runtime": "llama.cpp",
        "model": os.path.basename(model),
        "cuantizacion": cuantizacion,
        "batch_size": batch_size,
        "ctx": ctx,
        "vram_mib": resultado_vram["vram_mib"],
        "tokens_per_s": tokens_per_s,
        "returncode": proc.returncode,
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Mide tokens/s y VRAM de un LLM local con llama.cpp."
    )

    parser.add_argument(
        "--out",
        default="/content/hito-3/resultados.csv",
    )

    parser.add_argument(
        "--repeticiones",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    prompt = "Explicá Amdahl en dos oraciones."

    configs = [
        {
            "cuantizacion": "q4_k_m",
            "model": MODELOS["q4"],
            "batch_size": 512,
            "ctx": 2048,
        },
        {
            "cuantizacion": "q8_0",
            "model": MODELOS["q8"],
            "batch_size": 512,
            "ctx": 2048,
        },
        {
            "cuantizacion": "q4_k_m",
            "model": MODELOS["q4"],
            "batch_size": 512,
            "ctx": 4096,
        },
    ]

    rows = []

    for config in configs:

        print()
        print("=" * 60)
        print(
            f"Configuración: {config['cuantizacion']} "
            f"| batch={config['batch_size']} "
            f"| ctx={config['ctx']}"
        )
        print("=" * 60)

        for rep in range(1, args.repeticiones + 1):

            print(f"Repetición {rep}/{args.repeticiones}...")

            row = medir_llama(
                model=config["model"],
                prompt=prompt,
                cuantizacion=config["cuantizacion"],
                batch_size=config["batch_size"],
                ctx=config["ctx"],
                max_tokens=64,
            )

            row["repeticion"] = rep
            rows.append(row)

            print(
                f"Tokens/s: {row['tokens_per_s']} "
                f"| VRAM: {row['vram_mib']} MiB"
            )

    fieldnames = [
        "runtime",
        "model",
        "cuantizacion",
        "batch_size",
        "ctx",
        "repeticion",
        "tokens_per_s",
        "vram_mib",
        "returncode",
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"CSV generado: {args.out}")


if __name__ == "__main__":
    main()
