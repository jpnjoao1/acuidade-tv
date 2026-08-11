#!/usr/bin/env python3
"""Converte um logotipo raster (PNG/JPG) em curvas vetoriais.

Gera dois arquivos a partir da mesma geometria tracada:
  - <saida>.svg  : SVG com paths (curvas de Bezier), pronto para edicao
  - <saida>.ai   : arquivo PDF-compatible, aberto nativamente pelo Illustrator

Uso:
    python3 tools/vetorizar.py entrada.png saida [--limiar 128] [--cor "#111111"]
"""

import argparse
import sys
import zlib

import numpy as np
import potrace
from PIL import Image


# --------------------------------------------------------------------------
# Tracado
# --------------------------------------------------------------------------

def carregar_bitmap(caminho, limiar):
    """Le a imagem, achata sobre branco e devolve um bitmap booleano.

    True = tinta (pixel escuro), False = fundo.
    """
    img = Image.open(caminho)

    # Achata transparencia sobre branco para nao tracar o canal alfa como forma.
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        fundo = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(fundo, img)

    cinza = np.asarray(img.convert("L"), dtype=np.uint8)
    return cinza < limiar, img.size


def tracar(bitmap, alphamax=1.0, opttolerance=0.2, turdsize=2):
    """Roda o potrace e devolve as curvas encontradas.

    `potrace.Bitmap` inverte o array no construtor, entao passamos o
    complemento para que a tinta continue sendo o primeiro plano.
    """
    return potrace.Bitmap(~bitmap).trace(
        turdsize=turdsize,
        alphamax=alphamax,
        opticurve=True,
        opttolerance=opttolerance,
    )


# --------------------------------------------------------------------------
# Geracao de path
# --------------------------------------------------------------------------

def curvas_para_path(caminhos, flip_y=False, altura=0):
    """Converte as curvas do potrace em um unico atributo `d` de path.

    `flip_y` inverte o eixo vertical, necessario para o PDF (origem embaixo).
    """
    def pt(p):
        x, y = (p.x, p.y) if hasattr(p, "x") else (p[0], p[1])
        return (x, altura - y) if flip_y else (x, y)

    partes = []
    for curva in caminhos:
        x, y = pt(curva.start_point)
        partes.append(f"M{x:.2f} {y:.2f}")
        for seg in curva:
            if seg.is_corner:
                cx, cy = pt(seg.c)
                ex, ey = pt(seg.end_point)
                partes.append(f"L{cx:.2f} {cy:.2f}L{ex:.2f} {ey:.2f}")
            else:
                c1x, c1y = pt(seg.c1)
                c2x, c2y = pt(seg.c2)
                ex, ey = pt(seg.end_point)
                partes.append(
                    f"C{c1x:.2f} {c1y:.2f} {c2x:.2f} {c2y:.2f} {ex:.2f} {ey:.2f}"
                )
        partes.append("Z")
    return "".join(partes)


# --------------------------------------------------------------------------
# Saida SVG
# --------------------------------------------------------------------------

def gravar_svg(destino, d, largura, altura, cor):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {largura} {altura}" width="{largura}" height="{altura}">\n'
        f'  <path fill="{cor}" fill-rule="evenodd" d="{d}"/>\n'
        f"</svg>\n"
    )
    with open(destino, "w", encoding="utf-8") as f:
        f.write(svg)


# --------------------------------------------------------------------------
# Saida .ai (PDF-compatible)
# --------------------------------------------------------------------------

def path_svg_para_pdf(d):
    """Reescreve o `d` do SVG nos operadores de path do PDF."""
    saida = []
    i = 0
    n = len(d)
    while i < n:
        cmd = d[i]
        i += 1
        j = i
        while j < n and d[j] not in "MLCZ":
            j += 1
        nums = d[i:j].replace(",", " ").split()
        i = j
        if cmd == "M":
            saida.append(f"{nums[0]} {nums[1]} m")
        elif cmd == "L":
            saida.append(f"{nums[0]} {nums[1]} l")
        elif cmd == "C":
            saida.append(" ".join(nums[:6]) + " c")
        elif cmd == "Z":
            saida.append("h")
    return "\n".join(saida)


def gravar_ai(destino, d, largura, altura, rgb):
    """Escreve um PDF 1.4 minimo com o path preenchido.

    Illustrator abre e edita arquivos .ai gravados neste formato
    ("PDF Compatible File"), com as curvas ja editaveis.
    """
    r, g, b = (c / 255 for c in rgb)
    conteudo = (
        f"{r:.4f} {g:.4f} {b:.4f} rg\n"
        f"{path_svg_para_pdf(d)}\n"
        f"f*\n"
    ).encode("latin-1")
    fluxo = zlib.compress(conteudo)

    objetos = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".encode("latin-1"),
        (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {largura} {altura}] "
            f"/Contents 4 0 R /Resources << >> >>"
        ).encode("latin-1"),
        (
            f"<< /Length {len(fluxo)} /Filter /FlateDecode >>".encode("latin-1")
            + b"\nstream\n"
            + fluxo
            + b"\nendstream"
        ),
    ]

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    deslocamentos = []
    for num, obj in enumerate(objetos, start=1):
        deslocamentos.append(len(pdf))
        pdf += f"{num} 0 obj\n".encode("latin-1") + obj + b"\nendobj\n"

    inicio_xref = len(pdf)
    pdf += f"xref\n0 {len(objetos) + 1}\n".encode("latin-1")
    pdf += b"0000000000 65535 f \n"
    for off in deslocamentos:
        pdf += f"{off:010d} 00000 n \n".encode("latin-1")
    pdf += (
        f"trailer\n<< /Size {len(objetos) + 1} /Root 1 0 R >>\n"
        f"startxref\n{inicio_xref}\n%%EOF\n"
    ).encode("latin-1")

    with open(destino, "wb") as f:
        f.write(pdf)


# --------------------------------------------------------------------------

def hex_para_rgb(valor):
    valor = valor.lstrip("#")
    return tuple(int(valor[i:i + 2], 16) for i in (0, 2, 4))


def main():
    ap = argparse.ArgumentParser(description="Vetoriza um logotipo em curvas.")
    ap.add_argument("entrada", help="imagem de origem (PNG, JPG...)")
    ap.add_argument("saida", help="prefixo dos arquivos gerados (sem extensao)")
    ap.add_argument("--limiar", type=int, default=128,
                    help="corte de luminancia 0-255 (padrao: 128)")
    ap.add_argument("--cor", default="#111111",
                    help="cor de preenchimento das curvas (padrao: #111111)")
    ap.add_argument("--suavizacao", type=float, default=1.0,
                    help="alphamax do potrace, 0 = cantos vivos (padrao: 1.0)")
    ap.add_argument("--tolerancia", type=float, default=0.2,
                    help="opttolerance do potrace (padrao: 0.2)")
    ap.add_argument("--ruido", type=int, default=2,
                    help="descarta manchas menores que N pixels (padrao: 2)")
    args = ap.parse_args()

    bitmap, (largura, altura) = carregar_bitmap(args.entrada, args.limiar)
    if not bitmap.any():
        sys.exit("Nenhum pixel escuro encontrado — ajuste --limiar.")

    caminhos = tracar(bitmap, args.suavizacao, args.tolerancia, args.ruido)

    gravar_svg(f"{args.saida}.svg", curvas_para_path(caminhos),
               largura, altura, args.cor)
    gravar_ai(f"{args.saida}.ai",
              curvas_para_path(caminhos, flip_y=True, altura=altura),
              largura, altura, hex_para_rgb(args.cor))

    total = sum(1 for _ in caminhos)
    print(f"{total} contornos tracados de {largura}x{altura}px")
    print(f"gerado: {args.saida}.svg")
    print(f"gerado: {args.saida}.ai")


if __name__ == "__main__":
    main()
