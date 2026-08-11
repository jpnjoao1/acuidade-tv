#!/usr/bin/env python3
"""Converte um logotipo raster (PNG/JPG) em curvas vetoriais.

O fundo e detectado e descartado; o restante da arte e separado em N cores
e cada cor vira uma camada propria de curvas de Bezier.

Gera dois arquivos a partir da mesma geometria tracada:
  - <saida>.svg  : SVG com um <path> por cor, pronto para edicao
  - <saida>.ai   : arquivo PDF-compatible, aberto nativamente pelo Illustrator

Uso:
    python3 tools/vetorizar.py logo.jpg logo --cores 2
"""

import argparse
import sys
import zlib

import numpy as np
import potrace
from PIL import Image


# --------------------------------------------------------------------------
# Leitura e separacao de cores
# --------------------------------------------------------------------------

def carregar(caminho):
    """Le a imagem e devolve o array RGB, achatando alfa sobre branco."""
    img = Image.open(caminho)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        fundo = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(fundo, img)
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def cor_de_fundo(rgb):
    """Estima o fundo pela cor mais frequente nas bordas da imagem."""
    bordas = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    chaves, contagens = np.unique((bordas // 8).astype(int), axis=0,
                                  return_counts=True)
    return chaves[contagens.argmax()].astype(np.float32) * 8 + 4


def kmeans(amostras, k, iteracoes=24, semente=0):
    """K-means simples e deterministico sobre as cores dos pixels."""
    rng = np.random.default_rng(semente)
    centros = amostras[rng.choice(len(amostras), k, replace=False)].astype(float)
    rotulos = np.zeros(len(amostras), dtype=int)
    for _ in range(iteracoes):
        dist = ((amostras[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2)
        novos = dist.argmin(axis=1)
        if (novos == rotulos).all():
            break
        rotulos = novos
        for i in range(k):
            se_tem = rotulos == i
            if se_tem.any():
                centros[i] = amostras[se_tem].mean(axis=0)
    return centros, rotulos


def separar_camadas(rgb, k, tolerancia_fundo, max_amostras=200_000):
    """Devolve [(cor_rgb, mascara_booleana), ...], da maior para a menor area."""
    fundo = cor_de_fundo(rgb)
    distancia = np.sqrt(((rgb - fundo) ** 2).sum(axis=2))
    tinta = distancia > tolerancia_fundo
    if not tinta.any():
        sys.exit("Nenhuma arte encontrada — reduza --tolerancia-fundo.")

    pixels = rgb[tinta]
    rng = np.random.default_rng(0)

    # A paleta sai apenas dos pixels bem saturados. Os da franja de
    # antialiasing sao misturas com o fundo e falseariam os centros —
    # eles entram depois, atribuidos a cor solida mais proxima.
    solidos = rgb[distancia > max(tolerancia_fundo, 0.55 * distancia.max())]
    if len(solidos) < k:
        solidos = pixels

    if k == 1:
        return [(solidos.mean(axis=0).round().astype(int), tinta)]

    # K-means sobre uma amostra: o resultado e igual e roda em segundos.
    amostra = solidos[rng.choice(len(solidos), min(len(solidos), max_amostras),
                                 replace=False)]
    centros, _ = kmeans(amostra, k)

    dist = ((pixels[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2)
    rotulos = dist.argmin(axis=1)

    camadas = []
    indices = np.flatnonzero(tinta.ravel())
    for i in range(k):
        mascara = np.zeros(tinta.size, dtype=bool)
        mascara[indices[rotulos == i]] = True
        camadas.append((centros[i].round().astype(int),
                        mascara.reshape(tinta.shape)))
    camadas.sort(key=lambda c: c[1].sum(), reverse=True)
    return camadas


# --------------------------------------------------------------------------
# Tracado
# --------------------------------------------------------------------------

def tracar(mascara, alphamax, opttolerance, turdsize):
    """Roda o potrace sobre a mascara e devolve as curvas encontradas.

    `potrace.Bitmap` inverte o array no construtor, entao passamos o
    complemento para que a tinta continue sendo o primeiro plano.
    """
    return potrace.Bitmap(~mascara).trace(
        turdsize=turdsize,
        alphamax=alphamax,
        opticurve=True,
        opttolerance=opttolerance,
    )


def curvas_para_path(caminhos, flip_y=False, altura=0):
    """Converte as curvas do potrace num unico atributo `d` de path.

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

def hexa(rgb):
    return "#{:02X}{:02X}{:02X}".format(*(int(c) for c in rgb))


def gravar_svg(destino, camadas, largura, altura):
    linhas = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {largura} {altura}" width="{largura}" height="{altura}">'
    ]
    for nome, cor, d in camadas:
        linhas.append(f'  <g id="{nome}">')
        linhas.append(f'    <path fill="{hexa(cor)}" fill-rule="evenodd" d="{d}"/>')
        linhas.append("  </g>")
    linhas.append("</svg>")
    with open(destino, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")


# --------------------------------------------------------------------------
# Saida .ai (PDF-compatible)
# --------------------------------------------------------------------------

def path_svg_para_pdf(d):
    """Reescreve o `d` do SVG nos operadores de path do PDF."""
    saida = []
    i, n = 0, len(d)
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


def gravar_ai(destino, camadas, largura, altura):
    """Escreve um PDF 1.4 minimo com uma camada preenchida por cor.

    Illustrator abre e edita arquivos .ai gravados neste formato
    ("PDF Compatible File"), com as curvas ja editaveis.
    """
    blocos = []
    for _, cor, d in camadas:
        r, g, b = (int(c) / 255 for c in cor)
        blocos.append(f"{r:.4f} {g:.4f} {b:.4f} rg\n{path_svg_para_pdf(d)}\nf*")
    fluxo = zlib.compress("\n".join(blocos).encode("latin-1"))

    objetos = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {largura} {altura}] "
            f"/Contents 4 0 R /Resources << >> >>"
        ).encode("latin-1"),
        (
            f"<< /Length {len(fluxo)} /Filter /FlateDecode >>".encode("latin-1")
            + b"\nstream\n" + fluxo + b"\nendstream"
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

def main():
    ap = argparse.ArgumentParser(description="Vetoriza um logotipo em curvas.")
    ap.add_argument("entrada", help="imagem de origem (PNG, JPG...)")
    ap.add_argument("saida", help="prefixo dos arquivos gerados (sem extensao)")
    ap.add_argument("--cores", type=int, default=1,
                    help="quantas cores separar em camadas (padrao: 1)")
    ap.add_argument("--tolerancia-fundo", type=float, default=60,
                    help="distancia RGB minima do fundo para virar arte (padrao: 60)")
    ap.add_argument("--suavizacao", type=float, default=1.0,
                    help="alphamax do potrace, 0 = cantos vivos (padrao: 1.0)")
    ap.add_argument("--tolerancia", type=float, default=0.2,
                    help="opttolerance do potrace (padrao: 0.2)")
    ap.add_argument("--ruido", type=int, default=2,
                    help="descarta manchas menores que N pixels (padrao: 2)")
    args = ap.parse_args()

    rgb = carregar(args.entrada)
    altura, largura = rgb.shape[:2]

    camadas_svg, camadas_pdf = [], []
    for i, (cor, mascara) in enumerate(
        separar_camadas(rgb, args.cores, args.tolerancia_fundo), start=1
    ):
        caminhos = tracar(mascara, args.suavizacao, args.tolerancia, args.ruido)
        nome = f"cor-{i}-{hexa(cor).lstrip('#').lower()}"
        camadas_svg.append((nome, cor, curvas_para_path(caminhos)))
        camadas_pdf.append(
            (nome, cor, curvas_para_path(caminhos, flip_y=True, altura=altura))
        )
        print(f"camada {nome}: {sum(1 for _ in caminhos)} contornos, "
              f"{int(mascara.sum())} px")

    gravar_svg(f"{args.saida}.svg", camadas_svg, largura, altura)
    gravar_ai(f"{args.saida}.ai", camadas_pdf, largura, altura)
    print(f"gerado: {args.saida}.svg")
    print(f"gerado: {args.saida}.ai")


if __name__ == "__main__":
    main()
