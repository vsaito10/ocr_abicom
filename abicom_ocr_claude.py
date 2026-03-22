"""
Script OCR para extrair valores numéricos à esquerda de "(R$/L)" em imagens Abicom/PPI.

Dependências:
    pip install pytesseract pillow numpy

Uso:
    python extract_rsl_values.py                         # usa imagem padrão
    python extract_rsl_values.py caminho/para/imagem.png
"""

import re
import sys
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract


# Caminho do executável do pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ─── Pré-processamento ────────────────────────────────────────────────────────

def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Isola pixels brancos e amarelos (onde estão os números e legendas) e
    faz upscale 8× para melhorar a acurácia do Tesseract.
    """
    arr = np.array(img.convert("RGB"))

    # Texto branco: todos os canais altos
    white  = (arr[:, :, 0] > 180) & (arr[:, :, 1] > 180) & (arr[:, :, 2] > 180)

    # Texto amarelo (cabeçalhos): vermelho/verde alto, azul baixo
    yellow = (arr[:, :, 0] > 180) & (arr[:, :, 1] > 150) & (arr[:, :, 2] < 80)

    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    mask[white | yellow] = 255

    mask_img = Image.fromarray(mask)
    w, h = mask_img.size
    return mask_img.resize((w * 8, h * 8), Image.LANCZOS)


# ─── Normalização do texto OCR ────────────────────────────────────────────────

def normalize_ocr_text(raw: str) -> str:
    """
    Corrige erros típicos do OCR em imagens com fundo escuro:
      • ~ " ' ` -> sinal de menos
      • RS/     -> R$/
      • /1) /l) -> /L)
      • }]      -> ) em (R$/L)
    """
    text = raw
    text = re.sub(r'[~"\u2018\u2019`\u201c\u201d]', '-', text)
    text = text.replace('RS/', 'R$/')
    text = re.sub(r'/[1lI]\)', '/L)', text)
    text = re.sub(r'\(R\$\/L[}\]]', '(R$/L)', text, flags=re.IGNORECASE)
    return text


# ─── Detecção de seção ────────────────────────────────────────────────────────

def find_section_split(img: Image.Image) -> int:
    """
    Usa bounding boxes do Tesseract para localizar o texto 'PETROBRAS' e
    retorna o X onde a coluna direita começa.
    Fallback: metade da largura da imagem.
    """
    data = pytesseract.image_to_data(
        img, lang="eng", output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 11"
    )
    for text, left in zip(data["text"], data["left"]):
        if "PETROBRAS" in text.upper():
            return left
    return img.width // 2


# ─── Extração principal ───────────────────────────────────────────────────────

def extract_values_before_rsl(image_path: str, section: str = "principais") -> list:
    """
    Pipeline completo:
      1. Abre a imagem e detecta a divisão entre seções
      2. Recorta a coluna desejada ('principais' ou 'petrobras')
      3. Pré-processa e roda OCR (PSM 11)
      4. Normaliza e extrai valores antes de (R$/L)

    Retorna lista de dicts: {'value': str, 'context': str}
    """
    img = Image.open(image_path)
    width, height = img.size

    split_x = find_section_split(img)

    if section == "principais":
        img = img.crop((0, 0, split_x, height))
    else:
        img = img.crop((split_x, 0, width, height))

    img_proc = preprocess_image(img)
    w_p, h_p = img_proc.size

    # PSM 11 (esparso) cobre bem a parte superior; PSM 3 (auto) recupera
    # valores na metade inferior que o PSM 11 perde.
    raw_top    = pytesseract.image_to_string(img_proc, lang="eng", config="--oem 3 --psm 11")
    raw_bottom = pytesseract.image_to_string(
        img_proc.crop((0, h_p // 2, w_p, h_p)), lang="eng", config="--oem 3 --psm 3"
    )
    raw_text = raw_top + "\n" + raw_bottom

    print("=== Texto bruto extraído pelo OCR ===")
    print(raw_text)
    print("=====================================\n")

    normalized = normalize_ocr_text(raw_text)

    pattern = re.compile(
        r'([+-]?\d+[.,]\d+)'
        r'\s*'
        r'\(R\$\/L\)',
        re.IGNORECASE,
    )

    results = []
    for match in pattern.finditer(normalized):
        value   = match.group(1)
        start   = max(0, match.start() - 60)
        context = normalized[start: match.end()].strip().replace("\n", " ")
        results.append({"value": value, "context": context})

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def process_image(image_path: str) -> pd.DataFrame:
    date_str = re.search(r'(\d{8})_abicom', image_path).group(1)
    date_idx = pd.to_datetime([date_str], format="%Y%m%d")

    results = extract_values_before_rsl(image_path, section="principais")

    if not results:
        return pd.DataFrame({"gasolina": [float("nan")], "diesel": [float("nan")]}, index=date_idx)

    df = pd.DataFrame(results, columns=["value", "context"])
    df["value"] = pd.to_numeric(df["value"].str.replace(",", "."), errors="coerce")

    df = df[["value"]].T
    df.columns = ["gasolina", "diesel"]
    df.index = date_idx

    return df


def main():
    import glob

    image_dir = r"C:\B3\historico-arquivos\imagem-abicom\layout-atual-novo"
    image_paths = sorted(
        glob.glob(f"{image_dir}\\*_abicom.png") +
        glob.glob(f"{image_dir}\\*_abicom.jpeg") +
        glob.glob(f"{image_dir}\\*_abicom.jpg")
    )

    frames = []
    for image_path in image_paths:
        print(f"Processando: {image_path}")
        try:
            frames.append(process_image(image_path))
        except Exception as e:
            print(f"  Erro: {e}")
            date_str = re.search(r'(\d{8})_abicom', image_path).group(1)
            date_idx = pd.to_datetime([date_str], format="%Y%m%d")
            frames.append(pd.DataFrame({"gasolina": [float("nan")], "diesel": [float("nan")]}, index=date_idx))

    df_all = pd.concat(frames).sort_index()

    print("\nResultado final:")
    print(df_all.to_string(index=True))

    output_path = r"C:\B3\ocr_abicom\ocr_layout_atual_novo.csv"
    df_all.to_csv(output_path)
    print(f"\nSalvo em: {output_path}")

    return df_all


if __name__ == "__main__":
    main()
