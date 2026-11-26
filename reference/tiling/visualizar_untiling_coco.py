#!/usr/bin/env python3
"""Visualizador de anotaciones COCO reconstruidas sobre las imágenes originales (2448x2048).

Genera dos vistas PNG de alta resolución en `--output-dir` (por defecto `visualizaciones_untiling/`):
  1. Vista de auditoría (`<stem>_auditoria.png`):
     Muestra la cuadrícula de tiles anotados, sombrea zonas no incluidas, indica fusiones `[xN]`,
     etiquetas de clase y encabezado de resumen.
  2. Vista final del modelo (`<stem>_modelo_final.png`):
     Imagen limpia en su dimensión original exacta (2448x2048), sin encabezados, sin cuadrícula de tiles
     y sin textos; únicamente las cajas finales con:
       - Color ROJO (255, 0, 0) para la clase 0 (`particle`)
       - Color AZUL (0, 0, 255) para la clase 1 (`straw`)

Ejemplo de uso desde `/mnt/267EC92D7EC8F699/04_Proyectos/Activos/Dataset_lavado`:
  python visualizar_untiling_coco.py --coco _annotations_original.coco.json --original-dir .
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# Colores para la vista de auditoría
AUDIT_CATEGORY_COLORS = {
    0: (0, 210, 255),     # Clase 0 (particle): Cian brillante
    1: (255, 145, 0),     # Clase 1 (straw): Naranja ámbar
}
MERGED_HIGHLIGHT_COLOR = (50, 255, 120)  # Verde neón para cajas fusionadas entre tiles

# Colores estrictos para la vista final del modelo (sin texto):
# Rojo para clase 0, Azul para clase 1
MODEL_FINAL_COLORS = {
    0: (255, 0, 0),       # Clase 0 -> Rojo
    1: (0, 0, 255),       # Clase 1 -> Azul
}
DEFAULT_COLOR = (220, 80, 255)


def build_class_index_map(categories, annotations):
    """Mapea cada category_id real a un índice 0-based (0, 1, ...) ignorando la supercategoría vacía de Roboflow."""
    used_cat_ids = {ann['category_id'] for ann in annotations}
    real_cats = [
        c for c in sorted(categories, key=lambda x: x['id'])
        if not (c.get('supercategory') == 'none' and c['id'] not in used_cat_ids)
    ]
    cat_id_to_zero_idx = {}
    cat_id_to_name = {}
    for idx, c in enumerate(real_cats):
        cat_id_to_zero_idx[c['id']] = idx
        cat_id_to_name[c['id']] = c['name']
    return cat_id_to_zero_idx, cat_id_to_name


def find_original_image(img_entry, original_dir):
    """Busca el archivo PNG original usando file_name o buscando el stem en S-*/images/."""
    file_name = img_entry.get('file_name', '')
    stem = img_entry.get('extra', {}).get('stem') or Path(file_name).stem

    # 1. Ruta directa relativa a original_dir
    direct = original_dir / file_name
    if direct.is_file():
        return direct

    # 2. Búsqueda en S-*/images/<stem>.png
    matches = sorted(original_dir.glob(f'S-*/images/{stem}.png'))
    if matches:
        return matches[0]

    # 3. Búsqueda recursiva
    for p in sorted(original_dir.rglob(f'{stem}.png')):
        if p.is_file():
            return p

    return None


def build_base_canvas(img_entry, original_dir, roboflow_dir=None):
    """Carga la imagen original o reconstruye el lienzo pegando los tiles de Roboflow."""
    w = int(img_entry.get('width', 2448))
    h = int(img_entry.get('height', 2048))
    orig_path = find_original_image(img_entry, original_dir) if original_dir else None

    if orig_path and orig_path.is_file():
        with Image.open(orig_path) as im:
            return im.convert('RGB'), f'Original: {orig_path.relative_to(original_dir).as_posix()}'

    # Fallback: si no está el PNG original, intentar pegar los tiles .jpg de Roboflow si existe roboflow_dir
    canvas = Image.new('RGB', (w, h), (30, 34, 42))
    tiles_info = img_entry.get('extra', {}).get('tiles', [])
    pasted = 0
    if roboflow_dir and roboflow_dir.is_dir():
        for t in tiles_info:
            rf_name = t.get('roboflow_file', '')
            if not rf_name:
                continue
            matches = list(roboflow_dir.rglob(Path(rf_name).name))
            if matches:
                with Image.open(matches[0]) as tim:
                    tw, th = int(t.get('width', 640)), int(t.get('height', 640))
                    if tim.size != (tw, th):
                        tim = tim.resize((tw, th), Image.Resampling.BILINEAR)
                    canvas.paste(tim.convert('RGB'), (int(t['x0']), int(t['y0'])))
                    pasted += 1

    if pasted > 0:
        return canvas, f'Reconstruido desde {pasted} tiles de Roboflow'
    return canvas, 'Lienzo sintético (imagen original no encontrada)'


def load_font(size):
    """Carga una fuente TrueType legible o usa la fuente por defecto."""
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
    ]
    for c in candidates:
        if Path(c).is_file():
            try:
                return ImageFont.truetype(c, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def render_model_final_view(base_rgb, annotations, cat_id_to_zero_idx, line_width=3):
    """Genera la vista limpia que verá el modelo: sin nombres, sin grilla, Rojo=0 y Azul=1."""
    final_im = base_rgb.copy()
    draw = ImageDraw.Draw(final_im)

    for ann in annotations:
        zero_idx = cat_id_to_zero_idx.get(ann['category_id'], 0)
        color = MODEL_FINAL_COLORS.get(zero_idx, DEFAULT_COLOR)
        x, y, bw, bh = ann['bbox']
        x1, y1, x2, y2 = float(x), float(y), float(x + bw), float(y + bh)

        # Solo el rectángulo del color correspondiente (Rojo=0, Azul=1), sin texto ni etiquetas
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

    return final_im


def render_audit_view(
    base_rgb,
    source_label,
    img_entry,
    annotations,
    cat_id_to_zero_idx,
    cat_id_to_name,
    show_tiles=True,
    show_labels=True,
):
    """Dibuja la vista detallada de auditoría con cuadrícula de tiles, etiquetas y encabezado."""
    canvas_rgb = base_rgb.copy()
    w, h = canvas_rgb.size
    tiles_info = img_entry.get('extra', {}).get('tiles', [])

    # 1. Sombrear zonas sin tiles anotados
    if show_tiles and tiles_info:
        mask = Image.new('L', (w, h), 150)
        mask_draw = ImageDraw.Draw(mask)
        for t in tiles_info:
            x0, y0 = int(t['x0']), int(t['y0'])
            tw, th = int(t.get('width', 640)), int(t.get('height', 640))
            mask_draw.rectangle([x0, y0, x0 + tw, y0 + th], fill=0)

        dark_overlay = Image.new('RGB', (w, h), (15, 18, 24))
        canvas_rgb = Image.composite(dark_overlay, canvas_rgb, mask)

    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_small = load_font(16)
    font_tile = load_font(18)
    font_header = load_font(24)

    # 2. Dibujar bordes de los tiles presentes
    if show_tiles and tiles_info:
        for t in tiles_info:
            x0, y0 = int(t['x0']), int(t['y0'])
            tw, th = int(t.get('width', 640)), int(t.get('height', 640))
            draw.rectangle(
                [x0, y0, x0 + tw - 1, y0 + th - 1],
                outline=(255, 255, 255, 95),
                width=2,
            )
            tag = f"({x0},{y0})"
            draw.rectangle([x0 + 4, y0 + 4, x0 + 118, y0 + 26], fill=(0, 0, 0, 120))
            draw.text((x0 + 8, y0 + 5), tag, fill=(220, 230, 245, 210), font=font_tile)

    # 3. Dibujar cada anotación con etiquetas y marcas de fusión
    cat_counts = Counter()
    merged_boxes_count = 0

    for ann in annotations:
        zero_idx = cat_id_to_zero_idx.get(ann['category_id'], 0)
        cat_name = cat_id_to_name.get(ann['category_id'], f"cls_{zero_idx}")
        cat_counts[f"{zero_idx}:{cat_name}"] += 1
        base_color = AUDIT_CATEGORY_COLORS.get(zero_idx, DEFAULT_COLOR)

        merged_count = ann.get('extra', {}).get('merged_count', 1)
        if merged_count > 1:
            merged_boxes_count += 1

        x, y, bw, bh = ann['bbox']
        x1, y1, x2, y2 = float(x), float(y), float(x + bw), float(y + bh)

        draw.rectangle(
            [x1, y1, x2, y2],
            fill=(base_color[0], base_color[1], base_color[2], 38),
            outline=(base_color[0], base_color[1], base_color[2], 245),
            width=3,
        )

        if merged_count > 1:
            draw.rectangle(
                [x1 - 2, y1 - 2, x2 + 2, y2 + 2],
                outline=(MERGED_HIGHLIGHT_COLOR[0], MERGED_HIGHLIGHT_COLOR[1], MERGED_HIGHLIGHT_COLOR[2], 230),
                width=2,
            )

        if show_labels:
            label = f"{cat_name}" if merged_count == 1 else f"{cat_name} [x{merged_count}]"
            tb = draw.textbbox((0, 0), label, font=font_small)
            tw_txt = tb[2] - tb[0] + 8
            th_txt = tb[3] - tb[1] + 6
            ly = max(0.0, y1 - th_txt)
            bg_col = MERGED_HIGHLIGHT_COLOR if merged_count > 1 else base_color
            draw.rectangle([x1, ly, x1 + tw_txt, ly + th_txt], fill=(bg_col[0], bg_col[1], bg_col[2], 225))
            draw.text((x1 + 4, ly + 2), label, fill=(10, 14, 20, 255), font=font_small)

    composed = Image.alpha_composite(canvas_rgb.convert('RGBA'), overlay).convert('RGB')

    # 4. Barra superior de resumen
    header_h = 68
    final_im = Image.new('RGB', (w, h + header_h), (18, 22, 30))
    final_im.paste(composed, (0, header_h))
    hdraw = ImageDraw.Draw(final_im)

    stem = img_entry.get('extra', {}).get('stem') or Path(img_entry['file_name']).stem
    cls_summary = ' | '.join(f"{k}={v}" for k, v in sorted(cat_counts.items())) or 'Sin anotaciones'
    line1 = (
        f"{stem} ({w}x{h})  —  Tiles anotados: {len(tiles_info)}  |  "
        f"Total objetos: {len(annotations)} ({cls_summary})  |  "
        f"Fusionadas en solapamiento: {merged_boxes_count}"
    )
    line2 = (
        f"{source_label}   |   Auditoría: Cian = 0:particle, Naranja = 1:straw, "
        f"Borde verde [xN] = fusionada entre N tiles"
    )
    hdraw.text((16, 8), line1, fill=(245, 248, 255), font=font_header)
    hdraw.text((16, 40), line2, fill=(175, 190, 215), font=font_small)

    return final_im


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--coco',
        type=Path,
        default=Path('_annotations_original.coco.json'),
        help='Archivo COCO traducido por untiling_fibra_roboflow.py (default: _annotations_original.coco.json)',
    )
    parser.add_argument(
        '--original-dir',
        type=Path,
        default=Path('.'),
        help='Carpeta Dataset_lavado donde están las sesiones S-*/images/FLA_*.png (default: .)',
    )
    parser.add_argument(
        '--roboflow-dir',
        type=Path,
        default=Path('Dataset_Fibra_lavada_2026.v1i.coco'),
        help='Carpeta opcional descargada de Roboflow para usar los tiles .jpg si no se halla algún PNG original',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('visualizaciones_untiling'),
        help='Carpeta donde se guardarán las dos vistas por imagen (default: visualizaciones_untiling)',
    )
    parser.add_argument(
        '--image',
        action='append',
        help='Opcional: filtrar por nombre o substring (ej. --image M001_V01)',
    )
    parser.add_argument(
        '--line-width',
        type=int,
        default=3,
        help='Grosor de línea en píxeles para las cajas de la vista final del modelo (default: 3)',
    )
    parser.add_argument(
        '--no-tiles-grid',
        action='store_true',
        help='Oculta la cuadrícula de tiles en la vista de auditoría',
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Oculta las etiquetas de texto en la vista de auditoría',
    )
    args = parser.parse_args()

    if not args.coco.is_file():
        parser.error(
            f'No se encontró el archivo COCO: {args.coco}\n'
            f'Ejecuta primero:\n'
            f'  python untiling_fibra_roboflow.py --coco Dataset_Fibra_lavada_2026.v1i.coco --original-dir . --output _annotations_original.coco.json'
        )

    data = json.loads(args.coco.read_text(encoding='utf-8'))
    annotations = data.get('annotations', [])
    cat_id_to_zero_idx, cat_id_to_name = build_class_index_map(data.get('categories', []), annotations)

    anns_by_img = defaultdict(list)
    for ann in annotations:
        anns_by_img[ann['image_id']].append(ann)

    images = data.get('images', [])
    if args.image:
        images = [
            im for im in images
            if any(q.lower() in im['file_name'].lower() for q in args.image)
        ]
        if not images:
            parser.error(f'Ninguna imagen coincide con el filtro --image {args.image}')

    audit_dir = args.output_dir / 'auditoria'
    model_dir = args.output_dir / 'modelo_final'
    audit_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print('Mapeo de clases para la vista final del modelo:')
    for cid, zidx in sorted(cat_id_to_zero_idx.items(), key=lambda x: x[1]):
        cname = cat_id_to_name[cid]
        col_name = 'ROJO (255, 0, 0)' if zidx == 0 else ('AZUL (0, 0, 255)' if zidx == 1 else str(DEFAULT_COLOR))
        print(f'  - Clase {zidx} ({cname}, category_id={cid}) -> {col_name}')

    print(f'\nGenerando visualizaciones para {len(images)} imágenes...')
    for idx, img_entry in enumerate(images, 1):
        img_anns = anns_by_img.get(img_entry['id'], [])
        base_rgb, source_label = build_base_canvas(img_entry, args.original_dir, args.roboflow_dir)
        stem = img_entry.get('extra', {}).get('stem') or Path(img_entry['file_name']).stem

        # 1. Vista de auditoría (con tiles, nombres y fusiones)
        audit_im = render_audit_view(
            base_rgb=base_rgb,
            source_label=source_label,
            img_entry=img_entry,
            annotations=img_anns,
            cat_id_to_zero_idx=cat_id_to_zero_idx,
            cat_id_to_name=cat_id_to_name,
            show_tiles=not args.no_tiles_grid,
            show_labels=not args.no_labels,
        )
        audit_path = audit_dir / f'{stem}_auditoria.png'
        audit_im.save(audit_path, format='PNG', compress_level=3)

        # 2. Vista final del modelo (limpia, sin nombres, Rojo=0 y Azul=1)
        model_im = render_model_final_view(
            base_rgb=base_rgb,
            annotations=img_anns,
            cat_id_to_zero_idx=cat_id_to_zero_idx,
            line_width=args.line_width,
        )
        model_path = model_dir / f'{stem}_modelo_final.png'
        model_im.save(model_path, format='PNG', compress_level=3)

        print(f'  [{idx}/{len(images)}] {stem} ({len(img_anns)} cajas):')
        print(f'      -> Auditoría:    {audit_path}')
        print(f'      -> Modelo final: {model_path}')

    print(f'\nFinalizado. Revisa las dos carpetas dentro de: {args.output_dir.resolve()}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)