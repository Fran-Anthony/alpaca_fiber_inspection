#!/usr/bin/env python3
"""Traduce anotaciones COCO de tiles (Roboflow) de vuelta a las imágenes originales antes del tiling.

Soporta:
  - Extracción automática de coordenadas (x0, y0) y nombre original desde `extra.name`, `file_name` o `manifest_tiles.csv`.
  - Lectura de dimensiones originales exactas desde `--original-dir` (S-*/images/FLA_*.png), `--manifest` (manifest_tiles.csv)
    o mediante `--orig-width` y `--orig-height` (por defecto 2448x2048).
  - Re-escalado automático si Roboflow redimensionó los tiles al exportar.
  - Traslación de bounding boxes (`bbox`) y polígonos de segmentación (`segmentation`) al sistema de coordenadas original.
  - Fusión/deduplicación inteligente de anotaciones duplicadas o partidas en las zonas de solapamiento entre tiles distintos
    (`--dedup-mode union|largest|wbf|none`), respetando siempre las anotaciones dentro de un mismo tile.
  - Registro en `extra` de los tiles presentes por imagen y de las cajas fusionadas para visualización y auditoría.

Ejemplos de uso desde la carpeta Dataset_lavado:
  # 1. Traducir las anotaciones descargadas de Roboflow vinculándolas con las imágenes originales:
  python untiling_fibra_roboflow.py \
    --coco Dataset_Fibra_lavada_2026.v1i.coco \
    --original-dir . \
    --output _annotations_original.coco.json

  # 2. Si además tienes el manifest_tiles.csv generado al hacer el tiling:
  python untiling_fibra_roboflow.py \
    --coco Dataset_Fibra_lavada_2026.v1i.coco \
    --original-dir . \
    --manifest ../Dataset_lavado_tiles640/manifest_tiles.csv \
    --output _annotations_original.coco.json
"""
import argparse
import copy
import csv
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from alpaca_inspection.tiling_geometry import clip_bbox, deduplicate_cross_tile_annotations

try:
    from PIL import Image
except ImportError:
    Image = None

# Patrón para nombres generados por tiling_fibra_roboflow.py:
# Ejemplo en extra.name: FLA_S20260916A_L01_M002_V01_x1808_y0512.png
# Ejemplo en file_name de Roboflow: FLA_S20260916A_L01_M002_V01_x1808_y0512_png.rf.a0c2cd8964f12e659b77e0906c08874e.jpg
TILE_REGEX = re.compile(
    r'^(?P<stem>.+)_x(?P<x0>\d+)_y(?P<y0>\d+)(?:\.(?P<ext>png|jpg|jpeg|tif|tiff)|_(?P<ext2>png|jpg|jpeg)\.rf\..+)$',
    re.IGNORECASE,
)


def parse_tile_identity(img_entry):
    """Extrae (orig_stem, x0, y0, tile_key) desde un registro `image` de COCO (Roboflow)."""
    candidates = []
    extra = img_entry.get('extra')
    if isinstance(extra, dict) and extra.get('name'):
        candidates.append(Path(extra['name']).name)
    if img_entry.get('file_name'):
        candidates.append(Path(img_entry['file_name']).name)

    for name in candidates:
        m = TILE_REGEX.match(name)
        if m:
            stem = m.group('stem')
            x0 = int(m.group('x0'))
            y0 = int(m.group('y0'))
            tile_key = f'{stem}_x{x0:04d}_y{y0:04d}.png'
            return stem, x0, y0, tile_key

    raise ValueError(
        f'No se pudo extraer (stem, x0, y0) de la imagen id={img_entry.get("id")}: '
        f'file_name={img_entry.get("file_name")!r}, extra={img_entry.get("extra")!r}'
    )


def load_manifest(manifest_path):
    """Carga manifest_tiles.csv generado por tiling_fibra_roboflow.py."""
    by_tile_name = {}
    by_stem = {}
    with manifest_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tile_name = Path(row['tile_file']).name
            source_rel = row['source_file']
            stem = Path(source_rel).stem
            info = {
                'tile_file': row['tile_file'],
                'source_file': source_rel,
                'session': row.get('session', ''),
                'source_width': int(row['source_width']),
                'source_height': int(row['source_height']),
                'x0': int(row['x0']),
                'y0': int(row['y0']),
                'tile_width': int(row['tile_width']),
                'tile_height': int(row['tile_height']),
                'stem': stem,
            }
            by_tile_name[tile_name] = info
            if stem not in by_stem:
                by_stem[stem] = {
                    'source_file': source_rel,
                    'session': row.get('session', ''),
                    'width': int(row['source_width']),
                    'height': int(row['source_height']),
                    'expected_tiles': 0,
                }
            by_stem[stem]['expected_tiles'] += 1
    return by_tile_name, by_stem


def scan_original_dir(original_dir):
    """Escanea S-*/images/FLA_*.png en Dataset_lavado."""
    if Image is None:
        raise RuntimeError('Se requiere Pillow (pip install Pillow) para leer dimensiones desde --original-dir.')
    by_stem = {}
    candidates = sorted(original_dir.glob('S-*/images/FLA_*.png'))
    if not candidates:
        candidates = sorted(
            p for p in original_dir.rglob('FLA_*.png')
            if not TILE_REGEX.match(p.name)
        )
    for p in candidates:
        if not p.is_file():
            continue
        with Image.open(p) as im:
            w, h = im.size
        rel = p.relative_to(original_dir).as_posix()
        by_stem[p.stem] = {
            'abs_path': p,
            'source_file': rel,
            'width': w,
            'height': h,
        }
    return by_stem


def translate_segmentation(seg, x0, y0, scale_x, scale_y, max_w, max_h):
    """Traslada polígonos de segmentación COCO al sistema original."""
    if not seg or not isinstance(seg, list):
        return seg
    new_seg = []
    for poly in seg:
        if isinstance(poly, list) and len(poly) >= 6:
            new_poly = []
            for idx, val in enumerate(poly):
                if idx % 2 == 0:
                    px = max(0.0, min(float(max_w), float(val) * scale_x + x0))
                    new_poly.append(round(px, 2))
                else:
                    py = max(0.0, min(float(max_h), float(val) * scale_y + y0))
                    new_poly.append(round(py, 2))
            new_seg.append(new_poly)
        else:
            new_seg.append(poly)
    return new_seg


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--coco',
        type=Path,
        required=True,
        help='Ruta al archivo _annotations.coco.json o a la carpeta descargada de Roboflow (ej. Dataset_Fibra_lavada_2026.v1i.coco)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Ruta del archivo JSON COCO de salida (ej. _annotations_original.coco.json)',
    )
    parser.add_argument(
        '--original-dir',
        type=Path,
        default=Path('.'),
        help='Carpeta Dataset_lavado donde están las sesiones S-*/images/FLA_*.png (default: directorio actual .)',
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        help='Opcional: ruta a manifest_tiles.csv generado por tiling_fibra_roboflow.py',
    )
    parser.add_argument(
        '--orig-width',
        type=int,
        default=2448,
        help='Ancho de respaldo si no se encuentra la imagen original (default: 2448)',
    )
    parser.add_argument(
        '--orig-height',
        type=int,
        default=2048,
        help='Alto de respaldo si no se encuentra la imagen original (default: 2048)',
    )
    parser.add_argument(
        '--expected-tile-size',
        type=int,
        default=640,
        help='Tamaño nominal del tile en píxeles originales (default: 640)',
    )
    parser.add_argument(
        '--dedup-mode',
        choices=['union', 'largest', 'wbf', 'none'],
        default='union',
        help='Estrategia para duplicados en el solapamiento entre tiles distintos (default: union)',
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.40,
        help='Umbral IoU para fusionar cajas de la misma clase entre tiles distintos (default: 0.40)',
    )
    parser.add_argument(
        '--iomin-threshold',
        type=float,
        default=0.75,
        help='Umbral IoMin (intersección / área menor) para absorber fragmentos cortados en bordes (default: 0.75)',
    )
    parser.add_argument(
        '--path-mode',
        choices=['auto', 'basename', 'relative'],
        default='auto',
        help='Formato de file_name en el COCO de salida: relative (S-.../images/FLA_...png) o basename (FLA_...png)',
    )
    parser.add_argument(
        '--clean-supercategory',
        action='store_true',
        help='Elimina la categoría raíz supercategory="none" sin anotaciones que añade Roboflow',
    )
    parser.add_argument(
        '--copy-images-to',
        type=Path,
        help='Opcional: copia las imágenes originales referenciadas a esta carpeta',
    )
    args = parser.parse_args()

    # 1. Localizar archivos COCO (admite un archivo JSON o la carpeta de Roboflow con train/valid/test)
    coco_files = []
    if args.coco.is_dir():
        coco_files = sorted(args.coco.rglob('_annotations.coco.json'))
        if not coco_files:
            parser.error(f'No se encontraron archivos _annotations.coco.json dentro de {args.coco}')
    elif args.coco.is_file():
        coco_files = [args.coco]
    else:
        parser.error(f'No existe la ruta especificada en --coco: {args.coco}')

    manifest_tiles, manifest_stems = {}, {}
    if args.manifest:
        if not args.manifest.is_file():
            parser.error(f'No existe el archivo --manifest: {args.manifest}')
        manifest_tiles, manifest_stems = load_manifest(args.manifest)

    orig_stems = {}
    if args.original_dir and args.original_dir.is_dir():
        orig_stems = scan_original_dir(args.original_dir)

    # 2. Unificar entradas si hay varios JSON (ej. train/valid/test)
    base_info = {}
    base_licenses = []
    categories_by_id = {}
    raw_images = []
    raw_annotations = []
    seen_tile_keys = {}

    for cf in coco_files:
        data = json.loads(cf.read_text(encoding='utf-8'))
        if not base_info and 'info' in data:
            base_info = copy.deepcopy(data['info'])
        if not base_licenses and 'licenses' in data:
            base_licenses = copy.deepcopy(data['licenses'])
        for cat in data.get('categories', []):
            categories_by_id[cat['id']] = cat

        local_to_global_img = {}
        for img in data.get('images', []):
            stem, x0, y0, tile_key = parse_tile_identity(img)
            if tile_key in seen_tile_keys:
                local_to_global_img[img['id']] = seen_tile_keys[tile_key]
            else:
                new_id = len(raw_images)
                seen_tile_keys[tile_key] = new_id
                local_to_global_img[img['id']] = new_id
                img_copy = copy.deepcopy(img)
                img_copy['id'] = new_id
                img_copy['_parsed'] = (stem, x0, y0, tile_key)
                raw_images.append(img_copy)

        for ann in data.get('annotations', []):
            ann_copy = copy.deepcopy(ann)
            ann_copy['id'] = len(raw_annotations) + 1
            ann_copy['image_id'] = local_to_global_img[ann['image_id']]
            raw_annotations.append(ann_copy)

    # 3. Agrupar tiles por imagen original (stem)
    tiles_by_stem = defaultdict(list)
    tile_meta_by_id = {}

    for img in raw_images:
        stem, x0, y0, tile_key = img['_parsed']
        tw_coco = int(img.get('width', args.expected_tile_size))
        th_coco = int(img.get('height', args.expected_tile_size))

        if tile_key in manifest_tiles:
            m_info = manifest_tiles[tile_key]
            x0, y0 = m_info['x0'], m_info['y0']
            tw_orig, th_orig = m_info['tile_width'], m_info['tile_height']
        else:
            tw_orig, th_orig = args.expected_tile_size, args.expected_tile_size

        scale_x = tw_orig / float(tw_coco) if tw_coco else 1.0
        scale_y = th_orig / float(th_coco) if th_coco else 1.0

        meta = {
            'tile_image_id': img['id'],
            'stem': stem,
            'x0': x0,
            'y0': y0,
            'tile_key': tile_key,
            'roboflow_file': img.get('file_name', ''),
            'scale_x': scale_x,
            'scale_y': scale_y,
            'tw_orig': tw_orig,
            'th_orig': th_orig,
            'date_captured': img.get('date_captured'),
            'license': img.get('license', 1),
        }
        tile_meta_by_id[img['id']] = meta
        tiles_by_stem[stem].append(meta)

    # 4. Construir la lista de imágenes originales y resolver dimensiones / nombres de archivo
    new_images = []
    stem_to_new_img = {}
    coverage_report = []

    for new_img_id, stem in enumerate(sorted(tiles_by_stem.keys())):
        t_list = sorted(tiles_by_stem[stem], key=lambda t: (t['y0'], t['x0']))
        max_extent_w = max(t['x0'] + t['tw_orig'] for t in t_list)
        max_extent_h = max(t['y0'] + t['th_orig'] for t in t_list)

        if stem in orig_stems:
            orig_w = orig_stems[stem]['width']
            orig_h = orig_stems[stem]['height']
            rel_path = orig_stems[stem]['source_file']
        elif stem in manifest_stems:
            orig_w = manifest_stems[stem]['width']
            orig_h = manifest_stems[stem]['height']
            rel_path = manifest_stems[stem]['source_file']
        else:
            orig_w = max(args.orig_width, max_extent_w)
            orig_h = max(args.orig_height, max_extent_h)
            rel_path = f'{stem}.png'

        if args.path_mode == 'basename':
            file_name = f'{stem}.png'
        elif args.path_mode == 'relative':
            file_name = rel_path
        else:
            file_name = rel_path if (stem in orig_stems or stem in manifest_stems) else f'{stem}.png'

        expected_tiles = manifest_stems.get(stem, {}).get('expected_tiles')
        img_record = {
            'id': new_img_id,
            'license': t_list[0]['license'],
            'file_name': file_name,
            'height': orig_h,
            'width': orig_w,
            'date_captured': t_list[0]['date_captured'] or datetime.now(timezone.utc).isoformat(),
            'extra': {
                'stem': stem,
                'tiles_present': len(t_list),
                **({'tiles_expected': expected_tiles} if expected_tiles is not None else {}),
                'tiles': [
                    {
                        'x0': t['x0'],
                        'y0': t['y0'],
                        'width': t['tw_orig'],
                        'height': t['th_orig'],
                        'tile_key': t['tile_key'],
                        'roboflow_file': t['roboflow_file'],
                    }
                    for t in t_list
                ],
            },
        }
        new_images.append(img_record)
        stem_to_new_img[stem] = img_record
        coverage_report.append({
            'stem': stem,
            'file_name': file_name,
            'found_in_original_dir': stem in orig_stems,
            'width': orig_w,
            'height': orig_h,
            'tiles_present': len(t_list),
            'tiles_expected': expected_tiles,
        })

    # 5. Trasladar cada anotación a coordenadas de la imagen original
    anns_by_stem = defaultdict(list)
    for ann in raw_annotations:
        t_meta = tile_meta_by_id[ann['image_id']]
        stem = t_meta['stem']
        orig_img = stem_to_new_img[stem]
        orig_w, orig_h = orig_img['width'], orig_img['height']

        x, y, w, h = ann['bbox']
        gx = x * t_meta['scale_x'] + t_meta['x0']
        gy = y * t_meta['scale_y'] + t_meta['y0']
        gw = w * t_meta['scale_x']
        gh = h * t_meta['scale_y']

        clipped_bbox = clip_bbox([gx, gy, gw, gh], orig_w, orig_h)
        if clipped_bbox[2] <= 0 or clipped_bbox[3] <= 0:
            continue

        new_ann = copy.deepcopy(ann)
        new_ann['image_id'] = orig_img['id']
        new_ann['bbox'] = clipped_bbox
        new_ann['area'] = round(clipped_bbox[2] * clipped_bbox[3], 3)
        new_ann['segmentation'] = translate_segmentation(
            ann.get('segmentation', []),
            t_meta['x0'],
            t_meta['y0'],
            t_meta['scale_x'],
            t_meta['scale_y'],
            orig_w,
            orig_h,
        )
        new_ann['_tile_id'] = t_meta['tile_image_id']
        new_ann['_tile_coord'] = [t_meta['x0'], t_meta['y0']]
        anns_by_stem[stem].append(new_ann)

    # 6. Deduplicar anotaciones en zonas de solapamiento entre tiles distintos
    final_annotations = []
    total_merged = 0
    dedup_stats_by_stem = {}

    for stem in sorted(stem_to_new_img.keys()):
        orig_img = stem_to_new_img[stem]
        stem_anns = anns_by_stem.get(stem, [])
        deduped, merged_count = deduplicate_cross_tile_annotations(
            stem_anns,
            mode=args.dedup_mode,
            iou_thresh=args.iou_threshold,
            iomin_thresh=args.iomin_threshold,
            img_w=orig_img['width'],
            img_h=orig_img['height'],
        )
        total_merged += merged_count
        dedup_stats_by_stem[stem] = {
            'raw_annotations': len(stem_anns),
            'merged_duplicates': merged_count,
            'final_annotations': len(deduped),
        }
        for a in deduped:
            a.pop('_tile_id', None)
            a.pop('_tile_coord', None)
            a['id'] = len(final_annotations) + 1
            final_annotations.append(a)

    # 7. Limpiar categorías si se solicita
    used_cat_ids = {a['category_id'] for a in final_annotations}
    final_categories = []
    for cat_id in sorted(categories_by_id.keys()):
        cat = categories_by_id[cat_id]
        if args.clean_supercategory and cat.get('supercategory') == 'none' and cat_id not in used_cat_ids:
            continue
        final_categories.append(cat)

    # 8. Guardar JSON COCO traducido
    out_info = copy.deepcopy(base_info)
    out_info['description'] = (
        f"{out_info.get('description', 'COCO Dataset')} | "
        f"Untiled to original image dimensions (dedup={args.dedup_mode})"
    )
    out_coco = {
        'info': out_info,
        'licenses': base_licenses,
        'categories': final_categories,
        'images': new_images,
        'annotations': final_annotations,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_coco, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # 9. Copiar imágenes originales si se solicitó
    if args.copy_images_to:
        if not orig_stems:
            parser.error('--copy-images-to requiere que existan imágenes en --original-dir')
        args.copy_images_to.mkdir(parents=True, exist_ok=True)
        for img in new_images:
            stem = img['extra']['stem']
            if stem in orig_stems:
                src_p = orig_stems[stem]['abs_path']
                dst_p = args.copy_images_to / img['file_name']
                dst_p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_p, dst_p)

    # 10. Resumen en consola
    print('=== Traducción de Tiles a Dataset Original Finalizada ===')
    print(f'Archivos COCO leídos:      {len(coco_files)}')
    print(f'Imágenes originales halladas en {args.original_dir.resolve()}: {len(orig_stems)}')
    print(f'Tiles procesados:          {len(raw_images)}')
    print(f'Imágenes reconstruidas:    {len(new_images)}')
    print(f'Anotaciones en tiles:      {len(raw_annotations)}')
    print(f'Duplicados fusionados:     {total_merged} (modo: {args.dedup_mode})')
    print(f'Anotaciones finales:       {len(final_annotations)}')
    print(f'Archivo COCO generado:     {args.output}\n')
    print('Detalle por imagen original:')
    for rep in coverage_report:
        stem = rep['stem']
        st = dedup_stats_by_stem[stem]
        exp_str = f"/{rep['tiles_expected']}" if rep['tiles_expected'] is not None else ''
        status_orig = 'OK' if rep['found_in_original_dir'] else 'dimensiones por defecto'
        print(
            f"  - {rep['file_name']} ({rep['width']}x{rep['height']}, {status_orig}): "
            f"{rep['tiles_present']}{exp_str} tiles | "
            f"{st['raw_annotations']} anotaciones brutas -> "
            f"{st['final_annotations']} finales ({st['merged_duplicates']} fusionadas)"
        )


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)