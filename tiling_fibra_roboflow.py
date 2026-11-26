#!/usr/bin/env python3
"""Genera tiles PNG para Roboflow sin modificar imágenes ni metadatos originales.
Requiere Pillow: python -m pip install Pillow
Ejemplo:
  python tiling_fibra_roboflow.py --input . --output ../Dataset_lavado_tiles640 --dry-run
  python tiling_fibra_roboflow.py --input . --output ../Dataset_lavado_tiles640
Solo lee S-*/images/FLA_*.png. La salida debe ser nueva.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image


def positions(length, size, overlap):
    if length < size:
        raise ValueError(f'Dimensión {length} menor que tile {size}; no se rellena ni escala.')
    result = list(range(0, length - size + 1, size - overlap))
    if result[-1] != length - size:
        result.append(length - size)
    return result


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True, help='Carpeta Dataset_lavado')
    parser.add_argument('--output', type=Path, required=True, help='Carpeta NUEVA para los resultados')
    parser.add_argument('--tile-size', type=int, default=640)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--session', action='append', help='Opcional: sesión exacta; repetible')
    parser.add_argument('--exclude-file', type=Path, help='TXT opcional con rutas relativas de imágenes a excluir, una por línea')
    parser.add_argument('--dry-run', action='store_true', help='Valida archivos y cuenta tiles, sin escribir')
    a = parser.parse_args()
    if a.tile_size <= 0 or not 0 <= a.overlap < a.tile_size:
        parser.error('Debe cumplirse: tile-size > 0 y 0 <= overlap < tile-size.')
    src, dst = a.input.expanduser().resolve(), a.output.expanduser().resolve()
    if not src.is_dir():
        parser.error(f'No existe la carpeta de entrada: {src}')
    # Mantener los derivados fuera de las sesiones de adquisición.
    if dst == src or src in dst.parents:
        parser.error('La salida debe estar fuera de Dataset_lavado, por ejemplo ../Dataset_lavado_tiles640.')
    if dst.exists():
        parser.error(f'La salida ya existe: {dst}. Usa otra carpeta; no se sobrescribe ningún resultado.')
    staging = dst.with_name(dst.name + '.incomplete')
    if staging.exists() and not a.dry_run:
        parser.error(f'Hay una ejecución incompleta: {staging}. Usa otro nombre de salida o revisa esa carpeta.')
    excluded = set()
    if a.exclude_file:
        excluded = {x.strip() for x in a.exclude_file.read_text(encoding='utf-8-sig').splitlines()
                    if x.strip() and not x.lstrip().startswith('#')}
    sessions = sorted(p for p in src.glob('S-*') if p.is_dir())
    if a.session:
        missing = set(a.session) - {p.name for p in sessions}
        if missing:
            parser.error('Sesiones no encontradas: ' + ', '.join(sorted(missing)))
        sessions = [p for p in sessions if p.name in a.session]
    candidates = [p for s in sessions for p in sorted((s / 'images').glob('FLA_*.png')) if p.is_file()]
    known = {p.relative_to(src).as_posix() for p in candidates}
    unknown = excluded - known
    if unknown:
        parser.error('Exclusiones no encontradas en las sesiones seleccionadas: ' + ', '.join(sorted(unknown)))
    files = [p for p in candidates if p.relative_to(src).as_posix() not in excluded]
    if not files:
        parser.error('No se encontraron imágenes S-*/images/FLA_*.png para procesar.')
    if len({p.stem for p in files}) != len(files):
        parser.error('Hay nombres base repetidos entre sesiones; resolver antes de subir al mismo proyecto.')
    plan, totals = [], Counter()
    print('Validando imágenes originales...', flush=True)
    for f in files:
        with Image.open(f) as im:
            im.load()
            if im.format != 'PNG' or im.mode not in ('RGB', 'L', 'RGBA'):
                raise ValueError(f'Formato o modo no admitido sin conversión: {f.name} ({im.format}, {im.mode})')
            if im.getexif().get(274, 1) != 1:
                raise ValueError(f'Orientación EXIF distinta de 1 en {f.name}; revisar antes de recortar.')
            w, h = im.size
            xs = positions(w, a.tile_size, a.overlap)
            ys = positions(h, a.tile_size, a.overlap)
        session = f.parent.parent.name
        plan.append((f, session, w, h, xs, ys, sha256(f)))
        totals[session] += len(xs) * len(ys)
    for session, n in sorted(totals.items()):
        originals = sum(1 for item in plan if item[1] == session)
        print(f'{session}: {originals} originales -> {n} tiles')
    print(f'TOTAL: {len(plan)} originales -> {sum(totals.values())} tiles PNG')
    print(f'Excluidos explícitamente: {len(excluded)}. No se leen estados de metadata.csv.')
    if a.dry_run:
        print('Simulación finalizada. No se ha creado ningún archivo.')
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    fields = ['tile_file', 'source_file', 'session', 'source_sha256', 'source_width', 'source_height',
              'x0', 'y0', 'x1_exclusive', 'y1_exclusive', 'tile_width', 'tile_height', 'tile_sha256']
    with (staging / 'manifest_tiles.csv').open('w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for index, (f, session, w, h, xs, ys, digest) in enumerate(plan, 1):
            if sha256(f) != digest:
                raise RuntimeError(f'El original cambió durante el proceso: {f}')
            folder = staging / 'images' / session
            folder.mkdir(parents=True, exist_ok=True)
            with Image.open(f) as im:
                im.load()
                for y in ys:
                    for x in xs:
                        name = f'{f.stem}_x{x:04d}_y{y:04d}.png'
                        dest = folder / name
                        tile = im.crop((x, y, x + a.tile_size, y + a.tile_size))
                        tile.save(dest, format='PNG', compress_level=3)
                        writer.writerow(dict(zip(fields, [
                            dest.relative_to(staging).as_posix(), f.relative_to(src).as_posix(), session,
                            digest, w, h, x, y, x+a.tile_size, y+a.tile_size,
                            a.tile_size, a.tile_size, sha256(dest)])))
            print(f'[{index}/{len(plan)}] {f.name}: {len(xs)*len(ys)} tiles', flush=True)
    config = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_root': str(src),
              'tile_size': a.tile_size, 'nominal_overlap': a.overlap,
              'edge_policy': 'shift_last_window_to_edge', 'resize': False,
              'images': len(plan), 'tiles': sum(totals.values()), 'tiles_by_session': dict(totals),
              'excluded_files': sorted(excluded), 'split': 'not_assigned'}
    (staging / 'tiling_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (staging / 'LEEME.txt').write_text(
        'Subir a Roboflow únicamente los PNG de images/<sesion>/.\n'
        'Los tiles están SIN ANOTAR. No son negativos confirmados.\n'
        'No se aplicaron filtros por contenido ni división train/valid/test.\n'
        'Las posiciones del CSV usan origen superior izquierdo, x horizontal e y vertical.\n'
        'x1_exclusive e y1_exclusive son los límites exclusivos del recorte.\n'
        'En los últimos tiles el solapamiento puede superar el nominal.\n'
        'Todos los tiles relacionados con una misma muestra deben permanecer en una sola partición.\n'
        'Las pasadas repetidas de una misma porción también están relacionadas.\n'
        'Los conteos de anotaciones entre tiles incluyen duplicados del solapamiento.\n'
        'Originales y metadatos de captura permanecen intactos.\n', encoding='utf-8')
    if dst.exists():
        raise FileExistsError(f'La salida apareció durante el proceso: {dst}')
    staging.rename(dst)
    print(f'\nFinalizado: {dst}\nSubir PNG desde: {dst / "images"}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as e:
        print(f'ERROR: {e}\nLos originales no se modifican. Si hay una carpeta .incomplete, no la subas.', file=sys.stderr)
        sys.exit(1)
