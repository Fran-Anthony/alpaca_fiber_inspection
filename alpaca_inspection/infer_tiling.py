"""Predice impurezas en imágenes, video, cámara o RTSP."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2

from .tiler_detector import CLASS_NAMES, DEFAULT_WEIGHTS, TILE_SIZE, TiledImpurityDetector, render_detections

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_paths(source):
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return source.parent, [source]
    if source.is_dir():
        images = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        images.extend(source.glob("S-*/images/FLA_*.png"))
        if images:
            return source, sorted(images)
    raise ValueError(f"No se encontraron imágenes en {source}")


def process_images(source, output, predict, save_vis):
    root, paths = image_paths(source)
    output.mkdir(parents=True, exist_ok=False)
    coco = dict(images=[], annotations=[], categories=[
        {"id": cls, "name": name, "supercategory": "impurity"} for cls, name in CLASS_NAMES.items()
    ])
    summary = []
    for image_id, path in enumerate(paths, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"No se pudo leer {path}")
        try:
            detections, metrics = predict(frame, "cross_tile_merge")
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        height, width = frame.shape[:2]
        name = path.relative_to(root).as_posix()
        coco["images"].append(dict(id=image_id, file_name=name, width=width, height=height, extra={
            "stem": path.stem, "tiles_present": len(metrics["tiles"]),
            "tiles": [dict(x0=x, y0=y, width=TILE_SIZE, height=TILE_SIZE) for x, y in metrics["tiles"]],
        }))
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox_xyxy"]
            coco["annotations"].append(dict(
                id=len(coco["annotations"]) + 1, image_id=image_id, category_id=detection["class_id"],
                bbox=[x1, y1, x2 - x1, y2 - y1], area=(x2 - x1) * (y2 - y1),
                iscrowd=0, score=detection["score"], extra={
                    "merged_count": detection["merged_count"], "source_tiles": detection["source_tiles"],
                },
            ))
        particles = sum(d["class_id"] == 0 for d in detections)
        summary.append(dict(image=name, width=width, height=height, num_tiles=len(metrics["tiles"]),
                            particle_count=particles, straw_count=len(detections) - particles,
                            total_impurities=len(detections), total_ms=round(metrics["total_ms"], 2)))
        if save_vis:
            for folder, audit in (("modelo_final", False), ("auditoria", True)):
                target = output / folder / name
                target = target.with_name(target.name + ".png")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(target), render_detections(frame, detections, metrics, audit=audit)):
                    raise OSError(f"No se pudo guardar {target}")
    (output / "predicciones.coco.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    with (output / "resumen.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def process_stream(source, predict, audit):
    if source == "mindvision":
        from .mindvision_video import MindVisionCamera

        camera = MindVisionCamera(rgb_gain=(142, 71, 120))
        camera.open()
    else:
        camera = cv2.VideoCapture(int(source) if source.isdecimal() else source)
    try:
        if not camera.isOpened():
            raise OSError(f"No se pudo abrir {source}")
        previous = perf_counter()
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            detections, metrics = predict(frame, "cross_tile_merge" if audit else "batched_nms")
            canvas = render_detections(frame, detections, metrics, audit=audit)
            now = perf_counter()
            fps = 1 / (now - previous)
            previous = now
            line = f"{fps:.1f} FPS | {metrics['total_ms']:.0f} ms | {len(detections)} cajas"
            cv2.putText(canvas, line, (10, 54 if audit else 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Inspección de impurezas", canvas)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images", type=Path, help="Imagen, carpeta plana o carpeta con sesiones S-*.")
    inputs.add_argument("--stream", help="Video, índice de cámara OpenCV, mindvision o URL RTSP.")
    parser.add_argument("--output", type=Path, help="Carpeta nueva para los resultados de imágenes.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", help="Dispositivo para Ultralytics, por ejemplo cpu o 0.")
    parser.add_argument("--batch", type=int, default=20, help="Cantidad máxima de tiles por lote.")
    parser.add_argument("--engine-batch", type=int, help="Batch fijo de un engine, si se requiere indicarlo.")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--conf-particle", type=float, default=0.10)
    parser.add_argument("--conf-straw", type=float, default=0.10)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--dedup", choices=("cross_tile_merge", "batched_nms"))
    parser.add_argument("--merge", choices=("wbf", "union", "largest"), default="wbf")
    parser.add_argument("--class-agnostic", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--audit", action="store_true", help="Mostrar la cuadrícula y etiquetas en video.")
    args = parser.parse_args()
    if args.images and args.output is None:
        parser.error("--images requiere --output")

    try:
        detector = TiledImpurityDetector(
            args.weights, tile_batch_size=args.batch, device=args.device,
            half=not args.fp32, engine_batch_size=args.engine_batch,
        )

        def predict(frame, default_dedup):
            return detector.predict_frame(
                frame, conf_particle=args.conf_particle, conf_straw=args.conf_straw, roi=args.roi,
                dedup=args.dedup or default_dedup, merge=args.merge, class_agnostic=args.class_agnostic,
            )

        if args.images:
            process_images(args.images, args.output, predict, args.save_vis)
        else:
            process_stream(args.stream, predict, args.audit)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
