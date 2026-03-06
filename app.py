import os, sys, time, json
from pathlib import Path

import torch
import numpy as np
import rasterio
from rasterio.windows import Window
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Para importar SEN2SRLite.load cuando corres desde SR/
sys.path.insert(0, ".")
import SEN2SRLite.load as loader

BAND_KEYS = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]

app = FastAPI(title="SEN2SRLite API")

# Aseguramos que exista static
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    return FileResponse("static/index.html")


# -------------------------- Utils --------------------------
def list_tifs(folder: str):
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return []

    files = []
    for x in p.iterdir():
        if x.is_file() and x.suffix.lower() in {".tif", ".tiff"}:
            files.append(x.resolve())

    uniq = sorted({str(x): x for x in files}.keys(), key=lambda s: Path(s).name.lower())
    return uniq


def safe_delete(path: str):
    if os.path.exists(path):
        os.remove(path)


def pad_to_patch(x: np.ndarray, patch: int):
    c, h, w = x.shape
    out = np.zeros((c, patch, patch), dtype=x.dtype)
    out[:, :h, :w] = x
    return out, (h, w)


def normalize_input(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mx = float(np.nanmax(x)) if x.size else 0.0
    if mx > 1.5:  # típico Sentinel en 0..10000
        x = x / 10000.0
    return x


def denormalize_output(y: np.ndarray) -> np.ndarray:
    mx = float(np.nanmax(y)) if y.size else 0.0
    if mx <= 1.5:   # típico si está en [0,1]
        y = y * 10000.0
    y = np.clip(y, 0.0, 10000.0)
    return y.astype(np.uint16)


def resolve_device(device_mode: str) -> str:
    if device_mode == "cpu":
        return "cpu"
    if device_mode.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Seleccionaste GPU pero torch no detecta CUDA.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(weights_dir: Path, device: str):
    obj = loader.trainable_model(weights_dir, device=device)

    if callable(obj):
        return obj
    if isinstance(obj, dict):
        for k in ["model", "sr_model", "net", "network", "sr"]:
            if k in obj and callable(obj[k]):
                return obj[k]
        for v in obj.values():
            if callable(v):
                return v
    if isinstance(obj, (tuple, list)):
        for v in obj:
            if callable(v):
                return v
    raise RuntimeError(f"trainable_model() returned unsupported type: {type(obj)}")


@torch.no_grad()
def run_batch(model, device, batch_x_np):
    x = np.stack(batch_x_np, axis=0)  # (B,C,H,W)
    xt = torch.from_numpy(x).to(device)
    yt = model(xt)
    if isinstance(yt, (tuple, list)):
        yt = yt[0]
    if isinstance(yt, dict):
        yt = yt.get("raster_pixels", next(iter(yt.values())))
    y = yt.detach().cpu().numpy()
    return [y[i] for i in range(y.shape[0])]


# -------------------------- Inference Generator --------------------------
def run_sr_generator(
    weights_dir: Path,
    in_tif: str,
    out_tif: str,
    factor: int,
    patch: int,
    pad: int,
    batch: int,
    device_mode: str,
    ui_update_every: float = 0.25,
):
    start_t = time.time()
    step = patch - 2 * pad
    if step <= 0:
        yield {"error": "PATCH - 2*PAD must be > 0."}
        return

    try:
        device = resolve_device(device_mode)
        yield {"log": f"Dispositivo resuelto: {device}"}

        model = build_model(weights_dir, device)
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "to"):
            model.to(device)

        os.makedirs(os.path.dirname(out_tif) or ".", exist_ok=True)
        safe_delete(out_tif)

        with rasterio.open(in_tif) as src:
            if src.count != 10:
                yield {"error": f"Expected 10 bands input. Got {src.count}."}
                return

            src_desc = list(src.descriptions) if src.descriptions else [None] * src.count
            band_names = []
            for i in range(src.count):
                d = src_desc[i] if i < len(src_desc) else None
                if d is None or str(d).strip() == "":
                    d = BAND_KEYS[i] if i < len(BAND_KEYS) else f"Band {i+1}"
                band_names.append(str(d))

            global_tags = src.tags()
            per_band_tags = {i: src.tags(i) for i in range(1, src.count + 1)}

            profile = src.profile.copy()
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
            profile.pop("tiled", None)

            profile.update(
                driver="GTiff",
                dtype="uint16",
                count=src.count,
                width=src.width * factor,
                height=src.height * factor,
                transform=src.transform * src.transform.scale(1 / factor, 1 / factor),
                tiled=True,
                blockxsize=256,
                blockysize=256,
                compress="deflate",
                predictor=2,
                BIGTIFF="IF_SAFER",
            )

            total_tiles = ((src.height + step - 1) // step) * ((src.width + step - 1) // step)

            done_tiles = 0
            last_draw = 0.0

            yield {
                "progress": 0.0,
                "status": f"Iniciando... Total tiles: {total_tiles} | device={device}",
                "stage": "running"
            }

            with rasterio.open(out_tif, "w", **profile) as dst:
                if global_tags:
                    dst.update_tags(**global_tags)

                for i, name in enumerate(band_names, start=1):
                    dst.set_band_description(i, name)
                    tags_i = per_band_tags.get(i)
                    if tags_i:
                        dst.update_tags(i, **tags_i)

                batch_tiles = []
                batch_meta = []

                def flush():
                    nonlocal batch_tiles, batch_meta
                    if not batch_tiles:
                        return

                    outs = run_batch(model, device, batch_tiles)

                    for out_sr, meta in zip(outs, batch_meta):
                        (c0, r0, w, h, is_left, is_top, is_right, is_bottom) = meta

                        left = 0 if is_left else pad
                        top = 0 if is_top else pad
                        right = w if is_right else (w - pad)
                        bottom = h if is_bottom else (h - pad)

                        sr_left = left * factor
                        sr_top = top * factor
                        sr_right = right * factor
                        sr_bottom = bottom * factor

                        out_crop = out_sr[:, sr_top:sr_bottom, sr_left:sr_right]
                        out_crop = denormalize_output(out_crop)

                        out_win = Window(
                            (c0 + left) * factor,
                            (r0 + top) * factor,
                            (right - left) * factor,
                            (bottom - top) * factor,
                        )

                        dst.write(out_crop, window=out_win)

                    batch_tiles = []
                    batch_meta = []

                for r0 in range(0, src.height, step):
                    for c0 in range(0, src.width, step):
                        w = min(patch, src.width - c0)
                        h = min(patch, src.height - r0)

                        win = Window(c0, r0, w, h)
                        x = src.read(window=win)
                        x = normalize_input(x)
                        xpad, (hh, ww) = pad_to_patch(x, patch)

                        is_left = (c0 == 0)
                        is_top = (r0 == 0)
                        is_right = (c0 + w >= src.width)
                        is_bottom = (r0 + h >= src.height)

                        batch_tiles.append(xpad)
                        batch_meta.append((c0, r0, ww, hh, is_left, is_top, is_right, is_bottom))

                        done_tiles += 1
                        now = time.time()

                        if now - last_draw > ui_update_every:
                            pct = done_tiles / total_tiles if total_tiles else 1.0
                            elapsed = now - start_t
                            rate = done_tiles / elapsed if elapsed > 0 else 0.0
                            eta = (total_tiles - done_tiles) / rate if rate > 0 else 0.0

                            status_msg = (
                                f"device={device} | tiles {done_tiles}/{total_tiles} | "
                                f"{rate:.2f} tiles/s | ETA {int(eta//60):02d}:{int(eta%60):02d}"
                            )
                            yield {"progress": pct * 100, "status": status_msg, "stage": "running"}
                            last_draw = now

                        if len(batch_tiles) == batch:
                            flush()

                flush()

            yield {
                "progress": 100.0,
                "status": f"Completado. Guardado en {out_tif}",
                "stage": "completed"
            }

    except Exception as e:
        yield {"error": str(e), "stage": "error"}


# -------------------------- API Endpoints --------------------------
@app.get("/api/tifs")
def api_get_tifs(folder: str = "./input_10bands_LR"):
    return {"files": [Path(f).name for f in list_tifs(folder)]}

@app.get("/api/run")
def api_run_get(
    weights_dir: str = "./SEN2SRLite",
    in_tif: str = "",
    out_dir: str = "./output_10bands_SR",
    out_name: str = "",
    factor: int = 4,
    patch: int = 128,
    pad: int = 4,
    batch: int = 7,
    device: str = "auto"
):
    if not Path(weights_dir).exists():
        return StreamingResponse((f"data: {json.dumps({'error': f'No existe WEIGHTS_DIR: {weights_dir}'})}\n\n" for _ in range(1)), media_type="text/event-stream")
    if not Path(in_tif).exists():
        return StreamingResponse((f"data: {json.dumps({'error': f'No existe Input: {in_tif}'})}\n\n" for _ in range(1)), media_type="text/event-stream")
    
    out_path = str(Path(out_dir) / out_name)

    def event_generator():
        for event_data in run_sr_generator(Path(weights_dir), in_tif, out_path, factor, patch, pad, batch, device):
            yield f"data: {json.dumps(event_data)}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/device_info")
def api_device_info():
    cuda_av = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_av else "N/A"
    return {
        "cuda_available": cuda_av,
        "gpu_name": gpu_name
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
