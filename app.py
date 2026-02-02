import os, sys, time
from pathlib import Path
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
import streamlit as st

# Para importar SEN2SRLite.load cuando corres desde SR/
sys.path.insert(0, ".")
import SEN2SRLite.load as loader


# -------------------------- Utils --------------------------
def list_tifs(folder: str):
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return []
    tifs = list(p.glob("*.tif")) + list(p.glob("*.tiff")) + list(p.glob("*.TIF")) + list(p.glob("*.TIFF"))
    return sorted([str(x) for x in tifs])


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
    """
    device_mode: "auto" | "cuda (GPU)" | "cpu"
    """
    if device_mode == "cpu":
        return "cpu"
    if device_mode.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Seleccionaste GPU pero torch no detecta CUDA.")
        return "cuda"
    # auto
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


def run_sr(
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
        raise ValueError("PATCH - 2*PAD must be > 0.")

    device = resolve_device(device_mode)

    model = build_model(weights_dir, device)
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "to"):
        model.to(device)

    os.makedirs(os.path.dirname(out_tif) or ".", exist_ok=True)
    safe_delete(out_tif)

    with rasterio.open(in_tif) as src:
        if src.count != 10:
            raise ValueError(f"Expected 10 bands input. Got {src.count}.")

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

        # UI placeholders
        pbar = st.progress(0)
        status = st.empty()
        details = st.empty()

        done_tiles = 0
        last_draw = 0.0

        with rasterio.open(out_tif, "w", **profile) as dst:
            batch_tiles = []
            batch_meta = []

            def flush():
                nonlocal batch_tiles, batch_meta
                if not batch_tiles:
                    return

                outs = run_batch(model, device, batch_tiles)

                for out_sr, meta in zip(outs, batch_meta):
                    (c0, r0, w, h, is_left, is_top, is_right, is_bottom) = meta

                    left   = 0 if is_left   else pad
                    top    = 0 if is_top    else pad
                    right  = w if is_right  else (w - pad)
                    bottom = h if is_bottom else (h - pad)

                    sr_left   = left * factor
                    sr_top    = top * factor
                    sr_right  = right * factor
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

                        pbar.progress(min(int(pct * 100), 100))
                        status.write(
                            f"device={device} | tiles {done_tiles}/{total_tiles} | {rate:.2f} tiles/s | ETA {int(eta//60):02d}:{int(eta%60):02d}"
                        )
                        last_draw = now

                    if len(batch_tiles) == batch:
                        flush()

            flush()

        pbar.progress(100)
        details.success(f"Listo. Output: {out_tif}")


# -------------------------- UI --------------------------
st.set_page_config(page_title="SEN2SRLite - SR Inference", layout="wide")
st.title("SEN2SRLite - Super-Resolución (GeoTIFF)")

with st.sidebar:
    st.header("Rutas")
    weights_dir = st.text_input("WEIGHTS_DIR", value="./SEN2SRLite")

    st.caption("Input: elige carpeta y luego un .tif. No se suben datos, se leen del disco local.")
    input_dir = st.text_input("Input folder", value="./input_10bands_LR")
    tifs = list_tifs(input_dir)

    if tifs:
        in_tif = st.selectbox("IN_TIF", tifs)
    else:
        st.warning("No encontré .tif/.tiff en esa carpeta. Escribe una ruta válida o pon el archivo manual.")
        in_tif = st.text_input("IN_TIF (manual)", value="./input_10bands_LR/sept_2024_10bands.tif")

    st.caption("Output: carpeta + nombre (se creará si no existe).")
    output_dir = st.text_input("Output folder", value="./output_10bands_SR")
    out_name = st.text_input("Output filename", value="sept_2024_SR.tif")
    out_tif = str(Path(output_dir) / out_name)

    st.header("Compute")
    st.caption(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
    device_mode = st.selectbox("Device", ["auto", "cuda (GPU)", "cpu"], index=0)

    st.header("Parámetros")
    factor = st.number_input("FACTOR", min_value=1, max_value=8, value=4, step=1)
    patch = st.number_input("PATCH", min_value=32, max_value=512, value=128, step=16)
    pad = st.number_input("PAD", min_value=0, max_value=64, value=4, step=1)
    batch = st.number_input("BATCH", min_value=1, max_value=64, value=7, step=1)

    run = st.button("Run", type="primary")

# Resumen fuera del sidebar
st.write("### Config actual")
st.code(
    f"WEIGHTS_DIR = {weights_dir}\n"
    f"INPUT_DIR   = {input_dir}\n"
    f"IN_TIF      = {in_tif}\n"
    f"OUT_TIF     = {out_tif}\n"
    f"DEVICE      = {device_mode}\n"
    f"FACTOR={int(factor)} PATCH={int(patch)} PAD={int(pad)} BATCH={int(batch)}"
)

if run:
    # Validaciones mínimas
    if not Path(weights_dir).exists():
        st.error(f"No existe WEIGHTS_DIR: {weights_dir}")
    elif not Path(in_tif).exists():
        st.error(f"No existe IN_TIF: {in_tif}")
    else:
        try:
            run_sr(
                weights_dir=Path(weights_dir),
                in_tif=in_tif,
                out_tif=out_tif,
                factor=int(factor),
                patch=int(patch),
                pad=int(pad),
                batch=int(batch),
                device_mode=device_mode,
            )
        except Exception as e:
            st.exception(e)
