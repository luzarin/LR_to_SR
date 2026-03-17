# Low-Res to High-Res Sentinel-2 (Linux / WSL2)

Herramienta web para mejorar la resolución espacial de imágenes satelitales Sentinel-2 de Low-Res (10m) a Super-Res (2.5m), con salida GeoTIFF de 10 bandas.

## Entrada y salida

| Tipo | Formato | Notas |
|---|---|---|
| **Input** | GeoTIFF multibanda | Debe tener 10 bandas (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12) |
| **Output** | GeoTIFF multibanda | Mantiene metadata y descripciones de bandas |

## Requisitos

- Linux o WSL2
- Python 3.11.9 (recomendado con `pyenv`)
- Bash
- GPU NVIDIA opcional (CUDA) para acelerar inferencia

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/luzarin/LR_to_SR_S2.git
cd LR_to_SR_S2

# 2. Instalar y fijar versión de Python
pyenv install 3.11.9
pyenv local 3.11.9

# 3. Crear entorno virtual y activarlo
python -m venv .venv
source .venv/bin/activate

# 4. Instalar dependencias
pip install -U pip setuptools wheel
pip install opensr-utils opensr-model
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Verificación GPU (opcional)

```bash
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

## Uso

```bash
uv run uvicorn app:app --reload --host 127.0.0.1 --port 8010
```

Abrir en el navegador: [http://127.0.0.1:8010](http://127.0.0.1:8010)

### Flujo básico

1. Colocar los GeoTIFF de 10 bandas en `input_10bands_LR/`.
2. Abrir la UI y refrescar la lista de archivos.
3. Elegir `Device Mode` (`auto`, `cuda` o `cpu`).
4. Ajustar parámetros (`factor`, `patch`, `pad`, `batch`) si hace falta.
5. Ejecutar el proceso y revisar el progreso en tiempo real.
6. El resultado se guarda en `output_10bands_SR/`.

## Parámetros principales

| Parámetro | Default | Descripción |
|---|---|---|
| `factor` | `4` | Escala espacial (10m -> 2.5m cuando factor=4) |
| `patch` | `128` | Tamaño del tile de inferencia |
| `pad` | `4` | Padding para evitar bordes entre tiles |
| `batch` | `7` | Cantidad de tiles por batch |
| `device` | `auto` | Selección de cómputo (`auto`/`cuda`/`cpu`) |

## Estructura

```text
LR_to_SR_S2/
├── app.py                    # API FastAPI + motor de inferencia
├── static/
│   └── index.html            # Interfaz web
├── SEN2SRLite/               # Pesos .safetensor y loader
├── sen2sr/                   # Módulos/modelos auxiliares
├── input_10bands_LR/         # Inputs GeoTIFF (10 bandas)
├── output_10bands_SR/        # Outputs GeoTIFF super-resueltos
└── requirements.txt          # Dependencias
```

## API

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Sirve la UI |
| `GET` | `/api/device_info` | Informa disponibilidad CUDA y nombre de GPU |
| `GET` | `/api/tifs` | Lista TIFF disponibles en la carpeta input |
| `GET` | `/api/run` | Ejecuta inferencia SR y stream de progreso (SSE) |
