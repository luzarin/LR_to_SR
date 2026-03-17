# SEN2SRLite (Windows PowerShell)

Herramienta web para mejorar la resolución espacial de imágenes satelitales de Low-Res (10m) a Super-Res (2.5m) utilizando OpenSR y exportarlas en formato GeoTIFF.

## Instalación

Se recomienda el uso de **`uv`** para instalar una versión específica de Python.

```powershell
# 1. Instalar UV
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Instalar versión del proyecto
uv python install 3.11.9

# 3. Clonar el repositorio
git clone https://github.com/luzarin/LR_to_SR.git
cd LR_to_SR

# 4. Asignar versión, crear entorno virtual y activarlo
uv python pin 3.11.9
uv run python -m venv .venv
.\.venv\Scripts\activate

# 5. Instalar dependencias
uv pip install opensr-utils opensr-model
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
```

```powershell
# 6. Verificar si PyTorch está instalado con GPU
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

```powershell
# 7. Correr la app
uv run python -m uvicorn app:app --reload --port 8010
```