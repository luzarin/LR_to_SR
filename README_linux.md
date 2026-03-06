# SEN2SRLite (Linux / WSL2)

Herramienta web para mejorar la resolución espacial de imágenes satelitales de Low-Res (10m) a Super-Res (2.5m) utilizando OpenSR y exportarlas en formato GeoTIFF.

## Instalación

Se recomienda instalar las dependencias siguiendo estos pasos:

```bash
# 1. Clonar el repositorio
git clone https://github.com/luzarin/LR_to_SR.git
cd LR_to_SR

# 2. Instalar la versión del proyecto de Python
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

```bash
# 5. Verificar si PyTorch está instalado con GPU
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

```bash
# 6. Correr la app
uvicorn app:app --reload
```
