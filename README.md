## **Instalación en Windows - UV**

1. Instalar UV
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Verificar instalación UV
```bash
uv --version
```

3. Instalar python 3.11
```bash
uv python install 3.11
```

4. Revisar versiones instaladas de python
```bash
uv python list
```

## **Instalación Low-Res (10m) to High-Res (2.5m)**

1. Clonar repo
```bash
git clone https://github.com/luzarin/LR_to_SR.git
```

2. Cambiar dir al repo 
```bash
cd LR_to_SR
```

3. Crear un virtual environment con versión en específico
```bash
uv venv --python 3.11
```

4. Activar virtual environment
```bash
.\.venv\Scripts\activate
```

5. Install OpenSR
```bash
pip install opensr-utils opensr-model
```

6. Install requirements
```bash
uv pip install -r requirements.txt
```

7. Instalar libs restantes (según CPU o GPU)
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

8. Si instalaste GPU revisar instalación
```bash
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

9. Correr APP
```bash
streamlit run app.py
```
---
## **Instalación en WSL - Pyenv**

1. Clonar repo
```
git clone https://github.com/luzarin/LR_to_SR.git
cd LR_to_SR
```

2. Instalar Python 3.11.9
```
pyenv install 3.11.9
```

3. Definir python local del proyecto
```
cd ruta
pyenv local 3.11.9
python --version
```

4. Crear y activar el venv
```
python -m venv .venv
source .venv/bin/activate
```

5. Actualizar pip dentro del venv
```
pip install -U pip setuptools wheel
```

6. Instalar OpenSR
```
pip install opensr-utils opensr-model
```

7. Instalar requirements
```
pip install -r requirements.txt
```

8. Instalar PyTorch (GPU o CPU)
```
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```
```
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

9. Verificar instalación GPU
```
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

10. Correr app
```
streamlit run app.py
```