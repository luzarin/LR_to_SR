## **Instalación Low-Res (10m) to High-Res (2.5m)**

1. Instalar UV (virtual environments)
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

5. Clonar repo
```bash
git clone https://github.com/luzarin/LR_to_SR.git
```

6. Cambiar dir al repo 
```bash
cd LR_to_SR
```

7. Crear un virtual environment con versión en específico
```bash
uv venv --python 3.11
```

8. Activar virtual environment
```bash
.\.venv\Scripts\activate
```

9. Install requirements
```bash
uv pip install -r requirements.txt
```

10. Instalar libs restantes (según CPU o GPU)
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

11. Si instalaste GPU revisar instalación
```bash
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```
