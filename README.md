# Teoría de Aprendizaje de Máquina — 2026-2

Soluciones de talleres, parciales y actividades del curso.

## Estructura

```text
Actividades/      Actividades de clase
  1-Mnist/        Proyecto Python: dos modelos MNIST + servidor local
Parciales/        Parciales: enunciado y solución
Talleres/         Talleres: enunciado y solución
```

## Reproducir

Cada entregable es independiente: no existe un entorno único para todo el repo.
Cada carpeta se reproduce por separado.

> Requisito: **Python 3.12** (TensorFlow todavía no publica ruedas para 3.14). Si
> no está instalado, `uv python install 3.12` lo descarga; cada proyecto fija esa
> versión.

### Actividades/1-Mnist (proyecto Python)

Con [uv](https://docs.astral.sh/uv/) (recomendado):

```bash
cd Actividades/1-Mnist
uv sync                 # crea .venv con Python 3.12 (TF no soporta 3.14)
uv run mnist-serve      # http://127.0.0.1:5000
uv run pytest           # 11 tests
```

Con `venv` + `pip` (versiones fijadas en `requirements.txt`):

```bash
cd Actividades/1-Mnist
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mnist-serve
```

El detalle de la aplicación, el entrenamiento local y el entrenamiento en Kaggle
se encuentran en [Actividades/1-Mnist/README.md](Actividades/1-Mnist/README.md).

### Cuadernos (Talleres y Parciales)

Los cuadernos usan NumPy, Matplotlib, scikit-learn, PyTorch/torchvision y Jupyter:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install jupyterlab numpy matplotlib scikit-learn torch torchvision
jupyter lab
```

Sin GPU se recomienda la rueda CPU de PyTorch (mucho más liviana):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

El `.ipynb` de la carpeta del taller o parcial se abre en Jupyter y las celdas se
ejecutan en orden. Los enunciados originales no se versionan (viven en `.ref/`).

### Verificación

- Proyecto: `cd Actividades/1-Mnist && uv run pytest` (11 tests).
- Cuadernos: ejecutar las celdas y comparar con las salidas guardadas.
