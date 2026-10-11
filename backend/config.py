# config.py
# Configuración central del backend.
# Todos los parámetros ajustables se encuentran aquí.
# No distribuir estos valores en otros archivos.

# ── Rutas de los modelos ──────────────────────────────────────────────────────
# Carpetas donde se copiarán los modelos entrenados desde Google Colab.
BERT_MODEL_PATH = "models/bert"
ROBERTA_MODEL_PATH = "models/roberta"

# ── Pesos del ensemble ────────────────────────────────────────────────────────
# Valores TEMPORALES mientras no se tengan métricas de validación reales.
# Fórmula futura:
#   BERT_WEIGHT    = F1_BERT    / (F1_BERT + F1_RoBERTa)
#   ROBERTA_WEIGHT = F1_RoBERTa / (F1_BERT + F1_RoBERTa)
BERT_WEIGHT = 0.5
ROBERTA_WEIGHT = 0.5

# ── Umbral de clasificación ───────────────────────────────────────────────────
# Si la probabilidad de IA >= THRESHOLD  →  "IA"
# Si la probabilidad de IA <  THRESHOLD  →  "Humano"
# Ajustable después de analizar los resultados de validación.
THRESHOLD = 0.5

# ── Tokenización ─────────────────────────────────────────────────────────────
# Longitud máxima de tokens que aceptará cada modelo.
MAX_LENGTH = 256

# ── Modo simulado ────────────────────────────────────────────────────────────
# MOCK_MODE = True   →  respuestas simuladas, no se carga ningún modelo real.
# MOCK_MODE = False  →  se cargan BERT y RoBERTa desde las carpetas definidas arriba.
#
# Cambiar a False únicamente cuando las carpetas de modelos estén pobladas.
MOCK_MODE = True
