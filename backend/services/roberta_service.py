# services/roberta_service.py
# Responsabilidad: cargar el modelo RoBERTa y clasificar un texto.
#
# Proceso idéntico al de bert_service.py pero usando la carpeta models/roberta/
# y el tokenizer propio de RoBERTa.
#
# En MOCK_MODE = True  →  devuelve probabilidades fijas de ejemplo.
# En MOCK_MODE = False →  carga el modelo real desde models/roberta/ y hace inferencia.

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config


class RobertaService:
    """Encapsula la carga y la inferencia del modelo RoBERTa."""

    def __init__(self):
        self.mock_mode = config.MOCK_MODE
        self.model = None
        self.tokenizer = None
        self.device = None

        if not self.mock_mode:
            self._load_model()

    # ── Carga del modelo (solo cuando MOCK_MODE = False) ─────────────────────

    def _load_model(self):
        """Carga tokenizer y modelo desde la carpeta configurada en config.py."""

        model_path = config.ROBERTA_MODEL_PATH

        # Verificar que la carpeta existe y no está vacía
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Carpeta de modelo RoBERTa no encontrada: '{model_path}'. "
                "Copia los archivos del modelo entrenado en esa ruta."
            )

        archivos = os.listdir(model_path)
        if not archivos:
            raise FileNotFoundError(
                f"La carpeta '{model_path}' está vacía. "
                "Copia los archivos del modelo RoBERTa entrenado."
            )

        # Detectar dispositivo: GPU si está disponible, CPU en caso contrario
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # Cargar el tokenizer propio del modelo RoBERTa entrenado
            # RoBERTa usa vocab.json + merges.txt, distinto al vocab.txt de BERT
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,   # nunca descargar de Internet
            )

            # Cargar el modelo de clasificación binaria
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True,
            )

            self.model.to(self.device)
            self.model.eval()  # desactivar dropout, batch norm, etc.

        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo RoBERTa: {e}")

    # ── Inferencia ────────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Clasifica el texto y devuelve las probabilidades.

        Retorna:
            {
                "probability_human": float,  # posición 0 del softmax
                "probability_ai":    float   # posición 1 del softmax
            }
        """

        if self.mock_mode:
            return self._mock_predict()

        return self._real_predict(text)

    def _mock_predict(self) -> dict:
        """Probabilidades simuladas para cuando los modelos no están disponibles."""
        return {
            "probability_human": 0.25,
            "probability_ai": 0.75,
        }

    def _real_predict(self, text: str) -> dict:
        """Inferencia real con el modelo RoBERTa cargado."""

        # Tokenizar usando el tokenizer de RoBERTa (no modificar el texto más allá de strip)
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=config.MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )

        # Mover los tensores al mismo dispositivo que el modelo
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits

            # Aplicar softmax para obtener probabilidades entre 0 y 1
            probabilities = torch.softmax(logits, dim=1)

            # Etiquetas: 0 = Humano, 1 = IA
            probability_human = probabilities[0][0].item()
            probability_ai    = probabilities[0][1].item()

            return {
                "probability_human": probability_human,
                "probability_ai":    probability_ai,
            }

        except Exception as e:
            raise RuntimeError(f"Error durante la inferencia de RoBERTa: {e}")

    # ── Información del servicio ──────────────────────────────────────────────

    def get_device_name(self) -> str:
        """Devuelve el nombre del dispositivo en uso."""
        if self.mock_mode:
            return "cpu"
        return str(self.device)


# Instancia única: se carga una sola vez al iniciar el backend.
roberta_service = RobertaService()
