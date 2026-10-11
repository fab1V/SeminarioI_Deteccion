# services/ensemble_service.py
# Responsabilidad: combinar las probabilidades de BERT y RoBERTa
# en una predicción final ponderada.
#
# No contiene lógica de modelos de lenguaje.
# Solo matemáticas simples con los pesos definidos en config.py.

import config


class EnsembleService:
    """Combina las probabilidades de BERT y RoBERTa mediante promedio ponderado."""

    def __init__(self):
        self._load_weights()

    # ── Carga y validación de pesos ───────────────────────────────────────────

    def _load_weights(self):
        """
        Lee los pesos desde config.py y los normaliza si no suman exactamente 1.
        Esto garantiza que la fórmula siempre sea matemáticamente correcta.
        """
        bert_w    = config.BERT_WEIGHT
        roberta_w = config.ROBERTA_WEIGHT
        total     = bert_w + roberta_w

        if total <= 0:
            raise ValueError(
                "La suma de BERT_WEIGHT y ROBERTA_WEIGHT debe ser mayor que 0."
            )

        # Normalizar si no suman exactamente 1
        if abs(total - 1.0) > 1e-6:
            bert_w    = bert_w    / total
            roberta_w = roberta_w / total

        self.bert_weight    = bert_w
        self.roberta_weight = roberta_w

    # ── Combinación de probabilidades ─────────────────────────────────────────

    def combine(self, bert_probability_ai: float, roberta_probability_ai: float) -> dict:
        """
        Combina las probabilidades de IA de cada modelo.

        Fórmula:
            final_probability_ai = (BERT_WEIGHT * bert_probability_ai)
                                 + (ROBERTA_WEIGHT * roberta_probability_ai)

        Retorna:
            {
                "prediction":         "IA" | "Humano",
                "probability_ai":     float,
                "probability_human":  float
            }
        """

        # Promedio ponderado de la probabilidad de IA
        final_probability_ai = (
            self.bert_weight    * bert_probability_ai
            + self.roberta_weight * roberta_probability_ai
        )

        # La probabilidad humana es el complemento
        final_probability_human = 1.0 - final_probability_ai

        # Clasificación final según el umbral configurado
        if final_probability_ai >= config.THRESHOLD:
            prediction = "IA"
        else:
            prediction = "Humano"

        return {
            "prediction":        prediction,
            "probability_ai":    round(final_probability_ai,    4),
            "probability_human": round(final_probability_human, 4),
        }


# Instancia única compartida por toda la aplicación.
ensemble_service = EnsembleService()
