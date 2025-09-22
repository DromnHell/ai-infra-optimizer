import random
from typing import Dict

class MockLLM:
    """
    Simulates an LLM to test your pipeline without API calls.
    Returns a dummy but structured analysis.
    """

    def invoke(self, input: Dict):
        fake_findings = [
            "Utilisation élevée du processeur détectée. Recommandation : répartir la charge.",
            "Latence anormale observée. Recommandation : vérifier la passerelle API.",
            "Mémoire saturée. Recommandation : augmenter la RAM."
        ]

        response = random.choice(fake_findings)

        return response
