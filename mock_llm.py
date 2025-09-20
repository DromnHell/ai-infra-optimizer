import random
from typing import Dict

class MockLLM:
    """
    Simulates an LLM to test your pipeline without API calls.
    Returns a dummy but structured analysis.
    """

    def invoke(self, input: Dict) -> Dict:
        fake_findings = [
            "High CPU usage detected. Recommendation: distribute the load.",
            "Abnormal latency observed. Recommendation: check the API gateway.",
            "Memory saturated. Recommendation: increase RAM.",
        ]

        response = random.choice(fake_findings)

        return {"text": f"{response}"}
