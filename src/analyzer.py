import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.models import ContractAnalysis

load_dotenv(Path(__file__).parent.parent / ".env")

SYSTEM_PROMPT = """You are a financial contract analyst specialized in Latin American markets.
Your job is to analyze financial documents and extract structured information.

You must respond ONLY with a valid JSON object. No explanations, no markdown, no code blocks.
Just the raw JSON object.

The JSON must follow this exact structure:
{
    "document_type": "string (e.g. loan agreement, lease, insurance)",
    "parties": [
        {"name": "string", "role": "lender|borrower|guarantor"}
    ],
    "loan_amount": number or null,
    "currency": "USD|EUR|COP|GBP|MXN or null",
    "interest_rate": number or null,
    "term_months": integer or null,
    "start_date": "YYYY-MM-DD or null",
    "end_date": "YYYY-MM-DD or null",
    "risk_clauses": [
        {
            "clause": "short name of the clause",
            "severity": "high|medium|low",
            "description": "why this is a risk"
        }
    ],
    "summary": "2-3 sentence plain language summary of the contract"
}"""


class ContractAnalyzer:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def analyze(self, document_text: str) -> ContractAnalysis:
        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze this financial document:\n\n{document_text}"
                }
            ]
        )

        raw_text = response.content[0].text

        try:
            raw_json = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Claude returned invalid JSON: {e}\nRaw response: {raw_text}")

        return ContractAnalysis(**raw_json)

    def analyze_file(self, filepath: str) -> ContractAnalysis:
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        return self.analyze(text)

    def save_result(self, analysis: ContractAnalysis, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(analysis.model_dump_json(indent=2))

        print(f"Result saved to {path.resolve()}")