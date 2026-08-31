import pytest
from src.models import ContractAnalysis, Party, RiskClause

def test_party_valid():
    party = Party(name="ABC Bank", role="lender")
    assert party.role == "lender"
    assert party.name == "ABC Bank"


def test_party_invalid_role():
    with pytest.raises(Exception):
        Party(name="ABC Bank", role="invalid_role")


def test_risk_clause_valid():
    clause = RiskClause(
        clause="Late Payment",
        severity="high",
        description="Very high penalty"
    )
    assert clause.severity == "high"


def test_risk_clause_invalid_severity():
    with pytest.raises(Exception):
        RiskClause(
            clause="Late Payment",
            severity="critical",
            description="Very high penalty"
        )


def test_contract_analysis_valid():
    analysis = ContractAnalysis(
        document_type="loan agreement",
        parties=[
            Party(name="ABC Bank", role="lender"),
            Party(name="John Smith", role="borrower"),
        ],
        loan_amount=50000.0,
        currency="USD",
        interest_rate=12.5,
        term_months=36,
        start_date="2024-01-15",
        end_date="2027-01-15",
        risk_clauses=[
            RiskClause(
                clause="Late Payment",
                severity="high",
                description="High penalty"
            )
        ],
        summary="A 36-month loan agreement."
    )
    assert analysis.loan_amount == 50000.0
    assert analysis.currency == "USD"
    assert len(analysis.parties) == 2
    assert len(analysis.risk_clauses) == 1


def test_contract_analysis_invalid_currency():
    with pytest.raises(Exception):
        ContractAnalysis(
            document_type="loan agreement",
            parties=[],
            loan_amount=50000.0,
            currency="XYZ",
            interest_rate=12.5,
            term_months=36,
            start_date="2024-01-15",
            end_date="2027-01-15",
            risk_clauses=[],
            summary="A loan agreement."
        )