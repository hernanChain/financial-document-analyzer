from pydantic import BaseModel, field_validator
from datetime import date


class Party(BaseModel):
    name: str
    role: str  # "lender", "borrower", "guarantor"

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        allowed = {"lender", "borrower", "guarantor"}
        if value.lower() not in allowed:
            raise ValueError(f"role debe ser uno de {allowed}")
        return value.lower()


class RiskClause(BaseModel):
    clause: str
    severity: str  # "high", "medium", "low"
    description: str

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, value: str) -> str:
        allowed = {"high", "medium", "low"}
        if value.lower() not in allowed:
            raise ValueError(f"severity debe ser uno de {allowed}")
        return value.lower()


class ContractAnalysis(BaseModel):
    document_type: str
    parties: list[Party]
    loan_amount: float | None
    currency: str | None
    interest_rate: float | None
    term_months: int | None
    start_date: str | None
    end_date: str | None
    risk_clauses: list[RiskClause]
    summary: str

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str | None) -> str | None:
        if value is None:
            return value
        valid = {"USD", "EUR", "COP", "GBP", "MXN"}
        if value.upper() not in valid:
            raise ValueError(f"currency inválida: {value}")
        return value.upper()