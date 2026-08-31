import typer
from pathlib import Path
from src.analyzer import ContractAnalyzer

app = typer.Typer()


@app.command()
def analyze(
    filepath: str = typer.Argument(..., help="Path to the contract file"),
    output: str = typer.Option(None, "--output", "-o", help="Path to save JSON result"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full analysis details"),
):
    """Analyze a financial contract and extract structured information."""

    typer.echo(f"Analyzing: {filepath}")

    analyzer = ContractAnalyzer()

    try:
        analysis = analyzer.analyze_file(filepath)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"Analysis error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nDocument type: {analysis.document_type}")
    typer.echo(f"Summary: {analysis.summary}")

    # Partes del contrato
    typer.echo("\nParties:")
    for party in analysis.parties:
        typer.echo(f"  {party.role.upper()}: {party.name}")

    # Términos financieros
    if analysis.loan_amount:
        typer.echo(f"\nLoan amount: {analysis.currency} {analysis.loan_amount:,.2f}")
    if analysis.interest_rate:
        typer.echo(f"Interest rate: {analysis.interest_rate}%")
    if analysis.term_months:
        typer.echo(f"Term: {analysis.term_months} months")
    if analysis.start_date:
        typer.echo(f"Start date: {analysis.start_date}")
    if analysis.end_date:
        typer.echo(f"End date: {analysis.end_date}")

    # Cláusulas de riesgo
    typer.echo(f"\nRisk clauses ({len(analysis.risk_clauses)} found):")
    for risk in analysis.risk_clauses:
        icon = "🔴" if risk.severity == "high" else "🟡" if risk.severity == "medium" else "🟢"
        typer.echo(f"  {icon} [{risk.severity.upper()}] {risk.clause}")
        if verbose:
            typer.echo(f"      {risk.description}")

    # Guardá el resultado si se especificó output
    if output:
        analyzer.save_result(analysis, output)


@app.command()
def version():
    """Show the version of the analyzer."""
    typer.echo("Financial Document Analyzer v0.1.0")


if __name__ == "__main__":
    app()