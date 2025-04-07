"""
Utility functions for the CLI cybersecurity chatbot.
"""
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

def format_response(text):
    """Format the response text with highlighted keywords."""
    # List of important cybersecurity terms to highlight
    highlight_terms = {
        "phishing": "[bold red]phishing[/bold red]",
        "scam": "[bold red]scam[/bold red]",
        "password": "[bold yellow]password[/bold yellow]",
        "virus": "[bold red]virus[/bold red]",
        "malware": "[bold red]malware[/bold red]",
        "hacker": "[bold red]hacker[/bold red]",
        "security": "[bold green]security[/bold green]",
        "privacy": "[bold green]privacy[/bold green]",
        "data": "[bold blue]data[/bold blue]",
        "email": "[bold blue]email[/bold blue]",
        "link": "[bold yellow]link[/bold yellow]",
        "click": "[bold yellow]click[/bold yellow]",
        "download": "[bold yellow]download[/bold yellow]",
        "attachment": "[bold yellow]attachment[/bold yellow]",
        "update": "[bold green]update[/bold green]",
        "backup": "[bold green]backup[/bold green]",
    }
    
    # This is a simple implementation - for production, use regex to ensure whole word matching
    for term, highlight in highlight_terms.items():
        text = text.replace(f" {term} ", f" {highlight} ")
        text = text.replace(f" {term}.", f" {highlight}.")
        text = text.replace(f" {term},", f" {highlight},")
        text = text.replace(f" {term}:", f" {highlight}:")
        text = text.replace(f" {term}s ", f" {highlight}s ")
    
    return text
