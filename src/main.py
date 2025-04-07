#!/usr/bin/env python3
"""
CLI-cyber-elderly: A CLI chatbot for explaining cybersecurity to elderly users.
"""
import os
import sys
import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
import colorama
from dotenv import load_dotenv

# Try to import from src package first, then fall back to relative imports
try:
    from src.models import ModelManager
    from src.prompts import SYSTEM_PROMPT
    from src.utils import format_response
except ImportError:
    # If running directly from src directory
    from models import ModelManager
    from prompts import SYSTEM_PROMPT
    from utils import format_response

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Load environment variables from .env file
load_dotenv()

# Initialize Rich console for better terminal output
console = Console()
app = typer.Typer()

def display_welcome_message():
    """Display welcome message and instructions."""
    welcome_text = """
# Welcome to Cyber-Elderly Assistant!

This chatbot will help you understand cybersecurity concepts in simple terms.

## Commands:
- Type your questions about cybersecurity
- Type `/help` to see this message again
- Type `/model` to change the AI model
- Type `/exit` or `/quit` to exit

Let's start! What would you like to know about cybersecurity?
    """
    console.print(Panel(Markdown(welcome_text), title="Cyber-Elderly Assistant", border_style="blue"))

def display_help():
    """Display help information."""
    help_text = """
## Available Commands:
- `/help` - Show this help message
- `/model` - Change the AI model
- `/exit` or `/quit` - Exit the application

## Example Questions:
- "What is phishing?"
- "How do I create a strong password?"
- "What should I do if I receive a suspicious email?"
- "How can I protect my personal information online?"
    """
    console.print(Panel(Markdown(help_text), title="Help", border_style="green"))

def handle_model_change(model_manager):
    """Handle model change request."""
    console.print("\nAvailable models:")
    for idx, model in enumerate(model_manager.available_models, 1):
        console.print(f"{idx}. {model}")
    
    choice = Prompt.ask(
        "Select a model (number)",
        choices=[str(i) for i in range(1, len(model_manager.available_models) + 1)],
        default="1"
    )
    
    model_idx = int(choice) - 1
    model_name = model_manager.available_models[model_idx]
    
    # If OpenAI model is selected, check for API key
    if "openai" in model_name.lower() and not os.getenv("OPENAI_API_KEY"):
        console.print(Panel(
            "[bold red]OpenAI API key not found![/bold red]\n\n"
            "Please set your OpenAI API key using the OPENAI_API_KEY environment variable:\n"
            "export OPENAI_API_KEY=your_api_key_here\n\n"
            "Or add it to a .env file in the project directory.",
            title="API Key Required",
            border_style="red"
        ))
        return False
    
    model_manager.set_model(model_name)
    console.print(f"[green]Model changed to: {model_name}[/green]")
    return True

@app.command()
def main():
    """Run the CLI cybersecurity chatbot."""
    # Initialize the model manager
    model_manager = ModelManager()
    
    # Display welcome message
    display_welcome_message()
    
    # Main chat loop
    chat_history = []
    
    while True:
        # Get user input
        user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
        
        # Handle commands
        if user_input.lower() in ["/exit", "/quit"]:
            console.print("[yellow]Thank you for using Cyber-Elderly Assistant. Goodbye![/yellow]")
            break
        elif user_input.lower() == "/help":
            display_help()
            continue
        elif user_input.lower() == "/model":
            handle_model_change(model_manager)
            continue
        
        # Add user message to chat history
        chat_history.append({"role": "user", "content": user_input})
        
        try:
            # Get response from the model
            with console.status("[bold green]Thinking...[/bold green]"):
                response = model_manager.get_response(chat_history, SYSTEM_PROMPT)
            
            # Add assistant response to chat history
            chat_history.append({"role": "assistant", "content": response})
            
            # Display the response
            console.print("\n[bold green]Assistant[/bold green]:")
            # Format the response with highlighted keywords
            formatted_response = format_response(response)
            console.print(Panel(Markdown(formatted_response), border_style="green"))
            
        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
            if "API key" in str(e):
                console.print(
                    "[yellow]Please set your OpenAI API key using the OPENAI_API_KEY environment variable.[/yellow]"
                )

if __name__ == "__main__":
    app()
