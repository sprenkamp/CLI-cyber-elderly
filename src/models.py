"""
Model management for the CLI cybersecurity chatbot.
Handles switching between different LLM providers using LangChain.
"""
import os
import json
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import for OpenAI
import openai

# Import for Llama model
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Import LangChain components that we know work
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

console = Console()

class ModelManager:
    """Manages different language models for the chatbot using LangChain."""
    
    def __init__(self):
        """Initialize the model manager with available models."""
        self.available_models = [
            "OpenAI GPT-4o-mini",
            "Meta-Llama-3.2-1B (local)"
        ]
        self.current_model_name = self.available_models[0]
        self.current_model = None
        
        # Set OpenAI API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            openai.api_key = openai_api_key
            
        # Initialize the default model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the selected language model using LangChain."""
        if "openai" in self.current_model_name.lower() or "gpt" in self.current_model_name.lower():
            # Initialize OpenAI model with LangChain
            self.current_model = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.7,
                max_tokens=1000
            )
        else:
            # We'll initialize the Llama model when it's actually selected
            self.current_model = None
    
    def set_model(self, model_name: str):
        """Set the current model by name using LangChain."""
        if model_name not in self.available_models:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.current_model_name = model_name
        
        # Initialize the appropriate model with LangChain
        if "openai" in self.current_model_name.lower() or "gpt" in self.current_model_name.lower():
            # Initialize OpenAI model
            self.current_model = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.7,
                max_tokens=1000
            )
        elif "llama" in self.current_model_name.lower() or "meta" in self.current_model_name.lower():
            console.print("[yellow]Preparing to use Meta Llama 3.2 1B model...[/yellow]")
            try:
                # We'll use the transformers library with LangChain
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold green]Loading model... {task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("[cyan]Please wait", total=None)
                    
                    # Download and prepare the model
                    progress.update(task, description="[cyan]Loading tokenizer...")
                    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=os.getenv("HF_TOKEN"))
                    
                    progress.update(task, description="[cyan]Loading model...")
                    model = AutoModelForCausalLM.from_pretrained(
                        "meta-llama/Llama-3.2-1B", 
                        token=os.getenv("HF_TOKEN")
                    )
                    
                    progress.update(task, description="[cyan]Setting up pipeline...")
                    # Create a text generation pipeline
                    text_generation_pipeline = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=1000,
                        temperature=0.7,
                        do_sample=True
                    )
                    
                    # Create LangChain wrapper for the pipeline
                    self.current_model = HuggingFacePipeline(pipeline=text_generation_pipeline)
                    
                    progress.update(task, description="[green]Model ready!")
                
                console.print("[green]Meta Llama 3.2 1B model is ready to use![/green]")
                
            except Exception as e:
                console.print(f"[bold red]Failed to load Meta Llama 3.2 1B model: {str(e)}[/bold red]")
                console.print("[yellow]Reverting to OpenAI model. Please check your internet connection or provide a HF_TOKEN in your .env file.[/yellow]")
                self.current_model_name = self.available_models[0]  # Revert to OpenAI model
                # Initialize OpenAI model as fallback
                self.current_model = ChatOpenAI(
                    model_name="gpt-4o-mini",
                    temperature=0.7,
                    max_tokens=1000
                )
    
    def get_response(self, chat_history: List[Dict[str, str]], system_prompt: str) -> str:
        """
        Get a response from the current model based on chat history using LangChain.
        
        Args:
            chat_history: List of message dictionaries with 'role' and 'content'
            system_prompt: The system prompt to use
            
        Returns:
            The model's response as a string
        """
        try:
            if "openai" in self.current_model_name.lower() or "gpt" in self.current_model_name.lower():
                # For OpenAI models
                if not openai.api_key:
                    raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
                
                # Format messages for LangChain using proper message types
                messages = [
                    SystemMessage(content=system_prompt)
                ]
                
                # Add chat history
                for message in chat_history:
                    if message["role"] == "user":
                        messages.append(HumanMessage(content=message["content"]))
                    elif message["role"] == "assistant":
                        messages.append(AIMessage(content=message["content"]))
                
                # Call OpenAI through LangChain
                response = self.current_model.invoke(messages)
                return response.content
                
            elif "llama" in self.current_model_name.lower() or "meta" in self.current_model_name.lower():
                console.print("[cyan]Generating response with Meta Llama 3.2 1B...[/cyan]")
                
                # For Llama model, we need to use a different approach
                # First, create a formatted prompt for the Llama model
                formatted_prompt = f"{system_prompt}\n\n"
                
                for message in chat_history:
                    if message["role"] == "user":
                        formatted_prompt += f"Human: {message['content']}\n"
                    elif message["role"] == "assistant":
                        formatted_prompt += f"Assistant: {message['content']}\n"
                
                formatted_prompt += "Assistant: "
                
                # Generate response using the LangChain HuggingFacePipeline
                generated_text = self.current_model.invoke(formatted_prompt)
                
                # Extract just the assistant's response (after the last "Assistant: ")
                assistant_response = generated_text.split("Assistant: ")[-1].strip()
                
                # Remove any text after "Human:" if present (the model might hallucinate a follow-up question)
                if "Human:" in assistant_response:
                    assistant_response = assistant_response.split("Human:")[0].strip()
                
                return assistant_response
                
        except Exception as e:
            console.print(f"[bold red]Error using model {self.current_model_name}: {str(e)}[/bold red]")
            
            # If we're already using OpenAI, just re-raise the exception
            if "openai" in self.current_model_name.lower() or "gpt" in self.current_model_name.lower():
                raise
                
            # Fall back to OpenAI
            console.print("[yellow]Falling back to OpenAI model...[/yellow]")
            self.current_model_name = self.available_models[0]
            self.current_model = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.7,
                max_tokens=1000
            )
            return self.get_response(chat_history, system_prompt)
