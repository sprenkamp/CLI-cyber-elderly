"""
Model management for the CLI cybersecurity chatbot.
Handles switching between different LLM providers.
"""
import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import openai
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class ModelManager:
    """Manages different language models for the chatbot."""
    
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
        """Initialize the selected language model."""
        # We don't actually initialize the models here, just set the name
        # The actual model calls happen in get_response
        pass
    
    def set_model(self, model_name: str):
        """Set the current model by name."""
        if model_name not in self.available_models:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.current_model_name = model_name
    
    def get_response(self, chat_history: List[Dict[str, str]], system_prompt: str) -> str:
        """
        Get a response from the current model based on chat history.
        
        Args:
            chat_history: List of message dictionaries with 'role' and 'content'
            system_prompt: The system prompt to use
            
        Returns:
            The model's response as a string
        """
        if "openai" in self.current_model_name.lower() or "gpt" in self.current_model_name.lower():
            # For OpenAI models
            if not openai.api_key:
                raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
            
            # Format messages for OpenAI API
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add chat history
            for message in chat_history:
                messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })
            
            # Call OpenAI API
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        elif "llama" in self.current_model_name.lower() or "meta" in self.current_model_name.lower():
            # For Llama model
            # Define model paths and info
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            model_filename = "llama-3.2-1b.gguf"
            model_path = os.path.join(models_dir, model_filename)
            
            # Create models directory if it doesn't exist
            os.makedirs(models_dir, exist_ok=True)
            
            # Check if model file exists
            if not os.path.exists(model_path):
                console.print("[yellow]Model not found locally. Attempting to download from Hugging Face...[/yellow]")
                try:
                    # Define Hugging Face model info
                    repo_id = "meta-llama/Meta-Llama-3.2-1B-GGUF"
                    filename = "Meta-Llama-3.2-1B.Q4_K_M.gguf"
                    
                    # Download model with progress indicator
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[bold green]Downloading model... {task.description}"),
                        console=console
                    ) as progress:
                        task = progress.add_task("[cyan]Please wait", total=None)
                        # Download the model from Hugging Face
                        downloaded_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            token=os.getenv("HF_TOKEN")  # Use token if provided in .env
                        )
                        progress.update(task, description="[green]Download complete!")
                    
                    # Copy the downloaded file to our models directory
                    shutil.copy(downloaded_path, model_path)
                    console.print(f"[green]Model downloaded and saved to {model_path}[/green]")
                    
                except Exception as e:
                    raise FileNotFoundError(
                        f"Failed to download model from Hugging Face: {str(e)}\n"
                        "Please check your internet connection or provide a HF_TOKEN in your .env file."
                    )
            
            # Initialize Llama model
            llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=4
            )
            
            # Format prompt for Llama
            prompt = f"{system_prompt}\n\n"
            
            for message in chat_history:
                if message["role"] == "user":
                    prompt += f"Human: {message['content']}\n"
                elif message["role"] == "assistant":
                    prompt += f"Assistant: {message['content']}\n"
            
            prompt += "Assistant: "
            
            # Generate response
            response = llm(
                prompt,
                max_tokens=1000,
                temperature=0.7,
                stop=["Human:", "\n\n"]
            )
            
            return response["choices"][0]["text"]
