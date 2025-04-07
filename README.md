# CLI-cyber-elderly

A command-line interface (CLI) chatbot designed to explain cybersecurity concepts to elderly users. The application supports both closed-source (OpenAI) and open-source language models (Meta Llama).

## Features

- Simple CLI interface optimized for elderly users
- Explanations of common cybersecurity threats and best practices
- Ability to switch between different language models:
  - OpenAI GPT-4o-mini (requires API key)
  - Meta Llama 3.2 1B (local open-source model)
- Conversation history tracking

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/CLI-cyber-elderly.git
cd CLI-cyber-elderly
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Set up your environment variables by creating a `.env` file with your API keys:

```
# Create a .env file in the project root directory
touch .env

# Add the following content to the .env file, replacing the placeholder values with your actual keys:

# OpenAI API Key - Required for using the GPT-4o-mini model
# Get this key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Hugging Face Token - Required only if downloading gated/private Llama models
# Get this token from: https://huggingface.co/settings/tokens
HF_TOKEN=your_huggingface_token_here
```

**Why these keys are needed:**

- **OPENAI_API_KEY**: This key is required to access OpenAI's GPT-4o-mini model. Without this key, you won't be able to use the OpenAI model option in the chatbot.

- **HF_TOKEN**: This token is only required if you want to download Llama models that are gated or require authentication on Hugging Face. Some models are freely available without authentication, but others may require you to accept terms of use or have special access permissions.

## Usage

Run the chatbot:
```
python src/main.py
```

Follow the on-screen instructions to interact with the chatbot and learn about cybersecurity.

## Switching Between Models

The chatbot allows you to switch between different language models:

1. OpenAI GPT-4o-mini (requires API key)
2. Meta Llama 3.2 1B (runs locally)

To change the model, use the `/model` command during the chat session.

### Automatic Model Download

When selecting the Meta Llama 3.2 1B model for the first time, the application will automatically:

1. Check if the model exists in the `models/` directory
2. If not found, download it directly from Hugging Face
3. Save it to the `models/` directory for future use

Note: Some models on Hugging Face may require authentication. If needed, add your Hugging Face token to the `.env` file:
```
HF_TOKEN=your_huggingface_token_here
```

## License

MIT
