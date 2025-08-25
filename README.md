# Obsidian Auto Directory Name

This Python program automatically generates directory structures for Obsidian notes using LLM.

## Features

- Generate directories based on locale and keyword seeds
- Support for PARA and Zettelkasten organizational methods
- Preview mode (dry-run) available
- Preserves existing directories and integrates new ones
- Generates a tree preview of changes
- Saves directory structure to a JSON file

## Installation

```bash
pip install -r requirements.txt
```

## LLM Configuration

To use the LLM features, you need to configure your LLM provider and API key in the `.env` file.

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file to set your preferred LLM provider and API key:
   ```bash
   # Choose your LLM provider (openai, deepseek, or custom)
   LLM_PROVIDER=openai
   
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_BASE_URL=https://api.openai.com/v1
   LLM_MODEL=gpt-3.5-turbo-instruct
   
   # DeepSeek Configuration
   # LLM_PROVIDER=deepseek
   # DEEPSEEK_API_KEY=your_deepseek_api_key_here
   # DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
   # LLM_MODEL=deepseek-coder
   
   # Custom LLM Configuration
   # LLM_PROVIDER=custom
   # CUSTOM_LLM_API_KEY=your_custom_api_key_here
   # CUSTOM_LLM_BASE_URL=your_custom_endpoint_url_here
   # LLM_MODEL=your_model_name
   
   # LLM Parameters
   LLM_MAX_TOKENS=500
   LLM_TEMPERATURE=0.7
   ```

3. For OpenAI, get your API key from [OpenAI](https://platform.openai.com/api-keys)
4. For DeepSeek, get your API key from [DeepSeek](https://platform.deepseek.com/)

Note: If no API key is provided, the tool will use placeholder responses.

## Usage

```bash
python obsidian_auto_dirname.py [options]
```

### Options

- `--locale`: Set the locale (default: en)
- `--keywords`: Comma-separated list of keywords to use as seeds
- `--method`: Organizational method (PARA or Zettelkasten)
- `--dry-run`: Preview mode without generating directories

## Directory Structure

- Maximum 3 levels, minimum 2 levels
- Naming convention: `01. Directory Name`
- Generates an "Uncategorized" directory for difficult classifications

## Example Usage

```bash
# Generate directories with default settings
python obsidian_auto_dirname.py

# Generate directories based on keywords
python obsidian_auto_dirname.py --keywords="programming,design,marketing"

# Use PARA method with Chinese locale
python obsidian_auto_dirname.py --locale=zh --method=PARA --keywords="项目,文档,笔记"

# Preview changes without creating directories
python obsidian_auto_dirname.py --dry-run --keywords="programming,design"
```