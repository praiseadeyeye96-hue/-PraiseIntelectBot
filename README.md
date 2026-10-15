# PraiseBot — AI Writing, SEO, Marketing & Business Assistant

A self-hosted Telegram AI bot that runs entirely on local CPU inference.
**Telegram Username:** `@PraiseIntelectBot`

## Features

- **AI Writing** — Articles, emails, product descriptions, blog posts
- **SEO Assistant** — Titles, meta descriptions, content outlines
- **Marketing** — Ad copy, social captions, email campaigns
- **Business Content** — Proposals, brand descriptions, executive summaries
- **Conversation Memory** — Follow-up questions work naturally
- **Inline Menu** — Simple button-based navigation
- **No External AI APIs** — All inference runs locally

## Architecture

- **Bot Framework:** python-telegram-bot v21 (async polling)
- **AI Engine:** llama-cpp-python with Qwen2.5-3B-Instruct GGUF (Q4_K_M)
- **Deployment:** Railway with Docker, persistent volume at `/data`
- **Model Storage:** `/data/models/qwen2.5-3b-instruct-q4_k_m.gguf`

## Quick Start (Local Testing)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/praisebot.git
cd praisebot

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the environment template
cp .env.example .env
# Edit .env and add your BOT_TOKEN

# 5. Run the bot
python main.py
