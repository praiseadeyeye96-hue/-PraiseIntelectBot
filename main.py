#!/usr/bin/env python3
"""
PraiseBot — AI Writing, SEO, Marketing & Business Content Assistant
Telegram Username: @PraiseIntelectBot
Runs entirely on local CPU inference via llama-cpp-python.
"""

import asyncio
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
)
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BOT_TOKEN: str = os.environ.get("BOT_TOKEN", "")
MODEL_PATH: str = os.environ.get("MODEL_PATH", "/data/models/qwen2.5-3b-instruct-q4_k_m.gguf")
MODEL_URL: str = os.environ.get(
    "MODEL_URL",
    "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
)
N_CTX: int = int(os.environ.get("N_CTX", "2048"))
N_THREADS: int = int(os.environ.get("N_THREADS", "4"))
MAX_TOKENS: int = int(os.environ.get("MAX_TOKENS", "512"))
MAX_INPUT_CHARS: int = int(os.environ.get("MAX_INPUT_CHARS", "2000"))
USER_COOLDOWN_SECONDS: int = int(os.environ.get("USER_COOLDOWN_SECONDS", "10"))
MAX_HISTORY_TURNS: int = int(os.environ.get("MAX_HISTORY_TURNS", "5"))
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
TEMPERATURE: float = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P: float = float(os.environ.get("TOP_P", "0.9"))
DOWNLOAD_TIMEOUT: int = int(os.environ.get("DOWNLOAD_TIMEOUT", "600"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger("praisebot")

# ---------------------------------------------------------------------------
# Model Engine — Thread-Safe Singleton
# ---------------------------------------------------------------------------

class LocalAIEngine:
    """Manages the local GGUF model lifecycle and inference."""

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._initialized = False
        self._loading_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._model is not None

    def _ensure_model_file(self) -> bool:
        """Download model if missing. Returns True if file exists."""
        model_path = Path(MODEL_PATH)

        if model_path.exists() and model_path.stat().st_size > 1_000_000:
            logger.info("Model file found: %s (%.2f GB)",
                        model_path, model_path.stat().st_size / (1024 ** 3))
            return True

        if not MODEL_URL:
            logger.error("Model file missing and no MODEL_URL configured.")
            return False

        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading model from %s ...", MODEL_URL)

        try:
            import urllib.request

            tmp_path = model_path.with_suffix(".tmp")
            with urllib.request.urlopen(MODEL_URL, timeout=DOWNLOAD_TIMEOUT) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1 MB
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0 and downloaded % (100 * chunk_size) == 0:
                            logger.info(
                                "Downloaded %.1f / %.1f MB",
                                downloaded / (1024 ** 2),
                                total / (1024 ** 2),
                            )
            tmp_path.rename(model_path)
            logger.info("Model download complete: %s", model_path)
            return True
        except Exception as exc:
            logger.error("Model download failed: %s", exc, exc_info=True)
            tmp_path = model_path.with_suffix(".tmp")
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            return False

    def initialize(self) -> bool:
        """Load the model into memory. Safe to call multiple times."""
        if self._initialized:
            return True

        with self._loading_lock:
            if self._initialized:
                return True

            if not self._ensure_model_file():
                return False

            try:
                from llama_cpp import Llama

                logger.info(
                    "Loading model (n_ctx=%d, n_threads=%d)...",
                    N_CTX, N_THREADS,
                )
                self._model = Llama(
                    model_path=MODEL_PATH,
                    n_ctx=N_CTX,
                    n_threads=N_THREADS,
                    n_batch=256,
                    verbose=False,
                )
                self._initialized = True
                logger.info("Model loaded successfully.")
                return True
            except Exception as exc:
                logger.error("Model loading failed: %s", exc, exc_info=True)
                self._model = None
                self._initialized = False
                return False

    def generate(
        self,
        system_prompt: str,
        conversation: list[dict],
        user_message: str,
    ) -> str:
        """Run inference under a lock. Returns the generated text."""
        if not self.is_ready:
            raise RuntimeError("Model is not initialized.")

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation)
        messages.append({"role": "user", "content": user_message})

        with self._lock:
            output = self._model.create_chat_completion(
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repeat_penalty=1.1,
            )

        if not output or "choices" not in output or not output["choices"]:
            return ""

        content = output["choices"][0].get("message", {}).get("content", "")
        return content.strip() if content else ""


# Global engine instance
engine = LocalAIEngine()

# ---------------------------------------------------------------------------
# System Prompts per Mode
# ---------------------------------------------------------------------------

MODE_PROMPTS: dict[str, str] = {
    "writing": (
        "You are PraiseBot, a professional AI writing assistant. "
        "Generate well-structured, engaging, and grammatically correct content. "
        "Adapt your tone to the user's request (professional, casual, persuasive, etc.). "
        "If asked to revise, improve the previous output rather than starting from scratch."
    ),
    "seo": (
        "You are PraiseBot, an SEO content specialist. "
        "Generate SEO-optimized titles, meta descriptions, content outlines, and articles. "
        "Use relevant keywords naturally. Never promise specific rankings or traffic. "
        "Always note that SEO results depend on many factors beyond content quality."
    ),
    "marketing": (
        "You are PraiseBot, a marketing copywriting assistant. "
        "Create compelling ad copy, social media captions, email campaigns, and product descriptions. "
        "Focus on benefits, emotional resonance, and clear calls-to-action. "
        "Never promise guaranteed sales or engagement."
    ),
    "business": (
        "You are PraiseBot, a business content assistant. "
        "Generate professional business proposals, brand descriptions, executive summaries, "
        "and corporate communications. Maintain a polished, formal tone suitable for stakeholders."
    ),
}

DEFAULT_MODE = "writing"

# ---------------------------------------------------------------------------
# Conversation Memory (in-memory, per-user)
# ---------------------------------------------------------------------------

class ConversationMemory:
    """Stores short-term conversation history per user."""

    def __init__(self, max_turns: int = 5) -> None:
        self._store: OrderedDict[int, list[dict]] = OrderedDict()
        self._max_turns = max_turns

    def get(self, user_id: int) -> list[dict]:
        return list(self._store.get(user_id, []))

    def add_turn(self, user_id: int, user_msg: str, assistant_msg: str) -> None:
        history = self._store.setdefault(user_id, [])
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})
        # Trim to max_turns pairs
        max_msgs = self._max_turns * 2
        if len(history) > max_msgs:
            self._store[user_id] = history[-max_msgs:]

    def clear(self, user_id: int) -> None:
        self._store.pop(user_id, None)

    def cleanup(self) -> None:
        """Remove oldest entries if store grows too large."""
        max_users = 500
        while len(self._store) > max_users:
            self._store.popitem(last=False)


memory = ConversationMemory(max_turns=MAX_HISTORY_TURNS)

# ---------------------------------------------------------------------------
# Cooldown Tracker
# ---------------------------------------------------------------------------

class CooldownTracker:
    def __init__(self, seconds: int) -> None:
        self._seconds = seconds
        self._last: dict[int, float] = {}

    def check(self, user_id: int) -> Optional[float]:
        """Return remaining seconds if on cooldown, else None."""
        now = time.monotonic()
        last = self._last.get(user_id, 0.0)
        elapsed = now - last
        if elapsed < self._seconds:
            return round(self._seconds - elapsed, 1)
        return None

    def stamp(self, user_id: int) -> None:
        self._last[user_id] = time.monotonic()


cooldown = CooldownTracker(USER_COOLDOWN_SECONDS)

# ---------------------------------------------------------------------------
# User Mode Storage
# ---------------------------------------------------------------------------

user_modes: dict[int, str] = {}

def get_user_mode(user_id: int) -> str:
    return user_modes.get(user_id, DEFAULT_MODE)

# ---------------------------------------------------------------------------
# Menu Builders
# ---------------------------------------------------------------------------

def build_main_menu() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton("✍️ AI Writing", callback_data="mode_writing")],
        [InlineKeyboardButton("🔍 SEO Assistant", callback_data="mode_seo")],
        [InlineKeyboardButton("📣 Marketing Assistant", callback_data="mode_marketing")],
        [InlineKeyboardButton("💼 Business Content", callback_data="mode_business")],
        [InlineKeyboardButton("❓ Help", callback_data="help")],
    ]
    return InlineKeyboardMarkup(keyboard)

def build_back_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("◀️ Back to Menu", callback_data="menu")],
    ])

# ---------------------------------------------------------------------------
# Message Splitting
# ---------------------------------------------------------------------------

def split_message(text: str, max_len: int = 4000) -> list[str]:
    """Split long text into Telegram-compatible chunks."""
    if len(text) <= max_len:
        return [text]
    parts = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break
        cut = text.rfind("\n", 0, max_len)
        if cut == -1:
            cut = text.rfind(" ", 0, max_len)
        if cut == -1:
            cut = max_len
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    return parts

# ---------------------------------------------------------------------------
# Command Handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — welcome message and main menu."""
    user = update.effective_user
    name = user.first_name if user else "there"

    text = (
        f"👋 Hello, <b>{name}</b>!\n\n"
        "I'm <b>PraiseBot</b> — your AI assistant for writing, SEO, "
        "marketing, and business content.\n\n"
        "📝 <b>What I can do:</b>\n"
        "• Write articles, emails, and product descriptions\n"
        "• Generate SEO titles, meta descriptions, and outlines\n"
        "• Create marketing copy and social media captions\n"
        "• Draft business proposals and brand descriptions\n\n"
        "Choose a mode below, or simply <b>type your request</b> at any time."
    )
    await update.message.reply_text(
        text,
        parse_mode=ParseMode.HTML,
        reply_markup=build_main_menu(),
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help."""
    text = (
        "📖 <b>How to Use PraiseBot</b>\n\n"
        "<b>Commands:</b>\n"
        "/start — Main menu\n"
        "/write — AI Writing mode\n"
        "/seo — SEO Assistant mode\n"
        "/marketing — Marketing mode\n"
        "/business — Business Content mode\n"
        "/reset — Clear your conversation history\n"
        "/about — About this bot\n"
        "/help — This message\n\n"
        "<b>Example prompts:</b>\n"
        "• Write a product description for a leather handbag\n"
        "• Create 5 SEO titles for a blog post about coffee brewing\n"
        "• Write a Facebook ad for a fitness app\n"
        "• Draft a business proposal for a catering company\n\n"
        "💡 <i>Tip: After receiving a response, you can ask follow-ups "
        "like \"Make it more formal\" or \"Shorten it\" — I remember the "
        "recent conversation.</i>"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

async def cmd_about(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /about."""
    text = (
        "ℹ️ <b>About PraiseBot</b>\n\n"
        "PraiseBot is a self-hosted AI assistant that runs a local language "
        "model directly on the server. It generates writing, SEO content, "
        "marketing copy, and business drafts.\n\n"
        "🔒 <b>Privacy:</b> Your messages are processed locally. "
        "Conversation history is kept in memory only and is lost when the "
        "service restarts. We do not store personal information or share "
        "your data with third parties.\n\n"
        "⚠️ <b>Limitations:</b> The AI model has no live internet access and "
        "cannot verify current SEO metrics or real-time information. "
        "Generated content should be reviewed before publishing."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

async def cmd_write(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /write — AI Writing mode."""
    user_id = update.effective_user.id
    user_modes[user_id] = "writing"
    await update.message.reply_text(
        "✍️ <b>AI Writing mode activated.</b>\n\n"
        "Send me a prompt — an article idea, email draft, product description, "
        "or anything you'd like me to write.",
        parse_mode=ParseMode.HTML,
    )

async def cmd_seo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /seo."""
    user_id = update.effective_user.id
    user_modes[user_id] = "seo"
    await update.message.reply_text(
        "🔍 <b>SEO Assistant mode activated.</b>\n\n"
        "Ask me for SEO titles, meta descriptions, content outlines, or "
        "optimized articles. Include your target keyword if you have one.",
        parse_mode=ParseMode.HTML,
    )

async def cmd_marketing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /marketing."""
    user_id = update.effective_user.id
    user_modes[user_id] = "marketing"
    await update.message.reply_text(
        "📣 <b>Marketing mode activated.</b>\n\n"
        "Send me your product, campaign idea, or target audience, and I'll "
        "draft ad copy, social captions, or email campaigns.",
        parse_mode=ParseMode.HTML,
    )

async def cmd_business(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /business."""
    user_id = update.effective_user.id
    user_modes[user_id] = "business"
    await update.message.reply_text(
        "💼 <b>Business Content mode activated.</b>\n\n"
        "Ask me for business proposals, brand descriptions, executive "
        "summaries, or corporate communications.",
        parse_mode=ParseMode.HTML,
    )

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /reset."""
    user_id = update.effective_user.id
    memory.clear(user_id)
    user_modes.pop(user_id, None)
    await update.message.reply_text(
        "🔄 <b>Conversation history cleared.</b>\n\n"
        "Your mode has been reset to the default (AI Writing). "
        "You can start a fresh conversation.",
        parse_mode=ParseMode.HTML,
    )

# ---------------------------------------------------------------------------
# Callback Query Handler (Inline Buttons)
# ---------------------------------------------------------------------------

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button presses."""
    query = update.callback_query
    await query.answer()

    data = query.data
    user_id = query.from_user.id

    if data == "menu":
        await query.edit_message_text(
            "📋 <b>Main Menu</b>\n\nChoose a mode or type your request:",
            parse_mode=ParseMode.HTML,
            reply_markup=build_main_menu(),
        )
    elif data == "help":
        await query.edit_message_text(
            "📖 <b>Help</b>\n\n"
            "Use the buttons below or type commands directly:\n"
            "/write /seo /marketing /business /reset /about\n\n"
            "You can also just type your request at any time.",
            parse_mode=ParseMode.HTML,
            reply_markup=build_back_menu(),
        )
    elif data.startswith("mode_"):
        mode = data.replace("mode_", "")
        user_modes[user_id] = mode
        labels = {
            "writing": "✍️ AI Writing",
            "seo": "🔍 SEO Assistant",
            "marketing": "📣 Marketing Assistant",
            "business": "💼 Business Content",
        }
        label = labels.get(mode, mode.title())
        await query.edit_message_text(
            f"✅ <b>{label} mode activated.</b>\n\n"
            "Send me your request and I'll generate a response.",
            parse_mode=ParseMode.HTML,
            reply_markup=build_back_menu(),
        )

# ---------------------------------------------------------------------------
# Main Text Message Handler
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process ordinary text messages through the AI engine."""
    user = update.effective_user
    if not user or not update.message or not update.message.text:
        return

    user_id = user.id
    text = update.message.text.strip()

    # Input length guard
    if len(text) > MAX_INPUT_CHARS:
        await update.message.reply_text(
            f"⚠️ Your message is too long ({len(text)} characters). "
            f"Please shorten it to under {MAX_INPUT_CHARS} characters."
        )
        return

    # Cooldown check
    remaining = cooldown.check(user_id)
    if remaining is not None:
        await update.message.reply_text(
            f"⏳ Please wait <b>{remaining}s</b> before sending another request.",
            parse_mode=ParseMode.HTML,
        )
        return

    # Model readiness
    if not engine.is_ready:
        await update.message.reply_text(
            "⏳ The AI model is still loading. Please try again in a moment."
        )
        return

    mode = get_user_mode(user_id)
    system_prompt = MODE_PROMPTS.get(mode, MODE_PROMPTS[DEFAULT_MODE])

    # Send "typing" action
    await update.message.chat.send_action(ChatAction.TYPING)
    status_msg = await update.message.reply_text(
        "⏳ Generating your response..."
    )

    try:
        loop = asyncio.get_running_loop()
        conversation = memory.get(user_id)

        response = await loop.run_in_executor(
            None,
            engine.generate,
            system_prompt,
            conversation,
            text,
        )

        if not response:
            response = (
                "⚠️ The AI model returned an empty response. "
                "Please try rephrasing your request."
            )

        memory.add_turn(user_id, text, response)
        memory.cleanup()
        cooldown.stamp(user_id)

        # Delete status message
        try:
            await status_msg.delete()
        except Exception:
            pass

        # Send response in chunks
        chunks = split_message(response)
        for i, chunk in enumerate(chunks):
            if i == 0:
                await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
            else:
                await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)

    except Exception as exc:
        logger.error("Inference error for user %s: %s", user_id, exc, exc_info=True)
        try:
            await status_msg.edit_text(
                "❌ An error occurred while generating the response. "
                "Please try again."
            )
        except Exception:
            await update.message.reply_text(
                "❌ An error occurred while generating the response. "
                "Please try again."
            )

# ---------------------------------------------------------------------------
# Error Handler
# ---------------------------------------------------------------------------

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors without exposing the bot token."""
    logger.error("Update %s caused error: %s", update, context.error, exc_info=True)

# ---------------------------------------------------------------------------
# Post-Init: Set Bot Commands & Load Model
# ---------------------------------------------------------------------------

async def post_init(application: Application) -> None:
    """Set bot commands and pre-load the model."""
    commands = [
        BotCommand("start", "Main menu"),
        BotCommand("help", "Instructions & examples"),
        BotCommand("write", "AI Writing mode"),
        BotCommand("seo", "SEO Assistant mode"),
        BotCommand("marketing", "Marketing Assistant mode"),
        BotCommand("business", "Business Content mode"),
        BotCommand("reset", "Clear conversation history"),
        BotCommand("about", "About this bot"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Bot commands registered.")

    # Pre-load model in background thread
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, engine.initialize)

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    if not BOT_TOKEN:
        logger.critical("BOT_TOKEN environment variable is not set. Exiting.")
        raise SystemExit(1)

    logger.info("Starting PraiseBot...")
    logger.info("Model path: %s", MODEL_PATH)
    logger.info("Context: %d | Threads: %d | Max tokens: %d",
                N_CTX, N_THREADS, MAX_TOKENS)

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Command handlers
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("about", cmd_about))
    application.add_handler(CommandHandler("write", cmd_write))
    application.add_handler(CommandHandler("seo", cmd_seo))
    application.add_handler(CommandHandler("marketing", cmd_marketing))
    application.add_handler(CommandHandler("business", cmd_business))
    application.add_handler(CommandHandler("reset", cmd_reset))

    # Callback queries (inline buttons)
    application.add_handler(CallbackQueryHandler(handle_callback))

    # Text messages (anything not a command)
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    # Error handler
    application.add_error_handler(error_handler)

    logger.info("Polling for updates...")
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
