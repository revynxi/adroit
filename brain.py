import asyncio
import json
import logging
import logging.handlers
import os
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, quote_plus

import aiosqlite
import discord
import fasttext
from aiohttp import ClientSession, client_exceptions, web
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                    wait_random_exponential)
from thefuzz import fuzz

load_dotenv()

log_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger('discord_bot_v3')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

try:
    file_handler = logging.handlers.RotatingFileHandler(
        filename='adroit_bot.log',
        encoding='utf-8',
        maxBytes=5 * 1024 * 1024,  
        backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except Exception as e:
    logger.error(f"Failed to set up file logging: {e}", exc_info=True)


# --- Environment Variable Loading & Validation ---
DISCORD_TOKEN = os.getenv("ADROIT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FASTTEXT_MODEL_PATH = os.getenv("FASTTEXT_MODEL_PATH", "lid.176.ftz")
SIGHTENGINE_API_USER = os.getenv("SIGHTENGINE_API_USER")
SIGHTENGINE_API_SECRET = os.getenv("SIGHTENGINE_API_SECRET")

if not DISCORD_TOKEN:
    logger.critical("CRITICAL: ADROIT_TOKEN environment variable not set. Exiting.")
    exit(1)
if not OPENAI_API_KEY:
    logger.warning("Warning: OPENAI_API_KEY not set. AI content moderation will be disabled.")
if not (SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET):
    logger.warning("Warning: Sightengine API keys not set. Image moderation will be disabled.")


# --- Bot and Globals Initialization ---
intents = discord.Intents.default()
intents.members = True
intents.message_content = True
intents.presences = False

bot = commands.Bot(command_prefix=">>", intents=intents, help_command=None)

db_conn: aiosqlite.Connection | None = None
LANGUAGE_MODEL: fasttext.FastText._FastText | None = None
http_session: ClientSession | None = None

dynamic_rules = {
    "forbidden_words": set(),
    "forbidden_phrases": [],
    "forbidden_regex": []
}

# --- Static Configuration ---
class BotConfig:
    """Holds all static configurations for the bot for easy tuning."""
    def __init__(self):
        self.default_log_channel_id = 1113377818424922132
        self.default_review_channel_id = 1113377818424922132

        self.default_channel_configs = {
            1113377809440722974: {"language": ["en"]},
            1113377810476716132: {"language": ["en"]},
            1321499824926888049: {"language": ["fr"]},
            1122525009102000269: {"language": ["de"]},
            1122523546355245126: {"language": ["ru"]},
            1122524817904635904: {"language": ["zh"]},
            1242768362237595749: {"language": ["es"]}
        }
        self.forbidden_text_pattern = re.compile(
            r"(discord\.gg/|join\s+our|server\s+invite|free\s+nitro|check\s+out\s+my|follow\s+me|subscribe\s+to|buy\s+now|gift\s+card|giveaway\s+scam)",
            re.IGNORECASE
        )
        self.url_pattern = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|net|org|io|dev|xyz|gg|app|co|online|shop|site|fun|club|store|live)\b)")
        self.has_alphanumeric_pattern = re.compile(r'[a-zA-Z0-9]')

        self.permitted_domains = [
            "googleusercontent.com", "tenor.com", "giphy.com", "tiktok.com",
            "docs.google.com", "cdn.discordapp.com", "media.discordapp.net",
            "images-ext-1.discordapp.net", "images-ext-2.discordapp.net",
            "roblox.com", "github.com", "theuselessweb.com", "imgur.com", "i.imgur.com",
            "wikipedia.org", "wikimedia.org", "twitch.tv", "reddit.com", "x.com", "twitter.com",
            "fxtwitter.com", "vxtwitter.com", "spotify.com", "soundcloud.com",
            "pastebin.com", "hastebin.com", "gist.github.com", "youtube.com", "youtu.be"
        ]

        self.punishment_system = {
            "points_thresholds": {
                5: {"action": "warn", "reason_suffix": "Minor guideline violations.", "dm_message": "This is a formal warning. Please review the server rules carefully. Further violations will lead to stricter actions."},
                10: {"action": "mute", "duration_hours": 1, "reason_suffix": "Accumulated violations or spam.", "dm_message": "You have been muted for 1 hour due to repeated minor violations or spam. Please adhere to server guidelines upon your return."},
                20: {"action": "mute", "duration_hours": 6, "reason_suffix": "Significant or repeated violations.", "dm_message": "You have been muted for 6 hours due to significant or repeated violations. Continued disregard for rules will result in a kick or ban."},
                35: {"action": "kick", "reason_suffix": "Persistent serious violations after warnings/mutes.", "dm_message": "You have been kicked from the server due to persistent serious violations. You may rejoin, but further infractions will likely result in a ban."},
                50: {"action": "temp_ban", "duration_days": 3, "reason_suffix": "Severe or multiple major violations.", "dm_message": "You have been temporarily banned for 3 days due to severe or multiple major violations. Consider this a serious warning."},
                75: {"action": "temp_ban", "duration_days": 30, "reason_suffix": "Extreme or highly disruptive behavior.", "dm_message": "You have been temporarily banned for 30 days due to extreme or highly disruptive behavior. Any further issues after this ban may lead to a permanent ban."},
                100: {"action": "ban", "reason_suffix": "Egregious violations, repeat offenses after temp ban, or admin discretion.", "dm_message": "You have been permanently banned from the server due to egregious violations or continued disregard for server rules after previous sanctions."}
            },
            "violations": {
                "discrimination": {"points": 1, "severity": "Low"},
                "spam_rate": {"points": 1, "severity": "Low"},
                "spam_repetition": {"points": 1, "severity": "Low"},
                "nsfw_text": {"points": 1, "severity": "Low"},
                "nsfw_media": {"points": 5, "severity": "High"},
                "advertising_forbidden_text": {"points": 2, "severity": "Low"},
                "advertising_unpermitted_url": {"points": 2, "severity": "Low"},
                "politics_discussion_disallowed": {"points": 1, "severity": "Low"},
                "off_topic": {"points": 1, "severity": "Low"},
                "foreign_language": {"points": 1, "severity": "Low"},
                "openai_flagged_severe": {"points": 5, "severity": "High"},
                "openai_flagged_moderate": {"points": 5, "severity": "High"},
                "excessive_mentions": {"points": 1, "severity": "Low"},
                "excessive_attachments": {"points": 1, "severity": "Low"},
                "long_message": {"points": 1, "severity": "Low"},
                "gore_violence_media": {"points": 5, "severity": "High"},
                "offensive_symbols_media": {"points": 3, "severity": "Medium"},
                "dynamic_rule_violation": {"points": 3, "severity": "Medium"},
            }
        }
        self.spam_window_seconds = 10
        self.spam_message_limit = 5
        self.spam_repetition_history_count = 3
        self.spam_repetition_fuzzy_threshold = 85

        self.mention_limit = 5
        self.max_message_length = 1500
        self.max_attachments = 5

        self.min_msg_len_for_lang_check = 5 
        self.min_confidence_for_lang_flagging = 0.65
        self.min_confidence_short_msg_lang = 0.75
        self.short_msg_threshold_lang = 25
        self.common_safe_foreign_words = {"bonjour", "hola", "merci", "gracias", "oui", "si", "nyet", "da", "salut", "ciao", "hallo", "guten tag", "privet", "konnichiwa", "arigato", "sawasdee", "namaste", "scheiße", "scheisse"}
        
        self.fuzzy_match_threshold_keywords = 88

        self.sightengine_nudity_sexual_activity_threshold = 0.55
        self.sightengine_nudity_suggestive_threshold = 0.65
        self.sightengine_gore_threshold = 0.65
        self.sightengine_offensive_symbols_threshold = 0.55

        self.proactive_flagging_openai_threshold = 0.55

        self.delete_violating_messages = True
        self.send_in_channel_warning = True
        self.in_channel_warning_delete_delay = 30

bot_config = BotConfig()


# --- Utility Functions ---

def load_terms_from_file(filepath: str) -> tuple[set[str], list[str]]:
    """Loads terms from a text file, separating single words and multi-word phrases."""
    words = set()
    phrases = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                term = line.strip().lower()
                if not term or term.startswith("#"):
                    continue
                if ' ' in term:
                    phrases.append(term)
                else:
                    words.add(term)
        logger.info(f"Loaded {len(words)} words and {len(phrases)} phrases from {filepath}")
    except FileNotFoundError:
        logger.warning(f"Warning: Terms file '{filepath}' not found. No terms loaded for this category.")
    except Exception as e:
        logger.error(f"Error loading terms from {filepath}: {e}", exc_info=True)
    return words, phrases

discrimination_words_set, discrimination_phrases = load_terms_from_file('discrimination_terms.txt')
nsfw_text_words_set, nsfw_text_phrases = load_terms_from_file('nsfw_terms.txt')


def clean_message_for_language_detection(text: str) -> str:
    """
    Cleans message content specifically for language detection.
    This is crucial for improving fasttext's accuracy by removing "noise".
    """
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<@!?\d+>|<#\d+>|<@&\d+>', '', text)
    text = re.sub(r'<a?:\w+:\d+>', '', text)
    text = re.sub(r'(\*|_|`|~|>|\|)', '', text)
    normalized_text = re.sub(r'\s+', ' ', text).strip()
    return normalized_text


def clean_message_content(text: str) -> str:
    """Cleans and normalizes message content for general analysis."""
    normalized_text = re.sub(r'\s+', ' ', text).strip()
    return normalized_text.lower()


async def get_guild_config(guild_id: int, key: str, default_value=None):
    """Retrieves a guild-specific configuration value from the database."""
    if not db_conn:
        logger.error("get_guild_config: Database connection is not available.")
        return default_value
    try:
        async with db_conn.execute("SELECT config_value FROM guild_configs WHERE guild_id = ? AND config_key = ?", (guild_id, key)) as cursor:
            result = await cursor.fetchone()
            if result and result[0] is not None:
                try:
                    return json.loads(result[0])
                except json.JSONDecodeError:
                    return result[0]
            return default_value
    except Exception as e:
        logger.error(f"Error getting guild config for guild {guild_id}, key {key}: {e}", exc_info=True)
        return default_value


async def set_guild_config(guild_id: int, key: str, value_to_set):
    """Sets or updates a guild-specific configuration value in the database."""
    if not db_conn:
        logger.error("set_guild_config: Database connection is not available.")
        return

    stored_value = value_to_set
    if isinstance(value_to_set, (list, dict)):
        stored_value = json.dumps(value_to_set)
    elif value_to_set is None:
        stored_value = None
    else:
        stored_value = str(value_to_set)

    try:
        async with db_conn.cursor() as cursor:
            if stored_value is None:
                 await cursor.execute('DELETE FROM guild_configs WHERE guild_id = ? AND config_key = ?', (guild_id, key))
                 logger.info(f"Cleared guild config: Guild {guild_id}, Key '{key}'")
            else:
                await cursor.execute(
                    'INSERT OR REPLACE INTO guild_configs (guild_id, config_key, config_value) VALUES (?, ?, ?)',
                    (guild_id, key, stored_value)
                )
                logger.info(f"Set guild config: Guild {guild_id}, Key '{key}', Value '{value_to_set}'")
        await db_conn.commit()
    except Exception as e:
        logger.error(f"Error setting guild config for guild {guild_id}, key {key}: {e}", exc_info=True)


async def detect_language_ai(text: str) -> tuple[str | None, float]:
    """
    Detect the language of the given text using FastText.
    Returns (lang_code | None, confidence_score).
    """
    clean_text = clean_message_for_language_detection(text)
    if not clean_text or len(clean_text.split()) < bot_config.min_msg_len_for_lang_check:
        return None, 0.0

    if not LANGUAGE_MODEL:
        logger.warning("FastText model not loaded. Cannot detect language.")
        return None, 0.0
    try:
        prediction = LANGUAGE_MODEL.predict(clean_text, k=1)
        if prediction and prediction[0] and prediction[1] and len(prediction[0]) > 0:
            lang_code = prediction[0][0].replace("__label__", "")
            confidence = float(prediction[1][0])
            return lang_code, confidence
        else:
            logger.warning(f"FastText returned unexpected prediction format for: '{clean_text[:100]}...' -> {prediction}")
            return None, 0.0
    except Exception as e:
        logger.error(f"FastText language detection error for '{clean_text[:100]}...': {e}", exc_info=True)
        return None, 0.0


async def log_moderation_action(
    action_type: str,
    target_user: discord.User | discord.Member,
    reason: str,
    moderator: discord.User | discord.Member | None = None,
    guild: discord.Guild | None = None,
    color: discord.Color = discord.Color.orange(),
    extra_fields: list[tuple[str, str]] | None = None,
    message_url: str | None = None
):
    """Logs moderation actions to a specified channel and console with a standardized embed."""
    current_guild = guild or (target_user.guild if isinstance(target_user, discord.Member) else None)
    if not current_guild:
        logger.error(f"Cannot log action '{action_type}' for user {target_user.id}: Guild context missing.")
        return

    log_channel_id = await get_guild_config(current_guild.id, "log_channel_id", bot_config.default_log_channel_id)
    log_channel = bot.get_channel(log_channel_id) if log_channel_id else None

    embed_title = f"🛡️ Moderation: {action_type.replace('_', ' ').title()}"
    embed = discord.Embed(
        title=embed_title,
        description=reason,
        color=color,
        timestamp=datetime.now(timezone.utc)
    )
    embed.add_field(name="Target User", value=f"{target_user.mention} (`{target_user.id}`)", inline=True)
    
    user_avatar = target_user.display_avatar or target_user.avatar
    if user_avatar:
        embed.set_thumbnail(url=user_avatar.url)

    if moderator:
        embed.add_field(name="Moderator", value=f"{moderator.mention} (`{moderator.id}`)", inline=True)
    else:
        embed.add_field(name="Moderator", value="Automated Action", inline=True)

    if message_url:
        embed.add_field(name="Context", value=f"[Jump to Message]({message_url})", inline=False)

    if extra_fields:
        for name, value in extra_fields:
            embed.add_field(name=name, value=value, inline=False)

    log_message_console = (f"Log (Guild: {current_guild.name}, Action: {action_type}, Target: {target_user.id}, Reason: {reason})")
    logger.info(log_message_console)

    if log_channel:
        try:
            await log_channel.send(embed=embed)
        except discord.Forbidden:
            logger.error(f"Missing permissions to send logs to channel #{log_channel.name} ({log_channel.id}) in guild {current_guild.name}.")
        except Exception as e:
            logger.error(f"Error sending log embed to channel for guild {current_guild.name}: {e}", exc_info=True)
    else:
        logger.warning(f"Log channel ID {log_channel_id} not found or not configured for guild {current_guild.name}.")


# --- External API Calls with Retries ---

def retry_if_api_error(exception):
    """Retries on server errors (5xx), rate limits (429), or network issues for API calls."""
    if isinstance(exception, client_exceptions.ClientResponseError):
        return exception.status == 429 or exception.status >= 500
    return isinstance(exception, (asyncio.TimeoutError, client_exceptions.ClientOSError, client_exceptions.ClientConnectorError))


@retry(
    stop=stop_after_attempt(4), 
    wait=wait_random_exponential(multiplier=1, min=3, max=30), 
    retry=retry_if_api_error,
    reraise=True
)
async def check_openai_moderation_api(text_content: str) -> dict:
    """
    Checks text against the OpenAI moderation API with robust, exponential backoff retries.
    This decorator automatically handles the "429 Too Many Requests" error.
    """
    if not OPENAI_API_KEY:
        logger.debug("OPENAI_API_KEY not set. Skipping OpenAI moderation.")
        return {"flagged": False, "categories": {}, "category_scores": {}}
    if not text_content.strip():
        return {"flagged": False, "categories": {}, "category_scores": {}}

    url = "https://api.openai.com/v1/moderations"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"input": text_content.replace("\n", " ")}

    if not http_session or http_session.closed:
        logger.error("HTTP session not available for OpenAI moderation.")
        return {"flagged": False, "categories": {}, "category_scores": {}}

    try:
        async with http_session.post(url, headers=headers, json=data, timeout=15) as response:
            response.raise_for_status()
            json_response = await response.json()
            results = json_response.get("results", [])
            if results:
                return results[0]
            logger.warning(f"OpenAI moderation returned empty results for text: {text_content[:100]}...")
            return {"flagged": False, "categories": {}, "category_scores": {}}
    except client_exceptions.ClientResponseError as e:
        if e.status == 400:
            logger.warning(f"OpenAI moderation: 400 Bad Request (will not retry). Text: '{text_content[:100]}...'. Error: {e.message}")
            return {"flagged": False, "categories": {}, "category_scores": {}}
        logger.error(f"OpenAI moderation API error: {e.status} - {e.message}. Text: '{text_content[:100]}...'. Retrying if applicable.")
        raise
    except asyncio.TimeoutError:
        logger.error(f"OpenAI moderation API request timed out. Text: {text_content[:100]}...")
        raise
    except Exception as e:
        logger.error(f"Unexpected error with OpenAI moderation API: {e} for text: {text_content[:100]}...", exc_info=True)
        return {"flagged": False, "categories": {}, "category_scores": {}}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_api_error,
    reraise=True
)
async def check_sightengine_media_api(image_url: str) -> dict:
    """Checks image against Sightengine API for moderation flags."""
    if not SIGHTENGINE_API_USER or not SIGHTENGINE_API_SECRET:
        logger.debug("Sightengine API keys not set. Skipping image moderation.")
        return {}

    if not http_session or http_session.closed:
        logger.error("HTTP session not available for Sightengine moderation.")
        return {}

    models = "nudity-2.0,gore,offensive"
    encoded_url = quote_plus(image_url)
    url = f"https://api.sightengine.com/1.0/check.json?url={encoded_url}&models={models}&api_user={SIGHTENGINE_API_USER}&api_secret={SIGHTENGINE_API_SECRET}"

    try:
        async with http_session.get(url, timeout=20) as response:
            response.raise_for_status()
            json_response = await response.json()
            if json_response.get("status") == "success":
                return json_response
            else:
                logger.warning(f"Sightengine API returned non-success status: {json_response.get('status')} - {json_response.get('error', {}).get('message')} for URL: {image_url}")
                return {}
    except client_exceptions.ClientResponseError as e:
        logger.error(f"Sightengine API error: {e.status} - {e.message} for URL: {image_url}. Retrying if applicable.")
        raise
    except asyncio.TimeoutError:
        logger.error(f"Sightengine API request timed out for URL: {image_url}...")
        raise
    except Exception as e:
        logger.error(f"Unexpected error with Sightengine API for URL: {image_url}: {e}", exc_info=True)
        return {}

# --- Core Moderation Logic ---

async def apply_moderation_punishment(
    member: discord.Member,
    action_config: dict,
    total_points: int,
    violation_summary: str,
    moderator: discord.User | None = None
):
    """Applies a specified moderation action to a member and logs it."""
    action_type = action_config["action"]
    guild = member.guild
    reason_suffix = action_config.get("reason_suffix", f"Automated action due to reaching {total_points} points.")
    
    if moderator:
        full_reason = f"Manual action by {moderator.name}: {reason_suffix}"
    else:
        full_reason = f"Automated Action | {reason_suffix} | Violations: {violation_summary}"
    
    dm_message_base = action_config.get("dm_message", f"Action taken: {action_type}. Reason: {full_reason}")
    dm_message_text = f"Hello {member.name},\n\nRegarding your activity in **{guild.name}**:\n\n{dm_message_base}"
    
    log_color = discord.Color.orange()
    extra_log_fields = []
    duration: timedelta | None = None

    if "duration_hours" in action_config:
        duration = timedelta(hours=action_config["duration_hours"])
    elif "duration_days" in action_config:
        duration = timedelta(days=action_config["duration_days"])

    if duration:
        dm_message_text += f"\n\nThis action is effective for: **{str(duration)}**."
        extra_log_fields.append(("Duration", str(duration)))

    try:
        if action_type == "warn":
            log_color = discord.Color.gold()
        elif action_type == "mute":
            if duration:
                await member.timeout(duration, reason=full_reason)
                log_color = discord.Color.light_grey()
            else:
                logger.warning(f"Mute action for {member.display_name} ({member.id}) called without duration. Action skipped.")
                return
        elif action_type == "kick":
            log_color = discord.Color.red()
        elif action_type == "temp_ban":
            if duration:
                unban_time = datetime.now(timezone.utc) + duration
                if db_conn:
                    async with db_conn.cursor() as cursor:
                        await cursor.execute(
                            'INSERT OR REPLACE INTO temp_bans (user_id, guild_id, unban_time, ban_reason) VALUES (?, ?, ?, ?)',
                            (member.id, guild.id, unban_time.isoformat(), full_reason)
                        )
                        await db_conn.commit()
                dm_message_text += f"\n\nYou will be unbanned automatically around: {discord.utils.format_dt(unban_time, 'F')}."
                log_color = discord.Color.dark_red()
                extra_log_fields.append(("Unban Time", f"{discord.utils.format_dt(unban_time, 'R')} ({discord.utils.format_dt(unban_time, 'f')})"))
            else:
                logger.warning(f"Temp_ban action for {member.display_name} ({member.id}) called without duration. Action skipped.")
                return
        elif action_type == "ban":
            log_color = discord.Color.dark_red()
        
        try:
            if dm_message_text:
                await member.send(dm_message_text)
                logger.info(f"DM sent to {member.display_name} ({member.id}) for {action_type}.")
        except discord.Forbidden:
            logger.warning(f"Could not DM {action_type} notification to {member.display_name} ({member.id}) - DMs disabled or bot blocked.")
        except Exception as e:
            logger.error(f"Error sending DM to {member.display_name} ({member.id}) for {action_type}: {e}", exc_info=True)

        if action_type == "kick":
            await member.kick(reason=full_reason)
        elif action_type == "temp_ban" or action_type == "ban":
            await member.ban(reason=full_reason, delete_message_seconds=0)

        await log_moderation_action(action_type, member, full_reason, moderator, guild, log_color, extra_log_fields)

    except discord.Forbidden:
        logger.error(f"Missing permissions to {action_type} {member.display_name} ({member.id}) in {guild.name}. "
                     f"Please check bot role hierarchy and permissions (e.g., 'Timeout Members', 'Kick Members', 'Ban Members').")
        await log_moderation_action(f"{action_type}_failed_permission", member, f"Bot lacks permissions for '{action_type}'. Original reason: {full_reason}", moderator, guild, discord.Color.red(), extra_log_fields)
    except discord.HTTPException as e:
        logger.error(f"Discord API error while applying {action_type} to {member.display_name} ({member.id}): {e.status} - {e.text}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error applying {action_type} to {member.display_name} ({member.id}): {e}", exc_info=True)


async def process_infractions_and_punish(
    member: discord.Member,
    guild: discord.Guild,
    violation_types_this_message: list[str],
    message_content: str,
    message_url: str | None
):
    """
    Records infractions for a message, calculates total points, and applies punishment if a threshold is met.
    """
    if not db_conn:
        logger.error("process_infractions_and_punish: Database connection is not available.")
        return

    user_id = member.id
    guild_id = guild.id
    total_points_this_message = 0
    violation_summary_parts = []

    async with db_conn.cursor() as cursor:
        for violation_type in violation_types_this_message:
            violation_config = bot_config.punishment_system["violations"].get(violation_type)
            if not violation_config:
                logger.warning(f"Unknown violation type '{violation_type}' encountered for user {user_id}. Skipping.")
                continue

            points = violation_config["points"]
            total_points_this_message += points
            violation_summary_parts.append(violation_type.replace('_', ' ').title())

            try:
                await cursor.execute(
                    "INSERT INTO infractions (user_id, guild_id, violation_type, points, message_content_snippet, message_url, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (user_id, guild_id, violation_type, points, message_content[:200], message_url, datetime.now(timezone.utc).isoformat())
                )
            except Exception as e:
                logger.error(f"Failed to record infraction {violation_type} for user {user_id} in DB: {e}", exc_info=True)

        if not violation_summary_parts:
            await db_conn.commit()
            return

        await db_conn.commit()

        thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        await cursor.execute(
            "SELECT SUM(points) FROM infractions WHERE user_id = ? AND guild_id = ? AND timestamp >= ?",
            (user_id, guild_id, thirty_days_ago)
        )
        total_active_points_result = await cursor.fetchone()
        total_active_points = total_active_points_result[0] if total_active_points_result and total_active_points_result[0] is not None else 0
        logger.info(f"User {user_id} in guild {guild_id} now has {total_active_points} active points (after message adding {total_points_this_message} points).")

        applicable_punishment_config = None
        highest_threshold_met = -1

        for threshold_points, punishment_details in sorted(bot_config.punishment_system["points_thresholds"].items(), key=lambda item: item[0], reverse=True):
            if total_active_points >= threshold_points:
                applicable_punishment_config = punishment_details
                highest_threshold_met = threshold_points
                break

        if applicable_punishment_config:
            logger.info(f"User {user_id} met punishment threshold {highest_threshold_met} ({applicable_punishment_config['action']}) with {total_active_points} points.")

            violation_summary_str = ", ".join(violation_summary_parts)
            await apply_moderation_punishment(member, applicable_punishment_config, total_active_points, violation_summary_str)

            if applicable_punishment_config["action"] in ["ban"]:
                logger.info(f"Clearing infractions for user {user_id} in guild {guild_id} due to '{applicable_punishment_config['action']}' action.")
                await clear_user_infractions(user_id, guild_id, cursor)
                await db_conn.commit()

# --- Database Management ---

async def get_user_infractions_from_db(user_id: int, guild_id: int, days_limit: int = 0) -> tuple[list[dict], int]:
    """
    Retrieves infractions for a user, optionally limited by days.
    Returns a list of infraction dicts and the total active points (within 30 days).
    """
    if not db_conn: return [], 0
    infractions = []
    total_active_points = 0
    try:
        query = f"SELECT id, violation_type, points, message_content_snippet, message_url, timestamp FROM infractions WHERE user_id = ? AND guild_id = ? {'AND timestamp >= ?' if days_limit > 0 else ''} ORDER BY timestamp DESC"
        params = (user_id, guild_id, (datetime.now(timezone.utc) - timedelta(days=days_limit)).isoformat()) if days_limit > 0 else (user_id, guild_id)

        async with db_conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                infractions.append({
                    "id": row[0], "violation_type": row[1], "points": row[2],
                    "message_content_snippet": row[3], "message_url": row[4], "timestamp": row[5]
                })

        thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        async with db_conn.execute(
            "SELECT SUM(points) FROM infractions WHERE user_id = ? AND guild_id = ? AND timestamp >= ?",
            (user_id, guild_id, thirty_days_ago)
        ) as cursor:
            result = await cursor.fetchone()
            total_active_points = result[0] if result and result[0] is not None else 0

    except Exception as e:
        logger.error(f"Error retrieving infractions for user {user_id} in guild {guild_id}: {e}", exc_info=True)
    return infractions, total_active_points


async def clear_user_infractions(user_id: int, guild_id: int, cursor: aiosqlite.Cursor | None = None):
    """Clears all infractions for a user in a specific guild."""
    if not db_conn: return
    
    async def _clear(cur: aiosqlite.Cursor):
        await cur.execute("DELETE FROM infractions WHERE user_id = ? AND guild_id = ?", (user_id, guild_id))
    
    try:
        if cursor:
            await _clear(cursor)
        else:
            async with db_conn.cursor() as new_cursor:
                await _clear(new_cursor)
            await db_conn.commit()
        logger.info(f"All infractions cleared for user {user_id} in guild {guild_id}.")
    except Exception as e:
        logger.error(f"Error clearing infractions for user {user_id} in guild {guild_id}: {e}", exc_info=True)


async def remove_specific_infraction_from_db(infraction_id: int):
    """Removes a single infraction by its primary key ID."""
    if not db_conn: return False
    try:
        async with db_conn.execute("DELETE FROM infractions WHERE id = ?", (infraction_id,)) as cursor:
            await db_conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Infraction {infraction_id} removed.")
                return True
            return False
    except Exception as e:
        logger.error(f"Error removing infraction {infraction_id}: {e}", exc_info=True)
        return False


async def get_temp_bans_from_db() -> list[dict]:
    """Fetches all active temporary bans from the database."""
    if not db_conn: return []
    temp_bans = []
    try:
        async with db_conn.execute("SELECT user_id, guild_id, unban_time, ban_reason FROM temp_bans") as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                temp_bans.append({
                    "user_id": row[0], "guild_id": row[1],
                    "unban_time": datetime.fromisoformat(row[2]), "ban_reason": row[3]
                })
    except Exception as e:
        logger.error(f"Error retrieving temporary bans: {e}", exc_info=True)
    return temp_bans


async def remove_temp_ban_from_db(user_id: int, guild_id: int):
    """Removes a temp ban record after the user has been unbanned."""
    if not db_conn: return
    try:
        async with db_conn.execute("DELETE FROM temp_bans WHERE user_id = ? AND guild_id = ?", (user_id, guild_id)):
            await db_conn.commit()
        logger.info(f"Temp ban DB entry removed for user {user_id} in guild {guild_id}.")
    except Exception as e:
        logger.error(f"Error removing temp ban DB entry for user {user_id} in guild {guild_id}: {e}", exc_info=True)


async def setup_db():
    """Initializes the SQLite database and creates all necessary tables."""
    global db_conn
    try:
        db_conn = await aiosqlite.connect('adroit_bot_data.db')
        await db_conn.execute("""
            CREATE TABLE IF NOT EXISTS guild_configs (
                guild_id INTEGER NOT NULL,
                config_key TEXT NOT NULL,
                config_value TEXT,
                PRIMARY KEY (guild_id, config_key)
            )
        """)
        await db_conn.execute("""
            CREATE TABLE IF NOT EXISTS infractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL,
                violation_type TEXT NOT NULL,
                points INTEGER NOT NULL,
                message_content_snippet TEXT,
                message_url TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_infractions_user_guild_time ON infractions (user_id, guild_id, timestamp)')
        await db_conn.execute("""
            CREATE TABLE IF NOT EXISTS temp_bans (
                user_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL,
                unban_time TEXT NOT NULL,
                ban_reason TEXT,
                PRIMARY KEY (user_id, guild_id)
            )
        """)
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_temp_bans_unban_time ON temp_bans (unban_time)')
        await db_conn.execute("""
            CREATE TABLE IF NOT EXISTS dynamic_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                rule_type TEXT NOT NULL, -- 'forbidden_word', 'forbidden_phrase', 'forbidden_regex'
                pattern TEXT NOT NULL,
                added_by_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        await db_conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_dynamic_rules_guild_type_pattern ON dynamic_rules (guild_id, rule_type, pattern)')
        await db_conn.execute("""
            CREATE TABLE IF NOT EXISTS review_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                channel_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                message_content TEXT NOT NULL,
                reason TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_review_queue_guild_timestamp ON review_queue (guild_id, timestamp)')

        await db_conn.commit()
        logger.info("Database initialized and tables checked/created successfully.")
    except Exception as e:
        logger.critical(f"Failed to connect to or initialize database: {e}", exc_info=True)
        if db_conn: await db_conn.close()
        exit(1)


async def load_dynamic_rules_from_db():
    """Loads all dynamic rules from the database into memory for fast checking."""
    if not db_conn:
        logger.error("Cannot load dynamic rules: Database connection not available.")
        return
    
    dynamic_rules["forbidden_words"].clear()
    dynamic_rules["forbidden_phrases"].clear()
    dynamic_rules["forbidden_regex"] = []

    try:
        async with db_conn.execute("SELECT guild_id, rule_type, pattern FROM dynamic_rules") as cursor:
            rows = await cursor.fetchall()
            for guild_id, rule_type, pattern in rows:
                if rule_type == 'forbidden_word':
                    dynamic_rules["forbidden_words"].add(pattern.lower())
                elif rule_type == 'forbidden_phrase':
                    dynamic_rules["forbidden_phrases"].append(pattern.lower())
                elif rule_type == 'forbidden_regex':
                    try:
                        dynamic_rules["forbidden_regex"].append(re.compile(pattern, re.IGNORECASE))
                    except re.error as e:
                        logger.error(f"Failed to compile regex from DB: '{pattern}'. Error: {e}")
            logger.info(f"Loaded {len(dynamic_rules['forbidden_words'])} words, "
                        f"{len(dynamic_rules['forbidden_phrases'])} phrases, and "
                        f"{len(dynamic_rules['forbidden_regex'])} regex patterns from dynamic rules.")
    except Exception as e:
        logger.error(f"Error loading dynamic rules from database: {e}", exc_info=True)


# --- Discord Cogs (Command Groups) ---

class GeneralCog(commands.Cog, name="General"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="ping", description="Checks the bot's latency.")
    async def ping(self, interaction: discord.Interaction):
        latency = round(self.bot.latency * 1000)
        await interaction.response.send_message(f"Pong! Latency: {latency}ms", ephemeral=True)

    @app_commands.command(name="help", description="Shows information about bot commands.")
    async def help_command(self, interaction: discord.Interaction):
        embed = discord.Embed(title="Adroit Bot Help", description="Here are the available commands:", color=discord.Color.blue())
        
        for cog_name, cog in self.bot.cogs.items():
            command_list = []
            for cmd in cog.get_app_commands():
                if isinstance(cmd, app_commands.Command):
                     command_list.append(f"`/{cmd.name}` - {cmd.description}")

            if command_list:
                 embed.add_field(name=f"**{cog_name} Commands**", value="\n".join(command_list), inline=False)
        
        if not embed.fields:
            embed.description = "No commands found."

        await interaction.response.send_message(embed=embed, ephemeral=True)


class ConfigurationCog(commands.Cog, name="Configuration"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="set_log_channel", description="Sets the channel for moderation logs.")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(channel="The text channel to send logs to. Leave empty to clear.")
    async def set_log_channel(self, interaction: discord.Interaction, channel: discord.TextChannel | None = None):
        if not interaction.guild_id:
            await interaction.response.send_message("This command must be used in a server.", ephemeral=True)
            return
        
        if channel:
            await set_guild_config(interaction.guild_id, "log_channel_id", channel.id)
            await interaction.response.send_message(f"Moderation logs will now be sent to {channel.mention}.", ephemeral=True)
            await log_moderation_action("config_change", interaction.user, f"Set log channel to {channel.mention}", guild=interaction.guild, color=discord.Color.blue())
        else:
            await set_guild_config(interaction.guild_id, "log_channel_id", None)
            await interaction.response.send_message("Moderation log channel has been cleared.", ephemeral=True)
            await log_moderation_action("config_change", interaction.user, "Cleared log channel setting.", guild=interaction.guild, color=discord.Color.blue())

    @app_commands.command(name="set_channel_language", description="Sets expected language(s) for a channel (e.g., en,fr). 'any' to disable.")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(channel="The channel to configure.", languages="Comma-separated ISO 639-1 language codes (e.g., en,fr,es). Use 'any' to disable.")
    async def set_channel_language(self, interaction: discord.Interaction, channel: discord.TextChannel, languages: str):
        if not interaction.guild_id:
            await interaction.response.send_message("This command can only be used in a server.", ephemeral=True)
            return
        
        await interaction.response.defer(ephemeral=True)

        lang_list_raw = [lang.strip().lower() for lang in languages.split(',')]
        
        valid_langs = []
        is_any = False
        for lang_code in lang_list_raw:
            if lang_code == "any":
                is_any = True
                break
            if re.fullmatch(r"[a-z]{2,3}", lang_code):
                valid_langs.append(lang_code)
            else:
                await interaction.followup.send(f"Invalid language code: '{lang_code}'. Please use 2-letter ISO 639-1 codes (e.g., 'en', 'fr') or 'any'.")
                return

        config_key = f"channel_language_{channel.id}"
        if is_any:
            await set_guild_config(interaction.guild_id, config_key, ["any"])
            await interaction.followup.send(f"Language checks for {channel.mention} have been set to allow **any** language.")
        elif valid_langs:
            await set_guild_config(interaction.guild_id, config_key, valid_langs)
            await interaction.followup.send(f"Expected languages for {channel.mention} set to: **{', '.join(valid_langs)}**.")
        else:
             await set_guild_config(interaction.guild_id, config_key, None)
             await interaction.followup.send(f"Language configuration for {channel.mention} has been cleared/reset.")


    @app_commands.command(name="get_channel_config", description="Shows current language config for a channel.")
    @app_commands.default_permissions(manage_messages=True)
    @app_commands.describe(channel="The channel to check.")
    async def get_channel_config(self, interaction: discord.Interaction, channel: discord.TextChannel):
        if not interaction.guild_id:
            await interaction.response.send_message("This command can only be used in a server.", ephemeral=True)
            return

        lang_config_key = f"channel_language_{channel.id}"
        db_lang_setting = await get_guild_config(interaction.guild_id, lang_config_key, None)
        
        final_lang_setting = None
        source = "Not Configured"
        if db_lang_setting is not None:
            final_lang_setting = db_lang_setting
            source = "Server Setting"
        elif channel.id in bot_config.default_channel_configs and "language" in bot_config.default_channel_configs[channel.id]:
            final_lang_setting = bot_config.default_channel_configs[channel.id]["language"]
            source = "Bot Default"

        lang_display = "Any"
        if final_lang_setting and "any" not in final_lang_setting:
            lang_display = ", ".join(final_lang_setting).upper()

        embed = discord.Embed(title=f"Configuration for #{channel.name}", color=discord.Color.blurple())
        embed.add_field(name="Expected Language(s)", value=f"{lang_display} (Source: {source})", inline=False)
        await interaction.response.send_message(embed=embed, ephemeral=True)


class ModerationCog(commands.Cog, name="Moderation"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.user_message_timestamps: defaultdict[int, defaultdict[int, deque]] = defaultdict(lambda: defaultdict(deque))
        self.user_message_history: defaultdict[int, defaultdict[int, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=bot_config.spam_repetition_history_count))
        )
        self.openai_cooldowns: defaultdict[int, float] = defaultdict(float)

        self.temp_ban_check_task.start()
        self.cleanup_old_infractions_task.start()
        self.cleanup_spam_trackers_task.start()

    def cog_unload(self):
        """Gracefully stop background tasks when the cog is unloaded."""
        self.temp_ban_check_task.cancel()
        self.cleanup_old_infractions_task.cancel()
        self.cleanup_spam_trackers_task.cancel()

    async def get_effective_channel_language_config(self, guild_id: int, channel_id: int) -> list[str] | None:
        """Gets the effective language config: DB override > BotConfig default > None."""
        db_setting = await get_guild_config(guild_id, f"channel_language_{channel_id}", None)
        if db_setting is not None:
            if isinstance(db_setting, list) and "any" in db_setting: return ["any"]
            return db_setting if isinstance(db_setting, list) else None
        
        default_conf = bot_config.default_channel_configs.get(channel_id)
        return default_conf.get("language") if default_conf else None

    # --- Background Tasks ---
    @tasks.loop(minutes=1)
    async def temp_ban_check_task(self):
        """Periodically checks for expired temporary bans and unbans users."""
        if not db_conn:
            logger.warning("temp_ban_check_task: Database connection not available, skipping.")
            return

        now_utc = datetime.now(timezone.utc)
        expired_bans = await get_temp_bans_from_db()

        for ban_entry in expired_bans:
            user_id, guild_id, unban_time = ban_entry["user_id"], ban_entry["guild_id"], ban_entry["unban_time"]
            original_reason = ban_entry["ban_reason"]

            if now_utc >= unban_time:
                guild = self.bot.get_guild(guild_id)
                if not guild:
                    logger.warning(f"Guild {guild_id} not found for expired temp ban of user {user_id}. Removing DB entry.")
                    await remove_temp_ban_from_db(user_id, guild_id)
                    continue

                user_obj = discord.Object(id=user_id)
                unban_reason = f"Temporary ban expired. Original reason: {original_reason}"
                try:
                    await guild.unban(user_obj, reason=unban_reason)
                    await remove_temp_ban_from_db(user_id, guild_id)
                    logger.info(f"User {user_id} unbanned from {guild.name} (temp ban expired).")
                    
                    target_user_for_log = await self.bot.fetch_user(user_id)
                    await log_moderation_action("unban_temp_expired", target_user_for_log, unban_reason, guild=guild, color=discord.Color.green())
                except discord.NotFound:
                    logger.info(f"User {user_id} not found in ban list of {guild.name}. Removing temp ban entry.")
                    await remove_temp_ban_from_db(user_id, guild_id)
                except discord.Forbidden:
                    logger.error(f"Missing permissions to unban user {user_id} from {guild.name}.")
                except Exception as e:
                    logger.error(f"Error unbanning user {user_id} from {guild.name}: {e}", exc_info=True)

    @tasks.loop(hours=24)
    async def cleanup_old_infractions_task(self):
        """Periodically deletes very old infraction records from the database."""
        if not db_conn: return
        ninety_days_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        try:
            async with db_conn.execute("DELETE FROM infractions WHERE timestamp < ?", (ninety_days_ago,)) as cursor:
                await db_conn.commit()
                if cursor.rowcount > 0:
                    logger.info(f"Database Hygiene: Deleted {cursor.rowcount} infraction records older than 90 days.")
        except Exception as e:
            logger.error(f"Error during old infraction cleanup task: {e}", exc_info=True)

    @tasks.loop(hours=6)
    async def cleanup_spam_trackers_task(self):
        """Cleans up old entries from in-memory spam trackers to prevent memory leaks."""
        now_ts = datetime.now(timezone.utc).timestamp()
        for guild_id in list(self.user_message_timestamps.keys()):
            for user_id in list(self.user_message_timestamps[guild_id].keys()):
                self.user_message_timestamps[guild_id][user_id] = deque(
                    ts for ts in self.user_message_timestamps[guild_id][user_id]
                    if (now_ts - ts) < (bot_config.spam_window_seconds * 6)
                )
                if not self.user_message_timestamps[guild_id][user_id]:
                    del self.user_message_timestamps[guild_id][user_id]
            if not self.user_message_timestamps[guild_id]:
                del self.user_message_timestamps[guild_id]
        logger.info("Periodic spam tracker cleanup completed.")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if not message.guild or message.author.bot or message.webhook_id or not isinstance(message.author, discord.Member):
            return

        if message.author.guild_permissions.manage_messages:
            return await self.bot.process_commands(message)

        content_raw = message.content
        cleaned_content_for_matching = clean_message_content(content_raw)
        member, guild, channel = message.author, message.guild, message.channel
        
        violations_found = set()
        
        lang_violations = await self.check_language(guild.id, channel.id, content_raw)
        violations_found.update(lang_violations)


        violations_found.update(self.check_dynamic_rules(content_raw, cleaned_content_for_matching))
        violations_found.update(self.check_spam(member.id, guild.id, cleaned_content_for_matching))
        violations_found.update(self.check_advertising(content_raw))
        violations_found.update(self.check_message_limits(message))
        violations_found.update(self.check_keyword_violations(cleaned_content_for_matching))
        
        openai_proactive_flag_reason = None
        if not violations_found: 
            ai_text_violations, openai_proactive_flag_reason = await self.check_ai_text_moderation(content_raw, member.id)
            violations_found.update(ai_text_violations)
        
        violations_found.update(await self.check_ai_media_moderation(message.attachments))

        if violations_found:
            logger.info(f"User {member.id} ({member.name}) triggered violations: {list(violations_found)}. Message: {message.jump_url}")

            if bot_config.delete_violating_messages:
                try:
                    await message.delete()
                except (discord.Forbidden, discord.NotFound):
                    pass 

            if bot_config.send_in_channel_warning:
                viol_summary = ", ".join(v.replace('_', ' ').title() for v in violations_found)
                warn_text = f"{member.mention}, your message was moderated due to: **{viol_summary}**. Please review server rules."
                try:
                    warning_message = await channel.send(warn_text)
                    await asyncio.sleep(bot_config.in_channel_warning_delete_delay)
                    await warning_message.delete()
                except discord.Forbidden:
                    pass
                except discord.NotFound: 
                    pass
                except Exception as e:
                    logger.error(f"Error sending/deleting in-channel warning: {e}", exc_info=True)


            await process_infractions_and_punish(member, guild, list(violations_found), content_raw, message.jump_url)

        elif openai_proactive_flag_reason:
            await self.add_to_review_queue(message, openai_proactive_flag_reason)
        else:
            await self.bot.process_commands(message)

    def check_dynamic_rules(self, raw_content: str, cleaned_content: str) -> set[str]:
        """Checks against moderator-added custom rules."""
        violations = set()
        for pattern_re in dynamic_rules["forbidden_regex"]:
            if pattern_re.search(raw_content):
                violations.add("dynamic_rule_violation")
                break
        if not violations:
            if any(word in dynamic_rules["forbidden_words"] for word in cleaned_content.split()):
                violations.add("dynamic_rule_violation")
            elif any(phrase in cleaned_content for phrase in dynamic_rules["forbidden_phrases"]):
                violations.add("dynamic_rule_violation")
        return violations

    def check_spam(self, user_id: int, guild_id: int, cleaned_content: str) -> set[str]:
        """Checks for message rate and repetition spam."""
        violations = set()
        now_ts = datetime.now(timezone.utc).timestamp()
        
        timestamps = self.user_message_timestamps[guild_id][user_id]
        timestamps.append(now_ts)
        while timestamps and (now_ts - timestamps[0]) > bot_config.spam_window_seconds:
            timestamps.popleft()
        if len(timestamps) > bot_config.spam_message_limit:
            violations.add("spam_rate")

        if cleaned_content and "spam_rate" not in violations:
            history = self.user_message_history[guild_id][user_id]
            if len(history) == bot_config.spam_repetition_history_count:
                similar_count = sum(1 for old_msg in history if fuzz.ratio(cleaned_content, old_msg) >= bot_config.spam_repetition_fuzzy_threshold)
                if similar_count >= bot_config.spam_repetition_history_count - 1:
                    violations.add("spam_repetition")
                    history.clear() 
            history.append(cleaned_content)
        
        return violations
        
    def check_advertising(self, raw_content: str) -> set[str]:
        """Checks for forbidden phrases and unpermitted URLs."""
        if bot_config.forbidden_text_pattern.search(raw_content):
            return {"advertising_forbidden_text"}
        
        potential_urls = re.findall(r"(?:https?://)?(?:www\.)?[\w\.-]+\.\w{2,}(?:/[\S]*)?", raw_content, re.IGNORECASE)
        for purl in potential_urls:
            try:
                domain = urlparse(f'http://{purl}' if not purl.startswith(('http://', 'https://')) else purl).netloc.lower().lstrip('www.')
                if domain and not any(allowed == domain or domain.endswith(f'.{allowed}') for allowed in bot_config.permitted_domains):
                    return {"advertising_unpermitted_url"}
            except Exception:
                continue 
        return set()

    def check_message_limits(self, message: discord.Message) -> set[str]:
        """Checks for excessive mentions, attachments, or length."""
        violations = set()
        if len(message.mentions) > bot_config.mention_limit:
            violations.add("excessive_mentions")
        if len(message.attachments) > bot_config.max_attachments:
            violations.add("excessive_attachments")
        if len(message.content) > bot_config.max_message_length:
            violations.add("long_message")
        return violations
        
    async def check_language(self, guild_id: int, channel_id: int, raw_content: str) -> set[str]:
        """Checks if the message language is allowed in the channel."""
        if len(raw_content.split()) < bot_config.min_msg_len_for_lang_check or not bot_config.has_alphanumeric_pattern.search(raw_content):
            return set()
            
        channel_lang_config = await self.get_effective_channel_language_config(guild_id, channel_id)
        if not channel_lang_config or "any" in channel_lang_config:
            return set()

        lang_code, confidence = await detect_language_ai(raw_content)
        
        if not lang_code or confidence == 0.0:
            return set()

        if lang_code not in channel_lang_config:
            is_short_msg = len(raw_content) < bot_config.short_msg_threshold_lang
            threshold = bot_config.min_confidence_short_msg_lang if is_short_msg else bot_config.min_confidence_for_lang_flagging
            
            cleaned_for_safe_words = clean_message_content(raw_content)
            if any(safe_word in cleaned_for_safe_words.split() for safe_word in bot_config.common_safe_foreign_words):
                 return set()

            if confidence >= threshold:
                logger.info(f"Language violation by {guild_id}: Detected '{lang_code}' ({confidence:.2f}) in '{raw_content[:50]}...'. Allowed: {channel_lang_config}. Threshold: {threshold}")
                return {"foreign_language"}
        return set()


    def check_keyword_violations(self, cleaned_content: str) -> set[str]:
        """Checks for static lists of forbidden keywords/phrases."""
        violations = set()
        for phrase in discrimination_phrases:
            if fuzz.partial_ratio(phrase, cleaned_content) >= bot_config.fuzzy_match_threshold_keywords:
                violations.add("discrimination")
                break
        for phrase in nsfw_text_phrases:
            if fuzz.partial_ratio(phrase, cleaned_content) >= bot_config.fuzzy_match_threshold_keywords:
                violations.add("nsfw_text")
                break
        
        if any(word in discrimination_words_set for word in cleaned_content.split()):
            violations.add("discrimination")
        if any(word in nsfw_text_words_set for word in cleaned_content.split()):
            violations.add("nsfw_text")

        return violations

    async def check_ai_text_moderation(self, raw_content: str, user_id: int) -> tuple[set[str], str | None]:
        """Uses OpenAI API to check for complex text violations."""
        if not OPENAI_API_KEY: return set(), None
        
        now_ts = datetime.now(timezone.utc).timestamp()
        if now_ts < self.openai_cooldowns.get(user_id, 0):
            return set(), None
        self.openai_cooldowns[user_id] = now_ts + 10 

        try:
            openai_result = await check_openai_moderation_api(raw_content)
            
            if openai_result.get("flagged"):
                categories = openai_result.get("categories", {})
                
                severe_categories = [
                    "harassment", "harassment/threatening", "hate", "hate/threatening",
                    "self-harm", "self-harm/intent", "self-harm/instructions",
                    "sexual", "sexual/minors", "violence", "violence/graphic"
                ]
                if any(categories.get(cat) for cat in severe_categories):
                    logger.info(f"OpenAI flagged severe content from user {user_id}: {categories}. Message: {raw_content[:100]}...")
                    return {"openai_flagged_severe"}, None
                else: 
                    logger.info(f"OpenAI flagged moderate content from user {user_id}: {categories}. Message: {raw_content[:100]}...")
                    return {"openai_flagged_moderate"}, None

            category_scores = openai_result.get("category_scores", {})
            
            highest_score = 0
            if category_scores:
                highest_score = max(category_scores.values())

            if highest_score >= bot_config.proactive_flagging_openai_threshold:
                flagged_cats = {k for k, v in openai_result.get("categories", {}).items() if v}
                reason = f"Proactive OpenAI Flag (Score: {highest_score:.2f}, Categories: {', '.join(flagged_cats) or 'N/A'})"
                logger.info(f"OpenAI proactive flagging for user {user_id}: {reason}. Message: {raw_content[:100]}...")
                return set(), reason 
            
        except Exception as e:
            logger.error(f"OpenAI moderation call failed after retries for user {user_id}: {e}")
        
        return set(), None

    async def check_ai_media_moderation(self, attachments: list[discord.Attachment]) -> set[str]:
        """Uses Sightengine API to check for media violations."""
        if not (SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET): return set()

        violations = set() 
        for attachment in attachments:
            if attachment.content_type and attachment.content_type.startswith(("image/", "video/")): 
                try:
                    result = await check_sightengine_media_api(attachment.url)
                    if not result: continue
                    
                    if result.get("nudity", {}).get("sexual_activity", 0) >= bot_config.sightengine_nudity_sexual_activity_threshold or \
                       result.get("nudity", {}).get("suggestive", 0) >= bot_config.sightengine_nudity_suggestive_threshold:
                        violations.add("nsfw_media")
                    if result.get("gore", {}).get("prob", 0) >= bot_config.sightengine_gore_threshold:
                        violations.add("gore_violence_media")
                    if result.get("offensive", {}).get("prob", 0) >= bot_config.sightengine_offensive_symbols_threshold:
                        violations.add("offensive_symbols_media")
                except Exception as e:
                    logger.error(f"Sightengine moderation call failed for attachment {attachment.filename}: {e}")
        return violations

    async def add_to_review_queue(self, message: discord.Message, reason: str):
        """Adds a message to the human review queue in the database."""
        if not db_conn: return
        try:
            async with db_conn.cursor() as cursor:
                await cursor.execute("SELECT id FROM review_queue WHERE message_id = ?", (message.id,))
                if await cursor.fetchone() is None:
                    await cursor.execute(
                        "INSERT INTO review_queue (guild_id, user_id, channel_id, message_id, message_content, reason, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (message.guild.id, message.author.id, message.channel.id, message.id, message.content, reason, datetime.now(timezone.utc).isoformat())
                    )
                    await db_conn.commit()
                    logger.info(f"Message {message.id} from user {message.author.id} added to review queue. Reason: {reason}")
                    
                    review_channel_id = await get_guild_config(message.guild.id, "review_channel_id", bot_config.default_review_channel_id)
                    review_channel = self.bot.get_channel(review_channel_id)
                    if review_channel:
                        try:
                            embed = discord.Embed(
                                title="New Message Flagged for Review",
                                description=f"**Reason:** {reason}\n**Author:** {message.author.mention} (`{message.author.id}`)\n**Channel:** {message.channel.mention}\n[Jump to Message]({message.jump_url})",
                                color=discord.Color.yellow(),
                                timestamp=datetime.now(timezone.utc)
                            )
                            embed.set_footer(text=f"Message ID: {message.id}")
                            await review_channel.send(embed=embed)
                        except discord.Forbidden:
                            logger.error(f"Missing permissions to send review notification to channel #{review_channel.name} ({review_channel.id}).")
                        except Exception as e:
                            logger.error(f"Error sending review notification embed: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to add message {message.id} to review queue: {e}", exc_info=True)

  

    @app_commands.command(name="infractions", description="View a user's recent infractions and active points.")
    @app_commands.default_permissions(manage_messages=True)
    @app_commands.describe(member="The member to check infractions for.")
    async def infractions_command(self, interaction: discord.Interaction, member: discord.Member):
        if not interaction.guild_id: return
        infraction_list, active_points = await get_user_infractions_from_db(member.id, interaction.guild_id, days_limit=90)

        embed = discord.Embed(title=f"Infraction Report for {member.display_name}", color=discord.Color.orange() if active_points > 0 else discord.Color.green())
        if member.display_avatar: embed.set_thumbnail(url=member.display_avatar.url)
        embed.add_field(name="Total Active Points (Last 30 Days)", value=f"**{active_points}**", inline=False)

        if infraction_list:
            history_str = ""
            for infra in infraction_list[:10]: 
                ts_dt = datetime.fromisoformat(infra['timestamp'])
                entry = (f"ID: `{infra['id']}` - **{infra['points']} pts** ({infra['violation_type'].replace('_', ' ').title()})\n"
                         f"<t:{int(ts_dt.timestamp())}:f>\n")
                if len(history_str) + len(entry) > 1000: break
                history_str += entry
            embed.add_field(name="Recent Infraction History", value=history_str or "None", inline=False)
        else:
            embed.add_field(name="Recent Infraction History", value="No infractions recorded.", inline=False)

        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="clear_infractions", description="MANUALLY clears ALL or a specific infraction for a user.")
    @app_commands.default_permissions(administrator=True)
    @app_commands.describe(member="The member whose infractions to clear.", infraction_id="Optional ID of a specific infraction to remove.")
    async def clear_infractions_command(self, interaction: discord.Interaction, member: discord.Member, infraction_id: int | None = None):
        if not interaction.guild_id: return

        if infraction_id:
            success = await remove_specific_infraction_from_db(infraction_id)
            action_desc = f"infraction ID `{infraction_id}`"
        else:
            await clear_user_infractions(member.id, interaction.guild_id)
            success = True
            action_desc = "all infractions"

        if success:
            await interaction.response.send_message(f"Successfully cleared {action_desc} for {member.mention}.", ephemeral=True)
            await log_moderation_action("infractions_cleared", member, f"Moderator {interaction.user.mention} cleared {action_desc}.", interaction.user, interaction.guild, discord.Color.green())
        else:
            await interaction.response.send_message(f"Failed to clear {action_desc}.", ephemeral=True)
    
    async def _manual_action(self, interaction: discord.Interaction, member: discord.Member, action_key: str, reason: str, duration_value: float | None = None, duration_unit: str | None = None):
        """Helper for manual punishment commands."""
        if not interaction.guild_id:
            await interaction.response.send_message("This command must be used in a server.", ephemeral=True)
            return
        if member.id == self.bot.user.id:
            await interaction.response.send_message("I cannot moderate myself!", ephemeral=True)
            return
        if member.id == interaction.user.id and action_key != "warn": 
             await interaction.response.send_message("You cannot apply this moderation action to yourself.", ephemeral=True)
             return
        if isinstance(member, discord.Member) and member.top_role >= interaction.user.top_role and interaction.guild.owner_id != interaction.user.id :
            await interaction.response.send_message("You cannot moderate a member with an equal or higher role than yourself.", ephemeral=True)
            return

        mock_action_config = {"action": action_key, "reason_suffix": reason, "dm_message": f"You have been manually {action_key}ed by a moderator. Reason: {reason}"}
        
        if duration_value and duration_unit:
            if duration_unit == "hours":
                mock_action_config["duration_hours"] = duration_value
            elif duration_unit == "days":
                mock_action_config["duration_days"] = duration_value
        
        await apply_moderation_punishment(member, mock_action_config, 0, f"Manual Action: {reason}", moderator=interaction.user)
        
        duration_str = ""
        if duration_value and duration_unit: duration_str = f" for {duration_value} {duration_unit}"
        await interaction.response.send_message(f"Successfully applied **{action_key.upper()}** to {member.mention}{duration_str}. Reason: {reason}", ephemeral=True)

    @app_commands.command(name="manual_warn", description="Manually warn a member.")
    @app_commands.default_permissions(kick_members=True) 
    @app_commands.describe(member="The member to warn.", reason="Reason for the warning.")
    async def manual_warn_command(self, interaction: discord.Interaction, member: discord.Member, reason: str):
        await self._manual_action(interaction, member, "warn", reason)

    @app_commands.command(name="manual_mute", description="Manually mute a member.")
    @app_commands.default_permissions(moderate_members=True) 
    @app_commands.describe(member="The member to mute.", duration_hours="Duration of mute in hours.", reason="Reason for the mute.")
    async def manual_mute_command(self, interaction: discord.Interaction, member: discord.Member, duration_hours: app_commands.Range[float, 0.01], reason: str):
        await self._manual_action(interaction, member, "mute", reason, duration_hours, "hours")

    @app_commands.command(name="manual_kick", description="Manually kick a member.")
    @app_commands.default_permissions(kick_members=True)
    @app_commands.describe(member="The member to kick.", reason="Reason for the kick.")
    async def manual_kick_command(self, interaction: discord.Interaction, member: discord.Member, reason: str):
        await self._manual_action(interaction, member, "kick", reason)

    @app_commands.command(name="manual_ban", description="Manually ban a member (temporarily or permanently).")
    @app_commands.default_permissions(ban_members=True)
    @app_commands.describe(user="The user to ban (can be ID if not in server).", reason="Reason for the ban.", duration_days="Optional: Duration in days for a temporary ban. Omit for permanent.")
    async def manual_ban_command(self, interaction: discord.Interaction, user: discord.User, reason: str, duration_days: app_commands.Range[float, 0.01] | None = None):
        if not interaction.guild:
            await interaction.response.send_message("This command must be used in a server.", ephemeral=True)
            return
        
        target_member = interaction.guild.get_member(user.id) 

        if target_member: 
            if target_member.id == self.bot.user.id:
                await interaction.response.send_message("I cannot moderate myself!", ephemeral=True)
                return
            if target_member.id == interaction.user.id:
                 await interaction.response.send_message("You cannot ban yourself.", ephemeral=True)
                 return
            if target_member.top_role >= interaction.user.top_role and interaction.guild.owner_id != interaction.user.id:
                await interaction.response.send_message("You cannot moderate a member with an equal or higher role than yourself.", ephemeral=True)
                return
            
            action_key = "temp_ban" if duration_days else "ban"
            await self._manual_action(interaction, target_member, action_key, reason, duration_days, "days" if duration_days else None)
        
        else: 
            action_type = "temp_ban" if duration_days else "ban"
            full_reason = f"[{action_type.upper()}] Manual action by {interaction.user.name}: {reason}"
            dm_message_text = f"Hello {user.name},\n\nRegarding your status with **{interaction.guild.name}**:\n\nYou have been manually {action_type}ed. Reason: {reason}"
            log_color = discord.Color.dark_red()
            extra_log_fields = []
            
            try:
                if duration_days:
                    duration = timedelta(days=duration_days)
                    unban_time = datetime.now(timezone.utc) + duration
                    if db_conn:
                        async with db_conn.cursor() as cursor:
                            await cursor.execute(
                                'INSERT OR REPLACE INTO temp_bans (user_id, guild_id, unban_time, ban_reason) VALUES (?, ?, ?, ?)',
                                (user.id, interaction.guild_id, unban_time.isoformat(), full_reason)
                            )
                            await db_conn.commit()
                    dm_message_text += f"\n\nThis action is effective for: **{str(duration)}**.\nYou will be unbanned automatically around: {unban_time.strftime('%Y-%m-%d %H:%M:%S UTC')}."
                    extra_log_fields.append(("Duration", str(duration)))
                    extra_log_fields.append(("Unban Time", unban_time.strftime('%Y-%m-%d %H:%M:%S UTC')))

                try:
                    await user.send(dm_message_text)
                except discord.Forbidden:
                    logger.warning(f"Could not DM ban notification to user {user.id} (not in server or DMs blocked).")
                
                await interaction.guild.ban(user, reason=full_reason, delete_message_seconds=0)
                await log_moderation_action(action_type, user, full_reason, interaction.user, interaction.guild, log_color, extra_log_fields)
                
                duration_str = f" for {duration_days} days" if duration_days else " permanently"
                await interaction.response.send_message(f"Successfully banned {user.mention} (`{user.id}`){duration_str}. Reason: {reason}", ephemeral=True)

            except discord.Forbidden:
                await interaction.response.send_message(f"Failed to ban {user.mention}: Missing permissions.", ephemeral=True)
            except discord.HTTPException as e:
                await interaction.response.send_message(f"Failed to ban {user.mention}: Discord API error {e.status}.", ephemeral=True)
            except Exception as e:
                await interaction.response.send_message(f"An unexpected error occurred while banning {user.mention}.", ephemeral=True)
                logger.error(f"Error in manual_ban for user ID {user.id}: {e}", exc_info=True)

    @app_commands.command(name="manual_unban", description="Manually unban a user by their ID.")
    @app_commands.default_permissions(ban_members=True)
    @app_commands.describe(user_id="The ID of the user to unban.", reason="Reason for the unban.")
    async def manual_unban_command(self, interaction: discord.Interaction, user_id: str, reason: str):
        if not interaction.guild:
            await interaction.response.send_message("This command must be used in a server.", ephemeral=True)
            return
        
        try:
            uid = int(user_id)
            user_obj = discord.Object(id=uid)
        except ValueError:
            await interaction.response.send_message("Invalid User ID format. Please provide a numerical ID.", ephemeral=True)
            return

        full_reason = f"Manual unban by {interaction.user.name}: {reason}"
        try:
            try:
                await interaction.guild.fetch_ban(user_obj)
            except discord.NotFound:
                try: known_user = await self.bot.fetch_user(uid)
                except: known_user = None
                name_str = known_user.name if known_user else f"User ID {uid}"
                await interaction.response.send_message(f"{name_str} is not banned from this server.", ephemeral=True)
                return

            await interaction.guild.unban(user_obj, reason=full_reason)
            await remove_temp_ban_from_db(uid, interaction.guild.id)
            
            try: target_user_for_log = await self.bot.fetch_user(uid)
            except discord.NotFound: target_user_for_log = user_obj

            await log_moderation_action("unban_manual", target_user_for_log, full_reason, interaction.user, interaction.guild, discord.Color.green())
            await interaction.response.send_message(f"Successfully unbanned User ID `{uid}`. Reason: {reason}", ephemeral=True)
        
        except discord.Forbidden:
            await interaction.response.send_message("Failed to unban: Missing permissions.", ephemeral=True)
        except discord.HTTPException as e:
            await interaction.response.send_message(f"Failed to unban: Discord API error {e.status}.", ephemeral=True)
        except Exception as e:
            await interaction.response.send_message("An unexpected error occurred during unban.", ephemeral=True)
            logger.error(f"Error in manual_unban for user ID {uid}: {e}", exc_info=True)

    @app_commands.command(name="review", description="Review the oldest message in the moderation queue.")
    @app_commands.default_permissions(manage_messages=True)
    async def review_command(self, interaction: discord.Interaction):
        if not interaction.guild_id or not db_conn: return
        
        async with db_conn.execute("SELECT id, user_id, channel_id, message_id, message_content, reason, timestamp FROM review_queue WHERE guild_id = ? ORDER BY timestamp ASC LIMIT 1", (interaction.guild_id,)) as cursor:
            item = await cursor.fetchone()

        if not item:
            await interaction.response.send_message("The moderation review queue is empty!", ephemeral=True)
            return

        review_id, user_id, channel_id, message_id, content, reason, timestamp = item
        user = interaction.guild.get_member(user_id) 
        if not user: 
            try:
                user = await bot.fetch_user(user_id) 
            except discord.NotFound:
                user = None 
            except Exception as e:
                logger.error(f"Error fetching user {user_id} for review command: {e}")
                user = None

        channel = interaction.guild.get_channel(channel_id)
        message_url = f"https://discord.com/channels/{interaction.guild_id}/{channel_id}/{message_id}"

        embed = discord.Embed(title="Moderation Review Required", description=f"**Reason for Flag:** {reason}", color=discord.Color.orange(), timestamp=datetime.fromisoformat(timestamp))
        embed.add_field(name="Author", value=f"{user.mention if user else f'ID: {user_id}'}", inline=True)
        embed.add_field(name="Channel", value=f"{channel.mention if channel else f'ID: {channel_id}'}", inline=True)
        embed.add_field(name="Message", value=f"[Jump to Message]({message_url})", inline=False)
        embed.add_field(name="Content", value=f"```{discord.utils.escape_markdown(content[:1000])}```", inline=False)
        embed.set_footer(text=f"Review Item ID: {review_id}")

        view = ReviewActionView(review_id=review_id, member=user, message_id=message_id)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)


class AddRuleModal(discord.ui.Modal, title="Add New Violation Rule"):
    def __init__(self, review_id: int):
        super().__init__()
        self.review_id = review_id

    rule_type = discord.ui.Select(
        placeholder="Choose the type of rule to add...",
        options=[
            discord.SelectOption(label="Forbidden Word", value="forbidden_word", description="A single, case-insensitive word."),
            discord.SelectOption(label="Forbidden Phrase", value="forbidden_phrase", description="A case-insensitive sequence of words."),
            discord.SelectOption(label="Forbidden Regex", value="forbidden_regex", description="A regular expression pattern."),
        ]
    )
    pattern = discord.ui.TextInput(label="Pattern", style=discord.TextStyle.short, placeholder="Enter the word, phrase, or regex pattern.")

    async def on_submit(self, interaction: discord.Interaction):
        if not db_conn or not interaction.guild_id: return
        rule_type_val, pattern_val = self.rule_type.values[0], self.pattern.value.strip()

        try:
            async with db_conn.cursor() as cursor:
                await cursor.execute(
                    "INSERT INTO dynamic_rules (guild_id, rule_type, pattern, added_by_id, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (interaction.guild_id, rule_type_val, pattern_val, interaction.user.id, datetime.now(timezone.utc).isoformat())
                )
                await cursor.execute("DELETE FROM review_queue WHERE id = ?", (self.review_id,))
                await db_conn.commit()
            
            await load_dynamic_rules_from_db() 
            await interaction.response.send_message(f"✅ Rule added and review item `{self.review_id}` closed.", ephemeral=True)
            await log_moderation_action("dynamic_rule_added", interaction.user, f"Added `{rule_type_val}` rule: `{pattern_val}`", guild=interaction.guild, color=discord.Color.blue())
        except Exception as e:
            await interaction.response.send_message(f"An error occurred: {e}", ephemeral=True)
            logger.error(f"Error adding dynamic rule: {e}", exc_info=True)


class ReviewActionView(discord.ui.View):
    def __init__(self, review_id: int, member: discord.Member | None, message_id: int):
        super().__init__(timeout=300)
        self.review_id = review_id
        self.member = member
        self.message_id = message_id

    @discord.ui.button(label="Punish & Add Rule", style=discord.ButtonStyle.danger, emoji="�")
    async def punish_and_add_rule(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self.member:
            await interaction.response.send_message("Cannot punish user, they may have left the server. You can still add the rule.", ephemeral=True)
        modal = AddRuleModal(review_id=self.review_id)
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="Reject (Safe)", style=discord.ButtonStyle.success, emoji="✅")
    async def reject_as_safe(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not db_conn: return
        try:
            await db_conn.execute("DELETE FROM review_queue WHERE id = ?", (self.review_id,))
            await db_conn.commit()
            await interaction.response.edit_message(content=f"✅ Review item `{self.review_id}` marked as safe.", view=None)
            await log_moderation_action("review_rejected_safe", interaction.user, f"Review item `{self.review_id}` marked as safe.", guild=interaction.guild, color=discord.Color.green())
        except Exception as e:
            await interaction.followup.send(f"An error occurred: {e}", ephemeral=True)
            logger.error(f"Error rejecting review item: {e}", exc_info=True)


    @discord.ui.button(label="Ignore", style=discord.ButtonStyle.secondary, emoji="✖️")
    async def ignore_item(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="This review item was ignored.", view=None)
        await log_moderation_action("review_ignored", interaction.user, f"Review item `{self.review_id}` ignored.", guild=interaction.guild, color=discord.Color.light_grey())


# --- Bot Lifecycle Events & Web Server ---
@bot.event
async def setup_hook():
    """Asynchronous setup that runs before the bot logs in."""
    logger.info("Running setup_hook...")
    global http_session, LANGUAGE_MODEL, db_conn

    if not http_session or http_session.closed:
        http_session = ClientSession()
        logger.info("Aiohttp ClientSession initialized.")

    if not LANGUAGE_MODEL:
        try:
            if os.path.exists(FASTTEXT_MODEL_PATH):
                LANGUAGE_MODEL = fasttext.load_model(FASTTEXT_MODEL_PATH)
                logger.info(f"FastText model loaded from {FASTTEXT_MODEL_PATH}")
            else:
                logger.error(f"FastText model file not found at {FASTTEXT_MODEL_PATH}. Language detection will be disabled.")
        except Exception as e:
            logger.critical(f"Failed to load FastText model: {e}. Language detection will be disabled.", exc_info=True)

    if not db_conn:
        await setup_db()

    await load_dynamic_rules_from_db()

    await bot.add_cog(GeneralCog(bot))
    await bot.add_cog(ConfigurationCog(bot))
    await bot.add_cog(ModerationCog(bot))
    logger.info("All Cogs loaded.")

    try:
        synced_commands = await bot.tree.sync()
        logger.info(f"Synced {len(synced_commands)} slash commands.")
    except Exception as e:
        logger.error(f"Failed to sync slash commands: {e}", exc_info=True)

@bot.event
async def on_ready():
    """Event that fires when the bot is fully connected and ready."""
    logger.info(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    logger.info(f"Discord.py Version: {discord.__version__}")
    logger.info(f"{bot.user.name} is online and ready! 🚀")

async def health_check_handler(request):
    """Simple health check endpoint for the hosting service."""
    status_text = f"{bot.user.name} is running! Latency: {round(bot.latency * 1000)}ms. DB: {'OK' if db_conn else 'Error'}. LangModel: {'OK' if LANGUAGE_MODEL else 'Error'}."
    return web.Response(text=status_text, content_type="text/plain")

async def main_async_runner():
    """
    Handles the asynchronous running of both the bot and the web server.
    This resolves the "No open ports detected" error on hosting platforms like Render.
    """
    app = web.Application()
    app.router.add_get("/", health_check_handler)
    app.router.add_get("/health", health_check_handler)

    render_port = os.getenv("PORT")
    runner = None
    site = None

    if render_port:
        try:
            port = int(render_port)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', port)
            await site.start()
            logger.info(f"Health check web server started on port {port}.")
        except Exception as e:
            logger.error(f"Failed to start health check web server: {e}", exc_info=True)
    else:
        logger.info("PORT environment variable not set. Health check web server will not start.")

    try:
        await bot.start(DISCORD_TOKEN)
    finally:
        logger.info("Bot is shutting down.")
        if runner:
            await runner.cleanup()
            logger.info("Web server runner cleaned up.")
        await bot.close()


if __name__ == "__main__":
    try:
        asyncio.run(main_async_runner())
    except KeyboardInterrupt:
        logger.info("Shutdown requested via KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"💥 UNHANDLED EXCEPTION IN TOP LEVEL __main__: {e}", exc_info=True)
