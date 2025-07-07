import asyncio
import gettext
import json
import logging
import logging.handlers
import os
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus, urlparse

import aiohttp
import aiosqlite
import discord
import discord.ui
import fasttext
from googleapiclient import discovery
from aiohttp import ClientSession, client_exceptions
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
from tenacity import (retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential)
from thefuzz import fuzz

load_dotenv()

log_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger('adroit_perfected_v2')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

try:
    file_handler = logging.handlers.RotatingFileHandler(
        filename='adroit_perfected_v2.log',
        encoding='utf-8',
        maxBytes=5 * 1024 * 1024,
        backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except Exception as e:
    logger.error(f"Failed to set up file logging: {e}", exc_info=True)

DISCORD_TOKEN = os.getenv("ADROIT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY") 
FASTTEXT_MODEL_PATH = os.getenv("FASTTEXT_MODEL_PATH", "lid.176.ftz")
SIGHTENGINE_API_USER = os.getenv("SIGHTENGINE_API_USER")
SIGHTENGINE_API_SECRET = os.getenv("SIGHTENGINE_API_SECRET")

if not DISCORD_TOKEN:
    logger.critical("CRITICAL: ADROIT_TOKEN not set. Exiting.")
    exit(1)
if not OPENAI_API_KEY:
    logger.warning("Warning: OPENAI_API_KEY not set. OpenAI moderation will be disabled.")
if not PERSPECTIVE_API_KEY:
    logger.warning("Warning: PERSPECTIVE_API_KEY not set. Fallback moderation via Perspective API will be disabled.")
if not (SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET):
    logger.warning("Warning: Sightengine API keys not set. Image moderation will be disabled.")

intents = discord.Intents.default()
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix=">>", intents=intents, help_command=None)

db_conn: aiosqlite.Connection | None = None
LANGUAGE_MODEL: fasttext.FastText._FastText | None = None
http_session: ClientSession | None = None
perspective_client = None

def get_translator(lang_code: str = 'en'):
    """Gets the translator function for a given language code."""
    try:
        t = gettext.translation('adroit', localedir='locales', languages=[lang_code], fallback=True)
        return t.gettext
    except FileNotFoundError:
        return lambda s: s

class GlobalRateLimiter:
    """A simple in-memory global rate limiter."""
    def __init__(self, max_calls: int, period_seconds: int):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.call_timestamps = deque()

    def allow_call(self) -> bool:
        """Checks if a call is allowed under the rate limit."""
        now = time.monotonic()
        while self.call_timestamps and self.call_timestamps[0] <= now - self.period_seconds:
            self.call_timestamps.popleft()
        if len(self.call_timestamps) < self.max_calls:
            self.call_timestamps.append(now)
            return True
        logger.warning(f"Global rate limit hit ({self.max_calls}/{self.period_seconds}s). Call rejected.")
        return False

openai_rate_limiter = GlobalRateLimiter(max_calls=45, period_seconds=60)
perspective_rate_limiter = GlobalRateLimiter(max_calls=55, period_seconds=60) 

class BotConfig:
    """Houses all static configuration for the bot's moderation logic."""
    def __init__(self):
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
                1: {"action": "warn", "reason_suffix": "Minor guideline violations."},
                20: {"action": "mute", "duration_hours": 1, "reason_suffix": "Accumulated violations."},
                35: {"action": "mute", "duration_hours": 6, "reason_suffix": "Significant or repeated violations."},
                50: {"action": "kick", "reason_suffix": "Persistent serious violations."},
                75: {"action": "temp_ban", "duration_days": 3, "reason_suffix": "Severe or multiple major violations."},
                150: {"action": "temp_ban", "duration_days": 30, "reason_suffix": "Extreme or highly disruptive behavior."},
                300: {"action": "ban", "reason_suffix": "Egregious violations or repeat offenses."}
            },
            "violations": {
                "discrimination": {"points": 15, "severity": "High"}, "spam_rate": {"points": 1, "severity": "Low"},
                "spam_repetition": {"points": 2, "severity": "Low"}, "nsfw_text": {"points": 5, "severity": "Medium"},
                "nsfw_media": {"points": 10, "severity": "High"}, "advertising_forbidden": {"points": 3, "severity": "Medium"},
                "advertising_unpermitted_url": {"points": 3, "severity": "Medium"}, "politics_disallowed": {"points": 1, "severity": "Low"},
                "off_topic": {"points": 1, "severity": "Low"}, "foreign_language": {"points": 1, "severity": "Low"},
                "excessive_mentions": {"points": 1, "severity": "Low"}, "excessive_attachments": {"points": 1, "severity": "Low"},
                "long_message": {"points": 1, "severity": "Low"}, "gore_violence_media": {"points": 15, "severity": "High"},
                "offensive_symbols_media": {"points": 5, "severity": "Medium"},
                "dynamic_rule_violation_local": {"points": 3, "severity": "Medium"}, 
                "dynamic_rule_violation_global": {"points": 8, "severity": "High"}, 
                "manual_review_punishment": {"points": 2, "severity": "Medium"},
                "ai_hate": {"points": 15, "severity": "High"}, "ai_sexual": {"points": 10, "severity": "High"},
                "ai_violence": {"points": 10, "severity": "High"}, "ai_harassment": {"points": 5, "severity": "Medium"},
                "ai_self_harm": {"points": 15, "severity": "High"}, "ai_toxicity": {"points": 4, "severity": "Medium"},
            }
        }
        self.spam_window_seconds = 8
        self.spam_message_limit = 5
        self.spam_repetition_history_count = 3
        self.spam_repetition_fuzzy_threshold = 90
        self.mention_limit = 5
        self.max_message_length = 1500
        self.max_attachments = 5
        self.min_word_count_for_lang_check = 4
        self.min_confidence_for_lang_flagging = 0.65
        self.min_confidence_short_msg_lang = 0.75
        self.short_msg_threshold_lang = 25
        self.common_safe_foreign_words = {"bonjour", "hola", "merci", "gracias", "oui", "si", "nyet", "da", "salut", "ciao", "hallo", "guten tag", "privet", "konnichiwa", "arigato", "sawasdee", "namaste", "salaam"}
        self.fuzzy_match_threshold_keywords = 85
        self.sightengine_nudity_sexual_activity_threshold = 0.5
        self.sightengine_nudity_suggestive_threshold = 0.55
        self.sightengine_gore_threshold = 0.6
        self.sightengine_offensive_symbols_threshold = 0.5
        self.proactive_flagging_openai_threshold = 0.5
        self.proactive_flagging_perspective_threshold = 0.75 
        self.delete_violating_messages = True
        self.send_in_channel_warning = True
        self.in_channel_warning_delete_delay = 30

bot_config = BotConfig()
dynamic_rules_cache = {"global": {"forbidden_words": set(), "forbidden_phrases": [], "forbidden_regex": []}}

def load_terms_from_file(filepath: str) -> tuple[set[str], list[str]]:
    """Loads terms from a file into a set of words and a list of phrases."""
    words, phrases = set(), []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                term = line.strip().lower()
                if not term or term.startswith("#"): continue
                term = normalize_leetspeak(term)
                if ' ' in term: phrases.append(term)
                else: words.add(term)
    except FileNotFoundError:
        logger.warning(f"Terms file '{filepath}' not found.")
    except Exception as e:
        logger.error(f"Error loading terms from {filepath}: {e}", exc_info=True)
    return words, phrases

def normalize_leetspeak(text: str) -> str:
    """Normalizes common leetspeak substitutions in a string."""
    substitutions = {'@': 'a', '4': 'a', '8': 'b', '(': 'c', '3': 'e', 'ph': 'f', '6': 'g', '1': 'i', '!': 'i', '0': 'o', '$': 's', '5': 's', '+': 't', '7': 't'}
    for char, replacement in substitutions.items():
        text = text.replace(char, replacement)
    return text

discrimination_words_set, discrimination_phrases = load_terms_from_file('discrimination_terms.txt')
nsfw_text_words_set, nsfw_text_phrases = load_terms_from_file('nsfw_terms.txt')

def clean_message_for_language_detection(text: str) -> str:
    """Cleans a message to improve language detection accuracy."""
    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<@!?\d+>|<#\d+>|<@&\d+>', '', text)
    text = re.sub(r'<a?:\w+:\d+>', '', text)
    text = re.sub(r'\|\|.*?\|\|', '', text)
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)
    text = re.sub(r'(\*|_|~|>|#|\-|\+)', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def clean_message_content(text: str, normalize: bool = True) -> str:
    """Cleans message content for matching, optionally normalizing leetspeak."""
    cleaned_text = re.sub(r'\s+', ' ', text).strip().lower()
    if normalize:
        cleaned_text = normalize_leetspeak(cleaned_text)
    return cleaned_text

async def get_guild_config(guild_id: int, key: str, default_value=None):
    """Retrieves a configuration value for a guild from the database."""
    if not db_conn: return default_value
    try:
        async with db_conn.execute("SELECT config_value FROM guild_configs WHERE guild_id = ? AND config_key = ?", (guild_id, key)) as cursor:
            result = await cursor.fetchone()
            if result and result[0] is not None:
                try: return json.loads(result[0])
                except (json.JSONDecodeError, TypeError): return result[0]
            return default_value
    except Exception as e:
        logger.error(f"Error getting guild config for guild {guild_id}, key '{key}': {e}", exc_info=True)
        return default_value

async def set_guild_config(guild_id: int, key: str, value_to_set):
    """Sets a configuration value for a guild in the database."""
    if not db_conn: return
    stored_value = json.dumps(value_to_set) if isinstance(value_to_set, (list, dict)) else str(value_to_set) if value_to_set is not None else None
    try:
        async with db_conn.cursor() as cursor:
            if stored_value is None:
                await cursor.execute('DELETE FROM guild_configs WHERE guild_id = ? AND config_key = ?', (guild_id, key))
            else:
                await cursor.execute('INSERT OR REPLACE INTO guild_configs (guild_id, config_key, config_value) VALUES (?, ?, ?)', (guild_id, key, stored_value))
        await db_conn.commit()
    except Exception as e:
        logger.error(f"Error setting guild config for guild {guild_id}, key '{key}': {e}", exc_info=True)

async def update_user_profile(user_id: int, guild_id: int, points_to_add: int):
    """Updates a user's profile with new points and last infraction timestamp."""
    if not db_conn: return
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        await db_conn.execute(
            """
            INSERT INTO user_profiles (user_id, guild_id, total_points, last_infraction_timestamp)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, guild_id) DO UPDATE SET
            total_points = total_points + excluded.total_points,
            last_infraction_timestamp = excluded.last_infraction_timestamp
            """,
            (user_id, guild_id, points_to_add, now_iso)
        )
        await db_conn.commit()
    except Exception as e:
        logger.error(f"Error updating user profile for {user_id} in guild {guild_id}: {e}", exc_info=True)

async def detect_language_ai(text: str) -> tuple[str | None, float]:
    """Detects the language of a given text using the loaded FastText model."""
    if not LANGUAGE_MODEL: return None, 0.0
    clean_text = clean_message_for_language_detection(text)
    if not clean_text or not bot_config.has_alphanumeric_pattern.search(clean_text): return None, 0.0
    try:
        prediction = LANGUAGE_MODEL.predict(clean_text.replace("\n", " "), k=1)
        if prediction and prediction[0] and prediction[1]:
            lang_code = prediction[0][0].replace("__label__", "")
            confidence = float(prediction[1][0])
            return lang_code, confidence
        return None, 0.0
    except Exception as e:
        logger.error(f"FastText language detection error: {e}", exc_info=True)
        return None, 0.0

async def log_moderation_action(guild: discord.Guild, embed: discord.Embed, _=get_translator()):
    """Sends a moderation action log to the configured log channel."""
    log_channel_id = await get_guild_config(guild.id, "log_channel_id")
    if not log_channel_id: return
    log_channel = bot.get_channel(int(log_channel_id))
    if not isinstance(log_channel, discord.TextChannel): return

    try:
        await log_channel.send(embed=embed)
    except discord.Forbidden:
        logger.error(f"Missing permissions to send logs to channel #{log_channel.name} in guild {guild.name}.")
    except Exception as e:
        logger.error(f"Error sending log embed to channel {log_channel.id}: {e}", exc_info=True)

def retry_if_api_error(exception):
    """Retry condition for Tenacity: retries on 429 or 5xx errors."""
    if isinstance(exception, client_exceptions.ClientResponseError):
        return exception.status == 429 or exception.status >= 500
    return isinstance(exception, (asyncio.TimeoutError, client_exceptions.ClientOSError))

@retry(stop=stop_after_attempt(4), wait=wait_random_exponential(multiplier=1, max=30), retry=retry_if_api_error, reraise=True)
async def check_openai_moderation_api(text_content: str) -> dict:
    """Checks text content against the OpenAI Moderation API."""
    if not OPENAI_API_KEY or not text_content.strip() or not http_session or http_session.closed:
        return {}
    if not openai_rate_limiter.allow_call():
        return {}

    url = "https://api.openai.com/v1/moderations"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"input": text_content.replace("\n", " ")}

    try:
        async with http_session.post(url, headers=headers, json=data, timeout=15) as response:
            response.raise_for_status()
            json_response = await response.json()
            return json_response.get("results", [{}])[0]
    except client_exceptions.ClientResponseError as e:
        if e.status == 400: 
            logger.warning(f"OpenAI returned 400 Bad Request. Content: '{text_content[:100]}...'")
            return {}
        raise
    except Exception as e:
        logger.error(f"Unexpected error with OpenAI moderation: {e}", exc_info=True)
        raise 

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1.5, max=25), retry=retry_if_api_error, reraise=True)
async def check_perspective_api(text_content: str) -> dict:
    """Fallback moderation check using Google's Perspective API."""
    if not perspective_client or not text_content.strip():
        return {}
    if not perspective_rate_limiter.allow_call():
        return {}

    analyze_request = {
        'comment': {'text': text_content},
        'requestedAttributes': {
            'TOXICITY': {},
            'SEVERE_TOXICITY': {},
            'IDENTITY_ATTACK': {},
            'INSULT': {},
            'PROFANITY': {},
            'THREAT': {},
            'SEXUALLY_EXPLICIT': {}
        },
        'spanAnnotations': False,
        'languages': ['en'] 
    }
    try:
        response = await bot.loop.run_in_executor(
            None,
            lambda: perspective_client.comments().analyze(body=analyze_request).execute()
        )
        return response.get("attributeScores", {})
    except Exception as e:
        logger.error(f"Perspective API error: {e}", exc_info=True)
        return {} 

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=20), retry=retry_if_api_error, reraise=True)
async def check_sightengine_media_api(image_url: str) -> dict:
    """Checks an image URL against the Sightengine API for unsafe content."""
    if not (SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET and http_session and not http_session.closed):
        return {}
    models = "nudity-2.0,gore,offensive"
    api_url = f"https://api.sightengine.com/1.0/check.json?url={quote_plus(image_url)}&models={models}&api_user={SIGHTENGINE_API_USER}&api_secret={SIGHTENGINE_API_SECRET}"
    try:
        async with http_session.get(api_url, timeout=20) as response:
            response.raise_for_status()
            json_response = await response.json()
            return json_response if json_response.get("status") == "success" else {}
    except Exception as e:
        logger.error(f"Unexpected error with Sightengine API for URL {image_url}: {e}", exc_info=True)
        return {}

async def apply_moderation_punishment(member: discord.Member, action_config: dict, total_points: int, violation_summary: str, moderator: discord.User | None = None, original_content: str | None = None, _=get_translator()):
    """Applies a configured punishment to a member."""
    action_type = action_config["action"]
    guild = member.guild
    reason_suffix = action_config.get("reason_suffix", _("Automated action due to accumulated points."))

    moderator_str = _("Automated")
    if moderator:
        moderator_str = _("Manual by {moderator_name}").format(moderator_name=moderator.name)
        full_reason = f"{moderator_str}: {reason_suffix}"
    else:
        full_reason = f"{moderator_str} | Points: {total_points} | {reason_suffix} | Violations: {violation_summary}"

    dm_embed = discord.Embed(title=_("Action Taken in {guild_name}").format(guild_name=guild.name), color=discord.Color.orange())
    dm_embed.description = _("**Action:** {action}\n**Reason:** {reason}").format(action=action_type.replace('_', ' ').title(), reason=reason_suffix)
    if original_content:
        dm_embed.add_field(name=_("Original Content Snippet"), value=f"```\n{original_content[:200]}\n```", inline=False)
    
    duration = None
    if "duration_hours" in action_config:
        duration = timedelta(hours=action_config["duration_hours"])
    elif "duration_days" in action_config:
        duration = timedelta(days=action_config["duration_days"])
    
    if duration:
        dm_embed.add_field(name=_("Duration"), value=str(duration))

    if not guild.me or guild.me.top_role <= member.top_role:
        logger.warning(f"Cannot {action_type} {member.display_name} in {guild.name}: Check bot role hierarchy.")
        return

    try:
        await member.send(embed=dm_embed)
    except (discord.Forbidden, discord.HTTPException):
        logger.info(f"Could not DM user {member.id}, proceeding with action.")

    log_color = discord.Color.orange()
    extra_log_fields = []
    try:
        if action_type == "warn":
            log_color = discord.Color.gold()
        elif action_type == "mute" and duration:
            await member.timeout(duration, reason=full_reason)
            log_color = discord.Color.light_grey()
        elif action_type == "kick":
            await member.kick(reason=full_reason)
            log_color = discord.Color.red()
        elif action_type == "temp_ban" and duration and db_conn:
            unban_time = datetime.now(timezone.utc) + duration
            await db_conn.execute('INSERT OR REPLACE INTO temp_bans (user_id, guild_id, unban_time, ban_reason) VALUES (?, ?, ?, ?)', (member.id, guild.id, unban_time.isoformat(), full_reason))
            await db_conn.commit()
            await guild.ban(member, reason=full_reason, delete_message_days=0)
            log_color = discord.Color.dark_red()
            extra_log_fields.append((_("Unban Time"), f"{discord.utils.format_dt(unban_time, 'R')}"))
        elif action_type == "ban":
            await guild.ban(member, reason=full_reason, delete_message_days=0)
            log_color = discord.Color.dark_red()

        log_embed = discord.Embed(
            title=_("🛡️ Moderation: {action}").format(action=action_type.replace('_', ' ').title()),
            description=full_reason,
            color=log_color,
            timestamp=datetime.now(timezone.utc)
        )
        log_embed.set_author(name=f"{member} ({member.id})", icon_url=member.display_avatar.url)
        log_embed.add_field(name=_("Moderator"), value=moderator.mention if moderator else _("Automated"))
        for name, value in extra_log_fields:
            log_embed.add_field(name=name, value=value)
        await log_moderation_action(guild, log_embed, _=_)

    except discord.Forbidden:
        logger.error(f"Lacks permissions for '{action_type}' on {member.display_name} in {guild.name}.")
    except Exception as e:
        logger.error(f"Error applying {action_type} to {member.id}: {e}", exc_info=True)

async def process_infractions_and_punish(member: discord.Member, guild: discord.Guild, violation_types: list[str], content: str, message_url: str | None, _=get_translator()):
    """Processes new infractions, updates user points, and applies punishment if a threshold is met."""
    if not db_conn: return
    user_id, guild_id = member.id, guild.id
    points_this_action = 0
    summary_parts = []

    for v_type in violation_types:
        v_config = bot_config.punishment_system["violations"].get(v_type)
        if not v_config: continue
        
        points_this_action += v_config["points"]
        summary_parts.append(v_type.replace('_', ' ').title())

        await db_conn.execute(
            "INSERT INTO infractions (user_id, guild_id, violation_type, points, message_content_snippet, message_url, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, guild_id, v_type, v_config["points"], content[:500], message_url, datetime.now(timezone.utc).isoformat())
        )

    if not summary_parts:
        return await db_conn.commit()

    await update_user_profile(user_id, guild_id, points_this_action)

    ninety_days_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    async with db_conn.execute("SELECT SUM(points) FROM infractions WHERE user_id = ? AND guild_id = ? AND timestamp >= ?", (user_id, guild_id, ninety_days_ago)) as cursor:
        total_points_result = await cursor.fetchone()
    total_active_points = total_points_result[0] if total_points_result and total_points_result[0] else 0
    await db_conn.commit()

    applicable_punishment_config = None
    for threshold in sorted(bot_config.punishment_system["points_thresholds"].keys(), reverse=True):
        if total_active_points >= threshold:
            async with db_conn.execute("SELECT last_punishment_threshold FROM user_profiles WHERE user_id = ? AND guild_id = ?", (user_id, guild_id)) as cursor:
                profile = await cursor.fetchone()
            last_threshold = profile[0] if profile and profile[0] else 0
            
            if threshold > last_threshold:
                applicable_punishment_config = bot_config.punishment_system["points_thresholds"][threshold]
                await db_conn.execute("UPDATE user_profiles SET last_punishment_threshold = ? WHERE user_id = ? AND guild_id = ?", (threshold, user_id, guild_id))
                await db_conn.commit()
                break
    
    if applicable_punishment_config:
        await apply_moderation_punishment(member, applicable_punishment_config, total_active_points, ", ".join(summary_parts), original_content=content, _=_)
        if applicable_punishment_config["action"] == "ban":
            await clear_user_infractions(user_id, guild_id)

async def get_user_infractions_from_db(user_id: int, guild_id: int, days_limit: int = 90) -> tuple[list[dict], int]:
    """Retrieves a user's recent infractions and total active points."""
    if not db_conn: return [], 0
    try:
        time_limit = (datetime.now(timezone.utc) - timedelta(days=days_limit)).isoformat()
        async with db_conn.execute("SELECT id, violation_type, points, timestamp FROM infractions WHERE user_id = ? AND guild_id = ? AND timestamp >= ? ORDER BY timestamp DESC", (user_id, guild_id, time_limit)) as cursor:
            infractions = [{"id": r[0], "violation_type": r[1], "points": r[2], "timestamp": r[3]} for r in await cursor.fetchall()]
        
        ninety_days_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        async with db_conn.execute("SELECT SUM(points) FROM infractions WHERE user_id = ? AND guild_id = ? AND timestamp >= ?", (user_id, guild_id, ninety_days_ago)) as cursor:
            result = await cursor.fetchone()
        return infractions, result[0] if result and result[0] is not None else 0
    except Exception as e:
        logger.error(f"Error retrieving infractions for user {user_id}: {e}", exc_info=True)
        return [], 0

async def clear_user_infractions(user_id: int, guild_id: int):
    """Clears all infractions and resets the profile for a user in a guild."""
    if not db_conn: return
    try:
        await db_conn.execute("DELETE FROM infractions WHERE user_id = ? AND guild_id = ?", (user_id, guild_id))
        await db_conn.execute("UPDATE user_profiles SET total_points = 0, last_punishment_threshold = 0 WHERE user_id = ? AND guild_id = ?", (user_id, guild_id))
        await db_conn.commit()
    except Exception as e:
        logger.error(f"Error clearing infractions for user {user_id}: {e}", exc_info=True)

async def remove_specific_infraction_from_db(infraction_id: int) -> bool:
    """Removes a single infraction by its ID and adjusts user profile points."""
    if not db_conn: return False
    try:
        async with db_conn.execute("SELECT user_id, guild_id, points FROM infractions WHERE id = ?", (infraction_id,)) as cursor:
            infraction_data = await cursor.fetchone()
        
        if not infraction_data:
            return False 
            
        user_id, guild_id, points_to_remove = infraction_data

        cursor = await db_conn.execute("DELETE FROM infractions WHERE id = ?", (infraction_id,))
        if cursor.rowcount > 0:
            await db_conn.execute("UPDATE user_profiles SET total_points = total_points - ? WHERE user_id = ? AND guild_id = ?", (points_to_remove, user_id, guild_id))
            await db_conn.commit()
            return True
        return False
    except Exception as e:
        logger.error(f"Error removing infraction {infraction_id}: {e}", exc_info=True)
        return False

async def setup_database():
    """Initializes the database connection and creates tables if they don't exist."""
    global db_conn
    try:
        db_conn = await aiosqlite.connect('adroit_perfected_v2_data.db')
        await db_conn.execute("PRAGMA journal_mode=WAL;")
        await db_conn.execute("PRAGMA foreign_keys=ON;")
        await db_conn.execute("CREATE TABLE IF NOT EXISTS guild_configs (guild_id INTEGER NOT NULL, config_key TEXT NOT NULL, config_value TEXT, PRIMARY KEY (guild_id, config_key))")
        await db_conn.execute("CREATE TABLE IF NOT EXISTS infractions (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, guild_id INTEGER NOT NULL, violation_type TEXT NOT NULL, points INTEGER NOT NULL, message_content_snippet TEXT, message_url TEXT, timestamp TEXT NOT NULL)")
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_infractions_user_guild_time ON infractions (user_id, guild_id, timestamp)')
        await db_conn.execute("CREATE TABLE IF NOT EXISTS temp_bans (user_id INTEGER NOT NULL, guild_id INTEGER NOT NULL, unban_time TEXT NOT NULL, ban_reason TEXT, PRIMARY KEY (user_id, guild_id))")
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_temp_bans_unban_time ON temp_bans (unban_time)')
        await db_conn.execute("CREATE TABLE IF NOT EXISTS dynamic_rules (id INTEGER PRIMARY KEY, guild_id INTEGER, rule_type TEXT NOT NULL, pattern TEXT NOT NULL, added_by_id INTEGER NOT NULL, timestamp TEXT NOT NULL, UNIQUE (guild_id, rule_type, pattern))") # guild_id can be NULL for global rules
        await db_conn.execute("CREATE TABLE IF NOT EXISTS review_queue (id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL, user_id INTEGER NOT NULL, channel_id INTEGER, message_id INTEGER UNIQUE, message_content TEXT NOT NULL, reason TEXT NOT NULL, timestamp TEXT NOT NULL)")
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_review_queue_guild_timestamp ON review_queue (guild_id, timestamp)')
        await db_conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL,
                total_points INTEGER DEFAULT 0,
                last_infraction_timestamp TEXT,
                last_punishment_threshold INTEGER DEFAULT 0,
                role TEXT DEFAULT 'member',
                PRIMARY KEY (user_id, guild_id)
            )
        """)
        await db_conn.commit()
        logger.info("Database initialized successfully with all tables, including user_profiles.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize database: {e}", exc_info=True)
        if db_conn: await db_conn.close()
        exit(1)

async def load_dynamic_rules_from_db():
    """Loads all global and guild-specific rules from the database into an in-memory cache."""
    if not db_conn: return
    dynamic_rules_cache.clear()
    dynamic_rules_cache["global"] = {"forbidden_words": set(), "forbidden_phrases": [], "forbidden_regex": []}

    try:
        async with db_conn.execute("SELECT guild_id, rule_type, pattern FROM dynamic_rules") as cursor:
            async for guild_id, rule_type, pattern in cursor:
                key = "global" if guild_id is None else guild_id
                
                if key not in dynamic_rules_cache:
                    dynamic_rules_cache[key] = {"forbidden_words": set(), "forbidden_phrases": [], "forbidden_regex": []}

                if rule_type == 'forbidden_word':
                    dynamic_rules_cache[key]["forbidden_words"].add(pattern.lower())
                elif rule_type == 'forbidden_phrase':
                    dynamic_rules_cache[key]["forbidden_phrases"].append(pattern.lower())
                elif rule_type == 'forbidden_regex':
                    try:
                        dynamic_rules_cache[key]["forbidden_regex"].append(re.compile(pattern, re.IGNORECASE))
                    except re.error as e:
                        logger.error(f"Failed to compile regex '{pattern}' for guild {key}: {e}")
        logger.info(f"Loaded {sum(len(ruleset['forbidden_words']) + len(ruleset['forbidden_phrases']) + len(ruleset['forbidden_regex']) for ruleset in dynamic_rules_cache.values())} dynamic rules across all scopes.")
    except Exception as e:
        logger.error(f"Error loading dynamic rules into cache: {e}", exc_info=True)

class GeneralCog(commands.Cog, name="General"):
    """General purpose commands like ping and help."""
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="ping", description="Checks the bot's latency.")
    async def ping(self, interaction: discord.Interaction):
        _ = get_translator(interaction.locale.language if interaction.locale else 'en')
        latency = round(self.bot.latency * 1000)
        await interaction.response.send_message(_("Pong! Latency: {latency}ms").format(latency=latency), ephemeral=True)

    @app_commands.command(name="help", description="Shows information about bot commands.")
    async def help_command(self, interaction: discord.Interaction):
        _ = get_translator(interaction.locale.language if interaction.locale else 'en')
        embed = discord.Embed(
            title=_("Adroit Bot Help"),
            description=_("Below is a list of available command categories. Use the commands to configure and manage moderation in your server."),
            color=discord.Color.blue()
        )
        for cog_name, cog in self.bot.cogs.items():
            commands_list = []
            for cmd in cog.get_app_commands():
                if isinstance(cmd, app_commands.Group):
                    commands_list.extend(f"`/{cmd.name} {sub.name}` - {sub.description}" for sub in cmd.commands)
                else:
                    commands_list.append(f"`/{cmd.name}` - {cmd.description}")
            if commands_list:
                embed.add_field(name=_("**{cog_name} Commands**").format(cog_name=cog.qualified_name), value="\n".join(commands_list), inline=False)
        
        embed.set_footer(text=_("Adroit Bot | Enhanced Moderation"))
        await interaction.response.send_message(embed=embed, ephemeral=True)


class ConfigurationCog(commands.Cog, name="Configuration"):
    """Commands for configuring the bot's behavior in a guild."""
    def __init__(self, bot: commands.Bot):
        self.bot = bot
    
    group = app_commands.Group(name="config", description="Configure bot settings.", default_permissions=discord.Permissions(manage_guild=True))

    @group.command(name="set_log_channel", description="Sets the channel for moderation logs. (Leave empty to clear)")
    async def set_log_channel(self, interaction: discord.Interaction, channel: discord.TextChannel | None = None):
        _ = get_translator(interaction.locale.language if interaction.locale else 'en')
        if not interaction.guild_id: return
        await set_guild_config(interaction.guild_id, "log_channel_id", channel.id if channel else None)
        if channel:
            await interaction.response.send_message(_("Moderation logs will now be sent to {channel_mention}.").format(channel_mention=channel.mention), ephemeral=True)
        else:
            await interaction.response.send_message(_("Moderation log channel has been cleared."), ephemeral=True)

    @group.command(name="set_review_channel", description="Sets channel for messages needing review. (Leave empty to clear)")
    async def set_review_channel(self, interaction: discord.Interaction, channel: discord.TextChannel | None = None):
        _ = get_translator(interaction.locale.language if interaction.locale else 'en')
        if not interaction.guild_id: return
        await set_guild_config(interaction.guild_id, "review_channel_id", channel.id if channel else None)
        if channel:
            await interaction.response.send_message(_("Messages for review will be sent to {channel_mention}.").format(channel_mention=channel.mention), ephemeral=True)
        else:
            await interaction.response.send_message(_("Review channel has been cleared."), ephemeral=True)

    async def _set_language_config(self, interaction: discord.Interaction, languages_str: str, config_key: str, entity_mention: str):
        _ = get_translator(interaction.locale.language if interaction.locale else 'en')
        if not interaction.guild_id: return
        
        lang_list_raw = [lang.strip().lower() for lang in languages_str.split(',')]
        is_any = "any" in lang_list_raw
        is_default = "default" in lang_list_raw

        if is_any:
            await set_guild_config(interaction.guild_id, config_key, ["any"])
            msg = _("Language check for {entity} now allows **any** language.").format(entity=entity_mention)
        elif is_default:
            await set_guild_config(interaction.guild_id, config_key, None)
            msg = _("Language override for {entity} removed. Now uses server default.").format(entity=entity_mention)
        else:
            valid_langs = [lang for lang in lang_list_raw if re.fullmatch(r"[a-z]{2,3}", lang)]
            if len(valid_langs) != len(lang_list_raw):
                await interaction.response.send_message(_("Invalid language code format provided. Use 2-3 letter ISO codes (e.g., en, fr, es)."), ephemeral=True)
                return
            await set_guild_config(interaction.guild_id, config_key, valid_langs)
            msg = _("Languages for {entity} set to: **{langs}**.").format(entity=entity_mention, langs=', '.join(valid_langs).upper())
        
        await interaction.response.send_message(msg, ephemeral=True)

    @group.command(name="set_server_language", description="Sets default server language(s). Use 'any' to disable, 'en' for English.")
    @app_commands.describe(languages="Comma-separated ISO codes (e.g., en,fr). Use 'any' to allow all.")
    async def set_server_language(self, interaction: discord.Interaction, languages: str):
        if not interaction.guild: return
        await self._set_language_config(interaction, languages, "default_language", f"server **{interaction.guild.name}**")

    @group.command(name="set_channel_language", description="Overrides server language for a channel. Use 'any' or 'default'.")
    @app_commands.describe(channel="The channel to configure.", languages="Comma-separated ISO codes, 'any', or 'default'.")
    async def set_channel_language(self, interaction: discord.Interaction, channel: discord.TextChannel, languages: str):
        await self._set_language_config(interaction, languages, f"channel_language_{channel.id}", channel.mention)


class ModerationCog(commands.Cog, name="Moderation"):
    """The core moderation engine and related commands."""
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.user_message_timestamps: defaultdict[int, defaultdict[int, deque]] = defaultdict(lambda: defaultdict(deque))
        self.user_message_history: defaultdict[int, defaultdict[int, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=bot_config.spam_repetition_history_count)))

        self.report_message_context_menu = app_commands.ContextMenu(
            name="Report to Adroit",
            callback=self.report_message_from_context
        )
        self.bot.tree.add_command(self.report_message_context_menu)

        self.temp_ban_check_task.start()
        self.cleanup_old_infractions_task.start()
        self.cleanup_spam_trackers_task.start()

    def cog_unload(self):
        """Gracefully stop tasks and remove context menus on cog unload."""
        self.temp_ban_check_task.cancel()
        self.cleanup_old_infractions_task.cancel()
        self.cleanup_spam_trackers_task.cancel()
        self.bot.tree.remove_command(self.report_message_context_menu.name, type=self.report_message_context_menu.type)

    async def report_message_from_context(self, interaction: discord.Interaction, message: discord.Message):
        """Callback for the 'Report to Adroit' context menu command."""
        _ = get_translator(interaction.locale.language if interaction.locale else 'en')
        if message.author.id == interaction.user.id or message.author.bot:
            await interaction.response.send_message(_("You cannot report your own messages or messages from bots."), ephemeral=True)
            return
        await interaction.response.send_modal(ReportMessageModal(message=message, translator=_))

    async def get_effective_channel_language_config(self, guild_id: int, channel_id: int) -> list[str] | None:
        """Determines the effective language setting for a channel, falling back to server default."""
        channel_setting = await get_guild_config(guild_id, f"channel_language_{channel_id}")
        if channel_setting is not None:
            return channel_setting
        return await get_guild_config(guild_id, "default_language", ["en"]) 

    @tasks.loop(minutes=1)
    async def temp_ban_check_task(self):
        """Periodically checks for and lifts expired temporary bans."""
        now_utc = datetime.now(timezone.utc)
        try:
            async with db_conn.execute("SELECT user_id, guild_id, unban_time, ban_reason FROM temp_bans WHERE unban_time <= ?", (now_utc.isoformat(),)) as cursor:
                expired_bans = await cursor.fetchall()
            
            for user_id, guild_id, unban_time_iso, ban_reason in expired_bans:
                guild = self.bot.get_guild(guild_id)
                if not guild:
                    await self._remove_temp_ban_from_db(user_id, guild_id)
                    continue
                
                _ = get_translator(guild.preferred_locale.language if guild.preferred_locale else 'en')
                try:
                    user_obj = discord.Object(id=user_id)
                    await guild.unban(user_obj, reason=_("Temporary ban expired. Original reason: {reason}").format(reason=ban_reason))
                    embed = discord.Embed(title=_("✅ Temp Ban Expired"), description=_("User ID `{user_id}` has been unbanned.").format(user_id=user_id), color=discord.Color.green())
                    await log_moderation_action(guild, embed, _=_)
                except discord.NotFound:
                    pass
                except discord.Forbidden:
                    logger.error(f"Missing permissions to unban user {user_id} from guild {guild.name}.")
                except Exception as e:
                    logger.error(f"Error unbanning user {user_id} from guild {guild.id}: {e}", exc_info=True)
                finally:
                    await self._remove_temp_ban_from_db(user_id, guild_id)
        except Exception as e:
            logger.error(f"Error in temp_ban_check_task: {e}", exc_info=True)

    async def _remove_temp_ban_from_db(self, user_id: int, guild_id: int):
        """Helper to remove a temp ban entry from the database."""
        if not db_conn: return
        try:
            await db_conn.execute("DELETE FROM temp_bans WHERE user_id = ? AND guild_id = ?", (user_id, guild_id))
            await db_conn.commit()
        except Exception as e:
            logger.error(f"Error removing temp ban for user {user_id} in guild {guild_id}: {e}", exc_info=True)

    @tasks.loop(hours=24)
    async def cleanup_old_infractions_task(self):
        """Periodically deletes very old infraction records to save space."""
        if not db_conn: return
        one_hundred_eighty_days_ago = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
        try:
            cursor = await db_conn.execute("DELETE FROM infractions WHERE timestamp < ?", (one_hundred_eighty_days_ago,))
            await db_conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Cleaned up {cursor.rowcount} infractions older than 180 days.")
        except Exception as e:
            logger.error(f"Error during old infraction cleanup task: {e}", exc_info=True)
    
    @tasks.loop(hours=6)
    async def cleanup_spam_trackers_task(self):
        """Periodically cleans up the in-memory spam trackers to prevent memory leaks."""
        now_ts = datetime.now(timezone.utc).timestamp()
        cleanup_threshold = bot_config.spam_window_seconds * 10 
        
        for guild_id in list(self.user_message_timestamps):
            for user_id in list(self.user_message_timestamps[guild_id]):
                timestamps = self.user_message_timestamps[guild_id][user_id]
                while timestamps and (now_ts - timestamps[0] > cleanup_threshold):
                    timestamps.popleft()
                if not timestamps:
                    del self.user_message_timestamps[guild_id][user_id]
                    if user_id in self.user_message_history[guild_id]:
                        del self.user_message_history[guild_id][user_id]
            if not self.user_message_timestamps[guild_id]:
                del self.user_message_timestamps[guild_id]
                if guild_id in self.user_message_history:
                    del self.user_message_history[guild_id]
        logger.info("Finished cleaning up spam trackers.")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """The main entry point for message moderation."""
        if not message.guild or message.author.bot or not isinstance(message.author, discord.Member):
            return

        violations, proactive_flag_reason = await self._run_moderation_pipeline(message)
        
        if violations:
            await self._handle_violations(message, violations)
        elif proactive_flag_reason:
            await self.add_to_review_queue(message, proactive_flag_reason)

        await self.bot.process_commands(message)
        
    async def _run_moderation_pipeline(self, message: discord.Message) -> tuple[set[str], str | None]:
        """Executes all moderation checks on a message."""
        violations, proactive_flag_reason = set(), None
        content_raw = message.content
        cleaned_content = clean_message_content(content_raw)

        local_violations = (
            self.check_dynamic_rules(message.guild.id, content_raw, cleaned_content)
            | self.check_spam(message.author.id, message.guild.id, cleaned_content)
            | self.check_message_limits(message)
            | self.check_keyword_violations(cleaned_content)
        )

        if local_violations:
            return local_violations, None

        tasks = [
            self.check_advertising(content_raw, message.guild.id),
            self.check_language(message, content_raw),
            self.check_ai_text_moderation(content_raw, message.guild.id)
        ]
        if SIGHTENGINE_API_USER and message.attachments:
            tasks.append(self.check_ai_media_moderation(message.attachments))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"A moderation check failed in the pipeline: {res}", exc_info=True)
            elif isinstance(res, set):
                violations.update(res)
            elif isinstance(res, tuple) and len(res) == 2:
                ai_violations, proactive_reason = res
                violations.update(ai_violations)
                if proactive_reason and not proactive_flag_reason:
                    proactive_flag_reason = proactive_reason
                    
        return violations, proactive_flag_reason

    async def _handle_violations(self, message: discord.Message, violations: set[str]):
        """Handles the consequences of a detected violation (delete, warn, punish)."""
        _ = get_translator(message.guild.preferred_locale.language if message.guild.preferred_locale else 'en')

        if await get_guild_config(message.guild.id, "delete_violating_messages", bot_config.delete_violating_messages):
            try:
                await message.delete()
            except (discord.Forbidden, discord.NotFound):
                pass 

        if await get_guild_config(message.guild.id, "send_in_channel_warning", bot_config.send_in_channel_warning):
            viol_summary = ", ".join(v.replace('_', ' ').title() for v in violations)
            warn_text = _("{mention}, your message was moderated for: **{summary}**. Please review the rules.").format(
                mention=message.author.mention, summary=viol_summary
            )
            try:
                delete_delay_str = await get_guild_config(message.guild.id, "in_channel_warning_delete_delay", bot_config.in_channel_warning_delete_delay)
                delete_delay = int(delete_delay_str)
                await message.channel.send(warn_text, delete_after=delete_delay)
            except (discord.Forbidden, ValueError):
                pass 

        if isinstance(message.author, discord.Member):
            await process_infractions_and_punish(message.author, message.guild, list(violations), message.content, message.jump_url, _=_)

    def check_dynamic_rules(self, guild_id: int, raw_content: str, cleaned_content: str) -> set[str]:
        """Checks content against global and guild-specific dynamic rules."""
        violations = set()
        global_rules = dynamic_rules_cache.get("global", {})
        if any(regex.search(raw_content) for regex in global_rules.get("forbidden_regex", [])) or \
           any(word in cleaned_content.split() for word in global_rules.get("forbidden_words", set())) or \
           any(phrase in cleaned_content for phrase in global_rules.get("forbidden_phrases", [])):
            violations.add("dynamic_rule_violation_global")

        guild_rules = dynamic_rules_cache.get(guild_id, {})
        if any(regex.search(raw_content) for regex in guild_rules.get("forbidden_regex", [])) or \
           any(word in cleaned_content.split() for word in guild_rules.get("forbidden_words", set())) or \
           any(phrase in cleaned_content for phrase in guild_rules.get("forbidden_phrases", [])):
            violations.add("dynamic_rule_violation_local")
            
        return violations

    def check_spam(self, user_id: int, guild_id: int, cleaned_content: str) -> set[str]:
        """Checks for message rate spam and repetitive content."""
        violations = set()
        now_ts = datetime.now(timezone.utc).timestamp()

        timestamps = self.user_message_timestamps[guild_id][user_id]
        timestamps.append(now_ts)
        while timestamps and (now_ts - timestamps[0] > bot_config.spam_window_seconds):
            timestamps.popleft()
        if len(timestamps) > bot_config.spam_message_limit:
            violations.add("spam_rate")

        history = self.user_message_history[guild_id][user_id]
        if cleaned_content and "spam_rate" not in violations:
            if any(fuzz.ratio(cleaned_content, old_msg) > bot_config.spam_repetition_fuzzy_threshold for old_msg in history):
                violations.add("spam_repetition")
                history.clear()
        history.append(cleaned_content)
        
        return violations

    async def check_advertising(self, raw_content: str, guild_id: int) -> set[str]:
        """Checks for forbidden advertising text and unpermitted URLs."""
        if bot_config.forbidden_text_pattern.search(raw_content):
            return {"advertising_forbidden"}
        
        guild_permitted_domains = await get_guild_config(guild_id, "permitted_domains", bot_config.permitted_domains)
        
        for url_match in re.finditer(bot_config.url_pattern, raw_content):
            try:
                url_str = url_match.group(0)
                if not url_str.startswith(('http://', 'https://')):
                    url_str = 'http://' + url_str
                
                domain = urlparse(url_str).netloc.lower().lstrip('www.')
                if domain and not any(allowed == domain or domain.endswith(f'.{allowed}') for allowed in guild_permitted_domains):
                    return {"advertising_unpermitted_url"}
            except Exception:
                continue 
        return set()

    def check_message_limits(self, message: discord.Message) -> set[str]:
        """Checks for excessive mentions, attachments, or message length."""
        v = set()
        if len(message.mentions) > bot_config.mention_limit:
            v.add("excessive_mentions")
        if len(message.attachments) > bot_config.max_attachments:
            v.add("excessive_attachments")
        if len(message.content) > bot_config.max_message_length:
            v.add("long_message")
        return v

    async def check_language(self, message: discord.Message, raw_content: str) -> set[str]:
        """Checks if the message language is permitted in the channel."""
        if len(raw_content.split()) < bot_config.min_word_count_for_lang_check:
            return set()
            
        lang_config = await self.get_effective_channel_language_config(message.guild.id, message.channel.id)
        if not lang_config or "any" in lang_config:
            return set()
        
        lang_code, confidence = await detect_language_ai(raw_content)
        if not lang_code or lang_code in lang_config:
            return set()

        threshold = bot_config.min_confidence_short_msg_lang if len(raw_content) < bot_config.short_msg_threshold_lang else bot_config.min_confidence_for_lang_flagging

        if confidence >= threshold and not any(word in raw_content.lower() for word in bot_config.common_safe_foreign_words):
            return {"foreign_language"}
        return set()

    def check_keyword_violations(self, cleaned_content: str) -> set[str]:
        """Checks for static keyword/phrase violations using fuzzy matching."""
        violations = set()
        words = set(cleaned_content.split())

        if any(fuzz.partial_ratio(phrase, cleaned_content) > bot_config.fuzzy_match_threshold_keywords for phrase in discrimination_phrases) or \
           any(word in discrimination_words_set for word in words):
            violations.add("discrimination")

        if any(fuzz.partial_ratio(phrase, cleaned_content) > bot_config.fuzzy_match_threshold_keywords for phrase in nsfw_text_phrases) or \
           any(word in nsfw_text_words_set for word in words):
            violations.add("nsfw_text")
            
        return violations

    async def check_ai_text_moderation(self, content: str, guild_id: int) -> tuple[set[str], str | None]:
        """Primary AI text moderation using OpenAI with Perspective API as a fallback."""
        violations, proactive_reason = set(), None
        try:
            result = await check_openai_moderation_api(content)
            if result:
                categories, scores = result.get("categories", {}), result.get("category_scores", {})
                if result.get("flagged"):
                    if categories.get("hate") or categories.get("hate/threatening"): violations.add("ai_hate")
                    if categories.get("sexual") or categories.get("sexual/minors"): violations.add("ai_sexual")
                    if categories.get("violence") or categories.get("violence/graphic"): violations.add("ai_violence")
                    if categories.get("harassment") or categories.get("harassment/threatening"): violations.add("ai_harassment")
                    if categories.get("self-harm") or categories.get("self-harm/intent") or categories.get("self-harm/instructions"): violations.add("ai_self_harm")
                else:
                    threshold = await get_guild_config(guild_id, "proactive_flagging_openai_threshold", bot_config.proactive_flagging_openai_threshold)
                    if scores and (max_score := max(scores.values())) >= threshold:
                        category = max(scores, key=scores.get)
                        proactive_reason = f"Proactive OpenAI Flag ({category.replace('/', ' ')}: {max_score:.2f})"
                return violations, proactive_reason

        except Exception as e:
            logger.error(f"OpenAI moderation failed, attempting fallback. Error: {e}", exc_info=True)
            try:
                p_result = await check_perspective_api(content)
                if p_result:
                    if p_result.get("SEVERE_TOXICITY", {}).get("summaryScore", {}).get("value", 0) > 0.8: violations.add("ai_hate")
                    if p_result.get("SEXUALLY_EXPLICIT", {}).get("summaryScore", {}).get("value", 0) > 0.85: violations.add("ai_sexual")
                    if p_result.get("THREAT", {}).get("summaryScore", {}).get("value", 0) > 0.8: violations.add("ai_violence")
                    if p_result.get("TOXICITY", {}).get("summaryScore", {}).get("value", 0) > 0.9: violations.add("ai_toxicity")

                    threshold = await get_guild_config(guild_id, "proactive_flagging_perspective_threshold", bot_config.proactive_flagging_perspective_threshold)
                    max_p_score = max(attr.get("summaryScore", {}).get("value", 0) for attr in p_result.values())
                    if not violations and max_p_score >= threshold:
                        category = max(p_result, key=lambda k: p_result[k].get("summaryScore", {}).get("value", 0))
                        proactive_reason = f"Proactive Perspective API Flag ({category}: {max_p_score:.2f})"
            except Exception as p_e:
                logger.error(f"Perspective API fallback also failed: {p_e}", exc_info=True)

        return violations, proactive_reason

    async def check_ai_media_moderation(self, attachments: list[discord.Attachment]) -> set[str]:
        """Checks media attachments for unsafe content using Sightengine."""
        for attach in attachments:
            if not (attach.content_type and attach.content_type.startswith(("image/", "video/"))):
                continue
            try:
                result = await check_sightengine_media_api(attach.url)
                if not result: continue
                
                if result.get("nudity", {}).get("sexual_activity", 0) > bot_config.sightengine_nudity_sexual_activity_threshold or \
                   result.get("nudity", {}).get("suggestive", 0) > bot_config.sightengine_nudity_suggestive_threshold:
                    return {"nsfw_media"}
                if result.get("gore", {}).get("prob", 0) > bot_config.sightengine_gore_threshold:
                    return {"gore_violence_media"}
                if result.get("offensive", {}).get("prob", 0) > bot_config.sightengine_offensive_symbols_threshold:
                    return {"offensive_symbols_media"}
            except Exception as e:
                logger.error(f"Sightengine check failed for attachment {attach.url}: {e}", exc_info=True)
        return set()

    async def add_to_review_queue(self, message: discord.Message, reason: str):
        """Adds a message to the moderation review queue."""
        if not db_conn: return
        _ = get_translator(message.guild.preferred_locale.language if message.guild.preferred_locale else 'en')
        try:
            cursor = await db_conn.execute(
                "INSERT OR IGNORE INTO review_queue (guild_id, user_id, channel_id, message_id, message_content, reason, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (message.guild.id, message.author.id, message.channel.id, message.id, message.content, reason, datetime.now(timezone.utc).isoformat())
            )
            if cursor.rowcount == 0:
                return 
            await db_conn.commit()
            
            review_channel_id = await get_guild_config(message.guild.id, "review_channel_id")
            if review_channel_id and (review_channel := self.bot.get_channel(int(review_channel_id))):
                embed = discord.Embed(
                    title=_("🚨 Message Flagged for Review"),
                    description=_("**Reason:** {reason}").format(reason=reason),
                    color=discord.Color.yellow(),
                    timestamp=message.created_at
                )
                embed.add_field(name=_("Author"), value=f"{message.author.mention} (`{message.author.id}`)")
                embed.add_field(name=_("Channel"), value=f"{message.channel.mention}")
                embed.add_field(name=_("Context"), value=_("[Jump to Message]({url})").format(url=message.jump_url), inline=False)
                embed.add_field(name=_("Content"), value=f"```\n{discord.utils.escape_markdown(message.content[:500])}\n```", inline=False)
                await review_channel.send(embed=embed)
        except Exception as e:
            logger.error(f"Failed to add message {message.id} to review queue: {e}", exc_info=True)

    mod_group = app_commands.Group(name="mod", description="Moderation commands.", default_permissions=discord.Permissions(manage_messages=True))

    @mod_group.command(name="infractions", description="View a user's recent infractions and point total.")
    async def infractions_command(self, interaction: discord.Interaction, member: discord.Member, days: app_commands.Range[int, 1, 365] = 90):
        _ = get_translator(interaction.locale.language if interaction.locale else 'en')
        if not interaction.guild_id: return
        
        infraction_list, active_points = await get_user_infractions_from_db(member.id, interaction.guild_id, days)
        
        embed = discord.Embed(
            title=_("Infraction Report for {member_name}").format(member_name=member.display_name),
            color=discord.Color.red() if active_points > 20 else discord.Color.orange()
        )
        embed.set_thumbnail(url=member.display_avatar.url)
        embed.add_field(name=_("Active Points (Last 90 Days)"), value=f"**{active_points}**", inline=False)
        
        if infraction_list:
            history_str = "\n".join([
                _("**ID:** `{id}` | **{pts} pts** - {v_type}\n> <t:{ts}:R>").format(
                    id=i['id'],
                    pts=i['points'],
                    v_type=i['violation_type'].replace('_', ' ').title(),
                    ts=int(datetime.fromisoformat(i['timestamp']).timestamp())
                ) for i in infraction_list[:10]
            ])
            embed.add_field(name=_("Recent Infractions (Last {days} Days)").format(days=days), value=history_str, inline=False)
        else:
            embed.description = _("This user has no recent infractions.")
            
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @mod_group.command(name="clear_infractions", description="Clears ALL infractions or a specific one for a user.")
    @app_commands.default_permissions(administrator=True)
    @app_commands.describe(member="The member whose infractions to clear.", infraction_id="Optional ID of a specific infraction to remove.")
    async def clear_infractions_command(self, interaction: discord.Interaction, member: discord.Member, infraction_id: int | None = None):
        _ = get_translator(interaction.locale.language if interaction.locale else 'en')
        if not interaction.guild or not interaction.user: return
        
        if infraction_id:
            success = await remove_specific_infraction_from_db(infraction_id)
            action_desc = _("infraction ID `{id}`").format(id=infraction_id)
        else:
            await clear_user_infractions(member.id, interaction.guild.id)
            success = True
            action_desc = _("all infractions")
        
        if success:
            await interaction.response.send_message(_("Successfully cleared {desc} for {mention}.").format(desc=action_desc, mention=member.mention), ephemeral=True)
            log_embed = discord.Embed(
                title=_("✅ Infractions Cleared"),
                description=_("Moderator {mod_mention} cleared {desc} for {user_mention}.").format(mod_mention=interaction.user.mention, desc=action_desc, user_mention=member.mention),
                color=discord.Color.green()
            )
            await log_moderation_action(interaction.guild, log_embed, _=_)
        else:
            await interaction.response.send_message(_("Failed to clear {desc}. It may not exist.").format(desc=action_desc), ephemeral=True)

@bot.event
async def setup_hook():
    """Asynchronous setup that runs before the bot logs in."""
    global http_session, LANGUAGE_MODEL, perspective_client

    http_session = ClientSession(connector=aiohttp.TCPConnector(ssl=False))

    if os.path.exists(FASTTEXT_MODEL_PATH):
        try:
            LANGUAGE_MODEL = fasttext.load_model(FASTTEXT_MODEL_PATH)
            logger.info(f"FastText model loaded from {FASTTEXT_MODEL_PATH}")
        except Exception as e:
            logger.critical(f"Failed to load FastText model: {e}. Language detection will be disabled.", exc_info=True)
    else:
        logger.error(f"FastText model not found at {FASTTEXT_MODEL_PATH}. Language detection disabled.")

    if PERSPECTIVE_API_KEY:
        try:
            perspective_client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=PERSPECTIVE_API_KEY,
                static_discovery=False,
            )
            logger.info("Perspective API client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Perspective API client: {e}. Fallback moderation disabled.", exc_info=True)

    await setup_database()
    await load_dynamic_rules_from_db()

    await bot.add_cog(GeneralCog(bot))
    await bot.add_cog(ConfigurationCog(bot))
    await bot.add_cog(ModerationCog(bot))

    try:
        synced_commands = await bot.tree.sync()
        logger.info(f"Synced {len(synced_commands)} application commands globally.")
    except Exception as e:
        logger.error(f"Failed to sync application commands: {e}", exc_info=True)

@bot.event
async def on_ready():
    """Event that runs when the bot is connected and ready."""
    logger.info(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    logger.info(f"Discord.py Version: {discord.__version__}")
    logger.info(f"Adroit Perfected V2 is online and ready!")
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="for rule violations"))

async def main():
    """The main function to start the bot."""
    async with bot:
        await bot.start(DISCORD_TOKEN)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested via KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"💥 Unhandled exception at top level: {e}", exc_info=True)
    finally:
        if http_session and not http_session.closed:
            logger.info("Closing aiohttp session.")
            asyncio.run(http_session.close())
        if db_conn:
            logger.info("Closing database connection.")
            asyncio.run(db_conn.close())
