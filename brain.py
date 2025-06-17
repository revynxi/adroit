import asyncio
import json
import logging
import logging.handlers
import os
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, quote_plus
import io

import aiosqlite
import discord
import discord.ui
import fasttext
import aiohttp
from aiohttp import ClientSession, client_exceptions, web
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                     wait_random_exponential)
from thefuzz import fuzz

load_dotenv()

log_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger('adroit_perfected')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

try:
    file_handler = logging.handlers.RotatingFileHandler(
        filename='adroit_perfected.log',
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
FASTTEXT_MODEL_PATH = os.getenv("FASTTEXT_MODEL_PATH", "lid.176.ftz")
SIGHTENGINE_API_USER = os.getenv("SIGHTENGINE_API_USER")
SIGHTENGINE_API_SECRET = os.getenv("SIGHTENGINE_API_SECRET")

if not DISCORD_TOKEN:
    logger.critical("CRITICAL: ADROIT_TOKEN not set. Exiting.")
    exit(1)
if not OPENAI_API_KEY:
    logger.critical("CRITICAL: OPENAI_API_KEY not set. Text moderation disabled. Exiting.")
    exit(1)
if not (SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET):
    logger.warning("Warning: Sightengine API keys not set. Image moderation disabled.")

intents = discord.Intents.default()
intents.members = True
intents.message_content = True
intents.voice_states = True
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

class BotConfig:
    def __init__(self):
        self.default_log_channel_id = None
        self.default_review_channel_id = None
        self.default_channel_configs = {}
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
                5: {"action": "warn", "reason_suffix": "Minor guideline violations.", "dm_message": "This is a formal warning. Please review the server rules."},
                10: {"action": "mute", "duration_hours": 1, "reason_suffix": "Accumulated violations or spam.", "dm_message": "You have been muted for 1 hour due to repeated violations or spam."},
                20: {"action": "mute", "duration_hours": 6, "reason_suffix": "Significant or repeated violations.", "dm_message": "You have been muted for 6 hours due to significant violations."},
                35: {"action": "kick", "reason_suffix": "Persistent serious violations.", "dm_message": "You have been kicked from the server due to persistent violations."},
                50: {"action": "temp_ban", "duration_days": 3, "reason_suffix": "Severe or multiple major violations.", "dm_message": "You have been temporarily banned for 3 days due to severe violations."},
                75: {"action": "temp_ban", "duration_days": 30, "reason_suffix": "Extreme or highly disruptive behavior.", "dm_message": "You have been temporarily banned for 30 days due to extreme behavior."},
                100: {"action": "ban", "reason_suffix": "Egregious violations or repeat offenses.", "dm_message": "You have been permanently banned from the server."}
            },
            "violations": {
                "discrimination": {"points": 1, "severity": "Low"}, "spam_rate": {"points": 1, "severity": "Low"},
                "spam_repetition": {"points": 1, "severity": "Low"}, "nsfw_text": {"points": 1, "severity": "Low"},
                "nsfw_media": {"points": 5, "severity": "High"}, "advertising_forbidden_text": {"points": 2, "severity": "Low"},
                "advertising_unpermitted_url": {"points": 2, "severity": "Low"}, "politics_discussion_disallowed": {"points": 1, "severity": "Low"},
                "off_topic": {"points": 1, "severity": "Low"}, "foreign_language": {"points": 1, "severity": "Low"},
                "openai_flagged_severe": {"points": 5, "severity": "High"}, "openai_flagged_moderate": {"points": 5, "severity": "High"},
                "excessive_mentions": {"points": 1, "severity": "Low"}, "excessive_attachments": {"points": 1, "severity": "Low"},
                "long_message": {"points": 1, "severity": "Low"}, "gore_violence_media": {"points": 5, "severity": "High"},
                "offensive_symbols_media": {"points": 3, "severity": "Medium"}, "dynamic_rule_violation": {"points": 3, "severity": "Medium"}
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
        self.min_confidence_for_lang_flagging = 0.60
        self.min_confidence_short_msg_lang = 0.70
        self.short_msg_threshold_lang = 25
        self.common_safe_foreign_words = {"bonjour", "hola", "merci", "gracias", "oui", "si", "nyet", "da", "salut", "ciao", "hallo", "guten tag", "privet", "konnichiwa", "arigato", "sawasdee", "namaste"}
        self.fuzzy_match_threshold_keywords = 85
        self.openai_api_cooldown_seconds = 5
        self.sightengine_nudity_sexual_activity_threshold = 0.45
        self.sightengine_nudity_suggestive_threshold = 0.5
        self.sightengine_gore_threshold = 0.5
        self.sightengine_offensive_symbols_threshold = 0.45
        self.proactive_flagging_openai_threshold = 0.45
        self.delete_violating_messages = True
        self.send_in_channel_warning = True
        self.in_channel_warning_delete_delay = 30

bot_config = BotConfig()

def load_terms_from_file(filepath: str) -> tuple[set[str], list[str]]:
    words = set()
    phrases = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
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
        logger.warning(f"Terms file '{filepath}' not found.")
    except Exception as e:
        logger.error(f"Error loading terms from {filepath}: {e}", exc_info=True)
    return words, phrases

discrimination_words_set, discrimination_phrases = load_terms_from_file('discrimination_terms.txt')
nsfw_text_words_set, nsfw_text_phrases = load_terms_from_file('nsfw_terms.txt')

def clean_message_for_language_detection(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<@!?\d+>|<#\d+>|<@&\d+>', '', text)
    text = re.sub(r'<a?:\w+:\d+>', '', text)
    text = re.sub(r'(\*|_|`|~|>|\|)', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    normalized_text = re.sub(r'\s+', ' ', text).strip()
    return normalized_text

def clean_message_content(text: str) -> str:
    normalized_text = re.sub(r'\s+', ' ', text).strip()
    return normalized_text.lower()

async def get_guild_config(guild_id: int, key: str, default_value=None):
    if not db_conn:
        logger.error("get_guild_config: Database connection unavailable.")
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
        logger.error(f"Error getting guild config for guild {guild_id}, key '{key}': {e}", exc_info=True)
        return default_value

async def set_guild_config(guild_id: int, key: str, value_to_set):
    if not db_conn:
        logger.error("set_guild_config: Database connection unavailable.")
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
        logger.error(f"Error setting guild config for guild {guild_id}, key '{key}': {e}", exc_info=True)

async def detect_language_ai(text: str) -> tuple[str | None, float]:
    clean_text = clean_message_for_language_detection(text)
    if not clean_text or not bot_config.has_alphanumeric_pattern.search(clean_text):
        logger.debug(f"No language detection: Text '{clean_text[:50]}...' lacks alphanumeric content.")
        return None, 0.0
    if not LANGUAGE_MODEL:
        logger.warning("FastText model not loaded. Cannot detect language.")
        return None, 0.0
    try:
        prediction = LANGUAGE_MODEL.predict(clean_text.replace("\n", " "), k=1)
        if prediction and prediction[0] and prediction[1] and len(prediction[0]) > 0:
            lang_code = prediction[0][0].replace("__label__", "")
            confidence = float(prediction[1][0])
            logger.debug(f"Detected language: {lang_code} with confidence {confidence:.2f} for text: '{clean_text[:50]}...'")
            return lang_code, confidence
        else:
            logger.debug(f"FastText unexpected prediction for: '{clean_text[:50]}...' -> {prediction}")
            return None, 0.0
    except Exception as e:
        logger.error(f"FastText language detection error for '{clean_text[:50]}...': {e}", exc_info=True)
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
    current_guild = guild or (target_user.guild if isinstance(target_user, discord.Member) else None)
    if not current_guild:
        logger.error(f"Cannot log action '{action_type}' for user {target_user.id}: Guild context missing.")
        return
    log_channel_id = await get_guild_config(current_guild.id, "log_channel_id", bot_config.default_log_channel_id)
    log_channel = bot.get_channel(log_channel_id) if log_channel_id else None
    embed = discord.Embed(
        title=f"🛡️ Moderation: {action_type.replace('_', ' ').title()}",
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
    logger.info(f"Log (Guild: {current_guild.name}, Action: {action_type}, Target: {target_user.id}, Reason: {reason})")
    if log_channel:
        try:
            await log_channel.send(embed=embed)
        except discord.Forbidden:
            logger.error(f"Missing permissions to send logs to channel #{log_channel.name} ({log_channel.id}) in guild {current_guild.name}.")
        except Exception as e:
            logger.error(f"Error sending log embed to channel for guild {current_guild.name}: {e}", exc_info=True)
    else:
        logger.warning(f"Moderation log channel not configured for guild {current_guild.name}. Logs sent to console.")

def retry_if_api_error(exception):
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
    if not text_content.strip():
        return {"flagged": False, "categories": {}, "category_scores": {}}
    url = "https://api.openai.com/v1/moderations"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"input": text_content.replace("\n", " ")}
    if not http_session or http_session.closed:
        logger.error("HTTP session unavailable for OpenAI moderation.")
        return {"flagged": False, "categories": {}, "category_scores": {}}
    try:
        async with http_session.post(url, headers=headers, json=data, timeout=15) as response:
            response.raise_for_status()
            json_response = await response.json()
            results = json_response.get("results", [])
            if results:
                logger.debug(f"OpenAI moderation response: {json_response}")
                return results[0]
            logger.warning(f"OpenAI moderation empty results for text: {text_content[:50]}...")
            return {"flagged": False, "categories": {}, "category_scores": {}}
    except client_exceptions.ClientResponseError as e:
        if e.status == 400:
            logger.warning(f"OpenAI moderation: 400 Bad Request. Text: '{text_content[:50]}...'. Error: {e.message}")
            return {"flagged": False, "categories": {}, "category_scores": {}}
        logger.error(f"OpenAI moderation API error: {e.status} - {e.message}.")
        raise
    except asyncio.TimeoutError:
        logger.error(f"OpenAI moderation API timed out. Text: {text_content[:50]}...")
        raise
    except Exception as e:
        logger.error(f"Unexpected error with OpenAI moderation: {e} for text: {text_content[:50]}...", exc_info=True)
        return {"flagged": False, "categories": {}, "category_scores": {}}

@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_api_error,
    reraise=True
)
async def check_sightengine_media_api(image_url: str) -> dict:
    if not SIGHTENGINE_API_USER or not SIGHTENGINE_API_SECRET:
        return {}
    if not http_session or http_session.closed:
        logger.error("HTTP session unavailable for Sightengine.")
        return {}
    models = "nudity-2.0,gore,offensive"
    encoded_url = quote_plus(image_url)
    api_url = f"https://api.sightengine.com/1.0/check.json?url={encoded_url}&models={models}&api_user={SIGHTENGINE_API_USER}&api_secret={SIGHTENGINE_API_SECRET}"
    try:
        async with http_session.get(api_url, timeout=20) as response:
            response.raise_for_status()
            json_response = await response.json()
            if json_response.get("status") == "success":
                logger.debug(f"Sightengine response: {json_response}")
                return json_response
            logger.warning(f"Sightengine non-success: {json_response.get('error', {}).get('message')} for URL: {image_url}")
            return {}
    except client_exceptions.ClientResponseError as e:
        logger.error(f"Sightengine API error: {e.status} - {e.message} for URL: {image_url}.")
        raise
    except asyncio.TimeoutError:
        logger.error(f"Sightengine API timed out for URL: {image_url}...")
        raise
    except Exception as e:
        logger.error(f"Unexpected error with Sightengine API for URL: {image_url}: {e}", exc_info=True)
        return {}

async def apply_moderation_punishment(
    member: discord.Member,
    action_config: dict,
    total_points: int,
    violation_summary: str,
    moderator: discord.User | None = None,
    original_content: str | None = None
):
    action_type = action_config["action"]
    guild = member.guild
    reason_suffix = action_config.get("reason_suffix", f"Automated action due to {total_points} points.")
    if moderator:
        full_reason = f"Manual action by {moderator.name} ({moderator.id}): {reason_suffix}"
    else:
        full_reason = f"Automated Action | Points: {total_points} | {reason_suffix} | Violations: {violation_summary}"
    dm_message_base = action_config.get("dm_message", f"Action taken: {action_type}. Reason: {full_reason}")
    dm_message_text = f"Hello {member.name},\n\nIn **{guild.name}**:\n\n{dm_message_base}"
    if original_content:
        dm_message_text += f"\n\nContent: ```{original_content[:500]}```"
    log_color = discord.Color.orange()
    extra_log_fields = []
    duration: timedelta | None = None
    if "duration_hours" in action_config:
        duration = timedelta(hours=action_config["duration_hours"])
    elif "duration_days" in action_config:
        duration = timedelta(days=action_config["duration_days"])
    if duration:
        dm_message_text += f"\n\nDuration: **{str(duration)}**."
        extra_log_fields.append(("Duration", str(duration)))
    bot_member = guild.me
    if not bot_member:
        logger.error(f"Bot member not found in guild {guild.name}. Cannot apply punishment.")
        return
    required_perm = None
    if action_type == "mute": required_perm = "moderate_members"
    elif action_type in ["kick", "ban", "temp_ban"]: required_perm = f"{action_type.replace('temp_', '')}_members"
    if required_perm and not getattr(bot_member.guild_permissions, required_perm, False):
        logger.error(f"Bot lacks '{required_perm}' permission in {guild.name} to {action_type} {member.display_name}.")
        return
    if isinstance(member, discord.Member) and bot_member.top_role <= member.top_role:
        logger.warning(f"Bot's top role not higher than {member.display_name}'s. Cannot {action_type}.")
        return
    try:
        try:
            if dm_message_text:
                await member.send(dm_message_text)
                logger.info(f"DM sent to {member.display_name} ({member.id}) for {action_type}.")
        except discord.Forbidden:
            logger.warning(f"Could not DM {action_type} notification to {member.display_name}.")
        except Exception as e:
            logger.error(f"Error sending DM to {member.display_name} for {action_type}: {e}", exc_info=True)
        if action_type == "warn":
            log_color = discord.Color.gold()
        elif action_type == "mute":
            if duration:
                await member.timeout(duration, reason=full_reason)
                log_color = discord.Color.light_grey()
            else:
                logger.warning(f"Mute for {member.id} called without duration. Skipped.")
                return
        elif action_type == "kick":
            await member.kick(reason=full_reason)
            log_color = discord.Color.red()
        elif action_type == "temp_ban":
            if duration:
                unban_time = datetime.now(timezone.utc) + duration
                if db_conn:
                    await db_conn.execute(
                        'INSERT OR REPLACE INTO temp_bans (user_id, guild_id, unban_time, ban_reason) VALUES (?, ?, ?, ?)',
                        (member.id, guild.id, unban_time.isoformat(), full_reason)
                    )
                    await db_conn.commit()
                await guild.ban(member, reason=full_reason, delete_message_seconds=0)
                log_color = discord.Color.dark_red()
                extra_log_fields.append(("Unban Time", f"{discord.utils.format_dt(unban_time, 'R')}"))
            else:
                logger.warning(f"Temp_ban for {member.id} called without duration. Skipped.")
                return
        elif action_type == "ban":
            await guild.ban(member, reason=full_reason, delete_message_seconds=0)
            log_color = discord.Color.dark_red()
        await log_moderation_action(action_type, member, full_reason, moderator, guild, log_color, extra_log_fields)
    except discord.Forbidden:
        logger.error(f"Bot lacks permissions to perform '{action_type}' on {member.display_name} in {guild.name}.")
    except discord.HTTPException as e:
        logger.error(f"Discord API error applying {action_type} to {member.id}: Status {e.status} - {e.text}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error applying {action_type} to {member.id}: {e}", exc_info=True)

async def process_infractions_and_punish(
    member: discord.Member,
    guild: discord.Guild,
    violation_types: list[str],
    content: str,
    message_url: str | None
):
    if not db_conn:
        logger.error("process_infractions_and_punish: Database connection unavailable.")
        return
    user_id, guild_id = member.id, guild.id
    points_this_action = 0
    summary_parts = []
    async with db_conn.cursor() as cursor:
        for v_type in violation_types:
            v_config = bot_config.punishment_system["violations"].get(v_type)
            if not v_config:
                logger.warning(f"Unknown violation type '{v_type}' for user {user_id}. Skipped.")
                continue
            points = v_config["points"]
            points_this_action += points
            summary_parts.append(v_type.replace('_', ' ').title())
            try:
                await cursor.execute(
                    "INSERT INTO infractions (user_id, guild_id, violation_type, points, message_content_snippet, message_url, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (user_id, guild_id, v_type, points, content[:500], message_url, datetime.now(timezone.utc).isoformat())
                )
            except Exception as e:
                logger.error(f"Failed to record infraction '{v_type}' for user {user_id}: {e}", exc_info=True)
        if not summary_parts:
            await db_conn.commit()
            return
        ninety_days_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        await cursor.execute(
            "SELECT SUM(points) FROM infractions WHERE user_id = ? AND guild_id = ? AND timestamp >= ?",
            (user_id, guild_id, ninety_days_ago)
        )
        total_points_result = await cursor.fetchone()
        total_active_points = total_points_result[0] if total_points_result and total_points_result[0] is not None else 0
        logger.info(f"User {user_id} has {total_active_points} active points (added {points_this_action}).")
        applicable_punishment_config = None
        for threshold in sorted(bot_config.punishment_system["points_thresholds"].keys(), reverse=True):
            if total_active_points >= threshold:
                applicable_punishment_config = bot_config.punishment_system["points_thresholds"][threshold]
                break
        if applicable_punishment_config:
            summary_str = ", ".join(summary_parts)
            await apply_moderation_punishment(member, applicable_punishment_config, total_active_points, summary_str, original_content=content)
            if applicable_punishment_config["action"] == "ban":
                await cursor.execute("DELETE FROM infractions WHERE user_id = ? AND guild_id = ?", (user_id, guild_id))
                logger.info(f"Cleared infractions for user {user_id} due to ban.")
        await db_conn.commit()

async def get_user_infractions_from_db(user_id: int, guild_id: int, days_limit: int = 90) -> tuple[list[dict], int]:
    if not db_conn:
        return [], 0
    infractions, total_active_points = [], 0
    try:
        query = "SELECT id, violation_type, points, message_content_snippet, message_url, timestamp FROM infractions WHERE user_id = ? AND guild_id = ? AND timestamp >= ? ORDER BY timestamp DESC"
        past_time_limit = (datetime.now(timezone.utc) - timedelta(days=days_limit)).isoformat()
        async with db_conn.execute(query, (user_id, guild_id, past_time_limit)) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                infractions.append({
                    "id": row[0], "violation_type": row[1], "points": row[2],
                    "message_content_snippet": row[3], "message_url": row[4], "timestamp": row[5]
                })
        ninety_days_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        async with db_conn.execute(
            "SELECT SUM(points) FROM infractions WHERE user_id = ? AND guild_id = ? AND timestamp >= ?",
            (user_id, guild_id, ninety_days_ago)
        ) as cursor:
            result = await cursor.fetchone()
            total_active_points = result[0] if result and result[0] is not None else 0
    except Exception as e:
        logger.error(f"Error retrieving infractions for user {user_id}: {e}", exc_info=True)
    return infractions, total_active_points

async def clear_user_infractions(user_id: int, guild_id: int):
    if not db_conn: return
    try:
        async with db_conn.execute("DELETE FROM infractions WHERE user_id = ? AND guild_id = ?", (user_id, guild_id)):
            await db_conn.commit()
        logger.info(f"Cleared infractions for user {user_id} in guild {guild_id}.")
    except Exception as e:
        logger.error(f"Error clearing infractions for user {user_id}: {e}", exc_info=True)

async def remove_specific_infraction_from_db(infraction_id: int) -> bool:
    if not db_conn: return False
    try:
        async with db_conn.execute("DELETE FROM infractions WHERE id = ?", (infraction_id,)) as cursor:
            await db_conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Infraction {infraction_id} removed.")
                return True
            logger.warning(f"Attempted to remove non-existent infraction {infraction_id}.")
            return False
    except Exception as e:
        logger.error(f"Error removing infraction {infraction_id}: {e}", exc_info=True)
        return False

async def get_temp_bans_from_db() -> list[dict]:
    if not db_conn: return []
    bans = []
    try:
        async with db_conn.execute("SELECT user_id, guild_id, unban_time, ban_reason FROM temp_bans") as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                bans.append({
                    "user_id": row[0], "guild_id": row[1],
                    "unban_time": datetime.fromisoformat(row[2]),
                    "ban_reason": row[3]
                })
    except Exception as e:
        logger.error(f"Error retrieving temporary bans: {e}", exc_info=True)
    return bans

async def remove_temp_ban_from_db(user_id: int, guild_id: int):
    if not db_conn: return
    try:
        await db_conn.execute("DELETE FROM temp_bans WHERE user_id = ? AND guild_id = ?", (user_id, guild_id))
        await db_conn.commit()
        logger.info(f"Removed temp ban for user {user_id} in guild {guild_id}.")
    except Exception as e:
        logger.error(f"Error removing temp ban for user {user_id}: {e}", exc_info=True)

async def setup_db():
    global db_conn
    try:
        db_conn = await aiosqlite.connect('adroit_perfected_data.db')
        await db_conn.execute("PRAGMA journal_mode=WAL;")
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
                rule_type TEXT NOT NULL,
                pattern TEXT NOT NULL,
                added_by_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                UNIQUE (guild_id, rule_type, pattern)
            )
        """)
        await db_conn.execute("""
            CREATE TABLE IF NOT EXISTS review_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                channel_id INTEGER,
                message_id INTEGER UNIQUE,
                message_content TEXT NOT NULL,
                reason TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_review_queue_guild_timestamp ON review_queue (guild_id, timestamp)')
        await db_conn.commit()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize database: {e}", exc_info=True)
        if db_conn:
            await db_conn.close()
        exit(1)

async def load_dynamic_rules_from_db():
    if not db_conn:
        logger.error("Cannot load dynamic rules: Database connection unavailable.")
        return
    dynamic_rules["forbidden_words"].clear()
    dynamic_rules["forbidden_phrases"].clear()
    dynamic_rules["forbidden_regex"] = []
    try:
        async with db_conn.execute("SELECT guild_id, rule_type, pattern FROM dynamic_rules") as cursor:
            rows = await cursor.fetchall()
            for _, rule_type, pattern in rows:
                if rule_type == 'forbidden_word':
                    dynamic_rules["forbidden_words"].add(pattern.lower())
                elif rule_type == 'forbidden_phrase':
                    dynamic_rules["forbidden_phrases"].append(pattern.lower())
                elif rule_type == 'forbidden_regex':
                    try:
                        dynamic_rules["forbidden_regex"].append(re.compile(pattern, re.IGNORECASE))
                    except re.error as e:
                        logger.error(f"Failed to compile regex '{pattern}': {e}")
            logger.info(f"Loaded {len(dynamic_rules['forbidden_words'])} words, {len(dynamic_rules['forbidden_phrases'])} phrases, {len(dynamic_rules['forbidden_regex'])} regex patterns.")
    except Exception as e:
        logger.error(f"Error loading dynamic rules: {e}", exc_info=True)

class GeneralCog(commands.Cog, name="General"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="ping", description="Checks the bot's latency.")
    async def ping(self, interaction: discord.Interaction):
        latency = round(self.bot.latency * 1000)
        await interaction.response.send_message(f"Pong! Latency: {latency}ms", ephemeral=True)

    @app_commands.command(name="help", description="Shows information about bot commands.")
    async def help_command(self, interaction: discord.Interaction):
        embed = discord.Embed(title="Adroit Bot Help", description="Available commands:", color=discord.Color.blue())
        for cog_name, cog in self.bot.cogs.items():
            command_list = []
            for cmd in cog.get_app_commands():
                if isinstance(cmd, app_commands.Command):
                    command_list.append(f"`/{cmd.name}` - {cmd.description}")
                elif isinstance(cmd, app_commands.Group):
                    for sub_cmd in cmd.commands:
                        command_list.append(f"`/{cmd.name} {sub_cmd.name}` - {sub_cmd.description}")
            if command_list:
                embed.add_field(name=f"**{cog_name} Commands**", value="\n".join(command_list), inline=False)
        if not embed.fields:
            embed.description = "No commands found."
        await interaction.response.send_message(embed=embed, ephemeral=True)

class ConfigurationCog(commands.Cog, name="Configuration"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
    group = app_commands.Group(name="config", description="Configure bot settings.")
    @group.command(name="set_log_channel", description="Sets the moderation log channel.")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(channel="Text channel for logs. Leave empty to clear.")
    async def set_log_channel(self, interaction: discord.Interaction, channel: discord.TextChannel | None = None):
        if not interaction.guild_id: return
        if channel:
            await set_guild_config(interaction.guild_id, "log_channel_id", channel.id)
            await interaction.response.send_message(f"Moderation logs set to {channel.mention}.", ephemeral=True)
        else:
            await set_guild_config(interaction.guild_id, "log_channel_id", None)
            await interaction.response.send_message("Moderation log channel cleared.", ephemeral=True)
    @group.command(name="set_review_channel", description="Sets the channel for messages needing review.")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(channel="Text channel for review notifications. Leave empty to clear.")
    async def set_review_channel(self, interaction: discord.Interaction, channel: discord.TextChannel | None = None):
        if not interaction.guild_id: return
        if channel:
            await set_guild_config(interaction.guild_id, "review_channel_id", channel.id)
            await interaction.response.send_message(f"Review messages set to {channel.mention}.", ephemeral=True)
        else:
            await set_guild_config(interaction.guild_id, "review_channel_id", None)
            await interaction.response.send_message("Review channel cleared.", ephemeral=True)
    @group.command(name="set_default_language", description="Sets default server language(s).")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(languages="Comma-separated ISO 639-1 codes (e.g., en,fr). Use 'any' to disable.")
    async def set_default_language(self, interaction: discord.Interaction, languages: str):
        if not interaction.guild_id: return
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
                await interaction.followup.send(f"Invalid language code: '{lang_code}'. Use 2-3 letter ISO codes or 'any'.", ephemeral=True)
                return
        config_key = "default_language"
        if is_any:
            await set_guild_config(interaction.guild_id, config_key, ["any"])
            await interaction.followup.send(f"Server language check allows **any** language.", ephemeral=True)
        elif valid_langs:
            await set_guild_config(interaction.guild_id, config_key, valid_langs)
            await interaction.followup.send(f"Server languages set to: **{', '.join(valid_langs).upper()}**.", ephemeral=True)
        else:
            await set_guild_config(interaction.guild_id, config_key, None)
            await interaction.followup.send(f"Server language configuration cleared.", ephemeral=True)
    @group.command(name="set_channel_language", description="Overrides server language for a channel.")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(channel="Channel to configure.", languages="Comma-separated ISO codes, 'any', or 'default' to clear.")
    async def set_channel_language(self, interaction: discord.Interaction, channel: discord.TextChannel, languages: str):
        if not interaction.guild_id: return
        await interaction.response.defer(ephemeral=True)
        lang_list_raw = [lang.strip().lower() for lang in languages.split(',')]
        valid_langs = []
        is_any, is_default = False, False
        for lang_code in lang_list_raw:
            if lang_code == "any":
                is_any = True
                break
            if lang_code == "default":
                is_default = True
                break
            if re.fullmatch(r"[a-z]{2,3}", lang_code):
                valid_langs.append(lang_code)
            else:
                await interaction.followup.send(f"Invalid language code: '{lang_code}'. Use 2-3 letter ISO codes, 'any', or 'default'.", ephemeral=True)
                return
        config_key = f"channel_language_{channel.id}"
        if is_any:
            await set_guild_config(interaction.guild_id, config_key, ["any"])
            await interaction.followup.send(f"Language checks for {channel.mention} allow **any** language.", ephemeral=True)
        elif is_default:
            await set_guild_config(interaction.guild_id, config_key, None)
            await interaction.followup.send(f"Language override for {channel.mention} removed. Uses server default.", ephemeral=True)
        elif valid_langs:
            await set_guild_config(interaction.guild_id, config_key, valid_langs)
            await interaction.followup.send(f"Languages for {channel.mention} set to: **{', '.join(valid_langs).upper()}**.", ephemeral=True)
        else:
            await interaction.followup.send("No valid language options provided.", ephemeral=True)
    @group.command(name="set_ai_threshold", description="Adjusts AI text moderation sensitivity (0.0 to 1.0).")
    @app_commands.default_permissions(administrator=True)
    @app_commands.describe(threshold="Value from 0.0 (sensitive) to 1.0 (not sensitive). Default: 0.50.")
    async def set_ai_threshold(self, interaction: discord.Interaction, threshold: app_commands.Range[float, 0.0, 1.0]):
        if not interaction.guild_id: return
        await set_guild_config(interaction.guild_id, "proactive_flagging_openai_threshold", threshold)
        await interaction.response.send_message(f"AI flagging threshold set to `{threshold}`.", ephemeral=True)

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
        self.report_message_context_menu = app_commands.ContextMenu(
            name="Report to Adroit!",
            callback=self.report_message_from_context,
        )
        self.bot.tree.add_command(self.report_message_context_menu)
    def cog_unload(self):
        self.temp_ban_check_task.cancel()
        self.cleanup_old_infractions_task.cancel()
        self.cleanup_spam_trackers_task.cancel()
        self.bot.tree.remove_command(self.report_message_context_menu)
    async def report_message_from_context(self, interaction: discord.Interaction, message: discord.Message):
        if message.author.id == interaction.user.id:
            await interaction.response.send_message("You cannot report your own message.", ephemeral=True)
            return
        if message.author.bot:
            await interaction.response.send_message("You cannot report bot messages.", ephemeral=True)
            return
        modal = ReportMessageModal(message=message)
        await interaction.response.send_modal(modal)
    async def get_effective_channel_language_config(self, guild_id: int, channel_id: int) -> list[str] | None:
        channel_setting = await get_guild_config(guild_id, f"channel_language_{channel_id}")
        if channel_setting is not None:
            return channel_setting if channel_setting else []
        guild_setting = await get_guild_config(guild_id, "default_language")
        if guild_setting is not None:
            return guild_setting if guild_setting else []
        return bot_config.default_channel_configs.get(channel_id, {}).get("language")
    @tasks.loop(minutes=1)
    async def temp_ban_check_task(self):
        if not db_conn: return
        now_utc = datetime.now(timezone.utc)
        expired_bans = await get_temp_bans_from_db()
        for ban_entry in expired_bans:
            user_id, guild_id, unban_time = ban_entry["user_id"], ban_entry["guild_id"], ban_entry["unban_time"]
            if now_utc >= unban_time:
                guild = self.bot.get_guild(guild_id)
                if not guild:
                    await remove_temp_ban_from_db(user_id, guild_id)
                    continue
                user_obj = discord.Object(id=user_id)
                unban_reason = f"Temp ban expired. Reason: {ban_entry['ban_reason']}"
                try:
                    await guild.unban(user_obj, reason=unban_reason)
                    await remove_temp_ban_from_db(user_id, guild_id)
                    logger.info(f"Unbanned user {user_id} from {guild.name} (temp ban expired).")
                    target_user_for_log = await self.bot.fetch_user(user_id)
                    await log_moderation_action("unban_temp_expired", target_user_for_log, unban_reason, guild=guild, color=discord.Color.green())
                except discord.NotFound:
                    await remove_temp_ban_from_db(user_id, guild_id)
                except discord.Forbidden:
                    logger.error(f"Missing permissions to unban user {user_id} from {guild.name}.")
                except Exception as e:
                    logger.error(f"Error unbanning user {user_id}: {e}", exc_info=True)
    @tasks.loop(hours=24)
    async def cleanup_old_infractions_task(self):
        if not db_conn: return
        one_hundred_eighty_days_ago = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
        try:
            async with db_conn.execute("DELETE FROM infractions WHERE timestamp < ?", (one_hundred_eighty_days_ago,)) as cursor:
                await db_conn.commit()
                if cursor.rowcount > 0:
                    logger.info(f"Deleted {cursor.rowcount} infractions older than 180 days.")
        except Exception as e:
            logger.error(f"Error during infraction cleanup: {e}", exc_info=True)
    @tasks.loop(hours=6)
    async def cleanup_spam_trackers_task(self):
        now_ts = datetime.now(timezone.utc).timestamp()
        cleanup_threshold = bot_config.spam_window_seconds * 10
        for guild_id in list(self.user_message_timestamps.keys()):
            for user_id in list(self.user_message_timestamps[guild_id].keys()):
                timestamps = self.user_message_timestamps[guild_id][user_id]
                while timestamps and (now_ts - timestamps[0] > cleanup_threshold):
                    timestamps.popleft()
                if not timestamps:
                    del self.user_message_timestamps[guild_id][user_id]
            if not self.user_message_timestamps[guild_id]:
                del self.user_message_timestamps[guild_id]
        logger.info("Spam tracker cleanup completed.")
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if not message.guild or message.author.bot or message.webhook_id:
            return
        violations, proactive_flag_reason = await self._run_moderation_pipeline(message)
        if violations:
            logger.info(f"User {message.author.id} ({message.author.name}) in guild {message.guild.id} triggered: {list(violations)}. URL: {message.jump_url}")
            await self._handle_violations(message, violations)
        elif proactive_flag_reason:
            await self.add_to_review_queue(message, proactive_flag_reason)
        await self.bot.process_commands(message)
    async def _run_moderation_pipeline(self, message: discord.Message) -> tuple[set[str], str | None]:
        violations = set()
        proactive_flag_reason = None
        content_raw = message.content
        cleaned_content = clean_message_content(content_raw)
        violations.update(self.check_dynamic_rules(content_raw, cleaned_content))
        violations.update(self.check_spam(message.author.id, message.guild.id, cleaned_content))
        violations.update(self.check_message_limits(message))
        violations.update(self.check_keyword_violations(cleaned_content))
        if not violations:
            async_tasks = [
                self.check_advertising(content_raw, message.guild.id),
                self.check_language(message.guild.id, message.channel.id, content_raw),
            ]
            if OPENAI_API_KEY and not (datetime.now(timezone.utc).timestamp() < self.openai_cooldowns.get(message.author.id, 0)):
                async_tasks.append(self.check_ai_text_moderation(content_raw, message.author.id, message.guild.id))
            if message.attachments:
                async_tasks.append(self.check_ai_media_moderation(message.attachments, message.guild.id))
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    logger.error(f"Moderation check failed: {res}", exc_info=True)
                elif isinstance(res, set):
                    violations.update(res)
                elif isinstance(res, tuple):
                    ai_violations, proactive_reason = res
                    violations.update(ai_violations)
                    if proactive_reason and not proactive_flag_reason:
                        proactive_flag_reason = proactive_reason
        return violations, proactive_flag_reason
    async def _handle_violations(self, message: discord.Message, violations: set[str]):
        is_delete_enabled = await get_guild_config(message.guild.id, "delete_violating_messages", bot_config.delete_violating_messages)
        if is_delete_enabled:
            try:
                await message.delete()
                logger.info(f"Deleted message {message.id} from user {message.author.id}.")
            except (discord.Forbidden, discord.NotFound) as e:
                logger.warning(f"Failed to delete message {message.id}: {type(e).__name__}")
        is_warn_enabled = await get_guild_config(message.guild.id, "send_in_channel_warning", bot_config.send_in_channel_warning)
        if is_warn_enabled:
            viol_summary = ", ".join(v.replace('_', ' ').title() for v in violations)
            warn_text = f"{message.author.mention}, your message was moderated: **{viol_summary}**. Review server rules."
            try:
                delete_delay = await get_guild_config(message.guild.id, "in_channel_warning_delete_delay", bot_config.in_channel_warning_delete_delay)
                await message.channel.send(warn_text, delete_after=delete_delay)
            except discord.Forbidden as e:
                logger.error(f"Failed to send warning to channel {message.channel.id}: {e}")
        if isinstance(message.author, discord.Member):
            await process_infractions_and_punish(message.author, message.guild, list(violations), message.content, message.jump_url)
    def check_dynamic_rules(self, raw_content: str, cleaned_content: str) -> set[str]:
        violations = set()
        for pattern_re in dynamic_rules["forbidden_regex"]:
            if pattern_re.search(raw_content):
                violations.add("dynamic_rule_violation")
                logger.debug(f"Dynamic Regex Violation: '{pattern_re.pattern}'")
                return violations
        if any(word in dynamic_rules["forbidden_words"] for word in cleaned_content.split()):
            violations.add("dynamic_rule_violation")
        elif any(phrase in cleaned_content for phrase in dynamic_rules["forbidden_phrases"]):
            violations.add("dynamic_rule_violation")
        return violations
    def check_spam(self, user_id: int, guild_id: int, cleaned_content: str) -> set[str]:
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
            is_repetitive = False
            if len(history) >= bot_config.spam_repetition_history_count - 1:
                similar_count = sum(1 for old_msg in history if fuzz.ratio(cleaned_content, old_msg) >= bot_config.spam_repetition_fuzzy_threshold)
                if similar_count >= bot_config.spam_repetition_history_count - 1:
                    is_repetitive = True
            if is_repetitive:
                violations.add("spam_repetition")
                history.clear()
            history.append(cleaned_content)
        return violations
    async def check_advertising(self, raw_content: str, guild_id: int) -> set[str]:
        violations = set()
        if bot_config.forbidden_text_pattern.search(raw_content):
            violations.add("advertising_forbidden_text")
            return violations
        guild_permitted_domains = await get_guild_config(guild_id, "permitted_domains", bot_config.permitted_domains)
        urls = re.findall(bot_config.url_pattern, raw_content)
        for url_match in urls:
            url = url_match[0] if isinstance(url_match, tuple) and url_match[0] else url_match
            try:
                schemed_url = url if url.startswith(('http://', 'https://')) else 'http://' + url
                domain = urlparse(schemed_url).netloc.lower().lstrip('www.')
                if domain and not any(allowed_domain == domain or domain.endswith('.' + allowed_domain) for allowed_domain in guild_permitted_domains):
                    violations.add("advertising_unpermitted_url")
                    logger.debug(f"Unpermitted URL: '{domain}'")
                    return violations
            except Exception as e:
                logger.warning(f"Could not parse URL '{url}': {e}")
        return violations
    def check_message_limits(self, message: discord.Message) -> set[str]:
        v = set()
        if len(message.mentions) > bot_config.mention_limit: v.add("excessive_mentions")
        if len(message.attachments) > bot_config.max_attachments: v.add("excessive_attachments")
        if len(message.content) > bot_config.max_message_length: v.add("long_message")
        return v
    async def check_language(self, guild_id: int, channel_id: int, raw_content: str) -> set[str]:
        if len(raw_content.split()) < bot_config.min_msg_len_for_lang_check:
            logger.debug(f"Message too short for language check: '{raw_content[:50]}...'")
            return set()
        lang_config = await self.get_effective_channel_language_config(guild_id, channel_id)
        if not lang_config or "any" in lang_config:
            logger.debug(f"Language check skipped: Config allows any language in channel {channel_id}.")
            return set()
        lang_code, confidence = await detect_language_ai(raw_content)
        if not lang_code or lang_code in lang_config:
            logger.debug(f"Language allowed: Detected '{lang_code}' in channel {channel_id} with config {lang_config}.")
            return set()
        threshold = bot_config.min_confidence_short_msg_lang if len(raw_content) < bot_config.short_msg_threshold_lang else bot_config.min_confidence_for_lang_flagging
        if confidence >= threshold:
            if not any(word in raw_content.lower() for word in bot_config.common_safe_foreign_words):
                logger.info(f"Language violation: Detected '{lang_code}' ({confidence:.2f}) where {lang_config} required in channel {channel_id}.")
                return {"foreign_language"}
        logger.debug(f"Language not flagged: '{lang_code}' ({confidence:.2f}) below threshold {threshold}.")
        return set()
    def check_keyword_violations(self, cleaned_content: str) -> set[str]:
        violations = set()
        if any(word in cleaned_content.split() for word in discrimination_words_set) or \
           any(fuzz.partial_ratio(phrase, cleaned_content) >= bot_config.fuzzy_match_threshold_keywords for phrase in discrimination_phrases):
            violations.add("discrimination")
        if any(word in cleaned_content.split() for word in nsfw_text_words_set) or \
           any(fuzz.partial_ratio(phrase, cleaned_content) >= bot_config.fuzzy_match_threshold_keywords for phrase in nsfw_text_phrases):
            violations.add("nsfw_text")
        return violations
    async def check_ai_text_moderation(self, content: str, user_id: int, guild_id: int) -> tuple[set[str], str | None]:
        violations = set()
        proactive_reason = None
        now_ts = datetime.now(timezone.utc).timestamp()
        if now_ts < self.openai_cooldowns.get(user_id, 0):
            logger.debug(f"OpenAI check skipped for user {user_id} due to cooldown.")
            return violations, proactive_reason
        self.openai_cooldowns[user_id] = now_ts + bot_config.openai_api_cooldown_seconds
        try:
            result = await check_openai_moderation_api(content)
            if result.get("flagged"):
                categories = result.get("categories", {})
                severe_categories = {
                    "harassment", "harassment/threatening", "hate", "hate/threatening",
                    "self-harm", "self-harm/intent", "self-harm/instructions",
                    "sexual/minors", "violence/graphic"
                }
                if any(categories.get(cat) for cat in severe_categories):
                    violations.add("openai_flagged_severe")
                else:
                    violations.add("openai_flagged_moderate")
            else:
                threshold = await get_guild_config(guild_id, "proactive_flagging_openai_threshold", bot_config.proactive_flagging_openai_threshold)
                scores = result.get("category_scores", {})
                if scores and max(scores.values()) >= threshold:
                    flagged_category = max(scores, key=scores.get)
                    proactive_reason = f"Proactive OpenAI Flag ({flagged_category.replace('/', ' ')}: {scores[flagged_category]:.2f})"
        except Exception as e:
            logger.error(f"OpenAI moderation failed for user {user_id}: {e}", exc_info=True)
        return violations, proactive_reason
    async def check_ai_media_moderation(self, attachments: list[discord.Attachment], guild_id: int) -> set[str]:
        if not (SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET): return set()
        violations = set()
        nudity_sa_thresh = await get_guild_config(guild_id, "sightengine_nudity_sexual_activity_threshold", bot_config.sightengine_nudity_sexual_activity_threshold)
        nudity_s_thresh = await get_guild_config(guild_id, "sightengine_nudity_suggestive_threshold", bot_config.sightengine_nudity_suggestive_threshold)
        gore_thresh = await get_guild_config(guild_id, "sightengine_gore_threshold", bot_config.sightengine_gore_threshold)
        offensive_thresh = await get_guild_config(guild_id, "sightengine_offensive_symbols_threshold", bot_config.sightengine_offensive_symbols_threshold)
        for attach in attachments:
            if attach.content_type and (attach.content_type.startswith("image/") or attach.content_type.startswith("video/")):
                try:
                    result = await check_sightengine_media_api(attach.url)
                    if not result: continue
                    if result.get("nudity", {}).get("sexual_activity", 0) >= nudity_sa_thresh or \
                       result.get("nudity", {}).get("suggestive", 0) >= nudity_s_thresh:
                        violations.add("nsfw_media")
                    if result.get("gore", {}).get("prob", 0) >= gore_thresh:
                        violations.add("gore_violence_media")
                    if result.get("offensive", {}).get("prob", 0) >= offensive_thresh:
                        violations.add("offensive_symbols_media")
                    if violations:
                        logger.debug(f"Sightengine Flagged: {attach.filename}. Result: {result}")
                        return violations
                except Exception as e:
                    logger.error(f"Sightengine failed for {attach.url}: {e}", exc_info=True)
        return violations
    async def add_to_review_queue(self, message: discord.Message, reason: str):
        if not db_conn: return
        try:
            async with db_conn.cursor() as cursor:
                await cursor.execute(
                    "INSERT OR IGNORE INTO review_queue (guild_id, user_id, channel_id, message_id, message_content, reason, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (message.guild.id, message.author.id, message.channel.id, message.id, message.content, reason, datetime.now(timezone.utc).isoformat())
                )
                if cursor.rowcount > 0:
                    await db_conn.commit()
                    logger.info(f"Message {message.id} added to review queue. Reason: {reason}")
                    review_channel_id = await get_guild_config(message.guild.id, "review_channel_id", bot_config.default_review_channel_id)
                    if review_channel_id and (review_channel := self.bot.get_channel(review_channel_id)):
                        embed = discord.Embed(title="🚨 Message Flagged for Review", description=f"**Reason:** {reason}", color=discord.Color.yellow(), timestamp=message.created_at)
                        embed.add_field(name="Author", value=f"{message.author.mention} (`{message.author.id}`)")
                        embed.add_field(name="Channel", value=f"{message.channel.mention}")
                        embed.add_field(name="Context", value=f"[Jump to Message]({message.jump_url})", inline=False)
                        embed.add_field(name="Content", value=f"```\n{discord.utils.escape_markdown(message.content[:500])}\n```", inline=False)
                        await review_channel.send(embed=embed)
        except Exception as e:
            logger.error(f"Failed to add message {message.id} to review queue: {e}", exc_info=True)
    mod_group = app_commands.Group(name="mod", description="Moderation commands.")
    @mod_group.command(name="infractions", description="View a user's recent infractions.")
    @app_commands.default_permissions(manage_messages=True)
    async def infractions_command(self, interaction: discord.Interaction, member: discord.Member, days: app_commands.Range[int, 1, 365] = 90):
        if not interaction.guild_id: return
        infraction_list, active_points = await get_user_infractions_from_db(member.id, interaction.guild_id, days)
        embed = discord.Embed(title=f"Infraction Report for {member.display_name}", color=discord.Color.red() if active_points > 20 else discord.Color.orange())
        embed.set_thumbnail(url=member.display_avatar.url)
        embed.add_field(name="Active Points (Last 90 Days)", value=f"**{active_points}**", inline=False)
        if infraction_list:
            history_str = []
            for infra in infraction_list:
                ts = datetime.fromisoformat(infra['timestamp'])
                entry = f"**ID:** `{infra['id']}` | **{infra['points']} pts** - {infra['violation_type'].replace('_', ' ').title()}\n> <t:{int(ts.timestamp())}:R>\n"
                if len("\n".join(history_str) + entry) > 1000:
                    history_str.append("...")
                    break
                history_str.append(entry)
            embed.add_field(name=f"Recent Infractions (Last {days} Days)", value="\n".join(history_str), inline=False)
        await interaction.response.send_message(embed=embed, ephemeral=True)
    @app_commands.command(name="clear_infractions", description="Clears ALL or specific infraction for a user.")
    @app_commands.default_permissions(administrator=True)
    @app_commands.describe(member="Member to clear infractions.", infraction_id="Optional ID of specific infraction.")
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
            await interaction.response.send_message(f"Cleared {action_desc} for {member.mention}.", ephemeral=True)
            await log_moderation_action("infractions_cleared", member, f"Moderator {interaction.user.mention} cleared {action_desc}.", interaction.user, interaction.guild, discord.Color.green())
        else:
            await interaction.response.send_message(f"Failed to clear {action_desc}.", ephemeral=True)
    async def _manual_action(self, interaction: discord.Interaction, member: discord.Member, action_key: str, reason: str, duration_value: float | None = None, duration_unit: str | None = None):
        if not interaction.guild_id:
            await interaction.response.send_message("This command must be used in a server.", ephemeral=True)
            return
        if member.id == self.bot.user.id:
            await interaction.response.send_message("I cannot moderate myself!", ephemeral=True)
            return
        if member.id == interaction.user.id and action_key != "warn":
            await interaction.response.send_message("You cannot apply this action to yourself.", ephemeral=True)
            return
        if isinstance(member, discord.Member) and interaction.guild.owner_id != interaction.user.id and member.top_role >= interaction.user.top_role:
            await interaction.response.send_message("You cannot moderate a member with equal or higher role.", ephemeral=True)
            return
        mock_action_config = {"action": action_key, "reason_suffix": reason, "dm_message": f"You have been {action_key}ed by a moderator. Reason: {reason}"}
        if duration_value and duration_unit:
            if duration_unit == "hours":
                mock_action_config["duration_hours"] = duration_value
            elif duration_unit == "days":
                mock_action_config["duration_days"] = duration_value
        await apply_moderation_punishment(member, mock_action_config, 0, f"Manual Action: {reason}", moderator=interaction.user)
        duration_str = f" for {duration_value} {duration_unit}" if duration_value and duration_unit else ""
        await interaction.response.send_message(f"Applied **{action_key.upper()}** to {member.mention}{duration_str}. Reason: {reason}", ephemeral=True)
    @app_commands.command(name="manual_warn", description="Manually warn a member.")
    @app_commands.default_permissions(kick_members=True)
    @app_commands.describe(member="Member to warn.", reason="Reason for warning.")
    async def manual_warn_command(self, interaction: discord.Interaction, member: discord.Member, reason: str):
        await self._manual_action(interaction, member, "warn", reason)
    @app_commands.command(name="manual_mute", description="Manually mute a member.")
    @app_commands.default_permissions(moderate_members=True)
    @app_commands.describe(member="Member to mute.", duration_hours="Mute duration in hours.", reason="Reason for mute.")
    async def manual_mute_command(self, interaction: discord.Interaction, member: discord.Member, duration_hours: app_commands.Range[float, 0.01], reason: str):
        await self._manual_action(interaction, member, "mute", reason, duration_hours, "hours")
    @app_commands.command(name="manual_kick", description="Manually kick a member.")
    @app_commands.default_permissions(kick_members=True)
    @app_commands.describe(member="Member to kick.", reason="Reason for kick.")
    async def manual_kick_command(self, interaction: discord.Interaction, member: discord.Member, reason: str):
        await self._manual_action(interaction, member, "kick", reason)
    @app_commands.command(name="manual_ban", description="Manually ban a member (temporarily or permanently).")
    @app_commands.default_permissions(ban_members=True)
    @app_commands.describe(user="User to ban (ID if not in server).", reason="Reason for ban.", duration_days="Optional: Duration in days for temp ban.")
    async def manual_ban_command(self, interaction: discord.Interaction, user: discord.User, reason: str, duration_days: app_commands.Range[float, 0.01] | None = None):
        if not interaction.guild:
            await interaction.response.send_message("This command must be used in a server.", ephemeral=True)
            return
        target_member = interaction.guild.get_member(user.id)
        action_type = "temp_ban" if duration_days else "ban"
        full_reason = f"Manual {action_type} by {interaction.user.name}: {reason}"
        dm_message_text = f"Hello {user.name},\n\nRegarding **{interaction.guild.name}**:\n\nYou have been {action_type}ed. Reason: {reason}"
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
                dm_message_text += f"\n\nDuration: **{str(duration)}**."
                extra_log_fields.append(("Unban Time", f"{discord.utils.format_dt(unban_time, 'R')}"))
            try:
                await user.send(dm_message_text)
            except discord.Forbidden:
                logger.warning(f"Could not DM ban notification to user {user.id}.")
            await interaction.guild.ban(user, reason=full_reason, delete_message_seconds=0)
            await log_moderation_action(action_type, user, full_reason, interaction.user, interaction.guild, log_color, extra_log_fields)
            duration_str = f" for {duration_days} days" if duration_days else " permanently"
            await interaction.response.send_message(f"Banned {user.mention} (`{user.id}`){duration_str}. Reason: {reason}", ephemeral=True)
        except discord.Forbidden:
            await interaction.response.send_message(f"Failed to ban {user.mention}: Missing permissions.", ephemeral=True)
        except discord.HTTPException as e:
            await interaction.response.send_message(f"Failed to ban {user.mention}: Discord API error {e.status}.", ephemeral=True)
        except Exception as e:
            await interaction.response.send_message(f"An unexpected error occurred while banning {user.mention}.", ephemeral=True)
            logger.error(f"Error in manual_ban for user ID {user.id}: {e}", exc_info=True)
    @app_commands.command(name="manual_unban", description="Manually unban a user by their ID.")
    @app_commands.default_permissions(ban_members=True)
    @app_commands.describe(user_id="ID of user to unban.", reason="Reason for unban.")
    async def manual_unban_command(self, interaction: discord.Interaction, user_id: str, reason: str):
        if not interaction.guild:
            await interaction.response.send_message("This command must be used in a server.", ephemeral=True)
            return
        try:
            uid = int(user_id)
            user_obj = discord.Object(id=uid)
        except ValueError:
            await interaction.response.send_message("Invalid User ID. Please provide a numerical ID.", ephemeral=True)
            return
        full_reason = f"Manual unban by {interaction.user.name}: {reason}"
        try:
            try:
                await interaction.guild.fetch_ban(user_obj)
            except discord.NotFound:
                await interaction.response.send_message(f"User ID {uid} is not banned.", ephemeral=True)
                return
            await interaction.guild.unban(user_obj, reason=full_reason)
            await remove_temp_ban_from_db(uid, interaction.guild.id)
            try:
                target_user_for_log = await self.bot.fetch_user(uid)
            except discord.NotFound:
                target_user_for_log = discord.Object(id=uid)
            await log_moderation_action("unban_manual", target_user_for_log, full_reason, interaction.user, interaction.guild, discord.Color.green())
            await interaction.response.send_message(f"Unbanned User ID `{uid}`. Reason: {reason}", ephemeral=True)
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

class ReportMessageModal(discord.ui.Modal, title="Report Message to Moderators"):
    def __init__(self, message: discord.Message):
        super().__init__()
        self.message = message
    reason_input = discord.ui.TextInput(
        label="Reason for reporting this message",
        style=discord.TextStyle.paragraph,
        placeholder="e.g., 'Spam', 'Hate Speech', 'Offensive content'",
        required=True,
        max_length=500,
    )
    async def on_submit(self, interaction: discord.Interaction):
        if not db_conn or not self.message.guild: return
        report_reason = self.reason_input.value.strip()
        try:
            mod_cog = bot.get_cog("Moderation")
            if mod_cog and hasattr(mod_cog, 'add_to_review_queue'):
                await mod_cog.add_to_review_queue(self.message, f"User Report: {report_reason}")
                logger.info(f"User {interaction.user.id} reported message {self.message.id} for: {report_reason}")
            else:
                async with db_conn.cursor() as cursor:
                    await cursor.execute(
                        "INSERT OR IGNORE INTO review_queue (guild_id, user_id, channel_id, message_id, message_content, reason, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (self.message.guild.id, self.message.author.id, self.message.channel.id, self.message.id, self.message.content, f"User Report by {interaction.user.name}: {report_reason}", datetime.now(timezone.utc).isoformat())
                    )
                    await db_conn.commit()
            await interaction.response.send_message("✅ Your report has been submitted. Thank you!", ephemeral=True)
        except Exception as e:
            await interaction.response.send_message(f"An error occurred while submitting your report: {e}", ephemeral=True)
            logger.error(f"Error submitting user report for message {self.message.id}: {e}", exc_info=True)

class AddRuleModal(discord.ui.Modal, title="Add New Violation Rule"):
    def __init__(self, review_id: int):
        super().__init__()
        self.review_id = review_id
    rule_type = discord.ui.TextInput(label="Rule Type", placeholder="e.g., forbidden_word, forbidden_phrase, forbidden_regex")
    pattern = discord.ui.TextInput(label="Pattern", style=discord.TextStyle.short, placeholder="Enter the word, phrase, or regex pattern.")
    async def on_submit(self, interaction: discord.Interaction):
        if not db_conn or not interaction.guild_id: return
        rule_type_val = self.rule_type.value.strip()
        pattern_val = self.pattern.value.strip()
        if rule_type_val not in ["forbidden_word", "forbidden_phrase", "forbidden_regex"]:
            await interaction.response.send_message("Invalid rule type. Use 'forbidden_word', 'forbidden_phrase', or 'forbidden_regex'.", ephemeral=True)
            return
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
    def __init__(self, review_id: int, member: discord.Member | discord.User | None, message_id: int):
        super().__init__(timeout=300)
        self.review_id = review_id
        self.member = member
        self.message_id = message_id
    @discord.ui.button(label="Punish & Add Rule", style=discord.ButtonStyle.danger, emoji="⚖️")
    async def punish_and_add_rule(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self.member or not isinstance(self.member, discord.Member):
            await interaction.response.send_message("Cannot punish user, they may have left the server. You can still add the rule.", ephemeral=True)
        modal = AddRuleModal(review_id=self.review_id)
        await interaction.response.send_modal(modal)
    @discord.ui.button(label="Reject (Safe)", style=discord.ButtonStyle.success, emoji="✅")
    async def reject_as_safe(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not db_conn: return
        try:
            async with db_conn.cursor() as cursor:
                await cursor.execute("DELETE FROM review_queue WHERE id = ?", (self.review_id,))
                await db_conn.commit()
            await interaction.response.edit_message(content=f"✅ Review item `{self.review_id}` marked as safe.", view=None)
            await log_moderation_action("review_rejected_safe", interaction.user, f"Review item `{self.review_id}` marked as safe.", guild=interaction.guild, color=discord.Color.green())
        except Exception as e:
            await interaction.response.send_message(f"An error occurred: {e}", ephemeral=True)
            logger.error(f"Error rejecting review item: {e}", exc_info=True)
    @discord.ui.button(label="Ignore", style=discord.ButtonStyle.secondary, emoji="✖️")
    async def ignore_item(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="This review item was ignored.", view=None)

@bot.event
async def setup_hook():
    logger.info("Running setup_hook: Initializing bot components...")
    global http_session, LANGUAGE_MODEL, db_conn
    if not http_session or http_session.closed:
        http_session = ClientSession(connector=aiohttp.TCPConnector(ssl=False))
        logger.info("Aiohttp ClientSession initialized.")
    if not LANGUAGE_MODEL:
        if os.path.exists(FASTTEXT_MODEL_PATH):
            try:
                LANGUAGE_MODEL = fasttext.load_model(FASTTEXT_MODEL_PATH)
                logger.info(f"FastText model loaded from {FASTTEXT_MODEL_PATH}")
            except Exception as e:
                logger.critical(f"Failed to load FastText model: {e}. Language detection disabled.", exc_info=True)
        else:
            logger.error(f"FastText model not found at {FASTTEXT_MODEL_PATH}. Language detection disabled.")
    if not db_conn:
        await setup_db()
    await load_dynamic_rules_from_db()
    await bot.add_cog(GeneralCog(bot))
    await bot.add_cog(ConfigurationCog(bot))
    await bot.add_cog(ModerationCog(bot))
    logger.info("All Cogs loaded.")
    try:
        synced_commands = await bot.tree.sync()
        logger.info(f"Synced {len(synced_commands)} application commands.")
    except discord.Forbidden:
        logger.error("Bot lacks 'applications.commands' scope to sync slash commands.")
    except Exception as e:
        logger.error(f"Failed to sync slash commands: {e}", exc_info=True)

@bot.event
async def on_ready():
    await bot.tree.sync()
    logger.info("-" * 40)
    logger.info(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    logger.info(f"Discord.py Version: {discord.__version__}")
    logger.info(f"{bot.user.name} is online and ready! 🚀")
    logger.info("-" * 40)

async def health_check_handler(request):
    db_status = "OK" if db_conn else "Error"
    lang_model_status = "OK" if LANGUAGE_MODEL else "Error"
    http_session_status = "OK" if http_session and not http_session.closed else "Error"
    status_text = (f"{bot.user.name} is running!\n"
                   f"Latency: {round(bot.latency * 1000)}ms\n"
                   f"DB: {db_status}\n"
                   f"LangModel: {lang_model_status}\n"
                   f"HTTPSession: {http_session_status}\n"
                   f"Discord API OK: {bot.is_ready()}")
    return web.Response(text=status_text, content_type="text/plain")

async def main_async_runner():
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
        logger.info("PORT not set. Health check web server not started.")
    try:
        await bot.start(DISCORD_TOKEN)
    finally:
        logger.info("Bot is shutting down.")
        if runner:
            await runner.cleanup()
            logger.info("Web server runner cleaned up.")
        if http_session and not http_session.closed:
            await http_session.close()
            logger.info("Aiohttp ClientSession closed.")
        if db_conn:
            await db_conn.close()
            logger.info("Database connection closed.")
        await bot.close()

if __name__ == "__main__":
    try:
        if not discord.opus.is_loaded():
            try:
                discord.opus.load_opus('libopus.so')
                logger.info("Opus library loaded successfully.")
            except OSError:
                logger.warning("Could not load opus library. Voice functionality may not work.")
        else:
            logger.info("Opus library already loaded.")
        logger.info("Starting bot asynchronously...")
        asyncio.run(main_async_runner())
    except KeyboardInterrupt:
        logger.info("Shutdown requested via KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"💥 UNHANDLED EXCEPTION IN TOP LEVEL: {e}", exc_info=True)
