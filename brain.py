import asyncio
import json
import logging
import os
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger('discord_bot_v3')

DISCORD_TOKEN = os.getenv("ADROIT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FASTTEXT_MODEL_PATH = os.getenv("FASTTEXT_MODEL_PATH", "lid.176.ftz")
SIGHTENGINE_API_USER = os.getenv("SIGHTENGINE_API_USER")
SIGHTENGINE_API_SECRET = os.getenv("SIGHTENGINE_API_SECRET")

if not DISCORD_TOKEN:
    logger.critical("CRITICAL: ADROIT_TOKEN environment variable not set. Exiting.")
    exit(1)

intents = discord.Intents.default()
intents.members = True
intents.message_content = True
intents.presences = False

bot = commands.Bot(command_prefix=">>", intents=intents, help_command=None)

db_conn: aiosqlite.Connection | None = None
LANGUAGE_MODEL: fasttext.FastText._FastText | None = None 
http_session: ClientSession | None = None

class BotConfig:
    """Holds all static configurations for the bot."""
    def __init__(self):
        self.default_log_channel_id = 1113377818424922132
        self.default_channel_configs = { 
            1113377809440722974: {"language": ["en"]},
            1322517478365990984: {"language": ["en"], "topics": ["politics"]}, 
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
                "discrimination": {"points": 2, "severity": "Low"},
                "spam_rate": {"points": 2, "severity": "Low"},
                "spam_repetition": {"points": 3, "severity": "Medium"}, 
                "nsfw_text": {"points": 2, "severity": "Low"},
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
                "offensive_symbols_media": {"points": 4, "severity": "Medimum"}
            }
        }
        self.spam_window_seconds = 10
        self.spam_message_limit = 5 
        self.spam_repetition_history_count = 3
        self.spam_repetition_fuzzy_threshold = 85 
        self.mention_limit = 7 
        self.max_message_length = 1500 
        self.max_attachments = 5
        self.min_msg_len_for_lang_check = 5
        self.min_confidence_for_lang_flagging = 0.70 
        self.min_confidence_short_msg_lang = 0.80 
        self.short_msg_threshold_lang = 25
        self.common_safe_foreign_words = {"bonjour", "hola", "merci", "gracias", "oui", "si", "nyet", "da", "salut", "ciao", "hallo", "guten tag", "privet", "konnichiwa", "arigato", "sawasdee", "namaste"}
        self.fuzzy_match_threshold_keywords = 88 

        self.sightengine_nudity_sexual_activity_threshold = 0.6 
        self.sightengine_nudity_suggestive_threshold = 0.8 
        self.sightengine_gore_threshold = 0.7 
        self.sightengine_offensive_symbols_threshold = 0.85 

        self.delete_violating_messages = True
        self.send_in_channel_warning = True
        self.in_channel_warning_delete_delay = 20 

bot_config = BotConfig()

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
nsfw_text_words_set, nsfw_text_phrases = load_terms_from_file('nsfw_text_terms.txt')

def clean_message_content(text: str) -> str:
    """Cleans and normalizes message content for analysis."""
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
    """Detect the language of the given text using FastText. Returns (lang_code | None, confidence_score)."""
    clean_text = text.strip() 
    if not clean_text:
        return None, 0.0

    if not LANGUAGE_MODEL:
        logger.warning("FastText model not loaded. Cannot detect language.")
        return None, 0.0
    try:
        prediction = LANGUAGE_MODEL.predict(clean_text.replace('\n', ' '), k=1)
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
    if isinstance(target_user, discord.Member) and target_user.avatar:
        embed.set_thumbnail(url=target_user.avatar.url)
    elif target_user.avatar:
         embed.set_thumbnail(url=target_user.avatar.url)


    if moderator:
        embed.add_field(name="Moderator", value=f"{moderator.mention} (`{moderator.id}`)", inline=True)
    else:
        embed.add_field(name="Moderator", value="Automated Action", inline=True)

    if message_url:
        embed.add_field(name="Context", value=f"[Jump to Message]({message_url})", inline=False)

    if extra_fields:
        for name, value in extra_fields:
            embed.add_field(name=name, value=value, inline=False) 

    if log_channel:
        try:
            await log_channel.send(embed=embed)
        except discord.Forbidden:
            logger.error(f"Missing permissions to send logs to channel #{log_channel.name} ({log_channel.id}) in guild {current_guild.name}.")
        except Exception as e:
            logger.error(f"Error sending log embed to channel for guild {current_guild.name}: {e}", exc_info=True)
    else:
        log_message_console = (f"Log (Guild: {current_guild.name}, Action: {action_type}, Target: {target_user.id}, Reason: {reason}) "
                               f"- Log channel ID {log_channel_id} not found or not configured.")
        if extra_fields:
            log_message_console += f" Extra: {extra_fields}"
        logger.info(log_message_console)


def retry_if_api_error(exception):
    """Retries on server errors (5xx), rate limits (429), or network issues for API calls."""
    if isinstance(exception, client_exceptions.ClientResponseError):
        return exception.status == 429 or exception.status >= 500 
    return isinstance(exception, (asyncio.TimeoutError, client_exceptions.ClientOSError, client_exceptions.ClientConnectorError))


@retry(
    stop=stop_after_attempt(4), 
    wait=wait_random_exponential(multiplier=1, min=3, max=60), 
    retry=retry_if_api_error,
    reraise=True 
)
async def check_openai_moderation_api(text_content: str) -> dict:
    """Checks text against the OpenAI moderation API with robust retries."""
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
        logger.critical(f"OpenAI moderation failed definitively for text: '{text_content[:100]}...'. Moderation skipped.")
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

    models = "nudity-2.0,gore,offensive,properties,text-content" 
    url = f"https://api.sightengine.com/1.0/check.json?url={image_url}&models={models}&api_user={SIGHTENGINE_API_USER}&api_secret={SIGHTENGINE_API_SECRET}"
    
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
        logger.critical(f"Sightengine moderation failed definitively for URL: '{image_url}'. Moderation skipped.")
        return {} # Fallback

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
    full_reason = f"[{action_type.upper()}] {reason_suffix} (Violations: {violation_summary})"
    if moderator: 
        full_reason = f"[{action_type.upper()}] Manual action by {moderator.name}: {reason_suffix}"


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
                dm_message_text += f"\n\nYou will be unbanned automatically around: {unban_time.strftime('%Y-%m-%d %H:%M:%S UTC')}."
                log_color = discord.Color.dark_red()
                extra_log_fields.append(("Unban Time", unban_time.strftime('%Y-%m-%d %H:%M:%S UTC')))
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
        await log_moderation_action(f"{action_type}_failed_http_error", member, f"Discord API error: {e.status} - {e.text}. Original reason: {full_reason}", moderator, guild, discord.Color.red(), extra_log_fields)
    except Exception as e:
        logger.error(f"Unexpected error applying {action_type} to {member.display_name} ({member.id}): {e}", exc_info=True)
        await log_moderation_action(f"{action_type}_failed_unknown", member, f"Unexpected error: {e}. Original reason: {full_reason}", moderator, guild, discord.Color.red(), extra_log_fields)


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
                logger.info(f"Recorded infraction for user {user_id}: {violation_type} (+{points} points). Message: {message_url or 'N/A'}")
            except Exception as e:
                logger.error(f"Failed to record infraction {violation_type} for user {user_id} in DB: {e}", exc_info=True)
        
        if not violation_summary_parts: 
            await db_conn.commit() 
            return

        await db_conn.commit() 

        thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
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

            if applicable_punishment_config["action"] in ["kick", "ban", "temp_ban"]:
                logger.info(f"Clearing infractions for user {user_id} in guild {guild_id} due to '{applicable_punishment_config['action']}' action.")
                await clear_user_infractions(user_id, guild_id, cursor) 
                await db_conn.commit() 

async def get_user_infractions_from_db(user_id: int, guild_id: int, days_limit: int = 0) -> tuple[list[dict], int]:
    """
    Retrieves infractions for a user, optionally limited by days.
    Returns a list of infraction dicts and the total active points (within 30 days).
    """
    if not db_conn: return [], 0
    infractions = []
    total_active_points = 0
    try:
        async with db_conn.execute(
            f"SELECT id, violation_type, points, message_content_snippet, message_url, timestamp FROM infractions WHERE user_id = ? AND guild_id = ? {'AND timestamp >= ?' if days_limit > 0 else ''} ORDER BY timestamp DESC",
            (user_id, guild_id, (datetime.now(timezone.utc) - timedelta(days=days_limit)).isoformat()) if days_limit > 0 else (user_id, guild_id)
        ) as cursor:
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
    """Clears all infractions for a user in a specific guild. Uses provided cursor or creates one."""
    if not db_conn: return
    try:
        if cursor:
            await cursor.execute("DELETE FROM infractions WHERE user_id = ? AND guild_id = ?", (user_id, guild_id))
        else:
            async with db_conn.execute("DELETE FROM infractions WHERE user_id = ? AND guild_id = ?", (user_id, guild_id)) as cur:
                await db_conn.commit() 
        logger.info(f"All infractions cleared for user {user_id} in guild {guild_id}.")
    except Exception as e:
        logger.error(f"Error clearing infractions for user {user_id} in guild {guild_id}: {e}", exc_info=True)

async def remove_specific_infraction_from_db(infraction_id: int):
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
    if not db_conn: return
    try:
        async with db_conn.execute("DELETE FROM temp_bans WHERE user_id = ? AND guild_id = ?", (user_id, guild_id)):
            await db_conn.commit()
        logger.info(f"Temp ban DB entry removed for user {user_id} in guild {guild_id}.")
    except Exception as e:
        logger.error(f"Error removing temp ban DB entry for user {user_id} in guild {guild_id}: {e}", exc_info=True)


async def setup_db():
    """Initializes the SQLite database and creates necessary tables."""
    global db_conn
    try:
        db_conn = await aiosqlite.connect('bot_data_v3.db')
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
                message_content_snippet TEXT, -- Store a snippet of the message
                message_url TEXT,             -- Link to the message if available
                timestamp TEXT NOT NULL       -- ISO format UTC timestamp
            )
        """)
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_infractions_user_guild_time ON infractions (user_id, guild_id, timestamp)')

        await db_conn.execute("""
            CREATE TABLE IF NOT EXISTS temp_bans (
                user_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL,
                unban_time TEXT NOT NULL,    -- ISO format UTC timestamp for when to unban
                ban_reason TEXT,
                PRIMARY KEY (user_id, guild_id) -- User can only have one active temp ban per guild
            )
        """)
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_temp_bans_unban_time ON temp_bans (unban_time)')
        
        await db_conn.commit()
        logger.info("Database initialized and tables checked/created successfully.")
    except Exception as e:
        logger.critical(f"Failed to connect to or initialize database: {e}", exc_info=True)
        if db_conn: await db_conn.close() 
        exit(1) 


class GeneralCog(commands.Cog, name="General"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="ping", description="Checks the bot's latency.")
    async def ping(self, interaction: discord.Interaction):
        latency = round(self.bot.latency * 1000)
        await interaction.response.send_message(f"Pong! Latency: {latency}ms", ephemeral=True)

    @app_commands.command(name="help", description="Shows information about bot commands.")
    async def help_command(self, interaction: discord.Interaction):
        embed = discord.Embed(title="Bot Help", description="Here are the available commands:", color=discord.Color.blue())
        
        for cog_name, cog in self.bot.cogs.items():
            command_list = []
            if hasattr(cog, 'get_app_commands'):
                 for cmd in cog.get_app_commands():
                    if isinstance(cmd, app_commands.Command):
                         command_list.append(f"`/{cmd.name}` - {cmd.description}")

            if command_list:
                 embed.add_field(name=f"**{cog_name} Commands**", value="\n".join(command_list), inline=False)
        
        if not embed.fields:
            embed.description = "No commands found or Cogs not loaded properly."

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
            await log_moderation_action("config_change_log_channel_set", interaction.user, f"Set log channel to {channel.mention}", guild=interaction.guild, color=discord.Color.blue())
        else:
            await set_guild_config(interaction.guild_id, "log_channel_id", None)
            await interaction.response.send_message("Moderation log channel has been cleared. Logs may use default or not send to a channel.", ephemeral=True)
            await log_moderation_action("config_change_log_channel_cleared", interaction.user, "Cleared log channel setting.", guild=interaction.guild, color=discord.Color.blue())


    @app_commands.command(name="set_channel_language", description="Sets expected language(s) for a channel (e.g., en,fr). 'any' to disable.")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(channel="The channel to configure.", languages="Comma-separated ISO 639-1 language codes (e.g., en,fr,es). Use 'any' to disable.")
    async def set_channel_language(self, interaction: discord.Interaction, channel: discord.TextChannel, languages: str):
        if not interaction.guild_id:
            await interaction.response.send_message("This command can only be used in a server.", ephemeral=True)
            return
        
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
                await interaction.response.send_message(f"Invalid language code: '{lang_code}'. Please use 2-letter ISO 639-1 codes (e.g., 'en', 'fr') or 'any'.", ephemeral=True)
                return

        config_key = f"channel_language_{channel.id}"
        if is_any:
            await set_guild_config(interaction.guild_id, config_key, ["any"]) 
            await interaction.response.send_message(f"Language checks for {channel.mention} have been set to allow **any** language.", ephemeral=True)
        elif valid_langs:
            await set_guild_config(interaction.guild_id, config_key, valid_langs)
            await interaction.response.send_message(f"Expected languages for {channel.mention} set to: **{', '.join(valid_langs)}**.", ephemeral=True)
        else: 
             await set_guild_config(interaction.guild_id, config_key, None)
             await interaction.response.send_message(f"No valid languages provided for {channel.mention}. Language configuration cleared/reset to default.", ephemeral=True)


    @app_commands.command(name="get_channel_config", description="Shows current language/topic config for a channel.")
    @app_commands.default_permissions(manage_messages=True)
    @app_commands.describe(channel="The channel to check.")
    async def get_channel_config(self, interaction: discord.Interaction, channel: discord.TextChannel):
        if not interaction.guild_id:
            await interaction.response.send_message("This command can only be used in a server.", ephemeral=True)
            return

        lang_config_key = f"channel_language_{channel.id}"
        db_lang_setting = await get_guild_config(interaction.guild_id, lang_config_key, None)
        
        final_lang_setting = None
        if db_lang_setting is not None: 
            final_lang_setting = db_lang_setting
        elif channel.id in bot_config.default_channel_configs and "language" in bot_config.default_channel_configs[channel.id]:
            final_lang_setting = bot_config.default_channel_configs[channel.id]["language"]

        lang_display = "Any (or not configured)"
        if final_lang_setting:
            if "any" in final_lang_setting: lang_display = "Any"
            else: lang_display = ", ".join(final_lang_setting).upper()

        embed = discord.Embed(title=f"Configuration for #{channel.name}", color=discord.Color.blurple())
        embed.add_field(name="Expected Language(s)", value=lang_display, inline=False)
        await interaction.response.send_message(embed=embed, ephemeral=True)


class ModerationCog(commands.Cog, name="Moderation"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.user_message_timestamps: defaultdict[int, defaultdict[int, deque]] = defaultdict(lambda: defaultdict(deque))
        self.user_message_history: defaultdict[int, defaultdict[int, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=bot_config.spam_repetition_history_count))
        )
        self.temp_ban_check_task.start()
        self.cleanup_old_infractions_task.start()
        self.cleanup_spam_trackers_task.start()

    def cog_unload(self):
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
        if default_conf and "language" in default_conf:
            return default_conf["language"]
        return None 

    @tasks.loop(minutes=1) 
    async def temp_ban_check_task(self):
        """Checks for expired temporary bans and unbans users."""
        if not db_conn:
            logger.warning("temp_ban_check_task: Database connection not available, skipping.")
            return

        now_utc = datetime.now(timezone.utc)
        expired_bans = await get_temp_bans_from_db()

        for ban_entry in expired_bans:
            user_id = ban_entry["user_id"]
            guild_id = ban_entry["guild_id"]
            unban_time = ban_entry["unban_time"]
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
                    
                    try:
                        target_user_for_log = await self.bot.fetch_user(user_id)
                    except discord.NotFound:
                        target_user_for_log = user_obj 
                        
                    await log_moderation_action("unban_temp_expired", target_user_for_log, unban_reason, guild=guild, color=discord.Color.green())
                
                except discord.NotFound: 
                    logger.info(f"User {user_id} not found in ban list of {guild.name} (possibly already unbanned). Removing temp ban entry.")
                    await remove_temp_ban_from_db(user_id, guild_id)
                except discord.Forbidden:
                    logger.error(f"Missing permissions to unban user {user_id} from {guild.name}.")
                    await log_moderation_action("unban_failed_permission", user_obj, f"Bot lacks permissions. Manual unban required. Original reason: {original_reason}", guild=guild, color=discord.Color.red())
                except Exception as e:
                    logger.error(f"Error unbanning user {user_id} from {guild.name}: {e}", exc_info=True)
                    await log_moderation_action("unban_failed_error", user_obj, f"Error during unban: {e}. Manual intervention may be needed. Original reason: {original_reason}", guild=guild, color=discord.Color.red())

    @temp_ban_check_task.before_loop
    async def before_temp_ban_check_task(self):
        await self.bot.wait_until_ready()
        logger.info("Starting temporary ban check loop.")

    @tasks.loop(hours=24) 
    async def cleanup_old_infractions_task(self):
        """Periodically deletes very old infraction records from the database for hygiene."""
        if not db_conn:
            logger.warning("cleanup_old_infractions_task: Database connection not available.")
            return
        
        ninety_days_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        try:
            async with db_conn.execute("DELETE FROM infractions WHERE timestamp < ?", (ninety_days_ago,)) as cursor:
                await db_conn.commit()
                if cursor.rowcount > 0:
                    logger.info(f"Database Hygiene: Deleted {cursor.rowcount} infraction records older than 90 days.")
        except Exception as e:
            logger.error(f"Error during old infraction cleanup task: {e}", exc_info=True)

    @cleanup_old_infractions_task.before_loop
    async def before_cleanup_old_infractions_task(self):
        await self.bot.wait_until_ready()
        logger.info("Starting daily old infraction cleanup task.")

    @tasks.loop(hours=6) 
    async def cleanup_spam_trackers_task(self):
        """Cleans up old entries from in-memory spam trackers."""
        now = datetime.now(timezone.utc)
        for guild_id in list(self.user_message_timestamps.keys()):
            for user_id in list(self.user_message_timestamps[guild_id].keys()):
                self.user_message_timestamps[guild_id][user_id] = deque(
                    ts for ts in self.user_message_timestamps[guild_id][user_id]
                    if (now - ts).total_seconds() < (bot_config.spam_window_seconds * 6) 
                )
                if not self.user_message_timestamps[guild_id][user_id]:
                    del self.user_message_timestamps[guild_id][user_id]
            if not self.user_message_timestamps[guild_id]:
                del self.user_message_timestamps[guild_id]
        
        logger.info("Periodic spam tracker cleanup completed.")

    @cleanup_spam_trackers_task.before_loop
    async def before_cleanup_spam_trackers_task(self):
        await self.bot.wait_until_ready()
        logger.info("Starting periodic spam tracker cleanup task.")


    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if not message.guild or message.author.bot or message.webhook_id:
            return 

        if isinstance(message.author, discord.Member) and message.author.guild_permissions.manage_messages:
            return await self.bot.process_commands(message) 

        content_raw = message.content
        cleaned_content_for_matching = clean_message_content(content_raw)
        
        member = message.author 
        guild = message.guild
        channel = message.channel
        user_id = member.id
        guild_id = guild.id
        
        violations_found_this_message = set() 

        now = datetime.now(timezone.utc)
        
        self.user_message_timestamps[guild_id][user_id].append(now)
        while self.user_message_timestamps[guild_id][user_id] and \
              (now - self.user_message_timestamps[guild_id][user_id][0]).total_seconds() > bot_config.spam_window_seconds:
            self.user_message_timestamps[guild_id][user_id].popleft()

        if len(self.user_message_timestamps[guild_id][user_id]) > bot_config.spam_message_limit:
            violations_found_this_message.add("spam_rate")
            logger.debug(f"Spam (Rate): User {user_id} in guild {guild_id} sent {len(self.user_message_timestamps[guild_id][user_id])} messages in window.")

        if cleaned_content_for_matching and "spam_rate" not in violations_found_this_message:
            history = self.user_message_history[guild_id][user_id]
            is_repetitive = False
            if len(history) == bot_config.spam_repetition_history_count: 
                similar_count = 0
                for old_msg_cleaned in history:
                    if fuzz.ratio(cleaned_content_for_matching, old_msg_cleaned) >= bot_config.spam_repetition_fuzzy_threshold:
                        similar_count += 1
                if similar_count >= bot_config.spam_repetition_history_count -1 : 
                    is_repetitive = True
            
            if is_repetitive:
                violations_found_this_message.add("spam_repetition")
                logger.debug(f"Spam (Repetition): User {user_id} in guild {guild_id} sent repetitive messages.")
                self.user_message_history[guild_id][user_id].clear() 
            
            self.user_message_history[guild_id][user_id].append(cleaned_content_for_matching)

        if bot_config.forbidden_text_pattern.search(content_raw):
            violations_found_this_message.add("advertising_forbidden_text")
            logger.debug(f"Forbidden Text: User {user_id} used forbidden pattern. Content: {content_raw[:100]}")

        if "advertising_forbidden_text" not in violations_found_this_message:
            potential_urls = re.findall(r"(?:https?://)?(?:www\.)?[\w\.-]+\.\w{2,}(?:/[\S]*)?", content_raw, re.IGNORECASE)
            guild_permitted_domains = await get_guild_config(guild_id, "permitted_domains", bot_config.permitted_domains)

            for purl in potential_urls:
                try:
                    schemed_url = purl if purl.startswith(('http://', 'https://')) else 'http://' + purl
                    parsed_url = urlparse(schemed_url)
                    domain = parsed_url.netloc.lower().lstrip('www.')
                    
                    if domain and not any(allowed_domain == domain or domain.endswith('.' + allowed_domain) for allowed_domain in guild_permitted_domains):
                        violations_found_this_message.add("advertising_unpermitted_url")
                        logger.debug(f"Unpermitted URL: User {user_id} posted URL with domain '{domain}'.")
                        break 
                except Exception as e:
                    logger.warning(f"Error parsing potential URL '{purl}': {e}")


        if len(message.mentions) > bot_config.mention_limit:
            violations_found_this_message.add("excessive_mentions")
        if len(message.attachments) > bot_config.max_attachments:
            violations_found_this_message.add("excessive_attachments")
        if len(content_raw) > bot_config.max_message_length: 
            violations_found_this_message.add("long_message")

        if len(cleaned_content_for_matching.split()) >= bot_config.min_msg_len_for_lang_check and \
           bot_config.has_alphanumeric_pattern.search(cleaned_content_for_matching):
            
            channel_lang_config = await self.get_effective_channel_language_config(guild_id, channel.id)

            if channel_lang_config and "any" not in channel_lang_config:
                lang_code, confidence = await detect_language_ai(cleaned_content_for_matching) 
                
                if lang_code and lang_code not in channel_lang_config:
                    is_safe_word = any(safe_word in cleaned_content_for_matching for safe_word in bot_config.common_safe_foreign_words)
                    
                    current_confidence_threshold = bot_config.min_confidence_for_lang_flagging
                    if len(cleaned_content_for_matching) < bot_config.short_msg_threshold_lang:
                        current_confidence_threshold = bot_config.min_confidence_short_msg_lang

                    if not is_safe_word and confidence >= current_confidence_threshold:
                        violations_found_this_message.add("foreign_language")
                        logger.debug(f"Foreign Language: User {user_id}, lang '{lang_code}' (conf: {confidence:.2f}) not in {channel_lang_config} for channel {channel.id}.")

        if any(word in discrimination_words_set for word in cleaned_content_for_matching.split()) or \
           any(fuzz.partial_ratio(phrase, cleaned_content_for_matching) >= bot_config.fuzzy_match_threshold_keywords for phrase in discrimination_phrases):
            violations_found_this_message.add("discrimination")

        if "discrimination" not in violations_found_this_message and \
           (any(word in nsfw_text_words_set for word in cleaned_content_for_matching.split()) or \
           any(fuzz.partial_ratio(phrase, cleaned_content_for_matching) >= bot_config.fuzzy_match_threshold_keywords for phrase in nsfw_text_phrases)):
            violations_found_this_message.add("nsfw_text")
        
        if cleaned_content_for_matching and not ("discrimination" in violations_found_this_message or "nsfw_text" in violations_found_this_message):
            if OPENAI_API_KEY:
                try:
                    openai_result = await check_openai_moderation_api(content_raw) 
                    if openai_result.get("flagged"):
                        categories = openai_result.get("categories", {})

                        if categories.get("harassment", False) or categories.get("harassment/threatening", False) or \
                           categories.get("hate", False) or categories.get("hate/threatening", False) or \
                           categories.get("self-harm", False) or categories.get("self-harm/intent", False) or categories.get("self-harm/instructions", False):
                            violations_found_this_message.add("openai_flagged_severe")
                        elif categories.get("sexual", False) or categories.get("sexual/minors", False):
                             violations_found_this_message.add("openai_flagged_severe") 
                        elif categories.get("violence", False) or categories.get("violence/graphic", False):
                             violations_found_this_message.add("openai_flagged_severe")
                        else: 
                            violations_found_this_message.add("openai_flagged_moderate")
                        logger.debug(f"OpenAI Flagged: User {user_id}. Categories: {categories}. Scores: {openai_result.get('category_scores')}")
                except Exception as e: 
                    logger.error(f"OpenAI moderation call failed after retries for user {user_id}: {e}")

        for attachment in message.attachments:
            if attachment.content_type and (attachment.content_type.startswith("image/") or attachment.content_type.startswith("video/")):
                if SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET:
                    try:
                        sightengine_result = await check_sightengine_media_api(attachment.url)
                        if sightengine_result:
                            nudity_data = sightengine_result.get("nudity", {})
                            if nudity_data.get("sexual_activity", 0.0) >= bot_config.sightengine_nudity_sexual_activity_threshold or \
                               nudity_data.get("suggestive", 0.0) >= bot_config.sightengine_nudity_suggestive_threshold: 
                                violations_found_this_message.add("nsfw_media")
                            if sightengine_result.get("gore", {}).get("prob", 0.0) >= bot_config.sightengine_gore_threshold:
                                violations_found_this_message.add("gore_violence_media")
                            offensive_data = sightengine_result.get("offensive", {})
                            if offensive_data.get("prob", 0.0) >= bot_config.sightengine_offensive_symbols_threshold: 
                                violations_found_this_message.add("offensive_symbols_media")
                          try:
                            if offensive_data.get("nazi", 0.0) > 0.8 or offensive_data.get("confederate_flag", 0.0) > 0.8:
                                violations_found_this_message.add("offensive_symbols_media")
                            
                            if violations_found_this_message.intersection({"nsfw_media", "gore_violence_media", "offensive_symbols_media"}):
                                logger.debug(f"Sightengine Flagged: User {user_id}, Attachment {attachment.filename}. Result: {sightengine_result}")
                                break 
                    except Exception as e:
                        logger.error(f"Sightengine moderation call failed after retries for user {user_id}, attachment {attachment.filename}: {e}")
        
        if violations_found_this_message:
            logger.info(f"User {user_id} ({member.name}) in guild {guild_id} triggered violations: {list(violations_found_this_message)}. Message: {message.jump_url}")

            if bot_config.delete_violating_messages:
                try:
                    await message.delete()
                    logger.info(f"Deleted message {message.id} from user {user_id} due to violations.")
                except discord.Forbidden:
                    logger.error(f"Failed to delete message {message.id} (user {user_id}): Missing permissions.")
                except discord.NotFound:
                    logger.warning(f"Failed to delete message {message.id} (user {user_id}): Message already deleted.")
                except Exception as e:
                    logger.error(f"Error deleting message {message.id} (user {user_id}): {e}", exc_info=True)
            
            if bot_config.send_in_channel_warning:
                viol_summary = ", ".join(v.replace('_', ' ').title() for v in violations_found_this_message)
                warn_text = f"{member.mention}, your message was moderated due to: **{viol_summary}**. Please review server rules."
                try:
                    await channel.send(warn_text, delete_after=bot_config.in_channel_warning_delete_delay)
                except discord.Forbidden:
                    logger.error(f"Failed to send in-channel warning to channel {channel.id}: Missing permissions.")
                except Exception as e:
                    logger.error(f"Error sending in-channel warning to {channel.id}: {e}", exc_info=True)

            await process_infractions_and_punish(member, guild, list(violations_found_this_message), content_raw, message.jump_url)
        else:
            await self.bot.process_commands(message)

  
    @app_commands.command(name="infractions", description="View a user's recent infractions and active points.")
    @app_commands.default_permissions(manage_messages=True)
    @app_commands.describe(member="The member to check infractions for.")
    async def infractions_command(self, interaction: discord.Interaction, member: discord.Member):
        if not interaction.guild_id:
            await interaction.response.send_message("This command must be used in a server.", ephemeral=True)
            return

        infraction_list, active_points = await get_user_infractions_from_db(member.id, interaction.guild_id, days_limit=90)

        embed = discord.Embed(
            title=f"Infraction Report for {member.display_name}",
            description=f"User ID: `{member.id}`",
            color=discord.Color.orange() if active_points > 0 else discord.Color.green()
        )
        if member.avatar: embed.set_thumbnail(url=member.avatar.url)
        embed.add_field(name="Total Active Points (Last 30 Days)", value=f"**{active_points}**", inline=False)

        if infraction_list:
            history_str = ""
            for infra in infraction_list[:10]:
                ts_dt = datetime.fromisoformat(infra['timestamp'])
                is_active_for_points = ts_dt >= (datetime.now(timezone.utc) - timedelta(days=30))
                active_marker = " (Active)" if is_active_for_points else ""
                
                entry = (f"ID: `{infra['id']}` - **{infra['points']} pts** ({infra['violation_type'].replace('_', ' ').title()}){active_marker}\n"
                         f"<t:{int(ts_dt.timestamp())}:f> (UTC)\n")
                if infra['message_url']:
                    entry += f"[Context]({infra['message_url']})\n"
                if infra['message_content_snippet']:
                     entry += f"```{discord.utils.escape_markdown(infra['message_content_snippet'][:100])}```\n"
                entry += "---\n"
                if len(history_str) + len(entry) > 1000: break 
                history_str += entry
            
            embed.add_field(name="Recent Infraction History (Max 10 of last 90 days)", value=history_str if history_str else "None found in the last 90 days.", inline=False)
        else:
            embed.add_field(name="Recent Infraction History", value="No infractions recorded for this user in the last 90 days.", inline=False)

        embed.set_footer(text=f"Report generated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        await interaction.response.send_message(embed=embed, ephemeral=True)


    @app_commands.command(name="clear_infractions", description="MANUALLY clears ALL or a specific infraction for a user.")
    @app_commands.default_permissions(administrator=True) 
    @app_commands.describe(member="The member whose infractions to clear.", infraction_id="Optional ID of a specific infraction to remove. Clears all if omitted.")
    async def clear_infractions_command(self, interaction: discord.Interaction, member: discord.Member, infraction_id: int | None = None):
        if not interaction.guild_id:
            await interaction.response.send_message("This command must be used in a server.", ephemeral=True)
            return

        action_description = ""
        success = False
        if infraction_id:
            success = await remove_specific_infraction_from_db(infraction_id)
            action_description = f"infraction ID `{infraction_id}`"
        else:
            await clear_user_infractions(member.id, interaction.guild_id) 
            success = True
            action_description = "all infractions"
        
        if success:
            await interaction.response.send_message(f"Successfully cleared {action_description} for {member.mention}.", ephemeral=True)
            await log_moderation_action("infractions_cleared_manual", member, f"Moderator {interaction.user.mention} cleared {action_description}.", interaction.user, interaction.guild, discord.Color.green())
        else:
            await interaction.response.send_message(f"Failed to clear {action_description} for {member.mention}. Infraction ID might be invalid or no infractions to clear.", ephemeral=True)

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


@bot.event
async def on_ready():
    """Event that fires when the bot is ready."""
    logger.info(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    logger.info(f"Discord.py Version: {discord.__version__}")
    logger.info('------')

    global http_session, LANGUAGE_MODEL, db_conn 
    
    if not http_session or http_session.closed: 
        http_session = ClientSession()
        logger.info("Aiohttp ClientSession initialized/reinitialized.")

    if not LANGUAGE_MODEL: 
        try:
            if os.path.exists(FASTTEXT_MODEL_PATH):
                LANGUAGE_MODEL = fasttext.load_model(FASTTEXT_MODEL_PATH)
                logger.info(f"FastText model loaded from {FASTTEXT_MODEL_PATH}")
            else:
                logger.error(f"FastText model file not found at {FASTTEXT_MODEL_PATH}. Language detection will be disabled.")
                LANGUAGE_MODEL = None
        except Exception as e: 
            logger.critical(f"Failed to load FastText model: {e}. Language detection will be disabled.", exc_info=True)
            LANGUAGE_MODEL = None

    if not db_conn:
        await setup_db() 

    await bot.add_cog(GeneralCog(bot))
    await bot.add_cog(ConfigurationCog(bot))
    await bot.add_cog(ModerationCog(bot)) 
    logger.info("Cogs loaded.")

    try:
        synced_commands = await bot.tree.sync()
        logger.info(f"Synced {len(synced_commands)} slash commands globally.")
    except Exception as e:
        logger.error(f"Failed to sync slash commands: {e}", exc_info=True)

    logger.info(f"{bot.user.name} is online and ready! 🚀")


@bot.event
async def on_disconnect():
    logger.warning("Bot disconnected from Discord.")

@bot.event
async def on_resumed():
    logger.info("Bot has resumed its session with Discord.")
    global http_session
    if not http_session or http_session.closed:
        http_session = ClientSession()
        logger.info("Aiohttp ClientSession reinitialized after resume.")


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    """Handles errors for prefix commands (if any are kept/added)."""
    if hasattr(ctx.command, 'on_error'): 
        return

    error = getattr(error, 'original', error) 

    if isinstance(error, commands.CommandNotFound):
        return 
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Oops! You missed an argument: `{error.param.name}`. Check help for usage.", delete_after=15)
    elif isinstance(error, commands.BadArgument):
        await ctx.send(f"Hmm, that's not a valid argument. {error}", delete_after=15)
    elif isinstance(error, commands.NoPrivateMessage):
        await ctx.send("This command can only be used in a server.", delete_after=15)
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send(f"You don't have permission to use this command: `{', '.join(error.missing_permissions)}`", delete_after=15)
    elif isinstance(error, commands.BotMissingPermissions):
        await ctx.send(f"I'm missing permissions to do that: `{', '.join(error.missing_permissions)}`. Please grant them to me!", delete_after=15)
        logger.warning(f"Bot missing permissions in guild {ctx.guild.id if ctx.guild else 'DM'}, channel {ctx.channel.id if hasattr(ctx.channel, 'id') else 'DM'}: {error.missing_permissions} for command {ctx.command.name if ctx.command else 'UnknownCmd'}")
    elif isinstance(error, commands.CommandOnCooldown):
        await ctx.send(f"This command is on cooldown. Try again in {error.retry_after:.2f} seconds.", delete_after=10)
    else:
        logger.error(f"Unhandled command error for '{ctx.command.qualified_name if ctx.command else 'UnknownCmd'}': {error}", exc_info=True)
        try:
            await ctx.send("An unexpected error occurred while running this command. The developers have been notified.", delete_after=15)
        except discord.HTTPException:
            pass 

async def health_check_handler(request):
    """Simple health check endpoint."""
    status_text = f"{bot.user.name} is running! Latency: {round(bot.latency * 1000)}ms. DB: {'OK' if db_conn else 'Error'}. LangModel: {'OK' if LANGUAGE_MODEL else 'Error'}."
    return web.Response(text=status_text, content_type="text/plain")

async def main_async_runner():
    """Handles the asynchronous setup and running of the bot, including a web server."""
    global http_session, db_conn

    await setup_db()
    if not db_conn: 
        logger.critical("DB connection failed to establish in main_async_runner. Bot cannot start.")
        return

    app = web.Application()
    app.router.add_get("/", health_check_handler) 
    app.router.add_get("/health", health_check_handler)

    render_port = os.getenv("PORT")
    runner = None
    site = None
    web_server_task = None

    if render_port:
        try:
            port = int(render_port)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', port)
            web_server_task = asyncio.create_task(site.start()) 
            logger.info(f"Health check web server starting on port {port}.")
        except ValueError:
            logger.error(f"Invalid PORT environment variable: '{render_port}'. Must be an integer. Web server not starting.")
        except Exception as e:
            logger.error(f"Failed to start health check web server: {e}", exc_info=True)
    else:
        logger.info("PORT environment variable not set. Health check web server will not start.")

    discord_bot_task = asyncio.create_task(bot.start(DISCORD_TOKEN))

    try:
        if web_server_task:
            await asyncio.gather(discord_bot_task, web_server_task)
        else:
            await discord_bot_task
    except discord.LoginFailure:
        logger.critical("CRITICAL: Invalid Discord token. Check your ADROIT_TOKEN environment variable.")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        logger.critical(f"Critical error in main_async_runner: {e}", exc_info=True)
    finally:
        logger.info("Initiating final cleanup on bot shutdown...")
        
        if not bot.is_closed():
            logger.info("Closing Discord bot connection...")
            await bot.close() 
            logger.info("Discord bot connection closed.")

        if web_server_task and not web_server_task.done():
            web_server_task.cancel()
            try:
                await web_server_task
            except asyncio.CancelledError:
                logger.info("Web server task cancelled.")
            except Exception as e_ws_cancel:
                logger.error(f"Error cancelling web server task: {e_ws_cancel}", exc_info=True)
        
        if site: 
            logger.info("Stopping web server site...")
        if runner:
            await runner.cleanup()
            logger.info("Aiohttp web server runner cleaned up.")

        if http_session and not http_session.closed:
            await http_session.close()
            logger.info("Aiohttp client session closed.")
        
        if db_conn:
            await db_conn.close()
            logger.info("Database connection closed.")

        logger.info("✅ Cleanup complete. Bot is offline.")


if __name__ == "__main__":
    try:
        asyncio.run(main_async_runner())
    except Exception as e: 
        logger.critical(f"💥 UNHANDLED EXCEPTION IN TOP LEVEL __main__: {e}", exc_info=True)
