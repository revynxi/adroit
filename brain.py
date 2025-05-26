import asyncio
import json
import logging
import os
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta
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

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger('discord_bot')

# --- Environment Variables & Constants ---
DISCORD_TOKEN = os.getenv("ADROIT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FASTTEXT_MODEL_PATH = os.getenv("FASTTEXT_MODEL_PATH", "lid.176.ftz")
SIGHTENGINE_API_USER = os.getenv("SIGHTENGINE_API_USER")
SIGHTENGINE_API_SECRET = os.getenv("SIGHTENGINE_API_SECRET")

if not DISCORD_TOKEN:
    logger.critical("CRITICAL: ADROIT_TOKEN environment variable not set. Exiting.")
    exit(1)

# --- Bot Intents ---
intents = discord.Intents.default()
intents.members = True
intents.message_content = True
intents.presences = False

bot = commands.Bot(command_prefix=">>", intents=intents, help_command=None)

# --- Global Variables (to be initialized) ---
db_conn: aiosqlite.Connection | None = None
LANGUAGE_MODEL = None # FastText model instance
http_session: ClientSession | None = None

class BotConfig:
    """Holds all static configurations for the bot."""
    def __init__(self):
        self.default_log_channel_id = 1113377818424922132 # Example ID, replace with actual
        self.default_channel_configs = { # Default per-channel language/topic settings
            1113377809440722974: {"language": ["en"]}, # general
            1322517478365990984: {"language": ["en"], "topics": ["politics"]}, # politics
            1113377810476716132: {"language": ["en"]}, # gaming
            1321499824926888049: {"language": ["fr"]}, # french
            1122525009102000269: {"language": ["de"]}, # german
            1122523546355245126: {"language": ["ru"]}, # russian
            1122524817904635904: {"language": ["zh"]}, # chinese
            1242768362237595749: {"language": ["es"]}  # spanish
        }
        self.forbidden_text_pattern = re.compile(
            r"(discord\.gg/|join\s+our|server\s+invite|free\s+nitro|check\s+out\s+my|follow\s+me|subscribe\s+to|buy\s+now)",
            re.IGNORECASE
        )
        self.url_pattern = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|net|org|io|dev|xyz|gg|app|co|xyz|online|shop)\b)") # Expanded TLDs
        self.has_alphanumeric_pattern = re.compile(r'[a-zA-Z0-9]')
        self.permitted_domains = [ # Default list of globally permitted domains
            "googleusercontent.com", "tenor.com", "giphy.com", "tiktok.com", "youtube.com",
            "docs.google.com", "cdn.discordapp.com", "media.discordapp.net", "images-ext-1.discordapp.net", "images-ext-2.discordapp.net",
            "roblox.com", "github.com", "theuselessweb.com", "imgur.com", "i.imgur.com",
            "wikipedia.org", "wikimedia.org", "twitch.tv", "reddit.com", "x.com", "twitter.com", "fxtwitter.com", "vxtwitter.com",
            "spotify.com", "soundcloud.com", "pastebin.com", "hastebin.com", "gist.github.com"
        ]
        self.punishment_system = {
            "points_thresholds": {
                5: {"action": "warn", "message": "Warnings make your sins weigh heavier. Think twice before sending something inappropriate."},
                10: {"action": "mute", "duration_hours": 1, "reason": "Spam/Minor violations"},
                15: {"action": "kick", "reason": "Repeated violations"},
                25: {"action": "temp_ban", "duration_days": 1, "reason": "Serious/Persistent violations"},
                50: {"action": "temp_ban", "duration_days": 30, "reason": "Severe/Accumulated violations"},
                10000: {"action": "ban", "reason": "Extreme/Manual Escalation"} # Effectively permanent / admin override
            },
            "violations": {
                "discrimination": {"points": 10, "severity": "High"},
                "spam": {"points": 1, "severity": "Low"},
                "nsfw_text": {"points": 3, "severity": "Medium"},
                "nsfw_media": {"points": 5, "severity": "High"},
                "advertising": {"points": 2, "severity": "Medium"},
                "politics_discussion_disallowed": {"points": 1, "severity": "Low"},
                "off_topic": {"points": 1, "severity": "Low"},
                "foreign_language": {"points": 1, "severity": "Low"},
                "openai_flagged": {"points": 3, "severity": "Medium"},
                "excessive_mentions": {"points": 1, "severity": "Low"},
                "excessive_attachments": {"points": 1, "severity": "Low"},
                "long_message": {"points": 1, "severity": "Low"},
                "gore_violence_media": {"points": 7, "severity": "High"},
                "offensive_symbols_media": {"points": 5, "severity": "High"}
            }
        }
        # Moderation thresholds and limits
        self.spam_window_seconds = 10
        self.spam_message_limit = 5 # Max messages in spam_window_seconds
        self.spam_repetition_limit = 3 # Max identical messages in spam_window_seconds
        self.mention_limit = 5
        self.max_message_length = 1000
        self.max_attachments = 4
        self.min_msg_len_for_lang_check = 4
        self.min_confidence_for_lang_flagging = 0.65
        self.min_confidence_short_msg_lang = 0.75
        self.short_msg_threshold_lang = 20
        self.common_safe_foreign_words = {"bonjour", "hola", "merci", "gracias", "oui", "si", "nyet", "da", "salut", "ciao", "hallo", "guten tag", "privet", "konnichiwa", "arigato"}
        self.fuzzy_match_threshold = 85 # For keyword matching with fuzzy logic

        # Sightengine thresholds
        self.sightengine_nudity_sexual_activity_threshold = 0.6
        self.sightengine_nudity_suggestive_threshold = 0.8
        self.sightengine_gore_threshold = 0.7
        self.sightengine_violence_threshold = 0.7
        self.sightengine_offensive_symbols_threshold = 0.85
        self.sightengine_text_profanity_threshold = 0.9

bot_config = BotConfig()

# --- Term Loading for Keyword/Phrase Matching ---
def load_terms_from_file(filepath: str) -> tuple[set[str], list[str]]:
    """Loads terms from a text file, separating single words and multi-word phrases."""
    words = set()
    phrases = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                term = line.strip().lower()
                if not term or term.startswith("#"): # Skip empty lines and comments
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


# --- Utility Functions ---
def clean_message_content(text: str) -> str:
    """Cleans and normalizes message content for analysis."""
    return text.strip().lower()

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
                    return json.loads(result[0]) # Attempt to parse as JSON
                except json.JSONDecodeError:
                    return result[0] # Return as string if not valid JSON
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
        stored_value = json.dumps(value_to_set) # Serialize complex types to JSON string
    elif value_to_set is None: # Handle explicit None to delete/reset
        stored_value = None # Store as NULL or handle deletion logic
    else:
        stored_value = str(value_to_set) # Ensure other types are stored as strings

    try:
        async with db_conn.cursor() as cursor:
            if stored_value is None: # If value is None, delete the config key
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
            logger.warning(f"FastText returned unexpected prediction format for: '{clean_text[:100]}...'")
            return None, 0.0
    except Exception as e:
        logger.error(f"FastText language detection error for '{clean_text[:100]}...': {e}", exc_info=True)
        return None, 0.0

async def log_moderation_action(
    action: str,
    target_user: discord.User | discord.Member,
    reason: str,
    moderator: discord.User | discord.Member | None = None,
    guild: discord.Guild | None = None,
    color: discord.Color = discord.Color.orange(),
    extra_fields: list[tuple[str, str]] | None = None
):
    """Logs moderation actions to a specified channel and console with a standardized embed."""
    current_guild = guild or (target_user.guild if isinstance(target_user, discord.Member) else None)
    if not current_guild:
        logger.error(f"Cannot log action '{action}' for user {target_user.id}: Guild context missing.")
        return

    log_channel_id = await get_guild_config(current_guild.id, "log_channel_id", bot_config.default_log_channel_id)
    log_channel = bot.get_channel(log_channel_id) if log_channel_id else None

    embed = discord.Embed(
        title=f"🛡️ Moderation: {action.replace('_', ' ').title()}",
        description=reason,
        color=color,
        timestamp=datetime.utcnow()
    )
    embed.add_field(name="Target User", value=f"{target_user.mention} (`{target_user.id}`)", inline=True)
    if isinstance(target_user, discord.Member):
        embed.set_thumbnail(url=target_user.display_avatar.url)

    if moderator:
        embed.add_field(name="Moderator", value=f"{moderator.mention} (`{moderator.id}`)", inline=True)
    else:
        embed.add_field(name="Moderator", value="Automated Action", inline=True)

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
        logger.info(f"Log (Guild: {current_guild.name}, Action: {action}, Target: {target_user.id}, Reason: {reason}) "
                    f"- Log channel ID {log_channel_id} not found or not configured.")


def retry_if_api_error(exception):
    """Retries on server errors (5xx), rate limits (429), or network issues for API calls."""
    if isinstance(exception, client_exceptions.ClientResponseError):
        return exception.status == 429 or exception.status >= 500 # Rate limit or server error
    return isinstance(exception, (asyncio.TimeoutError, client_exceptions.ClientOSError, client_exceptions.ClientConnectorError))


@retry(
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=1, min=3, max=60),
    retry=retry_if_api_error
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
    data = {"input": text_content}

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
    wait=wait_random_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_api_error
)
async def check_sightengine_media_api(image_url: str) -> dict:
    """Checks image against Sightengine API for moderation flags."""
    if not SIGHTENGINE_API_USER or not SIGHTENGINE_API_SECRET:
        logger.debug("Sightengine API keys not set. Skipping image moderation.")
        return {}

    url = f"https://api.sightengine.com/1.0/check.json?url={image_url}&models=nudity,gore,offensive&api_user={SIGHTENGINE_API_USER}&api_secret={SIGHTENGINE_API_SECRET}"
    
    try:
        async with http_session.get(url, timeout=20) as response:
            response.raise_for_status()
            json_response = await response.json()
            return json_response
    except client_exceptions.ClientResponseError as e:
        logger.error(f"Sightengine API error: {e.status} - {e.message} for URL: {image_url}. Retrying if applicable.")
        raise
    except asyncio.TimeoutError:
        logger.error(f"Sightengine API request timed out for URL: {image_url}...")
        raise
    except Exception as e:
        logger.error(f"Unexpected error with Sightengine API for URL: {image_url}: {e}", exc_info=True)
        logger.critical(f"Sightengine moderation failed definitively for URL: '{image_url}'. Moderation skipped.")
        return {}


async def apply_moderation_punishment(member: discord.Member, action: str, reason: str, duration: timedelta | None = None, moderator: discord.User | None = None):
    """Applies a specified moderation action to a member and logs it."""
    dm_message_text = ""
    log_color = discord.Color.orange()
    extra_log_fields = []

    try:
        if action == "warn":
            warn_config = bot_config.punishment_system["points_thresholds"].get(5, {})
            warning_dm_detail = warn_config.get("message", "Please be mindful of the server rules.")
            dm_message_text = f"You have received a formal warning in **{member.guild.name}**.\nReason: {reason}\n\n*{warning_dm_detail}*"
            log_color = discord.Color.gold()
        elif action == "mute":
            if duration:
                await member.timeout(duration, reason=reason)
                dm_message_text = f"You have been muted in **{member.guild.name}** for **{str(duration)}**.\nReason: {reason}"
                log_color = discord.Color.light_grey()
                extra_log_fields.append(("Duration", str(duration)))
            else:
                logger.warning(f"Attempted to mute {member.display_name} without duration. Action skipped.")
                return
        elif action == "kick":
            dm_message_text = f"You have been kicked from **{member.guild.name}**.\nReason: {reason}"
            log_color = discord.Color.red()
        elif action == "temp_ban":
            if duration:
                unban_time = datetime.utcnow() + duration
                if db_conn:
                    async with db_conn.cursor() as cursor:
                        await cursor.execute(
                            'INSERT OR REPLACE INTO temp_bans (user_id, guild_id, unban_time, ban_reason) VALUES (?, ?, ?, ?)',
                            (member.id, member.guild.id, unban_time.isoformat(), reason)
                        )
                        await db_conn.commit()
                dm_message_text = f"You have been temporarily banned from **{member.guild.name}** until {unban_time.strftime('%Y-%m-%d %H:%M:%S UTC')}.\nReason: {reason}"
                log_color = discord.Color.dark_red()
                extra_log_fields.append(("Duration", str(duration)))
                extra_log_fields.append(("Unban Time", unban_time.strftime('%Y-%m-%d %H:%M:%S UTC')))
            else:
                logger.warning(f"Attempted to temp_ban {member.display_name} without duration. Action skipped.")
                return
        elif action == "ban":
            dm_message_text = f"You have been permanently banned from **{member.guild.name}**.\nReason: {reason}"
            log_color = discord.Color.dark_red()

        # Attempt to DM the user
        try:
            await member.send(dm_message_text)
            logger.info(f"DM sent to {member.display_name} ({member.id}) for {action}.")
        except discord.Forbidden:
            logger.warning(f"Could not DM {action} notification to {member.display_name} ({member.id}).")
        except Exception as e:
            logger.error(f"Error sending DM to {member.display_name} ({member.id}) for {action}: {e}", exc_info=True)

        # Perform the actual action
        if action == "kick":
            await member.kick(reason=reason)
        elif action == "temp_ban" or action == "ban":
            await member.ban(reason=reason, delete_message_days=0) # Don't delete messages by default

        await log_moderation_action(action, member, reason, moderator, member.guild, log_color, extra_log_fields)

    except discord.Forbidden:
        logger.error(f"Missing permissions to {action} {member.display_name} in {member.guild.name}. "
                     f"Please check bot role hierarchy and permissions.")
        if log_color != discord.Color.red(): # Log permission errors in red unless it's already a serious action
            await log_moderation_action(f"{action}_failed_permission", member, f"Bot lacks permissions to {action} user: {reason}", moderator, member.guild, discord.Color.red(), extra_log_fields)
    except discord.HTTPException as e:
        logger.error(f"Discord API error while applying {action} to {member.display_name}: {e.status} - {e.text}", exc_info=True)
        await log_moderation_action(f"{action}_failed_http_error", member, f"Discord API error: {e.status} - {e.text} for reason: {reason}", moderator, member.guild, discord.Color.red(), extra_log_fields)
    except Exception as e:
        logger.error(f"Unexpected error applying {action} to {member.display_name}: {e}", exc_info=True)
        await log_moderation_action(f"{action}_failed_unknown", member, f"Unexpected error: {e} for reason: {reason}", moderator, member.guild, discord.Color.red(), extra_log_fields)

async def add_user_infraction(user_id: int, guild_id: int, violation_type: str, message_content: str, message_url: str | None = None):
    """Adds an infraction for a user and checks if a punishment threshold is met."""
    if not db_conn:
        logger.error("add_user_infraction: Database connection is not available.")
        return

    violation_info = bot_config.punishment_system["violations"].get(violation_type)
    if not violation_info:
        logger.warning(f"Attempted to add infraction for unknown violation type: {violation_type}")
        return

    points = violation_info["points"]
    reason = f"Automated infraction: {violation_type.replace('_', ' ').title()}"
    
    try:
        async with db_conn.cursor() as cursor:
            # Add the new infraction
            await cursor.execute(
                "INSERT INTO infractions (user_id, guild_id, violation_type, points, message_content, message_url, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, guild_id, violation_type, points, message_content, message_url, datetime.utcnow().isoformat())
            )
            await db_conn.commit()
            logger.info(f"Infraction added for user {user_id} in guild {guild_id}: {violation_type} (+{points} points).")

            # Calculate total active points (e.g., in the last 30 days)
            thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
            await cursor.execute(
                "SELECT SUM(points) FROM infractions WHERE user_id = ? AND guild_id = ? AND timestamp >= ?",
                (user_id, guild_id, thirty_days_ago)
            )
            total_active_points = (await cursor.fetchone())[0] or 0
            logger.info(f"User {user_id} in guild {guild_id} has {total_active_points} active points.")

            # Check for punishment thresholds
            member = bot.get_guild(guild_id).get_member(user_id)
            if not member:
                logger.warning(f"Member {user_id} not found in guild {guild_id}. Cannot apply punishment.")
                return

            # Sort thresholds from lowest to highest points
            sorted_thresholds = sorted(bot_config.punishment_system["points_thresholds"].items(), key=lambda item: item[0])

            # Apply the highest applicable punishment
            applied_action = None
            for threshold_points, punishment_details in sorted_thresholds:
                if total_active_points >= threshold_points:
                    action = punishment_details["action"]
                    action_reason = punishment_details.get("reason", f"Accumulated {total_active_points} points for {violation_type} related infractions.")
                    duration = None
                    if "duration_hours" in punishment_details:
                        duration = timedelta(hours=punishment_details["duration_hours"])
                    elif "duration_days" in punishment_details:
                        duration = timedelta(days=punishment_details["duration_days"])
                    elif "duration_months" in punishment_details: # Convert months to days for timedelta
                        duration = timedelta(days=punishment_details["duration_months"] * 30) # Approx.
                    
                    # If this is a ban, we only apply it once.
                    # If it's a temp_ban, we only apply if a higher duration is needed or new temp ban.
                    # If it's a warn/mute/kick, always apply if threshold is met.
                    if action == "ban":
                        applied_action = action
                        await apply_moderation_punishment(member, action, action_reason)
                        await clear_user_infractions(user_id, guild_id) # Clear points on ban
                        break # Stop processing further thresholds

                    # For other actions, ensure we're not repeating the same lower level punishment without reason
                    # This logic needs refinement based on desired repeated punishment behavior.
                    # For now, let's just apply the highest applicable action if no action has been applied yet,
                    # or if the new action is a stricter or longer temp_ban.
                    if not applied_action or (action == "temp_ban" and applied_action != "ban"): # Simple logic to apply highest non-ban
                        # More sophisticated logic here: check if user is already muted/temp-banned for *longer*
                        # For simplicity, if current points reach a higher threshold, apply that punishment.
                        # This means a user could be muted, then kicked, then temp-banned as points increase.
                        applied_action = action
                        await apply_moderation_punishment(member, action, action_reason, duration)
                        # Don't break for warn/mute/kick, allow higher thresholds to be checked if points are very high.
                        # For example, if 5 points is warn, 10 is mute. If someone sends something that's 10 points, they should be muted, not just warned.
                        # If a punishment was applied (other than ban), we might want to clear some points or reset.
                        # For now, points accumulate.
                        if action in ["kick", "temp_ban"]: # Clear points on kick/temp_ban as these are significant
                            await clear_user_infractions(user_id, guild_id)
                            break # Stop processing after kick or temp_ban

    except Exception as e:
        logger.error(f"Error adding infraction or applying punishment for user {user_id} in guild {guild_id}: {e}", exc_info=True)


async def get_user_infractions(user_id: int, guild_id: int) -> list[dict]:
    """Retrieves all infractions for a given user in a guild."""
    if not db_conn:
        logger.error("get_user_infractions: Database connection is not available.")
        return []
    try:
        async with db_conn.execute("SELECT * FROM infractions WHERE user_id = ? AND guild_id = ?", (user_id, guild_id)) as cursor:
            rows = await cursor.fetchall()
            infractions = []
            for row in rows:
                infractions.append({
                    "id": row[0],
                    "user_id": row[1],
                    "guild_id": row[2],
                    "violation_type": row[3],
                    "points": row[4],
                    "message_content": row[5],
                    "message_url": row[6],
                    "timestamp": row[7]
                })
            return infractions
    except Exception as e:
        logger.error(f"Error retrieving infractions for user {user_id} in guild {guild_id}: {e}", exc_info=True)
        return []

async def remove_user_infraction(infraction_id: int):
    """Removes a specific infraction by its ID."""
    if not db_conn:
        logger.error("remove_user_infraction: Database connection is not available.")
        return
    try:
        async with db_conn.execute("DELETE FROM infractions WHERE id = ?", (infraction_id,)) as cursor:
            await db_conn.commit()
            logger.info(f"Infraction {infraction_id} removed.")
    except Exception as e:
        logger.error(f"Error removing infraction {infraction_id}: {e}", exc_info=True)

async def clear_user_infractions(user_id: int, guild_id: int):
    """Clears all infractions for a user in a specific guild."""
    if not db_conn:
        logger.error("clear_user_infractions: Database connection is not available.")
        return
    try:
        async with db_conn.execute("DELETE FROM infractions WHERE user_id = ? AND guild_id = ?", (user_id, guild_id)) as cursor:
            await db_conn.commit()
            logger.info(f"All infractions cleared for user {user_id} in guild {guild_id}.")
    except Exception as e:
        logger.error(f"Error clearing infractions for user {user_id} in guild {guild_id}: {e}", exc_info=True)

async def get_temp_bans() -> list[dict]:
    """Retrieves all active temporary bans."""
    if not db_conn:
        logger.error("get_temp_bans: Database connection is not available.")
        return []
    try:
        async with db_conn.execute("SELECT user_id, guild_id, unban_time, ban_reason FROM temp_bans") as cursor:
            rows = await cursor.fetchall()
            temp_bans = []
            for row in rows:
                temp_bans.append({
                    "user_id": row[0],
                    "guild_id": row[1],
                    "unban_time": datetime.fromisoformat(row[2]),
                    "ban_reason": row[3]
                })
            return temp_bans
    except Exception as e:
        logger.error(f"Error retrieving temporary bans: {e}", exc_info=True)
        return []

async def remove_temp_ban(user_id: int, guild_id: int):
    """Removes a temporary ban entry from the database."""
    if not db_conn:
        logger.error("remove_temp_ban: Database connection is not available.")
        return
    try:
        async with db_conn.execute("DELETE FROM temp_bans WHERE user_id = ? AND guild_id = ?", (user_id, guild_id)) as cursor:
            await db_conn.commit()
            logger.info(f"Temp ban removed for user {user_id} in guild {guild_id}.")
    except Exception as e:
        logger.error(f"Error removing temp ban for user {user_id} in guild {guild_id}: {e}", exc_info=True)


# --- Database Setup ---
async def setup_db():
    """Initializes the SQLite database and creates necessary tables."""
    global db_conn
    try:
        db_conn = await aiosqlite.connect('bot_data.db')
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
                message_content TEXT,
                message_url TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        await db_conn.execute("""
            CREATE TABLE IF NOT EXISTS temp_bans (
                user_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL,
                unban_time TEXT NOT NULL,
                ban_reason TEXT,
                PRIMARY KEY (user_id, guild_id)
            )
        """)
        await db_conn.commit()
        logger.info("Database initialized and tables checked/created.")
    except Exception as e:
        logger.critical(f"Failed to connect to or initialize database: {e}", exc_info=True)
        exit(1)


# --- Cogs ---
class General(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="ping", description="Checks the bot's latency.")
    async def ping(self, interaction: discord.Interaction):
        await interaction.response.send_message(f"Pong! {round(self.bot.latency * 1000)}ms", ephemeral=True)

    @app_commands.command(name="set_log_channel", description="Sets the channel where moderation logs will be sent.")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(channel="The channel to set as the log channel.")
    async def set_log_channel(self, interaction: discord.Interaction, channel: discord.TextChannel):
        if not interaction.guild:
            await interaction.response.send_message("This command can only be used in a server.", ephemeral=True)
            return
        await set_guild_config(interaction.guild.id, "log_channel_id", channel.id)
        await interaction.response.send_message(f"Moderation logs will now be sent to {channel.mention}.", ephemeral=True)

    @app_commands.command(name="get_log_channel", description="Shows the current channel configured for moderation logs.")
    @app_commands.default_permissions(manage_guild=True)
    async def get_log_channel(self, interaction: discord.Interaction):
        if not interaction.guild:
            await interaction.response.send_message("This command can only be used in a server.", ephemeral=True)
            return
        channel_id = await get_guild_config(interaction.guild.id, "log_channel_id", bot_config.default_log_channel_id)
        channel = self.bot.get_channel(channel_id)
        if channel:
            await interaction.response.send_message(f"The current moderation log channel is {channel.mention}.", ephemeral=True)
        else:
            await interaction.response.send_message(f"No log channel is currently set or the set channel ({channel_id}) is invalid.", ephemeral=True)

    @app_commands.command(name="set_channel_language", description="Sets the expected language(s) for a channel.")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(channel="The channel to configure.", languages="Comma-separated language codes (e.g., en,fr,de). Use 'any' to disable specific language checks.")
    async def set_channel_language(self, interaction: discord.Interaction, channel: discord.TextChannel, languages: str):
        if not interaction.guild:
            await interaction.response.send_message("This command can only be used in a server.", ephemeral=True)
            return
        
        lang_list = [lang.strip().lower() for lang in languages.split(',')]
        if "any" in lang_list:
            # Setting to None effectively disables language checks for this channel
            await set_guild_config(interaction.guild.id, f"channel_language_{channel.id}", None)
            await interaction.response.send_message(f"Language checks for {channel.mention} have been disabled.", ephemeral=True)
        else:
            await set_guild_config(interaction.guild.id, f"channel_language_{channel.id}", lang_list)
            await interaction.response.send_message(f"Expected languages for {channel.mention} set to: {', '.join(lang_list)}.", ephemeral=True)

    @app_commands.command(name="get_channel_language", description="Shows the expected language(s) for a channel.")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(channel="The channel to check.")
    async def get_channel_language(self, interaction: discord.Interaction, channel: discord.TextChannel):
        if not interaction.guild:
            await interaction.response.send_message("This command can only be used in a server.", ephemeral=True)
            return
        
        channel_configs = await get_guild_config(interaction.guild.id, f"channel_language_{channel.id}", bot_config.default_channel_configs.get(channel.id))
        if channel_configs and "language" in channel_configs:
            await interaction.response.send_message(f"The expected languages for {channel.mention} are: {', '.join(channel_configs['language'])}.", ephemeral=True)
        else:
            await interaction.response.send_message(f"No specific language is configured for {channel.mention}. Defaulting to server-wide or allowing any language.", ephemeral=True)

class Moderation(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.user_message_timestamps = defaultdict(lambda: defaultdict(deque)) # guild_id -> user_id -> deque of timestamps
        self.user_message_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=bot_config.spam_repetition_limit))) # guild_id -> user_id -> deque of messages
        self.temp_ban_check.start()

    def cog_unload(self):
        self.temp_ban_check.cancel()

    async def get_channel_language_config(self, guild_id: int, channel_id: int) -> list[str] | None:
        """Helper to get channel-specific language settings."""
        # Check channel-specific override first
        channel_lang_config = await get_guild_config(guild_id, f"channel_language_{channel_id}", None)
        if channel_lang_config is not None and isinstance(channel_lang_config, list):
            return channel_lang_config
        
        # Fallback to default channel configs defined in BotConfig
        default_channel_conf = bot_config.default_channel_configs.get(channel_id)
        if default_channel_conf and "language" in default_channel_conf:
            return default_channel_conf["language"]
        
        return None # No specific language config found

    @tasks.loop(minutes=1)
    async def temp_ban_check(self):
        """Checks for expired temporary bans and unbans users."""
        if not db_conn:
            logger.warning("temp_ban_check: Database connection is not available, skipping check.")
            return

        now = datetime.utcnow()
        expired_bans = await get_temp_bans()

        for ban_entry in expired_bans:
            user_id = ban_entry["user_id"]
            guild_id = ban_entry["guild_id"]
            unban_time = ban_entry["unban_time"]
            reason = ban_entry["ban_reason"]

            if now >= unban_time:
                guild = self.bot.get_guild(guild_id)
                if not guild:
                    logger.warning(f"Guild {guild_id} not found for expired temp ban of user {user_id}.")
                    await remove_temp_ban(user_id, guild_id) # Remove from DB even if guild not found
                    continue

                try:
                    banned_user = discord.Object(id=user_id) # User might not be in cache
                    await guild.unban(banned_user, reason="Temporary ban expired.")
                    await remove_temp_ban(user_id, guild_id)
                    logger.info(f"User {user_id} unbanned from {guild.name} (temp ban expired).")
                    await log_moderation_action("Unban_Temp_Expired", banned_user, f"Temporary ban expired. Original reason: {reason}", guild=guild, color=discord.Color.green())
                except discord.NotFound:
                    logger.info(f"User {user_id} not found as banned in {guild.name}, removing temp ban entry.")
                    await remove_temp_ban(user_id, guild_id)
                except discord.Forbidden:
                    logger.error(f"Missing permissions to unban user {user_id} from {guild.name}.")
                    await log_moderation_action("Unban_Failed_Permission", banned_user, f"Bot lacks permissions to unban. Manual unban required. Original reason: {reason}", guild=guild, color=discord.Color.red())
                except Exception as e:
                    logger.error(f"Error unbanning user {user_id} from {guild.name}: {e}", exc_info=True)
                    await log_moderation_action("Unban_Failed_Error", banned_user, f"Error during unban: {e}. Manual intervention may be needed. Original reason: {reason}", guild=guild, color=discord.Color.red())

    @temp_ban_check.before_loop
    async def before_temp_ban_check(self):
        await self.bot.wait_until_ready()
        logger.info("Starting temp ban check loop.")


   @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild or message.webhook_id:
            return

        if message.content.startswith(self.bot.command_prefix):
            return # Ignore commands

        member = message.author
        guild = message.guild
        channel = message.channel
        content = message.content
        cleaned_content = clean_message_content(content)
        message_url = message.jump_url
        
        violations_found = set() # Store violation types to add infractions

        # --- 1. Spam Detection ---
        now = datetime.utcnow()
        user_id = member.id
        guild_id = guild.id

        self.user_message_timestamps[guild_id][user_id].append(now)
        
        # Remove old timestamps outside the spam window
        while self.user_message_timestamps[guild_id][user_id] and \
              self.user_message_timestamps[guild_id][user_id][0] < now - timedelta(seconds=bot_config.spam_window_seconds):
            self.user_message_timestamps[guild_id][user_id].popleft()

        # Check message count spam
        if len(self.user_message_timestamps[guild_id][user_id]) > bot_config.spam_message_limit:
            violations_found.add("spam")
            logger.info(f"Spam detected: User {user_id} sent too many messages in {bot_config.spam_window_seconds}s.")
            # No need to process repetition if rate limit already hit for spam

        # Check identical message repetition spam
        # Add current message to history (cleaned content for comparison)
        if cleaned_content: # Only add if there's actual text content
            self.user_message_history[guild_id][user_id].append(cleaned_content)
            if len(self.user_message_history[guild_id][user_id]) >= bot_config.spam_repetition_limit:
                # Check if all messages in the history are identical or very similar
                first_msg = self.user_message_history[guild_id][user_id][0]
                is_repetitive = True
                for msg in list(self.user_message_history[guild_id][user_id])[1:]:
                    if fuzz.ratio(first_msg, msg) < bot_config.fuzzy_match_threshold: # Use fuzzy matching
                        is_repetitive = False
                        break
                if is_repetitive:
                    violations_found.add("spam")
                    logger.info(f"Repetitive spam detected: User {user_id} sent identical/similar messages.")
                    self.user_message_history[guild_id][user_id].clear() # Clear history to avoid immediate re-trigger

        # --- 2. Forbidden Text and URLs ---
        if bot_config.forbidden_text_pattern.search(content):
            violations_found.add("advertising") # Often associated with advertising
            logger.info(f"Forbidden text detected from {user_id}.")

        urls = bot_config.url_pattern.findall(content)
        for url in urls:
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
                if not domain and parsed_url.path: # Handle cases like 'example.com' without scheme
                    domain = parsed_url.path.split('/')[0]

                # Strip common prefixes like 'www.'
                domain = domain.lstrip('www.')
                
                # Check if the domain is not in the permitted list
                if domain and domain not in bot_config.permitted_domains:
                    violations_found.add("advertising")
                    logger.info(f"Unpermitted URL detected: {domain} from {user_id}.")
                    break # Only add once per message

            except Exception as e:
                logger.warning(f"Error parsing URL '{url}': {e}")
                # Continue to next URL or other checks


        # --- 3. Excessive Mentions / Attachments / Message Length ---
        if len(message.mentions) > bot_config.mention_limit:
            violations_found.add("excessive_mentions")
            logger.info(f"Excessive mentions detected from {user_id}.")

        if len(message.attachments) > bot_config.max_attachments:
            violations_found.add("excessive_attachments")
            logger.info(f"Excessive attachments detected from {user_id}.")

        if len(content) > bot_config.max_message_length:
            violations_found.add("long_message")
            logger.info(f"Long message detected from {user_id}.")

        # --- 4. Language Policy Enforcement ---
        channel_language_setting = await self.get_channel_language_config(guild_id, channel.id)

        if channel_language_setting is not None and "any" not in channel_language_setting:
            if len(cleaned_content) >= bot_config.min_msg_len_for_lang_check and bot_config.has_alphanumeric_pattern.search(cleaned_content):
                lang_code, confidence = await detect_language_ai(cleaned_content)
                logger.debug(f"Language detection for user {user_id} in channel {channel.id}: {lang_code} (confidence: {confidence})")

                # Adjust confidence threshold for short messages
                threshold = bot_config.min_confidence_for_lang_flagging
                if len(cleaned_content) < bot_config.short_msg_threshold_lang:
                    threshold = bot_config.min_confidence_short_msg_lang

                if lang_code and lang_code not in channel_language_setting:
                    # Also check against common safe foreign words
                    is_safe_foreign_word = False
                    for word in bot_config.common_safe_foreign_words:
                        if word in cleaned_content:
                            is_safe_foreign_word = True
                            break

                    if not is_safe_foreign_word and confidence >= threshold:
                        violations_found.add("foreign_language")
                        logger.info(f"Foreign language '{lang_code}' detected from {user_id} in channel {channel.id}.")

        # --- 5. Keyword/Phrase Matching (Discrimination, NSFW Text) ---
        words_in_message = set(cleaned_content.split())

        # Discrimination terms
        if discrimination_words_set.intersection(words_in_message):
            violations_found.add("discrimination")
            logger.info(f"Discrimination word detected from {user_id}.")
        for phrase in discrimination_phrases:
            if fuzz.partial_ratio(phrase, cleaned_content) >= bot_config.fuzzy_match_threshold:
                violations_found.add("discrimination")
                logger.info(f"Discrimination phrase detected from {user_id}.")
                break # Found, no need to check other phrases

        # NSFW text terms
        if nsfw_text_words_set.intersection(words_in_message):
            violations_found.add("nsfw_text")
            logger.info(f"NSFW text word detected from {user_id}.")
        for phrase in nsfw_text_phrases:
            if fuzz.partial_ratio(phrase, cleaned_content) >= bot_config.fuzzy_match_threshold:
                violations_found.add("nsfw_text")
                logger.info(f"NSFW text phrase detected from {user_id}.")
                break

        # --- 6. AI Moderation (OpenAI, Sightengine) ---
        # ONLY call OpenAI if no discrimination or NSFW text was found by local checks
        if not ("discrimination" in violations_found or "nsfw_text" in violations_found):
            if OPENAI_API_KEY and cleaned_content:
                openai_result = await check_openai_moderation_api(cleaned_content)
                if openai_result.get("flagged"):
                    violations_found.add("openai_flagged")
                    logger.info(f"OpenAI moderation flagged content from {user_id}.")
                    logger.debug(f"OpenAI categories: {openai_result.get('categories')}, scores: {openai_result.get('category_scores')}")
        else:
            logger.info(f"Skipping OpenAI moderation for {user_id} due to local keyword/phrase match.")


        # Check attachments with Sightengine
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                if SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET:
                    sightengine_result = await check_sightengine_media_api(attachment.url)
                    if sightengine_result:
                        if sightengine_result.get("nudity", {}).get("sexual_activity") and sightengine_result["nudity"]["sexual_activity"] >= bot_config.sightengine_nudity_sexual_activity_threshold:
                            violations_found.add("nsfw_media")
                            logger.info(f"Sightengine flagged sexual activity in image from {user_id}.")
                        elif sightengine_result.get("nudity", {}).get("suggestive") and sightengine_result["nudity"]["suggestive"] >= bot_config.sightengine_nudity_suggestive_threshold:
                            violations_found.add("nsfw_media")
                            logger.info(f"Sightengine flagged suggestive content in image from {user_id}.")
                        
                        if sightengine_result.get("gore", {}).get("gore") and sightengine_result["gore"]["gore"] >= bot_config.sightengine_gore_threshold:
                            violations_found.add("gore_violence_media")
                            logger.info(f"Sightengine flagged gore in image from {user_id}.")
                        elif sightengine_result.get("violence", {}).get("violence") and sightengine_result["violence"]["violence"] >= bot_config.sightengine_violence_threshold:
                            violations_found.add("gore_violence_media")
                            logger.info(f"Sightengine flagged violence in image from {user_id}.")

                        if sightengine_result.get("offensive", {}).get("symbols") and sightengine_result["offensive"]["symbols"] >= bot_config.sightengine_offensive_symbols_threshold:
                            violations_found.add("offensive_symbols_media")
                            logger.info(f"Sightengine flagged offensive symbols in image from {user_id}.")

        # --- Apply Infractions ---
        if violations_found:
            for violation_type in violations_found:
                # Add infraction for each unique violation found
                await add_user_infraction(user_id, guild_id, violation_type, content, message_url)
                logger.info(f"Added infraction for {violation_type} to user {user_id}.")


@bot.event
async def on_ready():
    """Event that fires when the bot is ready."""
    logger.info(f'Logged in as {bot.user} (ID: {bot.user.id})')
    logger.info('------')

    global http_session, LANGUAGE_MODEL
    http_session = ClientSession()
    logger.info("Aiohttp ClientSession created.")

    try:
        LANGUAGE_MODEL = fasttext.load_model(FASTTEXT_MODEL_PATH)
        logger.info(f"FastText model loaded from {FASTTEXT_MODEL_PATH}")
    except ValueError as e:
        logger.error(f"Failed to load FastText model from {FASTTEXT_MODEL_PATH}: {e}. Language detection will be disabled.", exc_info=True)
        LANGUAGE_MODEL = None
    except Exception as e:
        logger.critical(f"An unexpected error occurred while loading FastText model: {e}", exc_info=True)
        LANGUAGE_MODEL = None

    await setup_db()

    # Sync slash commands
    try:
        # Sync to specific guilds for faster updates during development
        # For production, consider syncing globally or using bot.tree.copy_global_to(guild)
        # and then await app_commands.sync() for faster guild-specific updates.
        synced_commands = await bot.tree.sync() # Syncs global commands
        logger.info(f"Synced {len(synced_commands)} slash commands globally.")
    except Exception as e:
        logger.error(f"Failed to sync slash commands: {e}", exc_info=True)

    logger.info("Adroit Bot is online and ready! 🚀")


@bot.event
async def on_disconnect():
    logger.warning("Bot disconnected from Discord. Running cleanup...")
    # Cleanup tasks are handled in the main_async_runner's finally block
    # or by explicit bot.close() where needed.

@bot.event
async def on_error(event_name, *args, **kwargs):
    """Logs errors that occur in event listeners."""
    logger.error(f"Error in event '{event_name}':", exc_info=True)
  
async def health_check(request):
    """Simple health check endpoint for Render."""
    return web.Response(text="Bot is running!")

async def main_async_runner():
    """Handles the asynchronous setup and running of the bot, including a web server for Render."""
    global http_session # Ensure http_session is accessible

    # Create an aiohttp web application
    app = web.Application()
    app.router.add_get("/", health_check) # Add a health check endpoint

    # Get the port from the environment variable (Render sets this)
    port = int(os.getenv("PORT", 8080)) # Default to 8080 if not set, though Render will set it

    # Create an aiohttp web server runner
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)

    # Start both the Discord bot and the web server concurrently
    discord_task = asyncio.create_task(bot.start(DISCORD_TOKEN))
    web_server_task = asyncio.create_task(site.start())

    try:
        logger.info(f"Starting web server on port {port} for Render health checks.")
        # Wait for both tasks to complete (or for the bot to shut down)
        await asyncio.gather(discord_task, web_server_task)
    except Exception as e:
        logger.critical(f"Bot or web server failed to start: {e}", exc_info=True)
    finally:
        logger.info("Initiating final cleanup on bot shutdown.")
        # Close aiohttp session used by the bot
        if http_session and not http_session.closed:
            await http_session.close()
            logger.info("Aiohttp client session closed.")

        # Clean up web server resources
        await runner.cleanup()
        logger.info("Aiohttp web server runner cleaned up.")

        # Close database connection
        if db_conn:
            await db_conn.close()
            logger.info("Database connection closed.")

        logger.info("✅ Cleanup complete. Adroit Bot is offline.")


if __name__ == "__main__":
    try:
        asyncio.run(main_async_runner())
    except KeyboardInterrupt:
        logger.info("⌨️ Bot shutdown initiated by KeyboardInterrupt (Ctrl+C).")
    except Exception as e:
        logger.critical(f"💥 UNHANDLED EXCEPTION IN MAIN RUNNER: {e}", exc_info=True)
