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

# Load environment variables from .env file
load_dotenv()

# --- Logging Setup ---
# Configure basic logging for the bot. This will output logs to the console.
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger('discord_bot')

# --- Environment Variables & Constants ---
# Retrieve necessary API keys and tokens from environment variables.
# It's crucial to set these in a .env file or directly in your environment.
DISCORD_TOKEN = os.getenv("ADROIT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FASTTEXT_MODEL_PATH = os.getenv("FASTTEXT_MODEL_PATH", "lid.176.ftz") # Default path for FastText model
SIGHTENGINE_API_USER = os.getenv("SIGHTENGINE_API_USER")
SIGHTENGINE_API_SECRET = os.getenv("SIGHTENGINE_API_SECRET")

# Critical check: Ensure Discord token is set, otherwise the bot cannot run.
if not DISCORD_TOKEN:
    logger.critical("CRITICAL: ADROIT_TOKEN environment variable not set. Exiting.")
    exit(1)

# --- Bot Intents ---
# Define the intents your bot needs. Intents tell Discord which events your bot wants to receive.
# discord.Intents.default() provides a good starting set.
# intents.members is crucial for moderation actions (kicking, banning, getting member info).
# intents.message_content is required to read message content for moderation.
# intents.presences is generally not needed for moderation and can be resource-intensive, so keep it False.
intents = discord.Intents.default()
intents.members = True
intents.message_content = True
intents.presences = False

# Initialize the bot instance.
# command_prefix=">>" is for traditional prefix commands (though slash commands are preferred).
# help_command=None disables the default help command, allowing custom implementation if desired.
bot = commands.Bot(command_prefix=">>", intents=intents, help_command=None)

# --- Global Variables (to be initialized at startup) ---
# These variables will hold instances of database connection, FastText model, and HTTP session.
db_conn: aiosqlite.Connection | None = None
LANGUAGE_MODEL = None # FastText model instance for language detection
http_session: ClientSession | None = None # aiohttp client session for external API calls


class BotConfig:
    """
    Holds all static and default configurations for the bot.
    These values serve as defaults and can often be overridden by guild-specific
    settings stored in the database.
    """
    def __init__(self):
        # Default channel ID for moderation logs. This should ideally be set per guild.
        self.default_log_channel_id = None # Changed to None, as it's better to require configuration.

        # Default per-channel language/topic settings.
        # IMPORTANT: These are example IDs and will NOT work for your server.
        # You should configure these using the `/configure` command after the bot is running.
        # The bot will use these as defaults if no specific configuration is found in the DB.
        self.default_channel_configs = {
            # Example: 1113377809440722974: {"language": ["en"]}, # 'general' channel
            # Example: 1322517478365990984: {"language": ["en"], "topics": ["politics"]}, # 'politics' channel
        }

        # Regex pattern for detecting forbidden text strings (e.g., common advertising phrases).
        self.forbidden_text_pattern = re.compile(
            r"(discord\.gg/|join\s+our|server\s+invite|free\s+nitro|check\s+out\s+my|follow\s+me|subscribe\s+to|buy\s+now)",
            re.IGNORECASE
        )
        # Regex pattern to find URLs in messages.
        self.url_pattern = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|net|org|io|dev|xyz|gg|app|co|uk|ai|art|blog|cloud|codes|design|digital|email|finance|games|group|info|live|media|money|online|photos|shop|site|store|tech|tools|top|travel|video|wiki|world|zone)\b)", re.IGNORECASE)
        # Regex pattern to check if a string contains any alphanumeric characters.
        self.has_alphanumeric_pattern = re.compile(r'[a-zA-Z0-9]')

        # Default list of globally permitted domains for links.
        # These are domains that are generally safe and allowed unless overridden by guild settings.
        self.permitted_domains = [
            "googleusercontent.com", "tenor.com", "giphy.com", "tiktok.com", "youtube.com",
            "docs.google.com", "cdn.discordapp.com", "media.discordapp.net", "images-ext-1.discordapp.net", "images-ext-2.discordapp.net",
            "roblox.com", "github.com", "theuselessweb.com", "imgur.com", "i.imgur.com",
            "wikipedia.org", "wikimedia.org", "twitch.tv", "reddit.com", "x.com", "twitter.com", "fxtwitter.com", "vxtwitter.com",
            "spotify.com", "soundcloud.com", "pastebin.com", "hastebin.com", "gist.github.com"
        ]

        # Configuration for the automated punishment system based on accumulated infraction points.
        self.punishment_system = {
            "points_thresholds": {
                5: {"action": "warn", "message": "Warnings make your sins weigh heavier. Think twice before sending something inappropriate."},
                10: {"action": "mute", "duration_hours": 1, "reason": "Spam/Minor violations"},
                15: {"action": "kick", "reason": "Repeated violations"},
                25: {"action": "temp_ban", "duration_days": 1, "reason": "Serious/Persistent violations"},
                50: {"action": "temp_ban", "duration_days": 30, "reason": "Severe/Accumulated violations"},
                10000: {"action": "ban", "reason": "Extreme/Manual Escalation"} # High threshold, effectively permanent / admin override
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

        # Moderation thresholds and limits for various checks.
        self.spam_window_seconds = 10 # Time window for spam detection (in seconds)
        self.spam_message_limit = 5 # Max messages allowed within spam_window_seconds
        self.spam_repetition_limit = 3 # Max identical messages allowed within spam_window_seconds
        self.mention_limit = 5 # Max mentions allowed in a single message
        self.max_message_length = 1000 # Max characters allowed in a message
        self.max_attachments = 4 # Max attachments allowed in a message
        self.min_msg_len_for_lang_check = 4 # Minimum message length to perform language detection
        self.min_confidence_for_lang_flagging = 0.65 # Minimum confidence for flagging a foreign language
        self.min_confidence_short_msg_lang = 0.75 # Higher confidence for short messages
        self.short_msg_threshold_lang = 20 # Messages shorter than this are considered "short" for language detection
        # Common safe foreign words that should not trigger foreign language flags if detected alone.
        self.common_safe_foreign_words = {"bonjour", "hola", "merci", "gracias", "oui", "si", "nyet", "da", "salut", "ciao", "hallo", "guten tag", "privet", "konnichiwa", "arigato"}
        self.fuzzy_match_threshold = 85 # Threshold for fuzzy string matching (0-100), higher is stricter

        # Sightengine API specific thresholds for content moderation.
        self.sightengine_nudity_sexual_activity_threshold = 0.6
        self.sightengine_nudity_suggestive_threshold = 0.8
        self.sightengine_gore_threshold = 0.7
        self.sightengine_violence_threshold = 0.7
        self.sightengine_offensive_symbols_threshold = 0.85 # For symbols like nazi, confederate, etc.
        self.sightengine_text_profanity_threshold = 0.9


# Instantiate the bot configuration.
bot_config = BotConfig()

# --- Term Loading for Keyword/Phrase Matching ---
def load_terms_from_file(filepath: str) -> tuple[set[str], list[str]]:
    """
    Loads terms from a text file, separating single words and multi-word phrases.
    Each line in the file is considered a term. Lines starting with '#' are comments.
    """
    words = set()
    phrases = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                term = line.strip().lower()
                if not term or term.startswith("#"): # Skip empty lines and comments
                    continue
                if ' ' in term: # If the term contains a space, it's a phrase
                    phrases.append(term)
                else: # Otherwise, it's a single word
                    words.add(term)
        logger.info(f"Loaded {len(words)} words and {len(phrases)} phrases from {filepath}")
    except FileNotFoundError:
        logger.warning(f"Warning: Terms file '{filepath}' not found. No terms loaded for this category.")
    except Exception as e:
        logger.error(f"Error loading terms from {filepath}: {e}", exc_info=True)
    return words, phrases

# Load discrimination and NSFW terms from their respective files.
# IMPORTANT: You need to create these files (`discrimination_terms.txt` and `nsfw_text_terms.txt`)
# in the same directory as your bot script.
# Example content for discrimination_terms.txt:
# hate speech
# racial slur
# Example content for nsfw_text_terms.txt:
# explicit word
# adult content
discrimination_words_set, discrimination_phrases = load_terms_from_file('discrimination_terms.txt')
nsfw_text_words_set, nsfw_text_phrases = load_terms_from_file('nsfw_text_terms.txt')


# --- Utility Functions ---
def clean_message_content(text: str) -> str:
    """Cleans and normalizes message content for analysis (e.g., case-insensitive checks)."""
    return text.strip().lower() # Simple cleaning: remove leading/trailing whitespace and convert to lowercase

async def get_guild_config(guild_id: int, key: str, default_value=None):
    """
    Retrieves a guild-specific configuration value from the database.
    Handles JSON deserialization for complex types (lists, dictionaries).
    """
    if not db_conn:
        logger.error("get_guild_config: Database connection is not available.")
        return default_value
    try:
        async with db_conn.execute("SELECT config_value FROM guild_configs WHERE guild_id = ? AND config_key = ?", (guild_id, key)) as cursor:
            result = await cursor.fetchone()
            if result and result[0] is not None:
                try:
                    # Attempt to parse the stored value as JSON.
                    # This is important for settings like 'allowed_languages' or 'permitted_domains'
                    # which are stored as JSON strings.
                    return json.loads(result[0])
                except json.JSONDecodeError:
                    # If it's not valid JSON, return it as a string.
                    return result[0]
            # If no result or value is None, return the provided default.
            return default_value
    except Exception as e:
        logger.error(f"Error getting guild config for guild {guild_id}, key {key}: {e}", exc_info=True)
        return default_value

async def set_guild_config(guild_id: int, key: str, value_to_set):
    """
    Sets or updates a guild-specific configuration value in the database.
    Serializes complex types (lists, dictionaries) to JSON strings for storage.
    If `value_to_set` is None, the configuration key is deleted from the database.
    """
    if not db_conn:
        logger.error("set_guild_config: Database connection is not available.")
        return

    stored_value = value_to_set
    if isinstance(value_to_set, (list, dict)):
        stored_value = json.dumps(value_to_set) # Serialize complex types to JSON string
    elif value_to_set is None:
        stored_value = None # Signal to delete the entry
    else:
        stored_value = str(value_to_set) # Ensure other types are stored as strings

    try:
        async with db_conn.cursor() as cursor:
            if stored_value is None: # If value is None, delete the config key
                 await cursor.execute('DELETE FROM guild_configs WHERE guild_id = ? AND config_key = ?', (guild_id, key))
                 logger.info(f"Cleared guild config: Guild {guild_id}, Key '{key}'")
            else:
                # Use INSERT OR REPLACE to either insert a new row or update an existing one.
                await cursor.execute(
                    'INSERT OR REPLACE INTO guild_configs (guild_id, config_key, config_value) VALUES (?, ?, ?)',
                    (guild_id, key, stored_value)
                )
                logger.info(f"Set guild config: Guild {guild_id}, Key '{key}', Value '{value_to_set}'")
        await db_conn.commit() # Commit changes to the database
    except Exception as e:
        logger.error(f"Error setting guild config for guild {guild_id}, key {key}: {e}", exc_info=True)


async def detect_language_ai(text: str) -> tuple[str | None, float]:
    """
    Detect the language of the given text using the FastText model.
    Returns a tuple of (language_code | None, confidence_score).
    """
    clean_text = text.strip()
    if not clean_text:
        return None, 0.0

    if not LANGUAGE_MODEL:
        logger.warning("FastText model not loaded. Cannot detect language.")
        return None, 0.0
    try:
        # FastText's predict method expects a single string. Replace newlines for better prediction.
        prediction = LANGUAGE_MODEL.predict(clean_text.replace('\n', ' '), k=1)
        if prediction and prediction[0] and prediction[1] and len(prediction[0]) > 0:
            lang_code = prediction[0][0].replace("__label__", "") # Extract language code
            confidence = float(prediction[1][0]) # Extract confidence score
            return lang_code, confidence
        else:
            logger.warning(f"FastText returned unexpected prediction format for: '{clean_text[:100]}...'")
            return None, 0.0
    except Exception as e:
        logger.error(f"FastText language detection error for '{clean_text[:100]}...': {e}", exc_info=True)
        return None, 0.0 # Return None for lang if an error occurs

async def log_moderation_action(
    action: str,
    target_user: discord.User | discord.Member,
    reason: str,
    moderator: discord.User | discord.Member | None = None,
    guild: discord.Guild | None = None,
    color: discord.Color = discord.Color.orange(),
    extra_fields: list[tuple[str, str]] | None = None
):
    """
    Logs moderation actions to a specified log channel and the console with a standardized embed.
    This provides a clear record of all moderation activities.
    """
    current_guild = guild or (target_user.guild if isinstance(target_user, discord.Member) else None)
    if not current_guild:
        logger.error(f"Cannot log action '{action}' for user {target_user.id}: Guild context missing.")
        return

    # Retrieve the log channel ID for the current guild.
    # If not set, it will use the default (which is None, so it will log only to console).
    log_channel_id = await get_guild_config(current_guild.id, "log_channel_id", bot_config.default_log_channel_id)
    log_channel = bot.get_channel(log_channel_id) if log_channel_id else None

    # Create a rich embed for the moderation log.
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

    # Add any extra fields provided (e.g., duration of mute/ban).
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
                    f"- Log channel ID {log_channel_id} not found or not configured. Logging to console only.")


def retry_if_api_error(exception):
    """
    Predicate function for tenacity.retry decorator.
    Retries on specific aiohttp client exceptions:
    - 429 (Too Many Requests - Rate Limit)
    - 5xx (Server Errors)
    - Network issues (Timeout, OS Error, Connector Error)
    """
    if isinstance(exception, client_exceptions.ClientResponseError):
        return exception.status == 429 or exception.status >= 500
    return isinstance(exception, (asyncio.TimeoutError, client_exceptions.ClientOSError, client_exceptions.ClientConnectorError))


@retry(
    stop=stop_after_attempt(4), # Max 4 attempts (1 initial + 3 retries)
    wait=wait_random_exponential(multiplier=1, min=3, max=60), # Exponential backoff between 3 and 60 seconds
    retry=retry_if_api_error # Custom retry condition
)
async def check_openai_moderation_api(text_content: str) -> dict:
    """
    Checks text against the OpenAI moderation API with robust retries.
    Returns the moderation result dictionary from OpenAI.
    """
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
            response.raise_for_status() # Raise an exception for 4xx or 5xx status codes
            json_response = await response.json()
            results = json_response.get("results", [])
            if results:
                return results[0] # Contains "flagged", "categories", "category_scores"
            logger.warning(f"OpenAI moderation returned empty results for text: {text_content[:100]}...")
            return {"flagged": False, "categories": {}, "category_scores": {}}
    except client_exceptions.ClientResponseError as e:
        if e.status == 400: # Bad Request (e.g., invalid input for OpenAI)
            logger.warning(f"OpenAI moderation: 400 Bad Request (will not retry). Text: '{text_content[:100]}...'. Error: {e.message}")
            return {"flagged": False, "categories": {}, "category_scores": {}} # Don't retry 400 errors
        logger.error(f"OpenAI moderation API error: {e.status} - {e.message}. Text: '{text_content[:100]}...'. Retrying if applicable.")
        raise # Re-raise the exception for tenacity to catch and retry
    except asyncio.TimeoutError:
        logger.error(f"OpenAI moderation API request timed out. Text: {text_content[:100]}...")
        raise # Re-raise for tenacity
    except Exception as e:
        logger.error(f"Unexpected error with OpenAI moderation API: {e} for text: {text_content[:100]}...", exc_info=True)
        # Fallback if tenacity gives up or for non-retryable errors not caught above
        logger.critical(f"OpenAI moderation failed definitively for text: '{text_content[:100]}...'. Moderation skipped.")
        return {"flagged": False, "categories": {}, "category_scores": {}}


async def apply_moderation_punishment(member: discord.Member, action: str, reason: str, duration: timedelta | None = None, moderator: discord.User | None = None):
    """
    Applies a specified moderation action (warn, mute, kick, temp_ban, ban) to a member
    and logs the action.
    """
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
                # Timeout a member. Requires 'Moderate Members' permission.
                await member.timeout(duration, reason=reason)
                dm_message_text = f"You have been muted in **{member.guild.name}** for **{str(duration)}**.\nReason: {reason}"
                log_color = discord.Color.light_grey()
                extra_log_fields.append(("Duration", str(duration)))
            else:
                logger.warning(f"Attempted to mute {member.display_name} without duration. Action skipped.")
                return
        elif action == "kick":
            dm_message_text = f"You have been kicked from **{member.guild.name}**.\nReason: {reason}"
            # Kick the member. Requires 'Kick Members' permission.
            await member.kick(reason=reason)
            log_color = discord.Color.red()
        elif action == "temp_ban":
            if duration:
                unban_time = datetime.utcnow() + duration
                if db_conn:
                    # Log temporary ban in the database for later automatic unbanning.
                    async with db_conn.cursor() as cursor:
                        await cursor.execute(
                            'INSERT OR REPLACE INTO temp_bans (user_id, guild_id, unban_time, ban_reason) VALUES (?, ?, ?, ?)',
                            (member.id, member.guild.id, unban_time.isoformat(), reason)
                        )
                        await db_conn.commit()
                dm_message_text = f"You have been temporarily banned from **{member.guild.name}** until **{unban_time.strftime('%Y-%m-%d %H:%M:%S UTC')}** ({str(duration)} from now).\nReason: {reason}"
                # Ban the member. Requires 'Ban Members' permission.
                await member.ban(reason=reason, delete_message_days=0) # delete_message_days=0 means no messages are deleted.
                log_color = discord.Color.dark_red()
                extra_log_fields.append(("Unban Time", unban_time.strftime('%Y-%m-%d %H:%M:%S UTC')))
            else:
                logger.warning(f"Attempted to temp_ban {member.display_name} without duration. Action skipped.")
                return
        elif action == "ban":
            dm_message_text = f"You have been permanently banned from **{member.guild.name}**.\nReason: {reason}"
            # Ban the member permanently. Requires 'Ban Members' permission.
            await member.ban(reason=reason, delete_message_days=0)
            log_color = discord.Color.black() # Very severe action

        # Attempt to DM the user about the moderation action.
        if dm_message_text:
            try:
                dm_embed = discord.Embed(title=f"Moderation Action: {action.title()}", description=dm_message_text, color=log_color)
                dm_embed.set_footer(text=f"Guild: {member.guild.name}")
                await member.send(embed=dm_embed)
            except discord.Forbidden:
                logger.warning(f"Could not DM {action} notification to {member.display_name} ({member.id}). They may have DMs disabled.")
            except discord.HTTPException as e:
                 logger.error(f"Failed to send DM for {action} to {member.display_name}: {e}", exc_info=True)


        # Log the action to the dedicated moderation log channel.
        await log_moderation_action(action, member, reason, moderator, member.guild, color=log_color, extra_fields=extra_log_fields)

    except discord.Forbidden:
        # This error occurs if the bot does not have the necessary permissions to perform the action.
        logger.error(f"BOT PERMISSION ERROR: Missing permissions to {action} member {member.display_name} ({member.id}) in guild {member.guild.name}. Check role hierarchy and bot permissions.")
        if moderator and isinstance(moderator, discord.Member):
            try:
                # Attempt to notify the moderator if the automated action failed due to bot permissions.
                await moderator.send(f"Error: I don't have sufficient permissions to `{action}` user `{member.display_name}` in `{member.guild.name}`. Please check my roles and permissions.")
            except discord.Forbidden:
                pass # Cannot notify moderator either
    except discord.HTTPException as e:
        logger.error(f"Discord API error while applying {action} to {member.display_name}: {e.status} - {e.text}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error applying {action} to {member.display_name}: {e}", exc_info=True)


async def add_user_infraction(member: discord.Member, points_to_add: int, violation_types_str: str, original_message: discord.Message):
    """
    Logs an infraction for a user, accumulates their points, and triggers automated
    punishments if their total points meet defined thresholds.
    """
    if not db_conn:
        logger.error("add_user_infraction: Database connection is not available.")
        return
    guild_id = member.guild.id
    user_id = member.id

    try:
        async with db_conn.cursor() as cursor:
            # Insert the new infraction record into the database.
            await cursor.execute(
                'INSERT INTO infractions (user_id, guild_id, points, timestamp, violation_type, message_id, channel_id, message_content) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (user_id, guild_id, points_to_add, datetime.utcnow().isoformat(), violation_types_str, original_message.id, original_message.channel.id, original_message.content[:200]) # Log snippet of message
            )
            # Calculate total active points for the user (e.g., infractions within the last 30 days).
            thirty_days_ago_iso = (datetime.utcnow() - timedelta(days=30)).isoformat()
            await cursor.execute(
                'SELECT SUM(points) FROM infractions WHERE user_id = ? AND guild_id = ? AND timestamp >= ?',
                (user_id, guild_id, thirty_days_ago_iso)
            )
            total_points_data = await cursor.fetchone()
            current_total_points = total_points_data[0] if total_points_data and total_points_data[0] is not None else 0
        await db_conn.commit() # Commit changes to the database

        logger.info(f"User {member.display_name} ({user_id}) in guild {guild_id} received {points_to_add} points for '{violation_types_str}'. Total active points: {current_total_points}.")

        # Check for automated punishment thresholds.
        # Iterate through thresholds from highest to lowest to ensure the most severe applicable
        # punishment is applied if multiple thresholds are met.
        for threshold in sorted(bot_config.punishment_system["points_thresholds"].keys(), reverse=True):
            if current_total_points >= threshold:
                punishment_details = bot_config.punishment_system["points_thresholds"][threshold]
                action_to_take = punishment_details["action"]

                base_reason = punishment_details.get("reason", f"Automated action due to reaching {current_total_points} infraction points.")
                full_reason = f"{base_reason} (Violations: {violation_types_str})"

                duration = None
                if "duration_hours" in punishment_details:
                    duration = timedelta(hours=punishment_details["duration_hours"])
                elif "duration_days" in punishment_details:
                    duration = timedelta(days=punishment_details["duration_days"])

                logger.info(f"Threshold of {threshold} points met by {member.display_name} (Total: {current_total_points}). Applying '{action_to_take}'.")
                # Apply the punishment. The bot itself is the moderator for automated actions.
                await apply_moderation_punishment(member, action_to_take, full_reason, duration, moderator=bot.user)
                break # Apply only one punishment (the highest one met)
    except Exception as e:
        logger.error(f"Error in add_user_infraction for {member.display_name}: {e}", exc_info=True)


def get_domain_from_url(url_string: str) -> str | None:
    """Extracts the TLD+1 domain from a given URL string."""
    try:
        # Prepend 'https://' if no scheme is present to help urlparse.
        if not url_string.startswith(('http://', 'https://')):
            url_string = 'https://' + url_string
        parsed_url = urlparse(url_string)
        domain = parsed_url.netloc
        if domain:
            domain = domain.lower()
            if domain.startswith('www.'):
                domain = domain[4:] # Remove 'www.' prefix
            return domain
        return None
    except ValueError: # Handles malformed URLs for urlparse
        logger.debug(f"Could not parse domain from potentially malformed URL: '{url_string}'")
        return None

def is_domain_permitted(domain_to_check: str, guild_specific_permitted_domains: list[str]) -> bool:
    """
    Checks if the `domain_to_check` is in the guild's permitted list or is a subdomain of a permitted one.
    This function supports matching subdomains (e.g., "sub.example.com" matches "example.com").
    """
    if not domain_to_check:
        return False

    for permitted_domain_pattern in guild_specific_permitted_domains:
        # Check for exact match or if it's a subdomain of a permitted domain.
        if domain_to_check == permitted_domain_pattern or domain_to_check.endswith('.' + permitted_domain_pattern):
            return True
    return False

# --- HTTP Server for Health Checks (Optional, for deployment platforms) ---
async def handle_health_check_request(request: web.Request):
    """Handles HTTP health check requests."""
    return web.Response(text=f"Adroit Bot is healthy and awake! Version: 1.0. Time: {datetime.utcnow()}", status=200, content_type="text/plain")

async def start_http_health_server():
    """
    Starts a simple aiohttp server for health checks.
    This is useful for deployment platforms (like Heroku, Render, etc.) to check if the bot process is running.
    """
    try:
        app = web.Application()
        app.router.add_get('/', handle_health_check_request) # Root path for health check

        port_str = os.getenv("PORT", "8080") # Default to 8080 if PORT env var not set
        port = int(port_str)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host='0.0.0.0', port=port)
        await site.start()
        logger.info(f"✅ HTTP health check server started on http://0.0.0.0:{port}")
        # The server runs in the background. asyncio.create_task ensures it doesn't block the main loop.
    except ValueError:
        logger.error(f"❌ Invalid PORT environment variable: '{port_str}'. Must be an integer. HTTP server not started.")
    except Exception as e:
        logger.error(f"❌ Failed to start HTTP health check server: {e}", exc_info=True)

# --- Background Tasks ---
@tasks.loop(hours=12) # This task runs every 12 hours.
async def cleanup_and_decay_task():
    """
    Periodically cleans up old data from the database (infractions, temp bans, message history)
    and processes expired temporary bans.
    """
    if not db_conn:
        logger.warning("cleanup_and_decay_task: Database connection not available. Skipping.")
        return

    logger.info("Starting cleanup and decay task...")
    try:
        async with db_conn.cursor() as cursor:
            # 1. Delete very old infraction records (e.g., older than 90 days).
            ninety_days_ago_iso = (datetime.utcnow() - timedelta(days=90)).isoformat()
            await cursor.execute('DELETE FROM infractions WHERE timestamp < ?', (ninety_days_ago_iso,))
            deleted_infractions = cursor.rowcount
            if deleted_infractions > 0:
                logger.info(f"Deleted {deleted_infractions} infraction records older than 90 days.")

            # 2. Process expired temporary bans.
            now_iso = datetime.utcnow().isoformat()
            await cursor.execute('SELECT user_id, guild_id, ban_reason FROM temp_bans WHERE unban_time <= ?', (now_iso,))
            expired_bans_data = await cursor.fetchall()

            for user_id, guild_id, ban_reason in expired_bans_data:
                guild = bot.get_guild(guild_id)
                if guild:
                    try:
                        user_to_unban_obj = discord.Object(id=user_id) # Unban by ID
                        await guild.unban(user_to_unban_obj, reason="Temporary ban automatically expired.")
                        logger.info(f"Automatically unbanned user ID {user_id} from guild {guild.name} ({guild_id}).")

                        # Try to fetch user for richer logging, but don't fail if user is gone.
                        target_user_for_log = None
                        try:
                            target_user_for_log = await bot.fetch_user(user_id)
                        except discord.NotFound:
                            logger.debug(f"User {user_id} not found via fetch_user in cleanup_and_decay_task, using ID for log.")
                        except discord.HTTPException as e:
                            logger.warning(f"HTTP error fetching user {user_id} in cleanup_and_decay_task: {e}. Using ID for log.")

                        log_reason = f"Temporary ban expired. Original reason: {ban_reason or 'Not specified.'}"
                        await log_moderation_action("unban", target_user_for_log or user_to_unban_obj, log_reason, bot.user, guild, color=discord.Color.green())
                    except discord.NotFound:
                        logger.warning(f"User {user_id} not found in ban list of guild {guild.name} for auto-unban, or already unbanned.")
                    except discord.Forbidden:
                        logger.error(f"BOT PERMISSION ERROR: Missing permissions to unban user {user_id} in guild {guild.name}.")
                    except Exception as e:
                        logger.error(f"Error during automatic unban for user {user_id} in guild {guild.name}: {e}", exc_info=True)
                else:
                    logger.warning(f"Cannot process temp ban expiry for user {user_id} in guild {guild_id}: Bot is no longer in this guild.")

                # Delete the processed temp_ban entry regardless of unban success to prevent re-processing.
                await cursor.execute('DELETE FROM temp_bans WHERE user_id = ? AND guild_id = ? AND unban_time <= ?', (user_id, guild_id, now_iso))

            if expired_bans_data:
                 logger.info(f"Processed {len(expired_bans_data)} expired temporary bans.")


            # 3. Clean up old message history for spam checks.
            # Retention should be reasonably short, just enough for spam/repetition context.
            message_history_retention_seconds = max(bot_config.spam_window_seconds * 60, 600) # e.g., 10 minutes
            message_history_cutoff_ts = (datetime.utcnow() - timedelta(seconds=message_history_retention_seconds)).timestamp()
            await cursor.execute("DELETE FROM message_history WHERE timestamp < ?", (message_history_cutoff_ts,))
            deleted_messages = cursor.rowcount
            if deleted_messages > 0:
                logger.info(f"Cleaned up {deleted_messages} message_history entries older than {message_history_retention_seconds}s.")

            await db_conn.commit()
        logger.info("Cleanup and decay task completed.")
    except Exception as e:
        logger.error(f"Error during cleanup_and_decay_task: {e}", exc_info=True)

@cleanup_and_decay_task.before_loop
async def before_cleanup_and_decay_task():
    """Waits until the bot is ready before starting the cleanup task."""
    await bot.wait_until_ready()
    logger.info("Cleanup and decay task is ready and will start looping.")


# --- Event Handlers ---
@bot.event
async def on_error(event_method_name: str, *args, **kwargs):
    """
    Global error handler for unhandled exceptions that occur within bot events.
    This catches errors that aren't specific to a command.
    """
    logger.error(f"Unhandled error in event '{event_method_name}': Args: {args}, Kwargs: {kwargs}", exc_info=True)
    # Potentially log to a Discord channel if critical, but be careful about rate limits/loops.

@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    """
    Centralized error handling for traditional prefix commands.
    Provides user-friendly error messages for common command-related issues.
    """
    embed_title = "Command Error"
    embed_color = discord.Color.red()
    error_message = f"An unexpected error occurred: {error}"

    if isinstance(error, commands.CommandNotFound):
        return # Usually best to ignore, or send a very subtle ephemeral message if it's a common typo.
    elif isinstance(error, commands.MissingRequiredArgument):
        embed_title = "Missing Argument"
        error_message = f"You missed the `{error.param.name}` argument. Please check the command's usage."
    elif isinstance(error, commands.BadArgument):
        embed_title = "Invalid Argument"
        error_message = f"One of the arguments you provided was invalid. {error}"
    elif isinstance(error, commands.NoPrivateMessage):
        embed_title = "Server Only"
        error_message = "This command can only be used inside a server."
    elif isinstance(error, commands.PrivateMessageOnly):
        embed_title = "DMs Only"
        error_message = "This command can only be used in Direct Messages."
    elif isinstance(error, commands.MissingPermissions):
        embed_title = "Permission Denied"
        missing_perms_str = ", ".join([f"`{perm.replace('_', ' ').title()}`" for perm in error.missing_permissions])
        error_message = f"You don't have the required permissions to use this command: {missing_perms_str}."
    elif isinstance(error, commands.BotMissingPermissions):
        embed_title = "Bot Lacks Permissions"
        missing_perms_str = ", ".join([f"`{perm.replace('_', ' ').title()}`" for perm in error.missing_permissions])
        error_message = f"I'm missing the following permissions to perform this action: {missing_perms_str}. Please grant them to me."
        logger.warning(f"Bot missing permissions in G:{ctx.guild.id if ctx.guild else 'DM'} C:{ctx.channel.id}: {error.missing_permissions} for cmd: {ctx.command.name if ctx.command else 'N/A'}")
    elif isinstance(error, commands.CommandOnCooldown):
        embed_title = "Command on Cooldown"
        error_message = f"This command is on cooldown. Please try again in **{error.retry_after:.2f} seconds**."
        embed_color = discord.Color.gold()
    elif isinstance(error, commands.CheckFailure): # Generic check failure (e.g., custom checks)
        embed_title = "Check Failed"
        error_message = "You do not meet the requirements to run this command." # Keep it generic
    elif isinstance(error, commands.CommandInvokeError):
        original_error = error.original
        logger.error(f"Error invoking command '{ctx.command.qualified_name if ctx.command else 'UnknownCmd'}': {original_error}", exc_info=original_error)
        embed_title = "Internal Command Error"
        error_message = "An internal error occurred while executing this command. The developers have been notified."
        embed_color = discord.Color.dark_red()
    else: # Fallback for other/unknown command errors
        logger.error(f"Unhandled command error for '{ctx.command.qualified_name if ctx.command else 'UnknownCmd'}': {error}", exc_info=True)

    embed = discord.Embed(title=f"❌ {embed_title}", description=error_message, color=embed_color)
    try:
        # Send ephemeral response for slash commands, regular for prefix commands.
        await ctx.send(embed=embed, ephemeral=isinstance(ctx, discord.Interaction))
    except discord.HTTPException:
        logger.error("Failed to send command error message to context.")


@bot.event
async def on_guild_join(guild: discord.Guild):
    """
    Logs when the bot joins a new guild and sends a welcome message if possible.
    Guides the guild owner on initial setup.
    """
    logger.info(f"Joined new guild: {guild.name} (ID: {guild.id}, Members: {guild.member_count}, Owner: {guild.owner_id})")

    welcome_embed = discord.Embed(
        title=f"👋 Hello, {guild.name}!",
        description=(
            f"Thanks for adding Adroit Bot! I'm here to help with moderation.\n\n"
            f"**Quick Start:**\n"
            f"- Use `/configure setting:log_channel_id value:<your_log_channel_id>` to set up a logging channel.\n"
            f"- Use `/configure setting:allowed_languages value:en target_channel:<your_general_channel>` to set English as default for a channel.\n"
            f"- View all settings with `/get_config`.\n"
            f"- See available commands by typing `/` in Discord and browsing the slash commands.\n\n"
            f"I'll do my best to keep your community safe and sound!"
        ),
        color=discord.Color.blue()
    )
    welcome_embed.set_footer(text="Adroit Bot | Advanced Moderation")
    welcome_embed.set_thumbnail(url=bot.user.display_avatar.url)

    sent_welcome = False
    # Try to send to the system channel first.
    if guild.system_channel and guild.system_channel.permissions_for(guild.me).send_messages:
        try:
            await guild.system_channel.send(embed=welcome_embed)
            sent_welcome = True
            logger.info(f"Sent welcome message to system channel in {guild.name}")
        except discord.HTTPException:
            pass # Fall through to try other channels

    # If not sent to system channel, try the first available text channel.
    if not sent_welcome:
        for channel in guild.text_channels:
            if channel.permissions_for(guild.me).send_messages:
                try:
                    await channel.send(embed=welcome_embed)
                    logger.info(f"Sent welcome message to first available channel #{channel.name} in {guild.name}")
                    break
                except discord.HTTPException:
                    continue # Try next channel

    if not sent_welcome:
        logger.warning(f"Could not send welcome message to any channel in {guild.name}")


# --- Cogs ---
class Moderation(commands.Cog):
    """
    Handles message moderation, infraction tracking, and punishment application.
    This cog contains the core logic for filtering messages based on rules.
    """
    def __init__(self, bot_instance: commands.Bot):
        self.bot = bot_instance
        # In-memory storage for rapid message spam detection.
        # defaultdict creates a new deque for a user if one doesn't exist.
        self.user_rate_limit_data = defaultdict(lambda: deque())

        # Load keyword/phrase lists for discrimination and NSFW text.
        self.discrimination_words, self.discrimination_phrases = discrimination_words_set, discrimination_phrases
        self.nsfw_text_words, self.nsfw_text_phrases = nsfw_text_words_set, nsfw_text_phrases

        # Start background tasks associated with this cog.
        self.cleanup_inmemory_rate_limit_data.start()

    def cog_unload(self):
        """Called when the cog is unloaded. Ensures background tasks are cancelled."""
        self.cleanup_inmemory_rate_limit_data.cancel()
        logger.info("Moderation cog's in-memory rate limit cleanup task cancelled.")

    @tasks.loop(minutes=10) # Runs every 10 minutes.
    async def cleanup_inmemory_rate_limit_data(self):
        """
        Cleans up old timestamps from the in-memory rate limit deques.
        This prevents the `user_rate_limit_data` from growing indefinitely.
        """
        now = datetime.utcnow()
        cleaned_users = 0
        # Iterate over a copy of keys to safely modify the dictionary during iteration.
        for user_id in list(self.user_rate_limit_data.keys()):
            user_deque = self.user_rate_limit_data[user_id]
            # Remove timestamps older than the spam window plus a small buffer.
            while user_deque and (now - user_deque[0]).total_seconds() >= (bot_config.spam_window_seconds + 5):
                user_deque.popleft()
            if not user_deque: # If deque becomes empty, remove user_id from dict to save memory.
                del self.user_rate_limit_data[user_id]
                cleaned_users +=1
        if cleaned_users > 0:
            logger.debug(f"Cleaned up in-memory rate limit data for {cleaned_users} users.")

    @cleanup_inmemory_rate_limit_data.before_loop
    async def before_cleanup_inmemory_rate_limit_data(self):
        """Waits until the bot is ready before starting the in-memory cleanup task."""
        await self.bot.wait_until_ready()
        logger.info("In-memory rate limit data cleanup task is ready.")


    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """
        This event listener processes every message sent in guilds for moderation.
        It applies various checks and triggers punishments based on configured rules.
        """
        # Ignore DMs, messages from bots, or if essential services (DB, HTTP session) are not ready.
        if not message.guild or message.author.bot or not db_conn or not http_session:
            return

        author_member = message.author # This is a Member object, providing guild-specific info.
        guild_id = message.guild.id
        user_id = author_member.id
        content_raw = message.content # Original message content
        content_lower = clean_message_content(content_raw) # Lowercased for case-insensitive checks
        now = datetime.utcnow()
        violations_found = set() # Store unique violation types detected for this message.

        # --- 0. Role-based Immunity ---
        # Users with 'manage_messages' permission (typically moderators/admins) are immune to automated moderation.
        if author_member.guild_permissions.manage_messages:
            await self.bot.process_commands(message) # Still process commands for them.
            return # Skip all moderation checks for immune users.

        # --- 1. Spam Detection (Rate Limiting & Repetition) ---
        # A. In-memory rate limiting for very rapid messages.
        user_timestamps = self.user_rate_limit_data[user_id]
        while user_timestamps and (now - user_timestamps[0]).total_seconds() >= bot_config.spam_window_seconds:
            user_timestamps.popleft() # Remove old timestamps
        user_timestamps.append(now) # Add current message timestamp
        if len(user_timestamps) > bot_config.spam_message_limit:
            violations_found.add("spam")
            logger.debug(f"SPAM (Rate Limit): User {user_id} sent {len(user_timestamps)} msgs in {bot_config.spam_window_seconds}s.")

        # B. DB-backed message history for content repetition.
        try:
            # Store a hash of the raw message content for efficient repetition checks.
            await db_conn.execute(
                "INSERT INTO message_history (user_id, guild_id, timestamp, message_content_hash) VALUES (?, ?, ?, ?)",
                (user_id, guild_id, now.timestamp(), hash(content_raw))
            )
            # Check for repetition within the spam window.
            repetition_check_ts_start = (now - timedelta(seconds=bot_config.spam_window_seconds)).timestamp()
            async with db_conn.execute(
                "SELECT COUNT(*) FROM message_history WHERE user_id = ? AND guild_id = ? AND message_content_hash = ? AND timestamp >= ?",
                (user_id, guild_id, hash(content_raw), repetition_check_ts_start)
            ) as cursor:
                repetition_count = (await cursor.fetchone())[0]
                if repetition_count >= bot_config.spam_repetition_limit:
                    violations_found.add("spam")
                    logger.debug(f"SPAM (Repetition): User {user_id} repeated content '{content_raw[:30]}...' {repetition_count} times.")
        except Exception as e:
            logger.error(f"DB error during spam check for user {user_id}: {e}", exc_info=True)


        # --- 2. Basic Message Properties Violations (Rule 5: Don't annoy our members) ---
        if len(message.mentions) > bot_config.mention_limit: violations_found.add("excessive_mentions")
        if len(message.attachments) > bot_config.max_attachments: violations_found.add("excessive_attachments")
        if len(content_raw) > bot_config.max_message_length: violations_found.add("long_message")

        # --- 3. Fetch Channel & Guild Configurations ---
        # Channel-specific settings (e.g., language, topics) are stored as a JSON dictionary.
        channel_db_config_key = f"channel_config_{message.channel.id}"
        raw_channel_db_settings = await get_guild_config(guild_id, channel_db_config_key, {})

        # Merge default channel config with database overrides.
        effective_channel_config = bot_config.default_channel_configs.get(message.channel.id, {}).copy()
        if isinstance(raw_channel_db_settings, dict):
            effective_channel_config.update(raw_channel_db_settings)

        # Guild-wide settings.
        guild_permitted_domains = await get_guild_config(guild_id, "permitted_domains", list(bot_config.permitted_domains))
        link_only_channel_id = await get_guild_config(guild_id, "link_channel_id", None)

        # --- 4. Language Check (Rule 7: Speak English in general) ---
        allowed_languages = effective_channel_config.get("language") # e.g., ["en", "fr"]
        if allowed_languages and content_lower: # Only perform if specific languages are enforced and content exists.
            # Skip language check for very short messages, URLs, or non-alphanumeric content.
            can_skip_lang_check = (
                len(content_lower) < bot_config.min_msg_len_for_lang_check or
                bot_config.url_pattern.fullmatch(content_lower.strip()) or
                not bot_config.has_alphanumeric_pattern.search(content_lower)
            )
            if not can_skip_lang_check:
                detected_lang, confidence = await detect_language_ai(content_raw)
                logger.debug(f"LangDetect: '{content_raw[:30]}...' -> {detected_lang} (Conf: {confidence:.2f}), Allowed: {allowed_languages}")
                if detected_lang and detected_lang not in allowed_languages:
                    is_short = len(content_lower) < bot_config.short_msg_threshold_lang
                    conf_threshold = bot_config.min_confidence_short_msg_lang if is_short else bot_config.min_confidence_for_lang_flagging

                    # Flag as foreign language unless it's a common safe word or confidence is too low.
                    if not (content_lower in bot_config.common_safe_foreign_words or confidence < conf_threshold):
                        violations_found.add("foreign_language")

        # --- 5. Advertising & Forbidden Links (Rule 9: Advertising is prohibited) ---
        # Check for forbidden text patterns.
        if bot_config.forbidden_text_pattern.search(content_lower):
            violations_found.add("advertising")

        urls_in_message = bot_config.url_pattern.findall(content_raw)
        if urls_in_message:
            # If a link-only channel is set, check domains only if this is NOT that channel.
            if link_only_channel_id and message.channel.id != link_only_channel_id:
                for url_match_groups in urls_in_message:
                    url_str = url_match_groups[0] if isinstance(url_match_groups, tuple) else url_match_groups
                    domain = get_domain_from_url(url_str)
                    if domain and not is_domain_permitted(domain, guild_permitted_domains):
                        violations_found.add("advertising")
                        logger.debug(f"ADVERTISING (Forbidden Domain): {domain} in {url_str} by {user_id}")
                        break # One forbidden domain is enough to flag
            elif not link_only_channel_id: # No specific link channel, check all domains everywhere.
                 for url_match_groups in urls_in_message:
                    url_str = url_match_groups[0] if isinstance(url_match_groups, tuple) else url_match_groups
                    domain = get_domain_from_url(url_str)
                    if domain and not is_domain_permitted(domain, guild_permitted_domains):
                        violations_found.add("advertising")
                        logger.debug(f"ADVERTISING (Forbidden Domain): {domain} in {url_str} by {user_id}")
                        break

        # --- 6. Keyword/Phrase-based Moderation (Rule 1: Respectful, Rule 3: No NSFW Text) ---
        # Uses fuzzy matching for better detection of variations.
        # Discrimination
        if not "discrimination" in violations_found:
            for word in self.discrimination_words:
                if fuzz.ratio(word, content_lower) >= bot_config.fuzzy_match_threshold:
                    violations_found.add("discrimination")
                    logger.debug(f"DISCRIMINATION (Fuzzy Word: '{word}') by {user_id}: '{content_raw[:30]}...'")
                    break
            if not "discrimination" in violations_found:
                for phrase in self.discrimination_phrases:
                    if fuzz.partial_ratio(phrase, content_lower) >= bot_config.fuzzy_match_threshold:
                        violations_found.add("discrimination")
                        logger.debug(f"DISCRIMINATION (Fuzzy Phrase: '{phrase}') by {user_id}: '{content_raw[:30]}...'")
                        break
        # NSFW Text
        if not "nsfw_text" in violations_found:
            for word in self.nsfw_text_words:
                if fuzz.ratio(word, content_lower) >= bot_config.fuzzy_match_threshold:
                    violations_found.add("nsfw_text")
                    logger.debug(f"NSFW_TEXT (Fuzzy Word: '{word}') by {user_id}: '{content_raw[:30]}...'")
                    break
            if not "nsfw_text" in violations_found:
                for phrase in self.nsfw_text_phrases:
                    if fuzz.partial_ratio(phrase, content_lower) >= bot_config.fuzzy_match_threshold:
                        violations_found.add("nsfw_text")
                        logger.debug(f"NSFW_TEXT (Fuzzy Phrase: '{phrase}') by {user_id}: '{content_raw[:30]}...'")
                        break

        # --- 7. Off-topic Check (Rule 8: Off-topic is prohibited, Rule 6: Disputes should remain secular) ---
        allowed_topics = effective_channel_config.get("topics")
        if allowed_topics and content_lower: # Only if topics are enforced for the channel.
            # This is a simple keyword-based check. For more advanced topic detection, NLP models would be needed.
            if not any(topic.lower() in content_lower for topic in allowed_topics):
                # If 'politics' or 'religion' are NOT in allowed_topics, then discussions on them are off-topic.
                if "politics" in content_lower or "religion" in content_lower:
                    violations_found.add("politics_discussion_disallowed")
                    logger.debug(f"POLITICS/RELIGION (Disallowed): {user_id}: '{content_raw[:30]}...' (Allowed: {allowed_topics})")
                else:
                    violations_found.add("off_topic")
                    logger.debug(f"OFF_TOPIC by {user_id}: '{content_raw[:30]}...' (Allowed: {allowed_topics})")


        # --- 8. Media NSFW/Gore/Violence Check (Sightengine for attachments) (Rule 3: No NSFW) ---
        # Only check if attachments exist, Sightengine API keys are configured, and no text-based NSFW is already found.
        if message.attachments and SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET and \
           not ("nsfw_text" in violations_found or "nsfw_media" in violations_found or "gore_violence_media" in violations_found):
            for attachment in message.attachments:
                if attachment.content_type and (attachment.content_type.startswith('image/') or attachment.content_type.startswith('video/')):
                    logger.debug(f"Checking attachment '{attachment.filename}' with Sightengine...")
                    try:
                        sightengine_flags = await self.check_media_content_sightengine(attachment.url)
                        if sightengine_flags:
                            violations_found.update(sightengine_flags) # Add all flags returned by Sightengine
                            logger.info(f"Sightengine flagged media from {user_id}: {attachment.url}. Flags: {sightengine_flags}")
                            break # One bad attachment is enough for this message
                    except Exception as e:
                        logger.error(f"Error during Sightengine check for {attachment.url}: {e}", exc_info=True)
                    break # Only check the first valid media attachment to save API calls.

        # --- 9. OpenAI Moderation API Check ---
        # Run if content exists and hasn't already hit a major local filter like discrimination or NSFW.
        # This helps save API calls if local filters are definitive.
        run_openai_check = (
            OPENAI_API_KEY and content_raw.strip() and
            not ("discrimination" in violations_found or "nsfw_text" in violations_found or "nsfw_media" in violations_found)
        )
        if run_openai_check:
            openai_result = await check_openai_moderation_api(content_raw)
            if openai_result.get("flagged"):
                violations_found.add("openai_flagged")
                logger.info(f"OpenAI flagged message from {user_id}: '{content_raw[:30]}...'. Categories: {openai_result.get('categories')}")
                # Map specific OpenAI categories to internal, more severe violations if desired.
                categories = openai_result.get("categories", {})
                if categories.get("hate", False) or categories.get("hate/threatening", False):
                    violations_found.add("discrimination") # Escalate to discrimination
                if categories.get("sexual", False) or categories.get("sexual/minors", False):
                    violations_found.add("nsfw_text") # Escalate to nsfw_text
                if categories.get("violence", False) or categories.get("self-harm", False):
                    violations_found.add("gore_violence_media") # Re-use for text-based violence/self-harm


        # --- Process Violations ---
        if violations_found:
            if db_conn: await db_conn.commit() # Commit message history now that we have violations.

            logger.info(f"Violations by {author_member.name} ({user_id}) in G:{guild_id}/C:{message.channel.id}: {', '.join(violations_found)}. Content: '{content_raw[:50]}...'")

            # Calculate total points for this message based on detected violations.
            total_points_for_message = sum(
                bot_config.punishment_system["violations"].get(v_type, {}).get("points", 0)
                for v_type in violations_found
            )

            # Log infraction and apply punishment if points > 0.
            if total_points_for_message > 0:
                await add_user_infraction(author_member, total_points_for_message, ", ".join(sorted(list(violations_found))), message)

            # Send a brief notification to the channel about message deletion.
            try:
                violation_titles_str = ", ".join(sorted([v.replace('_', ' ').title() for v in violations_found]))
                warning_embed = discord.Embed(
                    description=f"{author_member.mention}, your message was removed due to: **{violation_titles_str}**. Please review server rules.",
                    color=discord.Color.orange()
                )
                await message.channel.send(embed=warning_embed, delete_after=20) # Auto-delete after 20 seconds.
            except discord.Forbidden:
                logger.warning(f"Missing permissions to send violation notification in #{message.channel.name}.")
            except Exception as e:
                logger.error(f"Error sending violation notification: {e}", exc_info=True)

            # Delete the offending message.
            try:
                await message.delete()
                logger.info(f"Deleted message {message.id} from {author_member.name} due to violations.")
            except discord.Forbidden:
                logger.error(f"BOT PERMISSION ERROR: Missing permissions to delete message {message.id} in #{message.channel.name}.")
            except discord.NotFound:
                logger.warning(f"Message {message.id} not found for deletion (already deleted?).")
            except Exception as e:
                logger.error(f"Error deleting message {message.id}: {e}", exc_info=True)

        else: # No violations found, proceed to process commands.
            if db_conn: await db_conn.commit() # Commit message history even if no violation.
            await self.bot.process_commands(message) # Process traditional prefix commands.


    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=2, max=15), retry=retry_if_api_error)
    async def check_media_content_sightengine(self, media_url: str) -> set[str]:
        """
        Checks media (images/videos) using Sightengine API for various content types (nudity, gore, violence, offensive symbols, text profanity).
        Returns a set of violation strings if flagged, an empty set otherwise.
        """
        if not http_session:
            logger.error("Sightengine check: http_session is not available.")
            return set()
        if not SIGHTENGINE_API_USER or not SIGHTENGINE_API_SECRET:
            logger.debug("Sightengine API keys not set. Skipping media moderation.")
            return set()

        params = {
            'url': media_url,
            'models': 'nudity-2.0,wad,offensive,gore,violence,text', # Comprehensive models for image/video analysis
            'api_user': SIGHTENGINE_API_USER,
            'api_secret': SIGHTENGINE_API_SECRET
        }
        api_url = "https://api.sightengine.com/1.0/check.json"
        flagged_violations = set()

        try:
            async with http_session.get(api_url, params=params, timeout=20) as response:
                response_text = await response.text()
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

                data = json.loads(response_text) # Parse JSON response
                logger.debug(f"Sightengine API response for {media_url}: {data}")

                if data.get("status") != "success":
                    error_info = data.get("error", {})
                    logger.warning(f"Sightengine API returned non-success status for {media_url}: {error_info.get('message', 'Unknown error')}")
                    if error_info.get('type') == 'media_error' and error_info.get('code') == 22: # Specific error for large GIFs
                         logger.info(f"Sightengine: Media error (e.g. GIF too large) for {media_url}. Not flagging.")
                    return set() # Treat as not flagged if API had issues with the media itself

                # Nudity Check (nudity-2.0 model)
                nudity = data.get("nudity", {})
                if nudity.get("sexual_activity", 0.0) > bot_config.sightengine_nudity_sexual_activity_threshold or \
                   nudity.get("suggestive", 0.0) > bot_config.sightengine_nudity_suggestive_threshold:
                    flagged_violations.add("nsfw_media")

                # Gore & Violence Check
                if data.get("gore", {}).get("prob", 0.0) > bot_config.sightengine_gore_threshold or \
                   data.get("violence", {}).get("prob", 0.0) > bot_config.sightengine_violence_threshold:
                    flagged_violations.add("gore_violence_media")

                # Offensive Symbols Check (from 'offensive' model)
                offensive = data.get("offensive", {})
                if offensive.get("nazi", 0.0) > bot_config.sightengine_offensive_symbols_threshold or \
                   offensive.get("confederate", 0.0) > bot_config.sightengine_offensive_symbols_threshold or \
                   offensive.get("supremacist", 0.0) > bot_config.sightengine_offensive_symbols_threshold or \
                   offensive.get("terrorist", 0.0) > bot_config.sightengine_offensive_symbols_threshold:
                    flagged_violations.add("offensive_symbols_media")

                # Text in Image Profanity Check (from 'text' model)
                # Note: 'text' model can return multiple profanity detections, so it's a list of dicts.
                text_profanity_score = 0.0
                if data.get("text", {}).get("profanity"):
                    # Find the highest profanity score if multiple are detected.
                    text_profanity_score = max([p.get("prob", 0.0) for p in data["text"]["profanity"]])

                if text_profanity_score > bot_config.sightengine_text_profanity_threshold:
                     flagged_violations.add("nsfw_text") # Flag as nsfw_text if image contains profane text

                return flagged_violations

        except client_exceptions.ClientResponseError as e:
            logger.error(f"Sightengine API HTTP error: {e.status} - {e.message} for {media_url}. URL: {e.request_info.url if e.request_info else 'N/A'}")
            if e.status == 400 and "cannot_load_image" in e.message.lower():
                 logger.warning(f"Sightengine could not load image at {media_url}. Assuming safe and not retrying.")
                 return set() # Don't retry if image cannot be loaded.
            raise # Re-raise for tenacity to retry
        except asyncio.TimeoutError:
            logger.error(f"Sightengine API request timed out for {media_url}")
            raise # Re-raise for tenacity
        except json.JSONDecodeError:
            logger.error(f"Sightengine API: Failed to decode JSON response for {media_url}. Response: {response_text[:200]}")
            return set() # Cannot parse, assume safe
        except Exception as e:
            logger.error(f"Unexpected error with Sightengine API for {media_url}: {e}", exc_info=True)
            return set() # Fallback: assume safe if unexpected error after retries


class ConfigCommands(commands.Cog, name="Configuration"):
    """
    Commands for configuring bot settings per guild or a specific channel.
    Requires 'Manage Guild' permission to use.
    """
    def __init__(self, bot_instance: commands.Bot):
        self.bot = bot_instance

    @app_commands.command(name="configure", description="Configure bot settings for this guild or a specific channel.")
    @app_commands.default_permissions(manage_guild=True) # Only users with Manage Guild permission can use this.
    @app_commands.describe(
        setting_name="The setting to configure (e.g., log_channel_id, link_channel_id, allowed_languages, permitted_domains, allowed_topics).",
        value="The value for the setting. For lists (languages, domains, topics), use comma-separation. Use 'clear' to reset.",
        target_channel="[Optional] The channel for channel-specific settings (e.g., allowed_languages, allowed_topics)."
    )
    @app_commands.choices(setting_name=[
        app_commands.Choice(name="Log Channel ID", value="log_channel_id"),
        app_commands.Choice(name="Link-Only Channel ID", value="link_channel_id"),
        app_commands.Choice(name="Allowed Languages (Channel)", value="allowed_languages"),
        app_commands.Choice(name="Allowed Topics (Channel)", value="allowed_topics"), # Re-enabled topics
        app_commands.Choice(name="Permitted Link Domains (Guild)", value="permitted_domains"),
    ])
    async def configure_setting(self, interaction: discord.Interaction, setting_name: str, value: str, target_channel: discord.TextChannel | None = None):
        await interaction.response.defer(ephemeral=True) # Defer response as operations might take time.
        guild_id = interaction.guild_id

        is_channel_specific_type = setting_name in ["allowed_languages", "allowed_topics"]

        # Determine the actual channel context for the setting.
        # If target_channel is provided, use it; otherwise, use the channel where the command was invoked.
        config_context_channel = target_channel or interaction.channel

        db_key_to_use: str
        parsed_value: any = None
        success_message_detail = ""

        if value.lower() == 'clear':
            parsed_value = None # Signal to clear/delete the setting from DB.
        else:
            # Parse the input value based on the setting name.
            if setting_name in ["log_channel_id", "link_channel_id"]:
                try:
                    parsed_value = int(value)
                    # Validate if the provided channel ID actually exists in the guild.
                    if not interaction.guild.get_channel(parsed_value):
                        await interaction.followup.send(embed=discord.Embed(title="❌ Configuration Error", description=f"Channel ID `{parsed_value}` not found in this server.", color=discord.Color.red()))
                        return
                except ValueError:
                    await interaction.followup.send(embed=discord.Embed(title="❌ Configuration Error", description=f"Invalid channel ID: `{value}`. Must be a number.", color=discord.Color.red()))
                    return
            elif setting_name == "allowed_languages":
                # Parse comma-separated language codes, validate as 2-letter codes.
                parsed_value = [lang.strip().lower() for lang in value.split(',') if lang.strip() and len(lang.strip()) == 2]
                if not parsed_value:
                    await interaction.followup.send(embed=discord.Embed(title="❌ Configuration Error", description="No valid 2-letter language codes provided. Example: `en,fr,es`.", color=discord.Color.red()))
                    return
            elif setting_name == "allowed_topics":
                # Parse comma-separated topics.
                parsed_value = [topic.strip().lower() for topic in value.split(',') if topic.strip()]
                if not parsed_value:
                    await interaction.followup.send(embed=discord.Embed(title="❌ Configuration Error", description="No valid topics provided. Example: `gaming,politics`.", color=discord.Color.red()))
                    return
            elif setting_name == "permitted_domains":
                # Parse comma-separated domains, extract TLD+1.
                parsed_value = [get_domain_from_url(d.strip()) for d in value.split(',') if d.strip()]
                parsed_value = [d for d in parsed_value if d] # Remove None entries from failed parsing
                if not parsed_value:
                    await interaction.followup.send(embed=discord.Embed(title="❌ Configuration Error", description="No valid domains provided. Example: `google.com,discord.gg`.", color=discord.Color.red()))
                    return

        # Determine the database key to use and update the configuration.
        if is_channel_specific_type:
            db_key_to_use = f"channel_config_{config_context_channel.id}"
            # For channel-specific settings, we store a JSON dictionary in the DB.
            # We need to fetch the current settings, update the specific sub-key, then save it back.
            current_channel_settings = await get_guild_config(guild_id, db_key_to_use, {})
            if not isinstance(current_channel_settings, dict): current_channel_settings = {} # Ensure it's a dict

            json_subkey = setting_name.replace("allowed_", "") # e.g., "language" or "topics"
            if parsed_value is None: # Clearing the sub-key
                if json_subkey in current_channel_settings: del current_channel_settings[json_subkey]
            else:
                current_channel_settings[json_subkey] = parsed_value

            # Store the updated dictionary. If it becomes empty, store None to clear the DB entry.
            await set_guild_config(guild_id, db_key_to_use, current_channel_settings if current_channel_settings else None)
            success_message_detail = f"Setting `{setting_name}` for channel {config_context_channel.mention} " + \
                                     (f"set to `{parsed_value}`." if parsed_value is not None else "cleared.")
        else: # Guild-wide setting
            db_key_to_use = setting_name
            await set_guild_config(guild_id, db_key_to_use, parsed_value)
            success_message_detail = f"Guild setting `{setting_name}` " + \
                                     (f"set to `{parsed_value}`." if parsed_value is not None else "cleared (reverted to default).")

        embed = discord.Embed(title="✅ Configuration Updated", description=success_message_detail, color=discord.Color.green())
        await interaction.followup.send(embed=embed)
        # Log the configuration change to the moderation log channel.
        await log_moderation_action("config_change", interaction.user, success_message_detail, interaction.user, interaction.guild, color=discord.Color.blue())


    @app_commands.command(name="get_config", description="View current bot configurations for this guild or a channel.")
    @app_commands.default_permissions(manage_guild=True) # Only users with Manage Guild permission can use this.
    @app_commands.describe(target_channel="[Optional] Channel to view specific settings for (e.g., allowed_languages, allowed_topics).")
    async def view_configuration(self, interaction: discord.Interaction, target_channel: discord.TextChannel | None = None):
        await interaction.response.defer(ephemeral=True)
        guild_id = interaction.guild_id

        # Determine the channel context for displaying settings.
        display_channel_context = target_channel or interaction.channel

        embed = discord.Embed(title=f"⚙️ Bot Configuration for {interaction.guild.name}", color=discord.Color.blue())
        embed.set_footer(text=f"Requested by {interaction.user.display_name} | Guild ID: {guild_id}")

        # --- Guild-wide settings ---
        log_ch_id = await get_guild_config(guild_id, "log_channel_id", bot_config.default_log_channel_id)
        log_ch_obj = interaction.guild.get_channel(log_ch_id) if log_ch_id else None
        embed.add_field(name="🪵 Log Channel", value=f"{log_ch_obj.mention} (`{log_ch_id}`)" if log_ch_obj else "Not Set (Logs only to console)", inline=False)

        link_ch_id = await get_guild_config(guild_id, "link_channel_id", None)
        link_ch_obj = interaction.guild.get_channel(link_ch_id) if link_ch_id else None
        embed.add_field(name="🔗 Link-Only Channel", value=f"{link_ch_obj.mention} (`{link_ch_id}`)" if link_ch_obj else "Not Set (Domain checks apply everywhere)", inline=False)

        perm_domains = await get_guild_config(guild_id, "permitted_domains", list(bot_config.permitted_domains))
        perm_domains_str = ", ".join([f"`{d}`" for d in perm_domains]) if perm_domains else "Using Bot Defaults"
        if len(perm_domains_str) > 1000: perm_domains_str = perm_domains_str[:1000] + "..." # Truncate if too long
        embed.add_field(name="🌐 Permitted Link Domains (Guild)", value=perm_domains_str, inline=False)

        # --- Channel-specific settings for the displayed channel context ---
        embed.add_field(name="\u200b", value=f"--- Settings for Channel: {display_channel_context.mention} ---", inline=False) # Separator

        channel_settings_key = f"channel_config_{display_channel_context.id}"
        channel_db_cfg = await get_guild_config(guild_id, channel_settings_key, {})
        # Merge default channel config with DB overrides for effective settings.
        effective_channel_cfg = bot_config.default_channel_configs.get(display_channel_context.id, {}).copy()
        if isinstance(channel_db_cfg, dict): effective_channel_cfg.update(channel_db_cfg)

        allowed_langs = effective_channel_cfg.get("language")
        langs_str = ", ".join([f"`{lang}`" for lang in allowed_langs]) if allowed_langs else "Any Language Allowed"
        embed.add_field(name="🗣️ Allowed Languages", value=langs_str, inline=True)

        allowed_topics = effective_channel_cfg.get("topics")
        topics_str = ", ".join([f"`{t}`" for t in allowed_topics]) if allowed_topics else "Any Topic Allowed"
        embed.add_field(name="💬 Allowed Topics", value=topics_str, inline=True)

        if not target_channel:
             embed.add_field(name="ℹ️ Note", value=f"Channel settings shown for current channel ({display_channel_context.mention}). Use `target_channel` option for others.", inline=False)

        await interaction.followup.send(embed=embed)


class UtilityCommands(commands.Cog, name="Utilities"):
    """General utility commands for the bot."""
    def __init__(self, bot_instance: commands.Bot):
        self.bot = bot_instance

    @app_commands.command(name="awake", description="Check if the bot is responsive.")
    async def awake_slash(self, interaction: discord.Interaction):
        """Responds with bot's latency to check responsiveness."""
        latency_ms = self.bot.latency * 1000
        embed = discord.Embed(
            title="🤖 Adroit Bot Status",
            description="I am Adroit. I am always awake. Never sleeping.",
            color=discord.Color.green()
        )
        embed.add_field(name="Discord API Latency", value=f"{latency_ms:.2f} ms", inline=False)
        embed.set_footer(text=f"Bot Version 1.0 | Running on Discord.py {discord.__version__}")
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="classify_language", description="Classify the language of a given text.")
    @app_commands.describe(text_to_classify="The text you want to classify.")
    async def classify_language_slash(self, interaction: discord.Interaction, text_to_classify: str):
        """Classifies the language of input text using FastText and provides confidence."""
        await interaction.response.defer(ephemeral=True)
        if not LANGUAGE_MODEL:
            await interaction.followup.send("Language model is not loaded, cannot classify. Please check bot logs for FastText errors.", ephemeral=True)
            return
        if not text_to_classify.strip():
            await interaction.followup.send("Cannot classify empty text.", ephemeral=True)
            return

        lang_code, confidence = await detect_language_ai(text_to_classify)

        embed = discord.Embed(title="📝 Language Classification Result", color=discord.Color.blue())
        embed.add_field(name="Input Text Snippet", value=f"```\n{discord.utils.escape_markdown(text_to_classify[:250])}{'...' if len(text_to_classify) > 250 else ''}\n```", inline=False)
        if lang_code:
            embed.add_field(name="Detected Language", value=f"**{lang_code.upper()}**", inline=True)
            embed.add_field(name="Confidence", value=f"{confidence:.2%}", inline=True)
        else:
            embed.add_field(name="Detection Failed", value="Could not determine the language. The text might be too short or ambiguous.", inline=False)

        # Provide context based on channel's allowed languages.
        channel_cfg_key = f"channel_config_{interaction.channel_id}"
        channel_db_settings = await get_guild_config(interaction.guild_id, channel_cfg_key, {})
        effective_channel_cfg = bot_config.default_channel_configs.get(interaction.channel_id, {}).copy()
        if isinstance(channel_db_settings, dict): effective_channel_cfg.update(channel_db_settings)

        allowed_channel_langs = effective_channel_cfg.get("language")
        if allowed_channel_langs and lang_code:
            status = "✅ Allowed in this channel" if lang_code in allowed_channel_langs else "⚠️ Not typically allowed in this channel"
            embed.add_field(name="Channel Status", value=f"{status} (Configured: {', '.join(allowed_channel_langs)})", inline=False)

        await interaction.followup.send(embed=embed, ephemeral=True)


# --- Database Initialization ---
async def initialize_database():
    """
    Initializes the SQLite database and creates necessary tables if they don't exist.
    Configures PRAGMA settings for better performance and concurrency.
    """
    global db_conn
    try:
        db_conn = await aiosqlite.connect('adroit_bot_database.db')
        # Set PRAGMA for Write-Ahead Logging (WAL) mode for better concurrency with SQLite.
        await db_conn.execute("PRAGMA journal_mode=WAL;")
        await db_conn.execute("PRAGMA synchronous=NORMAL;") # Balances data integrity and performance.
        await db_conn.execute("PRAGMA foreign_keys=ON;") # Enforce foreign key constraints.

        # Create `guild_configs` table: Stores guild-specific settings.
        await db_conn.execute('''
            CREATE TABLE IF NOT EXISTS guild_configs (
                guild_id INTEGER NOT NULL,
                config_key TEXT NOT NULL,
                config_value TEXT, -- Stores values (e.g., JSON strings for lists/dicts)
                PRIMARY KEY (guild_id, config_key)
            )
        ''')
        # Create `infractions` table: Logs all moderation infractions for users.
        await db_conn.execute('''
            CREATE TABLE IF NOT EXISTS infractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL,
                points INTEGER NOT NULL,
                timestamp TEXT NOT NULL,    -- ISO format UTC datetime for when the infraction occurred
                violation_type TEXT,        -- Comma-separated list of violation types (e.g., "spam,nsfw_text")
                message_id INTEGER,         -- ID of the message that caused the infraction
                channel_id INTEGER,         -- ID of the channel where the infraction occurred
                message_content TEXT        -- Snippet of the offending message content
            )
        ''')
        # Index for faster queries on user, guild, and time for infraction history.
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_infractions_user_guild_time ON infractions (user_id, guild_id, timestamp)')

        # Create `temp_bans` table: Stores information about temporary bans for automatic unbanning.
        await db_conn.execute('''
            CREATE TABLE IF NOT EXISTS temp_bans (
                user_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL,
                unban_time TEXT NOT NULL,   -- ISO format UTC datetime for when the ban expires
                ban_reason TEXT,
                PRIMARY KEY (user_id, guild_id) -- A user can only have one active temp ban per guild
            )
        ''')
        # Index for faster lookup of expired temporary bans.
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_temp_bans_unban_time ON temp_bans (unban_time)')

        # Create `message_history` table: Stores recent message hashes for spam detection.
        await db_conn.execute('''
            CREATE TABLE IF NOT EXISTS message_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,        -- Unix timestamp for easy time calculations
                message_content_hash INTEGER    -- Hash of the message content for repetition checks
            )
        ''')
        # Index for faster queries on user, guild, and time for spam checks.
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_message_history_user_guild_time ON message_history (user_id, guild_id, timestamp)')

        await db_conn.commit() # Commit all table creation and PRAGMA changes.
        logger.info("✅ Database initialized successfully with WAL mode and schema.")
    except Exception as e:
        logger.critical(f"❌ CRITICAL: Failed to initialize database: {e}", exc_info=True)
        # If DB initialization fails, the bot cannot function correctly, so exit.
        if db_conn: await db_conn.close() # Attempt to close if partially opened.
        db_conn = None # Ensure it's None so other parts of the code know it failed.
        exit(1) # Exit the application.

# --- Bot Startup Sequence ---
@bot.event
async def on_ready():
    """
    Called when the bot is fully connected to Discord and ready to operate.
    This is where Cogs are added, background tasks are started, and slash commands are synced.
    """
    logger.info(f'🚀 Logged in as {bot.user.name} (ID: {bot.user.id})')
    logger.info(f'🔗 Discord.py Version: {discord.__version__}')
    logger.info(f'🌐 Connected to {len(bot.guilds)} guilds.')
    logger.info('------')

    # Initialize FastText model for language detection.
    global LANGUAGE_MODEL
    if FASTTEXT_MODEL_PATH:
        try:
            LANGUAGE_MODEL = fasttext.load_model(FASTTEXT_MODEL_PATH)
            logger.info(f"✅ FastText language model loaded from '{FASTTEXT_MODEL_PATH}'.")
        except ValueError as e:
            if "Cannot load" in str(e) or "not found" in str(e).lower():
                logger.error(f"❌ FastText model file '{FASTTEXT_MODEL_PATH}' not found or invalid. Language detection will be unavailable. {e}")
            else:
                logger.error(f"❌ Error loading FastText model: {e}. Language detection unavailable.")
            LANGUAGE_MODEL = None
        except Exception as e:
            logger.error(f"❌ Unexpected error loading FastText model: {e}", exc_info=True)
            LANGUAGE_MODEL = None
    else:
        logger.warning("FASTTEXT_MODEL_PATH not set. Language detection will be unavailable.")

    # Add Cogs (collections of commands and event listeners).
    await bot.add_cog(Moderation(bot))
    await bot.add_cog(ConfigCommands(bot))
    await bot.add_cog(UtilityCommands(bot))
    logger.info("📚 Cogs loaded.")

    # Start background tasks if they are not already running.
    if not cleanup_and_decay_task.is_running():
        cleanup_and_decay_task.start()

    # Sync slash commands with Discord. This makes your slash commands visible and usable.
    try:
        synced_cmds = await bot.tree.sync()
        logger.info(f"✅ Synced {len(synced_cmds)} application (slash) commands.")
    except Exception as e:
        logger.error(f"❌ Failed to sync application commands: {e}", exc_info=True)

    # Start HTTP health check server as a non-blocking asyncio task.
    asyncio.create_task(start_http_health_server())

    # Set the bot's activity status.
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="over servers | /help"))
    logger.info("🤖 Bot is ready and watching!")


@bot.event
async def on_disconnect():
    """Called when the bot loses connection to Discord."""
    logger.warning("🔌 Bot disconnected from Discord. Will attempt to reconnect automatically.")

@bot.event
async def on_resumed():
    """Called when the bot successfully resumes its connection to Discord."""
    logger.info("✅ Bot resumed connection to Discord.")


async def main_async_runner():
    """
    Main asynchronous function to initialize resources (HTTP session, database)
    and run the bot. This is the entry point for the bot's operations.
    """
    global http_session, db_conn

    # Initialize aiohttp client session for making HTTP requests to external APIs.
    http_session = ClientSession()
    logger.info("🌍 Aiohttp ClientSession initialized.")

    # Initialize the SQLite database.
    await initialize_database()
    if not db_conn: # If DB initialization failed, gracefully exit.
        logger.critical("Database initialization failed. Bot cannot start.")
        if http_session and not http_session.closed: await http_session.close()
        return

    # Start the bot. The `async with bot:` ensures proper startup and shutdown.
    try:
        async with bot:
            await bot.start(DISCORD_TOKEN)
    except discord.LoginFailure:
        logger.critical("CRITICAL: Invalid Discord token. Please check your ADROIT_TOKEN environment variable.")
    except Exception as e:
        logger.critical(f"CRITICAL: An unhandled exception occurred during bot runtime: {e}", exc_info=True)
    finally:
        logger.info("🔌 Bot shutting down. Cleaning up resources...")
        # Cancel all running background tasks to ensure a clean shutdown.
        if cleanup_and_decay_task.is_running():
            cleanup_and_decay_task.cancel()
            logger.info("Cleanup and decay task cancelled.")

        # Close aiohttp session.
        if http_session and not http_session.closed:
            await http_session.close()
            logger.info("Aiohttp session closed.")

        # Close database connection.
        if db_conn:
            await db_conn.close()
            logger.info("Database connection closed.")

        # Optional: Cancel any other outstanding asyncio tasks.
        # This part is commented out as it needs careful handling not to cancel the main loop itself.
        # current_task = asyncio.current_task()
        # all_other_tasks = [t for t in asyncio.all_tasks() if t is not current_task]
        # if all_other_tasks:
        #     logger.info(f"Cancelling {len(all_other_tasks)} outstanding asyncio tasks...")
        #     for task in all_other_tasks:
        #         task.cancel()
        #     await asyncio.gather(*all_other_tasks, return_exceptions=True)
        #     logger.info("Outstanding tasks processing complete.")

        logger.info("✅ Cleanup complete. Adroit Bot is offline.")

if __name__ == "__main__":
    # This block ensures the main async runner is executed when the script is run directly.
    try:
        asyncio.run(main_async_runner())
    except KeyboardInterrupt:
        logger.info("⌨️ Bot shutdown initiated by KeyboardInterrupt (Ctrl+C).")
    except Exception as e: # Catch-all for truly unhandled exceptions at the very top level.
        logger.critical(f"💥 UNHANDLED EXCEPTION IN MAIN RUNNER: {e}", exc_info=True)
    finally:
        logger.info("🏁 Main process finished.")

