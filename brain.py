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
intents.presences = False # Presences are not used, keep False for resource efficiency

bot = commands.Bot(command_prefix=">>", intents=intents, help_command=None)

# --- Global Variables (to be initialized) ---
db_conn: aiosqlite.Connection | None = None
LANGUAGE_MODEL = None # FastText model instance
http_session: ClientSession | None = None


class BotConfig:
    """Holds all static configurations for the bot."""
    def __init__(self):
        self.default_log_channel_id = 1113377818424922132 # Example ID
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
        self.url_pattern = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|net|org|io|dev|xyz|gg|app)\b)") # Expanded TLDs slightly
        self.has_alphanumeric_pattern = re.compile(r'[a-zA-Z0-9]')
        self.permitted_domains = [ # Default list of globally permitted domains
            "googleusercontent.com", "tenor.com", "giphy.com", "tiktok.com", "youtube.com", "youtu.be",
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
                50: {"action": "temp_ban", "duration_days": 30, "reason": "Severe/Accumulated violations"}, # Changed from months to days for timedelta
                10000: {"action": "ban", "reason": "Extreme/Manual Escalation"} # Effectively permanent / admin override
            },
            "violations": {
                "discrimination": {"points": 10, "severity": "High"}, # Increased points
                "spam": {"points": 1, "severity": "Low"},
                "nsfw_text": {"points": 3, "severity": "Medium"}, # Renamed from "nsfw" for clarity
                "nsfw_media": {"points": 5, "severity": "High"},
                "advertising": {"points": 2, "severity": "Medium"},
                "politics_discussion_disallowed": {"points": 1, "severity": "Low"}, # More specific name
                "off_topic": {"points": 1, "severity": "Low"},
                "foreign_language": {"points": 1, "severity": "Low"},
                "openai_flagged": {"points": 3, "severity": "Medium"}, # Renamed
                "excessive_mentions": {"points": 1, "severity": "Low"},
                "excessive_attachments": {"points": 1, "severity": "Low"},
                "long_message": {"points": 1, "severity": "Low"},
                "gore_violence_media": {"points": 7, "severity": "High"}, # New category for Sightengine
                "offensive_symbols_media": {"points": 5, "severity": "High"} # New category for Sightengine
            }
        }
        # Moderation thresholds and limits
        self.spam_window_seconds = 10
        self.spam_message_limit = 5 # Max messages in spam_window_seconds
        self.spam_repetition_limit = 3 # Max identical messages in spam_window_seconds
        self.mention_limit = 5
        self.max_message_length = 1000 # Increased slightly
        self.max_attachments = 4
        self.min_msg_len_for_lang_check = 4
        self.min_confidence_for_lang_flagging = 0.65
        self.min_confidence_short_msg_lang = 0.75
        self.short_msg_threshold_lang = 20
        self.common_safe_foreign_words = {"bonjour", "hola", "merci", "gracias", "oui", "si", "nyet", "da", "salut", "ciao", "hallo", "guten tag", "privet", "konnichiwa", "arigato"}
        self.fuzzy_match_threshold = 85 # Increased slightly for better accuracy

        # Sightengine thresholds
        self.sightengine_nudity_sexual_activity_threshold = 0.6
        self.sightengine_nudity_suggestive_threshold = 0.8 # Higher threshold for suggestive
        self.sightengine_gore_threshold = 0.7
        self.sightengine_violence_threshold = 0.7
        self.sightengine_offensive_symbols_threshold = 0.85 # For nazi, confederate, etc.
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
    return text.strip().lower() # Keep it simple: strip and lower

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
    clean_text = text.strip() # Clean for FastText, but don't lowercase here as model might be case-sensitive
    if not clean_text:
        return None, 0.0

    if not LANGUAGE_MODEL:
        logger.warning("FastText model not loaded. Cannot detect language.")
        return None, 0.0
    try:
        # FastText expects a single string, newlines are okay.
        prediction = LANGUAGE_MODEL.predict(clean_text.replace('\n', ' '), k=1) # Replace newlines for predict
        if prediction and prediction[0] and prediction[1] and len(prediction[0]) > 0:
            lang_code = prediction[0][0].replace("__label__", "")
            confidence = float(prediction[1][0])
            return lang_code, confidence
        else:
            logger.warning(f"FastText returned unexpected prediction format for: '{clean_text[:100]}...'")
            return None, 0.0
    except Exception as e:
        logger.error(f"FastText language detection error for '{clean_text[:100]}...': {e}", exc_info=True)
        return None, 0.0 # Return None for lang if error

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
    stop=stop_after_attempt(4), # Max 4 attempts
    wait=wait_random_exponential(multiplier=1, min=3, max=60), # Exponential backoff
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
                return results[0] # Contains "flagged", "categories", "category_scores"
            logger.warning(f"OpenAI moderation returned empty results for text: {text_content[:100]}...")
            return {"flagged": False, "categories": {}, "category_scores": {}}
    except client_exceptions.ClientResponseError as e:
        if e.status == 400: # Bad Request (e.g., invalid input for OpenAI)
            logger.warning(f"OpenAI moderation: 400 Bad Request (will not retry). Text: '{text_content[:100]}...'. Error: {e.message}")
            return {"flagged": False, "categories": {}, "category_scores": {}} # Don't retry 400
        logger.error(f"OpenAI moderation API error: {e.status} - {e.message}. Text: '{text_content[:100]}...'. Retrying if applicable.")
        raise # Re-raise for tenacity
    except asyncio.TimeoutError:
        logger.error(f"OpenAI moderation API request timed out. Text: {text_content[:100]}...")
        raise # Re-raise for tenacity
    except Exception as e:
        logger.error(f"Unexpected error with OpenAI moderation API: {e} for text: {text_content[:100]}...", exc_info=True)
        # Fallback if tenacity gives up or for non-retryable errors not caught above
        logger.critical(f"OpenAI moderation failed definitively for text: '{text_content[:100]}...'. Moderation skipped.")
        return {"flagged": False, "categories": {}, "category_scores": {}}


async def apply_moderation_punishment(member: discord.Member, action: str, reason: str, duration: timedelta | None = None, moderator: discord.User | None = None):
    """Applies a specified moderation action to a member and logs it."""
    dm_message_text = ""
    log_color = discord.Color.orange()
    extra_log_fields = []

    try:
        if action == "warn":
            warn_config = bot_config.punishment_system["points_thresholds"].get(5, {}) # Get the 5-point threshold config for the DM message
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
            await member.kick(reason=reason) # Kick after attempting DM
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
                dm_message_text = f"You have been temporarily banned from **{member.guild.name}** until **{unban_time.strftime('%Y-%m-%d %H:%M:%S UTC')}** ({str(duration)} from now).\nReason: {reason}"
                await member.ban(reason=reason, delete_message_days=0) # Ban after attempting DM & DB log
                log_color = discord.Color.dark_red()
                extra_log_fields.append(("Unban Time", unban_time.strftime('%Y-%m-%d %H:%M:%S UTC')))
            else:
                logger.warning(f"Attempted to temp_ban {member.display_name} without duration. Action skipped.")
                return
        elif action == "ban":
            dm_message_text = f"You have been permanently banned from **{member.guild.name}**.\nReason: {reason}"
            await member.ban(reason=reason, delete_message_days=0) # Ban after attempting DM
            log_color = discord.Color.black() # Very severe

        # Attempt to DM the user
        if dm_message_text:
            try:
                dm_embed = discord.Embed(title=f"Moderation Action: {action.title()}", description=dm_message_text, color=log_color)
                dm_embed.set_footer(text=f"Guild: {member.guild.name}")
                await member.send(embed=dm_embed)
            except discord.Forbidden:
                logger.warning(f"Could not DM {action} notification to {member.display_name} ({member.id}). They may have DMs disabled.")
            except discord.HTTPException as e:
                 logger.error(f"Failed to send DM for {action} to {member.display_name}: {e}", exc_info=True)


        # Log the action
        await log_moderation_action(action, member, reason, moderator, member.guild, color=log_color, extra_fields=extra_log_fields)

    except discord.Forbidden:
        logger.error(f"BOT PERMISSION ERROR: Missing permissions to {action} member {member.display_name} ({member.id}) in guild {member.guild.name}. Check role hierarchy and bot permissions.")
        if moderator and isinstance(moderator, discord.Member): # Notify moderator if manual action fails due to bot perms
            try:
                await moderator.send(f"Error: I don't have sufficient permissions to `{action}` user `{member.display_name}` in `{member.guild.name}`. Please check my roles and permissions.")
            except discord.Forbidden:
                pass # Can't notify moderator either
    except discord.HTTPException as e:
        logger.error(f"Discord API error while applying {action} to {member.display_name}: {e.status} - {e.text}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error applying {action} to {member.display_name}: {e}", exc_info=True)


async def add_user_infraction(member: discord.Member, points_to_add: int, violation_types_str: str, original_message: discord.Message):
    """Logs an infraction, accumulates points, and triggers automated punishments if thresholds are met."""
    if not db_conn:
        logger.error("add_user_infraction: Database connection is not available.")
        return
    guild_id = member.guild.id
    user_id = member.id

    try:
        async with db_conn.cursor() as cursor:
            await cursor.execute(
                'INSERT INTO infractions (user_id, guild_id, points, timestamp, violation_type, message_id, channel_id, message_content) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (user_id, guild_id, points_to_add, datetime.utcnow().isoformat(), violation_types_str, original_message.id, original_message.channel.id, original_message.content[:200]) # Log snippet
            )
            # Get total active points (e.g., last 30 days)
            thirty_days_ago_iso = (datetime.utcnow() - timedelta(days=30)).isoformat()
            await cursor.execute(
                'SELECT SUM(points) FROM infractions WHERE user_id = ? AND guild_id = ? AND timestamp >= ?',
                (user_id, guild_id, thirty_days_ago_iso)
            )
            total_points_data = await cursor.fetchone()
            current_total_points = total_points_data[0] if total_points_data and total_points_data[0] is not None else 0
        await db_conn.commit()

        logger.info(f"User {member.display_name} ({user_id}) in guild {guild_id} received {points_to_add} points for '{violation_types_str}'. Total active points: {current_total_points}.")

        # Check for automated punishment
        # Iterate thresholds from highest to lowest to apply the most severe one met
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
                # Removed "duration_months" as timedelta doesn't directly support it; use days.

                logger.info(f"Threshold of {threshold} points met by {member.display_name} (Total: {current_total_points}). Applying '{action_to_take}'.")
                await apply_moderation_punishment(member, action_to_take, full_reason, duration, moderator=bot.user) # Moderator is the bot for automated actions
                break # Apply only one punishment (the highest one met)
    except Exception as e:
        logger.error(f"Error in add_user_infraction for {member.display_name}: {e}", exc_info=True)


def get_domain_from_url(url_string: str) -> str | None:
    """Extracts the TLD+1 domain from a given URL string."""
    try:
        if not url_string.startswith(('http://', 'https://')):
            url_string = 'https://' + url_string
        parsed_url = urlparse(url_string)
        domain = parsed_url.netloc
        if domain:
            domain = domain.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            # Basic handling for common subdomains if needed, but usually TLD+1 is sufficient
            # e.g. for something like "cdn.example.co.uk", netloc is "cdn.example.co.uk"
            # For simplicity, we'll consider the full netloc (minus www) as the domain for matching.
            return domain
        return None
    except ValueError: # Handles malformed URLs for urlparse
        logger.debug(f"Could not parse domain from potentially malformed URL: '{url_string}'")
        return None

def is_domain_permitted(domain_to_check: str, guild_specific_permitted_domains: list[str]) -> bool:
    """Checks if the domain_to_check is in the guild's permitted list or is a subdomain of a permitted one."""
    if not domain_to_check:
        return False
    
    # Combine guild-specific with global defaults if guild list isn't overriding completely
    # For now, assume guild_specific_permitted_domains is the definitive list if set, else global.
    # This logic is handled by how get_guild_config provides the list.

    for permitted_domain_pattern in guild_specific_permitted_domains:
        # Permitted domains might be stored as "example.com"
        # We want to match "example.com" and "sub.example.com"
        if domain_to_check == permitted_domain_pattern or domain_to_check.endswith('.' + permitted_domain_pattern):
            return True
    return False

# --- HTTP Server for Health Checks (Optional, for deployment platforms) ---
async def handle_health_check_request(request: web.Request):
    """Handles HTTP health check requests."""
    return web.Response(text=f"Adroit Bot is healthy and awake! Version: 1.0. Time: {datetime.utcnow()}", status=200, content_type="text/plain")

async def start_http_health_server():
    """Starts a simple aiohttp server for health checks."""
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
        # Keep it running in the background. If this task is awaited, it blocks.
        # If it needs to be stoppable, it should be managed as an asyncio.Task.
    except ValueError:
        logger.error(f"❌ Invalid PORT environment variable: '{port_str}'. Must be an integer. HTTP server not started.")
    except Exception as e:
        logger.error(f"❌ Failed to start HTTP health check server: {e}", exc_info=True)

# --- Background Tasks ---
@tasks.loop(hours=12) # Reduced frequency, 24h might be too long for some cleanups
async def cleanup_and_decay_task():
    """Periodically cleans up old data: infractions, temp bans, message history."""
    if not db_conn:
        logger.warning("cleanup_and_decay_task: Database connection not available. Skipping.")
        return
    
    logger.info("Starting cleanup and decay task...")
    try:
        async with db_conn.cursor() as cursor:
            # 1. Delete very old infraction records (e.g., older than 90 days to keep DB lean)
            ninety_days_ago_iso = (datetime.utcnow() - timedelta(days=90)).isoformat()
            await cursor.execute('DELETE FROM infractions WHERE timestamp < ?', (ninety_days_ago_iso,))
            deleted_infractions = cursor.rowcount
            if deleted_infractions > 0:
                logger.info(f"Deleted {deleted_infractions} infraction records older than 90 days.")

            # 2. Process expired temporary bans
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
                        
                        # Try to fetch user for richer logging, but don't fail if user is gone
                        target_user_for_log = await bot.fetch_user(user_id) rescue None 
                        log_reason = f"Temporary ban expired. Original reason: {ban_reason or 'Not specified.'}"
                        await log_moderation_action("unban", target_user_for_log or user_to_unban_obj, log_reason, bot.user, guild, color=discord.Color.green())
                    except discord.NotFound: # Ban not found on Discord's side
                        logger.warning(f"User {user_id} not found in ban list of guild {guild.name} for auto-unban, or already unbanned.")
                    except discord.Forbidden:
                        logger.error(f"BOT PERMISSION ERROR: Missing permissions to unban user {user_id} in guild {guild.name}.")
                    except Exception as e:
                        logger.error(f"Error during automatic unban for user {user_id} in guild {guild.name}: {e}", exc_info=True)
                else:
                    logger.warning(f"Cannot process temp ban expiry for user {user_id} in guild {guild_id}: Bot is no longer in this guild.")
                
                # Delete the processed temp_ban entry regardless of unban success (to prevent re-processing)
                await cursor.execute('DELETE FROM temp_bans WHERE user_id = ? AND guild_id = ? AND unban_time <= ?', (user_id, guild_id, now_iso))
            
            if expired_bans_data: # Commit if any bans were processed
                 logger.info(f"Processed {len(expired_bans_data)} expired temporary bans.")


            # 3. Clean up old message history (e.g., older than 10 minutes for spam checks)
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
    await bot.wait_until_ready()
    logger.info("Cleanup and decay task is ready and will start looping.")


# --- Event Handlers ---
@bot.event
async def on_error(event_method_name: str, *args, **kwargs):
    """Global error handler for unhandled exceptions in bot events."""
    logger.error(f"Unhandled error in event '{event_method_name}': Args: {args}, Kwargs: {kwargs}", exc_info=True)
    # Potentially log to a Discord channel if critical, but be careful about rate limits/loops.

@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    """Centralized error handling for traditional prefix commands."""
    embed_title = "Command Error"
    embed_color = discord.Color.red()
    error_message = f"An unexpected error occurred: {error}"

    if isinstance(error, commands.CommandNotFound):
        return # Usually best to ignore, or send a very subtle ephemeral message if it's a common typo.
    elif isinstance(error, commands.MissingRequiredArgument):
        embed_title = "Missing Argument"
        error_message = f"You missed the `{error.param.name}` argument. Please check the command's usage."
        # Consider: await ctx.send_help(ctx.command)
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
    elif isinstance(error, commands.CheckFailure): # Generic check failure
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
        await ctx.send(embed=embed, ephemeral=True if isinstance(ctx, discord.Interaction) else False) # Ephemeral for slash, regular for prefix
    except discord.HTTPException:
        logger.error("Failed to send command error message to context.")


@bot.event
async def on_guild_join(guild: discord.Guild):
    """Logs when the bot joins a new guild and sends a welcome message if possible."""
    logger.info(f"Joined new guild: {guild.name} (ID: {guild.id}, Members: {guild.member_count}, Owner: {guild.owner_id})")
    
    # Try to send a welcome message to the system channel or first available text channel
    welcome_embed = discord.Embed(
        title=f"👋 Hello, {guild.name}!",
        description=(
            f"Thanks for adding Adroit Bot! I'm here to help with moderation.\n\n"
            f"**Quick Start:**\n"
            f"- Use `/configure setting:log_channel value:<your_log_channel_id>` to set up a logging channel.\n"
            f"- View all settings with `/get_config`.\n"
            f"- See available commands with `/help` (once a help command is added) or by exploring slash commands.\n\n"
            f"I'll do my best to keep your community safe and sound!"
        ),
        color=discord.Color.blue()
    )
    welcome_embed.set_footer(text="Adroit Bot | Advanced Moderation")
    welcome_embed.set_thumbnail(url=bot.user.display_avatar.url)

    sent_welcome = False
    if guild.system_channel and guild.system_channel.permissions_for(guild.me).send_messages:
        try:
            await guild.system_channel.send(embed=welcome_embed)
            sent_welcome = True
            logger.info(f"Sent welcome message to system channel in {guild.name}")
        except discord.HTTPException:
            pass # Fall through to try other channels

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
    """Handles message moderation, infraction tracking, and punishment application."""
    def __init__(self, bot_instance: commands.Bot):
        self.bot = bot_instance
        self.user_rate_limit_data = defaultdict(lambda: deque()) # For very rapid message spam

        # Load keyword/phrase lists (can be reloaded if files change, but not implemented here)
        self.discrimination_words, self.discrimination_phrases = discrimination_words_set, discrimination_phrases
        self.nsfw_text_words, self.nsfw_text_phrases = nsfw_text_words_set, nsfw_text_phrases
        
        # Start tasks associated with this cog
        self.cleanup_inmemory_rate_limit_data.start()
        # DB cleanup is handled by the global cleanup_and_decay_task

    def cog_unload(self):
        self.cleanup_inmemory_rate_limit_data.cancel()
        logger.info("Moderation cog's in-memory rate limit cleanup task cancelled.")

    @tasks.loop(minutes=10) # More frequent cleanup for in-memory rate limit data
    async def cleanup_inmemory_rate_limit_data(self):
        """Cleans up old timestamps from the in-memory rate limit deques."""
        now = datetime.utcnow()
        cleaned_users = 0
        # Iterate over a copy of keys if modifying the dict, though here we modify deques
        for user_id in list(self.user_rate_limit_data.keys()): # list() to avoid "dict changed size during iteration"
            user_deque = self.user_rate_limit_data[user_id]
            # Remove timestamps older than spam_window + a small buffer
            while user_deque and (now - user_deque[0]).total_seconds() >= (bot_config.spam_window_seconds + 5):
                user_deque.popleft()
            if not user_deque: # If deque becomes empty, remove user_id from dict
                del self.user_rate_limit_data[user_id]
                cleaned_users +=1
        if cleaned_users > 0:
            logger.debug(f"Cleaned up in-memory rate limit data for {cleaned_users} users.")

    @cleanup_inmemory_rate_limit_data.before_loop
    async def before_cleanup_inmemory_rate_limit_data(self):
        await self.bot.wait_until_ready()
        logger.info("In-memory rate limit data cleanup task is ready.")


    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if not message.guild or message.author.bot or not db_conn or not http_session:
            return # Ignore DMs, bot messages, or if essential services aren't ready

        # --- Basic Message Properties & User Context ---
        author_member = message.author # This is a Member object
        guild_id = message.guild.id
        user_id = author_member.id
        content_raw = message.content
        content_lower = clean_message_content(content_raw) # For case-insensitive checks
        now = datetime.utcnow()
        violations_found = set() # Store unique violation types for this message

        # --- 0. Role-based Immunity (Example) ---
        # This should be configurable per guild, e.g., via /configure
        # For now, a simple check for "Admin" or "Moderator" role names
        # immune_roles = await get_guild_config(guild_id, "immune_role_ids", []) 
        # if any(role.id in immune_roles for role in author_member.roles):
        #    return # User has an immune role
        # Or, by permission:
        if author_member.guild_permissions.manage_messages: # Users who can manage messages are likely mods
            await self.bot.process_commands(message) # Still process commands for them
            return # Skip moderation checks for users with manage_messages permission

        # --- 1. Spam Detection (Rate Limiting & Repetition) ---
        # A. In-memory rate limiting for very rapid messages
        user_timestamps = self.user_rate_limit_data[user_id]
        while user_timestamps and (now - user_timestamps[0]).total_seconds() >= bot_config.spam_window_seconds:
            user_timestamps.popleft()
        user_timestamps.append(now)
        if len(user_timestamps) > bot_config.spam_message_limit:
            violations_found.add("spam")
            logger.debug(f"SPAM (Rate Limit): User {user_id} sent {len(user_timestamps)} msgs in {bot_config.spam_window_seconds}s.")

        # B. DB-backed message history for content repetition
        try:
            await db_conn.execute(
                "INSERT INTO message_history (user_id, guild_id, timestamp, message_content_hash) VALUES (?, ?, ?, ?)",
                (user_id, guild_id, now.timestamp(), hash(content_raw)) # Store hash of raw content
            ) # Not committing yet, will commit after all DB ops for this message or if violation

            # Check for repetition within the spam window
            repetition_check_ts_start = (now - timedelta(seconds=bot_config.spam_window_seconds)).timestamp()
            async with db_conn.execute(
                "SELECT COUNT(*) FROM message_history WHERE user_id = ? AND guild_id = ? AND message_content_hash = ? AND timestamp >= ?",
                (user_id, guild_id, hash(content_raw), repetition_check_ts_start)
            ) as cursor:
                repetition_count = (await cursor.fetchone())[0]
                if repetition_count >= bot_config.spam_repetition_limit: # Current message is already inserted, so >= limit
                    violations_found.add("spam")
                    logger.debug(f"SPAM (Repetition): User {user_id} repeated content '{content_raw[:30]}...' {repetition_count} times.")
        except Exception as e:
            logger.error(f"DB error during spam check for user {user_id}: {e}", exc_info=True)


        # --- 2. Basic Message Properties Violations ---
        if len(message.mentions) > bot_config.mention_limit: violations_found.add("excessive_mentions")
        if len(message.attachments) > bot_config.max_attachments: violations_found.add("excessive_attachments")
        if len(content_raw) > bot_config.max_message_length: violations_found.add("long_message")

        # --- 3. Fetch Channel & Guild Configurations ---
        # Channel-specific settings (e.g., language, topics)
        channel_db_config_key = f"channel_config_{message.channel.id}"
        raw_channel_db_settings = await get_guild_config(guild_id, channel_db_config_key, {})
        
        # Merge default channel config with DB overrides
        effective_channel_config = bot_config.default_channel_configs.get(message.channel.id, {}).copy()
        if isinstance(raw_channel_db_settings, dict): # Ensure DB value is a dict before updating
            effective_channel_config.update(raw_channel_db_settings)
        
        # Guild-wide settings
        guild_permitted_domains = await get_guild_config(guild_id, "permitted_domains", list(bot_config.permitted_domains))
        link_only_channel_id = await get_guild_config(guild_id, "link_channel_id", None)

        # --- 4. Language Check ---
        allowed_languages = effective_channel_config.get("language") # e.g., ["en", "fr"]
        if allowed_languages and content_lower: # Only if specific languages are enforced
            # Skip language check for very short messages, URLs, or non-alphanumeric content
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
                    
                    # Check common safe words or low confidence
                    if not (content_lower in bot_config.common_safe_foreign_words or confidence < conf_threshold):
                        violations_found.add("foreign_language")

        # --- 5. Advertising & Forbidden Links ---
        if bot_config.forbidden_text_pattern.search(content_lower):
            violations_found.add("advertising")

        urls_in_message = bot_config.url_pattern.findall(content_raw)
        if urls_in_message:
            # If a link-only channel is set and this is NOT that channel, check domains
            if link_only_channel_id and message.channel.id != link_only_channel_id:
                for url_match_groups in urls_in_message:
                    # url_pattern returns tuple of groups, the first is the full URL
                    url_str = url_match_groups[0] if isinstance(url_match_groups, tuple) else url_match_groups
                    domain = get_domain_from_url(url_str)
                    if domain and not is_domain_permitted(domain, guild_permitted_domains):
                        violations_found.add("advertising")
                        logger.debug(f"ADVERTISING (Forbidden Domain): {domain} in {url_str} by {user_id}")
                        break # One forbidden domain is enough
            elif not link_only_channel_id: # No specific link channel, check all domains everywhere
                 for url_match_groups in urls_in_message:
                    url_str = url_match_groups[0] if isinstance(url_match_groups, tuple) else url_match_groups
                    domain = get_domain_from_url(url_str)
                    if domain and not is_domain_permitted(domain, guild_permitted_domains):
                        violations_found.add("advertising")
                        logger.debug(f"ADVERTISING (Forbidden Domain): {domain} in {url_str} by {user_id}")
                        break

        # --- 6. Keyword/Phrase-based Moderation (Discrimination, NSFW Text) using Fuzzy Matching ---
        # Discrimination
        if not "discrimination" in violations_found: # Avoid redundant checks
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
        
        # --- 7. Off-topic Check (if topics are defined for the channel) ---
        allowed_topics = effective_channel_config.get("topics")
        if allowed_topics and content_lower: # Only if topics are enforced
            # This is a simple check; more advanced would involve NLP topic modeling
            if not any(topic.lower() in content_lower for topic in allowed_topics):
                # Could add a check here to ensure the message isn't just a greeting or very short
                # before flagging as off-topic. For now, direct check.
                violations_found.add("off_topic")
                logger.debug(f"OFF_TOPIC by {user_id}: '{content_raw[:30]}...' (Allowed: {allowed_topics})")


        # --- 8. Media NSFW/Gore/Violence Check (Sightengine for attachments) ---
        # Check only if no text-based NSFW already found, and if Sightengine is configured
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
                    # Only check first valid media attachment to save API calls, or configure for more.
                    break 

        # --- 9. OpenAI Moderation API Check ---
        # Run if content exists and hasn't already hit a major local filter like discrimination or NSFW
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
                # Map specific OpenAI categories to internal, more severe violations if desired
                categories = openai_result.get("categories", {})
                if categories.get("hate", False) or categories.get("hate/threatening", False):
                    violations_found.add("discrimination") # Escalate to discrimination
                if categories.get("sexual", False) or categories.get("sexual/minors", False):
                    violations_found.add("nsfw_text") # Escalate to nsfw_text
                # Potentially add self-harm to a specific category if needed

        # --- Process Violations ---
        if violations_found:
            if db_conn: await db_conn.commit() # Commit message history now that we have violations

            logger.info(f"Violations by {author_member.name} ({user_id}) in G:{guild_id}/C:{message.channel.id}: {', '.join(violations_found)}. Content: '{content_raw[:50]}...'")
            
            # Calculate total points for this message
            total_points_for_message = sum(
                bot_config.punishment_system["violations"].get(v_type, {}).get("points", 0)
                for v_type in violations_found
            )

            # Log infraction and apply punishment if points > 0
            if total_points_for_message > 0:
                await add_user_infraction(author_member, total_points_for_message, ", ".join(sorted(list(violations_found))), message)

            # Send a brief notification to the channel about message deletion
            try:
                violation_titles_str = ", ".join(sorted([v.replace('_', ' ').title() for v in violations_found]))
                warning_embed = discord.Embed(
                    description=f"{author_member.mention}, your message was removed due to: **{violation_titles_str}**. Please review server rules.",
                    color=discord.Color.orange()
                )
                await message.channel.send(embed=warning_embed, delete_after=20) # Auto-delete after 20s
            except discord.Forbidden:
                logger.warning(f"Missing permissions to send violation notification in #{message.channel.name}.")
            except Exception as e:
                logger.error(f"Error sending violation notification: {e}", exc_info=True)

            # Delete the offending message
            try:
                await message.delete()
                logger.info(f"Deleted message {message.id} from {author_member.name} due to violations.")
            except discord.Forbidden:
                logger.error(f"BOT PERMISSION ERROR: Missing permissions to delete message {message.id} in #{message.channel.name}.")
            except discord.NotFound:
                logger.warning(f"Message {message.id} not found for deletion (already deleted?).")
            except Exception as e:
                logger.error(f"Error deleting message {message.id}: {e}", exc_info=True)
        
        else: # No violations found
            if db_conn: await db_conn.commit() # Commit message history even if no violation
            await self.bot.process_commands(message) # Process commands if no violations


    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=2, max=15), retry=retry_if_api_error)
    async def check_media_content_sightengine(self, media_url: str) -> set[str]:
        """
        Checks media using Sightengine for various content types.
        Returns a set of violation strings if flagged, empty set otherwise.
        """
        if not http_session: # Should be initialized by now
            logger.error("Sightengine check: http_session is not available.")
            return set()

        params = {
            'url': media_url,
            'models': 'nudity-2.0,wad,offensive,gore,violence,text', # Comprehensive models
            'api_user': SIGHTENGINE_API_USER,
            'api_secret': SIGHTENGINE_API_SECRET
        }
        api_url = "https://api.sightengine.com/1.0/check.json"
        flagged_violations = set()

        try:
            async with http_session.get(api_url, params=params, timeout=20) as response:
                response_text = await response.text()
                response.raise_for_status() # Raises for 4xx/5xx HTTP errors
                
                data = json.loads(response_text) # Parse after raise_for_status
                logger.debug(f"Sightengine API response for {media_url}: {data}")

                if data.get("status") != "success":
                    error_info = data.get("error", {})
                    logger.warning(f"Sightengine API returned non-success status for {media_url}: {error_info.get('message', 'Unknown error')}")
                    if error_info.get('type') == 'media_error' and error_info.get('code') == 22: # GIF too many frames
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
                # Check for specific offensive symbols if their probability is high
                # Example: nazi, confederate, extremist symbols
                if offensive.get("nazi", 0.0) > bot_config.sightengine_offensive_symbols_threshold or \
                   offensive.get("confederate", 0.0) > bot_config.sightengine_offensive_symbols_threshold or \
                   offensive.get("supremacist", 0.0) > bot_config.sightengine_offensive_symbols_threshold or \
                   offensive.get("terrorist", 0.0) > bot_config.sightengine_offensive_symbols_threshold:
                    flagged_violations.add("offensive_symbols_media")
                
                # Text in Image Profanity Check
                text_profanity_score = data.get("text", {}).get("profanity", [{}])[0].get("prob", 0.0) # text.profanity is a list
                if text_profanity_score > bot_config.sightengine_text_profanity_threshold:
                     flagged_violations.add("nsfw_text") # Flag as nsfw_text if image contains profane text

                # WAD (Weapons, Alcohol, Drugs) - can be less critical, or map to specific violations
                # wad = data.get("wad", {})
                # if wad.get("drugs_prob", 0.0) > 0.8: flagged_violations.add("drug_media_reference") # Example

                return flagged_violations

        except client_exceptions.ClientResponseError as e:
            logger.error(f"Sightengine API HTTP error: {e.status} - {e.message} for {media_url}. URL: {e.request_info.url if e.request_info else 'N/A'}")
            if e.status == 400 and "cannot_load_image" in e.message.lower():
                 logger.warning(f"Sightengine could not load image at {media_url}. Assuming safe.")
                 return set()
            raise # Re-raise for tenacity
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
    """Commands for configuring bot settings per guild or channel."""
    def __init__(self, bot_instance: commands.Bot):
        self.bot = bot_instance

    @app_commands.command(name="configure", description="Configure bot settings for this guild or a specific channel.")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(
        setting_name="The setting to configure (e.g., log_channel, link_channel, allowed_languages, permitted_domains).",
        value="The value for the setting. For lists (languages, domains), use comma-separation. Use 'clear' to reset.",
        target_channel="[Optional] The channel for channel-specific settings (e.g., allowed_languages)."
    )
    @app_commands.choices(setting_name=[
        app_commands.Choice(name="Log Channel ID", value="log_channel_id"),
        app_commands.Choice(name="Link-Only Channel ID", value="link_channel_id"),
        app_commands.Choice(name="Allowed Languages (Channel)", value="allowed_languages"),
        # app_commands.Choice(name="Allowed Topics (Channel)", value="allowed_topics"), # Topics less common, can add later
        app_commands.Choice(name="Permitted Link Domains (Guild)", value="permitted_domains"),
    ])
    async def configure_setting(self, interaction: discord.Interaction, setting_name: str, value: str, target_channel: discord.TextChannel | None = None):
        await interaction.response.defer(ephemeral=True)
        guild_id = interaction.guild_id
        
        is_channel_specific_type = setting_name in ["allowed_languages", "allowed_topics"]
        
        # Determine the actual channel context for the setting
        config_context_channel = target_channel or interaction.channel # Default to interaction's channel if relevant and not specified
        
        db_key_to_use: str
        parsed_value: any = None
        success_message_detail = ""

        if value.lower() == 'clear':
            parsed_value = None # Signal to clear/delete the setting
        else:
            # Parse value based on setting_name
            if setting_name in ["log_channel_id", "link_channel_id"]:
                try:
                    parsed_value = int(value)
                    # Validate if channel exists in guild
                    if not interaction.guild.get_channel(parsed_value):
                        await interaction.followup.send(embed=discord.Embed(title="❌ Configuration Error", description=f"Channel ID `{parsed_value}` not found in this server.", color=discord.Color.red()))
                        return
                except ValueError:
                    await interaction.followup.send(embed=discord.Embed(title="❌ Configuration Error", description=f"Invalid channel ID: `{value}`. Must be a number.", color=discord.Color.red()))
                    return
            elif setting_name == "allowed_languages":
                parsed_value = [lang.strip().lower() for lang in value.split(',') if lang.strip() and len(lang.strip()) == 2] # Basic 2-letter code validation
                if not parsed_value:
                    await interaction.followup.send(embed=discord.Embed(title="❌ Configuration Error", description="No valid 2-letter language codes provided. Example: `en,fr,es`.", color=discord.Color.red()))
                    return
            elif setting_name == "permitted_domains":
                parsed_value = [get_domain_from_url(d.strip()) for d in value.split(',') if d.strip()]
                parsed_value = [d for d in parsed_value if d] # Remove None entries from failed parsing
                if not parsed_value:
                    await interaction.followup.send(embed=discord.Embed(title="❌ Configuration Error", description="No valid domains provided. Example: `google.com,discord.gg`.", color=discord.Color.red()))
                    return
            # Add elif for "allowed_topics" if re-enabled

        # Determine DB key
        if is_channel_specific_type:
            db_key_to_use = f"channel_config_{config_context_channel.id}"
            # For channel-specific, we store a JSON dict. We need to update a sub-key within it.
            current_channel_settings = await get_guild_config(guild_id, db_key_to_use, {})
            if not isinstance(current_channel_settings, dict): current_channel_settings = {} # Ensure it's a dict

            json_subkey = setting_name.replace("allowed_", "") # e.g., "language" or "topics"
            if parsed_value is None: # Clearing the sub-key
                if json_subkey in current_channel_settings: del current_channel_settings[json_subkey]
            else:
                current_channel_settings[json_subkey] = parsed_value
            
            await set_guild_config(guild_id, db_key_to_use, current_channel_settings if current_channel_settings else None) # Store None if dict becomes empty
            success_message_detail = f"Setting `{setting_name}` for channel {config_context_channel.mention} " + \
                                     (f"set to `{parsed_value}`." if parsed_value is not None else "cleared.")
        else: # Guild-wide setting
            db_key_to_use = setting_name
            await set_guild_config(guild_id, db_key_to_use, parsed_value)
            success_message_detail = f"Guild setting `{setting_name}` " + \
                                     (f"set to `{parsed_value}`." if parsed_value is not None else "cleared (reverted to default).")

        embed = discord.Embed(title="✅ Configuration Updated", description=success_message_detail, color=discord.Color.green())
        await interaction.followup.send(embed=embed)
        await log_moderation_action("config_change", interaction.user, success_message_detail, interaction.user, interaction.guild, color=discord.Color.blue())


    @app_commands.command(name="get_config", description="View current bot configurations for this guild or a channel.")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(target_channel="[Optional] Channel to view specific settings for (e.g., allowed_languages).")
    async def view_configuration(self, interaction: discord.Interaction, target_channel: discord.TextChannel | None = None):
        await interaction.response.defer(ephemeral=True)
        guild_id = interaction.guild_id
        
        display_channel_context = target_channel or interaction.channel

        embed = discord.Embed(title=f"⚙️ Bot Configuration for {interaction.guild.name}", color=discord.Color.blue())
        embed.set_footer(text=f"Requested by {interaction.user.display_name} | Guild ID: {guild_id}")

        # Guild-wide settings
        log_ch_id = await get_guild_config(guild_id, "log_channel_id", bot_config.default_log_channel_id)
        log_ch_obj = interaction.guild.get_channel(log_ch_id) if log_ch_id else None
        embed.add_field(name="🪵 Log Channel", value=f"{log_ch_obj.mention} (`{log_ch_id}`)" if log_ch_obj else f"Not Set (Default: `{bot_config.default_log_channel_id}` or None)", inline=False)

        link_ch_id = await get_guild_config(guild_id, "link_channel_id", None)
        link_ch_obj = interaction.guild.get_channel(link_ch_id) if link_ch_id else None
        embed.add_field(name="🔗 Link-Only Channel", value=f"{link_ch_obj.mention} (`{link_ch_id}`)" if link_ch_obj else "Not Set (Domain checks apply everywhere)", inline=False)

        perm_domains = await get_guild_config(guild_id, "permitted_domains", list(bot_config.permitted_domains))
        perm_domains_str = ", ".join([f"`{d}`" for d in perm_domains]) if perm_domains else "Using Bot Defaults"
        if len(perm_domains_str) > 1000: perm_domains_str = perm_domains_str[:1000] + "..."
        embed.add_field(name="🌐 Permitted Link Domains (Guild)", value=perm_domains_str, inline=False)
        
        # Channel-specific settings for the display_channel_context
        embed.add_field(name="\u200b", value=f"--- Settings for Channel: {display_channel_context.mention} ---", inline=False) # Separator
        
        channel_settings_key = f"channel_config_{display_channel_context.id}"
        channel_db_cfg = await get_guild_config(guild_id, channel_settings_key, {})
        effective_channel_cfg = bot_config.default_channel_configs.get(display_channel_context.id, {}).copy()
        if isinstance(channel_db_cfg, dict): effective_channel_cfg.update(channel_db_cfg)

        allowed_langs = effective_channel_cfg.get("language")
        langs_str = ", ".join([f"`{lang}`" for lang in allowed_langs]) if allowed_langs else "Any Language Allowed"
        embed.add_field(name="🗣️ Allowed Languages", value=langs_str, inline=True)

        # allowed_topics = effective_channel_cfg.get("topics") # If topics are re-enabled
        # topics_str = ", ".join([f"`{t}`" for t in allowed_topics]) if allowed_topics else "Any Topic Allowed"
        # embed.add_field(name="💬 Allowed Topics", value=topics_str, inline=True)
        
        if not target_channel:
             embed.add_field(name="ℹ️ Note", value=f"Channel settings shown for current channel ({display_channel_context.mention}). Use `target_channel` option for others.", inline=False)

        await interaction.followup.send(embed=embed)


class UtilityCommands(commands.Cog, name="Utilities"):
    """General utility commands for the bot."""
    def __init__(self, bot_instance: commands.Bot):
        self.bot = bot_instance

    @app_commands.command(name="awake", description="Check if the bot is responsive.")
    async def awake_slash(self, interaction: discord.Interaction):
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
        await interaction.response.defer(ephemeral=True)
        if not LANGUAGE_MODEL:
            await interaction.followup.send("Language model is not loaded, cannot classify.", ephemeral=True)
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
            embed.add_field(name="Detection Failed", value="Could not determine the language.", inline=False)
        
        # Check channel's allowed languages for context
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
    """Initializes the SQLite database and creates tables if they don't exist."""
    global db_conn
    try:
        db_conn = await aiosqlite.connect('adroit_bot_database.db') # Changed DB name slightly
        # Use PRAGMA for better performance and WAL mode for concurrency
        await db_conn.execute("PRAGMA journal_mode=WAL;")
        await db_conn.execute("PRAGMA synchronous=NORMAL;")
        await db_conn.execute("PRAGMA foreign_keys=ON;") # Enforce foreign key constraints if used

        # Guild Configurations Table
        await db_conn.execute('''
            CREATE TABLE IF NOT EXISTS guild_configs (
                guild_id INTEGER NOT NULL,
                config_key TEXT NOT NULL,
                config_value TEXT, -- Store complex values as JSON strings
                PRIMARY KEY (guild_id, config_key)
            )
        ''')
        # Infractions Table
        await db_conn.execute('''
            CREATE TABLE IF NOT EXISTS infractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL,
                points INTEGER NOT NULL,
                timestamp TEXT NOT NULL,    -- ISO format UTC datetime
                violation_type TEXT,        -- Comma-separated list of violation types
                message_id INTEGER,
                channel_id INTEGER,
                message_content TEXT        -- Snippet of the offending message
            )
        ''')
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_infractions_user_guild_time ON infractions (user_id, guild_id, timestamp)')
        
        # Temporary Bans Table
        await db_conn.execute('''
            CREATE TABLE IF NOT EXISTS temp_bans (
                user_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL,
                unban_time TEXT NOT NULL,   -- ISO format UTC datetime for when to unban
                ban_reason TEXT,
                PRIMARY KEY (user_id, guild_id) -- A user can only have one active temp ban per guild
            )
        ''')
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_temp_bans_unban_time ON temp_bans (unban_time)')

        # Message History for Spam Detection
        await db_conn.execute('''
            CREATE TABLE IF NOT EXISTS message_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,        -- Unix timestamp for easy time calculations
                message_content_hash INTEGER    -- Hash of the message content for repetition checks
            )
        ''')
        # Index for faster queries on user, guild, and time for spam checks
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_message_history_user_guild_time ON message_history (user_id, guild_id, timestamp)')
        
        await db_conn.commit()
        logger.info("✅ Database initialized successfully with WAL mode and schema.")
    except Exception as e:
        logger.critical(f"❌ CRITICAL: Failed to initialize database: {e}", exc_info=True)
        # Consider exiting if DB init fails, as bot functionality will be severely limited.
        if db_conn: await db_conn.close() # Attempt to close if partially opened
        db_conn = None # Ensure it's None so other parts know it failed
        exit(1) # Exit if DB fails

# --- Bot Startup Sequence ---
@bot.event
async def on_ready():
    """Called when the bot is fully connected and ready."""
    logger.info(f'🚀 Logged in as {bot.user.name} (ID: {bot.user.id})')
    logger.info(f'🔗 Discord.py Version: {discord.__version__}')
    logger.info(f'🌐 Connected to {len(bot.guilds)} guilds.')
    logger.info('------')
    
    # Database initialization is now in main() before bot.start()
    # Initialize FastText model
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

    # Add Cogs
    await bot.add_cog(Moderation(bot))
    await bot.add_cog(ConfigCommands(bot))
    await bot.add_cog(UtilityCommands(bot))
    logger.info("📚 Cogs loaded.")

    # Start background tasks
    if not cleanup_and_decay_task.is_running():
        cleanup_and_decay_task.start()
    
    # Sync slash commands
    try:
        synced_cmds = await bot.tree.sync()
        logger.info(f"✅ Synced {len(synced_cmds)} application (slash) commands.")
    except Exception as e:
        logger.error(f"❌ Failed to sync application commands: {e}", exc_info=True)

    # Start HTTP health check server (non-blocking)
    asyncio.create_task(start_http_health_server())

    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="over servers | /help"))
    logger.info("🤖 Bot is ready and watching!")


@bot.event
async def on_disconnect():
    logger.warning("🔌 Bot disconnected from Discord. Will attempt to reconnect automatically.")

@bot.event
async def on_resumed():
    logger.info("✅ Bot resumed connection to Discord.")


async def main_async_runner():
    """Main async function to initialize resources and run the bot."""
    global http_session, db_conn

    # Initialize aiohttp session
    http_session = ClientSession()
    logger.info("🌍 Aiohttp ClientSession initialized.")

    # Initialize Database
    await initialize_database()
    if not db_conn: # If DB init failed, don't start the bot
        logger.critical("Database initialization failed. Bot cannot start.")
        if http_session and not http_session.closed: await http_session.close()
        return

    # Start the bot using its async context manager
    try:
        async with bot:
            await bot.start(DISCORD_TOKEN)
    except discord.LoginFailure:
        logger.critical("CRITICAL: Invalid Discord token. Please check your ADROIT_TOKEN environment variable.")
    except Exception as e:
        logger.critical(f"CRITICAL: An unhandled exception occurred during bot runtime: {e}", exc_info=True)
    finally:
        logger.info("🔌 Bot shutting down. Cleaning up resources...")
        if cleanup_and_decay_task.is_running():
            cleanup_and_decay_task.cancel()
            logger.info("Cleanup and decay task cancelled.")
        
        # Close aiohttp session
        if http_session and not http_session.closed:
            await http_session.close()
            logger.info("Aiohttp session closed.")
        
        # Close database connection
        if db_conn:
            await db_conn.close()
            logger.info("Database connection closed.")
        
        # Cancel any other outstanding asyncio tasks (optional, good practice)
        # This part needs to be careful not to cancel the main loop itself if still running from somewhere else
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
    try:
        asyncio.run(main_async_runner())
    except KeyboardInterrupt:
        logger.info("⌨️ Bot shutdown initiated by KeyboardInterrupt (Ctrl+C).")
    except Exception as e: # Catch-all for truly unhandled exceptions at the very top level
        logger.critical(f"💥 UNHANDLED EXCEPTION IN MAIN RUNNER: {e}", exc_info=True)
    finally:
        logger.info("🏁 Main process finished.")
