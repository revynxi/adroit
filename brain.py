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

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger('discord_bot')

DISCORD_TOKEN = os.getenv("ADROIT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FASTTEXT_MODEL_PATH = os.getenv("FASTTEXT_MODEL_PATH", "lid.176.ftz")
SIGHTENGINE_API_USER = os.getenv("SIGHTENGINE_API_USER")
SIGHTENGINE_API_SECRET = os.getenv("SIGHTENGINE_API_SECRET")

if not DISCORD_TOKEN:
    logger.error("ADROIT_TOKEN environment variable not set. Exiting.")
    exit(1)

intents = discord.Intents.default()
intents.members = True
intents.message_content = True
intents.presences = False

bot = commands.Bot(command_prefix=">>", intents=intents, help_command=None)

db_conn: aiosqlite.Connection = None
LANGUAGE_MODEL = None
http_session: ClientSession = None

class BotConfig:
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
            r"(discord\.gg/|join\s+our|server\s+invite|free\s+nitro|check\s+out\s+my|follow\s+me|subscribe\s+to|buy\s+now)",
            re.IGNORECASE
        )
        self.url_pattern = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|net|org|io|dev)\b)")
        self.has_alphanumeric_pattern = re.compile(r'[a-zA-Z0-9]')
        self.permitted_domains = [
            "googleusercontent.com", "tenor.com", "giphy.com", "tiktok.com",
            "youtube.com", "youtu.be", "docs.google.com", "cdn.discordapp.com",
            "roblox.com", "github.com", "theuselessweb.com",
            "wikipedia.org", "twitch.tv", "reddit.com", "x.com", "twitter.com"
        ]
        self.punishment_system = {
            "points_thresholds": {
                5: {"action": "warn", "message": "Warnings make your sins weigh heavier, think twice before sending something inappropriate"},
                10: {"action": "mute", "duration_hours": 1, "reason": "Spam/Minor violations"},
                15: {"action": "kick", "reason": "Repeated violations"},
                25: {"action": "temp_ban", "duration_days": 1, "reason": "Serious/Persistent violations"},
                50: {"action": "temp_ban", "duration_months": 1, "reason": "Severe/Accumulated violations"},
                10000: {"action": "ban", "reason": "A literal war criminal"}
            },
            "violations": {
                "discrimination": {"points": 2, "severity": "Medium"},
                "spam": {"points": 1, "severity": "Low"},
                "nsfw": {"points": 2, "severity": "Medium"},
                "nsfw_media": {"points": 5, "severity": "High"},
                "advertising": {"points": 2, "severity": "Medium"},
                "politics_discussion": {"points": 1, "severity": "Low"},
                "off_topic": {"points": 1, "severity": "Low"},
                "foreign_language": {"points": 1, "severity": "Low"},
                "openai_moderation": {"points": 2, "severity": "Medium"},
                "excessive_mentions": {"points": 1, "severity": "Low"},
                "excessive_attachments": {"points": 1, "severity": "Low"},
                "long_message": {"points": 1, "severity": "Low"}
            }
        }
        self.spam_window = 10
        self.spam_limit = 5
        self.mention_limit = 5
        self.max_message_length = 800
        self.max_attachments = 4
        self.min_msg_len_for_lang_check = 4
        self.min_confidence_for_flagging = 0.65
        self.min_confidence_short_msg = 0.75
        self.short_msg_threshold = 20
        self.common_safe_foreign_words = {"bonjour", "hola", "merci", "gracias", "oui", "si", "nyet", "da", "salut", "ciao", "hallo", "guten tag"}
        self.fuzzy_match_threshold = 80

bot_config = BotConfig()

def load_terms_from_file(filepath: str) -> tuple[set, list]:
    """Loads terms from a text file, one term per line, separating single words and phrases."""
    words = set()
    phrases = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                term = line.strip().lower()
                if not term:
                    continue
                if ' ' in term:
                    phrases.append(term)
                else:
                    words.add(term)
        logger.info(f"Loaded {len(words)} words and {len(phrases)} phrases from {filepath}")
    except FileNotFoundError:
        logger.warning(f"Warning: {filepath} not found. Using empty list/set for terms.")
    except Exception as e:
        logger.error(f"Error loading terms from {filepath}: {e}")
    return words, phrases

discrimination_words_set, discrimination_phrases = load_terms_from_file('discrimination_terms.txt')
nsfw_words_set, nsfw_phrases = load_terms_from_file('nsfw_terms.txt')


def clean_message_content(text: str) -> str:
    """Cleans and normalizes message content for analysis."""
    return text.strip().lower()

async def get_guild_config(guild_id: int, key: str, default):
    """Retrieves a guild-specific configuration value from the database."""
    try:
        async with db_conn.execute("SELECT config_value FROM guild_configs WHERE guild_id = ? AND config_key = ?", (guild_id, key)) as cursor:
            result = await cursor.fetchone()
            if result:
                try:
                    return json.loads(result[0])
                except json.JSONDecodeError:
                    return result[0]
            return default
    except Exception as e:
        logger.error(f"Error getting guild config for guild {guild_id}, key {key}: {e}", exc_info=True)
        return default

async def set_guild_config(guild_id: int, key: str, value):
    """Sets or updates a guild-specific configuration value in the database."""
    stored_value = value
    if isinstance(value, (list, dict)):
        stored_value = json.dumps(value)

    async with db_conn.cursor() as cursor:
        await cursor.execute(
            'INSERT OR REPLACE INTO guild_configs (guild_id, config_key, config_value) VALUES (?, ?, ?)',
            (guild_id, key, stored_value)
        )
        await db_conn.commit()
    logger.info(f"Updated guild config: Guild {guild_id}, Key '{key}', Value '{value}'")

async def detect_language_ai(text: str) -> tuple[str, float]:
    """Detect the language of the given text using FastText. Returns (lang_code, confidence_score)."""
    clean_text = clean_message_content(text)
    if not clean_text:
        return "und", 0.0

    if not LANGUAGE_MODEL:
        logger.error("FastText model not loaded. Defaulting to ('en', 0.0).")
        return "en", 0.0
    try:
        prediction = LANGUAGE_MODEL.predict(clean_text, k=1)
        if prediction and prediction[0] and prediction[1]:
            lang_code = prediction[0][0].replace("__label__", "")
            confidence = float(prediction[1][0])
            return lang_code, confidence
        else:
            logger.warning(f"FastText returned unexpected prediction format for: '{clean_text[:100]}...'")
            return "und", 0.0
    except Exception as e:
        logger.error(f"FastText language detection error for '{clean_text[:100]}...': {e}")
        return "en", 0.0

async def log_action(action: str, member_or_user: discord.User | discord.Member, reason: str, guild: discord.Guild = None):
    """Log moderation actions to a specified channel and console."""
    current_guild = guild if guild else (member_or_user.guild if isinstance(member_or_user, discord.Member) else None)
    if not current_guild:
        logger.error(f"Cannot log action '{action}' for user {member_or_user.id}: Guild context missing.")
        return

    log_channel_id = await get_guild_config(current_guild.id, "log_channel_id", bot_config.default_log_channel_id)
    log_channel = bot.get_channel(log_channel_id) or current_guild.get_channel(log_channel_id)

    user_mention = member_or_user.mention if isinstance(member_or_user, discord.Member) else f"{member_or_user.name}#{member_or_user.discriminator}"
    user_id = member_or_user.id
    display_name = member_or_user.display_name if isinstance(member_or_user, discord.Member) else member_or_user.name

    embed = discord.Embed(
        title=f"Moderation Action: {action.upper()}",
        description=f"**User:** {user_mention} (`{user_id}`)\n**Reason:** {reason}\n**Timestamp:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        color=discord.Color.red() if action in ["ban", "kick", "mute", "temp_ban"] else discord.Color.orange()
    )

    if log_channel:
        try:
            await log_channel.send(embed=embed)
        except discord.Forbidden:
            logger.error(f"Missing permissions to send logs to channel {log_channel.name} ({log_channel.id}) in guild {current_guild.name}.")
        except Exception as e:
            logger.error(f"Error sending log embed to channel for guild {current_guild.name}: {e}")
    else:
        logger.info(f"LOG (Guild {current_guild.name}): {action.upper()} applied to {display_name} ({user_id}) for: {reason}")
        logger.warning(f"Log channel (ID: {log_channel_id}) not found or accessible for guild {current_guild.name}.")

def retry_if_openai_api_error(exception):
    """Retries on server errors (5xx), rate limits (429), or network issues, but not client errors like 400."""
    if isinstance(exception, client_exceptions.ClientResponseError):
        return exception.status == 429 or exception.status >= 500
    return isinstance(exception, (asyncio.TimeoutError, client_exceptions.ClientOSError, client_exceptions.ClientConnectorError))

@retry(
    stop=stop_after_attempt(5),  
    wait=wait_random_exponential(multiplier=1, min=4, max=120), 
    retry=retry_if_openai_api_error 
)
async def check_openai_moderation(text: str) -> dict:
    """Checks text against the OpenAI moderation API with selective retries."""
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set. Skipping OpenAI moderation check.")
        return {"flagged": False, "categories": {}}
    if not text.strip():
        return {"flagged": False, "categories": {}}

    url = "https://api.openai.com/v1/moderations"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {"input": text}

    try:
        async with http_session.post(url, headers=headers, json=data, timeout=15) as response: 
            response.raise_for_status()
            json_response = await response.json()
            results_list = json_response.get("results", [])
            if results_list:
                return results_list[0]
            logger.warning(f"OpenAI moderation returned empty results list for text: {text[:100]}")
            return {"flagged": False, "categories": {}}
    except client_exceptions.ClientResponseError as e:
        if e.status == 400:
            logger.warning(f"OpenAI moderation API returned 400 Bad Request (will not retry). Text: '{text[:100]}...' Error: {e.message}")
            return {"flagged": False, "categories": {}} 
        logger.error(f"OpenAI moderation API error: {e.status} - {e.message}. Text: '{text[:100]}...'. Retrying if applicable.")
        raise 
    except asyncio.TimeoutError:
        logger.error(f"OpenAI moderation API request timed out after 15 seconds. Text: {text[:100]}")
        raise 
    except Exception as e: 
        logger.error(f"Unexpected error with OpenAI moderation API: {e} for text: {text[:100]}", exc_info=True)
        logger.critical(f"OpenAI moderation failed after all attempts or due to an unretryable/unexpected error for text: '{text[:100]}...'. Moderation for this text will be skipped.")
        return {"flagged": False, "categories": {}}


async def apply_punishment(member: discord.Member, action: str, reason: str, duration: timedelta = None):
    """Applies a moderation action to a member."""
    try:
        dm_message = ""
        if action == "warn":
            warn_config = bot_config.punishment_system["points_thresholds"].get(5, {})
            warning_message = warn_config.get("message", "You have received a warning. Make sure to not break our server rules!")
            dm_message = f"You have been warned. Reason: {reason}\n{warning_message}"
            await log_action("warn", member, reason)
        elif action == "mute":
            if duration:
                await member.timeout(duration, reason=reason)
                dm_message = f"You have been muted for {duration}. Reason: {reason}"
                await log_action("mute", member, reason)
            else:
                logger.warning(f"Attempted to mute {member.display_name} without duration. Skipping.")
        elif action == "kick":
            dm_message = f"You are being kicked from the server. Reason: {reason}"
            await member.kick(reason=reason)
            await log_action("kick", member, reason)
        elif action == "temp_ban":
            if duration:
                unban_time = datetime.utcnow() + duration
                async with db_conn.cursor() as cursor:
                    await cursor.execute(
                        'INSERT OR REPLACE INTO temp_bans (user_id, guild_id, unban_time, ban_reason) VALUES (?, ?, ?, ?)',
                        (member.id, member.guild.id, unban_time.isoformat(), reason)
                    )
                    await db_conn.commit()
                dm_message = f"You have been temporarily banned until {unban_time.strftime('%Y-%m-%d %H:%M:%S UTC')}. Reason: {reason}"
                await member.ban(reason=reason, delete_message_days=0)
                await log_action("temp_ban", member, reason)
            else:
                logger.warning(f"Attempted to temp_ban {member.display_name} without duration. Skipping.")
        elif action == "ban":
            dm_message = f"You have been permanently banned from the server. Reason: {reason}"
            await member.ban(reason=reason, delete_message_days=0)
            await log_action("ban", member, reason)

        if dm_message:
            try:
                await member.send(dm_message)
            except discord.Forbidden:
                logger.warning(f"Could not DM {action} notification to {member.display_name} ({member.id}).")

    except discord.Forbidden:
        logger.error(f"Missing permissions to {action} {member.display_name} in {member.guild.name}. "
                     f"Please check bot role hierarchy and permissions.")
    except discord.HTTPException as e:
        logger.error(f"Discord API error while applying {action} to {member.display_name}: {e.status} - {e.text}")
    except Exception as e:
        logger.error(f"Unexpected error applying {action} to {member.display_name}: {e}", exc_info=True)


async def add_infraction_points(member: discord.Member, points: int, violation_type: str, message: discord.Message):
    """Logs an infraction and applies punishment if points threshold is reached."""
    guild_id = member.guild.id
    user_id = member.id

    async with db_conn.cursor() as cursor:
        await cursor.execute(
            'INSERT INTO infractions (user_id, guild_id, points, timestamp, violation_type, message_id, channel_id) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (user_id, guild_id, points, datetime.utcnow().isoformat(), violation_type, message.id, message.channel.id)
        )
        await db_conn.commit()

        cutoff_date = (datetime.utcnow() - timedelta(days=30)).isoformat()
        await cursor.execute(
            'SELECT SUM(points) FROM infractions WHERE user_id = ? AND guild_id = ? AND timestamp >= ?',
            (user_id, guild_id, cutoff_date)
        )
        total_points_data = await cursor.fetchone()
        total_points = total_points_data[0] if total_points_data and total_points_data[0] is not None else 0

    logger.info(f"User {member.display_name} ({user_id}) in guild {guild_id} received {points} points for '{violation_type}'. "
                f"Total active points (last 30 days): {total_points}.")

    for threshold_points in sorted(bot_config.punishment_system["points_thresholds"].keys(), reverse=True):
        if total_points >= threshold_points:
            punishment_config = bot_config.punishment_system["points_thresholds"][threshold_points]
            action = punishment_config["action"]
            
            auto_punishment_reason = punishment_config.get(
                "reason",
                f"Automated action: Accumulated {total_points} infraction points."
            )
            if f"Accumulated {total_points} infraction points" not in auto_punishment_reason: 
                auto_punishment_reason = f"{auto_punishment_reason} (Triggered by {total_points} points)."


            duration = None
            if "duration_hours" in punishment_config:
                duration = timedelta(hours=punishment_config["duration_hours"])
            elif "duration_days" in punishment_config:
                duration = timedelta(days=punishment_config["duration_days"])
            elif "duration_months" in punishment_config: 
                duration = timedelta(days=punishment_config["duration_months"] * 30)

            logger.info(f"Applying punishment '{action}' to {member.display_name} due to reaching {total_points} points (threshold: {threshold_points}).")
            await apply_punishment(member, action, auto_punishment_reason, duration)
            break 


def get_domain_from_url(url: str) -> str | None:
    """Extracts the domain from a given URL."""
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url 
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain.lower() if domain else None
    except Exception as e: 
        logger.debug(f"Could not parse domain from URL '{url}': {e}")
        return None

def is_permitted_domain(domain: str, guild_permitted_domains: list[str]) -> bool:
    """Check if the extracted domain is in the guild's permitted list or is a subdomain of a permitted one."""
    if not domain:
        return False
    for perm_domain_raw in guild_permitted_domains:
        perm_domain_cleaned = get_domain_from_url(perm_domain_raw) 
        if perm_domain_cleaned:
            if domain == perm_domain_cleaned or domain.endswith('.' + perm_domain_cleaned):
                return True
    return False

async def handle_health_check(request):
    """Handles HTTP health check requests."""
    return web.Response(text="Bot is awake and healthy!", status=200, headers={"Content-Type": "text/plain"})

async def start_http_server():
    """Starts a simple HTTP server for health checks."""
    try:
        app = web.Application()
        app.router.add_get('/', handle_health_check)
        runner = web.AppRunner(app)
        await runner.setup()
        port = int(os.getenv("PORT", "8080"))
        site = web.TCPSite(runner, host='0.0.0.0', port=port)
        await site.start()
        logger.info(f"✅ HTTP server running on http://0.0.0.0:{port}")
    except Exception as e:
        logger.error(f"❌ Failed to start HTTP server: {e}", exc_info=True)

@tasks.loop(hours=24)
async def decay_points():
    """Periodically cleans up old infraction records and processes expired temporary bans."""
    if not db_conn:
        logger.warning("Database connection not available for decay_points task. Skipping.")
        return

    async with db_conn.cursor() as cursor:
        cutoff_date_infractions_deletion = (datetime.utcnow() - timedelta(days=90)).isoformat()
        await cursor.execute('DELETE FROM infractions WHERE timestamp < ?', (cutoff_date_infractions_deletion,))
        deleted_rows = cursor.rowcount
        if deleted_rows > 0:
            logger.info(f"Deleted {deleted_rows} infraction records older than 90 days.")
        else: 
            pass 

        now_iso = datetime.utcnow().isoformat()
        await cursor.execute('SELECT user_id, guild_id, ban_reason FROM temp_bans WHERE unban_time <= ?', (now_iso,))
        expired_bans = await cursor.fetchall()

        for user_id, guild_id, ban_reason in expired_bans:
            guild = bot.get_guild(guild_id)
            if guild:
                try:
                    user_to_unban = discord.Object(id=user_id) 
                    await guild.unban(user_to_unban, reason="Temporary ban expired automatically.")
                    logger.info(f"Unbanned user ID {user_id} from guild {guild.name} ({guild_id}) - temp ban expired.")

                    try:
                        user_obj = await bot.fetch_user(user_id)
                        await log_action("unban", user_obj, f"Temporary ban expired (original reason: {ban_reason or 'N/A'})", guild=guild)
                    except discord.NotFound: 
                        await log_action("unban_id", discord.Object(id=user_id) , f"User ID {user_id} unbanned. Temp ban expired (original reason: {ban_reason or 'N/A'})", guild=guild)

                except discord.NotFound: 
                    logger.warning(f"User {user_id} not found in ban list of guild {guild.name} for automatic unban, or already unbanned.")
                except discord.Forbidden:
                    logger.error(f"Missing permissions to unban user {user_id} in guild {guild.name}.")
                except Exception as e:
                    logger.error(f"Error during automatic unban process for user {user_id} in guild {guild.name}: {e}", exc_info=True)
            else:
                 logger.warning(f"Cannot process temp ban expiry for user {user_id} in guild {guild_id}: Bot is not in this guild.")

            await cursor.execute('DELETE FROM temp_bans WHERE user_id = ? AND guild_id = ? AND unban_time <= ?', (user_id, guild_id, now_iso))
        if expired_bans:
             await db_conn.commit()
    logger.info("Decay points and temporary bans cleanup task completed.")


@bot.event
async def on_error(event_name, *args, **kwargs):
    """Catches unhandled exceptions in Discord events."""
    logger.error(f"Unhandled error in event '{event_name}': Args: {args}, Kwargs: {kwargs}", exc_info=True)

@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    """Centralized error handling for bot commands."""
    if isinstance(error, commands.CommandNotFound):
        return
    elif isinstance(error, commands.MissingRequiredArgument):
        embed = discord.Embed(
            title="Missing Argument",
            description=f"Oops! You missed an argument: `{error.param.name}`. Check `>>help {ctx.command.qualified_name}` for usage.",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)
    elif isinstance(error, commands.BadArgument):
        embed = discord.Embed(
            title="Invalid Argument",
            description=f"Hmm, that's not a valid argument. {error}", 
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)
    elif isinstance(error, commands.NoPrivateMessage):
        embed = discord.Embed(
            title="Server Only Command",
            description="This command can only be used in a server.",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)
    elif isinstance(error, commands.MissingPermissions):
        embed = discord.Embed(
            title="Permission Denied",
            description=f"You don't have the required permissions to use this command: `{', '.join(error.missing_permissions)}`",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)
    elif isinstance(error, commands.BotMissingPermissions):
        embed = discord.Embed(
            title="Bot Missing Permissions",
            description=f"I'm missing permissions to do that: `{', '.join(error.missing_permissions)}`.",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)
        logger.warning(f"Bot missing permissions in guild {ctx.guild.id if ctx.guild else 'DM'}, channel {ctx.channel.id}: {error.missing_permissions} for command {ctx.command.name if ctx.command else 'N/A'}")
    elif isinstance(error, commands.CommandInvokeError):
        logger.error(f"Error invoking command '{ctx.command.qualified_name}': {error.original}", exc_info=error.original)
        embed = discord.Embed(
            title="Internal Error",
            description="An internal error occurred while running this command. The developers have been notified.",
            color=discord.Color.dark_red()
        )
        await ctx.send(embed=embed)
    else:
        logger.error(f"Unhandled command error for '{ctx.command.qualified_name if ctx.command else 'UnknownCmd'}': {error}", exc_info=True)
        embed = discord.Embed(
            title="Unexpected Error",
            description="An unexpected error occurred. Please try again later.",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)

@bot.event
async def on_guild_join(guild: discord.Guild):
    """Logs when the bot joins a new guild."""
    logger.info(f"Joined new guild: {guild.name} (ID: {guild.id}, Members: {guild.member_count})")

class Moderation(commands.Cog):
    def __init__(self, bot_instance: commands.Bot):
        self.bot = bot_instance
        self.sightengine_nsfw_threshold = 0.6
        self.sightengine_gore_threshold = 0.8
        self.sightengine_violence_threshold = 0.7
        self.sightengine_text_profanity_threshold = 0.9 
        self.sightengine_minor_offensive_threshold = 0.95

        self.discrimination_words_set = discrimination_words_set
        self.discrimination_phrases = discrimination_phrases
        self.nsfw_words_set = nsfw_words_set
        self.nsfw_phrases = nsfw_phrases

        self.rate_limit_data = defaultdict(lambda: deque())
        
        self.cleanup_message_history_db.start()
        self.cleanup_rate_limit_data.start()


    def cog_unload(self):
        self.cleanup_message_history_db.cancel()
        self.cleanup_rate_limit_data.cancel()
        logger.info("Moderation cog tasks (message history cleanup, rate limit data cleanup) cancelled.")

    @tasks.loop(minutes=5)
    async def cleanup_rate_limit_data(self):
        """Cleans up old rate limit timestamps from the in-memory deque."""
        now = datetime.utcnow()
        for user_id in list(self.rate_limit_data.keys()):
            user_deque = self.rate_limit_data[user_id]
            while user_deque and (now - user_deque[0]).total_seconds() >= bot_config.spam_window:
                user_deque.popleft()
            if not user_deque: 
                del self.rate_limit_data[user_id]
        logger.debug("Cleaned up rate limit data.")

    @cleanup_rate_limit_data.before_loop
    async def before_cleanup_rate_limit_data(self):
        await self.bot.wait_until_ready()
        logger.info("Starting rate limit data cleanup task.")


    @tasks.loop(hours=1)
    async def cleanup_message_history_db(self):
        """Periodically cleans up old message entries from the database to manage size and relevance for spam checks."""
        if db_conn:
            try:
                retention_period_seconds = max(bot_config.spam_window * 60, 600) 
                threshold_timestamp = (datetime.utcnow() - timedelta(seconds=retention_period_seconds)).timestamp()

                async with db_conn.cursor() as cursor: 
                    await cursor.execute("DELETE FROM message_history WHERE timestamp < ?", (threshold_timestamp,))
                    await db_conn.commit()
                logger.info(f"Cleaned up message_history table: deleted entries older than {retention_period_seconds} seconds ({cursor.rowcount} rows).")
            except Exception as e:
                logger.error(f"Error during message history cleanup task: {e}", exc_info=True)

    @cleanup_message_history_db.before_loop
    async def before_cleanup_message_history_db(self):
        """Wait until the bot is ready before starting the cleanup loop."""
        await self.bot.wait_until_ready()
        logger.info("Starting message history DB cleanup task.")


    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild or not db_conn:
            return

        violations = set()
        guild_id = message.guild.id
        user_id = message.author.id
        content_raw = message.content
        content_lower = clean_message_content(content_raw)
        now = datetime.utcnow()

        user_message_timestamps = self.rate_limit_data[user_id]
        while user_message_timestamps and (now - user_message_timestamps[0]).total_seconds() >= bot_config.spam_window:
            user_message_timestamps.popleft()
        user_message_timestamps.append(now)

        if len(user_message_timestamps) > bot_config.spam_limit:
            violations.add("spam")
            logger.debug(f"Spam (rate limit) by {message.author.name}: {len(user_message_timestamps)} msgs in {bot_config.spam_window}s")

        try:
            await db_conn.execute(
                "INSERT INTO message_history (user_id, guild_id, timestamp, message_content) VALUES (?, ?, ?, ?)",
                (user_id, guild_id, now.timestamp(), content_raw) 
            )
            await db_conn.commit()

            time_threshold_spam_window_ts = (now - timedelta(seconds=bot_config.spam_window)).timestamp()
            async with db_conn.execute(
                "SELECT COUNT(*) FROM message_history WHERE user_id = ? AND guild_id = ? AND message_content = ? AND timestamp >= ?",
                (user_id, guild_id, content_raw, time_threshold_spam_window_ts)
            ) as cursor:
                count_repetition = (await cursor.fetchone())[0]
                if count_repetition > 2 : 
                    violations.add("spam")
                    logger.debug(f"Spam (repetition, DB) by {message.author.name}: '{content_raw[:50]}...' found {count_repetition} times.")
        except Exception as e:
            logger.error(f"Error with message history DB for spam check: {e}", exc_info=True)

        channel_specific_db_config = await get_guild_config(guild_id, f"channel_config_{message.channel.id}", {})
        channel_cfg = bot_config.default_channel_configs.get(message.channel.id, {}).copy() 
        if isinstance(channel_specific_db_config, dict):
            channel_cfg.update(channel_specific_db_config) 
        else:
            logger.warning(f"channel_config_{message.channel.id} for guild {guild_id} was not a dict: {channel_specific_db_config}")

        guild_permitted_domains = await get_guild_config(guild_id, "permitted_domains", list(bot_config.permitted_domains))

        if len(message.mentions) > bot_config.mention_limit:
            violations.add("excessive_mentions")
        if len(message.attachments) > bot_config.max_attachments:
            violations.add("excessive_attachments")
        if len(content_raw) > bot_config.max_message_length:
            violations.add("long_message")

        allowed_languages = channel_cfg.get("language") 
        if allowed_languages and content_lower: 
            skip_lang_check_reason = None
            if bot_config.url_pattern.fullmatch(content_lower.strip()): 
                skip_lang_check_reason = "message is only a URL"
            elif not bot_config.has_alphanumeric_pattern.search(content_lower): 
                skip_lang_check_reason = "message has no alphanumeric characters"
            elif len(content_lower) < bot_config.min_msg_len_for_lang_check: 
                skip_lang_check_reason = f"message is too short ({len(content_lower)} vs min {bot_config.min_msg_len_for_lang_check} chars)"

            if skip_lang_check_reason:
                logger.debug(f"Skipping language check for '{content_raw[:50]}...': {skip_lang_check_reason}.")
            else:
                detected_lang_code, confidence = await detect_language_ai(content_raw)
                logger.debug(f"Language detection for '{content_raw[:50]}...': Lang={detected_lang_code}, Conf={confidence:.2f}, Allowed: {allowed_languages}")

                if detected_lang_code not in allowed_languages and detected_lang_code != "und":
                    is_short_msg = len(content_lower) < bot_config.short_msg_threshold
                    confidence_threshold = bot_config.min_confidence_short_msg if is_short_msg else bot_config.min_confidence_for_flagging

                    if content_lower in bot_config.common_safe_foreign_words:
                        logger.debug(f"Message '{content_lower}' is a common safe foreign word, detected as {detected_lang_code}. Not flagging.")
                    elif confidence < confidence_threshold:
                        logger.debug(f"Low confidence ({confidence:.2f} < {confidence_threshold}) for {('short ' if is_short_msg else '')}message '{content_lower}' (lang: {detected_lang_code}). Not flagging as foreign.")
                    else:
                        violations.add("foreign_language")
                        logger.debug(f"Foreign language violation by {message.author.name} in {message.channel.name}: '{detected_lang_code}' (Conf: {confidence:.2f}) not in {allowed_languages}. Message: '{content_raw[:50]}...'")

        link_channel_id_config = await get_guild_config(guild_id, "link_channel_id", None)
        link_channel_id = int(link_channel_id_config) if link_channel_id_config and str(link_channel_id_config).isdigit() else None

        if bot_config.forbidden_text_pattern.search(content_lower):
            violations.add("advertising")
            logger.debug(f"Advertising (forbidden pattern) by {message.author.name}: '{content_raw[:50]}...'")

        urls_found = bot_config.url_pattern.findall(content_raw)
        if urls_found:
            is_link_channel = (link_channel_id == message.channel.id)
            if not is_link_channel: 
                for url_match in urls_found:
                    url_str = url_match[0] if isinstance(url_match, tuple) and url_match[0] else url_match
                    domain = get_domain_from_url(url_str)
                    if domain and not is_permitted_domain(domain, guild_permitted_domains):
                        violations.add("advertising")
                        logger.debug(f"Advertising (forbidden domain: {domain}) by {message.author.name} in non-link channel. URL: {url_str}")
                        break 
                    elif not domain and url_str: 
                        logger.debug(f"Could not extract domain from detected URL-like pattern '{url_str}' for advertising check.")

        if not "discrimination" in violations: 
            for word in self.discrimination_words_set:
                if fuzz.ratio(word, content_lower) >= bot_config.fuzzy_match_threshold: 
                    violations.add("discrimination")
                    logger.debug(f"Discrimination (fuzzy word: '{word}') by {message.author.name}: '{content_raw[:50]}...'")
                    break
            if not "discrimination" in violations:
                for phrase in self.discrimination_phrases:
                    if fuzz.partial_ratio(phrase, content_lower) >= bot_config.fuzzy_match_threshold:
                        violations.add("discrimination")
                        logger.debug(f"Discrimination (fuzzy phrase: '{phrase}') by {message.author.name}: '{content_raw[:50]}...'")
                        break
        if not "nsfw" in violations:
            for word in self.nsfw_words_set:
                if fuzz.ratio(word, content_lower) >= bot_config.fuzzy_match_threshold:
                    violations.add("nsfw")
                    logger.debug(f"NSFW text (fuzzy word: '{word}') by {message.author.name}: '{content_raw[:50]}...'")
                    break
            if not "nsfw" in violations:
                for phrase in self.nsfw_phrases:
                    if fuzz.partial_ratio(phrase, content_lower) >= bot_config.fuzzy_match_threshold:
                        violations.add("nsfw")
                        logger.debug(f"NSFW text (fuzzy phrase: '{phrase}') by {message.author.name}: '{content_raw[:50]}...'")
                        break

        if message.attachments and not ("nsfw_media" in violations or "nsfw" in violations) : 
            for attachment in message.attachments:
                content_type = attachment.content_type
                if content_type and (content_type.startswith('image/') or content_type.startswith('video/')):
                    if SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET:
                        logger.info(f"Checking attachment '{attachment.filename}' ({content_type}, URL: {attachment.url}) with Sightengine...")
                        try:
                            is_media_nsfw = await self.check_media_nsfw_sightengine(attachment.url)
                            if is_media_nsfw:
                                violations.add("nsfw_media")
                                logger.info(f"NSFW media (Sightengine) violation: {attachment.url} by {message.author.name}")
                                break 
                        except Exception as e: 
                            logger.error(f"Error calling Sightengine NSFW check for {attachment.url}: {e}", exc_info=True)
                    else:
                        logger.debug("Sightengine API credentials not set. Skipping media NSFW check.")
                    break 

        if content_raw.strip():
            openai_mod_result = await check_openai_moderation(content_raw) 
            if openai_mod_result.get("flagged"):
                logger.info(f"OpenAI Moderation flagged message by {message.author.name}: {content_raw[:100]}...")
                categories_hit = {k: v for k, v in openai_mod_result.get("categories", {}).items() if v}
                if categories_hit:
                    violations.add("openai_moderation")
                    logger.debug(f"OpenAI flagged categories: {categories_hit}")


        if violations:
            total_points_for_message = 0
            violation_details = []
            for violation_type in violations:
                points_for_type = bot_config.punishment_system["violations"].get(violation_type, {}).get("points", 0)
                severity = bot_config.punishment_system["violations"].get(violation_type, {}).get("severity", "Unknown")
                total_points_for_message += points_for_type
                violation_details.append(f"{violation_type.replace('_', ' ')} ({points_for_type} pts, {severity})")
                logger.info(f"Violation recorded: {violation_type} for user {message.author.display_name} in {message.guild.name}. Points: {points_for_type}. Severity: {severity}.")

            if total_points_for_message > 0:
                await add_infraction_points(message.author, total_points_for_message, ", ".join(sorted(list(violations))), message)
                try:
                    reasons_str = "; ".join(violation_details)
                    embed = discord.Embed(
                        title="⚠️ Content Moderated",
                        description=f"Your recent message in {message.channel.mention} was flagged for: **{reasons_str}**.\nThis has been recorded.",
                        color=discord.Color.orange()
                    )
                    embed.set_footer(text="Please review server rules. Repeated violations may lead to further actions.")
                    await message.author.send(embed=embed)
                except discord.Forbidden:
                    logger.warning(f"Could not DM violation notification to {message.author.display_name} ({message.author.id}).")
                except Exception as e:
                    logger.error(f"Error sending violation embed to user {message.author.id}: {e}", exc_info=True)

                try:
                    await message.delete()
                    logger.info(f"Deleted message {message.id} from {message.author.display_name} in {message.channel.name} due to violations: {', '.join(violations)}.")
                except discord.Forbidden:
                    logger.error(f"Missing permissions to delete message {message.id} in {message.channel.name} for guild {message.guild.name}.")
                except discord.NotFound:
                    logger.warning(f"Attempted to delete message {message.id} but it was not found (already deleted?).")
                except Exception as e:
                    logger.error(f"Error deleting message {message.id}: {e}", exc_info=True)
        else:
            await self.bot.process_commands(message)


    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(client_exceptions.ClientResponseError))
    async def check_media_nsfw_sightengine(self, media_url: str) -> bool:
        """Checks media content for NSFW, gore, and violence using Sightengine API."""
        if not SIGHTENGINE_API_USER or not SIGHTENGINE_API_SECRET:
            logger.warning("Sightengine API credentials not set. Cannot check media.")
            return False

        params = {
            'url': media_url,
            'models': 'wad,gore',
            'api_user': SIGHTENGINE_API_USER,
            'api_secret': SIGHTENGINE_API_SECRET
        }
        url = "https://api.sightengine.com/1.0/check.json"

        try:
            async with http_session.get(url, params=params, timeout=20) as response: 
                response.raise_for_status() 
                json_response = await response.json()
                logger.debug(f"Sightengine API response for {media_url}: {json_response}")

                wad = json_response.get('wad', {}) 
                if wad.get('adult') and wad['adult'] > self.sightengine_nsfw_threshold:
                    logger.info(f"Sightengine detected NSFW (adult) content above threshold ({wad['adult']:.2f}) for {media_url}")
                    return True

                if wad.get('offensive') and wad['offensive'] > self.sightengine_minor_offensive_threshold:
                    logger.info(f"Sightengine detected offensive content above threshold ({wad['offensive']:.2f}) for {media_url}")
                    return True


                gore_data = json_response.get('gore', {}) 
                if gore_data.get('prob') and gore_data['prob'] > self.sightengine_gore_threshold:
                    logger.info(f"Sightengine detected gore content above threshold ({gore_data['prob']:.2f}) for {media_url}")
                    return True

                return False 

        except client_exceptions.ClientResponseError as e:
            logger.error(f"Sightengine API error: {e.status} - {e.message} for {media_url}. URL: {e.request_info.url if e.request_info else 'N/A'}")
            if e.status == 400 and "cannot_load_image" in e.message.lower(): # Example of specific error handling
                 logger.warning(f"Sightengine could not load image at {media_url}. Assuming safe for this check.")
                 return False
            raise 
        except asyncio.TimeoutError:
            logger.error(f"Sightengine API request timed out for {media_url}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error with Sightengine API for {media_url}: {e}", exc_info=True)
            return False


@bot.tree.command(name="configure", description="Configure bot settings for this guild or channel.")
@app_commands.default_permissions(manage_guild=True)
@app_commands.describe(
    setting="Setting: log_channel, link_channel, allowed_languages, allowed_topics, permitted_domains",
    value="Value for the setting (e.g., channel ID, 'en,fr', 'topic1,topic2', 'domain.com,site.org')",
    channel="[Optional] Channel for channel-specific settings (languages, topics)",
    clear="[Optional] Set to True to clear/reset the setting to default"
)
async def configure(
    interaction: discord.Interaction,
    setting: str,
    value: str = None,
    channel: discord.TextChannel = None,
    clear: bool = False
):
    """Handles configuration commands using slash commands."""
    await interaction.response.defer(ephemeral=True) 

    guild_id = interaction.guild.id
    is_channel_setting_type = setting in ["allowed_languages", "allowed_topics"]
    
    target_channel_for_config = channel if channel else interaction.channel 

    if is_channel_setting_type and not channel: 
        pass 
    elif channel: 
        pass 
    
    if is_channel_setting_type: 
        config_key_base = setting 
        db_key_for_channel_json = f"channel_config_{target_channel_for_config.id}"
    else: 
        config_key_base = None  
        db_key_for_guild_setting = setting 


    if clear:
        if is_channel_setting_type: 
            channel_cfg_json = await get_guild_config(guild_id, db_key_for_channel_json, {})
            if config_key_base in channel_cfg_json:
                del channel_cfg_json[config_key_base]
                await set_guild_config(guild_id, db_key_for_channel_json, channel_cfg_json)
                msg = f"Setting `{config_key_base}` for channel {target_channel_for_config.mention} has been cleared (reverts to default or none)."
            else:
                msg = f"Setting `{config_key_base}` for channel {target_channel_for_config.mention} was not set or already clear."
        else:
            default_value = None
            if db_key_for_guild_setting == "log_channel_id": default_value = bot_config.default_log_channel_id
            elif db_key_for_guild_setting == "permitted_domains": default_value = list(bot_config.permitted_domains)

            async with db_conn.cursor() as cursor: 
                await cursor.execute('DELETE FROM guild_configs WHERE guild_id = ? AND config_key = ?', (guild_id, db_key_for_guild_setting))
                await db_conn.commit()
            msg = f"Guild setting `{db_key_for_guild_setting}` has been cleared. It will now use bot defaults (if any)."
        
        embed = discord.Embed(title="Configuration Cleared", description=msg, color=discord.Color.green())
        await interaction.followup.send(embed=embed)
        return

    if not value:
        embed = discord.Embed(title="Configuration Error", description="You must provide a `value` to set.", color=discord.Color.red())
        await interaction.followup.send(embed=embed)
        return

    success_message = None
    error_message = None

    try:
        if setting == "log_channel": 
            new_value_id = int(value)
            if not interaction.guild.get_channel(new_value_id): 
                error_message = "Invalid channel ID. The specified channel was not found in this server."
            else:
                await set_guild_config(guild_id, "log_channel_id", new_value_id)
                success_message = f"Log channel set to <#{new_value_id}>."
        elif setting == "link_channel": 
            new_value_id = int(value)
            if not interaction.guild.get_channel(new_value_id):
                error_message = "Invalid channel ID. The specified channel was not found in this server."
            else:
                await set_guild_config(guild_id, "link_channel_id", new_value_id)
                success_message = f"Link-only channel set to <#{new_value_id}>. URL domain checks will be bypassed in this channel."
        
        elif setting == "allowed_languages": 
            langs = [lang.strip().lower() for lang in value.split(',') if lang.strip()]
            if not langs: error_message = "No valid languages provided. Please use 2-letter codes like 'en,fr'."
            else:
                channel_cfg_json = await get_guild_config(guild_id, db_key_for_channel_json, {})
                channel_cfg_json["language"] = langs 
                await set_guild_config(guild_id, db_key_for_channel_json, channel_cfg_json)
                success_message = f"Allowed languages for {target_channel_for_config.mention} set to: `{', '.join(langs)}`."
        
        elif setting == "allowed_topics": 
            topics_list = [topic.strip().lower() for topic in value.split(',') if topic.strip()]
            channel_cfg_json = await get_guild_config(guild_id, db_key_for_channel_json, {})
            channel_cfg_json["topics"] = topics_list 
            await set_guild_config(guild_id, db_key_for_channel_json, channel_cfg_json)
            success_message = f"Allowed topics for {target_channel_for_config.mention} set to: `{', '.join(topics_list)}`."

        elif setting == "permitted_domains": 
            domains_list = [d.strip().lower() for d in value.split(',') if '.' in d.strip()] 
            if not domains_list:
                error_message = "No valid domains provided. Enter comma-separated domains like `example.com, anothersite.org`."
            else:
                await set_guild_config(guild_id, "permitted_domains", domains_list)
                success_message = f"Permitted domains for this guild updated to: `{', '.join(domains_list)}`."
        else:
            error_message = f"Unknown setting: `{setting}`. Valid settings: `log_channel`, `link_channel`, `allowed_languages`, `allowed_topics`, `permitted_domains`."

    except ValueError: 
        error_message = "Invalid value format. Channel IDs must be numbers. For lists, use comma-separation."
    except Exception as e:
        logger.error(f"Error during /configure command for guild {guild_id}, setting {setting}: {e}", exc_info=True)
        error_message = "An unexpected error occurred while setting the configuration."

    if error_message:
        embed = discord.Embed(title="Configuration Error", description=error_message, color=discord.Color.red())
    elif success_message:
        embed = discord.Embed(title="✅ Configuration Updated", description=success_message, color=discord.Color.green())
    
    await interaction.followup.send(embed=embed)


@bot.tree.command(name="get_config", description="View bot configuration settings for this guild or a channel.")
@app_commands.default_permissions(manage_guild=True)
@app_commands.describe(
    setting="[Optional] Specific setting to view (e.g., 'log_channel', 'allowed_languages').",
    channel="[Optional] Channel to view specific settings for (e.g., for 'allowed_languages')."
)
async def get_config(
    interaction: discord.Interaction,
    setting: str = None,
    channel: discord.TextChannel = None
):
    """View bot configuration settings."""
    await interaction.response.defer(ephemeral=True)
    guild_id = interaction.guild.id
    
    target_channel_for_display = channel if channel else interaction.channel 

    embed = discord.Embed(title=f"⚙️ Bot Configuration for {interaction.guild.name}", color=discord.Color.blue())
    embed.set_footer(text=f"Requested by {interaction.user.display_name} | Guild ID: {guild_id}")

    async def add_config_field(name: str, value_to_display, default_value_text="Bot Default", is_channel_specific=False, channel_obj=None):
        display_text = f"`{value_to_display}`" if value_to_display is not None else f"Not set (Using {default_value_text})"
        if is_channel_specific and channel_obj:
            name = f"{name} (Channel: {channel_obj.mention})"
        embed.add_field(name=name, value=display_text, inline=False)

    if setting: 
        setting_lower = setting.lower()
        if setting_lower == "log_channel":
            log_ch_id = await get_guild_config(guild_id, "log_channel_id", bot_config.default_log_channel_id)
            log_ch_obj = interaction.guild.get_channel(log_ch_id)
            await add_config_field("Log Channel", f"<#{log_ch_id}> ({log_ch_id})" if log_ch_obj else f"ID: {log_ch_id} (Channel not found or default)", f"<#{bot_config.default_log_channel_id}>")
        elif setting_lower == "link_channel":
            link_ch_id = await get_guild_config(guild_id, "link_channel_id", None)
            link_ch_obj = interaction.guild.get_channel(link_ch_id) if link_ch_id else None
            await add_config_field("Link-Only Channel", f"<#{link_ch_id}> ({link_ch_id})" if link_ch_obj else "None", "None (links checked everywhere)")
        elif setting_lower == "permitted_domains":
            domains = await get_guild_config(guild_id, "permitted_domains", list(bot_config.permitted_domains))
            await add_config_field("Permitted Domains (Guild)", f"{', '.join(domains)}" if domains else "None", f"{', '.join(bot_config.permitted_domains)}")
        
        elif setting_lower in ["allowed_languages", "allowed_topics"]:
            channel_cfg_json = await get_guild_config(guild_id, f"channel_config_{target_channel_for_display.id}", {})
            key_in_json = "language" if setting_lower == "allowed_languages" else "topics"
            default_cfg_for_channel = bot_config.default_channel_configs.get(target_channel_for_display.id, {})
            default_val = default_cfg_for_channel.get(key_in_json)

            value = channel_cfg_json.get(key_in_json, default_val) 
            
            default_text = f"{', '.join(default_val)}" if default_val else "None/Bot Global"
            await add_config_field(setting.replace("_", " ").title(), f"{', '.join(value)}" if value else "Not Set", default_text, True, target_channel_for_display)
        else:
            embed.description = f"Unknown setting: `{setting}`. Please choose from `log_channel`, `link_channel`, `permitted_domains`, `allowed_languages`, `allowed_topics`."
            embed.color = discord.Color.orange()
    else: 
        log_ch_id = await get_guild_config(guild_id, "log_channel_id", bot_config.default_log_channel_id)
        log_ch_obj = interaction.guild.get_channel(log_ch_id)
        await add_config_field("Log Channel", f"<#{log_ch_id}> ({log_ch_id})" if log_ch_obj else f"ID: {log_ch_id}", f"<#{bot_config.default_log_channel_id}>")

        link_ch_id = await get_guild_config(guild_id, "link_channel_id", None)
        link_ch_obj = interaction.guild.get_channel(link_ch_id) if link_ch_id else None
        await add_config_field("Link-Only Channel", f"<#{link_ch_id}> ({link_ch_id})" if link_ch_obj else "None", "None")

        domains = await get_guild_config(guild_id, "permitted_domains", list(bot_config.permitted_domains))
        await add_config_field("Permitted Domains (Guild)", f"{', '.join(domains)}" if domains else "None", f"{', '.join(bot_config.permitted_domains)}")
        
        embed.add_field(name="\u200b", value="--- Channel Specific Settings ---", inline=False) 

        channel_cfg_json = await get_guild_config(guild_id, f"channel_config_{target_channel_for_display.id}", {})
        default_channel_cfg = bot_config.default_channel_configs.get(target_channel_for_display.id, {})

        lang_val = channel_cfg_json.get("language", default_channel_cfg.get("language"))
        await add_config_field("Allowed Languages", f"{', '.join(lang_val)}" if lang_val else "Any", "Any/Bot Global", True, target_channel_for_display)
        
        topic_val = channel_cfg_json.get("topics", default_channel_cfg.get("topics"))
        await add_config_field("Allowed Topics", f"{', '.join(topic_val)}" if topic_val else "Any", "Any/Bot Global", True, target_channel_for_display)

        if not channel: 
             embed.add_field(name="Note", value=f"Channel settings displayed for current channel ({target_channel_for_display.mention}). Use the `channel` option to see others.", inline=False)


    await interaction.followup.send(embed=embed)


async def init_db():
    global db_conn
    try:
        db_conn = await aiosqlite.connect('database.db')
        await db_conn.execute('''
            CREATE TABLE IF NOT EXISTS guild_configs (
                guild_id INTEGER NOT NULL,
                config_key TEXT NOT NULL,
                config_value TEXT,
                PRIMARY KEY (guild_id, config_key) 
            )
        ''')
        await db_conn.execute('''
            CREATE TABLE IF NOT EXISTS infractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL,
                points INTEGER NOT NULL,
                timestamp TEXT NOT NULL,    -- ISO format UTC datetime
                violation_type TEXT,        -- e.g., "spam", "nsfw_text", "foreign_language"
                message_id INTEGER,         -- ID of the offending message
                channel_id INTEGER          -- ID of the channel where infraction occurred
            )
        ''')
        await db_conn.execute('''
            CREATE TABLE IF NOT EXISTS temp_bans (
                user_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL,
                unban_time TEXT NOT NULL,   -- ISO format UTC datetime for when to unban
                ban_reason TEXT,
                PRIMARY KEY (user_id, guild_id)
            )
        ''')
        await db_conn.execute('''
            CREATE TABLE IF NOT EXISTS message_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,    -- Unix timestamp for easy comparison
                message_content TEXT        -- Raw message content
            )
        ''')
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_infractions_user_guild_time ON infractions (user_id, guild_id, timestamp)')
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_temp_bans_unban_time ON temp_bans (unban_time)')
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_message_history_user_guild_time ON message_history (user_id, guild_id, timestamp)')
        
        await db_conn.commit()
        logger.info("✅ Database initialized successfully with corrected schema and indexes.")
    except Exception as e:
        logger.critical(f"❌ Failed to initialize database: {e}", exc_info=True)
        exit(1) 

@bot.event
async def on_ready():
    """Event that fires when the bot is ready."""
    logger.info(f'Logged in as {bot.user} (ID: {bot.user.id})')
    logger.info(f'Discord.py Version: {discord.__version__}')
    logger.info('------')
    
    await init_db() 

    global LANGUAGE_MODEL
    try:
        LANGUAGE_MODEL = fasttext.load_model(FASTTEXT_MODEL_PATH)
        logger.info(f"✅ FastText model loaded from {FASTTEXT_MODEL_PATH}")
    except ValueError as e: 
        if "Cannot load lid.176.ftz" in str(e) or "not found" in str(e).lower():
            logger.error(f"❌ FastText model file '{FASTTEXT_MODEL_PATH}' not found or invalid. Language detection will default to 'en'. {e}")
        else:
            logger.error(f"❌ Error loading FastText model: {e}. Language detection will default to 'en'.")
        LANGUAGE_MODEL = None 
    except Exception as e:
        logger.error(f"❌ Unexpected error loading FastText model: {e}", exc_info=True)
        LANGUAGE_MODEL = None

    global http_session
    if not http_session or http_session.closed:
        http_session = ClientSession()
        logger.info("✅ Aiohttp ClientSession initialized.")

    if not decay_points.is_running():
        decay_points.start()
    
    moderation_cog = Moderation(bot)
    await bot.add_cog(moderation_cog) 

    try:
        synced = await bot.tree.sync() 
        logger.info(f"✅ Synced {len(synced)} slash commands.")
    except Exception as e:
        logger.error(f"❌ Failed to sync slash commands: {e}", exc_info=True)

    asyncio.create_task(start_http_server())

@bot.event
async def on_disconnect():
    logger.warning("Bot disconnected from Discord. Attempting to reconnect...")

@bot.event
async def on_resumed():
    logger.info("✅ Bot resumed connection to Discord.")

async def main():
    """Main function to run the bot."""
    async with bot: 
        await bot.start(DISCORD_TOKEN)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot shutting down due to KeyboardInterrupt (Ctrl+C)...")
    except Exception as e: 
        logger.critical(f"Unhandled exception at the very top level: {e}", exc_info=True)
    finally:
        logger.info("Initiating final cleanup...")
        
        if http_session and not http_session.closed:
            try:
                async def close_session(): 
                    await http_session.close()
                asyncio.run(close_session()) 
                logger.info("Aiohttp session closed.")
            except RuntimeError as e: 
                logger.error(f"Error closing aiohttp session during shutdown: {e}. Might already be closed or loop issue.")
            except Exception as e:
                 logger.error(f"Unexpected error closing aiohttp session: {e}", exc_info=True)


        if db_conn:
            try:
                async def close_db():
                    await db_conn.close()
                asyncio.run(close_db())
                logger.info("Database connection closed.")
            except RuntimeError as e:
                logger.error(f"Error closing database connection: {e}.")
            except Exception as e:
                 logger.error(f"Unexpected error closing database connection: {e}", exc_info=True)

        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if tasks:
        logger.info(f"Cancelling {len(tasks)} outstanding asyncio tasks...")
        for task in tasks:
            task.cancel()
        try:
            async def gather_cancelled_tasks(): 
            wait asyncio.gather(*tasks, return_exceptions=True)
            asyncio.run(gather_cancelled_tasks())
            logger.info("Outstanding tasks cancelled.")
        except RuntimeError as e: 
            logger.warning(f"Could not gather cancelled tasks, loop may be closed: {e}")
        except Exception as e:
            logger.error(f"Error during task cancellation: {e}", exc_info=True)

        logger.info("✅ Cleanup attempt complete. Bot shut down.")
