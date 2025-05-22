import os
import re
import asyncio
import logging
from collections import deque, defaultdict
from datetime import datetime, timedelta
from urllib.parse import urlparse
import aiosqlite
import discord
import fasttext
from aiohttp import ClientSession, web
from discord.ext import commands, tasks
from dotenv import load_dotenv
from discord import app_commands 
import httpx

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger('discord_bot')

DISCORD_TOKEN = os.getenv("ADROIT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FASTTEXT_MODEL_PATH = os.getenv("FASTTEXT_MODEL_PATH", "lid.176.ftz")

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

cached_guild_configs = defaultdict(dict) 
user_message_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=5)))
user_message_timestamps = defaultdict(lambda: defaultdict(deque))

DEFAULT_LOG_CHANNEL_ID = 1113377818424922132

DEFAULT_CHANNEL_CONFIGS = {
    1113377809440722974: {"language": ["en"]},
    1322517478365990984: {"language": ["en"], "topics": ["politics"]},
    1113377810476716132: {"language": ["en"]},
    1321499824926888049: {"language": ["fr"]},
    1122525009102000269: {"language": ["de"]},
    1122523546355245126: {"language": ["ru"]},
    1122524817904635904: {"language": ["zh"]},
    1242768362237595749: {"language": ["es"]}
}

FORBIDDEN_TEXT_PATTERN = re.compile(
    r"(discord\.gg/|join\s+our|server\s+invite|free\s+nitro|check\s+out\s+my|follow\s+me|subscribe\s+to|buy\s+now)",
    re.IGNORECASE
)
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|net|org|io|dev)\b)") 

PERMITTED_DOMAINS = [
    "googleusercontent.com", "tenor.com", "giphy.com", "tiktok.com",
    "youtube.com", "youtu.be", "docs.google.com", "cdn.discordapp.com",
    "roblox.com", "github.com", "theuselessweb.com",
    "wikipedia.com", "twitch.tv", "tiktok.com", "reddit.com", "x.com", "twitter.com"
]

PUNISHMENT_SYSTEM = {
    "points_thresholds": {
        5: {"action": "warn", "message": "Warnings make your sins weigh heavier, think twice before sending something inappropriate."},
        10: {"action": "mute", "duration_hours": 1, "reason": "Spam/Minor violations"},
        15: {"action": "kick", "reason": "Repeated violations"},
        25: {"action": "temp_ban", "duration_days": 1, "reason": "Serious/Persistent violations"},
        40: {"action": "ban", "reason": "Severe/Accumulated violations"}
    },
    "violations": {
        "discrimination": {"points": 3, "severity": "Medium"},
        "spam": {"points": 2, "severity": "Medium"},
        "nsfw": {"points": 5, "severity": "High"},
        "advertising": {"points": 3, "severity": "Medium"},
        "politics_discussion": {"points": 3, "severity": "Medium"}, 
        "off_topic": {"points": 1, "severity": "Low"},
        "foreign_language": {"points": 1, "severity": "Low"},
        "openai_moderation": {"points": 3, "severity": "Medium"},
        "excessive_mentions": {"points": 2, "severity": "Medium"},
        "excessive_attachments": {"points": 2, "severity": "Medium"},
        "long_message": {"points": 2, "severity": "Medium"}
    }
}

SPAM_WINDOW = 10  
SPAM_LIMIT = 5   
MENTION_LIMIT = 5
MAX_MESSAGE_LENGTH = 800
MAX_ATTACHMENTS = 4

discrimination_words = set()
discrimination_patterns = []
nsfw_words = set()
nsfw_patterns = []

def load_terms_from_file(filepath: str) -> tuple[set, list]:
    """Loads terms from a file, separating words and phrases for regex compilation."""
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
    except FileNotFoundError:
        logger.warning(f"Warning: {filepath} not found. Using empty list/set for terms.")
    except Exception as e:
        logger.error(f"Error loading terms from {filepath}: {e}")
    compiled_patterns = [re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE) for phrase in phrases]
    return words, compiled_patterns

discrimination_words, discrimination_patterns = load_terms_from_file('discrimination_terms.txt')
nsfw_words, nsfw_patterns = load_terms_from_file('nsfw_terms.txt')

def clean_message_content(text: str) -> str:
    """Clean the message text for analysis."""
    return text.strip().lower()

async def get_guild_config(guild_id: int, key: str, default=None):
    """Retrieve a guild-specific configuration from cache or DB."""
    if guild_id in cached_guild_configs and key in cached_guild_configs[guild_id]:
        return cached_guild_configs[guild_id][key]

    async with db_conn.cursor() as cursor:
        await cursor.execute('SELECT value FROM guild_configs WHERE guild_id = ? AND key = ?', (guild_id, key))
        result = await cursor.fetchone()
        if result:
            value = result[0]
            try:
                if key in ["allowed_languages", "allowed_topics", "permitted_domains"]: 
                    value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
            cached_guild_configs[guild_id][key] = value
            return value
    return default

async def set_guild_config(guild_id: int, key: str, value):
    """Set a guild-specific configuration and update cache."""
    stored_value = value
    if isinstance(value, (list, dict)):
        stored_value = json.dumps(value)

    async with db_conn.cursor() as cursor:
        await cursor.execute(
            'INSERT OR REPLACE INTO guild_configs (guild_id, key, value) VALUES (?, ?, ?)',
            (guild_id, key, stored_value)
        )
        await db_conn.commit()
    cached_guild_configs[guild_id][key] = value
    logger.info(f"Updated guild config: Guild {guild_id}, Key '{key}', Value '{value}'")

async def detect_language_ai(text: str) -> str:
    """Detect the language of the given text using FastText."""
    clean_text = clean_message_content(text)
    if not LANGUAGE_MODEL:
        logger.error("FastText model not loaded. Defaulting to 'en'.")
        return "en"
    try:
        prediction = LANGUAGE_MODEL.predict(clean_text)
        return prediction[0][0].replace("__label__", "")
    except Exception as e:
        logger.error(f"FastText language detection error: {e}")
        return "en" 

async def log_action(action: str, member: discord.Member, reason: str):
    """Log moderation actions to a specified channel and console."""
    log_channel_id = await get_guild_config(member.guild.id, "log_channel_id", DEFAULT_LOG_CHANNEL_ID)
    log_channel = bot.get_channel(log_channel_id) or member.guild.get_channel(log_channel_id)

    if log_channel:
        try:
            embed = discord.Embed(
                title=f"Moderation Action: {action.upper()}",
                description=f"**User:** {member.mention} (`{member.id}`)\n**Reason:** {reason}\n**Timestamp:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
                color=discord.Color.red() if action in ["ban", "kick", "mute"] else discord.Color.orange()
            )
            await log_channel.send(embed=embed)
        except discord.Forbidden:
            logger.error(f"Missing permissions to send logs to channel {log_channel.name} ({log_channel.id}).")
        except Exception as e:
            logger.error(f"Error sending log embed to channel: {e}")
    else:
        logger.info(f"LOG: {action.upper()} applied to {member.display_name} ({member.id}) for: {reason}")
        logger.warning(f"Log channel (ID: {log_channel_id}) not found or accessible for guild {member.guild.name}.")


async def check_openai_moderation(text: str) -> dict:
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set. Skipping OpenAI moderation check.")
        return {"flagged": False, "categories": {}}

    url = "https://api.openai.com/v1/moderations"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {"input": text}

    retries = 3
    for i in range(retries):
        try:
            async with http_session.post(url, headers=headers, json=data, timeout=5) as response:
                response.raise_for_status()
                return await response.json().get("results", [{}])[0]
        except aiohttp.client_exceptions.ClientResponseError as e:
            if e.status == 429:
                retry_after = e.headers.get("Retry-After") 
                wait_time = int(retry_after) if retry_after else (2 ** i) 
                logger.warning(f"OpenAI moderation API hit rate limit (429). Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"HTTP error during OpenAI moderation check: {e}")
                return {"flagged": False, "categories": {}}
        except asyncio.TimeoutError:
            logger.error("OpenAI moderation API request timed out.")
            return {"flagged": False, "categories": {}}
        except Exception as e:
            logger.error(f"Unexpected error with OpenAI moderation API: {e}")
            return {"flagged": False, "categories": {}}
    logger.error(f"Failed to get OpenAI moderation response after {retries} retries.")
    return {"flagged": False, "categories": {}}

async def apply_punishment(member: discord.Member, action: str, reason: str, duration: timedelta = None):
    """Apply a punishment to a member based on infraction points."""
    try:
        if action == "warn":
            warning_message = PUNISHMENT_SYSTEM["points_thresholds"][5]["message"] 
            await member.send(warning_message)
            await log_action("warn", member, reason)
        elif action == "mute":
            if duration:
                await member.timeout(duration, reason=reason)
                await member.send(f"You have been muted for {duration}. Reason: {reason}")
                await log_action("mute", member, reason)
            else:
                logger.warning(f"Attempted to mute {member.display_name} without duration. Skipping.")
        elif action == "kick":
            await member.kick(reason=reason)
            await log_action("kick", member, reason)
        elif action == "temp_ban":
            if duration:
                unban_time = datetime.utcnow() + duration
                async with db_conn.cursor() as cursor:
                    await cursor.execute(
                        'INSERT INTO temp_bans (user_id, guild_id, unban_time, ban_reason) VALUES (?, ?, ?, ?)',
                        (member.id, member.guild.id, unban_time.isoformat(), reason)
                    )
                    await db_conn.commit()
                await member.ban(reason=reason, delete_message_days=0) 
                await member.send(f"You have been temporarily banned until {unban_time.isoformat()} UTC. Reason: {reason}")
                await log_action("temp_ban", member, reason)
            else:
                logger.warning(f"Attempted to temp_ban {member.display_name} without duration. Skipping.")
        elif action == "ban":
            await member.ban(reason=reason, delete_message_days=0) 
            await member.send(f"You have been permanently banned. Reason: {reason}")
            await log_action("ban", member, reason)
    except discord.Forbidden:
        logger.error(f"Missing permissions to {action} {member.display_name} in {member.guild.name}. "
                     f"Please check bot permissions.")
    except discord.HTTPException as e:
        logger.error(f"Discord API error while applying {action} to {member.display_name}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error applying {action} to {member.display_name}: {e}", exc_info=True)

async def log_violation(member: discord.Member, violation_type: str, message: discord.Message):
    """Log a violation, add points, and apply punishment if thresholds are met."""
    points = PUNISHMENT_SYSTEM["violations"].get(violation_type, {"points": 0})["points"]
    reason = f"Violation: {violation_type} in message: '{message.content[:100]}...'"
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
            'SELECT SUM(points) FROM infractions WHERE user_id = ? AND guild_id = ? AND timestamp > ?',
            (user_id, guild_id, cutoff_date)
        )
        total_points = (await cursor.fetchone())[0] or 0

    logger.info(f"User {member.display_name} ({user_id}) accumulated {points} points for '{violation_type}'. "
                f"Total points (for a month): {total_points}")

    for threshold_points in sorted(PUNISHMENT_SYSTEM["points_thresholds"].keys(), reverse=True):
        if total_points >= threshold_points:
            punishment_config = PUNISHMENT_SYSTEM["points_thresholds"][threshold_points]
            action = punishment_config["action"]
            duration = None
            if "duration_hours" in punishment_config:
                duration = timedelta(hours=punishment_config["duration_hours"])
            elif "duration_days" in punishment_config:
                duration = timedelta(days=punishment_config["duration_days"])

            punishment_reason = punishment_config.get("reason", f"Accumulated {total_points} infraction points.")
            await apply_punishment(member, action, punishment_reason, duration)
            break 

def get_domain_from_url(url: str) -> str:
    """Extract the domain from a URL, handling common prefixes."""
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url 
    parsed = urlparse(url)
    domain = parsed.netloc
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain.lower()

def is_permitted_domain(domain: str) -> bool:
    """Check if the extracted domain is in the permitted list or is a subdomain of a permitted domain."""
    for perm_domain in PERMITTED_DOMAINS:
        perm_domain_cleaned = get_domain_from_url(perm_domain) 
        if domain == perm_domain_cleaned or domain.endswith('.' + perm_domain_cleaned):
            return True
    return False

async def handle_health_check(request):
    """Handle HTTP requests for health checks."""
    return web.Response(text="Bot is awake and healthy!", status=200, headers={"Content-Type": "text/plain"})

async def start_http_server():
    """Start an HTTP server to keep the bot running (e.g., on Heroku)."""
    try:
        app = web.Application()
        app.router.add_get('/', handle_health_check)
        runner = web.AppRunner(app)
        await runner.setup()
        port = int(os.getenv("PORT", "8080"))
        site = web.TCPSite(runner, host='0.0.0.0', port=port)
        await site.start()
        logger.info(f"✅ HTTP server running on port {port}")
    except Exception as e:
        logger.error(f"❌ Failed to start HTTP server: {e}", exc_info=True)


@tasks.loop(hours=24)
async def decay_points():
    """Remove infraction records older than 90 days and clean up temp_bans."""
    if not db_conn:
        logger.warning("Database connection not available for decay_points task. Skipping.")
        return

    async with db_conn.cursor() as cursor:
        cutoff_date_infractions = (datetime.utcnow() - timedelta(days=90)).isoformat()
        await cursor.execute('DELETE FROM infractions WHERE timestamp < ?', (cutoff_date_infractions,))
        logger.info(f"Deleted infraction records older than 90 days.")

        now_iso = datetime.utcnow().isoformat()
        await cursor.execute('SELECT user_id, guild_id FROM temp_bans WHERE unban_time <= ?', (now_iso,))
        expired_bans = await cursor.fetchall()

        for user_id, guild_id in expired_bans:
            guild = bot.get_guild(guild_id)
            if guild:
                try:
                    user = await bot.fetch_user(user_id) 
                    if user:
                        await guild.unban(user, reason="Temporary ban expired.")
                        logger.info(f"Unbanned user {user.name} ({user_id}) from guild {guild.name} ({guild_id}) - temp ban expired.")
                        await log_action("unban", user, "Temporary ban expired automatically.")
                except discord.NotFound:
                    logger.warning(f"Could not find user {user_id} for unban in guild {guild_id}.")
                except discord.Forbidden:
                    logger.error(f"Missing permissions to unban user {user_id} in guild {guild.name}.")
                except Exception as e:
                    logger.error(f"Error during unban process for user {user_id} in guild {guild_id}: {e}", exc_info=True)

            await cursor.execute('DELETE FROM temp_bans WHERE user_id = ? AND guild_id = ?', (user_id, guild_id))
        await db_conn.commit()
    logger.info("Decay points and temporary bans cleanup completed.")


@tasks.loop(hours=12) 
async def cleanup_message_tracking():
    """Clean up message count and history tracking for inactive users."""
    now = datetime.utcnow()
    for guild_id in list(user_message_timestamps.keys()):
        for user_id in list(user_message_timestamps[guild_id].keys()):
            user_message_timestamps[guild_id][user_id] = deque(
                t for t in user_message_timestamps[guild_id][user_id]
                if (now - t).total_seconds() < SPAM_WINDOW 
            )
            if not user_message_timestamps[guild_id][user_id]:
                del user_message_timestamps[guild_id][user_id]
        if not user_message_timestamps[guild_id]:
            del user_message_timestamps[guild_id]

    for guild_id in list(user_message_history.keys()):
        for user_id in list(user_message_history[guild_id].keys()):
            if user_message_history[guild_id][user_id] and \
               (now - user_message_history[guild_id][user_id][-1][0]).total_seconds() > timedelta(days=7).total_seconds():
                user_message_history[guild_id][user_id].clear()
            if not user_message_history[guild_id][user_id]:
                del user_message_history[guild_id][user_id]
        if not user_message_history[guild_id]:
            del user_message_history[guild_id]
    logger.info("Message tracking cleanup completed.")


@bot.event
async def on_ready():
    """Handle bot startup tasks."""
    logger.info(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    logger.info("------")

    global http_session
    http_session = ClientSession()

    global db_conn
    try:
        db_conn = await aiosqlite.connect('infractions.db')
        await init_db()
        logger.info("✅ Database initialized.")
    except Exception as e:
        logger.critical(f"❌ Failed to connect to or initialize database: {e}", exc_info=True)

    global LANGUAGE_MODEL
    try:
        LANGUAGE_MODEL = fasttext.load_model(FASTTEXT_MODEL_PATH)
        logger.info(f"✅ Successfully loaded FastText model from {FASTTEXT_MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to load FastText model from {FASTTEXT_MODEL_PATH}: {e}. Language detection may not work.", exc_info=True)

    decay_points.start()
    cleanup_message_tracking.start()
    await start_http_server()
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} command(s).")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}", exc_info=True)

@bot.event
async def on_error(event_name, *args, **kwargs):
    """General error handler for bot events."""
    logger.error(f"Unhandled error in event '{event_name}':", exc_info=True)

@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    """Error handler for command invocations."""
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Missing argument: {error.param}. Please check the command usage.")
    elif isinstance(error, commands.BadArgument):
        await ctx.send(f"Bad argument: {error}. Please provide a valid value.")
    elif isinstance(error, commands.CommandNotFound):
        pass
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("You don't have the necessary permissions to use this command.")
    elif isinstance(error, commands.BotMissingPermissions):
        await ctx.send(f"I don't have the necessary permissions to perform this action. I need: {', '.join(error.missing_permissions)}")
        logger.warning(f"Bot missing permissions in guild {ctx.guild.id}, channel {ctx.channel.id}: {error.missing_permissions}")
    elif isinstance(error, commands.NoPrivateMessage):
        await ctx.send("This command cannot be used in private messages.")
    else:
        logger.error(f"Unhandled command error in {ctx.command}: {error}", exc_info=True)
        await ctx.send("An unexpected error occurred while processing your command.")

@bot.event
async def on_guild_join(guild: discord.Guild):
    """Log when the bot joins a new guild."""
    logger.info(f"Joined guild: {guild.name} ({guild.id})")


class Moderation(commands.Cog):
    def __init__(self, bot_instance: commands.Bot):
        self.bot = bot_instance

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Handle incoming messages and apply moderation rules."""
        if message.author.bot or not message.guild:
            return

        violations = set()
        guild_id = message.guild.id
        user_id = message.author.id
        content_lower = message.content.lower()
        now = datetime.utcnow()

        channel_config_db = await get_guild_config(guild_id, f"channel_config_{message.channel.id}", {})
        channel_cfg = DEFAULT_CHANNEL_CONFIGS.get(message.channel.id, {})
        channel_cfg.update(channel_config_db) 

        user_message_timestamps[guild_id][user_id].append(now)
        while user_message_timestamps[guild_id][user_id] and \
              (now - user_message_timestamps[guild_id][user_id][0]).total_seconds() >= SPAM_WINDOW:
            user_message_timestamps[guild_id][user_id].popleft()

        if len(user_message_timestamps[guild_id][user_id]) > SPAM_LIMIT:
            violations.add("spam")

        user_message_history[guild_id][user_id].append((now, message.content))
        if message.content in [msg_content for _, msg_content in list(user_message_history[guild_id][user_id])[:-1]]:
            violations.add("spam")

        if len(message.mentions) > MENTION_LIMIT:
            violations.add("excessive_mentions")
        if len(message.attachments) > MAX_ATTACHMENTS:
            violations.add("excessive_attachments")
        if len(message.content) > MAX_MESSAGE_LENGTH:
            violations.add("long_message")

        allowed_languages = channel_cfg.get("language")
        if allowed_languages:
            detected_lang = await detect_language_ai(message.content)
            if detected_lang not in allowed_languages:
                violations.add("foreign_language")
                logger.debug(f"Foreign language detected for {message.author.name} in {message.channel.name}: {detected_lang}")

        link_channel_id = await get_guild_config(guild_id, "link_channel_id")

        if FORBIDDEN_TEXT_PATTERN.search(content_lower):
            violations.add("advertising")

        urls = URL_PATTERN.findall(message.content)
        if urls and message.channel.id != link_channel_id:
            for url_match in urls:
                url = url_match[0] if isinstance(url_match, tuple) else url_match
                domain = get_domain_from_url(url)
                if domain and not is_permitted_domain(domain):
                    violations.add("advertising")
                    logger.debug(f"Forbidden URL detected: {url} (Domain: {domain})")
                    break 

        words_in_message = set(re.findall(r'\b\w+\b', content_lower))
        if any(word in discrimination_words for word in words_in_message) or \
           any(pattern.search(content_lower) for pattern in discrimination_patterns):
            violations.add("discrimination")

        if any(word in nsfw_words for word in words_in_message) or \
           any(pattern.search(content_lower) for pattern in nsfw_patterns):
            violations.add("nsfw")

        allowed_topics = channel_cfg.get("topics")
        if allowed_topics:
            if not any(topic.lower() in content_lower for topic in allowed_topics):
                violations.add("off_topic")
        else: 
            general_sensitive_terms = ["politics", "religion", "god", "allah", "jesus", "church", "mosque",
                                       "temple", "bible", "quran", "torah", "democrat", "republican", "liberal", "conservative"]
            if any(term in content_lower for term in general_sensitive_terms):
                violations.add("politics_discussion")

        if not violations and OPENAI_API_KEY:
            moderation_result = await check_openai_moderation(message.content)
            if moderation_result.get("flagged", False):
                violations.add("openai_moderation")
                for category, flagged in moderation_result.get("categories", {}).items():
                    if flagged:
                        logger.debug(f"OpenAI flagged category: {category}")
                        if category in ["hate", "hate/threatening", "self-harm", "sexual", "sexual/minors", "violence", "violence/graphic"]:
                            violations.add("nsfw" if category.startswith("sexual") else "discrimination")

        if violations:
            logger.info(f"Message from {message.author.name} ({user_id}) in {message.channel.name} ({message.channel.id}) "
                        f"flagged with violations: {', '.join(violations)}")
            try:
                await message.delete()
                await log_action("message_deleted", message.author, f"Violations: {', '.join(violations)}")
            except discord.Forbidden:
                logger.error(f"Missing permissions to delete message by {message.author.name} in {message.channel.name}.")
            except discord.HTTPException as e:
                logger.error(f"Error deleting message: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during message deletion: {e}", exc_info=True)

            for violation in violations:
                await log_violation(message.author, violation, message)

            warning_msg = f"{message.author.mention}, your message has been deleted due to: **{', '.join(v.replace('_', ' ').title() for v in violations)}**. Please review the server rules."
            try:
                await message.channel.send(warning_msg, delete_after=15)
            except discord.Forbidden:
                logger.error(f"Missing permissions to send warning message in {message.channel.name}.")


        await self.bot.process_commands(message)

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def set_log_channel(self, ctx: commands.Context, channel: discord.TextChannel):
        """Set the channel where moderation logs are sent."""
        await set_guild_config(ctx.guild.id, "log_channel_id", channel.id)
        await ctx.send(f"Moderation logs will now be sent to {channel.mention}.")
        await log_action("config_change", ctx.author, f"Set log channel to {channel.name}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def set_link_channel(self, ctx: commands.Context, channel: discord.TextChannel):
        """Set the channel where links are allowed without being flagged as advertising."""
        await set_guild_config(ctx.guild.id, "link_channel_id", channel.id)
        await ctx.send(f"Link posting channel set to {channel.mention}")
        await log_action("config_change", ctx.author, f"Set link channel to {channel.name}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def add_permitted_domain(self, ctx: commands.Context, domain: str):
        """Add a domain to the list of permitted domains for links."""
        current_permitted_domains = await get_guild_config(ctx.guild.id, "permitted_domains", list(PERMITTED_DOMAINS))
        cleaned_domain = get_domain_from_url(domain)
        if cleaned_domain and cleaned_domain not in current_permitted_domains:
            current_permitted_domains.append(cleaned_domain)
            await set_guild_config(ctx.guild.id, "permitted_domains", current_permitted_domains)
            await ctx.send(f"Added `{cleaned_domain}` to permitted domains.")
            await log_action("config_change", ctx.author, f"Added permitted domain: {cleaned_domain}")
        else:
            await ctx.send(f"`{cleaned_domain}` is already in the permitted list or invalid.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def remove_permitted_domain(self, ctx: commands.Context, domain: str):
        """Remove a domain from the list of permitted domains for links."""
        current_permitted_domains = await get_guild_config(ctx.guild.id, "permitted_domains", list(PERMITTED_DOMAINS))
        cleaned_domain = get_domain_from_url(domain)
        if cleaned_domain and cleaned_domain in current_permitted_domains:
            current_permitted_domains.remove(cleaned_domain)
            await set_guild_config(ctx.guild.id, "permitted_domains", current_permitted_domains)
            await ctx.send(f"Removed `{cleaned_domain}` from permitted domains.")
            await log_action("config_change", ctx.author, f"Removed permitted domain: {cleaned_domain}")
        else:
            await ctx.send(f"`{cleaned_domain}` not found in the permitted list or invalid.")

    @commands.command()
    @commands.has_permissions(manage_messages=True)
    async def infractions(self, ctx: commands.Context, member: discord.Member):
        """View a user's infraction history."""
        if not db_conn:
            await ctx.send("Database is not connected.")
            return

        async with db_conn.cursor() as cursor:
            cutoff_date = (datetime.utcnow() - timedelta(days=30)).isoformat()
            await cursor.execute(
                'SELECT SUM(points) FROM infractions WHERE user_id = ? AND guild_id = ? AND timestamp > ?',
                (member.id, ctx.guild.id, cutoff_date)
            )
            total_points = (await cursor.fetchone())[0] or 0

            await cursor.execute('''
                SELECT points, timestamp, violation_type FROM infractions
                WHERE user_id = ? AND guild_id = ?
                ORDER BY timestamp DESC
                LIMIT 10
            ''', (member.id, ctx.guild.id))
            records = await cursor.fetchall()

        embed = discord.Embed(
            title=f"Infraction Report for {member.display_name}",
            color=discord.Color.blue()
        )
        embed.add_field(name="Total Active Points (Last for a Month)", value=str(total_points), inline=False)

        if records:
            response = ""
            for points, timestamp_str, violation_type in records:
                timestamp = datetime.fromisoformat(timestamp_str)
                response += f"- **{points} points** ({violation_type.replace('_', ' ').title()}) on {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            embed.add_field(name="Recent Infractions (Last 10)", value=response, inline=False)
        else:
            embed.add_field(name="Recent Infractions", value=f"No recent infractions found for {member.display_name}.", inline=False)

        await ctx.send(embed=embed)

    @commands.command()
    @commands.has_permissions(manage_guild=True)
    async def clear_infractions(self, ctx: commands.Context, member: discord.Member):
        """Manually clear all infraction points for a user in this guild."""
        if not db_conn:
            await ctx.send("Database is not connected.")
            return
        async with db_conn.cursor() as cursor:
            await cursor.execute('DELETE FROM infractions WHERE user_id = ? AND guild_id = ?', (member.id, ctx.guild.id))
            await db_conn.commit()
        await ctx.send(f"Cleared all infraction points for {member.display_name}.")
        await log_action("infractions_cleared", ctx.author, f"Cleared infractions for {member.display_name}")

    @commands.command()
    @commands.has_permissions(kick_members=True)
    async def manual_warn(self, ctx: commands.Context, member: discord.Member, *, reason: str = "No reason provided."):
        """Manually warn a user."""
        await apply_punishment(member, "warn", f"Manual warn by {ctx.author.name}: {reason}")
        await ctx.send(f"Warned {member.display_name}.")

    @commands.command()
    @commands.has_permissions(kick_members=True)
    async def manual_mute(self, ctx: commands.Context, member: discord.Member, duration_hours: float, *, reason: str = "No reason provided."):
        """Manually mute a user for a specified number of hours."""
        duration = timedelta(hours=duration_hours)
        await apply_punishment(member, "mute", f"Manual mute by {ctx.author.name}: {reason}", duration=duration)
        await ctx.send(f"Muted {member.display_name} for {duration_hours} hours.")

    @commands.command()
    @commands.has_permissions(kick_members=True)
    async def manual_kick(self, ctx: commands.Context, member: discord.Member, *, reason: str = "No reason provided."):
        """Manually kick a user."""
        await apply_punishment(member, "kick", f"Manual kick by {ctx.author.name}: {reason}")
        await ctx.send(f"Kicked {member.display_name}.")

    @commands.command()
    @commands.has_permissions(ban_members=True)
    async def manual_ban(self, ctx: commands.Context, member: discord.Member, duration_days: float = None, *, reason: str = "No reason provided."):
        """Manually ban a user (temporarily or permanently)."""
        if duration_days:
            duration = timedelta(days=duration_days)
            await apply_punishment(member, "temp_ban", f"Manual temp ban by {ctx.author.name}: {reason}", duration=duration)
            await ctx.send(f"Temporarily banned {member.display_name} for {duration_days} days.")
        else:
            await apply_punishment(member, "ban", f"Manual ban by {ctx.author.name}: {reason}")
            await ctx.send(f"Permanently banned {member.display_name}.")

    @commands.command()
    @commands.has_permissions(ban_members=True)
    async def manual_unban(self, ctx: commands.Context, user_id: int, *, reason: str = "No reason provided."):
        """Manually unban a user by their ID."""
        user = discord.Object(id=user_id)
        try:
            await ctx.guild.unban(user, reason=f"Manual unban by {ctx.author.name}: {reason}")
            async with db_conn.cursor() as cursor:
                await cursor.execute('DELETE FROM temp_bans WHERE user_id = ? AND guild_id = ?', (user_id, ctx.guild.id))
                await db_conn.commit()
            await ctx.send(f"Unbanned user with ID {user_id}.")
            unbanned_member = await bot.fetch_user(user_id)
            await log_action("unban", unbanned_member, f"Manual unban by {ctx.author.name}: {reason}")
        except discord.NotFound:
            await ctx.send(f"User with ID {user_id} not found in ban list.")
        except discord.Forbidden:
            await ctx.send("I don't have permissions to unban members.")
        except Exception as e:
            logger.error(f"Error manually unbanning user {user_id}: {e}", exc_info=True)
            await ctx.send("An error occurred while trying to unban the user.")


class BotInfo(commands.Cog):
    def __init__(self, bot_instance: commands.Bot):
        self.bot = bot_instance

    @app_commands.command(name="awake", description="Check if the bot is awake.")
    async def awake(self, interaction: discord.Interaction):
        """Respond to the awake command."""
        await interaction.response.send_message("Awake. Never Sleep.", ephemeral=True)

    @commands.command()
    async def beta_classify(self, ctx: commands.Context):
        """Example command using language detection."""
        if not ctx.message.reference:
            await ctx.send("Please reply to a message to classify its language.")
            return

        referenced_message = await ctx.message.channel.fetch_message(ctx.message.reference.message_id)
        if not referenced_message:
            await ctx.send("Could not find the referenced message.")
            return

        if not referenced_message.content:
            await ctx.send("Referenced message has no text content to classify.")
            return

        lang = await detect_language_ai(referenced_message.content)
        await ctx.send(f"Language of the replied message: **{lang.upper()}**")

async def init_db():
    """Initialize the SQLite database for infractions, temp bans, and guild configs."""
    if not db_conn:
        logger.critical("Database connection not established. Cannot initialize DB.")
        return

    async with db_conn.cursor() as cursor:
        await cursor.execute('''
            CREATE TABLE IF NOT EXISTS infractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL,
                points INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                violation_type TEXT,
                message_id INTEGER,
                channel_id INTEGER
            )
        ''')
        await cursor.execute('''
            CREATE TABLE IF NOT EXISTS temp_bans (
                user_id INTEGER PRIMARY KEY, -- User can only have one active temp ban per guild
                guild_id INTEGER NOT NULL,
                unban_time TEXT NOT NULL,
                ban_reason TEXT
            )
        ''')
        await cursor.execute('''
            CREATE TABLE IF NOT EXISTS guild_configs (
                guild_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT, -- Store values as text, potentially JSON for lists/dicts
                PRIMARY KEY (guild_id, key)
            )
        ''')
        await db_conn.commit()
    logger.info("Database schema checked/created.")


async def main():
    async with bot:
        await bot.add_cog(Moderation(bot))
        await bot.add_cog(BotInfo(bot))
        await bot.start(DISCORD_TOKEN) # Deus ex Machina

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot shutting down due to KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"Unhandled exception during bot runtime: {e}", exc_info=True)
    finally:
        if db_conn:
            asyncio.run(db_conn.close())
            logger.info("Database connection closed.")
        if http_session:
            asyncio.run(http_session.close())
            logger.info("HTTP session closed.")
    logger.info("Bot process ended.")
    
