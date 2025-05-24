import os
import re
import asyncio
import logging
from collections import deque, defaultdict
from datetime import datetime, timedelta
from urllib.parse import urlparse
import json 
import aiosqlite
import discord
import fasttext
from aiohttp import ClientSession, web, client_exceptions 
from discord.ext import commands, tasks
from dotenv import load_dotenv
from discord import app_commands

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

cached_guild_configs = defaultdict(dict)
# user_message_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=5))) 
# user_message_timestamps = defaultdict(lambda: defaultdict(deque)) 

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
HAS_ALPHANUMERIC_PATTERN = re.compile(r'[a-zA-Z0-9]')


PERMITTED_DOMAINS = [
    "googleusercontent.com", "tenor.com", "giphy.com", "tiktok.com",
    "youtube.com", "youtu.be", "docs.google.com", "cdn.discordapp.com",
    "roblox.com", "github.com", "theuselessweb.com",
    "wikipedia.org", "twitch.tv", "reddit.com", "x.com", "twitter.com"
] 

PUNISHMENT_SYSTEM = {
    "points_thresholds": {
        5: {"action": "warn", "message": "Warnings make your sins weigh heavier, think twice before sending something inappropriate"},
        10: {"action": "mute", "duration_hours": 1, "reason": "Spam/Minor violations"},
        15: {"action": "kick", "reason": "Repeated violations"},
        25: {"action": "temp_ban", "duration_days": 1, "reason": "Serious/Persistent violations"},
        50: {"action": "temp_ban", "duration_years": 1, "reason": "Severe/Accumulated violations"}
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

def load_terms_from_file(filepath: str) -> set[str]:
    """Loads terms from a text file, one term per line."""
    terms = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                term = line.strip()
                if term:
                    terms.add(term)
        logger.info(f"Loaded {len(terms)} terms from {filepath}")
    except FileNotFoundError:
        logger.warning(f"Terms file not found: {filepath}. No terms loaded for this category.")
    except Exception as e:
        logger.error(f"Error loading terms from {filepath}: {e}", exc_info=True)
    return terms

def compile_patterns(terms: set[str]) -> list[re.Pattern]:
    """Compiles a set of terms into a list of regex patterns."""
    patterns = []
    for term in terms:
        try:
            patterns.append(re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE))
        except re.error as e:
            logger.error(f"Error compiling regex for term '{term}': {e}")
    return patterns

SPAM_WINDOW = 10  
SPAM_LIMIT = 5    
MENTION_LIMIT = 5
MAX_MESSAGE_LENGTH = 800 
MAX_ATTACHMENTS = 4

MIN_MSG_LEN_FOR_LANG_CHECK = 4 
MIN_CONFIDENCE_FOR_FLAGGING = 0.65
MIN_CONFIDENCE_SHORT_MSG = 0.75 
SHORT_MSG_THRESHOLD = 20 

COMMON_SAFE_FOREIGN_WORDS = {"bonjour", "hola", "merci", "gracias", "oui", "si", "nyet", "da", "salut", "ciao", "hallo", "guten tag"}

discrimination_words = load_terms_from_file('discrimination_terms.txt')
discrimination_patterns = compile_patterns(discrimination_words)

nsfw_words = load_terms_from_file('nsfw_terms.txt')
nsfw_patterns = compile_patterns(nsfw_words)

def load_terms_from_file(filepath: str) -> tuple[set, list]:
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

def clean_message_content(text: str) -> str:
    return text.strip().lower()

async def get_guild_config(guild_id, key, default):
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

    log_channel_id = await get_guild_config(current_guild.id, "log_channel_id", DEFAULT_LOG_CHANNEL_ID)
    log_channel = bot.get_channel(log_channel_id) or current_guild.get_channel(log_channel_id)

    user_mention = member_or_user.mention if isinstance(member_or_user, discord.Member) else f"{member_or_user.name}#{member_or_user.discriminator}"
    user_id = member_or_user.id
    display_name = member_or_user.display_name if isinstance(member_or_user, discord.Member) else member_or_user.name


    if log_channel:
        try:
            embed = discord.Embed(
                title=f"Moderation Action: {action.upper()}",
                description=f"**User:** {user_mention} (`{user_id}`)\n**Reason:** {reason}\n**Timestamp:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
                color=discord.Color.red() if action in ["ban", "kick", "mute", "temp_ban"] else discord.Color.orange()
            )
            await log_channel.send(embed=embed)
        except discord.Forbidden:
            logger.error(f"Missing permissions to send logs to channel {log_channel.name} ({log_channel.id}) in guild {current_guild.name}.")
        except Exception as e:
            logger.error(f"Error sending log embed to channel for guild {current_guild.name}: {e}")
    else:
        logger.info(f"LOG (Guild {current_guild.name}): {action.upper()} applied to {display_name} ({user_id}) for: {reason}")
        logger.warning(f"Log channel (ID: {log_channel_id}) not found or accessible for guild {current_guild.name}.")


async def check_openai_moderation(text: str) -> dict:
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

    retries = 3
    for i in range(retries):
        try:
            async with http_session.post(url, headers=headers, json=data, timeout=10) as response:
                response.raise_for_status()
                json_response = await response.json()
                results_list = json_response.get("results", [])
                if results_list:
                    return results_list[0]
                logger.warning(f"OpenAI moderation returned empty results list for text: {text[:100]}")
                return {"flagged": False, "categories": {}}
        except client_exceptions.ClientResponseError as e: 
            if e.status == 429: 
                retry_after_header = e.headers.get("Retry-After")
                wait_time = int(retry_after_header) if retry_after_header and retry_after_header.isdigit() else (2 ** (i + 1)) 
                logger.warning(f"OpenAI moderation API hit rate limit (429). Retrying in {wait_time} seconds... (Attempt {i+1}/{retries})")
                await asyncio.sleep(wait_time)
            elif e.status == 400: 
                 logger.warning(f"OpenAI moderation API returned 400 Bad Request. Text: '{text[:100]}...' Error: {e.message}")
                 return {"flagged": False, "categories": {}} 
            else:
                logger.error(f"HTTP error during OpenAI moderation check: {e.status} - {e.message} for text: {text[:100]}")
                return {"flagged": False, "categories": {}} 
        except asyncio.TimeoutError:
            logger.error(f"OpenAI moderation API request timed out after 10 seconds. (Attempt {i+1}/{retries}) Text: {text[:100]}")
            if i < retries - 1:
                await asyncio.sleep(2 ** (i + 1)) 
            else:
                return {"flagged": False, "categories": {}}
        except Exception as e:
            logger.error(f"Unexpected error with OpenAI moderation API: {e} for text: {text[:100]}", exc_info=True)
            return {"flagged": False, "categories": {}} 
    logger.error(f"Failed to get OpenAI moderation response after {retries} retries for text: {text[:100]}")
    return {"flagged": False, "categories": {}}


async def apply_punishment(member: discord.Member, action: str, reason: str, duration: timedelta = None):
    try:
        if action == "warn":
            warn_config = PUNISHMENT_SYSTEM["points_thresholds"].get(5, {})
            warning_message = warn_config.get("message", "You have received a warning. Please be mindful of the server rules.")
            
            try:
                await member.send(f"You have been warned. Reason: {reason}\n{warning_message}")
            except discord.Forbidden: 
                logger.warning(f"Could not DM warning to {member.display_name} ({member.id}).")
            await log_action("warn", member, reason)
        elif action == "mute":
            if duration:
                await member.timeout(duration, reason=reason)
                try:
                    await member.send(f"You have been muted for {duration}. Reason: {reason}")
                except discord.Forbidden:
                    logger.warning(f"Could not DM mute notification to {member.display_name} ({member.id}).")
                await log_action("mute", member, reason)
            else:
                logger.warning(f"Attempted to mute {member.display_name} without duration. Skipping.")
        elif action == "kick":
            try:
                await member.send(f"You are being kicked from the server. Reason: {reason}")
            except discord.Forbidden:
                 logger.warning(f"Could not DM kick notification to {member.display_name} ({member.id}).")
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
                try:
                    await member.send(f"You have been temporarily banned until {unban_time.strftime('%Y-%m-%d %H:%M:%S UTC')}. Reason: {reason}")
                except discord.Forbidden:
                    logger.warning(f"Could not DM temp_ban notification to {member.display_name} ({member.id}).")
                await member.ban(reason=reason, delete_message_days=0)
                await log_action("temp_ban", member, reason)
            else:
                logger.warning(f"Attempted to temp_ban {member.display_name} without duration. Skipping.")
        elif action == "ban":
            try:
                await member.send(f"You have been permanently banned from the server. Reason: {reason}")
            except discord.Forbidden:
                 logger.warning(f"Could not DM ban notification to {member.display_name} ({member.id}).")
            await member.ban(reason=reason, delete_message_days=0)
            await log_action("ban", member, reason)
    except discord.Forbidden:
        logger.error(f"Missing permissions to {action} {member.display_name} in {member.guild.name}. "
                     f"Please check bot role hierarchy and permissions.")
    except discord.HTTPException as e:
        logger.error(f"Discord API error while applying {action} to {member.display_name}: {e.status} - {e.text}")
    except Exception as e:
        logger.error(f"Unexpected error applying {action} to {member.display_name}: {e}", exc_info=True)


async def log_violation(member: discord.Member, violation_type: str, message: discord.Message):
    points_config = PUNISHMENT_SYSTEM["violations"].get(violation_type)
    if not points_config:
        logger.warning(f"Unknown violation type '{violation_type}' encountered for user {member.id}. No points assigned.")
        return
    
    points = points_config["points"]
    message_summary = message.content[:75] + '...' if len(message.content) > 75 else message.content
    reason_details = f"{violation_type.replace('_', ' ').title()}"
    if message.content:
        reason_details += f" (message: '{message_summary}')"
    else:
        reason_details += f" (message ID: {message.id})"


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

    for threshold_points in sorted(PUNISHMENT_SYSTEM["points_thresholds"].keys(), reverse=True):
        if total_points >= threshold_points:

            punishment_config = PUNISHMENT_SYSTEM["points_thresholds"][threshold_points]
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
            elif "duration_years" in punishment_config:
                duration = timedelta(years=punishment_config["duration_years"])
            
            logger.info(f"Applying punishment '{action}' to {member.display_name} due to reaching {total_points} points (threshold: {threshold_points}).")
            await apply_punishment(member, action, auto_punishment_reason, duration)
            break 


def get_domain_from_url(url: str) -> str | None:
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
    """Check if the extracted domain is in the guild's permitted list or is a subdomain."""
    if not domain:
        return False
    for perm_domain_raw in guild_permitted_domains:
        perm_domain_cleaned = get_domain_from_url(perm_domain_raw)
        if perm_domain_cleaned:
            if domain == perm_domain_cleaned or domain.endswith('.' + perm_domain_cleaned):
                return True
    return False


async def handle_health_check(request):
    return web.Response(text="Bot is awake and healthy!", status=200, headers={"Content-Type": "text/plain"})

async def start_http_server():
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
            logger.info("No infraction records older than 90 days to delete.")

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
                        await log_action("unban", user_obj, f"Temporary ban expired (original reason: {ban_reason})", guild=guild)
                    except discord.NotFound:
                        await log_action("unban_id", discord.Object(id=user_id) , f"User ID {user_id} unbanned. Temp ban expired (original reason: {ban_reason})", guild=guild)

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


@tasks.loop(hours=12)
async def cleanup_message_tracking():
    now = datetime.utcnow()
    cleaned_users_timestamps = 0
    cleaned_guilds_timestamps = 0
    for guild_id in list(user_message_timestamps.keys()):
        for user_id in list(user_message_timestamps[guild_id].keys()):
            user_message_timestamps[guild_id][user_id] = deque(
                t for t in user_message_timestamps[guild_id][user_id]
                if (now - t).total_seconds() < (SPAM_WINDOW + 60) 
            )
            if not user_message_timestamps[guild_id][user_id]:
                del user_message_timestamps[guild_id][user_id]
                cleaned_users_timestamps +=1
        if not user_message_timestamps[guild_id]:
            del user_message_timestamps[guild_id]
            cleaned_guilds_timestamps +=1
    
    if cleaned_users_timestamps > 0 or cleaned_guilds_timestamps > 0:
        logger.info(f"Cleaned up message timestamps for {cleaned_users_timestamps} users across {cleaned_guilds_timestamps} guilds.")

    cleaned_users_history = 0
    cleaned_guilds_history = 0
    long_ago = now - timedelta(days=7) 
    for guild_id in list(user_message_history.keys()):
        for user_id in list(user_message_history[guild_id].keys()):
            if user_message_history[guild_id][user_id]:
                if user_message_history[guild_id][user_id][-1][0] < long_ago:
                    del user_message_history[guild_id][user_id] 
                    cleaned_users_history+=1
            elif not user_message_history[guild_id][user_id]: 
                 del user_message_history[guild_id][user_id]
                 cleaned_users_history+=1

        if not user_message_history[guild_id]: 
            del user_message_history[guild_id]
            cleaned_guilds_history+=1
            
    if cleaned_users_history > 0 or cleaned_guilds_history > 0:
        logger.info(f"Cleaned up message history for {cleaned_users_history} users across {cleaned_guilds_history} guilds.")
    if not (cleaned_users_timestamps or cleaned_guilds_timestamps or cleaned_users_history or cleaned_guilds_history):
        logger.info("Message tracking cleanup: No old data found to clean.")


@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    logger.info(f"Discord.py version: {discord.__version__}")
    logger.info("------")

    global http_session
    http_session = ClientSession()

    global db_conn
    try:
        db_conn = await aiosqlite.connect('infractions.db')
        await init_db() 
        logger.info("✅ Database initialized and connection established.")
    except Exception as e:
        logger.critical(f"❌ CRITICAL: Failed to connect to or initialize database: {e}", exc_info=True)
        exit(1) # I see no point in keeping bot online if it can't even access it's own databases

    global LANGUAGE_MODEL
    if FASTTEXT_MODEL_PATH:
        try:
            LANGUAGE_MODEL = fasttext.load_model(FASTTEXT_MODEL_PATH)
            logger.info(f"✅ Successfully loaded FastText model from {FASTTEXT_MODEL_PATH}")
        except ValueError as ve: 
            logger.error(f"❌ Failed to load FastText model: {ve}. Is '{FASTTEXT_MODEL_PATH}' a valid model file?", exc_info=True)
        except Exception as e:
            logger.error(f"❌ Failed to load FastText model from {FASTTEXT_MODEL_PATH}: {e}. Language detection will be impaired.", exc_info=True)
    else:
        logger.warning("FASTTEXT_MODEL_PATH not set. Language detection will be disabled.")


    if db_conn: 
        decay_points.start()
        cleanup_message_tracking.start()
    
    await start_http_server() 

    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} application command(s).")
    except Exception as e:
        logger.error(f"Failed to sync application commands: {e}", exc_info=True)

@bot.event
async def on_error(event_name, *args, **kwargs):
    logger.error(f"Unhandled error in event '{event_name}': Args: {args}, Kwargs: {kwargs}", exc_info=True)

@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.CommandNotFound):
        return 
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Oops! You missed an argument: `{error.param.name}`. Check `>>help {ctx.command.qualified_name}` for usage.")
    elif isinstance(error, commands.BadArgument):
        await ctx.send(f"Hmm, that's not a valid argument. {error}")
    elif isinstance(error, commands.NoPrivateMessage):
        await ctx.send("This command can only be used in a server.")
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send(f"You don't have the required permissions to use this command: `{', '.join(error.missing_permissions)}`")
    elif isinstance(error, commands.BotMissingPermissions):
        await ctx.send(f"I'm missing permissions to do that: `{', '.join(error.missing_permissions)}`. Please grant them to me!")
        logger.warning(f"Bot missing permissions in guild {ctx.guild.id}, channel {ctx.channel.id}: {error.missing_permissions} for command {ctx.command.name}")
    elif isinstance(error, commands.CommandInvokeError):
        logger.error(f"Error invoking command '{ctx.command.qualified_name}': {error.original}", exc_info=error.original)
        await ctx.send("An internal error occurred while running this command. The developers have been notified.")
    else:
        logger.error(f"Unhandled command error for '{ctx.command.qualified_name if ctx.command else 'UnknownCmd'}': {error}", exc_info=True)
        await ctx.send("An unexpected error occurred. Please try again later.")

@bot.event
async def on_guild_join(guild: discord.Guild):
    logger.info(f"Joined new guild: {guild.name} (ID: {guild.id}, Members: {guild.member_count})")

class Moderation(commands.Cog):
    def __init__(self, bot_instance: commands.Bot):
        self.bot = bot_instance
        self.sightengine_nsfw_threshold = 0.6  
        self.sightengine_gore_threshold = 0.8   
        self.sightengine_violence_threshold = 0.7 
        self.sightengine_text_profanity_threshold = 0.9  
        self.sightengine_minor_offensive_threshold = 0.95 

        self.discrimination_words = discrimination_words
        self.discrimination_patterns = discrimination_patterns
        self.nsfw_words = nsfw_words
        self.nsfw_patterns = nsfw_patterns

        self.cleanup_message_history.start() 

    def cog_unload(self):
        self.cleanup_message_history.cancel()
        logger.info("Message history cleanup task cancelled.")

    @tasks.loop(hours=1) 
    async def cleanup_message_history(self):
        """Periodically cleans up old message entries from the database."""
        if db_conn:
            try:
                retention_period_seconds = SPAM_WINDOW * 2 
                threshold_timestamp = (datetime.utcnow() - timedelta(seconds=retention_period_seconds)).timestamp()

                await db_conn.execute("DELETE FROM message_history WHERE timestamp < ?", (threshold_timestamp,))
                await db_conn.commit()
                logger.info(f"Cleaned up message_history table: deleted entries older than {retention_period_seconds} seconds.")
            except Exception as e:
                logger.error(f"Error during message history cleanup task: {e}", exc_info=True)

    @cleanup_message_history.before_loop
    async def before_cleanup_message_history(self):
        """Wait until the bot is ready before starting the cleanup loop."""
        await self.bot.wait_until_ready()
        logger.info("Starting message history cleanup task.")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild or not db_conn:
            return

        if isinstance(bot.command_prefix, str) and message.content.startswith(bot.command_prefix):
             pass 

        violations = set()
        guild_id = message.guild.id
        user_id = message.author.id
        content_raw = message.content 
        content_lower = clean_message_content(content_raw) 
        now = datetime.utcnow()

        try:
            await db_conn.execute(
                "INSERT INTO message_history (user_id, guild_id, timestamp, message_content) VALUES (?, ?, ?, ?)",
                (user_id, guild_id, current_timestamp, content_raw)
            )
            await db_conn.commit()
        except Exception as e:
            logger.error(f"Error inserting message into history DB: {e}", exc_info=True)

        time_threshold_spam_window = (now - timedelta(seconds=SPAM_WINDOW)).timestamp()
        try:
            cursor = await db_conn.execute(
                "SELECT COUNT(*) FROM message_history WHERE user_id = ? AND guild_id = ? AND timestamp >= ?",
                (user_id, guild_id, time_threshold_spam_window)
            )
            count_freq = (await cursor.fetchone())[0]
            if count_freq > SPAM_LIMIT:
                violations.add("spam")
                logger.debug(f"Spam (frequency, DB) by {message.author.name}: {count_freq} msgs in {SPAM_WINDOW}s")
        except Exception as e:
            logger.error(f"Error querying message history for frequency spam: {e}", exc_info=True)

        try:
            cursor = await db_conn.execute(
                "SELECT COUNT(*) FROM message_history WHERE user_id = ? AND guild_id = ? AND message_content = ? AND timestamp >= ?",
                (user_id, guild_id, content_raw, time_threshold_spam_window)
            )
            count_repetition = (await cursor.fetchone())[0]
            if count_repetition > 1:
                violations.add("spam")
                logger.debug(f"Spam (repetition, DB) by {message.author.name}: '{content_raw[:50]}...'")
        except Exception as e:
            logger.error(f"Error querying message history for repetition spam: {e}", exc_info=True)
            
        channel_specific_db_config = await get_guild_config(guild_id, f"channel_config_{message.channel.id}", {})
        # guild_wide_lang = await get_guild_config(guild_id, "allowed_languages", None) 
     
        channel_cfg = DEFAULT_CHANNEL_CONFIGS.get(message.channel.id, {}).copy() 
        if isinstance(channel_specific_db_config, dict): 
            channel_cfg.update(channel_specific_db_config)
        else: 
            logger.warning(f"channel_config_{message.channel.id} for guild {guild_id} was not a dict: {channel_specific_db_config}")

        guild_permitted_domains = await get_guild_config(guild_id, "permitted_domains", list(PERMITTED_DOMAINS))


        user_timestamps = user_message_timestamps[guild_id][user_id]
        user_timestamps.append(now)
        while user_timestamps and (now - user_timestamps[0]).total_seconds() >= SPAM_WINDOW:
            user_timestamps.popleft()
        if len(user_timestamps) > SPAM_LIMIT:
            violations.add("spam")
            logger.debug(f"Spam (frequency) by {message.author.name}: {len(user_timestamps)} msgs in {SPAM_WINDOW}s")

        user_hist = user_message_history[guild_id][user_id]
        if content_raw and content_raw in [msg_content for _, msg_content in list(user_hist)]:
            violations.add("spam")
            logger.debug(f"Spam (repetition) by {message.author.name}: '{content_raw[:50]}...'")
        user_hist.append((now, content_raw))


        if len(message.mentions) > MENTION_LIMIT:
            violations.add("excessive_mentions")
        if len(message.attachments) > MAX_ATTACHMENTS:
            violations.add("excessive_attachments")
        if len(content_raw) > MAX_MESSAGE_LENGTH: 
            violations.add("long_message")

        allowed_languages = channel_cfg.get("language") 
        if allowed_languages and content_lower: 
            skip_lang_check_reason = None
            if URL_PATTERN.fullmatch(content_lower):
                skip_lang_check_reason = "message is a URL"
            elif not HAS_ALPHANUMERIC_PATTERN.search(content_lower): 
                skip_lang_check_reason = "message has no alphanumeric characters"
            elif len(content_lower) <= MIN_MSG_LEN_FOR_LANG_CHECK: 
                skip_lang_check_reason = f"message is too short ({len(content_lower)} chars)"
            
            if skip_lang_check_reason:
                logger.debug(f"Skipping language check for '{content_raw[:50]}...': {skip_lang_check_reason}.")
            else:
                detected_lang_code, confidence = await detect_language_ai(content_raw) # Use raw for detection
                logger.debug(f"Language detection for '{content_raw[:50]}...': Lang={detected_lang_code}, Conf={confidence:.2f}")

                if detected_lang_code not in allowed_languages:
                    if detected_lang_code == "und":
                        logger.info(f"Language undetermined for '{content_raw[:50]}...'. Not flagging as foreign language.")
                    elif content_lower in COMMON_SAFE_FOREIGN_WORDS:
                        logger.info(f"Message '{content_lower}' is a common safe word, detected as {detected_lang_code}. Not flagging as foreign language unless confidence is very high.")
                    elif len(content_lower) < SHORT_MSG_THRESHOLD and confidence < MIN_CONFIDENCE_SHORT_MSG:
                        logger.info(f"Low confidence ({confidence:.2f} < {MIN_CONFIDENCE_SHORT_MSG}) for short message '{content_raw[:50]}...' (lang: {detected_lang_code}). Not flagging as foreign.")
                    elif confidence < MIN_CONFIDENCE_FOR_FLAGGING:
                         logger.info(f"Low confidence ({confidence:.2f} < {MIN_CONFIDENCE_FOR_FLAGGING}) for message '{content_raw[:50]}...' (lang: {detected_lang_code}). Not flagging as foreign.")
                    else:
                        violations.add("foreign_language")
                        logger.debug(f"Foreign language violation by {message.author.name} in {message.channel.name}: '{detected_lang_code}' (Conf: {confidence:.2f}) not in {allowed_languages}. Message: '{content_raw[:50]}...'")
        
        link_channel_id_config = await get_guild_config(guild_id, "link_channel_id") 
        link_channel_id = int(link_channel_id_config) if link_channel_id_config and str(link_channel_id_config).isdigit() else None


        if FORBIDDEN_TEXT_PATTERN.search(content_lower):
            violations.add("advertising")
            logger.debug(f"Advertising (forbidden pattern) by {message.author.name}: '{content_raw[:50]}...'")

        urls_found = URL_PATTERN.findall(content_raw)
        if urls_found:
            is_link_channel = (link_channel_id == message.channel.id)
            if not is_link_channel:
                for url_match in urls_found:
                    url_str = url_match[0] if isinstance(url_match, tuple) else url_match
                    domain = get_domain_from_url(url_str)
                    if domain and not is_permitted_domain(domain, guild_permitted_domains):
                        violations.add("advertising")
                        logger.debug(f"Advertising (forbidden domain: {domain}) by {message.author.name} in non-link channel. URL: {url_str}")
                        break
                    elif not domain:
                        logger.debug(f"Could not extract domain from URL '{url_str}' for advertising check.")


        words_in_message = set(re.findall(r'\b\w+\b', content_lower))
        if any(word in discrimination_words for word in words_in_message) or \
           any(pattern.search(content_lower) for pattern in discrimination_patterns):
            violations.add("discrimination")
            logger.debug(f"Discrimination (local list) by {message.author.name}: '{content_raw[:50]}...'")


        if any(word in nsfw_words for word in words_in_message) or \
           any(pattern.search(content_lower) for pattern in nsfw_patterns):
            violations.add("nsfw")
            logger.debug(f"NSFW (local list) by {message.author.name}: '{content_raw[:50]}...'")

        allowed_topics = channel_cfg.get("topics") 
        if allowed_topics: 
            if not any(topic.lower() in content_lower for topic in allowed_topics):
                violations.add("off_topic")
                logger.debug(f"Off-topic by {message.author.name}: '{content_raw[:50]}...' (Allowed: {allowed_topics})")
        else: 
            general_sensitive_terms = [
                "politics", "religion", 
                "democrat", "republican", "liberal", "conservative" 
            ]
            # if any(term in content_lower for term in general_sensitive_terms):
            # violations.add("politics_discussion") # Or a more generic "sensitive_topic"
            # logger.debug(f"Sensitive topic (politics/religion) by {message.author.name}: '{content_raw[:50]}...'")
            pass 


        if message.attachments and not ("nsfw" in violations):
            for attachment in message.attachments:
                content_type = attachment.content_type
                if content_type and (content_type.startswith('image/') or content_type.startswith('video/')):
                    if SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET:
                        logger.info(f"Checking attachment '{attachment.filename}' ({content_type}) with Sightengine...")
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

        run_openai_check = OPENAI_API_KEY and content_raw.strip() and not any(v in violations for v in ["nsfw", "discrimination"])

        if run_openai_check:
            moderation_result = await check_openai_moderation(content_raw)
            if moderation_result.get("flagged", False):
                violations.add("openai_moderation") 
                logger.debug(f"OpenAI flagged message by {message.author.name}: '{content_raw[:50]}...'. Categories: {moderation_result.get('categories')}")
                categories = moderation_result.get("categories", {})
                if categories.get("hate", False) or categories.get("hate/threatening", False):
                    violations.add("discrimination")
                if categories.get("sexual", False) or categories.get("sexual/minors", False):
                    violations.add("nsfw")
                if categories.get("self-harm", False):
                    violations.add("nsfw") 

        if violations:
            logger.info(f"Message from {message.author.name} ({user_id}) in #{message.channel.name} ({message.channel.id}) "
                        f"flagged with violations: {', '.join(sorted(list(violations)))}. Content: '{content_raw[:100]}...'")
            try:
                await message.delete()
                await log_action("message_deleted", message.author, f"Violations: {', '.join(sorted(list(violations)))}", guild=message.guild)

            except discord.Forbidden:
                logger.error(f"Missing permissions to delete message by {message.author.name} in #{message.channel.name}.")
            except discord.NotFound: 
                logger.warning(f"Message {message.id} by {message.author.name} was already deleted.")
            except discord.HTTPException as e:
                logger.error(f"HTTP error deleting message {message.id}: {e.status} - {e.text}")
            except Exception as e: 
                logger.error(f"Unexpected error during message deletion of {message.id}: {e}", exc_info=True)


            for violation in sorted(list(violations)): 
                await log_violation(message.author, violation, message) 

            violation_titles = sorted([v.replace('_', ' ').title() for v in violations])
            warning_msg_text = (f"{message.author.mention}, your message was removed due to: "
                                f"**{', '.join(violation_titles)}**. Please review server rules.")
            try:
                await message.channel.send(warning_msg_text, delete_after=20) 
            except discord.Forbidden:
                logger.error(f"Missing permissions to send warning message in #{message.channel.name}.")
            except discord.HTTPException as e:
                 logger.error(f"HTTP error sending warning message: {e.status} - {e.text}")

        else: 
            await self.bot.process_commands(message)

    async def check_media_nsfw_sightengine(self, media_url: str) -> bool:
        """
        Checks media content for NSFW using the Sightengine API.

        Args:
            media_url: The URL of the media to check.

        Returns:
            True if NSFW content is detected, False otherwise.
        """
        if not SIGHTENGINE_API_USER or not SIGHTENGINE_API_SECRET:
            logger.error("Sightengine API credentials not set. Skipping NSFW check for %s", media_url)
            return False

        if http_session is None or http_session.closed:
            logger.warning("http_session was not initialized or was closed. Cannot perform Sightengine check for %s.", media_url)
            return False

        api_url = "https://api.sightengine.com/1.0/check.json"
        params = {
            "url": media_url,
            "models": "nudity-2.0,offensive,gore,violence,text",
            "api_user": SIGHTENGINE_API_USER,
            "api_secret": SIGHTENGINE_API_SECRET,
        }

        def log_api_error(message: str, url: str, response_detail: str, level=logging.ERROR):
            logger.log(level, f"Sightengine API {message} for {url}. Details: {response_detail}")

        try:
            async with http_session.get(api_url, params=params, timeout=15) as response:
                response_text = await response.text()  

                if response.status == 200:
                    try:
                        data = json.loads(response_text)
                    except json.JSONDecodeError:
                        log_api_error("response was not valid JSON", media_url, response_text)
                        return False

                    if data.get("status") == "success":
                        nudity_scores = data.get("nudity", {})
                        sexual_activity_score = nudity_scores.get("sexual_activity", 0.0)
                        suggestive_score = nudity_scores.get("suggestive", 0.0)
                        gore_score = data.get("gore", {}).get("prob", 0.0) 
                        violence_score = data.get("violence", {}).get("prob", 0.0) 
                        text_data = data.get("text", {})
                        text_profanity_score = text_data.get("profanity", 0.0)

                        offensive_data = data.get("offensive", {})
                        offensive_prob = 0.0
                        if offensive_data:
                            offensive_prob = max(
                                offensive_data.get("nazi", 0.0),
                                offensive_data.get("asian_swastika", 0.0),
                                offensive_data.get("confederate", 0.0),
                                offensive_data.get("supremacist", 0.0),
                                offensive_data.get("terrorist", 0.0),
                            )
                        
                        logger.debug(f"Sightengine response for {media_url}: "
                                     f"Nudity SA: {sexual_activity_score:.2f}, Suggestive: {suggestive_score:.2f}, " # SA stands for Sexual Activity 
                                     f"Gore: {gore_score:.2f}, Violence: {violence_score:.2f}, "
                                     f"Text Profanity: {text_profanity_score:.2f}, "
                                     f"Severe Offensive: {severe_offensive_prob:.2f}, Middle Finger: {middle_finger_score:.2f}")
                        
                        is_nsfw = False
                        flagged_reasons = []

                        if sexual_activity_score > self.sightengine_nsfw_threshold:
                            is_nsfw = True
                            flagged_reasons.append(f"Nudity SA ({sexual_activity_score:.2f})") # SA stands for Sexual Activity 
                        if suggestive_score > (self.sightengine_nsfw_threshold + 0.2): 
                            is_nsfw = True
                            flagged_reasons.append(f"Suggestive Nudity ({suggestive_score:.2f})")
                        if gore_score > self.sightengine_gore_threshold:
                            is_nsfw = True
                            flagged_reasons.append(f"Gore ({gore_score:.2f})")
                        if violence_score > self.sightengine_violence_threshold:
                            is_nsfw = True
                            flagged_reasons.append(f"Violence ({violence_score:.2f})")
                        if text_profanity_score > self.sightengine_text_profanity_threshold:
                            is_nsfw = True
                            flagged_reasons.append(f"Image Text Profanity ({text_profanity_score:.2f})")
                        if offensive_prob > 0.85: 
                            is_nsfw = True
                            flagged_reasons.append(f"Severe Offensive ({severe_offensive_prob:.2f})")
                        if is_nsfw:
                            logger.info(f"NSFW media detected by Sightengine: {media_url} (Reasons: {', '.join(flagged_reasons)})")
                            return True
                        return False
                    else:
                        error_msg = data.get("error", {}).get("message", "Unknown error in Sightengine response data.")
                        if data.get("error", {}).get("code") == 22 and data.get("error", {}).get("type") == "media_error":
                            log_api_error("API rejected GIF (too many frames - cannot scan)", media_url, error_msg, level=logging.WARNING)
                            return False 
                        else:
                            log_api_error("API error", media_url, f"Status in JSON: {data.get('status')}, Message: {error_msg}, Full JSON: {json.dumps(data)}")
                            return False
                elif response.status == 429:
                    log_api_error("rate limit hit", media_url, f"Status: {response.status}, Response: {response_text}", level=logging.WARNING)
                    return False
                else:
                    log_api_error("API request failed", media_url, f"Status: {response.status}, Response: {response_text}")
                    return False
        except client_exceptions.ClientConnectorError as e:
            log_api_error("network connection error", media_url, f"Error: {e}")
            return False
        except asyncio.TimeoutError:
            log_api_error("request timed out", media_url, "Timeout after 15 seconds", level=logging.WARNING)
            return False
        except Exception as e:
            log_api_error("unexpected error during API call", media_url, f"Error: {e}", level=logging.CRITICAL)
            return False

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def set_log_channel(self, ctx: commands.Context, channel: discord.TextChannel = None):
        """Set or clear the channel where moderation logs are sent. No argument to clear."""
        if channel:
            await set_guild_config(ctx.guild.id, "log_channel_id", channel.id)
            await ctx.send(f"Moderation logs will now be sent to {channel.mention}.")
            await log_action("config_change", ctx.author, f"Set log channel to #{channel.name}", guild=ctx.guild)
        else:
            await set_guild_config(ctx.guild.id, "log_channel_id", None) 
            await ctx.send("Moderation log channel has been cleared. Logs will only go to console if no default is set.")
            await log_action("config_change", ctx.author, "Cleared log channel", guild=ctx.guild)


    @commands.command()
    @commands.has_permissions(administrator=True)
    async def set_link_channel(self, ctx: commands.Context, channel: discord.TextChannel = None):
        """Set or clear the channel where links are allowed without being flagged as advertising. No arg to clear."""
        if channel:
            await set_guild_config(ctx.guild.id, "link_channel_id", channel.id)
            await ctx.send(f"Link posting channel set to {channel.mention}. Links outside may be restricted.")
            await log_action("config_change", ctx.author, f"Set link channel to #{channel.name}", guild=ctx.guild)
        else:
            await set_guild_config(ctx.guild.id, "link_channel_id", None)
            await ctx.send("Link posting channel restriction has been cleared.")
            await log_action("config_change", ctx.author, "Cleared link channel", guild=ctx.guild)


    @commands.command()
    @commands.has_permissions(administrator=True)
    async def add_permitted_domain(self, ctx: commands.Context, domain: str):
        """Add a domain to the list of permitted domains for links."""
        current_guild_domains = await get_guild_config(ctx.guild.id, "permitted_domains", list(PERMITTED_DOMAINS))
        if not isinstance(current_guild_domains, list):
            logger.warning(f"Permitted domains for guild {ctx.guild.id} was not a list: {current_guild_domains}. Resetting from default.")
            current_guild_domains = list(PERMITTED_DOMAINS)

        cleaned_domain = get_domain_from_url(domain) 
        if cleaned_domain and cleaned_domain not in current_guild_domains:
            current_guild_domains.append(cleaned_domain)
            await set_guild_config(ctx.guild.id, "permitted_domains", current_guild_domains)
            await ctx.send(f"Added `{cleaned_domain}` to this guild's permitted domains.")
            await log_action("config_change", ctx.author, f"Added permitted domain: {cleaned_domain}", guild=ctx.guild)
        elif cleaned_domain in current_guild_domains:
            await ctx.send(f"`{cleaned_domain}` is already in the permitted list.")
        else:
            await ctx.send(f"Invalid domain format: `{domain}`.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def remove_permitted_domain(self, ctx: commands.Context, domain: str):
        """Remove a domain from this guild's list of permitted domains."""
        current_guild_domains = await get_guild_config(ctx.guild.id, "permitted_domains", list(PERMITTED_DOMAINS))
        if not isinstance(current_guild_domains, list):
             current_guild_domains = list(PERMITTED_DOMAINS) 

        cleaned_domain = get_domain_from_url(domain)
        if cleaned_domain and cleaned_domain in current_guild_domains:
            current_guild_domains.remove(cleaned_domain)
            await set_guild_config(ctx.guild.id, "permitted_domains", current_guild_domains)
            await ctx.send(f"Removed `{cleaned_domain}` from this guild's permitted domains.")
            await log_action("config_change", ctx.author, f"Removed permitted domain: {cleaned_domain}", guild=ctx.guild)
        elif cleaned_domain:
            await ctx.send(f"`{cleaned_domain}` not found in this guild's custom permitted list.")
        else:
            await ctx.send(f"Invalid domain format: `{domain}`.")
            
    @commands.command(name="list_permitted_domains")
    @commands.has_permissions(manage_messages=True)
    async def list_permitted_domains(self, ctx: commands.Context):
        """Lists all currently permitted domains for this guild."""
        guild_domains = await get_guild_config(ctx.guild.id, "permitted_domains", list(PERMITTED_DOMAINS))
        if not guild_domains:
            await ctx.send("No domains are currently specifically permitted for this guild (global defaults may apply if not overridden).")
            return
        
        embed = discord.Embed(title=f"Permitted Domains for {ctx.guild.name}", color=discord.Color.blue())
        domain_list_str = "\n".join([f"- `{d}`" for d in sorted(guild_domains)])
        if len(domain_list_str) > 1900 : 
             domain_list_str = domain_list_str[:1900] + "\n... (list too long to display fully)"
        embed.description = domain_list_str
        await ctx.send(embed=embed)


    @commands.command()
    @commands.has_permissions(manage_messages=True)
    async def infractions(self, ctx: commands.Context, member: discord.Member):
        """View a user's recent infractions and active points."""
        if not db_conn:
            await ctx.send("Database is not connected, cannot retrieve infractions.")
            return

        async with db_conn.cursor() as cursor:
            cutoff_date_active = (datetime.utcnow() - timedelta(days=30)).isoformat()
            await cursor.execute(
                'SELECT SUM(points) FROM infractions WHERE user_id = ? AND guild_id = ? AND timestamp >= ?',
                (member.id, ctx.guild.id, cutoff_date_active)
            )
            total_points_data = await cursor.fetchone()
            total_active_points = total_points_data[0] if total_points_data and total_points_data[0] is not None else 0


            await cursor.execute('''
                SELECT points, timestamp, violation_type, message_id FROM infractions
                WHERE user_id = ? AND guild_id = ?
                ORDER BY timestamp DESC
                LIMIT 10
            ''', (member.id, ctx.guild.id))
            records = await cursor.fetchall()

        embed = discord.Embed(
            title=f"Infraction Report for {member.display_name}",
            description=f"User ID: {member.id}",
            color=discord.Color.orange() if total_active_points > 0 else discord.Color.green()
        )
        embed.set_thumbnail(url=member.display_avatar.url)
        embed.add_field(name="Total Active Points (Last 30 Days)", value=f"**{total_active_points}**", inline=False)

        if records:
            history_str = ""
            for points, ts_str, vio_type, msg_id in records:
                ts_dt = datetime.fromisoformat(ts_str)
                is_active = ts_dt >= datetime.fromisoformat(cutoff_date_active)
                active_marker = " (Active)" if is_active else ""
                entry = (f"- **{points} pts** ({vio_type.replace('_', ' ').title()}){active_marker} "
                         f"on {ts_dt.strftime('%Y-%m-%d %H:%M')} UTC")
                if msg_id: 
                    entry += f" (Msg ID: {msg_id})"
                history_str += entry + "\n"
            
            if len(history_str) > 1020: history_str = history_str[:1020] + "..." 
            embed.add_field(name="Recent Infraction History (Max 10)", value=history_str if history_str else "None found.", inline=False)
        else:
            embed.add_field(name="Recent Infraction History", value="No infractions recorded for this user.", inline=False)

        embed.set_footer(text=f"Report generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        await ctx.send(embed=embed)

    @commands.command()
    @commands.has_permissions(manage_guild=True) 
    async def clear_infractions(self, ctx: commands.Context, member: discord.Member, specific_violation_id: int = None):
        """Manually clear all (or a specific) infraction points for a user in this guild."""
        if not db_conn:
            await ctx.send("Database is not connected.")
            return
        
        async with db_conn.cursor() as cursor:
            if specific_violation_id:
                await cursor.execute('DELETE FROM infractions WHERE user_id = ? AND guild_id = ? AND id = ?', 
                                     (member.id, ctx.guild.id, specific_violation_id))
                action_taken = f"infraction ID {specific_violation_id}"
            else:
                await cursor.execute('DELETE FROM infractions WHERE user_id = ? AND guild_id = ?', (member.id, ctx.guild.id))
                action_taken = "all infractions"

            if cursor.rowcount > 0:
                await db_conn.commit()
                await ctx.send(f"Cleared {action_taken} for {member.display_name}.")
                await log_action("infractions_cleared", ctx.author, f"Cleared {action_taken} for {member.mention} ({member.id})", guild=ctx.guild)
            else:
                await ctx.send(f"No matching infractions found for {member.display_name} to clear ({action_taken}).")


    @commands.command()
    @commands.has_permissions(kick_members=True) 
    async def manual_warn(self, ctx: commands.Context, member: discord.Member, *, reason: str = "No reason provided by moderator."):
        """Manually warn a user."""
        full_reason = f"Manual warn by {ctx.author.name} ({ctx.author.id}): {reason}"
        await apply_punishment(member, "warn", full_reason)
        await ctx.send(f"Warned {member.display_name}. Reason: {reason}")

    @commands.command()
    @commands.has_permissions(kick_members=True)
    async def manual_mute(self, ctx: commands.Context, member: discord.Member, duration_hours: float, *, reason: str = "No reason provided by moderator."):
        """Manually mute a user for a specified number of hours."""
        if duration_hours <= 0:
            await ctx.send("Mute duration must be positive.")
            return
        duration = timedelta(hours=duration_hours)
        full_reason = f"Manual mute by {ctx.author.name} ({ctx.author.id}): {reason}"
        await apply_punishment(member, "mute", full_reason, duration=duration)
        await ctx.send(f"Muted {member.display_name} for {duration_hours} hours. Reason: {reason}")

    @commands.command()
    @commands.has_permissions(kick_members=True)
    async def manual_kick(self, ctx: commands.Context, member: discord.Member, *, reason: str = "No reason provided by moderator."):
        """Manually kick a user."""
        full_reason = f"Manual kick by {ctx.author.name} ({ctx.author.id}): {reason}"
        await apply_punishment(member, "kick", full_reason)
        await ctx.send(f"Kicked {member.display_name}. Reason: {reason}") 

    @commands.command()
    @commands.has_permissions(ban_members=True)
    async def manual_ban(self, ctx: commands.Context, member_or_id: discord.User | int , duration_days: float = None, *, reason: str = "No reason provided by moderator."):
        """Manually ban a user (temporarily or permanently). Provide User ID or @User."""
        
        target_user: discord.User | None = None
        if isinstance(member_or_id, discord.User): 
            target_user = member_or_id
        elif isinstance(member_or_id, int): 
            try:
                target_user = await bot.fetch_user(member_or_id)
            except discord.NotFound:
                await ctx.send(f"User with ID `{member_or_id}` not found.")
                return
            except discord.HTTPException as e:
                 await ctx.send(f"Failed to fetch user ID `{member_or_id}`: {e}")
                 return
        else: 
            await ctx.send("Invalid user provided. Please use @User or UserID.")
            return
        
        if not target_user: 
            await ctx.send("Could not identify target user.")
            return

        member = ctx.guild.get_member(target_user.id) 

        full_reason = f"Manual ban by {ctx.author.name} ({ctx.author.id}): {reason}"
        action_type = "ban"
        duration = None

        if duration_days is not None:
            if duration_days <= 0:
                await ctx.send("Ban duration must be positive.")
                return
            duration = timedelta(days=duration_days)
            action_type = "temp_ban"
            if member: 
                await apply_punishment(member, action_type, full_reason, duration=duration)
                await ctx.send(f"Temporarily banned {target_user.name}#{target_user.discriminator} for {duration_days} days. Reason: {reason}")
            else: 
                unban_time = datetime.utcnow() + duration
                async with db_conn.cursor() as cursor: 
                    await cursor.execute(
                        'INSERT OR REPLACE INTO temp_bans (user_id, guild_id, unban_time, ban_reason) VALUES (?, ?, ?, ?)',
                        (target_user.id, ctx.guild.id, unban_time.isoformat(), full_reason)
                    )
                    await db_conn.commit()
                await ctx.guild.ban(target_user, reason=full_reason, delete_message_days=0)
                await log_action(action_type, target_user, full_reason, guild=ctx.guild)
                await ctx.send(f"Temporarily banned user ID {target_user.id} for {duration_days} days (not in server or DMs failed). Reason: {reason}")

        else: 
            await ctx.guild.ban(target_user, reason=full_reason, delete_message_days=0) 
            await log_action(action_type, member if member else target_user, full_reason, guild=ctx.guild)
            await ctx.send(f"Permanently banned {target_user.name}#{target_user.discriminator}. Reason: {reason}")


    @commands.command()
    @commands.has_permissions(ban_members=True)
    async def manual_unban(self, ctx: commands.Context, user_id: int, *, reason: str = "No reason provided by moderator."):
        """Manually unban a user by their ID."""
        try:
            user_to_unban = await bot.fetch_user(user_id) 
        except discord.NotFound:
            await ctx.send(f"User with ID `{user_id}` not found by Discord.")
            return
        except discord.HTTPException as e:
            await ctx.send(f"Error fetching user ID `{user_id}`: {e}")
            return

        full_reason = f"Manual unban by {ctx.author.name} ({ctx.author.id}): {reason}"
        try:
            bans = [entry async for entry in ctx.guild.bans(limit=None) if entry.user.id == user_id] 
            if not bans:
                await ctx.send(f"{user_to_unban.name}#{user_to_unban.discriminator} (ID: {user_id}) is not banned.")
                return
            
            await ctx.guild.unban(user_to_unban, reason=full_reason)
            async with db_conn.cursor() as cursor:
                await cursor.execute('DELETE FROM temp_bans WHERE user_id = ? AND guild_id = ?', (user_id, ctx.guild.id))
                await db_conn.commit()
            
            await ctx.send(f"Unbanned {user_to_unban.name}#{user_to_unban.discriminator} (ID: {user_id}). Reason: {reason}")
            await log_action("unban", user_to_unban, full_reason, guild=ctx.guild)
        except discord.NotFound: 
            await ctx.send(f"{user_to_unban.name}#{user_to_unban.discriminator} (ID: {user_id}) was not found in this server's ban list.")
        except discord.Forbidden:
            await ctx.send("I don't have permissions to unban members.")
        except discord.HTTPException as e:
            logger.error(f"Error manually unbanning user {user_id}: {e}", exc_info=True)
            await ctx.send(f"An API error occurred while trying to unban the user: {e.text}")
        except Exception as e:
            logger.error(f"Unexpected error manually unbanning user {user_id}: {e}", exc_info=True)
            await ctx.send("An unexpected error occurred.")

class BotInfo(commands.Cog):
    def __init__(self, bot_instance: commands.Bot):
        self.bot = bot_instance

    @app_commands.command(name="awake", description="Check if the bot is awake and responsive.")
    async def awake_slash(self, interaction: discord.Interaction):
        """Respond to the /awake slash command."""
        await interaction.response.send_message("I am Adroit. I am always awake. Never sleeping.", ephemeral=True)

    @commands.command(name="awake")
    async def awake_prefix(self, ctx: commands.Context):
        """Respond to the >>awake prefix command."""
        await ctx.send("I am Adroit. I am always awake. Never sleeping.", delete_after=10)


    @commands.command(name="classify")
    @commands.cooldown(1, 5, commands.BucketType.user) 
    async def classify_language(self, ctx: commands.Context, *, text_to_classify: str = None):
        """Classify the language of replied message or provided text. Uses FastText."""
        if not LANGUAGE_MODEL:
            await ctx.send("Language model is not loaded, cannot classify.")
            return

        target_text = text_to_classify
        if not target_text:
            if ctx.message.reference and ctx.message.reference.message_id:
                try:
                    referenced_message = await ctx.channel.fetch_message(ctx.message.reference.message_id)
                    target_text = referenced_message.content
                    if not target_text:
                        await ctx.send("The replied message has no text content to classify.")
                        return
                except discord.NotFound:
                    await ctx.send("Could not find the replied message.")
                    return
                except discord.HTTPException:
                    await ctx.send("Failed to fetch the replied message.")
                    return
            else:
                await ctx.send("Please reply to a message or provide text directly to classify its language. Usage: `>>classify [text]` or reply with `>>classify`")
                return
        
        if not target_text.strip():
            await ctx.send("Cannot classify empty text.")
            return

        lang_code, confidence = await detect_language_ai(target_text)
        
        embed = discord.Embed(title="Language Classification", color=discord.Color.blue())
        embed.add_field(name="Text Snippet", value=f"```{discord.utils.escape_markdown(target_text[:200])}...```", inline=False)
        embed.add_field(name="Detected Language", value=f"**{lang_code.upper()}**", inline=True)
        embed.add_field(name="Confidence", value=f"{confidence:.2%}", inline=True)
        
        allowed_langs = await get_guild_config(ctx.guild.id, f"channel_config_{ctx.channel.id}.language") or \
        await get_guild_config(ctx.guild.id, "allowed_languages") or \
        DEFAULT_CHANNEL_CONFIGS.get(ctx.channel.id, {}).get("language")
        if allowed_langs and isinstance(allowed_langs, list):
            status = "Allowed" if lang_code in allowed_langs else "Not Allowed"
            embed.add_field(name="Channel Status", value=f"{status} (Allowed: {', '.join(allowed_langs)})", inline=False)

        await ctx.send(embed=embed)

async def init_db():
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
                timestamp TEXT NOT NULL,     -- ISO format UTC timestamp
                violation_type TEXT,
                message_id INTEGER,          -- ID of the offending message
                channel_id INTEGER           -- ID of the channel where infraction occurred
            )
        ''')
        await cursor.execute('CREATE INDEX IF NOT EXISTS idx_infractions_user_guild_time ON infractions (user_id, guild_id, timestamp)')

        await cursor.execute('''
            CREATE TABLE IF NOT EXISTS temp_bans (
                user_id INTEGER NOT NULL,    -- User ID
                guild_id INTEGER NOT NULL,   -- Guild ID
                unban_time TEXT NOT NULL,    -- ISO format UTC timestamp for when to unban
                ban_reason TEXT,
                PRIMARY KEY (user_id, guild_id) -- User can only have one active temp ban per guild
            )
        ''')
        await cursor.execute('CREATE INDEX IF NOT EXISTS idx_temp_bans_unban_time ON temp_bans (unban_time)')


        await cursor.execute('''
            CREATE TABLE IF NOT EXISTS guild_configs (
                guild_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT,                  -- Store complex values as JSON strings
                PRIMARY KEY (guild_id, key)
            )
        ''')
        await db_conn.commit()
    logger.info("Database schema checked/created successfully.")


async def main():
    global db_conn, http_session, LANGUAGE_MODEL

    http_session = ClientSession()
    logger.info("Aiohttp session initialized.")

    try:
        db_conn = await aiosqlite.connect('moderation_data.db')
        await db_conn.execute("""
            CREATE TABLE IF NOT EXISTS guild_configs (
                guild_id INTEGER PRIMARY KEY,
                config_key TEXT NOT NULL,
                config_value TEXT
            )
        """)
        await db_conn.execute("""
            CREATE TABLE IF NOT EXISTS message_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL,
                timestamp REAL NOT NULL, -- Unix timestamp for easy time calculations
                message_content TEXT NOT NULL
            )
        """)
        await db_conn.commit()
        logger.info("Database connection initialized and schema checked.")
    except Exception as e:
        logger.critical(f"CRITICAL: Could not connect to database or create tables: {e}", exc_info=True)
        if http_session and not http_session.closed:
            await http_session.close()
        exit(1)

    try:
        LANGUAGE_MODEL = fasttext.load_model(FASTTEXT_MODEL_PATH)
        logger.info("FastText language model loaded.")
    except ValueError as e:
        logger.error(f"Error loading FastText model: {e}. Language detection will be skipped.", exc_info=True)
        LANGUAGE_MODEL = None
    except Exception as e:
        logger.error(f"Unexpected error loading FastText model: {e}", exc_info=True)
        LANGUAGE_MODEL = None

    async def health_check(request):
        return web.Response(text="Bot is running and accessible!")

    app = web.Application()
    app.router.add_get("/", health_check)

    runner = web.AppRunner(app)
    await runner.setup()

    web_server_task = None
    port = os.getenv("PORT")
    if port:
        try:
            port = int(port)
            site = web.TCPSite(runner, host="0.0.0.0", port=port)
            web_server_task = asyncio.create_task(site.start()) 
            logger.info(f"Web server background task created on port {port}.")
        except ValueError:
            logger.error(f"Invalid PORT environment variable: '{os.getenv('PORT')}'. Port must be an integer.")
        except Exception as e:
            logger.error(f"Error setting up web server: {e}", exc_info=True)
    else:
        logger.warning("PORT environment variable not set. Web server will not start.")
        
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot shutting down due to KeyboardInterrupt (Ctrl+C)...\n")
    except Exception as e:
        logger.critical(f"Unhandled exception at the very top level: {e}", exc_info=True)
    finally:
        logger.info("Initiating final cleanup...")
        # These lines caused premature shutdown and are now removed:
        # if http_session and not http_session.closed:
        #      asyncio.run(http_session.close())
        #      logger.info("Aiohttp session closed.")
        # if db_conn:
        #     asyncio.run(db_conn.close())
        #     logger.info("Database connection closed.")
        pass 

        # try:
        #     tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        #     if tasks:
        #         logger.info(f"Cancelling {len(tasks)} outstanding asyncio tasks...")
        #         [task.cancel() for task in tasks]
        #         asyncio.run(asyncio.gather(*tasks, return_exceptions=True)) # Allow tasks to clean up
        #         logger.info("Outstanding tasks cancelled.")
        # except RuntimeError as e: # Can happen if event loop is already closed
        #      logger.warning(f"Could not cancel tasks, event loop likely closed: {e}")


    logger.info("Bot process finished.")
