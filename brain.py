import asyncio
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
from aiohttp import ClientSession, client_exceptions
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_random_exponential)
from thefuzz import fuzz

load_dotenv()

log_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger('adroit_perfected')
logger.setLevel(logging.INFO)

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
    logger.warning("Warning: Sightengine API keys not set. Image moderation will be disabled.")

intents = discord.Intents.default()
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix=">>", intents=intents, help_command=None)

db_conn: aiosqlite.Connection | None = None
LANGUAGE_MODEL: fasttext.FastText._FastText | None = None
http_session: ClientSession | None = None

class GlobalRateLimiter:
    def __init__(self, max_calls: int, period_seconds: int):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.call_timestamps = deque()

    def allow_call(self) -> bool:
        now = time.monotonic()
        while self.call_timestamps and self.call_timestamps[0] <= now - self.period_seconds:
            self.call_timestamps.popleft()
        if len(self.call_timestamps) < self.max_calls:
            self.call_timestamps.append(now)
            return True
        logger.warning(f"Global rate limit hit ({self.max_calls}/{self.period_seconds}s). Call rejected.")
        return False

openai_rate_limiter = GlobalRateLimiter(max_calls=50, period_seconds=60)

class BotConfig:
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
                5: {"action": "warn", "reason_suffix": "Minor guideline violations."},
                10: {"action": "mute", "duration_hours": 1, "reason_suffix": "Accumulated violations or spam."},
                20: {"action": "mute", "duration_hours": 6, "reason_suffix": "Significant or repeated violations."},
                35: {"action": "kick", "reason_suffix": "Persistent serious violations."},
                50: {"action": "temp_ban", "duration_days": 3, "reason_suffix": "Severe or multiple major violations."},
                75: {"action": "temp_ban", "duration_days": 30, "reason_suffix": "Extreme or highly disruptive behavior."},
                100: {"action": "ban", "reason_suffix": "Egregious violations or repeat offenses."}
            },
            "violations": {
                "discrimination": {"points": 3, "severity": "Medium"}, "spam_rate": {"points": 1, "severity": "Low"},
                "spam_repetition": {"points": 2, "severity": "Low"}, "nsfw_text": {"points": 2, "severity": "Medium"},
                "nsfw_media": {"points": 10, "severity": "High"}, "advertising_forbidden": {"points": 3, "severity": "Medium"},
                "advertising_unpermitted_url": {"points": 3, "severity": "Medium"}, "politics_disallowed": {"points": 1, "severity": "Low"},
                "off_topic": {"points": 1, "severity": "Low"}, "foreign_language": {"points": 1, "severity": "Low"},
                "openai_hate": {"points": 15, "severity": "High"}, "openai_sexual": {"points": 10, "severity": "High"},
                "openai_violence": {"points": 10, "severity": "High"}, "openai_harassment": {"points": 5, "severity": "Medium"},
                "openai_self_harm": {"points": 15, "severity": "High"},
                "excessive_mentions": {"points": 1, "severity": "Low"}, "excessive_attachments": {"points": 1, "severity": "Low"},
                "long_message": {"points": 1, "severity": "Low"}, "gore_violence_media": {"points": 15, "severity": "High"},
                "offensive_symbols_media": {"points": 5, "severity": "Medium"}, "dynamic_rule_violation": {"points": 3, "severity": "Medium"},
                "manual_review_punishment": {"points": 2, "severity": "Medium"}
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
        self.delete_violating_messages = True
        self.send_in_channel_warning = True
        self.in_channel_warning_delete_delay = 30

bot_config = BotConfig()
dynamic_rules = {"forbidden_words": set(), "forbidden_phrases": [], "forbidden_regex": []}

def load_terms_from_file(filepath: str) -> tuple[set[str], list[str]]:
    words, phrases = set(), []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                term = line.strip().lower()
                if not term or term.startswith("#"): continue
                if ' ' in term: phrases.append(term)
                else: words.add(term)
    except FileNotFoundError:
        logger.warning(f"Terms file '{filepath}' not found.")
    except Exception as e:
        logger.error(f"Error loading terms from {filepath}: {e}", exc_info=True)
    return words, phrases

discrimination_words_set, discrimination_phrases = load_terms_from_file('discrimination_terms.txt')
nsfw_text_words_set, nsfw_text_phrases = load_terms_from_file('nsfw_terms.txt')

def clean_message_for_language_detection(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<@!?\d+>|<#\d+>|<@&\d+>', '', text)
    text = re.sub(r'<a?:\w+:\d+>', '', text)
    text = re.sub(r'\|\|.*?\|\|', '', text)
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)
    text = re.sub(r'(\*|_|~|>|#|\-|\+)', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def clean_message_content(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip().lower()

async def get_guild_config(guild_id: int, key: str, default_value=None):
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

async def detect_language_ai(text: str) -> tuple[str | None, float]:
    if not LANGUAGE_MODEL: return None, 0.0
    clean_text = clean_message_for_language_detection(text)
    if not clean_text or not bot_config.has_alphanumeric_pattern.search(clean_text): return None, 0.0
    try:
        prediction = LANGUAGE_MODEL.predict(clean_text.replace("\n", " "), k=1)
        if prediction and prediction[0] and prediction[1]:
            lang_code, confidence = prediction[0][0].replace("__label__", ""), float(prediction[1][0])
            return lang_code, confidence
        return None, 0.0
    except Exception as e:
        logger.error(f"FastText language detection error: {e}", exc_info=True)
        return None, 0.0

async def log_moderation_action(guild: discord.Guild, embed: discord.Embed):
    log_channel_id = await get_guild_config(guild.id, "log_channel_id")
    if not log_channel_id: return
    log_channel = bot.get_channel(int(log_channel_id))
    if not isinstance(log_channel, discord.TextChannel): return

    try: await log_channel.send(embed=embed)
    except discord.Forbidden: logger.error(f"Missing permissions to send logs to channel #{log_channel.name}.")
    except Exception as e: logger.error(f"Error sending log embed to channel {log_channel.id}: {e}", exc_info=True)

def retry_if_api_error(exception):
    if isinstance(exception, client_exceptions.ClientResponseError):
        return exception.status == 429 or exception.status >= 500
    return isinstance(exception, (asyncio.TimeoutError, client_exceptions.ClientOSError))

@retry(stop=stop_after_attempt(4), wait=wait_random_exponential(multiplier=1, max=30), retry=retry_if_api_error, reraise=True)
async def check_openai_moderation_api(text_content: str) -> dict:
    if not text_content.strip() or not http_session or http_session.closed: return {}
    if not openai_rate_limiter.allow_call(): return {}
    url, headers = "https://api.openai.com/v1/moderations", {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"input": text_content.replace("\n", " ")}
    try:
        async with http_session.post(url, headers=headers, json=data, timeout=15) as response:
            response.raise_for_status()
            json_response = await response.json()
            return json_response.get("results", [{}])[0]
    except client_exceptions.ClientResponseError as e:
        if e.status == 400: return {}
        raise
    except Exception as e:
        logger.error(f"Unexpected error with OpenAI moderation: {e}", exc_info=True)
        return {}

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=20), retry=retry_if_api_error, reraise=True)
async def check_sightengine_media_api(image_url: str) -> dict:
    if not (SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET and http_session and not http_session.closed): return {}
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

async def apply_moderation_punishment(member: discord.Member, action_config: dict, total_points: int, violation_summary: str, moderator: discord.User | None = None, original_content: str | None = None):
    action_type, guild = action_config["action"], member.guild
    reason_suffix = action_config.get("reason_suffix", f"Automated action due to {total_points} points.")
    full_reason = f"Automated | Points: {total_points} | {reason_suffix} | Violations: {violation_summary}"
    if moderator: full_reason = f"Manual by {moderator.name}: {reason_suffix}"

    dm_embed = discord.Embed(title=f"Action Taken in {guild.name}", color=discord.Color.orange())
    dm_embed.description = f"**Action:** {action_type.replace('_', ' ').title()}\n**Reason:** {reason_suffix}"
    if original_content: dm_embed.add_field(name="Original Content Snippet", value=f"```\n{original_content[:200]}\n```", inline=False)
    
    log_color, extra_log_fields, duration = discord.Color.orange(), [], None
    if "duration_hours" in action_config: duration = timedelta(hours=action_config["duration_hours"])
    elif "duration_days" in action_config: duration = timedelta(days=action_config["duration_days"])
    if duration:
        dm_embed.add_field(name="Duration", value=str(duration))
        extra_log_fields.append(("Duration", str(duration)))

    if not guild.me or guild.me.top_role <= member.top_role:
        logger.warning(f"Cannot {action_type} {member.display_name}: Check bot role hierarchy.")
        return

    try: await member.send(embed=dm_embed)
    except (discord.Forbidden, discord.HTTPException): pass
        
    try:
        if action_type == "warn": log_color = discord.Color.gold()
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
            log_color, extra_log_fields = discord.Color.dark_red(), [("Unban Time", f"{discord.utils.format_dt(unban_time, 'R')}")]
        elif action_type == "ban":
            await guild.ban(member, reason=full_reason, delete_message_days=0)
            log_color = discord.Color.dark_red()

        log_embed = discord.Embed(title=f"🛡️ Moderation: {action_type.replace('_', ' ').title()}", description=full_reason, color=log_color, timestamp=datetime.now(timezone.utc))
        log_embed.set_author(name=f"{member} ({member.id})", icon_url=member.display_avatar.url)
        log_embed.add_field(name="Moderator", value=moderator.mention if moderator else "Automated")
        for name, value in extra_log_fields: log_embed.add_field(name=name, value=value)
        await log_moderation_action(guild, log_embed)

    except discord.Forbidden: logger.error(f"Lacks permissions for '{action_type}' on {member.display_name}.")
    except Exception as e: logger.error(f"Error applying {action_type} to {member.id}: {e}", exc_info=True)

async def process_infractions_and_punish(member: discord.Member, guild: discord.Guild, violation_types: list[str], content: str, message_url: str | None):
    if not db_conn: return
    user_id, guild_id = member.id, guild.id
    points_this_action, summary_parts = 0, []

    for v_type in violation_types:
        v_config = bot_config.punishment_system["violations"].get(v_type)
        if not v_config: continue
        points_this_action += v_config["points"]
        summary_parts.append(v_type.replace('_', ' ').title())
        await db_conn.execute("INSERT INTO infractions (user_id, guild_id, violation_type, points, message_content_snippet, message_url, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)", (user_id, guild_id, v_type, v_config["points"], content[:500], message_url, datetime.now(timezone.utc).isoformat()))

    if not summary_parts: return await db_conn.commit()

    ninety_days_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    async with db_conn.execute("SELECT SUM(points) FROM infractions WHERE user_id = ? AND guild_id = ? AND timestamp >= ?", (user_id, guild_id, ninety_days_ago)) as cursor:
        total_points_result = await cursor.fetchone()
    total_active_points = total_points_result[0] if total_points_result and total_points_result[0] else 0
    await db_conn.commit()
    
    applicable_punishment_config = None
    for threshold in sorted(bot_config.punishment_system["points_thresholds"].keys(), reverse=True):
        if total_active_points >= threshold:
            applicable_punishment_config = bot_config.punishment_system["points_thresholds"][threshold]
            break
    
    if applicable_punishment_config:
        await apply_moderation_punishment(member, applicable_punishment_config, total_active_points, ", ".join(summary_parts), original_content=content)
        if applicable_punishment_config["action"] == "ban": await clear_user_infractions(user_id, guild_id)

async def get_user_infractions_from_db(user_id: int, guild_id: int, days_limit: int = 90) -> tuple[list[dict], int]:
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
    if not db_conn: return
    try:
        await db_conn.execute("DELETE FROM infractions WHERE user_id = ? AND guild_id = ?", (user_id, guild_id))
        await db_conn.commit()
    except Exception as e: logger.error(f"Error clearing infractions for user {user_id}: {e}", exc_info=True)

async def remove_specific_infraction_from_db(infraction_id: int) -> bool:
    if not db_conn: return False
    try:
        cursor = await db_conn.execute("DELETE FROM infractions WHERE id = ?", (infraction_id,))
        await db_conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Error removing infraction {infraction_id}: {e}", exc_info=True)
        return False

async def get_temp_bans_from_db() -> list[dict]:
    if not db_conn: return []
    try:
        async with db_conn.execute("SELECT user_id, guild_id, unban_time, ban_reason FROM temp_bans") as cursor:
            return [{"user_id": r[0], "guild_id": r[1], "unban_time": datetime.fromisoformat(r[2]), "ban_reason": r[3]} for r in await cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error retrieving temporary bans: {e}", exc_info=True)
        return []

async def remove_temp_ban_from_db(user_id: int, guild_id: int):
    if not db_conn: return
    try:
        await db_conn.execute("DELETE FROM temp_bans WHERE user_id = ? AND guild_id = ?", (user_id, guild_id))
        await db_conn.commit()
    except Exception as e: logger.error(f"Error removing temp ban for user {user_id}: {e}", exc_info=True)

async def setup_database():
    global db_conn
    try:
        db_conn = await aiosqlite.connect('adroit_perfected_data.db')
        await db_conn.execute("PRAGMA journal_mode=WAL;")
        await db_conn.execute("PRAGMA foreign_keys=ON;")
        await db_conn.execute("CREATE TABLE IF NOT EXISTS guild_configs (guild_id INTEGER NOT NULL, config_key TEXT NOT NULL, config_value TEXT, PRIMARY KEY (guild_id, config_key))")
        await db_conn.execute("CREATE TABLE IF NOT EXISTS infractions (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, guild_id INTEGER NOT NULL, violation_type TEXT NOT NULL, points INTEGER NOT NULL, message_content_snippet TEXT, message_url TEXT, timestamp TEXT NOT NULL)")
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_infractions_user_guild_time ON infractions (user_id, guild_id, timestamp)')
        await db_conn.execute("CREATE TABLE IF NOT EXISTS temp_bans (user_id INTEGER NOT NULL, guild_id INTEGER NOT NULL, unban_time TEXT NOT NULL, ban_reason TEXT, PRIMARY KEY (user_id, guild_id))")
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_temp_bans_unban_time ON temp_bans (unban_time)')
        await db_conn.execute("CREATE TABLE IF NOT EXISTS dynamic_rules (id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL, rule_type TEXT NOT NULL, pattern TEXT NOT NULL, added_by_id INTEGER NOT NULL, timestamp TEXT NOT NULL, UNIQUE (guild_id, rule_type, pattern))")
        await db_conn.execute("CREATE TABLE IF NOT EXISTS review_queue (id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL, user_id INTEGER NOT NULL, channel_id INTEGER, message_id INTEGER UNIQUE, message_content TEXT NOT NULL, reason TEXT NOT NULL, timestamp TEXT NOT NULL)")
        await db_conn.execute('CREATE INDEX IF NOT EXISTS idx_review_queue_guild_timestamp ON review_queue (guild_id, timestamp)')
        await db_conn.commit()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize database: {e}", exc_info=True)
        if db_conn: await db_conn.close()
        exit(1)

async def load_dynamic_rules_from_db():
    if not db_conn: return
    dynamic_rules["forbidden_words"].clear(); dynamic_rules["forbidden_phrases"].clear(); dynamic_rules["forbidden_regex"] = []
    try:
        async with db_conn.execute("SELECT rule_type, pattern FROM dynamic_rules") as cursor:
            async for rule_type, pattern in cursor:
                if rule_type == 'forbidden_word': dynamic_rules["forbidden_words"].add(pattern.lower())
                elif rule_type == 'forbidden_phrase': dynamic_rules["forbidden_phrases"].append(pattern.lower())
                elif rule_type == 'forbidden_regex':
                    try: dynamic_rules["forbidden_regex"].append(re.compile(pattern, re.IGNORECASE))
                    except re.error as e: logger.error(f"Failed to compile regex '{pattern}': {e}")
    except Exception as e: logger.error(f"Error loading dynamic rules: {e}", exc_info=True)

class GeneralCog(commands.Cog, name="General"):
    def __init__(self, bot: commands.Bot): self.bot = bot
    @app_commands.command(name="ping", description="Checks the bot's latency.")
    async def ping(self, interaction: discord.Interaction):
        await interaction.response.send_message(f"Pong! Latency: {round(self.bot.latency * 1000)}ms", ephemeral=True)

    @app_commands.command(name="help", description="Shows information about bot commands.")
    async def help_command(self, interaction: discord.Interaction):
        embed = discord.Embed(title="Adroit Bot Help", description="Below is a list of available command categories.", color=discord.Color.blue())
        for cog in self.bot.cogs.values():
            commands_list = []
            for cmd in cog.get_app_commands():
                if isinstance(cmd, app_commands.Group):
                    commands_list.extend(f"`/{cmd.name} {sub.name}` - {sub.description}" for sub in cmd.commands)
                else: commands_list.append(f"`/{cmd.name}` - {cmd.description}")
            if commands_list: embed.add_field(name=f"**{cog.qualified_name} Commands**", value="\n".join(commands_list), inline=False)
        await interaction.response.send_message(embed=embed, ephemeral=True)

class ConfigurationCog(commands.Cog, name="Configuration"):
    def __init__(self, bot: commands.Bot): self.bot = bot
    group = app_commands.Group(name="config", description="Configure bot settings.", default_permissions=discord.Permissions(manage_guild=True))

    @group.command(name="set_log_channel", description="Sets the moderation log channel. (Leave empty to clear)")
    async def set_log_channel(self, interaction: discord.Interaction, channel: discord.TextChannel | None = None):
        if not interaction.guild_id: return
        await set_guild_config(interaction.guild_id, "log_channel_id", channel.id if channel else None)
        await interaction.response.send_message(f"Moderation logs set to {channel.mention}." if channel else "Moderation log channel cleared.", ephemeral=True)

    @group.command(name="set_review_channel", description="Sets channel for messages needing review. (Leave empty to clear)")
    async def set_review_channel(self, interaction: discord.Interaction, channel: discord.TextChannel | None = None):
        if not interaction.guild_id: return
        await set_guild_config(interaction.guild_id, "review_channel_id", channel.id if channel else None)
        await interaction.response.send_message(f"Review messages set to {channel.mention}." if channel else "Review channel cleared.", ephemeral=True)

    async def _set_language_config(self, interaction: discord.Interaction, languages_str: str, config_key: str, entity_mention: str):
        if not interaction.guild_id: return
        lang_list_raw = [lang.strip().lower() for lang in languages_str.split(',')]
        is_any, is_default = "any" in lang_list_raw, "default" in lang_list_raw
        if is_any: await set_guild_config(interaction.guild_id, config_key, ["any"])
        elif is_default: await set_guild_config(interaction.guild_id, config_key, None)
        else:
            valid_langs = [lang for lang in lang_list_raw if re.fullmatch(r"[a-z]{2,3}", lang)]
            if len(valid_langs) != len(lang_list_raw):
                await interaction.response.send_message("Invalid language code format provided. Use 2-3 letter ISO codes.", ephemeral=True)
                return
            await set_guild_config(interaction.guild_id, config_key, valid_langs)
        
        msg = f"Language check for {entity_mention} now allows **any** language." if is_any else \
              f"Language override for {entity_mention} removed. Now uses server default." if is_default else \
              f"Languages for {entity_mention} set to: **{', '.join(valid_langs).upper()}**." if 'valid_langs' in locals() and valid_langs else \
              f"Language configuration for {entity_mention} cleared."
        await interaction.response.send_message(msg, ephemeral=True)

    @group.command(name="set_server_language", description="Sets default server language(s). Use 'any' to disable.")
    @app_commands.describe(languages="Comma-separated ISO codes (e.g., en,fr).")
    async def set_server_language(self, interaction: discord.Interaction, languages: str):
        if not interaction.guild: return
        await self._set_language_config(interaction, languages, "default_language", f"server **{interaction.guild.name}**")

    @group.command(name="set_channel_language", description="Overrides server language for a channel. Use 'any' or 'default'.")
    @app_commands.describe(channel="Channel to configure.", languages="Comma-separated ISO codes, 'any', or 'default' to clear.")
    async def set_channel_language(self, interaction: discord.Interaction, channel: discord.TextChannel, languages: str):
        await self._set_language_config(interaction, languages, f"channel_language_{channel.id}", channel.mention)

class ModerationCog(commands.Cog, name="Moderation"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.user_message_timestamps: defaultdict[int, defaultdict[int, deque]] = defaultdict(lambda: defaultdict(deque))
        self.user_message_history: defaultdict[int, defaultdict[int, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=bot_config.spam_repetition_history_count)))
        self.report_message_context_menu = app_commands.ContextMenu(name="Report to Adroit", callback=self.report_message_from_context)
        self.bot.tree.add_command(self.report_message_context_menu)
        self.temp_ban_check_task.start()
        self.cleanup_old_infractions_task.start()
        self.cleanup_spam_trackers_task.start()

    def cog_unload(self):
        self.temp_ban_check_task.cancel()
        self.cleanup_old_infractions_task.cancel()
        self.cleanup_spam_trackers_task.cancel()
        self.bot.tree.remove_command(self.report_message_context_menu.name, type=self.report_message_context_menu.type)

    async def report_message_from_context(self, interaction: discord.Interaction, message: discord.Message):
        if message.author.id == interaction.user.id or message.author.bot:
            await interaction.response.send_message("You cannot report your own or a bot's message.", ephemeral=True)
            return
        await interaction.response.send_modal(ReportMessageModal(message=message))

    async def get_effective_channel_language_config(self, guild_id: int, channel_id: int) -> list[str] | None:
        channel_setting = await get_guild_config(guild_id, f"channel_language_{channel_id}")
        return channel_setting if channel_setting is not None else await get_guild_config(guild_id, "default_language", ["en"])

    @tasks.loop(minutes=1)
    async def temp_ban_check_task(self):
        now_utc = datetime.now(timezone.utc)
        for ban in [b for b in await get_temp_bans_from_db() if now_utc >= b["unban_time"]]:
            guild = self.bot.get_guild(ban["guild_id"])
            if not guild:
                await remove_temp_ban_from_db(ban["user_id"], ban["guild_id"])
                continue
            try:
                user_obj = discord.Object(id=ban["user_id"])
                await guild.unban(user_obj, reason=f"Temp ban expired. Original reason: {ban['ban_reason']}")
                embed = discord.Embed(title="✅ Temp Ban Expired", description=f"User ID `{ban['user_id']}` has been unbanned.", color=discord.Color.green())
                await log_moderation_action(guild, embed)
            except discord.NotFound: pass
            except discord.Forbidden: logger.error(f"Missing permissions to unban user {ban['user_id']} from {guild.name}.")
            except Exception as e: logger.error(f"Error unbanning user {ban['user_id']}: {e}", exc_info=True)
            finally: await remove_temp_ban_from_db(ban["user_id"], ban["guild_id"])

    @tasks.loop(hours=24)
    async def cleanup_old_infractions_task(self):
        if not db_conn: return
        one_hundred_eighty_days_ago = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
        try:
            cursor = await db_conn.execute("DELETE FROM infractions WHERE timestamp < ?", (one_hundred_eighty_days_ago,))
            await db_conn.commit()
            if cursor.rowcount > 0: logger.info(f"Deleted {cursor.rowcount} infractions older than 180 days.")
        except Exception as e: logger.error(f"Error during infraction cleanup: {e}", exc_info=True)
    
    @tasks.loop(hours=6)
    async def cleanup_spam_trackers_task(self):
        now_ts, cleanup_threshold = datetime.now(timezone.utc).timestamp(), bot_config.spam_window_seconds * 10
        for guild_id in list(self.user_message_timestamps):
            for user_id in list(self.user_message_timestamps[guild_id]):
                timestamps = self.user_message_timestamps[guild_id][user_id]
                while timestamps and (now_ts - timestamps[0] > cleanup_threshold): timestamps.popleft()
                if not timestamps: del self.user_message_timestamps[guild_id][user_id]
            if not self.user_message_timestamps[guild_id]: del self.user_message_timestamps[guild_id]

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if not message.guild or message.author.bot or not isinstance(message.author, discord.Member): return
        violations, proactive_flag_reason = await self._run_moderation_pipeline(message)
        if violations:
            await self._handle_violations(message, violations)
        elif proactive_flag_reason:
            await self.add_to_review_queue(message, proactive_flag_reason)
        await self.bot.process_commands(message)
        
    async def _run_moderation_pipeline(self, message: discord.Message) -> tuple[set[str], str | None]:
        violations, proactive_flag_reason = set(), None
        content_raw, cleaned_content = message.content, clean_message_content(message.content)
        
        local_violations = self.check_dynamic_rules(content_raw, cleaned_content) \
            | self.check_spam(message.author.id, message.guild.id, cleaned_content) \
            | self.check_message_limits(message) \
            | self.check_keyword_violations(cleaned_content)
        
        if local_violations: return local_violations, None

        tasks = [self.check_advertising(content_raw, message.guild.id), self.check_language(message, content_raw)]
        if OPENAI_API_KEY: tasks.append(self.check_ai_text_moderation(content_raw, message.guild.id))
        if SIGHTENGINE_API_USER and message.attachments: tasks.append(self.check_ai_media_moderation(message.attachments))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception): logger.error(f"Moderation check failed: {res}", exc_info=True)
            elif isinstance(res, set): violations.update(res)
            elif isinstance(res, tuple):
                ai_violations, proactive_reason = res
                violations.update(ai_violations)
                if proactive_reason and not proactive_flag_reason: proactive_flag_reason = proactive_reason
        return violations, proactive_flag_reason

    async def _handle_violations(self, message: discord.Message, violations: set[str]):
        if await get_guild_config(message.guild.id, "delete_violating_messages", bot_config.delete_violating_messages):
            try: await message.delete()
            except (discord.Forbidden, discord.NotFound): pass
        if await get_guild_config(message.guild.id, "send_in_channel_warning", bot_config.send_in_channel_warning):
            viol_summary = ", ".join(v.replace('_', ' ').title() for v in violations)
            warn_text = f"{message.author.mention}, your message was moderated for: **{viol_summary}**. Please review the rules."
            try:
                delete_delay = await get_guild_config(message.guild.id, "in_channel_warning_delete_delay", bot_config.in_channel_warning_delete_delay)
                await message.channel.send(warn_text, delete_after=int(delete_delay))
            except (discord.Forbidden, ValueError): pass
        if isinstance(message.author, discord.Member):
            await process_infractions_and_punish(message.author, message.guild, list(violations), message.content, message.jump_url)

    def check_dynamic_rules(self, raw_content: str, cleaned_content: str) -> set[str]:
        if any(regex.search(raw_content) for regex in dynamic_rules["forbidden_regex"]) or \
           any(word in cleaned_content.split() for word in dynamic_rules["forbidden_words"]) or \
           any(phrase in cleaned_content for phrase in dynamic_rules["forbidden_phrases"]):
            return {"dynamic_rule_violation"}
        return set()

    def check_spam(self, user_id: int, guild_id: int, cleaned_content: str) -> set[str]:
        violations, now_ts = set(), datetime.now(timezone.utc).timestamp()
        timestamps = self.user_message_timestamps[guild_id][user_id]
        timestamps.append(now_ts)
        while timestamps and (now_ts - timestamps[0] > bot_config.spam_window_seconds): timestamps.popleft()
        if len(timestamps) > bot_config.spam_message_limit: violations.add("spam_rate")
        
        history = self.user_message_history[guild_id][user_id]
        if cleaned_content and "spam_rate" not in violations:
            if any(fuzz.ratio(cleaned_content, old) > bot_config.spam_repetition_fuzzy_threshold for old in history):
                violations.add("spam_repetition")
                history.clear()
        history.append(cleaned_content)
        return violations

    async def check_advertising(self, raw_content: str, guild_id: int) -> set[str]:
        if bot_config.forbidden_text_pattern.search(raw_content): return {"advertising_forbidden"}
        guild_permitted_domains = await get_guild_config(guild_id, "permitted_domains", bot_config.permitted_domains)
        for url_match in re.finditer(bot_config.url_pattern, raw_content):
            try:
                schemed_url = url_match.group(0) if url_match.group(0).startswith(('http', 'www')) else 'http://' + url_match.group(0)
                domain = urlparse(schemed_url).netloc.lower().lstrip('www.')
                if domain and not any(allowed == domain or domain.endswith(f'.{allowed}') for allowed in guild_permitted_domains):
                    return {"advertising_unpermitted_url"}
            except Exception: continue
        return set()

    def check_message_limits(self, message: discord.Message) -> set[str]:
        v = set()
        if len(message.mentions) > bot_config.mention_limit: v.add("excessive_mentions")
        if len(message.attachments) > bot_config.max_attachments: v.add("excessive_attachments")
        if len(message.content) > bot_config.max_message_length: v.add("long_message")
        return v

    async def check_language(self, message: discord.Message, raw_content: str) -> set[str]:
        if len(raw_content.split()) < bot_config.min_word_count_for_lang_check: return set()
        lang_config = await self.get_effective_channel_language_config(message.guild.id, message.channel.id)
        if not lang_config or "any" in lang_config: return set()
        
        lang_code, confidence = await detect_language_ai(raw_content)
        if not lang_code or lang_code in lang_config: return set()

        threshold = bot_config.min_confidence_short_msg_lang if len(raw_content) < bot_config.short_msg_threshold_lang else bot_config.min_confidence_for_lang_flagging
        if confidence >= threshold and not any(word in raw_content.lower() for word in bot_config.common_safe_foreign_words):
            return {"foreign_language"}
        return set()

    def check_keyword_violations(self, cleaned_content: str) -> set[str]:
        violations = set()
        words = cleaned_content.split()
        if any(word in words for word in discrimination_words_set) or \
           any(phrase in cleaned_content for phrase in discrimination_phrases):
            violations.add("discrimination")
        if any(word in words for word in nsfw_text_words_set) or \
           any(phrase in cleaned_content for phrase in nsfw_text_phrases):
            violations.add("nsfw_text")
        return violations

    async def check_ai_text_moderation(self, content: str, guild_id: int) -> tuple[set[str], str | None]:
        violations, proactive_reason = set(), None
        try:
            result = await check_openai_moderation_api(content)
            if not result: return violations, proactive_reason
            
            categories, scores = result.get("categories", {}), result.get("category_scores", {})
            if result.get("flagged"):
                if categories.get("hate") or categories.get("hate/threatening"): violations.add("openai_hate")
                if categories.get("sexual") or categories.get("sexual/minors"): violations.add("openai_sexual")
                if categories.get("violence") or categories.get("violence/graphic"): violations.add("openai_violence")
                if categories.get("harassment") or categories.get("harassment/threatening"): violations.add("openai_harassment")
                if categories.get("self-harm") or categories.get("self-harm/intent") or categories.get("self-harm/instructions"): violations.add("openai_self_harm")
            else:
                threshold = await get_guild_config(guild_id, "proactive_flagging_openai_threshold", bot_config.proactive_flagging_openai_threshold)
                if scores and (max_score := max(scores.values())) >= threshold:
                    category = max(scores, key=scores.get)
                    proactive_reason = f"Proactive OpenAI Flag ({category.replace('/', ' ')}: {max_score:.2f})"
        except Exception as e: logger.error(f"AI text moderation check failed: {e}", exc_info=True)
        return violations, proactive_reason

    async def check_ai_media_moderation(self, attachments: list[discord.Attachment]) -> set[str]:
        for attach in attachments:
            if not (attach.content_type and attach.content_type.startswith(("image/", "video/"))): continue
            try:
                result = await check_sightengine_media_api(attach.url)
                if not result: continue
                if result.get("nudity", {}).get("sexual_activity", 0) > bot_config.sightengine_nudity_sexual_activity_threshold or \
                   result.get("nudity", {}).get("suggestive", 0) > bot_config.sightengine_nudity_suggestive_threshold:
                    return {"nsfw_media"}
                if result.get("gore", {}).get("prob", 0) > bot_config.sightengine_gore_threshold: return {"gore_violence_media"}
                if result.get("offensive", {}).get("prob", 0) > bot_config.sightengine_offensive_symbols_threshold: return {"offensive_symbols_media"}
            except Exception as e: logger.error(f"Sightengine failed for {attach.url}: {e}", exc_info=True)
        return set()

    async def add_to_review_queue(self, message: discord.Message, reason: str):
        if not db_conn: return
        try:
            cursor = await db_conn.execute("INSERT OR IGNORE INTO review_queue (guild_id, user_id, channel_id, message_id, message_content, reason, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)", (message.guild.id, message.author.id, message.channel.id, message.id, message.content, reason, datetime.now(timezone.utc).isoformat()))
            if cursor.rowcount == 0: return
            await db_conn.commit()
            
            review_channel_id = await get_guild_config(message.guild.id, "review_channel_id")
            if review_channel_id and (review_channel := self.bot.get_channel(int(review_channel_id))):
                embed = discord.Embed(title="🚨 Message Flagged for Review", description=f"**Reason:** {reason}", color=discord.Color.yellow(), timestamp=message.created_at)
                embed.add_field(name="Author", value=f"{message.author.mention} (`{message.author.id}`)")
                embed.add_field(name="Channel", value=f"{message.channel.mention}")
                embed.add_field(name="Context", value=f"[Jump to Message]({message.jump_url})", inline=False)
                embed.add_field(name="Content", value=f"```\n{discord.utils.escape_markdown(message.content[:500])}\n```", inline=False)
                await review_channel.send(embed=embed)
        except Exception as e: logger.error(f"Failed to add message {message.id} to review queue: {e}", exc_info=True)

    mod_group = app_commands.Group(name="mod", description="Moderation commands.", default_permissions=discord.Permissions(manage_messages=True))

    @mod_group.command(name="infractions", description="View a user's recent infractions.")
    async def infractions_command(self, interaction: discord.Interaction, member: discord.Member, days: app_commands.Range[int, 1, 365] = 90):
        if not interaction.guild_id: return
        infraction_list, active_points = await get_user_infractions_from_db(member.id, interaction.guild_id, days)
        embed = discord.Embed(title=f"Infraction Report for {member.display_name}", color=discord.Color.red() if active_points > 20 else discord.Color.orange())
        embed.set_thumbnail(url=member.display_avatar.url)
        embed.add_field(name="Active Points (Last 90 Days)", value=f"**{active_points}**", inline=False)
        if infraction_list:
            history_str = "\n".join([f"**ID:** `{i['id']}` | **{i['points']} pts** - {i['violation_type'].replace('_', ' ').title()}\n> <t:{int(datetime.fromisoformat(i['timestamp']).timestamp())}:R>" for i in infraction_list[:10]])
            embed.add_field(name=f"Recent Infractions (Last {days} Days)", value=history_str, inline=False)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @mod_group.command(name="clear_infractions", description="Clears ALL or a specific infraction for a user.")
    @app_commands.default_permissions(administrator=True)
    @app_commands.describe(member="Member to clear infractions.", infraction_id="Optional ID of a specific infraction to remove.")
    async def clear_infractions_command(self, interaction: discord.Interaction, member: discord.Member, infraction_id: int | None = None):
        if not interaction.guild or not interaction.user: return
        if infraction_id:
            success, action_desc = await remove_specific_infraction_from_db(infraction_id), f"infraction ID `{infraction_id}`"
        else:
            await clear_user_infractions(member.id, interaction.guild.id)
            success, action_desc = True, "all infractions"
        
        if success:
            await interaction.response.send_message(f"Cleared {action_desc} for {member.mention}.", ephemeral=True)
            log_embed = discord.Embed(title="✅ Infractions Cleared", description=f"Moderator {interaction.user.mention} cleared {action_desc} for {member.mention}.", color=discord.Color.green())
            await log_moderation_action(interaction.guild, log_embed)
        else: await interaction.response.send_message(f"Failed to clear {action_desc}. It may not exist.", ephemeral=True)

    async def _manual_action(self, interaction: discord.Interaction, user: discord.User, action_key: str, reason: str, duration: timedelta | None = None):
        if not interaction.guild or not interaction.user: return
        if user.id == self.bot.user.id or (user.id == interaction.user.id and action_key != "warn"):
            return await interaction.response.send_message("You cannot perform this action on this user.", ephemeral=True)
        member = interaction.guild.get_member(user.id)
        if member and interaction.guild.owner_id != interaction.user.id and member.top_role >= interaction.user.top_role:
            return await interaction.response.send_message("You cannot moderate a member with an equal or higher role.", ephemeral=True)
        
        action_config = {"action": action_key, "reason_suffix": reason}
        if duration: action_config["duration_days"] = duration.days
        
        if action_key in ["ban", "temp_ban"]:
            try:
                await interaction.guild.ban(user, reason=f"Manual by {interaction.user.name}: {reason}", delete_message_days=0)
                if action_key == "temp_ban" and duration and db_conn:
                    unban_time = datetime.now(timezone.utc) + duration
                    await db_conn.execute('INSERT OR REPLACE INTO temp_bans (user_id, guild_id, unban_time, ban_reason) VALUES (?, ?, ?, ?)', (user.id, interaction.guild.id, unban_time.isoformat(), reason))
                    await db_conn.commit()
                log_embed = discord.Embed(title=f"🔨 Manual {action_key.title()}", description=f"**Target:** {user.mention}\n**Moderator:** {interaction.user.mention}\n**Reason:** {reason}", color=discord.Color.dark_red())
                if duration: log_embed.add_field(name="Duration", value=str(duration))
                await log_moderation_action(interaction.guild, log_embed)
                await interaction.response.send_message(f"Applied **{action_key.upper()}** to {user.mention}.", ephemeral=True)
            except (discord.Forbidden, discord.HTTPException) as e: await interaction.response.send_message(f"Failed to ban user: {e}", ephemeral=True)
        elif member:
             await apply_moderation_punishment(member, action_config, 0, f"Manual Action: {reason}", moderator=interaction.user)
             await interaction.response.send_message(f"Applied **{action_key.upper()}** to {member.mention}.", ephemeral=True)
        else: await interaction.response.send_message("Cannot apply this action, user is not in the server.", ephemeral=True)

    @mod_group.command(name="warn", description="Manually warn a member.")
    async def manual_warn_command(self, interaction: discord.Interaction, member: discord.Member, reason: str):
        await self._manual_action(interaction, member, "warn", reason)

    @mod_group.command(name="mute", description="Manually mute a member for a duration.")
    @app_commands.describe(duration_hours="How many hours to mute for.")
    async def manual_mute_command(self, interaction: discord.Interaction, member: discord.Member, duration_hours: app_commands.Range[float, 0.01], reason: str):
        await self._manual_action(interaction, member, "mute", reason, timedelta(hours=duration_hours))

    @mod_group.command(name="kick", description="Manually kick a member.")
    async def manual_kick_command(self, interaction: discord.Interaction, member: discord.Member, reason: str):
        await self._manual_action(interaction, member, "kick", reason)

    @mod_group.command(name="ban", description="Manually ban a user (temporarily or permanently).")
    @app_commands.describe(user="User to ban (ID if not in server).", duration_days="Optional duration for a temp ban.")
    async def manual_ban_command(self, interaction: discord.Interaction, user: discord.User, reason: str, duration_days: app_commands.Range[float, 0.01] | None = None):
        action = "temp_ban" if duration_days else "ban"
        duration = timedelta(days=duration_days) if duration_days else None
        await self._manual_action(interaction, user, action, reason, duration)

    @mod_group.command(name="unban", description="Manually unban a user by their ID.")
    async def manual_unban_command(self, interaction: discord.Interaction, user_id: str, reason: str):
        if not interaction.guild or not interaction.user: return
        try: uid = int(user_id)
        except ValueError: return await interaction.response.send_message("Invalid User ID format.", ephemeral=True)
        
        try:
            user_obj = discord.Object(id=uid)
            await interaction.guild.unban(user_obj, reason=f"Manual unban by {interaction.user.name}: {reason}")
            await remove_temp_ban_from_db(uid, interaction.guild.id)
            target_user = await self.bot.fetch_user(uid)
            log_embed = discord.Embed(title="✅ Manual Unban", description=f"**Target:** {target_user.mention}\n**Moderator:** {interaction.user.mention}\n**Reason:** {reason}", color=discord.Color.green())
            await log_moderation_action(interaction.guild, log_embed)
            await interaction.response.send_message(f"Unbanned User ID `{uid}`.", ephemeral=True)
        except discord.NotFound: await interaction.response.send_message(f"User ID {uid} is not banned.", ephemeral=True)
        except discord.Forbidden: await interaction.response.send_message("Failed to unban: Missing permissions.", ephemeral=True)

    @mod_group.command(name="review", description="Review the oldest message in the moderation queue.")
    async def review_command(self, interaction: discord.Interaction):
        if not interaction.guild_id or not db_conn: return
        async with db_conn.execute("SELECT id, user_id, channel_id, message_id, message_content, reason, timestamp FROM review_queue WHERE guild_id = ? ORDER BY timestamp ASC LIMIT 1", (interaction.guild_id,)) as cursor:
            item = await cursor.fetchone()
        if not item: return await interaction.response.send_message("The moderation review queue is empty.", ephemeral=True)

        review_id, user_id, channel_id, message_id, content, reason, timestamp = item
        user = interaction.guild.get_member(user_id) or await self.bot.fetch_user(user_id)
        message_url = f"https://discord.com/channels/{interaction.guild_id}/{channel_id}/{message_id}"
        
        embed = discord.Embed(title="Moderation Review Required", description=f"**Reason for Flag:** {reason}", color=discord.Color.orange(), timestamp=datetime.fromisoformat(timestamp))
        embed.add_field(name="Author", value=f"{user.mention if user else f'ID: {user_id}'}", inline=True)
        embed.add_field(name="Channel", value=f"<#{channel_id}>", inline=True)
        embed.add_field(name="Message", value=f"[Jump to Message]({message_url})", inline=False)
        embed.add_field(name="Content", value=f"```{discord.utils.escape_markdown(content[:1000])}```", inline=False)
        embed.set_footer(text=f"Review Item ID: {review_id}")
        
        view = ReviewActionView(review_id=review_id, member=user, content=content, reason=reason, message_url=message_url)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
        view.message = await interaction.original_response()

class ReportMessageModal(discord.ui.Modal, title="Report Message to Moderators"):
    def __init__(self, message: discord.Message):
        super().__init__()
        self.message = message
    reason_input = discord.ui.TextInput(label="Reason for reporting", style=discord.TextStyle.paragraph, placeholder="e.g., 'Spam', 'Hate Speech', 'Offensive content'", required=True, max_length=500)

    async def on_submit(self, interaction: discord.Interaction):
        if not interaction.guild or not interaction.user: return
        mod_cog = bot.get_cog("Moderation")
        if not mod_cog: return
        report_reason = f"User Report by {interaction.user.name}: {self.reason_input.value}"
        await mod_cog.add_to_review_queue(self.message, report_reason)
        await interaction.response.send_message("✅ Your report has been submitted. Thank you!", ephemeral=True)

class AddRuleModal(discord.ui.Modal, title="Add New Violation Rule"):
    def __init__(self, parent_view: "ReviewActionView"):
        super().__init__()
        self.parent_view = parent_view
    rule_type = discord.ui.Select(placeholder="Select the rule type to add...", options=[
        discord.SelectOption(label="Forbidden Word", value="forbidden_word"),
        discord.SelectOption(label="Forbidden Phrase", value="forbidden_phrase"),
        discord.SelectOption(label="Forbidden Regex", value="forbidden_regex")
    ])
    pattern = discord.ui.TextInput(label="Pattern", style=discord.TextStyle.short, placeholder="Enter the word, phrase, or regex pattern.")

    async def on_submit(self, interaction: discord.Interaction):
        if not db_conn or not interaction.guild_id or not interaction.user or not self.rule_type.values:
            return await interaction.response.send_message("An error occurred.", ephemeral=True, delete_after=10)
            
        rule_type_val, pattern_val = self.rule_type.values[0], self.pattern.value.strip()
        log_reason = ""
        try:
            if rule_type_val == 'forbidden_regex': re.compile(pattern_val)
            await db_conn.execute("INSERT INTO dynamic_rules (guild_id, rule_type, pattern, added_by_id, timestamp) VALUES (?, ?, ?, ?, ?)", (interaction.guild_id, rule_type_val, pattern_val, interaction.user.id, datetime.now(timezone.utc).isoformat()))
            await db_conn.commit()
            await load_dynamic_rules_from_db()
            
            await interaction.response.send_message(f"✅ Rule added and review item `{self.parent_view.review_id}` closed.", ephemeral=True, delete_after=5)
            log_reason = f"Added new rule: `{pattern_val}`"
            
        except re.error:
            return await interaction.response.send_message("⚠️ Invalid Regex. Rule not added.", ephemeral=True, delete_after=10)
        except aiosqlite.IntegrityError:
            await interaction.response.send_message(f"⚠️ Rule already exists. Review item `{self.parent_view.review_id}` closed.", ephemeral=True, delete_after=5)
            log_reason = "Rule already existed."
        except Exception as e:
            return await interaction.response.send_message(f"An error occurred: {e}", ephemeral=True, delete_after=10)

        await self.parent_view._close_review_item_backend(interaction, discord.Color.blue(), log_reason)
        if self.parent_view.message:
            await self.parent_view.message.edit(content=f"✅ Review item `{self.parent_view.review_id}` has been resolved.", view=None, embed=None)


class ReviewActionView(discord.ui.View):
    def __init__(self, review_id: int, member: discord.Member | discord.User | None, content: str, reason: str, message_url: str):
        super().__init__(timeout=600)
        self.review_id, self.member, self.content, self.reason, self.message_url = review_id, member, content, reason, message_url
        self.message: discord.WebhookMessage | None = None

    async def _close_review_item_backend(self, interaction: discord.Interaction, color: discord.Color, log_reason: str):
        if not db_conn or not interaction.guild: return
        await db_conn.execute("DELETE FROM review_queue WHERE id = ?", (self.review_id,))
        await db_conn.commit()
        log_embed = discord.Embed(title="Review Item Closed", description=f"**Item ID:** `{self.review_id}`\n**Moderator:** {interaction.user.mention}\n**Action:** {log_reason}", color=color)
        log_embed.add_field(name="Context", value=f"[Original Message]({self.message_url})")
        await log_moderation_action(interaction.guild, log_embed)

    @discord.ui.button(label="Punish & Add Rule", style=discord.ButtonStyle.danger, emoji="⚖️")
    async def punish_and_add_rule(self, interaction: discord.Interaction, button: discord.ui.Button):
        if isinstance(self.member, discord.Member):
            await process_infractions_and_punish(self.member, self.member.guild, ["manual_review_punishment"], self.content, self.message_url)
        await interaction.response.send_modal(AddRuleModal(parent_view=self))

    @discord.ui.button(label="Punish Only", style=discord.ButtonStyle.primary, emoji="🔨")
    async def punish_only(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not isinstance(self.member, discord.Member):
            return await interaction.response.send_message("Cannot punish user, they may have left.", ephemeral=True, delete_after=10)
        await process_infractions_and_punish(self.member, self.member.guild, ["manual_review_punishment"], self.content, self.message_url)
        await self._close_review_item_backend(interaction, discord.Color.orange(), "Punished based on manual review.")
        await interaction.response.edit_message(content=f"✅ User punished and review item `{self.review_id}` closed.", view=None, embed=None)

    @discord.ui.button(label="Mark as Safe", style=discord.ButtonStyle.success, emoji="✅")
    async def mark_safe(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._close_review_item_backend(interaction, discord.Color.green(), "Moderator marked as safe.")
        await interaction.response.edit_message(content=f"✅ Review item `{self.review_id}` marked as safe.", view=None, embed=None)

@bot.event
async def setup_hook():
    global http_session, LANGUAGE_MODEL
    http_session = ClientSession(connector=aiohttp.TCPConnector(ssl=False))
    if os.path.exists(FASTTEXT_MODEL_PATH):
        try:
            LANGUAGE_MODEL = fasttext.load_model(FASTTEXT_MODEL_PATH)
            logger.info(f"FastText model loaded from {FASTTEXT_MODEL_PATH}")
        except Exception as e: logger.critical(f"Failed to load FastText model: {e}. Language detection disabled.", exc_info=True)
    else: logger.error(f"FastText model not found at {FASTTEXT_MODEL_PATH}. Language detection disabled.")
    
    await setup_database()
    await load_dynamic_rules_from_db()
    
    await bot.add_cog(GeneralCog(bot))
    await bot.add_cog(ConfigurationCog(bot))
    await bot.add_cog(ModerationCog(bot))
    
    synced_commands = await bot.tree.sync()
    logger.info(f"Synced {len(synced_commands)} application commands.")

@bot.event
async def on_ready():
    logger.info(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    logger.info(f"Discord.py Version: {discord.__version__}")
    logger.info(f"{bot.user.name} is online and ready!")

async def main():
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
