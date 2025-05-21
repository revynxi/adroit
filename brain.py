import os
import re
import asyncio
import discord
from discord.ext import commands, tasks
from datetime import datetime, timedelta
from aiohttp import web, ClientSession
import aiosqlite
import fasttext
from dotenv import load_dotenv
from urllib.parse import urlparse
from luga import language
from collections import deque

load_dotenv()
init_db()

intents = discord.Intents.default()
intents.members = True
intents.message_content = True
bot = commands.Bot(command_prefix=">>", intents=intents)

LANGUAGE_MODEL = None
user_message_count = {}
user_message_history = {}  # guild_id: {user_id: deque of last 5 messages}
LOG_CHANNEL_ID = 1113377818424922132

CHANNEL_CONFIG = {
    1113377809440722974: {"language": ["en"]},
    1322517478365990984: {"language": ["en"], "topics": ["politics"]},
    1113377810476716132: {"language": ["en"]},
    1321499824926888049: {"language": ["fr"]},
    1122525009102000269: {"language": ["de"]},
    1122523546355245126: {"language": ["ru"]},
    1122524817904635904: {"language": ["zh"]},
    1242768362237595749: {"language": ["es"]}
}

forbidden_text_pattern = re.compile(
    r"(discord\.gg/|join\s+our|server\s+invite|free\s+nitro|check out my|follow me|subscribe to|buy now)",
    re.IGNORECASE
)
url_pattern = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|net|org)\b)")
permitted_domains = ["youtube.com", "youtu.be", "tenor.com", "giphy.com", "tiktok.com"]

PUNISHMENT_SYSTEM = {
    "points_thresholds": {
        3: {"action": "warn", "message": "Warnings make your sins weigh harder, think twice before sending something"},
        5: {"action": "mute", "duration": timedelta(hours=1)},
        7: {"action": "kick"},
        10: {"action": "temp_ban", "duration": timedelta(days=1)},
        20: {"action": "ban"}
    },
    "violations": {
        "discrimination": {"points": 5},
        "spam": {"points": 2},
        "nsfw": {"points": 4},
        "advertising": {"points": 3},
        "politics": {"points": 3},
        "off_topic": {"points": 1},
        "foreign_language": {"points": 2},
        "openai_moderation": {"points": 3}
    }
}

SPAM_WINDOW = 10
SPAM_LIMIT = 5
MENTION_LIMIT = 5

# Load terms for discrimination and NSFW detection
discrimination_words = set()
discrimination_phrases = []
nsfw_words = set()
nsfw_phrases = []

try:
    with open('discrimination_terms.txt', 'r') as f:
        for line in f:
            term = line.strip().lower()
            if ' ' in term:
                discrimination_phrases.append(term)
            else:
                discrimination_words.add(term)
except FileNotFoundError:
    print("Warning: discrimination_terms.txt not found. Using empty list.")
    discrimination_phrases = []
    discrimination_words = set()

try:
    with open('nsfw_terms.txt', 'r') as f:
        for line in f:
            term = line.strip().lower()
            if ' ' in term:
                nsfw_phrases.append(term)
            else:
                nsfw_words.add(term)
except FileNotFoundError:
    print("Warning: nsfw_terms.txt not found. Using empty list.")
    nsfw_phrases = []
    nsfw_words = set()

discrimination_patterns = [re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE) for phrase in discrimination_phrases]
nsfw_patterns = [re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE) for phrase in nsfw_phrases]

def clean_message_content(text):
    """Clean the message text."""
    return text.strip()

async def load_model():
    """Load the FastText language model once at startup."""
    model_path = "lid.176.ftz"
    try:
        global LANGUAGE_MODEL
        LANGUAGE_MODEL = fasttext.load_model(model_path)
        print(f"Successfully loaded FastText model from {model_path}")
    except Exception as e:
        print(f"Failed to load FastText model: {e}")
        raise

async def detect_language_ai(text):
    """Detect the language of the given text using FastText."""
    clean_text = clean_message_content(text)
    try:
        await load_model()
        prediction = LANGUAGE_MODEL.predict(clean_text)
        return prediction[0][0].replace("__label__", "")
    except Exception as e:
        print(f"Language detection error: {e}")
        return "en"

async def init_db():
    """Initialize the SQLite database for infractions and guild configs."""
    async with aiosqlite.connect('infractions.db') as conn:
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS infractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                guild_id INTEGER,
                points INTEGER,
                timestamp TEXT
            )
        ''')
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS guild_configs (
                guild_id INTEGER PRIMARY KEY,
                link_channel_id INTEGER
            )
        ''')
        await conn.commit()

async def log_action(action, member, reason):
    """Log moderation actions to a specified channel or console."""
    log_channel = bot.get_channel(LOG_CHANNEL_ID)
    if log_channel:
        await log_channel.send(f"{action.upper()} applied to {member.mention} ({member.id}) for: {reason}")
    else:
        print(f"{action.upper()} applied to {member.display_name} ({member.id}) for: {reason}")

async def check_openai_moderation(text):
    """Check message content using OpenAI's moderation API."""
    url = "https://api.openai.com/v1/moderations"
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
    data = {"input": text}
    async with ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            result = await response.json()
            return result.get("results", [{}])[0]

async def apply_punishment(member, action, duration=None):
    """Apply a punishment to a member based on infraction points."""
    try:
        if action == "warn":
            warning_message = PUNISHMENT_SYSTEM["points_thresholds"][3]["message"]
            await member.send(warning_message)
            await log_action("warn", member, "Infraction threshold reached")
        elif action == "mute":
            if duration:
                unmute_time = datetime.utcnow() + duration
                await member.timeout(until=unmute_time)
                await member.send(f"You have been muted until {unmute_time}.")
                await log_action("mute", member, f"Duration: {duration}")
        elif action == "kick":
            await member.kick(reason="Auto-moderation punishment")
            await log_action("kick", member, "Auto-moderation punishment")
        elif action == "temp_ban":
            await member.ban(reason="Temporary ban")
            await log_action("temp_ban", member, f"Duration: {duration}")
            if duration:
                await asyncio.sleep(duration.total_seconds())
                await member.guild.unban(member)
        elif action == "ban":
            await member.ban(reason="Permanent ban")
            await log_action("ban", member, "Permanent ban")
    except discord.Forbidden:
        print(f"Missing permissions to {action} {member.display_name}")
    except discord.HTTPException as e:
        print(f"HTTP error while applying {action}: {e}")
    except Exception as e:
        print(f"Unexpected error while applying {action}: {e}")

async def log_violation(member, violation_type, message):
    """Log a violation and apply punishment if thresholds are met."""
    points = PUNISHMENT_SYSTEM["violations"][violation_type]["points"]
    async with aiosqlite.connect('infractions.db') as conn:
        await conn.execute('''
            INSERT INTO infractions (user_id, guild_id, points, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (member.id, member.guild.id, points, datetime.utcnow().isoformat()))
        await conn.commit()
        
        cutoff_date = (datetime.utcnow() - timedelta(days=28)).isoformat()
        cursor = await conn.execute('''
            SELECT SUM(points) FROM infractions 
            WHERE user_id = ? AND guild_id = ? AND timestamp > ?
        ''', (member.id, member.guild.id, cutoff_date))
        total_points = (await cursor.fetchone())[0] or 0
    
    for threshold in sorted(PUNISHMENT_SYSTEM["points_thresholds"].keys(), reverse=True):
        if total_points >= threshold:
            punishment = PUNISHMENT_SYSTEM["points_thresholds"][threshold]
            await apply_punishment(member, action=punishment["action"], duration=punishment.get("duration"))
            break

async def handle(request):
    """Handle HTTP requests to keep the bot awake."""
    print(f"Received request from {request.remote}")
    return web.Response(text="Bot is awake", status=200, headers={"Content-Type": "text/plain"})

async def start_http_server():
    """Start an HTTP server to keep the bot running."""
    try:
        app = web.Application()
        app.router.add_get('/', handle)
        runner = web.AppRunner(app)
        await runner.setup()
        port = int(os.getenv("PORT", "8080"))
        site = web.TCPSite(runner, host='0.0.0.0', port=port)
        await site.start()
        print(f"✅ HTTP server running on port {port}")
    except Exception as e:
        print(f"❌ Failed to start HTTP server: {e}")

@tasks.loop(hours=24)
async def decay_points():
    """Remove infraction records older than 90 days."""
    async with aiosqlite.connect('infractions.db') as conn:
        cutoff_date = (datetime.utcnow() - timedelta(days=90)).isoformat()
        await conn.execute('DELETE FROM infractions WHERE timestamp < ?', (cutoff_date,))
        await conn.commit()

@tasks.loop(hours=1)
async def cleanup_message_counts():
    """Clean up message count tracking for inactive users."""
    now = datetime.utcnow()
    for guild_id in list(user_message_count.keys()):
        for user_id in list(user_message_count[guild_id].keys()):
            user_message_count[guild_id][user_id] = [
                t for t in user_message_count[guild_id][user_id]
                if (now - t).total_seconds() < 86400
            ]
            if not user_message_count[guild_id][user_id]:
                del user_message_count[guild_id][user_id]
        if not user_message_count[guild_id]:
            del user_message_count[guild_id]

@bot.event
async def setup_hook():
    """Setup tasks and load resources before the bot starts."""
    try:
        language("test")
        print("Luga model initialized")
    except Exception as e:
        print(f"Error initializing Luga: {e}")

@bot.event
async def on_ready():
    """Handle bot startup tasks."""
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    print("------")
    await init_db()
    await start_http_server()
    decay_points.start()
    cleanup_message_counts.start()
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s).")
    except Exception as e:
        print(f"Failed to sync commands: {e}")

@bot.tree.command(name="awake", description="Check if the bot is awake.")
async def awake(interaction: discord.Interaction):
    """Respond to the awake command."""
    await interaction.response.send_message("Awake. Never Sleep.")

@bot.command()
async def beta_classify(ctx):
    """Example command using language detection."""
    lang = await detect_language_ai(ctx.message.content)
    await ctx.send(f"LANG: {lang}")

@bot.command()
@commands.has_permissions(administrator=True)
async def set_link_channel(ctx, channel: discord.TextChannel):
    """Set the channel where links are allowed."""
    async with aiosqlite.connect('infractions.db') as conn:
        await conn.execute('''
            INSERT OR REPLACE INTO guild_configs (guild_id, link_channel_id)
            VALUES (?, ?)
        ''', (ctx.guild.id, channel.id))
        await conn.commit()
    await ctx.send(f"Link posting channel set to {channel.mention}")

@bot.command()
@commands.has_permissions(manage_messages=True)
async def infractions(ctx, member: discord.Member):
    """View a user's infraction history."""
    async with aiosqlite.connect('infractions.db') as conn:
        cursor = await conn.execute('''
            SELECT points, timestamp FROM infractions 
            WHERE user_id = ? AND guild_id = ?
            ORDER BY timestamp DESC
        ''', (member.id, ctx.guild.id))
        records = await cursor.fetchall()
        if records:
            response = f"Infractions for {member.display_name}:\n"
            for points, timestamp in records:
                response += f"- {points} points on {timestamp}\n"
            await ctx.send(response)
        else:
            await ctx.send(f"No infractions found for {member.display_name}.")

def get_domain(url):
    """Extract the domain from a URL."""
    if not url.startswith('http'):
        url = 'http://' + url
    parsed = urlparse(url)
    return parsed.netloc

def is_permitted_domain(domain):
    """Check if the domain is in the permitted list."""
    return any(domain == perm or domain.endswith('.' + perm) for perm in permitted_domains)

@bot.event
async def on_message(message):
    """Handle incoming messages and apply moderation rules."""
    if message.author.bot or not message.guild:
        return

    violations = set()
    channel_cfg = CHANNEL_CONFIG.get(message.channel.id, {})
    content_lower = message.content.lower()
    guild_id = message.guild.id
    user_id = message.author.id

    # Spam detection: message frequency
    if guild_id not in user_message_count:
        user_message_count[guild_id] = {}
    if user_id not in user_message_count[guild_id]:
        user_message_count[guild_id][user_id] = []
    
    now = datetime.utcnow()
    user_message_count[guild_id][user_id] = [
        t for t in user_message_count[guild_id][user_id]
        if (now - t).total_seconds() < SPAM_WINDOW
    ]
    user_message_count[guild_id][user_id].append(now)
    
    if len(user_message_count[guild_id][user_id]) > SPAM_LIMIT:
        violations.add("spam")
    elif len(message.content) > 850 or len(message.attachments) > 4:
        violations.add("spam")

    # Spam detection: repeated messages and excessive mentions
    if guild_id not in user_message_history:
        user_message_history[guild_id] = {}
    if user_id not in user_message_history[guild_id]:
        user_message_history[guild_id][user_id] = deque(maxlen=5)
    else:
        if message.content in user_message_history[guild_id][user_id]:
            violations.add("spam")
        user_message_history[guild_id][user_id].append(message.content)
    
    if len(message.mentions) > MENTION_LIMIT:
        violations.add("spam")

    # Language detection
    if "language" in channel_cfg:
        detected_lang = await detect_language_ai(message.content)
        if detected_lang not in channel_cfg["language"]:
            violations.add("foreign_language")

    # Advertising detection
    async with aiosqlite.connect('infractions.db') as conn:
        cursor = await conn.execute('SELECT link_channel_id FROM guild_configs WHERE guild_id = ?', (message.guild.id,))
        result = await cursor.fetchone()
        link_channel_id = result[0] if result else None

    if forbidden_text_pattern.search(content_lower):
        violations.add("advertising")

    if message.channel.id != link_channel_id:
        urls = url_pattern.findall(message.content)
        for url in urls:
            domain = get_domain(url[0] if isinstance(url, tuple) else url)
            if domain and not is_permitted_domain(domain):
                violations.add("advertising")
                break

    # Discrimination and NSFW detection
    words = re.findall(r'\b\w+\b', content_lower)
    if any(word in discrimination_words for word in words) or any(discrimination_automaton.iter(content_lower)):
        violations.add("discrimination")
    
    if any(word in nsfw_words for word in words) or any(nsfw_automaton.iter(content_lower)):
        violations.add("nsfw")

    # Topic enforcement
    if "topics" in channel_cfg:
        if not any(topic in content_lower for topic in channel_cfg["topics"]):
            violations.add("off_topic")
    else:
        if any(term in content_lower for term in [
            "politics", "religion", "god", "allah", "jesus", "church", "mosque",
            "temple", "bible", "quran", "torah", "democrat", "republican", "liberal", "conservative"
        ]):
            violations.add("politics")

    # OpenAI moderation
    if not violations:
        try:
            moderation_result = await check_openai_moderation(message.content)
            if moderation_result.get("flagged", False):
                violations.add("openai_moderation")
        except Exception as e:
            print(f"OpenAI moderation error: {e}")

    if violations:
        try:
            await message.delete()
            await log_action("message_deleted", message.author, f"Violations: {', '.join(violations)}")
        except discord.Forbidden:
            print(f"Missing permissions to delete message in {message.channel.name}")
        except Exception as e:
            print(f"Error deleting message: {e}")

        for violation in violations:
            await log_violation(message.author, violation, message)
        
        warning_msg = f"{message.author.mention} Violation detected: {', '.join(violations)}"
        await message.channel.send(warning_msg, delete_after=10)

    await bot.process_commands(message)

bot.run(os.getenv("ADROIT_TOKEN"))
