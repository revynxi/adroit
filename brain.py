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

load_dotenv()

intents = discord.Intents.default()
intents.members = True
intents.message_content = True
bot = commands.Bot(command_prefix=">>", intents=intents)

LANGUAGE_MODEL = None
user_message_count = {}
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

RESTRICTED_PATTERNS = {
    "discrimination": re.compile(
        r"\b(nigg(a|er)|chink|spic|kike|fag|retard|tranny|"
        r"white\s+power|black\s+lives|all\s+lives\s+matter|"
        r"islamophobi(a|c)|anti[\s-]?semiti(sm|c))\b",
        re.IGNORECASE | re.VERBOSE
    ),
    "advertising": re.compile(
        r"(discord\.gg/|join\s+our|server\s+invite|"
        r"free\s+nitro|http(s)?://|www\.|\.com|\.net|\.org)",
        re.IGNORECASE | re.VERBOSE
    ),
    "nsfw": re.compile(
        r"\b(sex|porn|onlyfans|nsfw|dick|pussy|tits|anal|"
        r"masturbat(e|ion)|rape|pedo|underage)\b",
        re.IGNORECASE | re.VERBOSE
    )
}

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

def clean_message_content(text):
    """Clean message content by removing mentions and URLs."""
    text = re.sub(r'<@!?\d+>|https?://\S+', '', text)
    return text[:850]

async def load_model():
    """Load the FastText language model once at startup."""
    global LANGUAGE_MODEL
    if LANGUAGE_MODEL is None:
        LANGUAGE_MODEL = fasttext.load_model("lid.176.bin")

async def detect_language_ai(text):
    """Detect the language of the given text using FastText."""
    clean_text = clean_message_content(text)
    await load_model()  # Ensure model is loaded
    prediction = LANGUAGE_MODEL.predict(clean_text)
    return prediction[0][0].replace("__label__", "")

async def init_db():
    """Initialize the SQLite database for infractions."""
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
    to_remove = [uid for uid, times in user_message_count.items()
                 if all((now - t).total_seconds() > 86400 for t in times)]
    for uid in to_remove:
        del user_message_count[uid]

@bot.event
async def setup_hook():
    """Setup tasks and load resources before the bot starts."""
    await load_model()
    await init_db()

@bot.event
async def on_ready():
    """Handle bot startup tasks."""
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    print("------")
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
async def classify(ctx, *, text):
    """Classify the language of the provided text."""
    try:
        lang = await detect_language_ai(text)
        await ctx.send(f"Detected language: {lang}")
    except Exception as e:
        print(f"Classification error: {e}")
        await ctx.send("Error processing request.")

@bot.event
async def on_message(message):
    """Handle incoming messages and apply moderation rules."""
    if message.author.bot or not message.guild:
        return

    violations = set()
    channel_cfg = CHANNEL_CONFIG.get(message.channel.id, {})

    now = datetime.utcnow()
    user_id = message.author.id
    if user_id not in user_message_count:
        user_message_count[user_id] = []
    user_message_count[user_id] = [t for t in user_message_count[user_id] if (now - t).total_seconds() < SPAM_WINDOW]
    user_message_count[user_id].append(now)
    if len(user_message_count[user_id]) > SPAM_LIMIT:
        violations.add("spam")
    elif len(message.content) > 850 or len(message.attachments) > 4:
        violations.add("spam")

    if "language" in channel_cfg:
        detected_lang = await detect_language_ai(message.content)
        if detected_lang not in channel_cfg["language"]:
            violations.add("foreign_language")

    content_lower = message.content.lower()
    for pattern_type, pattern in RESTRICTED_PATTERNS.items():
        if pattern.search(content_lower):
            violations.add(pattern_type)

    if "topics" in channel_cfg:
        if not any(topic in content_lower for topic in channel_cfg["topics"]):
            violations.add("off_topic")
    else:
        if any(topic in content_lower for topic in ["politics"]):
            violations.add("politics")

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
