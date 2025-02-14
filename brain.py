from datetime import datetime, timedelta
from aiohttp import web, ClientSession
import asyncio
import os
import re
import discord
import fasttext
import sqlite3
from discord.ext import commands
from dotenv import load_dotenv

def clean_message_content(text):
    text = re.sub(r'<@!?\d+>|https?://\S+', '', text)  
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text) 
    return text[:850]  
    
async def detect_language_ai(text):
    clean_text = clean_message_content(text)
    model = fasttext.load_model("lid.176.bin")
    lang = model.predict(clean_text)[0][0].replace("__label__", "")
    return lang
    
async def handle(request):
    return web.Response(
        text="Bot is awake",
        status=200,
        headers={"Content-Type": "text/plain"}
    )

async def start_http_server():
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

load_dotenv()

def init_db():
    with sqlite3.connect('infractions.db') as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS infractions
                     (user_id INTEGER, guild_id INTEGER, points INTEGER, timestamp TEXT)''')

init_db()

ACTIVE_MUTES = {}

def save_mutes():
    conn = sqlite3.connect('mutes.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS mutes
                 (guild_id INTEGER, user_id INTEGER, unmute_time TEXT)''')
    c.execute('DELETE FROM mutes')
    for guild_id, guild_mutes in ACTIVE_MUTES.items():
        for user_id, unmute_time in guild_mutes.items():
            c.execute('INSERT INTO mutes VALUES (?, ?, ?)',
                     (guild_id, user_id, unmute_time.isoformat()))
    conn.commit()
    conn.close()

def load_mutes():
    ACTIVE_MUTES = {}
    conn = sqlite3.connect('mutes.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS mutes
                 (guild_id INTEGER, user_id INTEGER, unmute_time TEXT)''')
    c.execute('SELECT * FROM mutes')
    for row in c.fetchall():
        guild_id, user_id, unmute_time = row
        if guild_id not in ACTIVE_MUTES:
            ACTIVE_MUTES[guild_id] = {}
        ACTIVE_MUTES[guild_id][user_id] = datetime.fromisoformat(unmute_time)
    conn.close()
    return ACTIVE_MUTES

intents = discord.Intents.default()
intents.members = True
intents.message_content = True
bot = commands.Bot(command_prefix=">>", intents=intents)
    
@bot.event
async def setup_hook():
    try:
        await load_model()
    except Exception as e:
        print(f"Error during setup_hook: {e}")

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
        "foreign_language": {"points": 2}
    }
}

user_message_count = {}
last_message_times = {}

async def check_openai_moderation(text):
    url = "https://api.openai.com/v1/moderations"
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
    data = {"input": text}

    async with ClientSession() as session: 
        async with session.post(url, headers=headers, json=data) as response:
            result = await response.json()
            return result.get("results", [{}])[0]

async def apply_punishment(member, action, duration=None):
    try:
        if action == "mute":
            role = discord.utils.get(member.guild.roles, name="『Arrested』") or \
                   await member.guild.create_role(name="『Arrested』", color=discord.Color.dark_red())
            
            for channel in member.guild.text_channels:
                await channel.set_permissions(role, send_messages=False)
                
            await member.add_roles(role)
            if duration:
                await asyncio.sleep(duration.total_seconds())
                await member.remove_roles(role)
                
        elif action == "temp_ban":
            await member.ban(reason="Temporary ban")
            if duration:
                await asyncio.sleep(duration.total_seconds())
                await member.guild.unban(member)
                
        elif action == "ban":
            await member.ban(reason="Permanent ban")
            
    except Exception as e:
        print(f"Punishment error: {e}")
            
async def log_violation(member, violation_type, message):
    with sqlite3.connect('infractions.db') as conn:
        cursor = conn.cursor()
        
        points = PUNISHMENT_SYSTEM["violations"][violation_type]["points"]
        cursor.execute('''
            INSERT INTO infractions (user_id, guild_id, points, timestamp)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, guild_id) DO UPDATE SET
                points = points + excluded.points,
                timestamp = excluded.timestamp
        ''', (member.id, member.guild.id, points, datetime.utcnow().isoformat()))
        
        total_points = cursor.execute('SELECT SUM(points) FROM infractions WHERE user_id=?', 
                                    (member.id,)).fetchone()[0]
        
    for threshold in sorted(PUNISHMENT_SYSTEM["points_thresholds"].keys(), reverse=True):
        if total_points >= threshold:
            punishment = PUNISHMENT_SYSTEM["points_thresholds"][threshold]
            await apply_punishment(member, **punishment)
            break

async def decay_points():
    while True:
        await asyncio.sleep(86400) 
        with sqlite3.connect('infractions.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM infractions
                WHERE datetime(timestamp) < datetime('now', '-28 days')
            ''')
            conn.commit()

    
async def cleanup_message_counts():
    while True:
        await asyncio.sleep(3600)  
        now = datetime.utcnow()
        to_remove = [uid for uid, t in last_message_times.items() 
                     if (now - t).total_seconds() > 86400] 
        for uid in to_remove:
            del user_message_count[uid]
            del last_message_times[uid]

@bot.event
async def on_ready():
    import gc

    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    print("------")

    await start_http_server()
    
    global ACTIVE_MUTES
    ACTIVE_MUTES = load_mutes()
    
    bot.loop.create_task(check_mutes_loop())
    bot.loop.create_task(cleanup_message_counts())
    bot.loop.create_task(start_http_server())
    bot.loop.create_task(decay_points())
    gc.collect()
    
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s).")
    except Exception as e:
        print(f"Failed to sync commands: {e}")

async def check_mutes_loop():
    await bot.wait_until_ready()
    while not bot.is_closed():
        await check_active_mutes()
        await asyncio.sleep(60)

async def check_active_mutes():
    current_time = datetime.utcnow()
    
    for guild in bot.guilds:
        guild_mutes = ACTIVE_MUTES.get(guild.id, {})
        to_remove = []
        
        for user_id, unmute_time in guild_mutes.items():
            if current_time >= unmute_time:
                member = guild.get_member(user_id)
                if member:
                    muted_role = discord.utils.get(guild.roles, name="『Arrested』")
                    if muted_role and muted_role in member.roles:
                        await member.remove_roles(muted_role)
                        print(f"Unmuted {member.display_name} in {guild.name}")
                to_remove.append(user_id)
                
        for user_id in to_remove:
            del guild_mutes[user_id]
        ACTIVE_MUTES[guild.id] = guild_mutes
        
    save_mutes()

@bot.tree.command(name="awake", description="Hey, Adroit, are you awake?")
async def awake(interaction: discord.Interaction):
    await interaction.response.send_message(f"Awake. Never Sleep.")

@bot.command()
async def classify(ctx, *, text):
    try:
        if LANGUAGE_PIPELINE is None:
            await load_model()
        result = await asyncio.to_thread(LANGUAGE_PIPELINE, text)
        await ctx.send(f"Result: {result}")
    except Exception as e:
        print(f"Classification error: {e}")
        await ctx.send("Error processing request.")

@bot.event
async def on_message(message):
    if message.author.bot or not message.guild:
        return
    
    violations = set()
    channel_cfg = CHANNEL_CONFIG.get(message.channel.id, {})

    if "language" in channel_cfg:
        detected_lang = await detect_language_ai(message.content)
        if detected_lang not in channel_cfg["language"]:
            violations.add("foreign_language")

    if len(message.content) > 850 or len(message.attachments) > 4:
        violations.add("spam")

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

    if violations:
        await message.delete()
        for violation in violations:
            await log_violation(message.author, violation, message.content)
        
        warning_msg = f"{message.author.mention} Violation detected: {', '.join(violations)}"
        await message.channel.send(warning_msg, delete_after=10)

    await bot.process_commands(message)

bot.run(os.getenv("ADROIT_TOKEN"))
