from flask import Flask
from threading import Thread
from datetime import datetime, timedelta
from aiohttp import web
import asyncio
import os
import re
import discord
import json
import fasttext
import sqlite3
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from waitress import serve

async def detect_language_ai(text):
    clean_text = re.sub(r'<@!?\d+>|https?://\S+', '', text)[:512]
    model = fasttext.load_model("lid.176.bin")
    lang = model.predict(clean_text)[0][0].replace("__label__", "")
    return lang  
    
async def handle(request):
    return web.Response(text="Bot is awake")

def run_flask():
    app = web.Application()
    app.router.add_get('/', handle)
    web.run_app(app, host='0.0.0.0', port=8080)

load_dotenv()

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

CHANNEL_LANGUAGES = {
    1321499824926888049: ["fr"],
    1122525009102000269: ["de"],
    1122523546355245126: ["ru"],
    1122524817904635904: ["zh"],  
    1242768362237595749: ["es"],
    1113377809440722974: ["en"],
    1322517478365990984: ["en"],
    1113377810476716132: ["en"],
}

DESIGNATED_TOPICS_CHANNELS = {
    1322517478365990984: ["politics"]
}

RESTRICTED_TOPICS = ["religion", "politics"]

RESTRICTED_PATTERNS = {
    "politics": re.compile(r"\b(protest|riot)\b", re.I),
    "conflict": re.compile(r"\b(terrorism)\b", re.I)
}

PUNISHMENTS = {
    "discrimination": {"action": "mute", "duration": timedelta(minutes=15), "severity": 5, "aka": 'Discrimination'},
    "spam": {"action": "mute", "duration": timedelta(minutes=20), "severity": 3, "aka": 'Spam'},
    "nsfw": {"action": "mute", "duration": timedelta(minutes=45), "severity": 7, "aka": 'NSFW'},
    "tos_violation": {"action": "mute", "duration": timedelta(hours=1), "severity": 8, "aka": 'ToS Violation'},
    "off_topic": {"action": "mute", "duration": timedelta(minutes=10), "severity": 2, "aka": 'Off-topic'},
    "restricted_topic": {"action": "mute", "duration": timedelta(minutes=15), "severity": 4, "aka": 'Restricted topic'},
    "advertising": {"action": "mute", "duration": timedelta(minutes=30), "severity": 6, "aka": 'Advertising'},
    "foreign_language": {"action": "mute", "duration": timedelta(minutes=5), "severity": 1, "aka": 'Foreign language'}
}

DISCRIMINATION_PATTERNS = [
    re.compile(r"\b(nigg(a|er)|chink|spic|kike|fag)\b", re.I),
    re.compile(r"\b(white power|black lives)\b", re.I)
]

user_message_count = {}
last_message_times = {}

async def detect_language_ai(text):
    clean_text = re.sub(r'<@!?\d+>|https?://\S+', '', text)[:512]
    model = fasttext.load_model("lid.176.bin")
    lang = model.predict(clean_text)[0][0].replace("__label__", "")
    return lang

async def check_openai_moderation(text):
    url = "https://api.openai.com/v1/moderations"
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
    data = {"input": text}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            result = await response.json()
            return result.get("results", [{}])[0]

async def enforce_punishment(member, action, duration=None):
    try:
        if action == "mute":
            muted_role = discord.utils.get(member.guild.roles, name="『Arrested』")
            if not muted_role:
                muted_role = await member.guild.create_role(
                    name="『Arrested』",
                    color=discord.Color.dark_red(),
                    reason="Automatic role creation for moderation"
                )
                for channel in member.guild.channels:
                    if isinstance(channel, discord.TextChannel):
                        await channel.set_permissions(
                            muted_role,
                            send_messages=False,
                            add_reactions=False,
                            create_public_threads=False,
                            send_messages_in_threads=False
                        )
            await member.add_roles(muted_role, reason="Automatic moderation action")
            if duration:
                unmute_time = datetime.utcnow() + duration
                if member.guild.id not in ACTIVE_MUTES:
                    ACTIVE_MUTES[member.guild.id] = {}
                ACTIVE_MUTES[member.guild.id][member.id] = unmute_time
                save_mutes()
                
        elif action == "ban":
            await member.ban(reason="Severe ToS violation", delete_message_days=1)
            
    except Exception as e:
        print(f"Failed to enforce punishment: {e}")
        log_channel = discord.utils.get(member.guild.channels, name="『📄』staff-logs")
        if log_channel:
            await log_channel.send(f"⚠️ Failed to punish {member.mention}: {str(e)}")

async def log_action(guild, violations, message_content, punishment, author):
    log_channel = discord.utils.get(guild.channels, name="『📄』staff-logs")
    if log_channel:
        aka_violations = [
            PUNISHMENTS.get(violation, {}).get("aka", violation.title())
            for violation in violations
        ]
        
        embed = discord.Embed(
            title="🚨 Moderation Action",
            description=f"**User:** {author.mention}\n"
                       f"**Action Taken:** {punishment['action'].title()}\n"
                       f"**Duration:** {str(punishment['duration']) if punishment.get('duration') else 'Permanent'}",
            color=discord.Color.red() if punishment.get('severity', 0) >=5 else discord.Color.orange(),
            timestamp=datetime.utcnow()
        )
        
        embed.add_field(
            name="Detected Violations",
            value=", ".join(aka_violations),
            inline=False
        )
        embed.add_field(
            name="Message Content",
            value=f"```{message_content[:1000]}```",
            inline=False
        )
        
        embed.set_footer(text=f"User ID: {author.id}")
        await log_channel.send(embed=embed)

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
    
    global ACTIVE_MUTES
    ACTIVE_MUTES = load_mutes()
    
    bot.loop.create_task(check_mutes_loop())
    bot.loop.create_task(cleanup_message_counts())
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
    if message.author.bot:
        return
        
    violations = set()
    punishment = None
    current_time = datetime.utcnow()

    channel_id = message.channel.id
    allowed_languages = CHANNEL_LANGUAGES.get(channel_id, ["any"])
    
    if allowed_languages != ["any"]:  
        lang = await detect_language_ai(message.content)
        if lang and lang not in allowed_languages:
            violations.add("foreign_language")

    content_lower = message.content.lower()
    if channel_id not in DESIGNATED_TOPICS_CHANNELS:
        for topic, pattern in RESTRICTED_PATTERNS.items():
            if pattern.search(message.content):
                violations.add("restricted_topic")
                break
    else:
        allowed_topics = DESIGNATED_TOPICS_CHANNELS[channel_id]
        if not any(topic in content_lower for topic in allowed_topics):
            violations.add("off_topic")

    user_id = message.author.id
    now = datetime.utcnow()
    if user_id in last_message_times:
        time_diff = (now - last_message_times[user_id]).total_seconds()
        if time_diff > 10:
            user_message_count[user_id] = 0

    user_message_count[user_id] = user_message_count.get(user_id, 0) + 1
    last_message_times[user_id] = now
    if user_message_count[user_id] > 5:
        violations.add("spam")

    for pattern in DISCRIMINATION_PATTERNS:
        if pattern.search(message.content):
            violations.add("discrimination")
            break
    
    moderation_result = await check_openai_moderation(message.content)
    if moderation_result.get("flagged"):
        categories = moderation_result.get("categories", {})
        if categories.get("sexual") or categories.get("nsfw"):
            violations.add("nsfw")
        if categories.get("hate"):
            violations.add("discrimination")
        if categories.get("violence"):
            violations.add("tos_violation")

    try:
        await bot.process_commands(message)
        if violations:
            max_severity = max(PUNISHMENTS[violation]["severity"] for violation in violations)
            punishment = next(
                (PUNISHMENTS[v] for v in violations if PUNISHMENTS[v]["severity"] == max_severity),
                None
            )

            if punishment:
                try:
                    await message.delete()
                except discord.NotFound:
                    pass
                
                duration = punishment.get('duration')
                
                await asyncio.gather(
                    enforce_punishment(message.author, **punishment),
                    log_action(
                        guild=message.guild,
                        violations=violations,
                        message_content=message.content,
                        punishment=punishment,
                        author=message.author
                    )
                )
    except Exception as e:
        error_punishment = {
            "action": "error",
            "duration": None,
            "severity": 0
        }    
        error_msg = f"❌ Error processing message from {message.author}: {str(e)}"
        await log_action(
            message.guild, 
            {"system_error"},
            error_msg,
            error_punishment,
            message.author
        )

@bot.event
async def on_message_delete(message):
    await asyncio.sleep(15)
    user_id = message.author.id
    if user_id in user_message_count:
        del user_message_count[user_id]

thread = Thread(target=run_flask, daemon=True)
thread.start()

bot.run(os.getenv("ADROIT_TOKEN"))
