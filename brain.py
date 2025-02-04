from flask import Flask
from threading import Thread
from datetime import datetime, timedelta
import asyncio
import aiohttp
import os
import re
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is now awake"

def run_flask():
    app.run(host='0.0.0.0', port=8080)

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=">>", intents=intents)

CHANNEL_LANGUAGES = {
    1243854715872084019: ["ru", "en"], 
    1321499824926888049: ["fr"],
    1122525009102000269: ["de"],
    1122523546355245126: ["ru"],
    1122524817904635904: ["zh-cn", "zh-tw"],  
    1242768362237595749: ["es"],
    1113377809440722974: ["en"],
    1322517478365990984: ["en"],
    1322517478365990984: ["en"],
    1113377810476716132: ["en"],
}

DESIGNATED_TOPICS_CHANNELS = {
    1322517478365990984: ["politics"]
}

RESTRICTED_TOPICS = ["religion", "politics"]

RESTRICTED_PATTERNS = {
    # "religion": re.compile(r"\b(god|jesus|allah|buddha|hindu|church|mosque|temple|pray)\b", re.I),
    "politics": re.compile(r"\b(protest|riot)\b", re.I),
    "conflict": re.compile(r"\b(terrorism)\b", re.I)
}

PUNISHMENTS = {
    "discrimination": {"action": "mute", "duration": timedelta(minutes=15), "severity": 5},
    "spam": {"action": "mute", "duration": timedelta(minutes=20), "severity": 3},
    "nsfw": {"action": "mute", "duration": timedelta(minutes=45), "severity": 7},
    "tos_violation": {"action": "mute", "duration": timedelta(hours=1), "severity": 8},
    "off_topic": {"action": "mute", "duration": timedelta(minutes=10), "severity": 2},
    "restricted_topic": {"action": "mute", "duration": timedelta(minutes=15), "severity": 4},
    "advertising": {"action": "mute", "duration": timedelta(minutes=30), "severity": 6},
    "foreign_language": {"action": "mute", "duration": timedelta(minutes=5), "severity": 1}
}

DISCRIMINATION_PATTERNS = [
    re.compile(r"\b(nigg(a|er)|chink|spic|kike|fag)\b", re.I),
    re.compile(r"\b(white power|black lives)\b", re.I)
]

user_message_count = {}
last_message_times = {}

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
                await asyncio.sleep(duration.total_seconds())
                await member.remove_roles(muted_role, reason="Mute duration expired")
                
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
        embed = discord.Embed(
            title="🚨 Moderation Action",
            description=f"**User:** {author.mention}\n"
                       f"**Action Taken:** {punishment['action'].title()}\n"
                       f"**Duration:** {punishment['duration'] if punishment['duration'] else 'Permanent'}",
            color=discord.Color.red() if punishment['severity'] >= 5 else discord.Color.orange(),
            timestamp=datetime.utcnow()
        )
        
        embed.add_field(
            name="Detected Violations",
            value=", ".join(violations),
            inline=False
        )
        embed.add_field(
            name="Message Content",
            value=f"```{message_content[:1000]}```",  
            inline=False
        )
        
        embed.set_footer(text=f"User ID: {author.id}")
        await log_channel.send(embed=embed)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    print("------")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s).")
    except Exception as e:
        print(f"Failed to sync commands: {e}")

@bot.tree.command(name="awake", description="Hey, Adroit, are you awake?")
async def awake(interaction: discord.Interaction):
    await interaction.response.send_message(f"Awake. Never Sleep.")

@bot.event
async def on_message(message):
    if message.author.bot:
        return
        
    violations = set()
    punishment_to_apply = None
    current_time = datetime.utcnow()

    channel_id = message.channel.id
    allowed_languages = CHANNEL_LANGUAGES.get(channel_id, ["en"]) 
    
    if allowed_languages != ["en"]:
        try:
            lang = detect(message.content)
            if lang not in allowed_languages:
                violations.add("foreign_language")
        except:
            pass

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
        if (now - last_message_times[user_id]).total_seconds() > 10:
            user_message_count[user_id] = 0

    user_message_count[user_id] = user_message_count.get(user_id, 0) + 1
    last_message_times[user_id] = now
    if user_message_count[user_id] > 5: 
        violations.append("spam")

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

    await bot.process_commands(message)

    if violations:
        try:
            max_severity = max(PUNISHMENTS[violation]["severity"] for violation in violations)
            punishment = next(
                (PUNISHMENTS[v] for v in violations if PUNISHMENTS[v]["severity"] == max_severity),
                None
            )

            if punishment:
                punishment_copy = punishment.copy()
                severity = punishment_copy.pop("severity", None)

                await message.delete()
                await asyncio.gather(
                    enforce_punishment(message.author, **punishment_copy),
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
                error_punishment if not punishment else punishment,
                message.author
            )

@bot.event
async def on_message_delete(message):
    await asyncio.sleep(15)
    user_id = message.author.id
    if user_id in user_message_count:
        del user_message_count[user_id]

Thread(target=run_flask).start()

bot.run(os.getenv("ADROIT_TOKEN"))
