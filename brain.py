from flask import Flask
from threading import Thread
from datetime import datetime, timedelta
import asyncio
import aiohttp
import os
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from langdetect import detect


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

ALLOWED_LANGUAGES = ["en"]
DESIGNATED_TOPICS = {
    1322517478365990984: ["politics"]
}
RESTRICTED_TOPICS = ["religion", "politics"]
FOREIGN_CHANNELS = {
    1321499824926888049: ["fr"],
    1122525009102000269: ["de"],
    1122523546355245126: ["ru"],
    1122524817904635904: ["cn"],
    1242768362237595749: ["es"]
}
PUNISHMENTS = {
    "discrimination": {"action": "mute", "duration": timedelta(minutes=15), "severity": 5},
    "spam": {"action": "mute", "duration": timedelta(minutes=20), "severity": 3},
    "nsfw": {"action": "mute", "duration": timedelta(minutes=30), "severity": 7},
    "tos_violation": {"action": "mute", "duration": timedelta(minutes=30), "severity": 8},
    "off_topic": {"action": "mute", "duration": timedelta(minutes=10), "severity": 2},
    "restricted_topic": {"action": "mute", "duration": timedelta(minutes=15), "severity": 4},
    "advertising": {"action": "mute", "duration": timedelta(hours=12), "severity": 6},
    "foreign_language": {"action": "mute", "duration": timedelta(minutes=5), "severity": 1}
}


user_message_count = {}


async def check_openai_moderation(text):
    url = "https://api.openai.com/v1/moderations"
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
    data = {"input": text}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            result = await response.json()
            return result.get("results", [{}])[0].get("categories", {}) 

async def enforce_punishment(member, action, duration=None):
    try:
        if action == "mute":
            muted_role = discord.utils.get(member.guild.roles, name="『Arrested』")
            if not muted_role:
                muted_role = await member.guild.create_role(name="Muted")
                for channel in member.guild.channels:
                    await channel.set_permissions(muted_role, send_messages=False)
            await member.add_roles(muted_role)
            if duration:
                await asyncio.sleep(duration.total_seconds())
                await member.remove_roles(muted_role)
        elif action == "kick":
            await member.kick()
        elif action == "ban":
            await member.ban()
    except Exception as e:
        print(f"Failed to enforce punishment: {e}") 


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
        
    violations = []
    punishment_to_apply = None

    channel_id = message.channel.id
    allowed_languages = CHANNEL_LANGUAGES.get(channel_id, ["en"])
    if allowed_languages != "any":
        try:
            lang = detect(message.content)
            if lang not in allowed_languages:
                violations.append("foreign_language")
        except:
            pass

    content_lower = message.content.lower()
    if channel_id not in DESIGNATED_TOPICS_CHANNELS:
        if any(topic in content_lower for topic in RESTRICTED_TOPICS):
            violations.append("restricted_topic")
    else:
        allowed_topics = DESIGNATED_TOPICS_CHANNELS[channel_id]
        if not any(topic in content_lower for topic in allowed_topics):
            violations.append("off_topic")
    
    user_id = message.author.id
    user_message_count[user_id] = user_message_count.get(user_id, 0) + 1
    if user_message_count[user_id] > 5: 
        await message.delete()
        await enforce_punishment(message.author, **PUNISHMENTS["spam"])
        return

    if message.channel.id in RESTRICTED_CHANNELS:
        content_lower = message.content.lower()
        banned_keywords = RESTRICTED_CHANNELS[message.channel.id]
        if any(keyword in content_lower for keyword in banned_keywords):
            await message.delete()
            try:
                await message.author.send(f"{message.author.mention}  mentioning religion or politics is not allowed.")
            except discord.errors.Forbidden:
                print(f"Could not send DM to {message.author.name}.")
            await enforce_punishment(message.author, action="mute", duration=timedelta(hours=1))
            return

    openai_categories = await check_openai_moderation(message.content)
    if openai_categories.get("sexual") or openai_categories.get("nsfw"):
        violations.append("nsfw")
    if openai_categories.get("hate"):
        violations.append("discrimination")
    if openai_categories.get("violence"):
        violations.append("tos_violation")

    await bot.process_commands(message)

    if violations:
        max_severity = max(PUNISHMENTS[violation]["severity"] for violation in violations)
        for violation in violations:
            if PUNISHMENTS[violation]["severity"] == max_severity:
                punishment_to_apply = PUNISHMENTS[violation]
                break

        await message.delete()
        await enforce_punishment(message.author, **punishment_to_apply)

        log_channel = discord.utils.get(message.guild.channels, name="『📄』staff-logs")
        if log_channel:
            await log_channel.send(
                f"{message.author.mention} violated rules: {', '.join(violations)}. "
                f"Action taken: {punishment_to_apply['action']}."
            )

    await bot.process_commands(message)

    
@bot.event
async def on_message_delete(message):
    await asyncio.sleep(10)
    user_id = message.author.id
    if user_id in user_message_count:
        del user_message_count[user_id]

Thread(target=run_flask).start()


bot.run(os.getenv("ADROIT_TOKEN"))
