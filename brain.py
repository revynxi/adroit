import os
import discord
from discord import app_commands
from discord.ext import commands

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

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
async def hello(interaction: discord.Interaction):
    await interaction.response.send_message(f"Awake. Never Sleep.")

bot.run(os.getenv("ADROIT_TOKEN"))
