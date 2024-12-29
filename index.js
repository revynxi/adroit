const logger = require('pino')();
logger.info("Starting bot...");

setInterval(() => {
  logger.info("Bot is still alive!");
}, 60000);

const { SlashCommandBuilder } = require('@discordjs/builders');
const { Client, GatewayIntentBits } = require('discord.js');
const { REST } = require('@discordjs/rest');
const { Routes } = require('discord-api-types/v9');

const express = require('express');
const keepAlive = require('./keep_alive');

const TOKEN = process.env.MUNIFICUS_TOKEN;
const CLIENT_ID = process.env.MUNIFICUS_CLIENT;

const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.MessageContent,
    ],
});

const rest = new REST({ version: '10' }).setToken(TOKEN);

const echoCommand = new SlashCommandBuilder()
    .setName('echo')
    .setDescription('Forces bot to repeat after you!')
    .addStringOption(option =>
        option.setName('text')
            .setDescription('Text that bot should repeat')
            .setRequired(true)
    );

const app = express();

app.get('/', (req, res) => {
    res.send("Hello. I am alive!");
});

app.listen(process.env.PORT || 3000, () => {
    console.log('Server is running...');
});

client.once('ready', async () => {
    try {
        await rest.put(Routes.applicationCommands(CLIENT_ID), { body: [helloCommand.toJSON(), echoCommand.toJSON()] });
        console.log('Slash commands are successfuly registered!');
    } catch (error) {
        console.error('Failure while registering slash commands:');
        console.error(error);
    }
});

client.on('interactionCreate', async interaction => {
    if (!interaction.isChatInputCommand()) return;

    if (interaction.commandName === 'echo') {
        const text = interaction.options.getString('text');
        await interaction.reply(text);
    }
});

keepAlive(app);

client.login(TOKEN);
