const { SlashCommandBuilder } = require('@discordjs/builders');
const { Client, GatewayIntentBits } = require('discord.js');
const { REST } = require('@discordjs/rest');
const { Routes } = require('discord-api-types/v9');

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

client.once('ready', async () => {
    try {
        await rest.put(Routes.applicationCommands(CLIENT_ID), { body: [helloCommand.toJSON(), echoCommand.toJSON()] });
        console.log('Slah commands are successfuly registered!');
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

client.login(TOKEN);
