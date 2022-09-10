import random
from vectorizer import intent_by_vectorizer, BOT_CONFIG
from get_int import get_intent
from bot_token import bot
import telebot

def generate_response(text):
    intent = get_intent(
        text)  # 1. попытаться понять намерение сравнением по Левинштейну

    if intent is None:
        intent = intent_by_vectorizer(
            text)  # 2. попытаться понять намерение с помощью ML-модели

    return random.choice(BOT_CONFIG['intents'][intent]['responses'])



@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, generate_response(message.text))

if __name__ == "__main__":
    bot.polling()
