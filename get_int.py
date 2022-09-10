import nltk
from vectorizer import BOT_CONFIG
import re

def clean_text(text):
    """чистит текст"""
    c_text = ''
    for letter in re.findall(r'\w', text):
        c_text = c_text + letter
    return c_text

def mistakes(s1, s2):
    """считает расстояние Левенштейна"""
    return nltk.edit_distance(s1, s2) / ((len(s1) + len(s2)) / 2) < 0.4


def get_intent(question):
    """Сравнивает сообщение пользователя со словарем и выдает ключ совпадения"""
    for intento in BOT_CONFIG['intents']:
        for example in BOT_CONFIG['intents'][intento]['examples']:
            if mistakes(clean_text(question), clean_text(example)):
                return intento