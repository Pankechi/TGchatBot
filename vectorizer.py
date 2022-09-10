from bot_config import BOT_CONFIG
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

"""Собирает массив данных из конфига для передачи в векторайзер"""
x = []
y = []

for intent in BOT_CONFIG['intents']:
    if 'examples' in BOT_CONFIG['intents'][intent]:
        x += BOT_CONFIG['intents'][intent]['examples']
        y += [intent for i in
              range(len(BOT_CONFIG['intents'][intent]['examples']))]

vectorizer = CountVectorizer(ngram_range=(1, 3))#переменная на основе CV
vectorizer.fit(x)#передаем массив х
x_vect = vectorizer.transform(x)

"""Создаем классифайер и передаем ему векторизированный массив"""
sgd = SGDClassifier()
sgd.fit(x_vect, y)


def intent_by_vectorizer(text):
    """Функция получает текст от пользователя и предсказывает тематику
    сообщения с помощью классификатора. Возвращает тему по данным конфига"""
    return sgd.predict(vectorizer.transform([text]))[0]
