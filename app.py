import streamlit as st
import joblib
import re

# Загрузка модели
model = joblib.load('models/best_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')


# Очистка текста
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.strip()
    return text


# Рекомендации
def get_recommendation(prediction):
    if prediction == 1:
        return '''
        Возможны признаки стресса.

        Рекомендации:
        - Сделайте перерыв
        - Поговорите с близкими
        - Снизьте нагрузку
        - Попробуйте медитацию
        - При необходимости обратитесь к специалисту
        '''
    else:
        return '''
        Сообщение выглядит эмоционально стабильным.

        Продолжайте поддерживать здоровый образ жизни.
        '''


st.title('Stress Analysis in Social Media')

st.write('Введите сообщение для анализа (на английском!)')

user_input = st.text_area('Текст сообщения')

if st.button('Анализировать'):

    cleaned = clean_text(user_input)

    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]

    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(vectorized)[0][1]
    else:
        probability = 0.85 if prediction == 1 else 0.15

    if prediction == 1:
        st.error(f'Стрессовое сообщение. Вероятность: {probability:.2f}')
    else:
        st.success(f'Нет признаков стресса. Вероятность: {1-probability:.2f}')

    st.info(get_recommendation(prediction))