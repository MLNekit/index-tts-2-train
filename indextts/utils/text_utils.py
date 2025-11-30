import re

from textstat import textstat


# --- Новая функция для обнаружения кириллицы ---
def contains_russian(text):
    """Проверяет, содержит ли текст символы кириллицы."""
    # Регулярное выражение для основных символов кириллицы
    if re.search(r'[а-яА-ЯёЁ]', text):
        return True
    return False

# --- Расширенная функция для обнаружения китайского ---
def contains_chinese(text):
    """Проверяет, содержит ли текст китайские символы."""
    # \u4e00-\u9fff - основной диапазон CJK Unified Ideographs
    # 0-9 - цифры
    if re.search(r'[\u4e00-\u9fff0-9]', text):
        return True
    return False


def count_russian_syllables(text):
    """
    Упрощенный подсчет слогов для русского текста по количеству гласных.
    (Для точного подсчета лучше использовать стороннюю библиотеку)
    """
    vowels = 'ауоыиэяюёе'
    # Считаем все гласные в тексте
    count = sum(text.lower().count(v) for v in vowels)
    # Если слогов 0, ставим минимум 1
    return max(1, count)


def get_text_syllable_num(text):
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
    number_char_pattern = re.compile(r'[0-9]')
    
    # --- НОВЫЙ БЛОК ДЛЯ РУССКОГО ЯЗЫКА ---
    if contains_russian(text):
        # Если это русский текст, используем специальную функцию подсчета слогов
        return count_russian_syllables(text)
    # --- КОНЕЦ НОВОГО БЛОКА ---
    
    syllable_num = 0
    tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|[0-9]+', text)
    
    if contains_chinese(text):
        for token in tokens:
            if chinese_char_pattern.search(token) or number_char_pattern.search(token):
                syllable_num += len(token) # 1 символ = 1 "слог"/единица
            else:
                # Если среди китайских токенов есть латиница (английский), считаем слоги textstat
                syllable_num += textstat.syllable_count(token)
    else:
        # Считаем слоги textstat (для английского/латиницы)
        syllable_num = textstat.syllable_count(text)

    return syllable_num


def get_text_tts_dur(text):
    min_speed = 3  
    max_speed = 5.50

    # 1. Определяем коэффициент ratio в зависимости от языка
    ratio = 1.0 
    
    if contains_chinese(text):
        ratio = 0.8517 # Коэффициент для китайского
    elif contains_russian(text):
        # --- НОВЫЙ КОЭФФИЦИЕНТ ДЛЯ РУССКОГО ЯЗЫКА ---
        # Оставим 1.0, как и для латиницы, но это можно настроить
        ratio = 1.0 
        # --- КОНЕЦ НОВОГО БЛОКА ---

    syllable_num = get_text_syllable_num(text)
    max_dur = syllable_num * ratio / max_speed
    min_dur = syllable_num * ratio / min_speed

    return max_dur, min_dur
