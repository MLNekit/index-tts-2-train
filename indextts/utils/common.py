import os
import random
import re
from typing import List, Union

import torch
import torchaudio

MATPLOTLIB_FLAG = False


def load_audio(audiopath, sampling_rate):
    audio, sr = torchaudio.load(audiopath)
    # print(f"wave shape: {audio.shape}, sample_rate: {sr}")

    if audio.size(0) > 1:  # mix to mono
        audio = audio[0].unsqueeze(0)

    if sr != sampling_rate:
        try:
            audio = torchaudio.functional.resample(audio, sr, sampling_rate)
        except Exception as e:
            print(f"Warning: {audiopath}, wave shape: {audio.shape}, sample_rate: {sr}")
            return None
    # clip audio invalid values
    audio.clip_(-1, 1)
    return audio


def tokenize_by_CJK_char(line: str, do_upper_case=True) -> str:
    """
    Tokenize a line of text with CJK char, but preserves word integrity 
    for alphabetic languages (Cyrillic/Latin) to ensure long BPE tokens are possible.

    Example (before fix): "ПРИВЕТ" -> "П Р И В Е Т" (Fragmented)
    Example (after fix): "ПРИВЕТ" -> "ПРИВЕТ" (Word preserved)

    Args:
      line:
        The input text.

    Return:
      A new string tokenized by CJK char, with alphabetic words preserved.
    """
    # Паттерн для обнаружения и захвата CJK-символов
    CJK_RANGE_PATTERN_WITH_CAPTURE = (
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    )
    
    # 1. Разделяем строку по CJK-символам (CJK-символы остаются в списке из-за скобок)
    chars_or_words = re.split(CJK_RANGE_PATTERN_WITH_CAPTURE, line.strip())
    
    result_parts = []
    
    for part in chars_or_words:
        if not part.strip():
            continue
        
        processed_part = part.strip()

        # 2. Проверяем, является ли часть CJK-символом. 
        # Если да, оставляем ее как есть (она уже токенизирована по символам).
        if re.match(CJK_RANGE_PATTERN_WITH_CAPTURE, processed_part):
            # CJK символы не меняются, регистр не важен для CJK
            result_parts.append(processed_part)
        else:
            # 3. Если это не CJK (кириллица, латиница, знаки препинания)
            # Приводим к верхнему регистру, но НЕ фрагментируем (не вставляем пробелы внутри слова)
            if do_upper_case:
                processed_part = processed_part.upper()

            # Добавляем часть. Если это слово типа "ПРИВЕТ", оно добавится целиком.
            result_parts.append(processed_part)

    # 4. Объединяем части через пробел. BPE-токенизатор получит "ПРИВЕТ МИР" 
    # или "你 好 世 界 HELLO WORLD", и сможет выбрать длинный токен "ПРИВЕТ".
    return " ".join([p for p in result_parts if p])


# Новая вспомогательная функция для чистых алфавитных языков
def de_tokenized_by_alphabetic_char(line: str, do_lower_case=False) -> str:
    """
    Restores spaces for alphabetic languages (like English or Russian) 
    that were tokenized by SentencePiece.
    """
    if do_lower_case:
        line = line.lower()
        
    # sp_model.Decode уже заменил BPE-пробелы на ' ', но его нужно нормализовать.
    # split() и join(' ') восстанавливают правильные пробелы между словами.
    return ' '.join(line.split())


def de_tokenized_by_CJK_char(line: str, do_lower_case=False) -> str:
    """
    Смешанная логика. Сливает CJK-символы, но сохраняет пробелы вокруг 
    латиницы/кириллицы (если они не CJK).
    
    ПОПЫТКА ИСПОЛЬЗОВАТЬ СУЩЕСТВУЮЩУЮ ЛОГИКУ:
    Оставляем существующую логику только для CJK-текста. 
    Для русского/английского — вызываем новую функцию.
    """
    
    # 1. Проверяем наличие CJK-символов
    # (Используем паттерн из tokenize_by_CJK_char для обнаружения)
    # Используем паттерн без захвата, чтобы просто проверить наличие
    CJK_RANGE_PATTERN = (
        r"[\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF]"
    )
    
    has_cjk = re.search(CJK_RANGE_PATTERN, line)

    if not has_cjk:
        # Если CJK символов нет (т.е. это чистый русский или английский текст)
        # используем простую логику восстановления пробелов.
        return de_tokenized_by_alphabetic_char(line, do_lower_case=do_lower_case)

    # 2. Если есть CJK-символы, применяем оригинальную CJK-логику.
    # Это позволяет сохранить слияние CJK-символов.

    # replace english words in the line with placeholders
    english_word_pattern = re.compile(r"([A-Z]+(?:[\s-][A-Z-]+)*)", re.IGNORECASE)
    english_sents = english_word_pattern.findall(line)
    for i, sent in enumerate(english_sents):
        line = line.replace(sent, f"<sent_{i}>")

    words = line.split()
    # restore english sentences
    sent_placeholder_pattern = re.compile(r"^.*?(<sent_(\d+)>)")
    for i in range(len(words)):
        m = sent_placeholder_pattern.match(words[i])
        if m:
            # restore the english word
            placeholder_index = int(m.group(2))
            words[i] = words[i].replace(m.group(1), english_sents[placeholder_index])
            if do_lower_case:
                words[i] = words[i].lower()
                
    # ВНИМАНИЕ: Если русский текст токенизировался как отдельные слова, 
    # а не как CJK-токены, то здесь он будет слит! 
    # Это может быть проблемой для смешанного текста.
    return "".join(words)


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))
