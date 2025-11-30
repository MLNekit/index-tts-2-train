# -*- coding: utf-8 -*-
import os
import traceback
import re
import unicodedata
from typing import List, Union, overload
import warnings
try:
    from num2words import num2words
except ImportError:
    warnings.warn("num2words is not installed. Russian number normalization will not work.", ImportWarning)
    # Заглушка, если num2words не установлен
    def num2words(num, lang, to):
        return str(num)

from indextts.utils.common import tokenize_by_CJK_char, de_tokenized_by_CJK_char
from sentencepiece import SentencePieceProcessor


class TextNormalizer:
    _HIRAGANA_PATTERN = re.compile(r"[\u3040-\u309f]")
    _KATAKANA_PATTERN = re.compile(r"[\u30a0-\u30ff\u31f0-\u31ff\uFF66-\uFF9F]")
    _JAPANESE_PUNCT = re.compile(r"[ー〜〝〞〟・]")
    _CYRILLIC_PATTERN = re.compile(r'[а-яА-ЯёЁ]')
    
    # Регулярные выражения для улучшения русской нормализации (добавлены/изменены)
    # 1. Стандартизация всех форм тире/дефисов в ' - '
    _ru_re_dashes = re.compile(r'[\u2013\u2014—–-]\s*')
    
    # 2. Стандартизация многоточия в ' ... '
    _re_ellipsis_1 = re.compile(r'\.\s*\.\s*\.')
    _re_ellipsis_2 = re.compile(r'\.{2,}')
    _re_ellipsis_3 = re.compile(r'…')
    
    # 3. Обеспечение пробела перед критической пунктуацией (.,?!:;) для токенов типа ' ?'
    _ru_re_punctuation_spacing = re.compile(r'\s*([.,?!:;])\s*')
    
    # 4. Удаление избыточных знаков препинания (!!! -> !, ??? -> ?)
    _re_multi_punct = re.compile(r'([?!])\1+')


    def __init__(self, preferred_language: str | None = None):
        self.zh_normalizer = None
        self.en_normalizer = None
        self.preferred_language = preferred_language.lower() if preferred_language else None
        self.char_rep_map = {
            "：": ",",
            "；": ",",
            ";": ",",
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "\n": " ",
            "·": "-",
            "、": ",",
            # Многоточие будет обрабатываться через regex _re_ellipsis_N
            "...": "…", 
            ",,,": "…",
            "，，，": "…",
            "……": "…",
            "“": "'",
            "”": "'",
            '"': "'",
            "‘": "'",
            "’": "'",
            "（": "'",
            "）": "'",
            "(": "'",
            ")": "'",
            "《": "'",
            "》": "'",
            "【": "'",
            "】": "'",
            "[": "'",
            "]": "'",
            "—": "-", # Будет переопределено в ru_char_rep_map, но оставлено для базовой чистки
            "～": "-",
            "~": "-",
            "「": "'",
            "」": "'",
            ":": ",",
        }
        self.zh_char_rep_map = {
            "$": ".",
            **self.char_rep_map,
        }
        self.jp_char_rep_map = {
            **self.char_rep_map,
        }
        # Для русского языка используем regex для точной настройки пробелов,
        # поэтому здесь оставляем только замену широких символов на узкие, 
        # но основные знаки пунктуации удаляем, чтобы их обработал _ru_re_punctuation_spacing
        self.ru_char_rep_map = {
            "：": ",",
            "；": ",",
            "，": ",",
            "！": "!",
            "？": "?",
            "“": "'",
            "”": "'",
            "‘": "'",
            "’": "'",
            "（": "'",
            "）": "'",
            "《": "'",
            "》": "'",
            "【": "'",
            "】": "'",
            "「": "'",
            "」": "'",
            "~": "-",
            "～": "-",
            "\n": " ",
            "·": "-",
            # Убираем стандартное тире из карты, чтобы его обработал _ru_re_dashes regex
            #"—": "-", 
            # ... остальные символы, которые не конфликтуют с regex для пробелов
        }

        self._base_cleanup_pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map.keys()))
        self._zh_cleanup_pattern = re.compile("|".join(re.escape(p) for p in self.zh_char_rep_map.keys()))
        self._jp_cleanup_pattern = re.compile("|".join(re.escape(p) for p in self.jp_char_rep_map.keys()))
        # Создаем карту только для символов, которые НЕ обрабатываются через _ru_re_punctuation_spacing
        self._ru_cleanup_pattern = re.compile("|".join(re.escape(p) for p in self.ru_char_rep_map.keys()))

    def match_email(self, email):
        # 正则表达式匹配邮箱格式：数字英文@数字英文.英文
        pattern = r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$"
        return re.match(pattern, email) is not None

    PINYIN_TONE_PATTERN = r"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])"
    """
    匹配拼音声调格式：pinyin+数字，声调1-5，5表示轻声
    例如：xuan4, jve2, ying1, zhong4, shang5
    不匹配：beta1, voice2
    """
    NAME_PATTERN = r"[\u4e00-\u9fff]+(?:[-·—][\u4e00-\u9fff]+){1,2}"
    """
    匹配人名，格式：中文·中文，中文·中文-中文
    例如：克里斯托弗·诺兰，约瑟夫·高登-莱维特
    """

    # 匹配常见英语缩写 's，仅用于替换为 is，不匹配所有 's
    ENGLISH_CONTRACTION_PATTERN = r"(what|where|who|which|how|t?here|it|s?he|that|this)'s"


    def use_chinese(self, s):
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", s))
        has_alpha = bool(re.search(r"[a-zA-Z]", s))
        is_email = self.match_email(s)
        if has_chinese or not has_alpha or is_email:
            return True

        has_pinyin = bool(re.search(TextNormalizer.PINYIN_TONE_PATTERN, s, re.IGNORECASE))
        return has_pinyin
    
    def is_russian(self, text: str) -> bool:
        """Проверяет, содержит ли текст символы кириллицы (русский)."""
        if self._CYRILLIC_PATTERN.search(text):
            # Простая проверка на наличие кириллицы.
            return True
        return False

    def load(self):
        # print(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        # sys.path.append(model_dir)
        import platform
        if self.zh_normalizer is not None and self.en_normalizer is not None:
            return
        if platform.system() != "Linux":  # Mac and Windows
            from wetext import Normalizer

            self.zh_normalizer = Normalizer(remove_erhua=False, lang="zh", operator="tn")
            self.en_normalizer = Normalizer(lang="en", operator="tn")
        else:
            from tn.chinese.normalizer import Normalizer as NormalizerZh
            from tn.english.normalizer import Normalizer as NormalizerEn
            # use new cache dir for build tagger rules with disable remove_interjections and remove_erhua
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tagger_cache")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                with open(os.path.join(cache_dir, ".gitignore"), "w") as f:
                    f.write("*\n")
            self.zh_normalizer = NormalizerZh(
                cache_dir=cache_dir, remove_interjections=False, remove_erhua=False, overwrite_cache=False
            )
            self.en_normalizer = NormalizerEn(overwrite_cache=False)

    def _ensure_normalizers(self) -> None:
        if self.zh_normalizer is None or self.en_normalizer is None:
            self.load()

    def _basic_cleanup(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        return self._base_cleanup_pattern.sub(lambda x: self.char_rep_map[x.group()], text)

    def is_japanese(self, text: str) -> bool:
        if self._HIRAGANA_PATTERN.search(text) or self._KATAKANA_PATTERN.search(text):
            return True
        if self._JAPANESE_PUNCT.search(text):
            return True
        # iteration marks and katakana-hiragana prolonged sound mark
        if any(ch in text for ch in ("々", "〆", "ゝ", "ゞ", "ゝ", "ゞ", "ー")):
            return True
        return False

    def normalize_japanese(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        text = re.sub(r"^\s*(?:speaker|spk)\s*\d+\s*[:：]\s*", "", text, flags=re.IGNORECASE)
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text)
        text = self._jp_cleanup_pattern.sub(lambda x: self.jp_char_rep_map[x.group()], text)
        return text.strip()

    def normalize_russian(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        
        # 1. Приведение к нижнему регистру и очистка Unicode
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()
        
        # 2. Обработка сокращений (можно расширить)
        text = text.replace('г.', 'году')
        text = text.replace('тыс.', 'тысяч')
        text = text.replace('руб.', 'рублей')
        text = text.replace('ул.', 'улица')
        text = text.replace('д.', 'дом')
        
        # 3. Нормализация чисел с num2words (важно для TTS)
        def replace_num(match):
            num_str = match.group(0)
            # Пробуем целые числа
            try:
                # num2words имеет проблемы с русским языком (не всегда склоняет правильно),
                # но для базовой нормализации чисел это лучший вариант.
                return num2words(int(num_str), lang='ru', to='cardinal')
            except ValueError:
                return num_str
        
        # Замена чисел, которые не являются частями слов/дат (упрощенно)
        # Улучшенное regex для обработки чисел
        text = re.sub(r'(?<![a-zа-яё])\d+(?![a-zа-яё])', replace_num, text)

        # 4. Обработка пунктуации (КРИТИЧЕСКАЯ ЧАСТЬ для соответствия токенам)

        # Стандартизация многоточия в ' ... '
        text = self._re_ellipsis_1.sub(' ... ', text)
        text = self._re_ellipsis_2.sub(' ... ', text)
        text = self._re_ellipsis_3.sub(' ... ', text)
        
        # Стандартизация тире/дефисов в ' - '
        text = self._ru_re_dashes.sub(' - ', text)

        # Удаление избыточных знаков препинания (!!! -> !, ??? -> ?)
        text = self._re_multi_punct.sub(r'\1', text)
        
        # Обработка широких символов (из ru_char_rep_map)
        text = self._ru_cleanup_pattern.sub(lambda x: self.ru_char_rep_map[x.group()], text)

        # Обеспечиваем один пробел перед критической пунктуацией (.,?!:;).
        # Это преобразует 'слово.' в 'слово .' и позволяет токенизатору увидеть ' .'
        text = self._ru_re_punctuation_spacing.sub(r' \1', text)

        # Финальная очистка лишних пробелов
        text = re.sub(r"\s+", " ", text).strip()
        
        return text.strip()

    def normalize(self, text: str, language: str | None = None) -> str:
        if text is None:
            return ""
        lang = language.strip().lower() if language else self.preferred_language
        if lang is None:
            if self.is_japanese(text):
                lang = "ja"
            elif self.is_russian(text): # НОВОЕ: Проверяем кириллицу
                lang = "ru"
            elif self.use_chinese(text):
                lang = "zh"
            else:
                lang = "en"

        if lang == "ja":
            return self.normalize_japanese(text)
        
        if lang == "ru":
            return self.normalize_russian(text)

        if lang == "zh":
            self._ensure_normalizers()
            text = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
            replaced_text, pinyin_list = self.save_pinyin_tones(text.rstrip())
            replaced_text, original_name_list = self.save_names(replaced_text)
            try:
                result = self.zh_normalizer.normalize(replaced_text)
            except Exception:
                result = replaced_text
                print(traceback.format_exc())
            result = self.restore_names(result, original_name_list)
            result = self.restore_pinyin_tones(result, pinyin_list)
            result = self._zh_cleanup_pattern.sub(lambda x: self.zh_char_rep_map[x.group()], result)
            return result

        if lang == "en":
            self._ensure_normalizers()
            text_processed = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
            try:
                result = self.en_normalizer.normalize(text_processed)
            except Exception:
                print(traceback.format_exc())
                return self._basic_cleanup(text_processed)
            result = self._base_cleanup_pattern.sub(lambda x: self.char_rep_map[x.group()], result)
            return result

        return self._basic_cleanup(text)

    def correct_pinyin(self, pinyin: str):
        """
        将 jqx 的韵母为 u/ü 的拼音转换为 v
        如：ju -> jv , que -> qve, xün -> xvn
        """
        if pinyin[0] not in "jqxJQX":
            return pinyin
        # 匹配 jqx 的韵母为 u/ü 的拼音
        pattern = r"([jqx])[uü](n|e|an)*(\d)"
        repl = r"\g<1>v\g<2>\g<3>"
        pinyin = re.sub(pattern, repl, pinyin, flags=re.IGNORECASE)
        return pinyin.upper()

    def save_names(self, original_text):
        """
        替换人名为占位符 <n_a>、 <n_b>, ...
        例如：克里斯托弗·诺兰 -> <n_a>
        """
        # 人名
        name_pattern = re.compile(TextNormalizer.NAME_PATTERN, re.IGNORECASE)
        original_name_list = re.findall(name_pattern, original_text)
        if len(original_name_list) == 0:
            return (original_text, None)
        original_name_list = list(set("".join(n) for n in original_name_list))
        transformed_text = original_text
        # 替换占位符 <n_a>、 <n_b>, ...
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(name, f"<n_{number}>")

        return transformed_text, original_name_list

    def restore_names(self, normalized_text, original_name_list):
        """
        恢复人名为原来的文字
        例如：<n_a> -> original_name_list[0]
        """
        if not original_name_list or len(original_name_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # 替换为占位符 <n_a>、 <n_b>, ...
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(f"<n_{number}>", name)
        return transformed_text

    def save_pinyin_tones(self, original_text):
        """
        替换拼音声调为占位符 <pinyin_a>, <pinyin_b>, ...
        例如：xuan4 -> <pinyin_a>
        """
        # 声母韵母+声调数字
        origin_pinyin_pattern = re.compile(TextNormalizer.PINYIN_TONE_PATTERN, re.IGNORECASE)
        original_pinyin_list = re.findall(origin_pinyin_pattern, original_text)
        if len(original_pinyin_list) == 0:
            return (original_text, None)
        original_pinyin_list = list(set("".join(p) for p in original_pinyin_list))
        transformed_text = original_text
        # 替换为占位符 <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(pinyin, f"<pinyin_{number}>")

        # print("original_text: ", original_text)
        # print("transformed_text: ", transformed_text)
        return transformed_text, original_pinyin_list

    def restore_pinyin_tones(self, normalized_text, original_pinyin_list):
        """
        恢复拼音中的音调数字（1-5）为原来的拼音
        例如：<pinyin_a> -> original_pinyin_list[0]
        """
        if not original_pinyin_list or len(original_pinyin_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # 替换占位符 <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            pinyin = self.correct_pinyin(pinyin)
            transformed_text = transformed_text.replace(f"<pinyin_{number}>", pinyin)
        # print("normalized_text: ", normalized_text)
        # print("transformed_text: ", transformed_text)
        return transformed_text


class TextTokenizer:
    def __init__(self, vocab_file: str, normalizer: TextNormalizer = None):
        self.vocab_file = vocab_file
        self.normalizer = normalizer

        if self.vocab_file is None:
            raise ValueError("vocab_file is None")
        if not os.path.exists(self.vocab_file):
            raise ValueError(f"vocab_file {self.vocab_file} does not exist")
        if self.normalizer:
            self.normalizer.load()
        # 加载词表
        self.sp_model = SentencePieceProcessor(model_file=self.vocab_file)

        self.pre_tokenizers = [
            # 预处理器
            tokenize_by_CJK_char,
        ]

    @property
    def vocab_size(self):
        return self.sp_model.GetPieceSize()

    @property
    def unk_token(self):
        return "<unk>"

    @property
    def pad_token(self):
        return None

    @property
    def bos_token(self):
        return "<s>"

    @property
    def eos_token(self):
        return "</s>"

    @property
    def pad_token_id(self):
        return -1

    @property
    def bos_token_id(self):
        return 0

    @property
    def eos_token_id(self):
        return 1

    @property
    def unk_token_id(self):
        return self.sp_model.unk_id()

    @property
    def special_tokens_map(self):
        return {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    @overload
    def convert_ids_to_tokens(self, ids: int) -> str: ...

    @overload
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]: ...

    def convert_ids_to_tokens(self, ids: Union[List[int], int]):
        return self.sp_model.IdToPiece(ids)

    def convert_tokens_to_ids(self, tokens: Union[List[str], str]) -> List[int]:
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self.sp_model.PieceToId(token) for token in tokens]

    def tokenize(self, text: str, language: str | None = None) -> List[str]:
        return self.encode(text, out_type=str, language=language)

    def encode(self, text: str, language: str | None = None, **kwargs):
        if len(text) == 0:
            return []
        if len(text.strip()) == 1:
            return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)
        
        # 预处理
        if self.normalizer:
            text = self.normalizer.normalize(text, language=language)
        
        text = text.upper()

        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                text = pre_tokenizer(text)
        return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)

    def batch_encode(self, texts: List[str], language: str | None = None, **kwargs):
        # 预处理
        if self.normalizer:
            texts = [self.normalizer.normalize(text, language=language) for text in texts]
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                texts = [pre_tokenizer(text) for text in texts]
        return self.sp_model.Encode(texts, out_type=kwargs.pop("out_type", int), **kwargs)

    def decode(self, ids: Union[List[int], int], do_lower_case=False, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        decoded = self.sp_model.Decode(ids, out_type=kwargs.pop("out_type", str), **kwargs)
        return de_tokenized_by_CJK_char(decoded, do_lower_case=do_lower_case)

    @staticmethod
    def split_segments_by_token(
        tokenized_str: List[str],
        split_tokens: List[str],
        max_text_tokens_per_segment: int,
        quick_streaming_tokens: int = 0
    ) -> List[List[str]]:
        """
        将tokenize后的结果按特定token进一步分割
        """
        # 处理特殊情况
        if len(tokenized_str) == 0:
            return []
        segments: List[List[str]] = []
        current_segment = []
        current_segment_tokens_len = 0
        for i in range(len(tokenized_str)):
            token = tokenized_str[i]
            current_segment.append(token)
            current_segment_tokens_len += 1
            if not  ("," in split_tokens or " ," in split_tokens ) and ("," in current_segment or " ," in current_segment): 
                # Если в текущих токенах есть запятая, но она не является токеном-разделителем, разделяем по ней
                sub_segments = TextTokenizer.split_segments_by_token(
                    current_segment, [",", " ,"], max_text_tokens_per_segment=max_text_tokens_per_segment, quick_streaming_tokens = quick_streaming_tokens
                )
            elif "-" not in split_tokens and "-" in current_segment:
                # Нет запятой, разделяем по дефису
                sub_segments = TextTokenizer.split_segments_by_token(
                    current_segment, ["-"], max_text_tokens_per_segment=max_text_tokens_per_segment, quick_streaming_tokens = quick_streaming_tokens
                )
            elif current_segment_tokens_len <= max_text_tokens_per_segment:
                if token in split_tokens and current_segment_tokens_len > 2:
                    if i < len(tokenized_str) - 1:
                        if tokenized_str[i + 1] in ["'", " '"]:
                            # Если следующий токен - кавычка, не разделяем
                            current_segment.append(tokenized_str[i + 1])
                            i += 1
                    segments.append(current_segment)
                    current_segment = []
                    current_segment_tokens_len = 0
                continue
            # Если текущие токены превышают максимальную длину
            else:
                # Разделяем по длине
                sub_segments = []
                for j in range(0, len(current_segment), max_text_tokens_per_segment):
                    if j + max_text_tokens_per_segment < len(current_segment):
                        sub_segments.append(current_segment[j : j + max_text_tokens_per_segment])
                    else:
                        sub_segments.append(current_segment[j:])
                warnings.warn(
                    f"The tokens length of segment exceeds limit: {max_text_tokens_per_segment}, "
                    f"Tokens in segment: {current_segment}."
                    "Maybe unexpected behavior",
                    RuntimeWarning,
                )
            segments.extend(sub_segments)
            current_segment = []
            current_segment_tokens_len = 0
        if current_segment_tokens_len > 0:
            assert current_segment_tokens_len <= max_text_tokens_per_segment
            segments.append(current_segment)
        # Если соседние предложения вместе короче лимита и общее число токенов превысило quick_streaming_tokens, объединяем
        merged_segments = []
        total_token = 0
        for segment in segments:
            total_token += len(segment)
            if len(segment) == 0:
                continue
            if len(merged_segments) == 0:
                merged_segments.append(segment)
            elif len(merged_segments[-1]) + len(segment) <= max_text_tokens_per_segment and total_token > quick_streaming_tokens:
                merged_segments[-1] = merged_segments[-1] + segment
            # Или если меньше половины максимальной длины, объединяем
            elif len(merged_segments[-1]) + len(segment) <= max_text_tokens_per_segment / 2:
                merged_segments[-1] = merged_segments[-1] + segment
            else:
                merged_segments.append(segment)
        return merged_segments

    punctuation_marks_tokens = [
        ".",
        "!",
        "?",
        " .",
        # " !", # unk
        " ?",
        " ...", # ellipsis
    ]
    def split_segments(self, tokenized: List[str], max_text_tokens_per_segment=120, quick_streaming_tokens = 0) -> List[List[str]]:
        return TextTokenizer.split_segments_by_token(
            tokenized, self.punctuation_marks_tokens, max_text_tokens_per_segment=max_text_tokens_per_segment, quick_streaming_tokens = quick_streaming_tokens
        )


if __name__ == "__main__":
    # Тестовый код

    text_normalizer = TextNormalizer()

    cases = [
        "IndexTTS 正式发布1.0版本了，效果666",
        "晕XUAN4是一种GAN3觉",
        "我爱你！",
        "I love you!",
        "“我爱你”的英语是“I love you”",
        "2.5平方电线",
        "共465篇，约315万字",
        "2002年的第一场雪，下在了2003年",
        "速度是10km/h",
        "现在是北京时间2025年01月11日 20:00",
        "他这条裤子是2012年买的，花了200块钱",
        "电话：135-4567-8900",
        "1键3连",
        "他这条视频点赞3000+，评论1000+，收藏500+",
        "这是1024元的手机，你要吗？",
        "受不liao3你了",
        "“衣裳”不读衣chang2，而是读衣shang5",
        "最zhong4要的是：不要chong2蹈覆辙",
        "不zuo1死就不会死",
        "See you at 8:00 AM",
        "8:00 AM 开会",
        "Couting down 3, 2, 1, go!",
        "数到3就开始：1、2、3",
        "This sales for 2.5% off, only $12.5.",
        "5G网络是4G网络的升级版，2G网络是3G网络的前身",
        "苹果于2030/1/2发布新 iPhone 2X 系列手机，最低售价仅 ¥12999",
        "这酒...里...有毒...",
        # 异常case
        "只有,,,才是最好的",
        "babala2是什么？",  # babala二是什么?
        "用beta1测试",  # 用beta一测试
        "have you ever been to beta2?",  # have you ever been to beta two?
        "such as XTTS, CosyVoice2, Fish-Speech, and F5-TTS",  # such as xtts,cosyvoice two,fish-speech,and f five-tts
        "where's the money?",  # where is the money?
        "who's there?",  # who is there?
        "which's the best?",  # which is the best?
        "how's it going?",  # how is it going?
        "今天是个好日子 it's a good day",  # 今天是个好日子 it is a good day
        # Русские примеры с улучшенной нормализацией
        "Привет, это IndexTTS, версия 1.0! Я стою 1024 руб.",
        "Это произошло 20.09.2025 г. на ул. Ленина, д. 5.",
        "Скорость была 50 км/ч, а цена 300 тыс. рублей (или 450€).",
        "Она сказала: 'Спасибо!!!' и пошла домой.",
        "Это — очень сложно. Почему? Потому что это так...",
        "Я люблю математику: $E=mc^2$ и $x^3+y^2=10$.",
        "Текст с разными символами: #хештег@почта&ко.",
        "Наша цель: 15.5 млн. метров в час. Дата 12.12.2012.",
        "Символы ^ и = тоже должны быть обработаны: 2^3=8.",
        "Очень много пробелов , и пунктуации . ",
        "Что вершит судьбу человечества в этом мире? Некое незримое существо или закон, подобно Длани Господней парящей над миром? По крайне мере истинно то, что человек не властен даже над своей волей"
    ]
    # Тест токенизатора
    tokenizer = TextTokenizer(
        vocab_file="checkpoints/extended_bpe.model",
        normalizer=text_normalizer,
    )

    codes = tokenizer.batch_encode(
        cases,
        out_type=int,
    )

    print(f"vocab_size: {tokenizer.vocab_size}")
    # print(f"pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    print(f"bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    print(f"eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    print(f"unk_token: {tokenizer.unk_token}, unk_token_id: {tokenizer.unk_token_id}")
    # Тест пиньинь (8474-10201)
    for id in range(8474, 10201):
        pinyin = tokenizer.convert_ids_to_tokens(id)
        if re.match(TextNormalizer.PINYIN_TONE_PATTERN, pinyin, re.IGNORECASE) is None:
            print(f"{pinyin} should be matched")
    for badcase in [
        "beta1", "better1", "voice2", "bala2", "babala2", "hunger2"
    ]:
        if re.match(TextNormalizer.PINYIN_TONE_PATTERN, badcase, re.IGNORECASE) is not None:
            print(f"{badcase} should not be matched!")
    # Не должно быть unk_token_id
    for t in set([*TextTokenizer.punctuation_marks_tokens, ",", " ,", "-", " ..."]):
        tokens = tokenizer.convert_tokens_to_ids(t)
        if tokenizer.unk_token_id in tokens:
            print(f"Warning: {t} is unknown token")
        print(f"`{t}`", "->", tokens, "->", tokenizer.convert_ids_to_tokens(tokens))
    for ch in set(tokenizer.normalizer.zh_char_rep_map.values()):
        # Тест: символы после нормализации должны быть распознаны токенизатором
        print(f"`{ch}`", "->", tokenizer.sp_model.Encode(ch, out_type=str))
        print(f"` {ch}`", "->", tokenizer.sp_model.Encode(f" {ch}", out_type=str))
    max_text_tokens_per_segment=120
    for i in range(len(cases)):
        print(f"Исходный текст: {cases[i]}")
        normalized = text_normalizer.normalize(cases[i])
        print(f"Normalized: {normalized}")
        tokens = tokenizer.tokenize(cases[i])
        print("Tokenized: ", ", ".join([f"`{t}`" for t in tokens]))
        segments = tokenizer.split_segments(tokens, max_text_tokens_per_segment=max_text_tokens_per_segment)
        print("Segments count:", len(segments))
        if len(segments) > 1:
            for j in range(len(segments)):
                print(f"  {j}, count:", len(segments[j]), ", tokens:", "".join(segments[j]))
                if len(segments[j]) > max_text_tokens_per_segment:
                    print(f"Warning: segment {j} is too long, length: {len(segments[j])}")
        #print(f"Token IDs (first 10): {codes[i][:10]}")
        if tokenizer.unk_token in codes[i]:
            print(f"Warning: `{cases[i]}` contains UNKNOWN token")
        print(f"Decoded: {tokenizer.decode(codes[i], do_lower_case=True)}")
        print("-" * 50)
