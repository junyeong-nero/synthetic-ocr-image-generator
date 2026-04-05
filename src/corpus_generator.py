import re
import logging
from typing import List
from datasets import load_dataset
from tqdm import tqdm

from src.utils import save_txt

logger = logging.getLogger(__name__)

# --- Modified Part: Define a list of special symbols to remove as a constant ---
# Includes all requested symbols (geometric shapes, math symbols, arrows, special quotes, etc.).
SPECIAL_SYMBOLS_TO_REMOVE = (
    "▲▼◀▶◢◣◥◤△▽◿◺◹◸▴▾◂▸▵▿◃▹◁▷◅▻◬⟁⧋⧊⊿"
    "○◌◍◎◯❍◉⦾⊙⦿⊜⊖⊘⊚⊛⊝●⚫⦁◐◑◒◓◔◕⦶⦸◵◴◶◷"
    "□■▰▪◼▮◾▗▖▫▭▱◽◻▢⊞⊡⊟⊠▣▤▥▦⬚▧▨▩⬓◧⬒◨◩◪⬔⬕⊞⊟▯◚◛◫❏❐❑❒❘❙❚⊡▀ ▂▃▄▅▆▇█▉▊▋▌▍▎▏░▒▓▔"
    "◇◆◈⬖⬗⬘⬙⬠⬡⎔⋄◊⧫⬢⬣"
    '❝❞❛❜‘’‛‚“”„‟«»‹›Ꞌ"'
    "+-×÷=<>±∞√∑∏∆∇∫∬∭∮∯∰∱∲∳≠≈≡≤≥≪≫∂∅∈∉⊂⊃⊆⊇⊕⊖⊗⊘⊙⊚⊛⊜⊝∀∃∄∴∵∶∷∸∹∺∻⁄|‖‗†‡•‣․‥…⁖⁘⁙⁏⁐⁓⁑⁒⁔⁕⁗⁘⁙⁚⁛⁜⁝⁞"
    "←→↑↓↔↕↖↗↘↙↚↛↜↝↞↟↠↡↢↣↤↦↥↧↨↫↬↭↮↯↰↱↲↳↴↵↶↷↸↹↺↻↼↽↾↿⇀⇁⇂⇃⇄⇅⇆⇇⇈⇉⇊⇋⇌⇍⇏⇎⇐⇒⇔⇕⇖⇗⇘⇙⇚⇛⇜⇝⇞⇟⇠⇡⇢⇣⇤⇥⇦⇧⇨⇩⇪⟲⟳⟴⟵⟶⟷⟸⟹⟺⟻⟼⟿"
    "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳⓵⓶⓷⓸⓹⓺⓻⓼⓽⓾⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⒒⒓⒔⒕⒖⒗⒘⒙⒚⒛⓪"
    "❶❷❸❹❺❻❼❽❾❿➊➋➌➍➎➏➐➑➒➓⓫⓬⓭⓮⓯⓰⓱⓲⓳⓴"
    "¹²³↉½⅓¼⅕⅙⅐⅛⅑⅒⅔⅖¾⅗⅜⅘⅚⅝⅞"
    "✿☺☻☹☼☂☃⌇⚛⌨✆☎⌘⇧×☓✕✖⨉⨯☒✗✘Χχᚷ⊗⨷ₓˣ𒉽⛒⛝🆇🅧Ⓧ𝕏✔✓☐☑★☆♺⚑⚐✉✄⌲✈♦♣♠♥❤♡♪♩♫♬♯♀♂⚢⚣"
    "❑❒◈◐◑✖✚✜⧓⧗⧑⧒⧖_⚊╴╼╾‐⁃‑‒–⎯—―╶╺╸©®™℠℻℅℁⅍℄¶⁋❡⁌⁍⸖⸗⸚⸓§₿⚽⚾☘❦❧☙❢❣✁✂✃✄"
    "·•・"  # Including middle dots and similar symbols
)
# Compile the regex for efficiency.
SPECIAL_SYMBOLS_REGEX = re.compile(f"[{re.escape(SPECIAL_SYMBOLS_TO_REMOVE)}]")


LANG_CONFIG = {
    "ko": {
        "dataset_id": "20231101.ko",
        "char_regex": r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9\s.?!]",
    },
    "en": {
        "dataset_id": "20231101.en",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "ja": {
        "dataset_id": "20231101.ja",
        "char_regex": r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF0-9\s.?!]",
    },
    "ar": {
        "dataset_id": "20231101.ar",
        "char_regex": r"[^\u0600-\u06FF0-9\s.?!]",
    },
    "hi": {
        "dataset_id": "20231101.hi",
        "char_regex": r"[^\u0900-\u097F0-9\s.?!]",
    },
    "vi": {
        "dataset_id": "20231101.vi",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "th": {
        "dataset_id": "20231101.th",
        "char_regex": r"[^\u0E00-\u0E7F0-9\s.?!]",
    },
    "zh": {
        "dataset_id": "20231101.zh",
        "char_regex": r"[^\u3400-\u4DBF\u4E00-\u9FFF0-9\s.?!]",
    },
    "fr": {
        "dataset_id": "20231101.fr",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "de": {
        "dataset_id": "20231101.de",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "es": {
        "dataset_id": "20231101.es",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "pt": {
        "dataset_id": "20231101.pt",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "it": {
        "dataset_id": "20231101.it",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "nl": {
        "dataset_id": "20231101.nl",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "sv": {
        "dataset_id": "20231101.sv",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "no": {
        "dataset_id": "20231101.no",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "da": {
        "dataset_id": "20231101.da",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "fi": {
        "dataset_id": "20231101.fi",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "is": {
        "dataset_id": "20231101.is",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "et": {
        "dataset_id": "20231101.et",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "lv": {
        "dataset_id": "20231101.lv",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "lt": {
        "dataset_id": "20231101.lt",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "pl": {
        "dataset_id": "20231101.pl",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "cs": {
        "dataset_id": "20231101.cs",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "sk": {
        "dataset_id": "20231101.sk",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "sl": {
        "dataset_id": "20231101.sl",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "hr": {
        "dataset_id": "20231101.hr",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "ro": {
        "dataset_id": "20231101.ro",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "hu": {
        "dataset_id": "20231101.hu",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "tr": {
        "dataset_id": "20231101.tr",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "id": {
        "dataset_id": "20231101.id",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "ms": {
        "dataset_id": "20231101.ms",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "sw": {
        "dataset_id": "20231101.sw",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "af": {
        "dataset_id": "20231101.af",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "sq": {
        "dataset_id": "20231101.sq",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "ca": {
        "dataset_id": "20231101.ca",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "gl": {
        "dataset_id": "20231101.gl",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "eu": {
        "dataset_id": "20231101.eu",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "ga": {
        "dataset_id": "20231101.ga",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "mt": {
        "dataset_id": "20231101.mt",
        "char_regex": r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9\s.?!]",
    },
    "ru": {
        "dataset_id": "20231101.ru",
        "char_regex": r"[^\u0400-\u04FF\u0500-\u052F0-9\s.?!]",
    },
    "uk": {
        "dataset_id": "20231101.uk",
        "char_regex": r"[^\u0400-\u04FF\u0500-\u052F0-9\s.?!]",
    },
    "bg": {
        "dataset_id": "20231101.bg",
        "char_regex": r"[^\u0400-\u04FF\u0500-\u052F0-9\s.?!]",
    },
    "sr": {
        "dataset_id": "20231101.sr",
        "char_regex": r"[^\u0400-\u04FF\u0500-\u052F0-9\s.?!]",
    },
    "mk": {
        "dataset_id": "20231101.mk",
        "char_regex": r"[^\u0400-\u04FF\u0500-\u052F0-9\s.?!]",
    },
    "be": {
        "dataset_id": "20231101.be",
        "char_regex": r"[^\u0400-\u04FF\u0500-\u052F0-9\s.?!]",
    },
    "kk": {
        "dataset_id": "20231101.kk",
        "char_regex": r"[^\u0400-\u04FF\u0500-\u052F0-9\s.?!]",
    },
    "el": {
        "dataset_id": "20231101.el",
        "char_regex": r"[^\u0370-\u03FF0-9\s.?!]",
    },
    "he": {
        "dataset_id": "20231101.he",
        "char_regex": r"[^\u0590-\u05FF0-9\s.?!]",
    },
    "fa": {
        "dataset_id": "20231101.fa",
        "char_regex": r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF0-9\s.?!]",
    },
    "ur": {
        "dataset_id": "20231101.ur",
        "char_regex": r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF0-9\s.?!]",
    },
    "bn": {
        "dataset_id": "20231101.bn",
        "char_regex": r"[^\u0980-\u09FF0-9\s.?!]",
    },
    "pa": {
        "dataset_id": "20231101.pa",
        "char_regex": r"[^\u0A00-\u0A7F0-9\s.?!]",
    },
    "gu": {
        "dataset_id": "20231101.gu",
        "char_regex": r"[^\u0A80-\u0AFF0-9\s.?!]",
    },
    "ta": {
        "dataset_id": "20231101.ta",
        "char_regex": r"[^\u0B80-\u0BFF0-9\s.?!]",
    },
    "te": {
        "dataset_id": "20231101.te",
        "char_regex": r"[^\u0C00-\u0C7F0-9\s.?!]",
    },
    "kn": {
        "dataset_id": "20231101.kn",
        "char_regex": r"[^\u0C80-\u0CFF0-9\s.?!]",
    },
    "ml": {
        "dataset_id": "20231101.ml",
        "char_regex": r"[^\u0D00-\u0D7F0-9\s.?!]",
    },
    "si": {
        "dataset_id": "20231101.si",
        "char_regex": r"[^\u0D80-\u0DFF0-9\s.?!]",
    },
    "lo": {
        "dataset_id": "20231101.lo",
        "char_regex": r"[^\u0E80-\u0EFF0-9\s.?!]",
    },
    "my": {
        "dataset_id": "20231101.my",
        "char_regex": r"[^\u1000-\u109F0-9\s.?!]",
    },
    "km": {
        "dataset_id": "20231101.km",
        "char_regex": r"[^\u1780-\u17FF0-9\s.?!]",
    },
    "ka": {
        "dataset_id": "20231101.ka",
        "char_regex": r"[^\u10A0-\u10FF0-9\s.?!]",
    },
    "hy": {
        "dataset_id": "20231101.hy",
        "char_regex": r"[^\u0530-\u058F0-9\s.?!]",
    },
    "am": {
        "dataset_id": "20231101.am",
        "char_regex": r"[^\u1200-\u137F0-9\s.?!]",
    },
}


def clean_wiki_text(text: str, lang: str) -> str:
    """Cleans and removes unnecessary markup and special characters from Wikipedia text."""
    # --- Modified Part: First, remove all special symbols defined above ---
    words = text.split()
    text = " ".join([SPECIAL_SYMBOLS_REGEX.sub("", word) for word in words])

    # Existing logic to remove Wikipedia markup
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r" ", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r" ", text)
    text = re.sub(r"https?://[^ ]+", "", text)
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"==+\s*(.*?)\s*==+", r" .", text)

    # Remove all characters except those allowed for the specific language
    if lang in LANG_CONFIG:
        text = re.sub(LANG_CONFIG[lang]["char_regex"], "", text)

    # Normalize whitespace
    text = " ".join(text.split()).strip()
    return text


# --------------------------------------------------------------------------
# 1.2 Function to Create a Corpus File from Wikipedia
# --------------------------------------------------------------------------
def create_corpus_from_wiki(output_path: str, lang: str, num_sentences: int = 100000):
    """
    Collects sentences from the Wikimedia dataset for a specified language to create a corpus file.
    """
    if lang not in LANG_CONFIG:
        logger.error(
            f"Error: Language '{lang}' is not supported. Supported languages are: {list(LANG_CONFIG.keys())}"
        )
        return

    lang_settings = LANG_CONFIG[lang]
    logger.info(
        f"Starting to create '{output_path}' for language '{lang}'. Target sentences: {num_sentences:,}"
    )

    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            lang_settings["dataset_id"],
            split="train",
            streaming=True,
        )
        shuffled_dataset = dataset.shuffle(buffer_size=10000)
    except Exception as e:
        logger.error(f"Error loading dataset for language '{lang}': {e}")
        return

    collected_sentences: List[str] = []
    with tqdm(total=num_sentences, desc=f"Collecting '{lang}' sentences") as pbar:
        for data in shuffled_dataset:
            if len(collected_sentences) >= num_sentences:
                break

            cleaned_text = clean_wiki_text(data["text"], lang)
            sentences = re.split(r"(?<=[.?!])\s+", cleaned_text)

            for sentence in sentences:
                s = sentence.strip()
                if 10 < len(s) < 100:
                    collected_sentences.append(s)
                    pbar.update(1)
                    if len(collected_sentences) >= num_sentences:
                        break

    if pbar.n < num_sentences:
        pbar.update(num_sentences - pbar.n)

    text = "\n".join(collected_sentences)
    save_txt(output_path, text)

    logger.info(
        f"Saved a total of {len(collected_sentences):,} sentences to '{output_path}'."
    )
