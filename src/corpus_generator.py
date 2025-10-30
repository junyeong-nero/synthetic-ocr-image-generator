import re
import yaml
import logging
from typing import List
from datasets import load_dataset
from tqdm import tqdm

from utils import save_txt

logger = logging.getLogger(__name__)

# --- 수정된 부분: 제거할 특수 기호 목록을 상수로 정의 ---
# 요청된 모든 기호(기하학적 도형, 수학 기호, 화살표, 특수 따옴표 등)를 포함합니다.
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
    "·•・"  # 가운데점 및 유사 기호 포함
)
# 효율성을 위해 정규식을 컴파일합니다.
SPECIAL_SYMBOLS_REGEX = re.compile(f"[{re.escape(SPECIAL_SYMBOLS_TO_REMOVE)}]")


LANG_CONFIG = {
    "ko": {
        "dataset_id": "20231101.ko",
        "char_regex": r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9\s.?!]",
    },
    "en": {
        "dataset_id": "20231101.en",
        "char_regex": r"[^a-zA-Z0-9\s.?!]",
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
        "char_regex": r"[^a-zA-Z\u00C0-\u017F0-9\s.?!]",
    },
    "th": {
        "dataset_id": "20231101.th",
        "char_regex": r"[^\u0E00-\u0E7F0-9\s.?!]",
    },
}


def clean_wiki_text(text: str, lang: str) -> str:
    """Cleans and removes unnecessary markup and special characters from Wikipedia text."""
    # --- 수정된 부분: 위에서 정의한 모든 특수 기호를 먼저 제거 ---
    words = text.split()
    text = " ".join([SPECIAL_SYMBOLS_REGEX.sub("", word) for word in words])

    # 기존 위키피디아 마크업 제거 로직
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r" ", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r" ", text)
    text = re.sub(r"https?://[^ ]+", "", text)
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"==+\s*(.*?)\s*==+", r" .", text)

    # 언어별로 허용된 문자 외에는 모두 제거
    if lang in LANG_CONFIG:
        text = re.sub(LANG_CONFIG[lang]["char_regex"], "", text)

    # 공백 정규화
    text = " ".join(text.split()).strip()
    return text


# --------------------------------------------------------------------------
# 1.2 Function to Create a Corpus File from Wikipedia
# --------------------------------------------------------------------------
def create_corpus_from_wiki(output_path: str, lang: str, num_sentences: int = 5000):
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
                    pbar.update(1)  # 진행률 1 증가
                    if len(collected_sentences) >= num_sentences:
                        break

    if pbar.n < num_sentences:
        pbar.update(num_sentences - pbar.n)

    text = "\n".join(collected_sentences)
    save_txt(output_path, text)

    logger.info(
        f"Saved a total of {len(collected_sentences):,} sentences to '{output_path}'."
    )
