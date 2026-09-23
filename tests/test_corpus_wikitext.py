from src.corpus_wikitext import clean_text, parse_wikitext, write_corpus


def test_clean_text_removes_markup_debris() -> None:
    assert clean_text("스폴리아텔레(, )는 과자이다.") == "스폴리아텔레는 과자이다."
    assert clean_text("〈LIFE〉(라이프, )는 싱글") == "〈LIFE〉(라이프)는 싱글"
    assert clean_text("연구자이다.완벽한 환상") == "연구자이다. 완벽한 환상"


def test_parse_wikitext_splits_titles_and_paragraphs(tmp_path) -> None:
    lines = [
        " = 봉은사 =\n",
        "\n",
        "봉은사(奉恩寺)는 대한민국 서울특별시 강남구 삼성동 수도산 기슭에 있는 사찰이다. 조계종 사찰이다.\n",
        " = = 역사 = =\n",
        "짧다.\n",
        "1990 2000 2010 2020 3030 4040 5050 6060 7070 8080 9090 1111 2222 3333\n",
        " = 분류:사찰 =\n",
    ]
    corpus = parse_wikitext(lines, min_chars=20)
    assert corpus.titles == ["봉은사", "역사"]
    assert len(corpus.paragraphs) == 1 and corpus.paragraphs[0].startswith("봉은사")

    paragraphs_path, titles_path = write_corpus(corpus, tmp_path, "ko")
    assert paragraphs_path.read_text(encoding="utf-8").strip() == corpus.paragraphs[0]
    assert titles_path.read_text(encoding="utf-8").splitlines() == ["봉은사", "역사"]


def test_clean_text_strips_wiki_list_markup() -> None:
    assert clean_text("# 수행 유도하기 예를 들어") == "수행 유도하기 예를 들어"
    assert clean_text("있다. : 가천의대가 태동한 곳") == "있다. 가천의대가 태동한 곳"
    assert clean_text("펼쳤다. # 수행 유도하기") == "펼쳤다. 수행 유도하기"


def test_parse_wikitext_drops_markup_debris_paragraphs() -> None:
    lines = [
        'style="background:#CCD3D" colspan=10 | 1회 대회 결과 표에 나온 여러 팀들의 기록이 정리되어 있다.\n',
        "--Rhapsody(토론) 2010년 4월 3일 (토) 12:15 (KST) 의견을 남깁니다 여기에 길게 적습니다.\n",
        "자세한 내용은 https://example.org 문서에서 확인할 수 있으며 추가 설명이 이어진다.\n",
        "봉은사는 대한민국 서울특별시 강남구 삼성동 수도산 기슭에 있는 사찰이다.\n",
    ]
    corpus = parse_wikitext(lines, min_chars=20)
    assert corpus.paragraphs == ["봉은사는 대한민국 서울특별시 강남구 삼성동 수도산 기슭에 있는 사찰이다."]
