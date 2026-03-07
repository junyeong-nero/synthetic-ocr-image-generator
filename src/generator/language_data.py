from dataclasses import dataclass, field


@dataclass
class LanguageData:
    items: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    store_names: list[str] = field(default_factory=list)
    product_names: list[str] = field(default_factory=list)
    departments: list[str] = field(default_factory=list)
    positions: list[str] = field(default_factory=list)
    headers: dict[str, list[str]] = field(default_factory=dict)
    titles: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    code_comments: list[str] = field(default_factory=list)
    currency: str = ""
    currency_format: str = ""


KOREAN_DATA = LanguageData(
    items=[
        "사과", "바나나", "우유", "빵", "커피", "라면", "김치", "두부", "계란", "생수",
        "요거트", "치즈", "햄", "소시지", "과자", "초콜릿", "아이스크림", "주스", "탄산수", "맥주",
        "와인", "소주", "막걸리", "참치캔", "스팸", "라면", "쌀", "밀가루", "설탕", "소금",
    ],
    categories=["식품", "음료", "생활용품", "전자제품", "의류", "화장품", "가구", "스포츠", "도서", "완구"],
    subjects=["국어", "수학", "영어", "과학", "사회", "체육", "음악", "미술", "도덕", "기술"],
    store_names=[
        "GS25 강남점", "CU 서초역점", "세븐일레븐 역삼점", "이마트24 삼성점",
        "스타벅스 테헤란로점", "투썸플레이스 강남역점", "이디야커피 선릉점",
        "맥도날드 강남DT점", "버거킹 역삼점", "롯데리아 삼성역점",
        "올리브영 강남본점", "다이소 서초점", "미니스톱 역삼점",
        "홈플러스 익스프레스 논현점", "이마트 에브리데이 신사점",
    ],
    product_names=[
        "아메리카노", "카페라떼", "녹차라떼", "바닐라라떼", "카라멜마키아또",
        "삼각김밥", "컵라면", "도시락", "샌드위치", "김밥",
        "우유", "요거트", "주스", "생수", "탄산수",
        "초콜릿", "과자", "빵", "아이스크림", "껌",
        "핫도그", "떡볶이", "순대", "어묵", "튀김",
    ],
    departments=["영업부", "마케팅부", "개발부", "인사부", "재무부", "기획부", "총무부", "연구소", "품질관리부", "고객지원부"],
    positions=["사원", "대리", "과장", "차장", "부장", "이사", "상무", "전무", "부사장", "사장"],
    headers={
        "invoice": ["품목", "수량", "가격", "합계"],
        "schedule": ["시간", "월", "화", "수", "목", "금"],
        "product": ["제품", "분류", "가격", "재고"],
        "contact": ["이름", "전화", "이메일"],
    },
    titles=[
        "프로젝트 시작하기", "설치 가이드", "API 레퍼런스", "사용자 매뉴얼",
        "개발 환경 설정", "배포 가이드", "테스트 작성법", "성능 최적화",
        "보안 가이드", "마이그레이션 가이드", "트러블슈팅", "FAQ",
    ],
    paragraphs=[
        "이 프로젝트는 사용자의 생산성을 높이기 위해 설계되었습니다.",
        "다양한 기능을 제공하며 확장 가능한 아키텍처를 가지고 있습니다.",
        "설치가 간단하고 문서화가 잘 되어 있어 빠르게 시작할 수 있습니다.",
        "커뮤니티의 지원을 받아 지속적으로 개선되고 있습니다.",
        "오픈소스로 제공되어 누구나 기여할 수 있습니다.",
        "최신 기술 스택을 사용하여 안정적이고 빠른 성능을 제공합니다.",
        "다양한 플랫폼에서 동작하며 호환성이 뛰어납니다.",
        "상세한 에러 메시지와 로깅으로 디버깅이 용이합니다.",
    ],
    features=[
        "빠른 성능", "간편한 설치", "다양한 플러그인", "상세한 문서", "활발한 커뮤니티",
        "크로스 플랫폼 지원", "자동 업데이트", "보안 강화", "확장 가능한 API", "실시간 동기화",
    ],
    code_comments=[
        "# 설정 파일을 로드합니다",
        "# 데이터베이스 연결을 설정합니다",
        "# 사용자 인증을 처리합니다",
        "# 캐시를 초기화합니다",
        "# 로깅을 설정합니다",
    ],
    currency="원",
    currency_format="{:,}원",
)


ENGLISH_DATA = LanguageData(
    items=[
        "Apple", "Banana", "Milk", "Bread", "Coffee", "Pasta", "Cheese", "Eggs", "Water", "Juice",
        "Yogurt", "Ham", "Sausage", "Chips", "Chocolate", "Ice Cream", "Cereal", "Rice", "Flour", "Sugar",
        "Salt", "Butter", "Olive Oil", "Tuna", "Salmon", "Chicken", "Beef", "Pork", "Tofu", "Beans",
    ],
    categories=["Food", "Beverage", "Household", "Electronics", "Clothing", "Beauty", "Furniture", "Sports", "Books", "Toys"],
    subjects=["Math", "English", "Science", "History", "Art", "PE", "Music", "Geography", "Biology", "Chemistry"],
    store_names=[
        "Starbucks Coffee", "McDonald's", "Burger King", "Subway",
        "7-Eleven", "Walgreens", "CVS Pharmacy", "Target",
        "Walmart", "Costco", "Whole Foods", "Trader Joe's",
        "Safeway", "Kroger", "Publix",
    ],
    product_names=[
        "Americano", "Latte", "Cappuccino", "Mocha", "Espresso",
        "Sandwich", "Salad", "Burger", "Pizza Slice", "Hot Dog",
        "Water", "Juice", "Soda", "Coffee", "Tea",
        "Chips", "Cookies", "Candy", "Gum", "Chocolate",
        "Donut", "Bagel", "Muffin", "Croissant", "Pretzel",
    ],
    departments=["Sales", "Marketing", "Engineering", "HR", "Finance", "Operations", "Legal", "R&D", "QA", "Support"],
    positions=["Associate", "Senior", "Lead", "Manager", "Director", "VP", "SVP", "EVP", "President", "CEO"],
    headers={
        "invoice": ["Item", "Qty", "Price", "Total"],
        "schedule": ["Time", "Mon", "Tue", "Wed", "Thu", "Fri"],
        "product": ["Product", "Category", "Price", "Stock"],
        "contact": ["Name", "Phone", "Email"],
    },
    titles=[
        "Getting Started", "Installation Guide", "API Reference", "User Manual",
        "Development Setup", "Deployment Guide", "Writing Tests", "Performance Optimization",
        "Security Guide", "Migration Guide", "Troubleshooting", "FAQ",
    ],
    paragraphs=[
        "This project is designed to enhance user productivity.",
        "It provides various features with an extensible architecture.",
        "Easy to install and well-documented for quick onboarding.",
        "Continuously improved with community support.",
        "Open source and open for contributions from anyone.",
        "Built with modern technology stack for stability and performance.",
        "Works across multiple platforms with excellent compatibility.",
        "Detailed error messages and logging for easy debugging.",
    ],
    features=[
        "Fast performance", "Easy installation", "Various plugins", "Detailed documentation", "Active community",
        "Cross-platform support", "Auto updates", "Enhanced security", "Extensible API", "Real-time sync",
    ],
    code_comments=[
        "# Load configuration file",
        "# Setup database connection",
        "# Handle user authentication",
        "# Initialize cache",
        "# Configure logging",
    ],
    currency="$",
    currency_format="${:,.2f}",
)


JAPANESE_DATA = LanguageData(
    items=[
        "りんご", "バナナ", "牛乳", "パン", "コーヒー", "ラーメン", "寿司", "豆腐", "卵", "水",
        "ヨーグルト", "チーズ", "ハム", "ソーセージ", "お菓子", "チョコレート", "アイスクリーム", "ジュース", "炭酸水", "ビール",
        "ワイン", "日本酒", "焼酎", "ツナ缶", "カレー", "うどん", "そば", "米", "小麦粉", "砂糖",
    ],
    categories=["食品", "飲料", "日用品", "電子機器", "衣料品", "化粧品", "家具", "スポーツ", "書籍", "玩具"],
    subjects=["国語", "数学", "英語", "理科", "社会", "体育", "音楽", "美術", "道徳", "技術"],
    store_names=[
        "セブンイレブン渋谷店", "ローソン新宿店", "ファミリーマート池袋店", "ミニストップ品川店",
        "スターバックス表参道店", "ドトールコーヒー銀座店", "タリーズコーヒー六本木店",
        "マクドナルド渋谷センター街店", "モスバーガー新宿西口店", "吉野家秋葉原店",
        "マツモトキヨシ原宿店", "ダイソー渋谷店", "ドン・キホーテ新宿店",
        "イオン幕張店", "ヨドバシカメラ秋葉原店",
    ],
    product_names=[
        "アメリカーノ", "カフェラテ", "抹茶ラテ", "バニララテ", "キャラメルマキアート",
        "おにぎり", "カップ麺", "弁当", "サンドイッチ", "おでん",
        "牛乳", "ヨーグルト", "ジュース", "ミネラルウォーター", "炭酸水",
        "チョコレート", "お菓子", "パン", "アイスクリーム", "ガム",
        "たこ焼き", "焼きそば", "肉まん", "あんまん", "ドーナツ",
    ],
    departments=["営業部", "マーケティング部", "開発部", "人事部", "財務部", "企画部", "総務部", "研究所", "品質管理部", "カスタマーサポート部"],
    positions=["社員", "係長", "課長", "次長", "部長", "取締役", "常務", "専務", "副社長", "社長"],
    headers={
        "invoice": ["品目", "数量", "価格", "合計"],
        "schedule": ["時間", "月", "火", "水", "木", "金"],
        "product": ["製品", "カテゴリ", "価格", "在庫"],
        "contact": ["名前", "電話", "メール"],
    },
    titles=[
        "プロジェクトを始める", "インストールガイド", "APIリファレンス", "ユーザーマニュアル",
        "開発環境の設定", "デプロイガイド", "テストの書き方", "パフォーマンス最適化",
        "セキュリティガイド", "マイグレーションガイド", "トラブルシューティング", "FAQ",
    ],
    paragraphs=[
        "このプロジェクトは、ユーザーの生産性を向上させるために設計されています。",
        "拡張可能なアーキテクチャで様々な機能を提供しています。",
        "インストールが簡単で、ドキュメントが充実しているため、すぐに始められます。",
        "コミュニティのサポートにより継続的に改善されています。",
        "オープンソースで提供され、誰でも貢献できます。",
        "最新の技術スタックを使用し、安定性と高速パフォーマンスを提供します。",
        "複数のプラットフォームで動作し、互換性に優れています。",
        "詳細なエラーメッセージとログにより、デバッグが容易です。",
    ],
    features=[
        "高速なパフォーマンス", "簡単なインストール", "豊富なプラグイン", "詳細なドキュメント", "活発なコミュニティ",
        "クロスプラットフォーム対応", "自動更新", "セキュリティ強化", "拡張可能なAPI", "リアルタイム同期",
    ],
    code_comments=[
        "# 設定ファイルを読み込みます",
        "# データベース接続を設定します",
        "# ユーザー認証を処理します",
        "# キャッシュを初期化します",
        "# ログを設定します",
    ],
    currency="円",
    currency_format="{:,}円",
)


HINDI_DATA = LanguageData(
    items=[
        "सेब", "केला", "दूध", "रोटी", "कॉफी", "चावल", "दाल", "पनीर", "अंडा", "पानी",
        "दही", "मक्खन", "आटा", "चीनी", "नमक", "तेल", "चाय", "बिस्कुट", "नूडल्स", "मसाले",
        "सब्जी", "फल", "मांस", "मछली", "आइसक्रीम", "चॉकलेट", "जूस", "सोडा", "बीयर", "शराब",
    ],
    categories=["खाद्य", "पेय", "घरेलू", "इलेक्ट्रॉनिक्स", "कपड़े", "सौंदर्य", "फर्नीचर", "खेल", "पुस्तकें", "खिलौने"],
    subjects=["हिंदी", "गणित", "अंग्रेजी", "विज्ञान", "सामाजिक", "शारीरिक", "संगीत", "कला", "नैतिक", "कंप्यूटर"],
    store_names=[
        "रिलायंस फ्रेश दिल्ली", "बिग बाजार मुंबई", "डी-मार्ट बैंगलोर", "स्पेंसर्स कोलकाता",
        "कैफे कॉफी डे कनॉट प्लेस", "बरिस्ता जुहू", "स्टारबक्स गुड़गांव",
        "मैकडॉनल्ड्स साकेत", "डोमिनोज पिज्जा अंधेरी", "केएफसी नोएडा",
        "अपोलो फार्मेसी चेन्नई", "मेडप्लस हैदराबाद", "नेटमेड्स पुणे",
        "विश्वास मार्ट दिल्ली", "ईजी डे चेन्नई",
    ],
    product_names=[
        "अमेरिकानो", "कैफे लाते", "ग्रीन टी लाते", "वनीला लाते", "कारमेल मैकियाटो",
        "समोसा", "कप नूडल्स", "थाली", "सैंडविच", "पराठा",
        "दूध", "दही", "जूस", "पानी", "लस्सी",
        "चॉकलेट", "बिस्कुट", "रोटी", "आइसक्रीम", "च्युइंग गम",
        "पकौड़ा", "चाट", "डोसा", "इडली", "वड़ा",
    ],
    departments=["बिक्री विभाग", "मार्केटिंग विभाग", "विकास विभाग", "मानव संसाधन विभाग", "वित्त विभाग", "योजना विभाग", "प्रशासन विभाग", "अनुसंधान विभाग", "गुणवत्ता विभाग", "ग्राहक सेवा विभाग"],
    positions=["कार्यकारी", "सहायक प्रबंधक", "प्रबंधक", "वरिष्ठ प्रबंधक", "उप निदेशक", "निदेशक", "उप अध्यक्ष", "अध्यक्ष", "सीईओ", "एमडी"],
    headers={
        "invoice": ["वस्तु", "मात्रा", "मूल्य", "कुल"],
        "schedule": ["समय", "सोम", "मंगल", "बुध", "गुरु", "शुक्र"],
        "product": ["उत्पाद", "श्रेणी", "मूल्य", "स्टॉक"],
        "contact": ["नाम", "फोन", "ईमेल"],
    },
    titles=[
        "प्रोजेक्ट शुरू करना", "इंस्टॉलेशन गाइड", "एपीआई संदर्भ", "उपयोगकर्ता मैनुअल",
        "विकास वातावरण सेटअप", "डिप्लॉयमेंट गाइड", "टेस्ट लिखना", "प्रदर्शन अनुकूलन",
        "सुरक्षा गाइड", "माइग्रेशन गाइड", "समस्या निवारण", "FAQ",
    ],
    paragraphs=[
        "यह प्रोजेक्ट उपयोगकर्ता की उत्पादकता बढ़ाने के लिए डिज़ाइन किया गया है।",
        "यह विस्तार योग्य आर्किटेक्चर के साथ विभिन्न सुविधाएं प्रदान करता है।",
        "इंस्टॉल करना आसान है और अच्छी तरह से प्रलेखित है।",
        "समुदाय के समर्थन से लगातार सुधार किया जा रहा है।",
        "ओपन सोर्स और किसी के भी योगदान के लिए खुला है।",
        "आधुनिक तकनीकी स्टैक के साथ स्थिरता और प्रदर्शन प्रदान करता है।",
        "कई प्लेटफार्मों पर काम करता है और उत्कृष्ट संगतता है।",
        "विस्तृत त्रुटि संदेश और लॉगिंग से डिबगिंग आसान है।",
    ],
    features=[
        "तेज़ प्रदर्शन", "आसान इंस्टॉलेशन", "विभिन्न प्लगइन्स", "विस्तृत दस्तावेज़", "सक्रिय समुदाय",
        "क्रॉस-प्लेटफॉर्म समर्थन", "स्वचालित अपडेट", "बढ़ी हुई सुरक्षा", "विस्तार योग्य API", "रीयल-टाइम सिंक",
    ],
    code_comments=[
        "# कॉन्फ़िगरेशन फ़ाइल लोड करें",
        "# डेटाबेस कनेक्शन सेटअप करें",
        "# उपयोगकर्ता प्रमाणीकरण संभालें",
        "# कैश प्रारंभ करें",
        "# लॉगिंग कॉन्फ़िगर करें",
    ],
    currency="₹",
    currency_format="₹{:,.2f}",
)


LANGUAGE_DATA: dict[str, LanguageData] = {
    "ko": KOREAN_DATA,
    "en": ENGLISH_DATA,
    "ja": JAPANESE_DATA,
    "hi": HINDI_DATA,
}
