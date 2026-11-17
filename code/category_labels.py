"""Shared English display labels for the released category taxonomy."""

CATEGORY_ENGLISH_NAMES = {
    "健身": "Fitness",
    "民生资讯": "Livelihood news",
    "宠物": "Pets",
    "军事": "Military",
    "运动": "Sports",
    "时尚": "Fashion",
    "喜剧": "Comedy",
    "其他": "Other",
    "其他(合并)": "Other (merged)",
    "科学与法律": "Science & law",
    "美食": "Food",
    "健康": "Health",
    "短剧": "Short drama",
    "财经": "Finance",
    "情感": "Sentiment",
    "艺术": "Arts",
    "颜值": "Beauty",
    "舞蹈": "Dance",
    "时政资讯": "Current affairs",
    "高新数码": "Hi-tech",
    "三农": "Rural life",
    "二次元": "ACG",
    "亲子": "Parenting",
    "人文": "Humanities",
    "奇人异象": "Oddities",
    "影视综": "Film & TV",
    "房产家居": "Home & living",
    "摄影": "Photography",
    "教育": "Education",
    "旅游": "Travel",
    "明星娱乐": "Celebrity",
    "星座命理": "Astrology",
    "汽车": "Automotive",
    "游戏": "Gaming",
    "生活": "Lifestyle",
    "美妆": "Makeup",
    "自拍": "Selfie",
    "读书": "Reading",
    "音乐": "Music",
}


def english_category_label(category: object) -> str:
    """Return the released English label, preserving ASCII fixture labels."""
    original = str(category)
    return str(CATEGORY_ENGLISH_NAMES.get(original, original))
