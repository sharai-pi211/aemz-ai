"""
Гибридная рекомендательная система для подбора косметики Socrimea.

Логика:
1. Фильтры по категории, типам продукта, аллергиям и нежелательным компонентам.
2. Экспертный скоринг по зоне применения, типу кожи/волос, проблемам и эффектам.
3. TF-IDF похожесть: учитывает общий текстовый смысл запроса и карточки товара.
4. Итоговая рекомендация: топ-N товаров с объяснением, почему они подходят.

Это не медицинская система и не заменяет консультацию специалиста.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


LIST_COLUMNS = [
    "target_area",
    "skin_type",
    "hair_type",
    "concern_tags",
    "effect_tags",
    "active_ingredients",
    "ingredients_list",
    "potential_allergens_or_irritants",
]

ALIASES = {
    "голова": "кожа головы",
    "для головы": "кожа головы",
    "кожа головы": "кожа головы",
    "скальп": "кожа головы",
    "лицо": "лицо",
    "для лица": "лицо",
    "тело": "тело",
    "для тела": "тело",
    "руки": "руки",
    "ноги": "ноги",
    "губы": "губы",
    "увлажнить": "увлажнение",
    "увлажняющий": "увлажнение",
    "сухость": "сухость",
    "сухая": "сухая",
    "жирность": "жирность",
    "жирная": "жирная",
    "акне": "акне",
    "прыщи": "акне",
    "высыпания": "высыпания",
    "раздражение": "раздражение",
    "покраснение": "покраснение",
    "шелушение": "шелушение",
    "перхоть": "перхоть",
    "зуд": "зуд",
    "выпадение": "выпадение волос",
    "от выпадения": "выпадение волос",
    "восстановить": "восстановление",
    "питание": "питание",
    "питать": "питание",
    "очищение": "очищение",
    "очистить": "очищение",
    "антиэйдж": "антивозрастной уход",
    "морщины": "морщины",
}

AVOID_GROUPS = {
    # Группы для ограничений по составу и аллергий.
    # Важно: аллергии проверяются не только по чистому составу, но и по названию,
    # описанию, активным компонентам, аллергенам и ссылке товара.
    "лаванда": [
        "лаванда",
        "лаванды",
        "лаванд",
        "лавандовое",
        "лавандовая",
        "масло лаванды",
        "эфирное масло лаванды",
        "lavanda",
        "lavandy",
    ],
    "роза": ["роза", "розы", "розовое", "масло розы", "экстракт розы"],
    "розмарин": ["розмарин", "розмарина", "масло розмарина", "эфирное масло розмарина"],
    "шалфей": ["шалфей", "шалфея", "масло шалфея", "эфирное масло шалфея"],
    "иссоп": ["иссоп", "иссопа", "масло иссопа", "эфирное масло иссопа"],
    "глициния": ["глициния", "глицинии", "масло глицинии"],
    "прополис": ["прополис", "прополиса", "экстракт прополиса"],
    "мед": ["мед", "меда", "медовый", "медовая", "медовое", "медовый"],
    "мёд": ["мед", "меда", "медовый", "медовая", "медовое"],
    "пчелиная пыльца": ["пчелиная пыльца", "пыльца"],
    "сульфаты": [
        "натрия лауретсульфат",
        "натрия лаурилсульфат",
        "sls",
        "sles",
        "sodium lauryl sulfate",
        "sodium laureth sulfate",
    ],
    "sls": ["натрия лаурилсульфат", "sodium lauryl sulfate", "sls"],
    "sles": ["натрия лауретсульфат", "sodium laureth sulfate", "sles"],
    "отдушки": ["парфюмерная композиция", "отдушка", "отдушки", "ароматизатор"],
    "эфирные масла": ["эфирное масло", "эфирные масла"],
    "спирты": ["бензиловый спирт", "цетеариловый спирт", "цетиловый спирт"],
}


# Категорийные справочники для интерфейса. Они убирают нерелевантные теги из формы:
# например, «возрастные изменения» и «антивозрастной уход» не показываются в подборе для волос.
CATEGORY_ALLOWED = {
    "волосы": {
        "concerns": {
            "выпадение волос", "перхоть", "зуд", "шелушение", "жирность",
            "сухость", "ломкость", "повреждение", "отсутствие объема",
            "раздражение", "чувствительность", "дерматит", "псориаз",
            "секущиеся кончики", "тусклость", "ослабленные волосы",
        },
        "effects": {
            "очищение", "увлажнение", "питание", "восстановление", "укрепление",
            "рост волос", "блеск", "объем", "снятие раздражения", "успокаивающий эффект",
            "регуляция жирности", "против перхоти", "легкое расчесывание", "смягчение",
        },
    },
    "лицо": {
        "concerns": {
            "акне", "высыпания", "воспаления", "покраснение", "раздражение",
            "сухость", "шелушение", "жирность", "расширенные поры", "постакне",
            "тусклый тон", "возрастные изменения", "морщины", "отечность", "чувствительность",
        },
        "effects": {
            "увлажнение", "питание", "восстановление", "очищение", "снятие раздражения",
            "успокаивающий эффект", "антибактериальный эффект", "антивозрастной уход",
            "выравнивание тона", "сияние", "смягчение", "матирование", "сужение пор",
            "защита", "регенерация",
        },
    },
    "тело": {
        "concerns": {
            "сухость", "шелушение", "раздражение", "чувствительность", "дерматит",
            "псориаз", "целлюлит", "растяжки", "огрубение", "трещины",
            "потеря упругости", "возрастные изменения", "воспаления",
        },
        "effects": {
            "очищение", "увлажнение", "питание", "смягчение", "восстановление",
            "снятие раздражения", "успокаивающий эффект", "тонизирование", "упругость",
            "антицеллюлитный эффект", "расслабление", "регенерация", "защита",
            "антивозрастной уход",
        },
    },
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).replace("\xa0", " ").lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_pipe_list(value: Any) -> List[str]:
    text = normalize_text(value)
    if not text or text == "nan":
        return []
    parts = [p.strip() for p in re.split(r"\s*\|\s*|,\s*", text) if p.strip()]
    seen = set()
    result = []
    for p in parts:
        p = ALIASES.get(p, p)
        if p not in seen:
            result.append(p)
            seen.add(p)
    return result


def normalize_list(values: Optional[Iterable[Any]]) -> List[str]:
    if not values:
        return []
    result = []
    for value in values:
        text = normalize_text(value)
        if not text:
            continue
        result.append(ALIASES.get(text, text))
    seen = set()
    out = []
    for item in result:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def expand_avoid_terms(values: Optional[Iterable[str]]) -> List[str]:
    values = normalize_list(values)
    expanded = []
    for value in values:
        expanded.append(value)
        expanded.extend(AVOID_GROUPS.get(value, []))
    seen = set()
    out = []
    for item in expanded:
        item = normalize_text(item)
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def contains_any(text: str, words: Iterable[str]) -> bool:
    """Проверяет наличие запрещенных слов/фраз в тексте.

    Для длинных терминов используется обычное вхождение подстроки, чтобы ловить
    формы вроде «лаванды», «лавандовый» через корень «лаванд». Для очень коротких
    слов используется граница слова, чтобы не ловить случайные совпадения внутри
    других слов.
    """
    text = normalize_text(text)
    if not text:
        return False

    for word in words:
        term = normalize_text(word)
        if not term:
            continue
        if len(term) <= 3:
            if re.search(rf"(?<![а-яa-z0-9]){re.escape(term)}(?![а-яa-z0-9])", text):
                return True
        elif term in text:
            return True
    return False


def build_safety_text(row: pd.Series) -> str:
    """Собирает широкий текст для проверки аллергий и запретов.

    Аллергии должны исключать товар даже тогда, когда компонент указан не в
    ingredients_text, а в названии, описании, активных компонентах, аллергенах,
    ml_features_text или URL.
    """
    fields = [
        "product_name",
        "ingredients_text",
        "ingredients_list",
        "active_ingredients",
        "potential_allergens_or_irritants",
        "clean_description",
        "raw_description",
        "recommendation_text",
        "ml_features_text",
        "source_url",
    ]
    parts = []
    for field in fields:
        value = row.get(field, "")
        if isinstance(value, list):
            parts.extend(normalize_text(x) for x in value)
        else:
            parts.append(normalize_text(value))
    return " ".join(p for p in parts if p)


def list_overlap(product_values: Iterable[str], user_values: Iterable[str]) -> List[str]:
    product_set = set(normalize_list(product_values))
    user_set = set(normalize_list(user_values))
    return sorted(product_set & user_set)


@dataclass
class UserRequest:
    category: Optional[str] = None
    product_type: Optional[Union[str, List[str]]] = None
    product_types: Optional[List[str]] = None
    target_area: Optional[str] = None
    skin_type: Optional[str] = None
    hair_type: Optional[str] = None
    concerns: Optional[List[str]] = None
    desired_effects: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    avoid_ingredients: Optional[List[str]] = None
    query_text: Optional[str] = None


class CosmeticsRecommender:
    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        self.df = self._load_dataset(self.dataset_path)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=8000,
        )
        self.product_matrix = self.vectorizer.fit_transform(self.df["search_text"].fillna(""))

    def _load_dataset(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)

        for col in LIST_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        text_cols = [
            "product_name", "category", "subcategory", "product_type", "target_area",
            "skin_type", "hair_type", "concern_tags", "effect_tags", "active_ingredients",
            "ingredients_text", "ingredients_list", "potential_allergens_or_irritants",
            "clean_description", "recommendation_text", "ml_features_text"
        ]
        for col in text_cols:
            if col not in df.columns:
                df[col] = ""

        df["search_text"] = (
            df["product_name"].fillna("") + " " +
            df["category"].fillna("") + " " +
            df["subcategory"].fillna("") + " " +
            df["product_type"].fillna("") + " " +
            df["target_area"].fillna("") + " " +
            df["skin_type"].fillna("") + " " +
            df["hair_type"].fillna("") + " " +
            df["concern_tags"].fillna("") + " " +
            df["effect_tags"].fillna("") + " " +
            df["active_ingredients"].fillna("") + " " +
            df["clean_description"].fillna("") + " " +
            df["ml_features_text"].fillna("")
        ).map(normalize_text)
        return df

    def _request_product_types(self, request: UserRequest) -> List[str]:
        values: List[str] = []
        if request.product_types:
            values.extend(request.product_types)
        if request.product_type:
            if isinstance(request.product_type, list):
                values.extend(request.product_type)
            else:
                values.append(request.product_type)
        return normalize_list(values)

    def _build_query_text(self, request: UserRequest) -> str:
        product_types = self._request_product_types(request)
        parts = [
            request.category,
            " ".join(product_types),
            request.target_area,
            request.skin_type,
            request.hair_type,
            " ".join(request.concerns or []),
            " ".join(request.desired_effects or []),
            request.query_text,
        ]
        return normalize_text(" ".join([str(p) for p in parts if p]))

    def recommend(
        self,
        request: Dict[str, Any] | UserRequest,
        top_n: int = 5,
        min_score: float = 1.0,
        strict_product_type: bool = False,
        strict_category: bool = True,
    ) -> List[Dict[str, Any]]:
        if isinstance(request, dict):
            request = UserRequest(**request)

        category = ALIASES.get(normalize_text(request.category), normalize_text(request.category))
        product_types = self._request_product_types(request)
        target_area = ALIASES.get(normalize_text(request.target_area), normalize_text(request.target_area))
        skin_type = ALIASES.get(normalize_text(request.skin_type), normalize_text(request.skin_type))
        hair_type = ALIASES.get(normalize_text(request.hair_type), normalize_text(request.hair_type))

        concerns = normalize_list(request.concerns)
        desired_effects = normalize_list(request.desired_effects)
        allergies = expand_avoid_terms(request.allergies)
        avoid_ingredients = expand_avoid_terms(request.avoid_ingredients)

        query_text = self._build_query_text(request)
        query_vector = self.vectorizer.transform([query_text])
        similarity_scores = cosine_similarity(query_vector, self.product_matrix).ravel()

        recommendations = []

        for idx, row in self.df.iterrows():
            score = 0.0
            reasons = []
            warnings = []

            row_category = normalize_text(row.get("category"))
            row_product_type = normalize_text(row.get("product_type"))
            row_ingredients = normalize_text(row.get("ingredients_text"))
            row_ingredient_list = normalize_text(row.get("ingredients_list"))
            row_allergens = normalize_text(row.get("potential_allergens_or_irritants"))
            safety_text = build_safety_text(row)

            if category:
                if row_category == category:
                    score += 8
                    reasons.append(f"категория совпадает: {category}")
                elif strict_category:
                    continue
                else:
                    score -= 3

            if product_types:
                if row_product_type in product_types:
                    score += 8
                    reasons.append(f"тип продукта совпадает: {row_product_type}")
                elif strict_product_type:
                    continue
                else:
                    score -= 2

            # Аллергии: персональная непереносимость. Это всегда жесткое исключение.
            # Проверяем широко: название, состав, активные компоненты, описание, теги и URL.
            # Так запрос «лаванда» отсекает товары с «Лаванда» даже если чистый состав пустой.
            if allergies and contains_any(safety_text, allergies):
                continue

            # Нежелательные компоненты: пользовательское ограничение/предпочтение. Тоже исключаем.
            if avoid_ingredients and contains_any(safety_text, avoid_ingredients):
                continue

            if target_area:
                areas = split_pipe_list(row.get("target_area"))
                if target_area in areas or target_area in normalize_text(row.get("target_area")):
                    score += 6
                    reasons.append(f"подходит для зоны: {target_area}")

            if skin_type:
                product_skin = split_pipe_list(row.get("skin_type"))
                if skin_type in product_skin or skin_type in normalize_text(row.get("skin_type")):
                    score += 5
                    reasons.append(f"подходит для типа кожи: {skin_type}")

            if hair_type:
                product_hair = split_pipe_list(row.get("hair_type"))
                if hair_type in product_hair or hair_type in normalize_text(row.get("hair_type")):
                    score += 5
                    reasons.append(f"подходит для типа волос: {hair_type}")

            product_concerns = split_pipe_list(row.get("concern_tags"))
            matched_concerns = list_overlap(product_concerns, concerns)
            if matched_concerns:
                score += 7 * len(matched_concerns)
                reasons.append("совпадают проблемы: " + ", ".join(matched_concerns))

            product_effects = split_pipe_list(row.get("effect_tags"))
            matched_effects = list_overlap(product_effects, desired_effects)
            if matched_effects:
                score += 5 * len(matched_effects)
                reasons.append("совпадают эффекты: " + ", ".join(matched_effects))

            semantic_score = float(similarity_scores[idx]) * 15
            score += semantic_score
            if semantic_score >= 2:
                reasons.append("похож по текстовому описанию запроса")

            if row_allergens:
                warnings.append("проверьте возможные раздражители/аллергены: " + str(row.get("potential_allergens_or_irritants")))

            data_quality = normalize_text(row.get("data_quality"))
            if data_quality == "poor":
                score -= 4
                warnings.append("у карточки неполные данные")
            elif data_quality == "partial":
                score -= 1
                warnings.append("у карточки частично заполнены данные")

            if score >= min_score:
                recommendations.append({
                    "id": row.get("id"),
                    "product_name": row.get("product_name"),
                    "category": row.get("category"),
                    "subcategory": row.get("subcategory"),
                    "product_type": row.get("product_type"),
                    "target_area": row.get("target_area"),
                    "skin_type": row.get("skin_type"),
                    "hair_type": row.get("hair_type"),
                    "concern_tags": row.get("concern_tags"),
                    "effect_tags": row.get("effect_tags"),
                    "active_ingredients": row.get("active_ingredients"),
                    "ingredients_text": row.get("ingredients_text"),
                    "recommendation_text": row.get("recommendation_text"),
                    "score": round(score, 2),
                    "semantic_score": round(semantic_score, 2),
                    "reasons": reasons,
                    "warnings": warnings,
                    "source_url": row.get("source_url"),
                    "main_image": row.get("main_image"),
                    "data_quality": row.get("data_quality"),
                })

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:top_n]

    def available_options(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        df = self.df.copy()
        category_norm = normalize_text(category)
        if category_norm:
            df = df[df["category"].map(normalize_text) == category_norm]

        def uniq(col: str) -> List[str]:
            if col not in df.columns:
                return []
            values = []
            for item in df[col].dropna().astype(str).tolist():
                values.extend(split_pipe_list(item))
            values = [v for v in values if v and v != "nan"]

            # Для проблем и эффектов применяем категорийную онтологию, чтобы в волосах
            # не появлялись теги лица/тела вроде «возрастные изменения».
            if category_norm in CATEGORY_ALLOWED and col == "concern_tags":
                allowed = CATEGORY_ALLOWED[category_norm]["concerns"]
                values = [v for v in values if v in allowed]
            elif category_norm in CATEGORY_ALLOWED and col == "effect_tags":
                allowed = CATEGORY_ALLOWED[category_norm]["effects"]
                values = [v for v in values if v in allowed]

            return sorted(set(values))

        categories = sorted(set(normalize_text(x) for x in self.df["category"].dropna().astype(str).tolist() if normalize_text(x)))
        return {
            "categories": categories,
            "product_types": uniq("product_type"),
            "target_areas": uniq("target_area"),
            "skin_types": uniq("skin_type"),
            "hair_types": uniq("hair_type") if category_norm in ["", "волосы"] else [],
            "concerns": uniq("concern_tags"),
            "effects": uniq("effect_tags"),
            "potential_allergens": uniq("potential_allergens_or_irritants"),
        }


def print_recommendations(items: List[Dict[str, Any]]) -> None:
    if not items:
        print("Подходящих товаров не найдено. Попробуйте ослабить фильтры.")
        return

    for i, item in enumerate(items, start=1):
        print(f"\n{i}. {item['product_name']}")
        print(f"   Балл: {item['score']}")
        print(f"   Категория: {item.get('category')} / {item.get('product_type')}")
        if item.get("recommendation_text"):
            print(f"   Описание подбора: {item['recommendation_text']}")
        if item["reasons"]:
            print("   Почему подходит: " + "; ".join(item["reasons"]))
        if item["warnings"]:
            print("   Предупреждения: " + "; ".join(item["warnings"]))
        if item.get("source_url"):
            print(f"   Ссылка: {item['source_url']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Рекомендательная система для косметики")
    parser.add_argument("--dataset", default="socrimea_cosmetics_dataset.csv", help="Путь к CSV/XLSX датасету")
    parser.add_argument("--request", default=None, help="JSON-строка с параметрами пользователя")
    parser.add_argument("--top", type=int, default=5, help="Сколько рекомендаций вывести")
    args = parser.parse_args()

    recommender = CosmeticsRecommender(args.dataset)

    if args.request:
        request = json.loads(args.request)
    else:
        request = {
            "category": "волосы",
            "product_types": ["шампунь", "тоник"],
            "target_area": "кожа головы",
            "skin_type": "чувствительная",
            "concerns": ["перхоть", "зуд", "шелушение"],
            "desired_effects": ["очищение", "снятие раздражения"],
            "allergies": [],
            "avoid_ingredients": [],
            "query_text": "нужен шампунь или тоник для чувствительной кожи головы от перхоти и зуда",
        }

    results = recommender.recommend(args.request if False else request, top_n=args.top)
    print_recommendations(results)


if __name__ == "__main__":
    main()
