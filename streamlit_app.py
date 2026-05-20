import math

import streamlit as st

from cosmetics_recommender import CosmeticsRecommender


st.set_page_config(page_title="Подбор косметики", page_icon="🧴", layout="wide")

st.title("Подбор косметики")
st.write("Выберите параметры ухода — система подберет подходящие товары из датасета.")

DATASET_PATH = "socrimea_cosmetics_dataset.csv"

def load_recommender():
    return CosmeticsRecommender(DATASET_PATH)


recommender = load_recommender()
all_options = recommender.available_options()


def split_text(value):
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        items = [str(x).strip() for x in value if str(x).strip() and str(x).strip().lower() != "nan"]
        return items

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    for sep in ["|", ";"]:
        if sep in text:
            return [x.strip() for x in text.split(sep) if x.strip() and x.strip().lower() != "nan"]
    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip() and x.strip().lower() != "nan"]
    return [text]


def format_bar_list(value):
    items = split_text(value)
    return " | ".join(items) if items else "не указаны"


def format_comma_list(value):
    items = split_text(value)
    return ", ".join(items) if items else "не указаны"


def build_short_explanation(item):
    # В карточке оставляем только короткое пояснение, без технических баллов и предупреждений.
    effects = format_comma_list(item.get("effect_tags"))
    concerns = format_comma_list(item.get("concern_tags"))
    reasons = item.get("reasons") or []

    if effects != "не указаны" and concerns != "не указаны":
        return f"подходит по проблемам: {concerns}; эффекты: {effects}"
    if effects != "не указаны":
        return f"эффекты: {effects}"
    if concerns != "не указаны":
        return f"подходит по проблемам: {concerns}"
    if reasons:
        return "; ".join(str(x) for x in reasons[:2])
    return "подходит по выбранным параметрам"


with st.sidebar:
    st.header("Параметры подбора")

    category = st.selectbox(
        "Категория ухода",
        [""] + all_options["categories"],
        help="Сначала выберите категорию: лицо, волосы или тело.",
    )

    category_options = recommender.available_options(category if category else None)

    product_types = st.multiselect(
        "Тип продукта",
        category_options["product_types"],
        help="Можно выбрать несколько типов продукта.",
    )

    target_area = None
    skin_type = None
    hair_type = None

    if category:
        target_area = st.selectbox(
            "Зона применения",
            [""] + category_options["target_areas"],
        )

    if category == "волосы":
        st.subheader("Параметры волос")
        hair_type = st.multiselect(
            "Тип / состояние волос",
            category_options["hair_types"],
            help="Можно выбрать несколько вариантов, например: сухие, тонкие, поврежденные.",
        )
        skin_type = st.selectbox("Тип / состояние кожи головы", [""] + category_options["skin_types"])

    elif category in ["лицо", "тело"]:
        st.subheader("Параметры кожи")
        skin_type = st.selectbox("Тип / состояние кожи", [""] + category_options["skin_types"])

    concerns = st.multiselect("Проблемы", category_options["concerns"])
    effects = st.multiselect("Желаемые эффекты", category_options["effects"])

    st.subheader("Ограничения по составу")
    allergies_text = st.text_input(
        "Аллергии / непереносимость",
        placeholder="например: прополис, лаванда, мед",
        help="Личная реакция: такие товары полностью исключаются.",
    )

    avoid_text = st.text_input(
        "Исключить компоненты",
        placeholder="например: сульфаты, отдушки, эфирные масла",
        help="Не аллергия, а нежелательные компоненты или предпочтения по составу.",
    )

    query_text = st.text_area(
        "Свободный запрос",
        placeholder="например: нужен шампунь для кожи головы от зуда или крем для лица от сухости",
    )

    top_n = st.slider("Количество рекомендаций", 1, 10, 5)
    strict_category = st.checkbox("Строго соблюдать категорию", value=True)
    strict_product_type = st.checkbox("Строго соблюдать выбранные типы продукта", value=False)

    run_recommendation = st.button("Подобрать косметику", type="primary", use_container_width=True)


allergies = [x.strip() for x in allergies_text.split(",") if x.strip()]
avoid_ingredients = [x.strip() for x in avoid_text.split(",") if x.strip()]

request = {
    "category": category or None,
    "product_types": product_types,
    "target_area": target_area or None,
    "skin_type": skin_type or None,
    "hair_type": hair_type or None,
    "concerns": concerns,
    "desired_effects": effects,
    "allergies": allergies,
    "avoid_ingredients": avoid_ingredients,
    "query_text": query_text or None,
}

if not run_recommendation:
    st.info("Заполните параметры в боковой панели и нажмите «Подобрать косметику».")
else:
    results = recommender.recommend(
        request,
        top_n=top_n,
        strict_category=strict_category,
        strict_product_type=strict_product_type,
    )

    if not results:
        st.warning("Подходящих товаров не найдено. Попробуйте убрать часть ограничений.")
    else:
        st.subheader("Рекомендации")
        for item in results:
            with st.container(border=True):
                c1, c2 = st.columns([1, 3], vertical_alignment="top")

                with c1:
                    image = str(item.get("main_image") or "")
                    if image and "nophoto" not in image:
                        st.image(image, use_container_width=True)
                    else:
                        st.write("Нет изображения")

                with c2:
                    st.markdown(f"### {item.get('product_name', 'Без названия')}")
                    st.write(f"{build_short_explanation(item)}")

                    if item.get("source_url"):
                        st.link_button("Открыть товар", item["source_url"])
