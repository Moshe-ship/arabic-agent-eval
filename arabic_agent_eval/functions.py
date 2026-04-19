"""Arabic function/tool definitions for evaluation.

These are the tools available to agents during evaluation. Each function
has an Arabic description and Arabic parameter names alongside English
ones, plus `x-mtg` annotations declaring linguistic slot constraints
on every parameter. The x-mtg coverage drives `schema_bound_rate`
toward 1.0 on default-map runs — eval results don't need a separate
Hurmoz schema map to produce schema-grounded numbers.

Slot type choices, applied consistently across the registry:
- `named_entity` (ar) for cities / person names / brand names that must
  stay in Arabic script and not be transliterated.
- `free_text` (ar) for message bodies / search queries / translation
  input — Arabic expected but less structured than named entities.
- `temporal` (any) for dates / times — language-agnostic typography.
- `numeric` (any) for counts / amounts — `_free_text_overflow` suppressed.
- `identifier` (latn) for opaque codes, enums, currency codes, ISO
  country codes, ticker symbols — pure-Latin by convention.
"""

from __future__ import annotations

from typing import Any


def _func(
    name: str,
    name_ar: str,
    description: str,
    description_ar: str,
    parameters: dict[str, dict],
    required: list[str] | None = None,
) -> dict:
    """Build an OpenAI-compatible function definition.

    Each `parameters` entry may declare an `x_mtg` key; it is lifted
    onto the emitted JSON Schema property as `x-mtg` (the on-wire
    JSON Schema extension name).
    """
    props = {}
    for pname, pinfo in parameters.items():
        props[pname] = {
            "type": pinfo.get("type", "string"),
            "description": pinfo.get("description", ""),
        }
        if "enum" in pinfo:
            props[pname]["enum"] = pinfo["enum"]
        if "x_mtg" in pinfo:
            props[pname]["x-mtg"] = pinfo["x_mtg"]

    return {
        "name": name,
        "name_ar": name_ar,
        "description": description,
        "description_ar": description_ar,
        "parameters": {
            "type": "object",
            "properties": props,
            "required": required or list(parameters.keys()),
        },
    }


# Reusable x-mtg presets — keeping them centralized so a tweak to one
# slot type ripples across every tool that uses it.
_NAMED_ENTITY_AR = {
    "slot_type": "named_entity",
    "script": "ar",
    "transliteration_allowed": False,
    "mode": "advisory",
    "post_call_contract": ["script_match", "no_surface_corruption"],
}
_FREE_TEXT_AR = {
    "slot_type": "free_text",
    "script": "ar",
    "transliteration_allowed": False,
    "mode": "advisory",
    "post_call_contract": ["script_match", "no_surface_corruption"],
}
_FREE_TEXT_AR_MSA = {
    "slot_type": "free_text",
    "script": "ar",
    "dialect_expected": "msa",
    "dialect_enforcement": "preserve",
    "transliteration_allowed": False,
    "mode": "advisory",
    "post_call_contract": ["script_match", "no_surface_corruption"],
}
_MIXED_FREE_TEXT = {
    "slot_type": "free_text",
    "script": "mixed",
    "transliteration_allowed": True,
    "mode": "advisory",
}
_TEMPORAL = {
    "slot_type": "temporal",
    "script": "any",
    "mode": "advisory",
}
_NUMERIC = {
    "slot_type": "numeric",
    "script": "any",
    "mode": "advisory",
}
_IDENTIFIER_LATN = {
    "slot_type": "identifier",
    "script": "latn",
    "mode": "advisory",
}


FUNCTIONS: list[dict] = [
    _func(
        name="search_flights",
        name_ar="البحث عن رحلات",
        description="Search for flights between two cities",
        description_ar="البحث عن رحلات طيران بين مدينتين",
        parameters={
            "from_city": {"type": "string", "description": "Departure city",
                           "x_mtg": _NAMED_ENTITY_AR},
            "to_city": {"type": "string", "description": "Arrival city",
                         "x_mtg": _NAMED_ENTITY_AR},
            "date": {"type": "string", "description": "Travel date",
                      "x_mtg": _TEMPORAL},
            "passengers": {"type": "integer", "description": "Number of passengers",
                            "x_mtg": _NUMERIC},
        },
        required=["from_city", "to_city", "date"],
    ),
    _func(
        name="book_hotel",
        name_ar="حجز فندق",
        description="Book a hotel room",
        description_ar="حجز غرفة في فندق",
        parameters={
            "city": {"type": "string", "description": "City name",
                      "x_mtg": _NAMED_ENTITY_AR},
            "check_in": {"type": "string", "description": "Check-in date",
                          "x_mtg": _TEMPORAL},
            "check_out": {"type": "string", "description": "Check-out date",
                           "x_mtg": _TEMPORAL},
            "guests": {"type": "integer", "description": "Number of guests",
                        "x_mtg": _NUMERIC},
        },
        required=["city", "check_in", "check_out"],
    ),
    _func(
        name="send_message",
        name_ar="إرسال رسالة",
        description="Send a message to a contact",
        description_ar="إرسال رسالة إلى جهة اتصال",
        parameters={
            "recipient": {"type": "string",
                           "description": "Recipient name or number",
                           "x_mtg": {**_NAMED_ENTITY_AR, "script": "mixed",
                                      "transliteration_allowed": True}},
            "platform": {
                "type": "string",
                "description": "Messaging platform",
                "enum": ["whatsapp", "sms", "telegram", "email"],
                "x_mtg": _IDENTIFIER_LATN,
            },
            "message": {"type": "string", "description": "Message content",
                         "x_mtg": _FREE_TEXT_AR},
        },
    ),
    _func(
        name="get_weather",
        name_ar="حالة الطقس",
        description="Get current weather for a city",
        description_ar="الحصول على حالة الطقس الحالية لمدينة",
        parameters={
            "city": {"type": "string", "description": "City name",
                      "x_mtg": _NAMED_ENTITY_AR},
        },
    ),
    _func(
        name="search_restaurants",
        name_ar="البحث عن مطاعم",
        description="Search for restaurants by cuisine or location",
        description_ar="البحث عن مطاعم حسب نوع المطبخ أو الموقع",
        parameters={
            "city": {"type": "string", "description": "City name",
                      "x_mtg": _NAMED_ENTITY_AR},
            "cuisine": {"type": "string", "description": "Type of cuisine",
                         "x_mtg": _FREE_TEXT_AR},
            "budget": {
                "type": "string",
                "description": "Budget level",
                "enum": ["cheap", "moderate", "expensive"],
                "x_mtg": _IDENTIFIER_LATN,
            },
        },
        required=["city"],
    ),
    _func(
        name="book_table",
        name_ar="حجز طاولة",
        description="Reserve a table at a restaurant",
        description_ar="حجز طاولة في مطعم",
        parameters={
            "restaurant": {"type": "string", "description": "Restaurant name",
                            "x_mtg": _NAMED_ENTITY_AR},
            "date": {"type": "string", "description": "Reservation date",
                      "x_mtg": _TEMPORAL},
            "time": {"type": "string", "description": "Reservation time",
                      "x_mtg": _TEMPORAL},
            "guests": {"type": "integer", "description": "Number of guests",
                        "x_mtg": _NUMERIC},
        },
    ),
    _func(
        name="get_prayer_times",
        name_ar="مواقيت الصلاة",
        description="Get prayer times for a city",
        description_ar="الحصول على مواقيت الصلاة لمدينة",
        parameters={
            "city": {"type": "string", "description": "City name",
                      "x_mtg": _NAMED_ENTITY_AR},
            "date": {"type": "string",
                      "description": "Date (optional, defaults to today)",
                      "x_mtg": _TEMPORAL},
        },
        required=["city"],
    ),
    _func(
        name="convert_currency",
        name_ar="تحويل العملات",
        description="Convert between currencies",
        description_ar="تحويل بين العملات",
        parameters={
            "amount": {"type": "number", "description": "Amount to convert",
                        "x_mtg": _NUMERIC},
            "from_currency": {"type": "string", "description": "Source currency code",
                               "x_mtg": _IDENTIFIER_LATN},
            "to_currency": {"type": "string", "description": "Target currency code",
                             "x_mtg": _IDENTIFIER_LATN},
        },
    ),
    _func(
        name="translate_text",
        name_ar="ترجمة النص",
        description="Translate text between languages",
        description_ar="ترجمة نص بين اللغات",
        parameters={
            "text": {"type": "string", "description": "Text to translate",
                      "x_mtg": _MIXED_FREE_TEXT},
            "from_lang": {"type": "string", "description": "Source language",
                           "x_mtg": _IDENTIFIER_LATN},
            "to_lang": {"type": "string", "description": "Target language",
                         "x_mtg": _IDENTIFIER_LATN},
        },
    ),
    _func(
        name="get_news",
        name_ar="آخر الأخبار",
        description="Get latest news headlines",
        description_ar="الحصول على آخر عناوين الأخبار",
        parameters={
            "category": {
                "type": "string",
                "description": "News category",
                "enum": ["politics", "sports", "technology", "business", "entertainment"],
                "x_mtg": _IDENTIFIER_LATN,
            },
            "country": {"type": "string", "description": "Country code",
                         "x_mtg": _IDENTIFIER_LATN},
        },
        required=["category"],
    ),
    _func(
        name="set_reminder",
        name_ar="تعيين تذكير",
        description="Set a reminder for a specific time",
        description_ar="تعيين تذكير في وقت محدد",
        parameters={
            "message": {"type": "string", "description": "Reminder message",
                         "x_mtg": _FREE_TEXT_AR},
            "datetime": {"type": "string", "description": "When to remind",
                          "x_mtg": _TEMPORAL},
        },
    ),
    _func(
        name="calculate_zakat",
        name_ar="حساب الزكاة",
        description="Calculate zakat on wealth",
        description_ar="حساب الزكاة على الأموال",
        parameters={
            "amount": {"type": "number", "description": "Total wealth amount",
                        "x_mtg": _NUMERIC},
            "currency": {"type": "string", "description": "Currency code",
                          "x_mtg": _IDENTIFIER_LATN},
            "type": {
                "type": "string",
                "description": "Type of wealth",
                "enum": ["cash", "gold", "silver", "stocks", "business"],
                "x_mtg": _IDENTIFIER_LATN,
            },
        },
        required=["amount", "currency"],
    ),
    _func(
        name="find_quran_verse",
        name_ar="البحث في القرآن",
        description="Search for a Quran verse by text or topic",
        description_ar="البحث عن آية قرآنية بالنص أو الموضوع",
        parameters={
            "query": {"type": "string", "description": "Search query",
                       "x_mtg": _FREE_TEXT_AR_MSA},
            "surah": {"type": "integer",
                       "description": "Surah number (optional)",
                       "x_mtg": _NUMERIC},
        },
        required=["query"],
    ),
    _func(
        name="get_stock_price",
        name_ar="سعر السهم",
        description="Get current stock price",
        description_ar="الحصول على سعر السهم الحالي",
        parameters={
            "symbol": {"type": "string", "description": "Stock ticker symbol",
                        "x_mtg": _IDENTIFIER_LATN},
            "market": {
                "type": "string",
                "description": "Stock market",
                "enum": ["tadawul", "adx", "dfm", "nasdaq", "nyse"],
                "x_mtg": _IDENTIFIER_LATN,
            },
        },
    ),
    _func(
        name="order_food",
        name_ar="طلب طعام",
        description="Order food delivery",
        description_ar="طلب توصيل طعام",
        parameters={
            "restaurant": {"type": "string", "description": "Restaurant name",
                            "x_mtg": _NAMED_ENTITY_AR},
            "items": {"type": "string", "description": "Food items to order",
                       "x_mtg": _FREE_TEXT_AR},
            "address": {"type": "string", "description": "Delivery address",
                         "x_mtg": _MIXED_FREE_TEXT},
        },
    ),
    _func(
        name="get_traffic",
        name_ar="حالة المرور",
        description="Get traffic conditions between two points",
        description_ar="الحصول على حالة المرور بين نقطتين",
        parameters={
            "from_location": {"type": "string", "description": "Starting point",
                               "x_mtg": _NAMED_ENTITY_AR},
            "to_location": {"type": "string", "description": "Destination",
                             "x_mtg": _NAMED_ENTITY_AR},
        },
    ),
    _func(
        name="schedule_meeting",
        name_ar="جدولة اجتماع",
        description="Schedule a meeting with participants",
        description_ar="جدولة اجتماع مع المشاركين",
        parameters={
            "title": {"type": "string", "description": "Meeting title",
                       "x_mtg": _FREE_TEXT_AR},
            "date": {"type": "string", "description": "Meeting date",
                      "x_mtg": _TEMPORAL},
            "time": {"type": "string", "description": "Meeting time",
                      "x_mtg": _TEMPORAL},
            "participants": {"type": "string",
                              "description": "Participant names",
                              "x_mtg": _FREE_TEXT_AR},
            "location": {"type": "string",
                          "description": "Meeting location or link",
                          "x_mtg": _MIXED_FREE_TEXT},
        },
        required=["title", "date", "time", "participants"],
    ),
    _func(
        name="send_money",
        name_ar="تحويل أموال",
        description="Send money to a recipient",
        description_ar="تحويل أموال إلى مستلم",
        parameters={
            "recipient": {"type": "string", "description": "Recipient name",
                           "x_mtg": {**_NAMED_ENTITY_AR, "script": "mixed",
                                      "transliteration_allowed": True}},
            "amount": {"type": "number", "description": "Amount to send",
                        "x_mtg": _NUMERIC},
            "currency": {"type": "string", "description": "Currency code",
                          "x_mtg": _IDENTIFIER_LATN},
        },
    ),
    _func(
        name="get_time",
        name_ar="الوقت الحالي",
        description="Get current time in a city",
        description_ar="الحصول على الوقت الحالي في مدينة",
        parameters={
            "city": {"type": "string", "description": "City name",
                      "x_mtg": _NAMED_ENTITY_AR},
        },
    ),
    _func(
        name="check_visa_status",
        name_ar="حالة التأشيرة",
        description="Check visa application status",
        description_ar="التحقق من حالة طلب التأشيرة",
        parameters={
            "application_id": {"type": "string", "description": "Application ID",
                                "x_mtg": _IDENTIFIER_LATN},
            "passport_number": {"type": "string", "description": "Passport number",
                                 "x_mtg": _IDENTIFIER_LATN},
        },
    ),
    _func(
        name="book_car",
        name_ar="حجز سيارة",
        description="Book a ride or rent a car",
        description_ar="حجز رحلة أو استئجار سيارة",
        parameters={
            "pickup": {"type": "string", "description": "Pickup location",
                        "x_mtg": _NAMED_ENTITY_AR},
            "destination": {"type": "string", "description": "Destination",
                             "x_mtg": _NAMED_ENTITY_AR},
            "type": {
                "type": "string",
                "description": "Service type",
                "enum": ["ride", "rental"],
                "x_mtg": _IDENTIFIER_LATN,
            },
        },
    ),
    _func(
        name="search_jobs",
        name_ar="البحث عن وظائف",
        description="Search for job listings",
        description_ar="البحث عن فرص عمل",
        parameters={
            "title": {"type": "string", "description": "Job title or keyword",
                       "x_mtg": _FREE_TEXT_AR},
            "city": {"type": "string", "description": "City name",
                      "x_mtg": _NAMED_ENTITY_AR},
            "type": {
                "type": "string",
                "description": "Employment type",
                "enum": ["full-time", "part-time", "remote", "contract"],
                "x_mtg": _IDENTIFIER_LATN,
            },
        },
        required=["title"],
    ),
]


def get_function_by_name(name: str) -> dict | None:
    """Get a function definition by name."""
    for f in FUNCTIONS:
        if f["name"] == name:
            return f
    return None


def get_function_names() -> list[str]:
    """Get all function names."""
    return [f["name"] for f in FUNCTIONS]


def to_openai_tools(functions: list[dict] | None = None) -> list[dict]:
    """Convert to OpenAI tools format for API calls."""
    funcs = functions or FUNCTIONS
    tools = []
    for f in funcs:
        tool = {
            "type": "function",
            "function": {
                "name": f["name"],
                "description": f["description_ar"],
                "parameters": f["parameters"],
            },
        }
        tools.append(tool)
    return tools
