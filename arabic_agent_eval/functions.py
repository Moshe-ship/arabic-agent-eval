"""Arabic function/tool definitions for evaluation.

These are the tools available to agents during evaluation.
Each function has an Arabic description and Arabic parameter names alongside English ones.
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
    """Build an OpenAI-compatible function definition."""
    props = {}
    for pname, pinfo in parameters.items():
        props[pname] = {
            "type": pinfo.get("type", "string"),
            "description": pinfo.get("description", ""),
        }
        if "enum" in pinfo:
            props[pname]["enum"] = pinfo["enum"]

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


FUNCTIONS: list[dict] = [
    _func(
        name="search_flights",
        name_ar="البحث عن رحلات",
        description="Search for flights between two cities",
        description_ar="البحث عن رحلات طيران بين مدينتين",
        parameters={
            "from_city": {"type": "string", "description": "Departure city"},
            "to_city": {"type": "string", "description": "Arrival city"},
            "date": {"type": "string", "description": "Travel date"},
            "passengers": {"type": "integer", "description": "Number of passengers"},
        },
        required=["from_city", "to_city", "date"],
    ),
    _func(
        name="book_hotel",
        name_ar="حجز فندق",
        description="Book a hotel room",
        description_ar="حجز غرفة في فندق",
        parameters={
            "city": {"type": "string", "description": "City name"},
            "check_in": {"type": "string", "description": "Check-in date"},
            "check_out": {"type": "string", "description": "Check-out date"},
            "guests": {"type": "integer", "description": "Number of guests"},
        },
        required=["city", "check_in", "check_out"],
    ),
    _func(
        name="send_message",
        name_ar="إرسال رسالة",
        description="Send a message to a contact",
        description_ar="إرسال رسالة إلى جهة اتصال",
        parameters={
            "recipient": {"type": "string", "description": "Recipient name or number"},
            "platform": {
                "type": "string",
                "description": "Messaging platform",
                "enum": ["whatsapp", "sms", "telegram", "email"],
            },
            "message": {"type": "string", "description": "Message content"},
        },
    ),
    _func(
        name="get_weather",
        name_ar="حالة الطقس",
        description="Get current weather for a city",
        description_ar="الحصول على حالة الطقس الحالية لمدينة",
        parameters={
            "city": {"type": "string", "description": "City name"},
        },
    ),
    _func(
        name="search_restaurants",
        name_ar="البحث عن مطاعم",
        description="Search for restaurants by cuisine or location",
        description_ar="البحث عن مطاعم حسب نوع المطبخ أو الموقع",
        parameters={
            "city": {"type": "string", "description": "City name"},
            "cuisine": {"type": "string", "description": "Type of cuisine"},
            "budget": {
                "type": "string",
                "description": "Budget level",
                "enum": ["cheap", "moderate", "expensive"],
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
            "restaurant": {"type": "string", "description": "Restaurant name"},
            "date": {"type": "string", "description": "Reservation date"},
            "time": {"type": "string", "description": "Reservation time"},
            "guests": {"type": "integer", "description": "Number of guests"},
        },
    ),
    _func(
        name="get_prayer_times",
        name_ar="مواقيت الصلاة",
        description="Get prayer times for a city",
        description_ar="الحصول على مواقيت الصلاة لمدينة",
        parameters={
            "city": {"type": "string", "description": "City name"},
            "date": {"type": "string", "description": "Date (optional, defaults to today)"},
        },
        required=["city"],
    ),
    _func(
        name="convert_currency",
        name_ar="تحويل العملات",
        description="Convert between currencies",
        description_ar="تحويل بين العملات",
        parameters={
            "amount": {"type": "number", "description": "Amount to convert"},
            "from_currency": {"type": "string", "description": "Source currency code"},
            "to_currency": {"type": "string", "description": "Target currency code"},
        },
    ),
    _func(
        name="translate_text",
        name_ar="ترجمة النص",
        description="Translate text between languages",
        description_ar="ترجمة نص بين اللغات",
        parameters={
            "text": {"type": "string", "description": "Text to translate"},
            "from_lang": {"type": "string", "description": "Source language"},
            "to_lang": {"type": "string", "description": "Target language"},
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
            },
            "country": {"type": "string", "description": "Country code"},
        },
        required=["category"],
    ),
    _func(
        name="set_reminder",
        name_ar="تعيين تذكير",
        description="Set a reminder for a specific time",
        description_ar="تعيين تذكير في وقت محدد",
        parameters={
            "message": {"type": "string", "description": "Reminder message"},
            "datetime": {"type": "string", "description": "When to remind"},
        },
    ),
    _func(
        name="calculate_zakat",
        name_ar="حساب الزكاة",
        description="Calculate zakat on wealth",
        description_ar="حساب الزكاة على الأموال",
        parameters={
            "amount": {"type": "number", "description": "Total wealth amount"},
            "currency": {"type": "string", "description": "Currency code"},
            "type": {
                "type": "string",
                "description": "Type of wealth",
                "enum": ["cash", "gold", "silver", "stocks", "business"],
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
            "query": {"type": "string", "description": "Search query"},
            "surah": {"type": "integer", "description": "Surah number (optional)"},
        },
        required=["query"],
    ),
    _func(
        name="get_stock_price",
        name_ar="سعر السهم",
        description="Get current stock price",
        description_ar="الحصول على سعر السهم الحالي",
        parameters={
            "symbol": {"type": "string", "description": "Stock ticker symbol"},
            "market": {
                "type": "string",
                "description": "Stock market",
                "enum": ["tadawul", "adx", "dfm", "nasdaq", "nyse"],
            },
        },
    ),
    _func(
        name="order_food",
        name_ar="طلب طعام",
        description="Order food delivery",
        description_ar="طلب توصيل طعام",
        parameters={
            "restaurant": {"type": "string", "description": "Restaurant name"},
            "items": {"type": "string", "description": "Food items to order"},
            "address": {"type": "string", "description": "Delivery address"},
        },
    ),
    _func(
        name="get_traffic",
        name_ar="حالة المرور",
        description="Get traffic conditions between two points",
        description_ar="الحصول على حالة المرور بين نقطتين",
        parameters={
            "from_location": {"type": "string", "description": "Starting point"},
            "to_location": {"type": "string", "description": "Destination"},
        },
    ),
    _func(
        name="schedule_meeting",
        name_ar="جدولة اجتماع",
        description="Schedule a meeting with participants",
        description_ar="جدولة اجتماع مع المشاركين",
        parameters={
            "title": {"type": "string", "description": "Meeting title"},
            "date": {"type": "string", "description": "Meeting date"},
            "time": {"type": "string", "description": "Meeting time"},
            "participants": {"type": "string", "description": "Participant names"},
            "location": {"type": "string", "description": "Meeting location or link"},
        },
        required=["title", "date", "time", "participants"],
    ),
    _func(
        name="send_money",
        name_ar="تحويل أموال",
        description="Send money to a recipient",
        description_ar="تحويل أموال إلى مستلم",
        parameters={
            "recipient": {"type": "string", "description": "Recipient name"},
            "amount": {"type": "number", "description": "Amount to send"},
            "currency": {"type": "string", "description": "Currency code"},
        },
    ),
    _func(
        name="get_time",
        name_ar="الوقت الحالي",
        description="Get current time in a city",
        description_ar="الحصول على الوقت الحالي في مدينة",
        parameters={
            "city": {"type": "string", "description": "City name"},
        },
    ),
    _func(
        name="check_visa_status",
        name_ar="حالة التأشيرة",
        description="Check visa application status",
        description_ar="التحقق من حالة طلب التأشيرة",
        parameters={
            "application_id": {"type": "string", "description": "Application ID"},
            "passport_number": {"type": "string", "description": "Passport number"},
        },
    ),
    _func(
        name="book_car",
        name_ar="حجز سيارة",
        description="Book a ride or rent a car",
        description_ar="حجز رحلة أو استئجار سيارة",
        parameters={
            "pickup": {"type": "string", "description": "Pickup location"},
            "destination": {"type": "string", "description": "Destination"},
            "type": {
                "type": "string",
                "description": "Service type",
                "enum": ["ride", "rental"],
            },
        },
    ),
    _func(
        name="search_jobs",
        name_ar="البحث عن وظائف",
        description="Search for job listings",
        description_ar="البحث عن فرص عمل",
        parameters={
            "title": {"type": "string", "description": "Job title or keyword"},
            "city": {"type": "string", "description": "City name"},
            "type": {
                "type": "string",
                "description": "Employment type",
                "enum": ["full-time", "part-time", "remote", "contract"],
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
