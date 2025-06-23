import os
import re
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import langdetect
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score
import numpy as np
from intent_model import intent_classifier


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CSV_PATH = "C:/Users/avish/OneDrive/Desktop/projectChat/AB_NYC_2019_NEW.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}. Please check the path.")

df = pd.read_csv(CSV_PATH)

df = df.dropna(subset=["host_name"])
df["name"] = df["name"].fillna("Unnamed Listing")
df = df[df["availability_365"] > 0]
df = df[df["minimum_nights"] > 0]
df = df[df["price"] <= 500]
df["value_score"] = (df["number_of_reviews"] * (df["reviews_per_month"].fillna(0) + 1)) / df["price"].clip(lower=1)

danger_zones = ["Bronx", "Staten Island"]
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

def apply_clustering(df):
    features = df[["price", "number_of_reviews", "reviews_per_month", "minimum_nights", "availability_365"]].fillna(0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=5, random_state=42)
    df["cluster"] = kmeans.fit_predict(scaled_features)
    return df

df = apply_clustering(df)

def create_binary_vector(row):
    quiet_area = int(row["neighbourhood_group"] in ["Staten Island", "Queens"])
    entire_place = int(row["room_type"].lower() == "entire home/apt")
    many_reviews = int(row["number_of_reviews"] > 50)
    return np.array([quiet_area, entire_place, many_reviews])

def user_preferences_vector(user_info):
    quiet = 1 if user_info.get("location", "").lower() in ["staten island", "queens"] else 0
    entire = 1 if "entire" in user_info.get("style", "").lower() else 0
    many_reviews = 1 if user_info.get("min_reviews", 10) > 20 else 0
    return np.array([quiet, entire, many_reviews])

neighbourhood_descriptions = {
    "Manhattan": "the heart of NYC, with famous landmarks and a lively atmosphere",
    "Brooklyn": "a hip and artistic borough known for its parks and food",
    "Queens": "diverse and residential, with great food and culture",
    "Bronx": "home of the Yankees and the Bronx Zoo",
    "Staten Island": "a quieter borough accessible by ferry"
}

default_group_options = ["solo traveler", "couple", "family", "group"]
default_style_options = ["cozy and budget-friendly", "luxury", "entire place", "private room"]

UNCERTAIN_ANSWERS = [
    "don't know", "dont know", "not sure", "maybe", "no idea", "undecided", "i'm undecided", "unsure", "no clue", "not certain"
]

def is_uncertain_answer(text):
    text = text.lower()
    return any(phrase in text for phrase in UNCERTAIN_ANSWERS)

class AIBot:
    def __init__(self):
        self.chat_history = []
        self.user_info = {}
        self.filters = {}
        self.state = "greeting"
        self.start_index = 0

    def reset(self):
        self.__init__()

    def extract_budget_from_text(self, text):
        range_pattern = r"\$?(\d+)\s*[-to]+\s*\$?(\d+)"
        single_pattern = r"\$?(\d+)"
        text = text.lower()
        match_range = re.search(range_pattern, text)
        if match_range:
            low = int(match_range.group(1))
            high = int(match_range.group(2))
            return (low, high) if low <= high else (high, low)
        else:
            match_single = re.search(single_pattern, text)
            if match_single:
                val = int(match_single.group(1))
                return (val, val)
        return None

    def detect_language(self, text):
        try:
            return langdetect.detect(text)
        except:
            return "unknown"

    def process_message(self, user_input: str) -> str:
        user_input = user_input.strip()
        user_input_lower = user_input.lower()

        # טיפול בסגירת שיחה בכל שלב
        if any(word in user_input_lower for word in ["bye", "exit", "quit", "thank you", "thanks"]):
            self.reset()
            return "You're very welcome! 😊 If you need anything else, just say hi. Goodbye!"

        intent = intent_classifier.predict([user_input])[0]

        # טיפול בשפה עברית - הבוט מבין רק אנגלית
        if self.detect_language(user_input) == "he":
            return "הצ'אט מבין כרגע רק אנגלית. אנא כתוב את ההודעה שלך באנגלית."

        if (not self.chat_history and not user_input) or any(
                word in user_input_lower for word in ["hello", "hi", "hey"]):
            self.reset()
            self.state = "purpose"
            self.chat_history.append({"role": "user", "content": user_input})
            return "1/6 - Hi there! 👋 I’m your NYC apartment finder. What brings you to New York?"

        # סיווג כוונה
        intent = intent_classifier.predict([user_input])[0]

        if is_uncertain_answer(user_input_lower):
            if self.state == "purpose":
                return "Are you coming for work, vacation, or a mix?"
            if self.state == "group":
                options = ", ".join(default_group_options)
                return f"I'm happy to help! Are you traveling solo, with a partner, or with a group? Options: {options}"
            if self.state == "style":
                options = ", ".join(default_style_options)
                return f"Would you prefer something cozy and budget-friendly, or do you prefer luxury? Options: {options}"
            if self.state == "location":
                options = "\n".join([f"- {k}: {v}" for k, v in neighbourhood_descriptions.items()])
                return f"Here are some boroughs to choose from:\n{options}\nWhich one sounds good to you?"
            if self.state == "budget":
                return "Please share your budget, for example: $100 or $80-$150."
            if self.state == "review_pref":
                return "Do you prefer places with lots of reviews, or are you open to new listings too?"
            if self.state in ["confirm", "search", "showing", "adjust"]:
                return "Would you like me to help you adjust your preferences or show more options?"

        # המשך הלוגיקה שלך ללא שינוי
        if self.state == "purpose":
            self.user_info["purpose"] = user_input
            self.state = "group"
            return "2/6 - Are you traveling solo, with friends, or with family?"

        if self.state == "group":
            self.user_info["group"] = user_input
            self.state = "style"
            return "3/6 - Are you looking for something cozy and budget-friendly, or do you prefer a bit more luxury?"

        if self.state == "style":
            self.user_info["style"] = user_input
            self.state = "location"
            return "4/6 - Any favorite neighborhoods or vibes you want? Or should I show you the best-value spots anywhere in NYC?"

        if self.state == "location":
            self.user_info["location"] = user_input
            self.state = "budget"
            return "5/6 - What’s your budget per night? (e.g. $100, or between $80-$150)"

        if self.state == "budget":
            budget_range = self.extract_budget_from_text(user_input)
            if budget_range:
                self.user_info["budget_low"], self.user_info["budget_high"] = budget_range
            else:
                self.user_info["budget_low"] = None
                self.user_info["budget_high"] = None
            self.state = "review_pref"
            return "6/6 - Do you prefer places with lots of reviews, or are you open to new listings too?"

        if self.state == "review_pref":
            self.user_info["min_reviews"] = 0 if "new" in user_input_lower else 10
            summary = self.summarize_preferences()
            self.state = "confirm"
            return summary + "\n\nShould I look for options now? (yes/no)"

        if self.state == "confirm":
            if "yes" in user_input_lower:
                self.state = "search"
                self.start_index = 0
            else:
                self.reset()
                return "No worries! If you want to start over, just say hi."

        if self.state == "search":
            self.build_filters_from_user_info()
            results = self.search_apartments_with_jaccard(self.filters, self.user_info, self.start_index, limit=5)
            if results.empty:
                self.state = "adjust"
                return "📭 Sorry, I couldn't find listings with those preferences. Want to adjust them?"
            self.state = "showing"
            return self.format_listings(results)

        if self.state == "showing":
            more_triggers = ["more", "show me more", "show", "עוד", "תראה עוד"]
            if any(word in user_input_lower for word in more_triggers):
                self.start_index += 5
                results = self.search_apartments_with_jaccard(self.filters, self.user_info, self.start_index, limit=5)
                if results.empty:
                    self.state = "adjust"
                    return "That's all I found! Want to adjust your preferences?"
                return self.format_listings(results)
            elif any(word in user_input_lower for word in
                     ["adjust", "change", "start over", "different", "preferences"]):
                self.reset()
                self.state = "purpose"
                return "Sure! Let's start over. What brings you to New York?"
            else:
                return "Say 'show me more' to see more results, or 'change preferences' to update your choices."

        if self.state == "adjust":
            if "yes" in user_input_lower or "adjust" in user_input_lower:
                self.reset()
                self.state = "purpose"
                return "Let's update your preferences! What brings you to New York?"
            else:
                return "Say 'yes' to adjust your preferences, or 'bye' to exit."

        return "I'm here to help you find your perfect NYC apartment! Shall we start? 😊"

    def summarize_preferences(self):
        s = self.user_info
        budget_str = "No set budget"
        if s.get("budget_low") and s.get("budget_high"):
            if s["budget_low"] == s["budget_high"]:
                budget_str = f"${s['budget_low']}"
            else:
                budget_str = f"${s['budget_low']} - ${s['budget_high']}"
        elif s.get("budget_low"):
            budget_str = f"From ${s['budget_low']}"

        return (f"Here’s what I’ve got:\n"
                f"• Purpose: {s.get('purpose','N/A')}\n"
                f"• Group: {s.get('group','N/A')}\n"
                f"• Style: {s.get('style','N/A')}\n"
                f"• Neighborhood/Vibe: {s.get('location','Anywhere')}\n"
                f"• Budget: {budget_str}/night\n"
                f"• Reviews: {'Lots of reviews' if s.get('min_reviews',10) > 0 else 'Open to new listings'}")

    def build_filters_from_user_info(self):
        s = self.user_info
        self.filters = {}
        if s.get("budget_low"):
            self.filters["min_price"] = s["budget_low"]
        if s.get("budget_high"):
            self.filters["max_price"] = s["budget_high"]
        if s.get("location") and "any" not in s["location"].lower():
            self.filters["area"] = s["location"]
        self.filters["min_reviews"] = s.get("min_reviews", 10)
        self.filters["safe"] = True
        style = s.get("style", "").lower()
        if any(word in style for word in ["cozy", "budget", "private"]):
            self.filters["room"] = "Private room"
        elif any(word in style for word in ["luxury", "entire"]):
            self.filters["room"] = "Entire home/apt"

    def search_apartments_with_jaccard(self, filters, user_info, start_index=0, limit=5):
        result = df.copy()
        if "min_price" in filters:
            result = result[result["price"] >= filters["min_price"]]
        if "max_price" in filters:
            result = result[result["price"] <= filters["max_price"]]
        if "area" in filters:
            area = filters["area"].lower()
            mask_group = result["neighbourhood_group"].str.lower().str.contains(area)
            mask_neigh = result["neighbourhood"].str.lower().str.contains(area)
            result = result[mask_group | mask_neigh]
        if "room" in filters:
            result = result[result["room_type"].str.lower() == filters["room"].lower()]
        if "min_reviews" in filters:
            result = result[result["number_of_reviews"] >= filters["min_reviews"]]
        if "safe" in filters and filters["safe"]:
            result = result[~result["neighbourhood_group"].isin(danger_zones)]

        user_vec = user_preferences_vector(user_info)
        result = result.copy()
        result["jaccard_similarity"] = result.apply(lambda row: jaccard_score(user_vec, create_binary_vector(row)), axis=1)

        result = result.sort_values(by=["jaccard_similarity", "value_score"], ascending=[False, False])
        return result.iloc[start_index:start_index + limit]

    def format_listings(self, listings):
        if listings.empty:
            return "📭 Oh no! I couldn't find anything matching your preferences. Want to adjust them a bit?"

        output = ["🏡 **Here are some of the best value options for you:**\n"]
        for _, row in listings.iterrows():
            maps_url = f"https://www.google.com/maps/search/?api=1&query={row['neighbourhood'].replace(' ', '+')},+{row['neighbourhood_group'].replace(' ', '+')}"
            output.append(
                f"• **{row['name']}**\n"
                f"  📍 [{row['neighbourhood']}, {row['neighbourhood_group']}]({maps_url})\n"
                f"  🛏️ {row['room_type']} — 💵 ${row['price']} / night\n"
                f"  ⭐ {row['number_of_reviews']} reviews | Minimum stay: {row['minimum_nights']} nights\n"
                f"  🧑 Host: {row['host_name']} (ID: {row['host_id']})\n"
                f"  📅 Available {row['availability_365']} days/year\n"
                f"  🔗 [View on Google Maps]({maps_url})\n"
            )
        output.append("\nWant to adjust your preferences? Just say 'change preferences'.\nOr say 'bye' to end the chat. 😊")
        return "\n".join(output)
