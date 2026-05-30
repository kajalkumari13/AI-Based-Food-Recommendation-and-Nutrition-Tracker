"""Utility helpers for health metrics, nutrition targets, and chatbot integration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


# Activity multipliers used with BMR to estimate TDEE.
ACTIVITY_FACTORS: Dict[str, float] = {
    "Sedentary": 1.2,
    "Lightly Active": 1.375,
    "Moderately Active": 1.55,
    "Very Active": 1.725,
    "Extra Active": 1.9,
}

# Goal-wise calorie delta applied over maintenance calories.
GOAL_ADJUSTMENTS: Dict[str, int] = {
    "Weight Loss": -400,
    "Maintenance": 0,
    "Weight Gain": 350,
}


@dataclass(frozen=True)
class MacroTargets:
    """Daily macro targets in grams."""

    protein_g: float
    carbs_g: float
    fat_g: float


def normalize_diet_preference(preference: str) -> str:
    """Normalize user-entered diet preference to a standard title-cased label."""
    text = (preference or "Any").strip().lower()
    aliases = {
        "vegetarian": "Veg",
        "veg": "Veg",
        "vegan": "Vegan",
        "keto": "Keto",
        "low carb": "Low-Carb",
        "low-carb": "Low-Carb",
        "high protein": "High-Protein",
        "high-protein": "High-Protein",
        "paleo": "Paleo",
        "any": "Any",
    }
    return aliases.get(text, preference.strip().title() if preference else "Any")


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Compute Body Mass Index from weight and height."""
    if weight_kg <= 0 or height_cm <= 0:
        raise ValueError("Weight and height must be positive numbers.")
    height_m = height_cm / 100
    return weight_kg / (height_m**2)


def bmi_category(bmi: float) -> str:
    """Map BMI value to WHO standard category."""
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def calculate_bmr(age: int, gender: str, height_cm: float, weight_kg: float) -> float:
    """Calculate BMR using the Mifflin-St Jeor equation."""
    if age <= 0 or height_cm <= 0 or weight_kg <= 0:
        raise ValueError("Age, height, and weight must be positive numbers.")

    gender_key = (gender or "").strip().lower()
    base = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)

    if gender_key in {"male", "m"}:
        return base + 5
    if gender_key in {"female", "f"}:
        return base - 161

    # For non-binary/other, we return midpoint between male and female constants.
    return base - 78


def calculate_daily_calories(
    age: int,
    gender: str,
    height_cm: float,
    weight_kg: float,
    activity_level: str,
    goal: str,
) -> Tuple[float, float]:
    """Compute BMR and goal-adjusted daily calorie requirement."""
    bmr = calculate_bmr(age, gender, height_cm, weight_kg)
    activity_factor = ACTIVITY_FACTORS.get(activity_level, ACTIVITY_FACTORS["Moderately Active"])
    maintenance = bmr * activity_factor
    target = maintenance + GOAL_ADJUSTMENTS.get(goal, 0)
    return bmr, max(target, 1200.0)


def macro_targets(calories: float, diet_preference: str, goal: str) -> MacroTargets:
    """Estimate macro split in grams based on calories and dietary style."""
    preference = normalize_diet_preference(diet_preference)

    # Default balanced split: P 25%, C 45%, F 30%.
    split = {"protein": 0.25, "carbs": 0.45, "fat": 0.30}

    if preference == "Keto":
        split = {"protein": 0.25, "carbs": 0.10, "fat": 0.65}
    elif preference == "Low-Carb":
        split = {"protein": 0.30, "carbs": 0.25, "fat": 0.45}
    elif preference == "High-Protein":
        split = {"protein": 0.35, "carbs": 0.35, "fat": 0.30}
    elif preference == "Vegan":
        split = {"protein": 0.22, "carbs": 0.50, "fat": 0.28}

    # Slight protein bump for weight-loss goals to preserve lean mass.
    if goal == "Weight Loss":
        split["protein"] += 0.03
        split["carbs"] -= 0.02
        split["fat"] -= 0.01

    protein_g = (calories * split["protein"]) / 4
    carbs_g = (calories * split["carbs"]) / 4
    fat_g = (calories * split["fat"]) / 9

    return MacroTargets(protein_g=protein_g, carbs_g=carbs_g, fat_g=fat_g)


def compose_diet_context(user_profile: dict, recent_logs: List[dict] | None = None) -> str:
    """Build a compact context string to ground chatbot advice."""
    recent_logs = recent_logs or []
    profile_lines = [
        f"Age: {user_profile.get('age', 'N/A')}",
        f"Gender: {user_profile.get('gender', 'N/A')}",
        f"Height(cm): {user_profile.get('height_cm', 'N/A')}",
        f"Weight(kg): {user_profile.get('weight_kg', 'N/A')}",
        f"Activity: {user_profile.get('activity_level', 'N/A')}",
        f"Diet preference: {user_profile.get('dietary_preference', 'Any')}",
        f"Goal: {user_profile.get('goal', 'Maintenance')}",
        f"Daily calorie target: {user_profile.get('daily_calorie_target', 'N/A')}",
    ]

    logs_text = "\n".join(
        f"- {item['food_name']} x{item['servings']} ({item['calories']:.0f} kcal)"
        for item in recent_logs[:8]
    )
    return "\n".join(profile_lines) + "\nRecent meal logs:\n" + (logs_text or "- None")


def get_diet_chatbot_response(
    question: str,
    user_profile: dict,
    recent_logs: List[dict] | None = None,
    model: str = "gpt-4o-mini",
) -> str:
    """Call OpenAI API for contextual diet guidance with safe fallback behavior."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY is not configured. Add it to your environment to enable chatbot advice."

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        context = compose_diet_context(user_profile, recent_logs)
        response = client.responses.create(
            model=model,
            temperature=0.3,
            max_output_tokens=400,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a certified nutrition assistant. Give concise, practical, safe diet advice. "
                                "Avoid diagnosing diseases and advise consulting a professional for medical issues."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"User context:\n{context}\n\nQuestion: {question}",
                        }
                    ],
                },
            ],
        )
        return response.output_text
    except Exception as exc:  # noqa: BLE001
        return f"Unable to fetch chatbot advice right now: {exc}"
