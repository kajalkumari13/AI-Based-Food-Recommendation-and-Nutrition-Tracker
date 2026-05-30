"""Machine learning logic for food recommendations and meal planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import MacroTargets, normalize_diet_preference


@dataclass
class RecommendationInput:
    """Input payload required by the recommender."""

    dietary_preference: str
    daily_calorie_target: float
    macro_targets: MacroTargets
    goal: str


class FoodRecommender:
    """Content-based recommender using cosine similarity on food features."""

    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path
        self.food_df = self._load_dataset(dataset_path)

        self.numeric_features = ["calories", "protein", "carbs", "fat"]
        self.categorical_features = ["diet_type"]

        # Shared preprocessing for recommendation and optional classification.
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
            ]
        )

        self.food_feature_matrix = self.preprocessor.fit_transform(self.food_df)
        self.health_model = self._train_health_classifier()

    @staticmethod
    def _load_dataset(dataset_path: str) -> pd.DataFrame:
        """Load and validate the required food dataset schema."""
        df = pd.read_csv(dataset_path)
        rename_map = {
            "Food name": "food_name",
            "Calories": "calories",
            "Protein": "protein",
            "Carbs": "carbs",
            "Fat": "fat",
            "Diet type": "diet_type",
        }
        df = df.rename(columns=rename_map)

        required_cols = ["food_name", "calories", "protein", "carbs", "fat", "diet_type"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        df = df.dropna(subset=required_cols).copy()
        df["diet_type"] = df["diet_type"].astype(str).str.strip()

        for col in ["calories", "protein", "carbs", "fat"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["calories", "protein", "carbs", "fat"])
        return df.reset_index(drop=True)

    def _train_health_classifier(self) -> Pipeline:
        """Optional classifier to estimate whether food is healthy."""
        df = self.food_df.copy()

        # Simple domain-rule labels for a weakly supervised classifier.
        healthy_flag = (
            (df["calories"].between(150, 430))
            & (df["protein"] >= 10)
            & (df["fat"] <= 24)
            & (df["carbs"] <= 60)
        ).astype(int)

        model = Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            ("num", StandardScaler(), self.numeric_features),
                            ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
                        ]
                    ),
                ),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=5,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        model.fit(df[self.numeric_features + self.categorical_features], healthy_flag)
        return model

    @staticmethod
    def _diet_compatibility(preference: str) -> List[str]:
        """Return acceptable diet types for the selected user preference."""
        pref = normalize_diet_preference(preference)
        mapping = {
            "Any": [],
            "Veg": ["Veg", "Vegan"],
            "Vegan": ["Vegan"],
            "Keto": ["Keto", "Low-Carb"],
            "Low-Carb": ["Low-Carb", "Keto", "High-Protein"],
            "High-Protein": ["High-Protein", "Low-Carb"],
            "Paleo": ["Paleo", "Low-Carb"],
        }
        return mapping.get(pref, [pref])

    def _build_user_vector(self, request: RecommendationInput) -> pd.DataFrame:
        """Create a synthetic user preference vector in food-feature space."""
        calories_per_meal = request.daily_calorie_target / 4
        user_vector = pd.DataFrame(
            [
                {
                    "calories": calories_per_meal,
                    "protein": request.macro_targets.protein_g / 4,
                    "carbs": request.macro_targets.carbs_g / 4,
                    "fat": request.macro_targets.fat_g / 4,
                    "diet_type": normalize_diet_preference(request.dietary_preference),
                }
            ]
        )
        return user_vector

    def recommend_foods(
        self,
        request: RecommendationInput,
        top_n: int = 12,
        feedback_adjustments: Dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Rank foods with cosine similarity plus optional user feedback adjustments."""
        feedback_adjustments = feedback_adjustments or {}

        candidates = self.food_df.copy()
        allowed_diets = self._diet_compatibility(request.dietary_preference)
        if allowed_diets:
            candidates = candidates[candidates["diet_type"].isin(allowed_diets)].copy()

        if candidates.empty:
            return pd.DataFrame(columns=self.food_df.columns.tolist() + ["similarity_score", "final_score"])

        user_vector = self._build_user_vector(request)
        user_features = self.preprocessor.transform(user_vector)
        candidate_features = self.preprocessor.transform(candidates)

        sims = cosine_similarity(user_features, candidate_features).flatten()
        candidates["similarity_score"] = sims

        # Feedback boost: 1-5 rating converted to -0.10..+0.10.
        candidates["feedback_boost"] = candidates["food_name"].map(
            lambda name: (feedback_adjustments.get(name, 3.0) - 3.0) * 0.05
        )
        candidates["final_score"] = candidates["similarity_score"] + candidates["feedback_boost"]

        ranked = candidates.sort_values("final_score", ascending=False).head(top_n)
        ranked["healthy_probability"] = self.predict_healthiness(ranked)
        return ranked.reset_index(drop=True)

    def predict_healthiness(self, foods_df: pd.DataFrame) -> np.ndarray:
        """Return healthy-food probability from optional classification model."""
        if foods_df.empty:
            return np.array([])
        x = foods_df[self.numeric_features + self.categorical_features]
        return self.health_model.predict_proba(x)[:, 1]

    def build_7_day_plan(self, recommendations: pd.DataFrame, daily_target: float) -> pd.DataFrame:
        """Generate a varied 7-day meal plan from recommended foods."""
        if recommendations.empty:
            return pd.DataFrame(
                columns=["Day", "Breakfast", "Lunch", "Dinner", "Snack", "Estimated Calories"]
            )

        recs = recommendations.copy()
        recs = recs.sort_values("final_score", ascending=False).reset_index(drop=True)

        slots = {
            "Breakfast": recs[recs["calories"].between(180, 360)],
            "Lunch": recs[recs["calories"].between(280, 470)],
            "Dinner": recs[recs["calories"].between(260, 480)],
            "Snack": recs[recs["calories"].between(150, 300)],
        }

        for slot_name, slot_df in slots.items():
            if slot_df.empty:
                slots[slot_name] = recs

        used_recent: List[str] = []
        rows = []

        for day_index in range(7):
            day_meals = {}
            calories_sum = 0.0

            for slot_name, slot_df in slots.items():
                # Prefer foods not used in recent picks to increase diversity.
                pool = slot_df[~slot_df["food_name"].isin(used_recent[-10:])]
                if pool.empty:
                    pool = slot_df

                selected = pool.sample(n=1, random_state=42 + day_index + len(slot_name)).iloc[0]
                day_meals[slot_name] = selected["food_name"]
                calories_sum += float(selected["calories"])
                used_recent.append(selected["food_name"])

            # Light scaling note shown by difference from target.
            rows.append(
                {
                    "Day": f"Day {day_index + 1}",
                    "Breakfast": day_meals["Breakfast"],
                    "Lunch": day_meals["Lunch"],
                    "Dinner": day_meals["Dinner"],
                    "Snack": day_meals["Snack"],
                    "Estimated Calories": round(calories_sum, 0),
                    "Delta vs Target": round(calories_sum - daily_target, 0),
                }
            )

        return pd.DataFrame(rows)
