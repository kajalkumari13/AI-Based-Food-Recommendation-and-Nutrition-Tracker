"""Main Streamlit application for AI-Based Food Recommendation and Nutrition Tracking."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database import (
    add_feedback,
    add_food_log,
    get_daily_totals,
    get_feedback_adjustments,
    get_food_logs,
    get_recent_feedback,
    get_recent_food_logs,
    get_user_profile,
    get_weekly_calorie_trend,
    get_weekly_weight_trend,
    init_db,
    upsert_user_profile,
    upsert_weight_log,
)
from model import FoodRecommender, RecommendationInput
from utils import (
    ACTIVITY_FACTORS,
    GOAL_ADJUSTMENTS,
    bmi_category,
    calculate_bmi,
    calculate_daily_calories,
    get_diet_chatbot_response,
    macro_targets,
    normalize_diet_preference,
)

# -----------------------------------------------------------------------------
# Configuration and cached resources
# -----------------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parent
DATASET_PATH = APP_ROOT / "dataset" / "food_dataset.csv"

st.set_page_config(
    page_title="AI Food Recommendation & Nutrition Tracker",
    page_icon="🥗",
    layout="wide",
)


@st.cache_data
def load_food_dataset(path: Path) -> pd.DataFrame:
    """Load food dataset once per session."""
    return pd.read_csv(path)


@st.cache_resource
def load_recommender(path: Path) -> FoodRecommender:
    """Load and cache ML recommender instance."""
    return FoodRecommender(str(path))


def apply_custom_style() -> None:
    """Apply lightweight production-style visual polish to Streamlit UI."""
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(120deg, #f8fff8 0%, #edf9ff 100%);
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f5132 0%, #14532d 60%, #1f7a5a 100%);
            }
            [data-testid="stSidebar"] * {
                color: #f0fff4 !important;
            }
            .block-container {
                padding-top: 1.5rem;
            }
            .kpi-card {
                background: white;
                border: 1px solid #e6f0ea;
                border-radius: 14px;
                padding: 10px 14px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Core page renderers
# -----------------------------------------------------------------------------
def render_profile_and_recommendations(
    user_id: str,
    profile: dict | None,
    recommender: FoodRecommender,
    food_df: pd.DataFrame,
) -> None:
    """Render profile input form, health metrics, and personalized recommendations."""
    st.title("AI-Based Food Recommendation and Nutrition Tracker")
    st.caption("Personalized meals, calorie targets, nutrition tracking, and adaptive recommendations")

    with st.form("profile_form", clear_on_submit=False):
        st.subheader("1) User Profile")
        c1, c2, c3, c4 = st.columns(4)

        age = c1.number_input("Age", min_value=10, max_value=90, value=int(profile["age"]) if profile else 24)
        gender = c2.selectbox(
            "Gender",
            options=["Male", "Female", "Other"],
            index=["Male", "Female", "Other"].index(profile["gender"]) if profile and profile["gender"] in ["Male", "Female", "Other"] else 0,
        )
        height_cm = c3.number_input(
            "Height (cm)",
            min_value=120.0,
            max_value=230.0,
            value=float(profile["height_cm"]) if profile else 170.0,
            step=0.5,
        )
        weight_kg = c4.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=220.0,
            value=float(profile["weight_kg"]) if profile else 70.0,
            step=0.5,
        )

        c5, c6, c7 = st.columns(3)
        activity_level = c5.selectbox(
            "Activity Level",
            options=list(ACTIVITY_FACTORS.keys()),
            index=list(ACTIVITY_FACTORS.keys()).index(profile["activity_level"])
            if profile and profile["activity_level"] in ACTIVITY_FACTORS
            else 2,
        )

        food_diets = sorted({d for d in food_df["Diet type"].dropna().astype(str).tolist()})
        preference_choices = ["Any"] + food_diets
        current_pref = normalize_diet_preference(profile["dietary_preference"]) if profile else "Any"
        dietary_preference = c6.selectbox(
            "Dietary Preference",
            options=preference_choices,
            index=preference_choices.index(current_pref) if current_pref in preference_choices else 0,
        )

        goal = c7.selectbox(
            "Goal",
            options=list(GOAL_ADJUSTMENTS.keys()),
            index=list(GOAL_ADJUSTMENTS.keys()).index(profile["goal"])
            if profile and profile["goal"] in GOAL_ADJUSTMENTS
            else 1,
        )

        submitted = st.form_submit_button("Save Profile & Recalculate Targets", use_container_width=True)

    if submitted:
        bmi = calculate_bmi(weight_kg, height_cm)
        bmr, daily_calorie_target = calculate_daily_calories(
            age=age,
            gender=gender,
            height_cm=height_cm,
            weight_kg=weight_kg,
            activity_level=activity_level,
            goal=goal,
        )

        upsert_user_profile(
            user_id,
            {
                "age": age,
                "gender": gender,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "activity_level": activity_level,
                "dietary_preference": normalize_diet_preference(dietary_preference),
                "goal": goal,
                "bmi": round(bmi, 2),
                "bmr": round(bmr, 2),
                "daily_calorie_target": round(daily_calorie_target, 0),
            },
        )
        st.success("Profile saved successfully. Recommendations refreshed.")
        profile = get_user_profile(user_id)

    if not profile:
        st.info("Fill your profile and click Save to unlock personalized recommendations.")
        return

    st.subheader("2) Health Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BMI", f"{profile['bmi']:.2f}", bmi_category(profile["bmi"]))
    c2.metric("BMR (kcal/day)", f"{profile['bmr']:.0f}")
    c3.metric("Target Calories", f"{profile['daily_calorie_target']:.0f} kcal")
    c4.metric("Goal", profile["goal"])

    targets = macro_targets(
        calories=float(profile["daily_calorie_target"]),
        diet_preference=profile["dietary_preference"],
        goal=profile["goal"],
    )

    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    st.write(
        f"Suggested Macros: **Protein {targets.protein_g:.0f}g**, "
        f"**Carbs {targets.carbs_g:.0f}g**, **Fat {targets.fat_g:.0f}g**"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("3) Personalized Meal Recommendations")
    feedback_adjustments = get_feedback_adjustments(user_id)
    rec_input = RecommendationInput(
        dietary_preference=profile["dietary_preference"],
        daily_calorie_target=float(profile["daily_calorie_target"]),
        macro_targets=targets,
        goal=profile["goal"],
    )

    recommendations = recommender.recommend_foods(
        rec_input,
        top_n=15,
        feedback_adjustments=feedback_adjustments,
    )

    if recommendations.empty:
        st.warning("No recommendations found for selected preferences. Try 'Any' diet preference.")
        return

    st.session_state[f"recommendations_{user_id}"] = recommendations

    display_cols = [
        "food_name",
        "diet_type",
        "calories",
        "protein",
        "carbs",
        "fat",
        "final_score",
        "healthy_probability",
    ]
    view_df = recommendations[display_cols].copy()
    view_df["final_score"] = view_df["final_score"].round(3)
    view_df["healthy_probability"] = (view_df["healthy_probability"] * 100).round(1)
    view_df = view_df.rename(
        columns={
            "food_name": "Food",
            "diet_type": "Diet Type",
            "calories": "Calories",
            "protein": "Protein",
            "carbs": "Carbs",
            "fat": "Fat",
            "final_score": "Recommendation Score",
            "healthy_probability": "Healthy %",
        }
    )
    st.dataframe(view_df, use_container_width=True, hide_index=True)

    st.subheader("4) Quick Log from Recommendations")
    with st.form("quick_log_form"):
        c1, c2, c3 = st.columns([2, 1, 1])
        selected_food = c1.selectbox("Choose recommended food", options=recommendations["food_name"].tolist())
        meal_type = c2.selectbox("Meal Type", options=["Breakfast", "Lunch", "Dinner", "Snack"])
        servings = c3.slider("Servings", min_value=0.5, max_value=3.0, value=1.0, step=0.5)
        add_log = st.form_submit_button("Add to Today's Log")

    if add_log:
        row = recommendations.loc[recommendations["food_name"] == selected_food].iloc[0]
        add_food_log(
            user_id=user_id,
            log_date=date.today().isoformat(),
            meal_type=meal_type,
            food_name=row["food_name"],
            calories=float(row["calories"]) * servings,
            protein=float(row["protein"]) * servings,
            carbs=float(row["carbs"]) * servings,
            fat=float(row["fat"]) * servings,
            servings=float(servings),
        )
        st.success(f"Logged {selected_food} ({servings} serving).")


def render_nutrition_tracker(user_id: str, profile: dict | None, food_df: pd.DataFrame) -> None:
    """Render daily logging and nutrition visualizations."""
    st.title("Daily Nutrition Tracker")

    if not profile:
        st.info("Create a profile first in Dashboard to start tracking.")
        return

    selected_date = st.date_input("Track Date", value=date.today())
    selected_date_str = selected_date.isoformat()

    st.subheader("Log a Meal")
    with st.form("manual_log_form"):
        c1, c2, c3 = st.columns([2, 1, 1])
        food_name = c1.selectbox("Food", options=food_df["Food name"].tolist())
        meal_type = c2.selectbox("Meal", options=["Breakfast", "Lunch", "Dinner", "Snack"], key="tracker_meal_type")
        servings = c3.slider("Servings", min_value=0.5, max_value=3.0, value=1.0, step=0.5, key="tracker_servings")
        submit_log = st.form_submit_button("Add Meal")

    if submit_log:
        food_row = food_df[food_df["Food name"] == food_name].iloc[0]
        add_food_log(
            user_id=user_id,
            log_date=selected_date_str,
            meal_type=meal_type,
            food_name=food_name,
            calories=float(food_row["Calories"]) * servings,
            protein=float(food_row["Protein"]) * servings,
            carbs=float(food_row["Carbs"]) * servings,
            fat=float(food_row["Fat"]) * servings,
            servings=float(servings),
        )
        st.success(f"Added {food_name} to {meal_type} on {selected_date_str}.")

    logs_df = get_food_logs(user_id, selected_date_str)
    daily_totals = get_daily_totals(user_id, selected_date_str)
    required_calories = float(profile.get("daily_calorie_target", 0) or 0)

    st.subheader("Logged Meals")
    if logs_df.empty:
        st.caption("No meals logged yet for this date.")
    else:
        show_logs = logs_df[["meal_type", "food_name", "servings", "calories", "protein", "carbs", "fat"]].copy()
        show_logs.columns = ["Meal", "Food", "Servings", "Calories", "Protein", "Carbs", "Fat"]
        st.dataframe(show_logs, use_container_width=True, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Consumed Calories", f"{daily_totals['calories']:.0f} kcal")
    c2.metric("Target Calories", f"{required_calories:.0f} kcal")
    c3.metric("Protein", f"{daily_totals['protein']:.1f} g")
    c4.metric("Carbs / Fat", f"{daily_totals['carbs']:.1f} / {daily_totals['fat']:.1f} g")

    st.subheader("Nutrition Charts")
    left, right = st.columns(2)

    with left:
        fig_calorie = go.Figure()
        fig_calorie.add_bar(
            x=["Consumed", "Required"],
            y=[daily_totals["calories"], required_calories],
            marker_color=["#2f855a", "#2b6cb0"],
        )
        fig_calorie.update_layout(
            title="Calorie Intake vs Required",
            yaxis_title="Calories (kcal)",
            height=380,
        )
        st.plotly_chart(fig_calorie, use_container_width=True)

    with right:
        fig_macros = go.Figure(
            data=[
                go.Pie(
                    labels=["Protein", "Carbs", "Fat"],
                    values=[
                        max(daily_totals["protein"], 0),
                        max(daily_totals["carbs"], 0),
                        max(daily_totals["fat"], 0),
                    ],
                    hole=0.35,
                )
            ]
        )
        fig_macros.update_layout(title="Macro Distribution", height=380)
        st.plotly_chart(fig_macros, use_container_width=True)

    st.subheader("Weekly Trend")
    weekly = get_weekly_calorie_trend(user_id, selected_date_str, days=7)
    fig_weekly = go.Figure()
    fig_weekly.add_trace(
        go.Scatter(
            x=weekly["log_date"],
            y=weekly["calories"],
            mode="lines+markers",
            name="Consumed",
            line=dict(color="#2f855a", width=3),
        )
    )
    fig_weekly.add_trace(
        go.Scatter(
            x=weekly["log_date"],
            y=[required_calories] * len(weekly),
            mode="lines",
            name="Required",
            line=dict(color="#dd6b20", dash="dash"),
        )
    )
    fig_weekly.update_layout(
        title="7-Day Calorie Trend",
        xaxis_title="Date",
        yaxis_title="Calories",
        height=420,
    )
    st.plotly_chart(fig_weekly, use_container_width=True)

    st.subheader("Body Weight Progress")
    c1, c2 = st.columns([2, 1])
    weight_today = c1.number_input(
        "Update Weight (kg)",
        min_value=30.0,
        max_value=220.0,
        value=float(profile["weight_kg"]),
        step=0.2,
    )
    save_weight = c2.button("Save Weight", use_container_width=True)

    if save_weight:
        upsert_weight_log(user_id, selected_date_str, weight_today)
        st.success(f"Saved weight {weight_today:.1f} kg for {selected_date_str}.")

    weight_trend = get_weekly_weight_trend(user_id, selected_date_str, days=7)
    if not weight_trend.empty:
        fig_weight = go.Figure()
        fig_weight.add_trace(
            go.Scatter(
                x=weight_trend["log_date"],
                y=weight_trend["weight_kg"],
                mode="lines+markers",
                name="Weight",
                line=dict(color="#1a365d", width=3),
            )
        )
        fig_weight.update_layout(title="7-Day Weight Trend", xaxis_title="Date", yaxis_title="Weight (kg)", height=350)
        st.plotly_chart(fig_weight, use_container_width=True)
    else:
        st.caption("No weight logs available in the selected 7-day window.")


def render_meal_planner(user_id: str, profile: dict | None, recommender: FoodRecommender) -> None:
    """Render 7-day auto meal planner."""
    st.title("7-Day Meal Planner")

    if not profile:
        st.info("Create your profile first from Dashboard.")
        return

    targets = macro_targets(
        calories=float(profile["daily_calorie_target"]),
        diet_preference=profile["dietary_preference"],
        goal=profile["goal"],
    )
    rec_input = RecommendationInput(
        dietary_preference=profile["dietary_preference"],
        daily_calorie_target=float(profile["daily_calorie_target"]),
        macro_targets=targets,
        goal=profile["goal"],
    )
    recommendations = recommender.recommend_foods(rec_input, top_n=28, feedback_adjustments=get_feedback_adjustments(user_id))

    if st.button("Generate 7-Day Plan", use_container_width=True):
        plan_df = recommender.build_7_day_plan(recommendations, daily_target=float(profile["daily_calorie_target"]))
        st.session_state[f"meal_plan_{user_id}"] = plan_df

    plan_df = st.session_state.get(f"meal_plan_{user_id}")
    if plan_df is None or plan_df.empty:
        st.caption("Click 'Generate 7-Day Plan' to create your personalized meal plan.")
        return

    st.dataframe(plan_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download Meal Plan as CSV",
        data=plan_df.to_csv(index=False).encode("utf-8"),
        file_name=f"meal_plan_{user_id}.csv",
        mime="text/csv",
    )


def render_chatbot(user_id: str, profile: dict | None) -> None:
    """Render OpenAI diet-advice chatbot interface."""
    st.title("Diet Advice Chatbot (OpenAI)")

    if not profile:
        st.info("Please create your profile first from Dashboard for contextual diet advice.")
        return

    model = st.selectbox("OpenAI Model", options=["gpt-4o-mini", "gpt-4.1-mini"], index=0)
    question = st.text_area(
        "Ask your diet/nutrition question",
        placeholder="Example: I have a 1900 kcal target. Suggest how to split meals if I workout in the evening.",
        height=120,
    )

    if st.button("Get Advice", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Fetching personalized advice..."):
                recent_logs = get_recent_food_logs(user_id)
                response = get_diet_chatbot_response(
                    question=question,
                    user_profile=profile,
                    recent_logs=recent_logs,
                    model=model,
                )
            st.markdown("### Chatbot Response")
            st.write(response)

    st.caption("Set OPENAI_API_KEY environment variable to activate chatbot responses.")


def render_feedback(user_id: str, food_df: pd.DataFrame) -> None:
    """Render recommendation feedback interface for adaptive learning."""
    st.title("Feedback & Recommendation Tuning")

    with st.form("feedback_form"):
        food_name = st.selectbox("Food Item", options=food_df["Food name"].tolist())
        rating = st.slider("Rating", min_value=1, max_value=5, value=4)
        comment = st.text_area("Comment (optional)", placeholder="What did you like/dislike?")
        submit = st.form_submit_button("Submit Feedback", use_container_width=True)

    if submit:
        add_feedback(user_id=user_id, food_name=food_name, rating=rating, comment=comment.strip())
        st.success("Feedback saved. Future recommendations will adapt to your ratings.")

    recent_feedback = get_recent_feedback(user_id, limit=12)
    st.subheader("Recent Feedback")
    if recent_feedback.empty:
        st.caption("No feedback submitted yet.")
    else:
        st.dataframe(recent_feedback, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# App entry point
# -----------------------------------------------------------------------------
def main() -> None:
    """Initialize dependencies and route pages."""
    apply_custom_style()
    init_db()

    if not DATASET_PATH.exists():
        st.error(f"Dataset not found: {DATASET_PATH}")
        st.stop()

    food_df = load_food_dataset(DATASET_PATH)
    recommender = load_recommender(DATASET_PATH)

    st.sidebar.title("Navigation")
    user_id = st.sidebar.text_input("User ID", value="iit_capstone_user")
    page = st.sidebar.radio(
        "Go to",
        options=[
            "Dashboard",
            "Nutrition Tracker",
            "Meal Planner",
            "Diet Chatbot",
            "Feedback",
        ],
    )

    if not user_id.strip():
        st.warning("Please enter a valid User ID in sidebar.")
        st.stop()

    profile = get_user_profile(user_id.strip())

    if page == "Dashboard":
        render_profile_and_recommendations(user_id.strip(), profile, recommender, food_df)
    elif page == "Nutrition Tracker":
        render_nutrition_tracker(user_id.strip(), profile, food_df)
    elif page == "Meal Planner":
        render_meal_planner(user_id.strip(), profile, recommender)
    elif page == "Diet Chatbot":
        render_chatbot(user_id.strip(), profile)
    elif page == "Feedback":
        render_feedback(user_id.strip(), food_df)


if __name__ == "__main__":
    main()
