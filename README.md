# AI-Based Food Recommendation and Nutrition Tracker

A complete end-to-end Streamlit web application for personalized food recommendation, calorie tracking, nutrition analytics, and diet guidance chatbot support.

## Project Structure

```text
capstone2/
├── app.py
├── model.py
├── utils.py
├── database.py
├── requirements.txt
├── nutrition_tracker.db (auto-created)
└── dataset/
    └── food_dataset.csv
```

## Features

- Personalized meal recommendations using content-based filtering (cosine similarity)
- BMI and BMR calculation (Mifflin-St Jeor Equation)
- Goal-adjusted daily calorie target estimation
- Macro target generation (Protein, Carbs, Fat)
- Daily nutrition tracker with SQLite-backed food logs
- Weekly calorie and weight progress charts
- 7-day meal planner
- Feedback system to tune recommendations
- OpenAI chatbot integration for diet advice
- Optional food healthiness probability model

## Tech Stack

- Frontend: Streamlit
- Backend: Python (modular)
- ML: scikit-learn (cosine similarity + optional classifier)
- Visualization: Plotly
- Database: SQLite

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Enable chatbot:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_openai_api_key"
```

4. Run the app:

```bash
streamlit run app.py
```

## Dataset Schema

The CSV file in `dataset/food_dataset.csv` includes:

- Food name
- Calories
- Protein
- Carbs
- Fat
- Diet type

## Notes for Capstone Submission

- Modular design with clear separation of UI, ML, utility logic, and database operations.
- Ready for extension with larger datasets, authentication, and deployment.
- Includes persistent logs and explainable recommendation scores.

## Future Enhancements

- Replace heuristic health labels with supervised labels from expert dataset.
- Add user authentication and role-based dashboards.
- Deploy on Streamlit Cloud/AWS/GCP with managed database.
- Add micronutrient-level recommendation optimization.
