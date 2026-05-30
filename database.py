"""SQLite operations for profiles, nutrition logs, feedback, and progress tracking."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd

DB_PATH = Path("nutrition_tracker.db")


@contextmanager
def get_connection(db_path: str | Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Yield SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | Path = DB_PATH) -> None:
    """Create all required tables if they do not exist."""
    with get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                height_cm REAL NOT NULL,
                weight_kg REAL NOT NULL,
                activity_level TEXT NOT NULL,
                dietary_preference TEXT NOT NULL,
                goal TEXT NOT NULL,
                bmi REAL,
                bmr REAL,
                daily_calorie_target REAL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS food_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                log_date TEXT NOT NULL,
                meal_type TEXT NOT NULL,
                food_name TEXT NOT NULL,
                calories REAL NOT NULL,
                protein REAL NOT NULL,
                carbs REAL NOT NULL,
                fat REAL NOT NULL,
                servings REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                food_name TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                comment TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS weight_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                log_date TEXT NOT NULL,
                weight_kg REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE,
                UNIQUE (user_id, log_date)
            );
            """
        )


def upsert_user_profile(user_id: str, payload: Dict, db_path: str | Path = DB_PATH) -> None:
    """Insert or update user profile metrics."""
    now = datetime.utcnow().isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO user_profiles (
                user_id, age, gender, height_cm, weight_kg,
                activity_level, dietary_preference, goal, bmi, bmr,
                daily_calorie_target, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                age = excluded.age,
                gender = excluded.gender,
                height_cm = excluded.height_cm,
                weight_kg = excluded.weight_kg,
                activity_level = excluded.activity_level,
                dietary_preference = excluded.dietary_preference,
                goal = excluded.goal,
                bmi = excluded.bmi,
                bmr = excluded.bmr,
                daily_calorie_target = excluded.daily_calorie_target,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                payload["age"],
                payload["gender"],
                payload["height_cm"],
                payload["weight_kg"],
                payload["activity_level"],
                payload["dietary_preference"],
                payload["goal"],
                payload.get("bmi"),
                payload.get("bmr"),
                payload.get("daily_calorie_target"),
                now,
            ),
        )


def get_user_profile(user_id: str, db_path: str | Path = DB_PATH) -> Optional[Dict]:
    """Fetch a single user profile by ID."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def add_food_log(
    user_id: str,
    log_date: str,
    meal_type: str,
    food_name: str,
    calories: float,
    protein: float,
    carbs: float,
    fat: float,
    servings: float,
    db_path: str | Path = DB_PATH,
) -> None:
    """Insert a consumed meal entry for a given date."""
    now = datetime.utcnow().isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO food_logs (
                user_id, log_date, meal_type, food_name,
                calories, protein, carbs, fat, servings, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                log_date,
                meal_type,
                food_name,
                calories,
                protein,
                carbs,
                fat,
                servings,
                now,
            ),
        )


def get_food_logs(user_id: str, log_date: str, db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """Return all logged foods for a user on a date."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, log_date, meal_type, food_name, servings,
                   calories, protein, carbs, fat, created_at
            FROM food_logs
            WHERE user_id = ? AND log_date = ?
            ORDER BY created_at DESC
            """,
            (user_id, log_date),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def get_daily_totals(user_id: str, log_date: str, db_path: str | Path = DB_PATH) -> Dict[str, float]:
    """Aggregate daily nutrition totals for charting and KPI display."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                COALESCE(SUM(calories), 0) AS calories,
                COALESCE(SUM(protein), 0) AS protein,
                COALESCE(SUM(carbs), 0) AS carbs,
                COALESCE(SUM(fat), 0) AS fat
            FROM food_logs
            WHERE user_id = ? AND log_date = ?
            """,
            (user_id, log_date),
        ).fetchone()
    return dict(row)


def get_weekly_calorie_trend(
    user_id: str,
    end_date: str,
    days: int = 7,
    db_path: str | Path = DB_PATH,
) -> pd.DataFrame:
    """Return day-wise calorie intake trend for the last N days."""
    end_dt = date.fromisoformat(end_date)
    start_dt = end_dt - timedelta(days=days - 1)

    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT log_date, COALESCE(SUM(calories), 0) AS calories
            FROM food_logs
            WHERE user_id = ?
              AND log_date BETWEEN ? AND ?
            GROUP BY log_date
            ORDER BY log_date ASC
            """,
            (user_id, start_dt.isoformat(), end_dt.isoformat()),
        ).fetchall()

    frame = pd.DataFrame([dict(r) for r in rows])
    all_dates = pd.DataFrame(
        {
            "log_date": [
                (start_dt + timedelta(days=i)).isoformat()
                for i in range(days)
            ]
        }
    )

    if frame.empty:
        all_dates["calories"] = 0
        return all_dates

    merged = all_dates.merge(frame, on="log_date", how="left").fillna(0)
    return merged


def add_feedback(
    user_id: str,
    food_name: str,
    rating: int,
    comment: str,
    db_path: str | Path = DB_PATH,
) -> None:
    """Store user feedback to influence future recommendations."""
    now = datetime.utcnow().isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO feedback (user_id, food_name, rating, comment, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, food_name, rating, comment, now),
        )


def get_feedback_adjustments(user_id: str, db_path: str | Path = DB_PATH) -> Dict[str, float]:
    """Return average per-food rating for a user."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT food_name, AVG(rating) AS avg_rating
            FROM feedback
            WHERE user_id = ?
            GROUP BY food_name
            """,
            (user_id,),
        ).fetchall()

    return {row["food_name"]: float(row["avg_rating"]) for row in rows}


def get_recent_feedback(user_id: str, limit: int = 10, db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """Fetch recent feedback entries for transparency in UI."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT food_name, rating, comment, created_at
            FROM feedback
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    return pd.DataFrame([dict(r) for r in rows])


def upsert_weight_log(
    user_id: str,
    log_date: str,
    weight_kg: float,
    db_path: str | Path = DB_PATH,
) -> None:
    """Insert or update a daily body weight entry."""
    now = datetime.utcnow().isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO weight_logs (user_id, log_date, weight_kg, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, log_date) DO UPDATE SET
                weight_kg = excluded.weight_kg,
                created_at = excluded.created_at
            """,
            (user_id, log_date, weight_kg, now),
        )


def get_weekly_weight_trend(
    user_id: str,
    end_date: str,
    days: int = 7,
    db_path: str | Path = DB_PATH,
) -> pd.DataFrame:
    """Return body-weight trend for the last N days."""
    end_dt = date.fromisoformat(end_date)
    start_dt = end_dt - timedelta(days=days - 1)

    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT log_date, weight_kg
            FROM weight_logs
            WHERE user_id = ?
              AND log_date BETWEEN ? AND ?
            ORDER BY log_date ASC
            """,
            (user_id, start_dt.isoformat(), end_dt.isoformat()),
        ).fetchall()

    frame = pd.DataFrame([dict(r) for r in rows])
    if frame.empty:
        return pd.DataFrame(columns=["log_date", "weight_kg"])
    return frame


def get_recent_food_logs(user_id: str, limit: int = 8, db_path: str | Path = DB_PATH) -> List[Dict]:
    """Return latest food logs used to ground chatbot context."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT food_name, servings, calories, log_date
            FROM food_logs
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    return [dict(r) for r in rows]
