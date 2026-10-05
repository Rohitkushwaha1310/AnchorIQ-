import uuid
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION & REPRODUCIBILITY
# ==========================================
SEED = 42
np.random.seed(SEED)

NUM_USERS = 5000
TOTAL_TARGET_ROWS = 100000  # Change to 500000 or 1000000 as needed
START_DATE = datetime(2025, 1, 1, 0, 0, 0)

ACTIVITY_TYPES = [
    "mood",
    "meditation",
    "journal",
    "community",
    "music",
    "chat",
    "exercise",
    "communication",
]

# Baseline probabilities for activity selection
ACTIVITY_PROBS = [0.20, 0.15, 0.10, 0.10, 0.20, 0.10, 0.10, 0.05]

CHAT_CATEGORIES = {
    "positive": (0.7, 0.9, 70, 95),
    "calm": (0.5, 0.8, 65, 85),
    "motivation": (0.6, 0.9, 65, 90),
    "gratitude": (0.6, 0.9, 70, 90),
    "excitement": (0.7, 0.95, 75, 95),
    "neutral": (0.0, 0.4, 45, 60),
    "stress": (-0.8, -0.4, 20, 40),
    "anxiety": (-0.85, -0.5, 15, 35),
    "sadness": (-0.75, -0.3, 20, 45),
    "anger": (-0.9, -0.6, 10, 30),
    "loneliness": (-0.7, -0.3, 20, 40),
    "frustration": (-0.8, -0.4, 25, 45),
}

CHAT_TEXT_TEMPLATES = {
    "stress": "I had a really difficult day and feel exhausted.",
    "positive": "I feel amazing after talking to my friends today.",
    "calm": "Taking a moment to breathe and unwind.",
    "anxiety": "Feeling overwhelmed with upcoming deadlines.",
    "gratitude": "Grateful for the support from my team today.",
    "neutral": "Checking in for the day.",
}

EXERCISE_TYPES = ["walking", "running", "cycling", "gym", "yoga", "stretching"]
MOOD_LABELS = [
    "happy",
    "calm",
    "sad",
    "anxious",
    "angry",
    "stressed",
    "neutral",
    "motivated",
    "lonely",
    "excited",
    "tired",
]
MEDITATION_TYPES = [
    "mindfulness",
    "guided",
    "breathwork",
    "body_scan",
    "loving_kindness",
]
COMMUNICATION_TYPES = [
    "friend",
    "family",
    "community",
    "support_group",
    "general_social",
    "online_chat",
]


# ==========================================
# LATENT USER PROFILES GENERATOR
# ==========================================
def generate_user_profiles(num_users):
    users = []
    for i in range(num_users):
        user_id = f"user_{i+1:05d}"
        profile = {
            "user_id": user_id,
            "baseline_wellness": np.random.normal(55, 12),  # Centered around 55
            "stress_sensitivity": np.clip(np.random.normal(1.0, 0.3), 0.2, 2.0),
            "social_response": np.clip(
                np.random.normal(1.0, 0.4), -0.5, 2.0
            ),  # Some introspective users dislike social contact
            "music_response": np.clip(np.random.normal(1.0, 0.25), 0.2, 1.8),
            "meditation_response": np.clip(np.random.normal(1.0, 0.35), 0.1, 2.0),
            "exercise_response": np.clip(np.random.normal(1.0, 0.3), 0.1, 1.9),
            "recovery_rate": np.clip(
                np.random.normal(0.05, 0.02), 0.01, 0.12
            ),  # Passive recovery towards baseline per hour
            "activity_frequency": np.random.choice(
                [2, 3, 4, 5, 6], p=[0.1, 0.3, 0.3, 0.2, 0.1]
            ),
        }
        users.append(profile)
    return users


# ==========================================
# SYNTHETIC DATA GENERATION ENGINE
# ==========================================
def generate_synthetic_svs_dataset(num_users, total_rows):
    user_profiles = generate_user_profiles(num_users)
    rows_per_user = total_rows // num_users

    dataset = []

    for u_idx, user in enumerate(user_profiles):
        curr_time = START_DATE + timedelta(
            days=np.random.randint(0, 30), hours=np.random.randint(0, 24)
        )
        curr_energy = np.clip(
            user["baseline_wellness"] + np.random.normal(0, 5), 10, 90
        )

        history = []  # List of past records for this user: (timestamp, activity_type, activity_svs)

        for r in range(rows_per_user):
            # 1. Simulate Time Progression (Realistic sleep/awake rhythms)
            hours_gap = np.random.exponential(scale=16.0 / user["activity_frequency"])
            hours_gap = max(0.2, min(hours_gap, 36.0))  # Cap gap between 12 min and 36 hrs

            # Sleep schedule modeling: jump overnight if past 11 PM
            curr_time += timedelta(hours=hours_gap)
            if curr_time.hour >= 23 or curr_time.hour < 6:
                if np.random.rand() > 0.15:  # 85% chance user sleeps
                    curr_time = curr_time.replace(
                        hour=7, minute=np.random.randint(0, 59)
                    ) + timedelta(days=1 if curr_time.hour >= 23 else 0)

            hour = curr_time.hour
            day_of_week = curr_time.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0

            # 2. Passive Energy Recovery/Decay towards baseline over time
            time_since_last_min = (
                hours_gap * 60.0
                if r > 0
                else np.random.randint(120, 1440)
            )
            energy_drift = (
                user["baseline_wellness"] - curr_energy
            ) * (1 - np.exp(-user["recovery_rate"] * (time_since_last_min / 60.0)))
            energy_before_activity = np.clip(
                curr_energy + energy_drift, 5, 95
            )

            # 3. Select Activity Type
            activity_type = np.random.choice(
                ACTIVITY_TYPES, p=ACTIVITY_PROBS
            )

            # 4. Generate Activity Specific Details
            activity_intensity = np.round(
                np.clip(np.random.beta(2, 2), 0.05, 0.98), 2
            )

            # Initialize activity-specific optional fields
            mood_svs, meditation_svs, journal_svs, community_svs = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )
            music_svs, chat_svs, exercise_svs, communication_svs = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )

            mood_intensity = np.nan
            meditation_duration_minutes = np.nan
            exercise_duration_minutes = np.nan
            social_interaction_quality = np.nan
            chat_sentiment, chat_intensity, chat_category, chat_text = (
                None,
                np.nan,
                None,
                None,
            )

            raw_act_svs = 50.0  # Default baseline value

            if activity_type == "mood":
                mood_label = np.random.choice(MOOD_LABELS)
                mood_intensity = activity_intensity
                # Map mood label to SVS scale
                mood_svs_map = {
                    "happy": 85,
                    "calm": 80,
                    "excited": 85,
                    "motivated": 78,
                    "neutral": 50,
                    "tired": 40,
                    "sad": 25,
                    "anxious": 20,
                    "stressed": 22,
                    "angry": 15,
                    "lonely": 20,
                }
                raw_act_svs = mood_svs_map[mood_label] + np.random.normal(
                    0, 5
                )
                mood_svs = np.clip(raw_act_svs, 5, 98)

            elif activity_type == "meditation":
                meditation_duration_minutes = int(
                    np.random.choice([5, 10, 15, 20, 30, 45, 60])
                )
                # Diminishing returns after 30 mins
                duration_factor = np.log1p(meditation_duration_minutes) / np.log1p(
                    30
                )
                raw_act_svs = 40 + (
                    45 * activity_intensity * duration_factor
                ) + np.random.normal(0, 4)
                meditation_svs = np.clip(raw_act_svs, 10, 98)

            elif activity_type == "journal":
                journal_length = np.random.randint(20, 500)
                raw_act_svs = 30 + (
                    55 * activity_intensity
                ) + np.random.normal(0, 6)
                journal_svs = np.clip(raw_act_svs, 15, 95)

            elif activity_type == "community":
                social_interaction_quality = np.round(
                    np.clip(
                        activity_intensity + np.random.normal(0, 0.15),
                        0.0,
                        1.0,
                    ),
                    2,
                )
                raw_act_svs = 20 + (
                    70 * social_interaction_quality
                ) + np.random.normal(0, 5)
                community_svs = np.clip(raw_act_svs, 10, 98)

            elif activity_type == "music":
                raw_act_svs = 25 + (
                    65 * activity_intensity
                ) + np.random.normal(0, 7)
                music_svs = np.clip(raw_act_svs, 10, 95)

            elif activity_type == "chat":
                chat_category = np.random.choice(list(CHAT_CATEGORIES.keys()))
                s_min, s_max, svs_min, svs_max = CHAT_CATEGORIES[chat_category]
                chat_sentiment = np.round(
                    np.random.uniform(s_min, s_max), 2
                )
                chat_intensity = activity_intensity
                raw_act_svs = np.random.uniform(
                    svs_min, svs_max
                ) + np.random.normal(0, 3)
                chat_svs = np.clip(raw_act_svs, 5, 98)
                chat_text = CHAT_TEXT_TEMPLATES.get(
                    chat_category, "Interactive messaging session logged."
                )

            elif activity_type == "exercise":
                exercise_duration_minutes = int(
                    np.random.choice([15, 30, 45, 60, 90])
                )
                # Fatigue penalty if intensity is very high and duration > 60 mins
                fatigue = (
                    15
                    if (activity_intensity > 0.85 and exercise_duration_minutes > 60)
                    else 0
                )
                raw_act_svs = (
                    35 + (55 * activity_intensity) - fatigue + np.random.normal(0, 5)
                )
                exercise_svs = np.clip(raw_act_svs, 10, 98)

            elif activity_type == "communication":
                social_interaction_quality = np.round(
                    np.clip(
                        activity_intensity + np.random.normal(0, 0.2), 0.0, 1.0
                    ),
                    2,
                )
                raw_act_svs = 15 + (
                    75 * social_interaction_quality
                ) + np.random.normal(0, 6)
                communication_svs = np.clip(raw_act_svs, 10, 95)

            activity_svs = float(np.round(raw_act_svs, 2))

            # 5. Compute Historical Lag Features (No Data Leakage)
            hist_df = (
                pd.DataFrame(history, columns=["ts", "type", "svs"])
                if history
                else pd.DataFrame()
            )

            if not hist_df.empty:
                prev_act_svs = hist_df.iloc[-1]["svs"]
                avg_svs_3 = hist_df.iloc[-3:]["svs"].mean()
                avg_svs_7 = hist_df.iloc[-7:]["svs"].mean()
                avg_svs_14 = hist_df.iloc[-14:]["svs"].mean()

                ts_24h = curr_time - timedelta(hours=24)
                ts_7d = curr_time - timedelta(days=7)

                act_count_24h = len(hist_df[hist_df["ts"] >= ts_24h])
                act_count_7d = len(hist_df[hist_df["ts"] >= ts_7d])

                df_7d = hist_df[hist_df["ts"] >= ts_7d]
                mood_avg_7d = df_7d[df_7d["type"] == "mood"]["svs"].mean()
                meditation_avg_7d = df_7d[df_7d["type"] == "meditation"]["svs"].mean()
                journal_avg_7d = df_7d[df_7d["type"] == "journal"]["svs"].mean()
                community_avg_7d = df_7d[df_7d["type"] == "community"]["svs"].mean()
                music_avg_7d = df_7d[df_7d["type"] == "music"]["svs"].mean()
                chat_avg_7d = df_7d[df_7d["type"] == "chat"]["svs"].mean()
                exercise_avg_7d = df_7d[df_7d["type"] == "exercise"]["svs"].mean()
                comm_avg_7d = df_7d[df_7d["type"] == "communication"]["svs"].mean()
            else:
                prev_act_svs = np.nan
                avg_svs_3 = np.nan
                avg_svs_7 = np.nan
                avg_svs_14 = np.nan
                act_count_24h = 0
                act_count_7d = 0
                mood_avg_7d, meditation_avg_7d, journal_avg_7d = (
                    np.nan,
                    np.nan,
                    np.nan,
                )
                community_avg_7d, music_avg_7d, chat_avg_7d = (
                    np.nan,
                    np.nan,
                    np.nan,
                )
                exercise_avg_7d, comm_avg_7d = np.nan, np.nan

            # 6. CALCULATE TARGET VALUE (Nonlinear, Latent, Stochastic)
            # Personalization multiplier based on latent user traits
            trait_multiplier = 1.0
            if activity_type == "meditation":
                trait_multiplier = user["meditation_response"]
            elif activity_type in ["community", "communication"]:
                trait_multiplier = user["social_response"]
            elif activity_type == "music":
                trait_multiplier = user["music_response"]
            elif activity_type == "exercise":
                trait_multiplier = user["exercise_response"]

            # Baseline delta calculation
            delta_base = (activity_svs - energy_before_activity) * 0.25

            # Nonlinear scaling & diminishing returns
            if delta_base > 0:
                # Diminishing returns if previous energy is already high
                diminishing_factor = np.power(
                    (100.0 - energy_before_activity) / 100.0, 0.7
                )
                effective_delta = delta_base * trait_multiplier * diminishing_factor
            else:
                # High stress sensitivity amplifies negative events
                effective_delta = (
                    delta_base * user["stress_sensitivity"]
                )

            # History momentum effect
            recent_trend = (
                (avg_svs_3 - energy_before_activity) * 0.08
                if not np.isnan(avg_svs_3)
                else 0.0
            )

            # Environmental context penalty (e.g. middle of night stress)
            night_penalty = -3.5 if (hour >= 0 and hour <= 5) else 0.0

            # Realistic stochastic noise
            noise = np.random.normal(0, 4.5)

            # Compute final energy level & clip to [0, 100]
            next_energy_level = float(
                np.clip(
                    energy_before_activity
                    + effective_delta
                    + recent_trend
                    + night_penalty
                    + noise,
                    0.0,
                    100.0,
                )
            )

            # Outlier Injection (~0.5% probability)
            if np.random.rand() < 0.005:
                # Sudden crash or peak spike
                outlier_shift = np.random.choice([-35.0, 30.0])
                next_energy_level = float(
                    np.clip(next_energy_level + outlier_shift, 0.0, 100.0)
                )

            svs_change = float(
                np.round(next_energy_level - energy_before_activity, 2)
            )
            next_energy_level = float(np.round(next_energy_level, 2))

            # Update loop state
            curr_energy = next_energy_level
            history.append((curr_time, activity_type, activity_svs))

            # 7. Build Row Dictionary
            row = {
                "user_id": user["user_id"],
                "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                "activity_type": activity_type,
                "activity_svs": activity_svs,
                "activity_intensity": activity_intensity,
                "previous_energy_level": float(
                    np.round(energy_before_activity, 2)
                ),
                "previous_activity_svs": (
                    float(np.round(prev_act_svs, 2))
                    if not np.isnan(prev_act_svs)
                    else np.nan
                ),
                "avg_activity_svs_3": (
                    float(np.round(avg_svs_3, 2))
                    if not np.isnan(avg_svs_3)
                    else np.nan
                ),
                "avg_activity_svs_7": (
                    float(np.round(avg_svs_7, 2))
                    if not np.isnan(avg_svs_7)
                    else np.nan
                ),
                "avg_activity_svs_14": (
                    float(np.round(avg_svs_14, 2))
                    if not np.isnan(avg_svs_14)
                    else np.nan
                ),
                "activity_count_24h": act_count_24h,
                "activity_count_7d": act_count_7d,
                "mood_avg_7d": (
                    float(np.round(mood_avg_7d, 2))
                    if not np.isnan(mood_avg_7d)
                    else np.nan
                ),
                "meditation_avg_7d": (
                    float(np.round(meditation_avg_7d, 2))
                    if not np.isnan(meditation_avg_7d)
                    else np.nan
                ),
                "journal_avg_7d": (
                    float(np.round(journal_avg_7d, 2))
                    if not np.isnan(journal_avg_7d)
                    else np.nan
                ),
                "community_avg_7d": (
                    float(np.round(community_avg_7d, 2))
                    if not np.isnan(community_avg_7d)
                    else np.nan
                ),
                "music_avg_7d": (
                    float(np.round(music_avg_7d, 2))
                    if not np.isnan(music_avg_7d)
                    else np.nan
                ),
                "chat_avg_7d": (
                    float(np.round(chat_avg_7d, 2))
                    if not np.isnan(chat_avg_7d)
                    else np.nan
                ),
                "exercise_avg_7d": (
                    float(np.round(exercise_avg_7d, 2))
                    if not np.isnan(exercise_avg_7d)
                    else np.nan
                ),
                "communication_avg_7d": (
                    float(np.round(comm_avg_7d, 2))
                    if not np.isnan(comm_avg_7d)
                    else np.nan
                ),
                "mood_svs": (
                    float(np.round(mood_svs, 2))
                    if not np.isnan(mood_svs)
                    else np.nan
                ),
                "meditation_svs": (
                    float(np.round(meditation_svs, 2))
                    if not np.isnan(meditation_svs)
                    else np.nan
                ),
                "journal_svs": (
                    float(np.round(journal_svs, 2))
                    if not np.isnan(journal_svs)
                    else np.nan
                ),
                "community_svs": (
                    float(np.round(community_svs, 2))
                    if not np.isnan(community_svs)
                    else np.nan
                ),
                "music_svs": (
                    float(np.round(music_svs, 2))
                    if not np.isnan(music_svs)
                    else np.nan
                ),
                "chat_svs": (
                    float(np.round(chat_svs, 2))
                    if not np.isnan(chat_svs)
                    else np.nan
                ),
                "exercise_svs": (
                    float(np.round(exercise_svs, 2))
                    if not np.isnan(exercise_svs)
                    else np.nan
                ),
                "communication_svs": (
                    float(np.round(communication_svs, 2))
                    if not np.isnan(communication_svs)
                    else np.nan
                ),
                "mood_intensity": mood_intensity,
                "meditation_duration_minutes": meditation_duration_minutes,
                "exercise_duration_minutes": exercise_duration_minutes,
                "social_interaction_quality": social_interaction_quality,
                "chat_sentiment": chat_sentiment,
                "chat_intensity": chat_intensity,
                "chat_category": chat_category,
                "chat_text": chat_text,
                "hour": hour,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "time_since_previous_activity_minutes": int(
                    time_since_last_min
                ),
                "svs_change": svs_change,
                "next_energy_level": next_energy_level,
            }

            dataset.append(row)

    df = pd.DataFrame(dataset)

    # 8. INTRODUCE REALISTIC MISSINGNESS (5-12%)
    # Randomly drop specific behavioral logs to simulate sparse user activity
    cols_for_missingness = [
        "mood_avg_7d",
        "meditation_avg_7d",
        "journal_avg_7d",
        "community_avg_7d",
        "music_avg_7d",
        "chat_avg_7d",
        "exercise_avg_7d",
        "avg_activity_svs_14",
    ]
    for col in cols_for_missingness:
        mask = np.random.rand(len(df)) < np.random.uniform(0.05, 0.12)
        df.loc[mask, col] = np.nan

    return df


# Execute generation
if __name__ == "__main__":
    df_synthetic = generate_synthetic_svs_dataset(
        num_users=NUM_USERS, total_rows=TOTAL_TARGET_ROWS
    )
    df_synthetic.to_csv("svs_synthetic_dataset.csv", index=False)
    print(f"Successfully generated {len(df_synthetic)} rows.")