import pandas as pd
import numpy as np

# Hum har genre ke liye 250 samples banayenge (250 * 4 = 1000)
SAMPLES_PER_GENRE = 250

def generate_genre_data(genre_name, tempo_mean, tempo_std, loudness_mean, loudness_std, pitch_mean, pitch_std):
    """Ek specific genre ke liye data generate karne ka helper function."""
    data = {
        'tempo': np.random.normal(loc=tempo_mean, scale=tempo_std, size=SAMPLES_PER_GENRE),
        'instrument_loudness_db': np.random.normal(loc=loudness_mean, scale=loudness_std, size=SAMPLES_PER_GENRE),
        'vocal_pitch_hz': np.random.normal(loc=pitch_mean, scale=pitch_std, size=SAMPLES_PER_GENRE),
        'genre': genre_name
    }
    return pd.DataFrame(data)

print(f"1000 lines ka dummy dataset generate ho raha hai...")

# 1. Rock Data (Fast tempo, high loudness)
rock_df = generate_genre_data(
    genre_name='rock',
    tempo_mean=130, tempo_std=15,
    loudness_mean=90, loudness_std=5,
    pitch_mean=350, pitch_std=50
)

# 2. Pop Data (Medium tempo, medium loudness, high pitch)
pop_df = generate_genre_data(
    genre_name='pop',
    tempo_mean=110, tempo_std=10,
    loudness_mean=80, loudness_std=5,
    pitch_mean=450, pitch_std=50
)

# 3. Jazz Data (Slow tempo, low loudness)
jazz_df = generate_genre_data(
    genre_name='jazz',
    tempo_mean=90, tempo_std=10,
    loudness_mean=65, loudness_std=5,
    pitch_mean=220, pitch_std=40
)

# 4. Classical Data (Variable tempo, medium-low loudness)
classical_df = generate_genre_data(
    genre_name='classical',
    tempo_mean=100, tempo_std=20, # Zyada variation
    loudness_mean=70, loudness_std=7,
    pitch_mean=180, pitch_std=40
)

# Sab dataframes ko milana
final_df = pd.concat([rock_df, pop_df, jazz_df, classical_df])

# Data ko shuffle karna (taake sab genres mix ho jayen)
final_df = final_df.sample(frac=1).reset_index(drop=True)

# Data ko file mein save karna
final_df.to_csv('dataset.csv', index=False)

print(f"Done! 'dataset.csv' file with 1000 lines is created successfully.")