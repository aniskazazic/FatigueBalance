import os
import pandas as pd

path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Workout_Routine_Dirty.csv')
path = os.path.normpath(path)

df = pd.read_csv(path)
if 'Unnamed: 18' in df.columns:
    df = df.drop(columns=['Unnamed: 18'])

defaults = {
    'Sleep_Duration': 7.0,
    'Sleep_Score': 75.0,
    'Sleep_Quality': 7.0,
    'Soreness': 5.0,
    'Stress': 5.0,
    'RPE': 5.0,
}
for col, default in defaults.items():
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)

if 'Injury_Illness' in df.columns:
    df['Injury_Illness'] = df['Injury_Illness'].astype(str).str.strip().replace({'nan': 'No', '': 'No'})
    df['Injury_Illness'] = df['Injury_Illness'].fillna('No')
else:
    df['Injury_Illness'] = 'No'

sleep = df['Sleep_Duration'].clip(0, 10)
sleep_quality = df['Sleep_Quality'].clip(0, 10)
stress = df['Stress'].clip(1, 10)
rpe = df['RPE'].clip(1, 10)
soreness = df['Soreness'].clip(1, 10)

fatigue = ((10 - sleep) * 4) + ((10 - sleep_quality) * 3) + stress * 4 + rpe * 3 + soreness * 2
fatigue = fatigue.clip(0, 100)
df['Fatigue'] = fatigue.round(1)
df['FatigueScore'] = df['Fatigue']

bins = [0, 40, 60, 80, 100]
labels = ['low', 'medium', 'high', 'critical']
df['RiskLevel'] = pd.cut(df['Fatigue'], bins=bins, labels=labels, include_lowest=True).astype(str)

extra = [
    {'Date': '01/10/2023', 'Name': 'Marco', 'Position': 'Goalkeeper', 'Session_Type': 'Practice', 'Sleep_Duration': 9.5, 'Sleep_Score': 94, 'Sleep_Quality': 9, 'Soreness': 2, 'Stress': 2, 'RPE': 3, 'Distance': 4200, 'Acceleration_Count': 38, 'Max_Acceleration': 4.1, 'Deceleration_Count': 28, 'Max_Deceleration': 4.9, 'Max_Speed': 23.1, 'Injury_Illness': 'No', 'Injury_Type': ''},
    {'Date': '01/11/2023', 'Name': 'Luka', 'Position': 'Midfielder', 'Session_Type': 'Practice', 'Sleep_Duration': 8.0, 'Sleep_Score': 90, 'Sleep_Quality': 8, 'Soreness': 3, 'Stress': 3, 'RPE': 4, 'Distance': 6800, 'Acceleration_Count': 58, 'Max_Acceleration': 5.1, 'Deceleration_Count': 52, 'Max_Deceleration': 5.7, 'Max_Speed': 29.2, 'Injury_Illness': 'No', 'Injury_Type': ''},
    {'Date': '01/12/2023', 'Name': 'Ivan', 'Position': 'Forward', 'Session_Type': 'Game', 'Sleep_Duration': 7.0, 'Sleep_Score': 82, 'Sleep_Quality': 7, 'Soreness': 5, 'Stress': 6, 'RPE': 7, 'Distance': 10900, 'Acceleration_Count': 75, 'Max_Acceleration': 6.8, 'Deceleration_Count': 68, 'Max_Deceleration': 7.1, 'Max_Speed': 33.0, 'Injury_Illness': 'No', 'Injury_Type': ''},
    {'Date': '01/13/2023', 'Name': 'Jovan', 'Position': 'Defender', 'Session_Type': 'Game', 'Sleep_Duration': 6.0, 'Sleep_Score': 76, 'Sleep_Quality': 6, 'Soreness': 6, 'Stress': 7, 'RPE': 8, 'Distance': 11100, 'Acceleration_Count': 83, 'Max_Acceleration': 7.1, 'Deceleration_Count': 77, 'Max_Deceleration': 7.8, 'Max_Speed': 31.2, 'Injury_Illness': 'No', 'Injury_Type': ''},
    {'Date': '01/14/2023', 'Name': 'Filip', 'Position': 'Midfielder', 'Session_Type': 'Practice', 'Sleep_Duration': 5.0, 'Sleep_Score': 68, 'Sleep_Quality': 5, 'Soreness': 7, 'Stress': 8, 'RPE': 9, 'Distance': 11200, 'Acceleration_Count': 88, 'Max_Acceleration': 7.3, 'Deceleration_Count': 80, 'Max_Deceleration': 8.0, 'Max_Speed': 30.6, 'Injury_Illness': 'Yes', 'Injury_Type': 'Muscle pain'},
    {'Date': '01/15/2023', 'Name': 'Nikola', 'Position': 'Outside Back', 'Session_Type': 'Practice', 'Sleep_Duration': 4.5, 'Sleep_Score': 61, 'Sleep_Quality': 5, 'Soreness': 8, 'Stress': 9, 'RPE': 9, 'Distance': 11300, 'Acceleration_Count': 90, 'Max_Acceleration': 7.9, 'Deceleration_Count': 84, 'Max_Deceleration': 8.2, 'Max_Speed': 32.0, 'Injury_Illness': 'Yes', 'Injury_Type': 'Hamstring discomfort'},
    {'Date': '01/16/2023', 'Name': 'Andrej', 'Position': 'Right Wing', 'Session_Type': 'Game', 'Sleep_Duration': 8.5, 'Sleep_Score': 85, 'Sleep_Quality': 8, 'Soreness': 4, 'Stress': 5, 'RPE': 6, 'Distance': 11995, 'Acceleration_Count': 92, 'Max_Acceleration': 7.5, 'Deceleration_Count': 83, 'Max_Deceleration': 7.2, 'Max_Speed': 34.4, 'Injury_Illness': 'No', 'Injury_Type': ''},
    {'Date': '01/17/2023', 'Name': 'Petar', 'Position': 'Center Forward', 'Session_Type': 'Practice', 'Sleep_Duration': 7.5, 'Sleep_Score': 88, 'Sleep_Quality': 8, 'Soreness': 3, 'Stress': 4, 'RPE': 4, 'Distance': 7200, 'Acceleration_Count': 56, 'Max_Acceleration': 5.0, 'Deceleration_Count': 45, 'Max_Deceleration': 5.3, 'Max_Speed': 29.8, 'Injury_Illness': 'No', 'Injury_Type': ''},
    {'Date': '01/18/2023', 'Name': 'Sasha', 'Position': 'Left Wing', 'Session_Type': 'Practice', 'Sleep_Duration': 9.2, 'Sleep_Score': 95, 'Sleep_Quality': 9, 'Soreness': 2, 'Stress': 2, 'RPE': 3, 'Distance': 5200, 'Acceleration_Count': 42, 'Max_Acceleration': 4.2, 'Deceleration_Count': 29, 'Max_Deceleration': 4.8, 'Max_Speed': 28.7, 'Injury_Illness': 'No', 'Injury_Type': ''},
    {'Date': '01/19/2023', 'Name': 'Milan', 'Position': 'Central Midfielder', 'Session_Type': 'Game', 'Sleep_Duration': 5.8, 'Sleep_Score': 72, 'Sleep_Quality': 6, 'Soreness': 6, 'Stress': 7, 'RPE': 8, 'Distance': 10050, 'Acceleration_Count': 70, 'Max_Acceleration': 6.4, 'Deceleration_Count': 69, 'Max_Deceleration': 7.0, 'Max_Speed': 30.4, 'Injury_Illness': 'No', 'Injury_Type': ''},
    {'Date': '01/20/2023', 'Name': 'Viktor', 'Position': 'Defender', 'Session_Type': 'Practice', 'Sleep_Duration': 6.8, 'Sleep_Score': 80, 'Sleep_Quality': 7, 'Soreness': 5, 'Stress': 5, 'RPE': 5, 'Distance': 8350, 'Acceleration_Count': 61, 'Max_Acceleration': 5.8, 'Deceleration_Count': 59, 'Max_Deceleration': 6.1, 'Max_Speed': 27.5, 'Injury_Illness': 'No', 'Injury_Type': ''},
    {'Date': '01/21/2023', 'Name': 'Bojan', 'Position': 'Goalkeeper', 'Session_Type': 'Game', 'Sleep_Duration': 7.8, 'Sleep_Score': 88, 'Sleep_Quality': 8, 'Soreness': 4, 'Stress': 4, 'RPE': 4, 'Distance': 4100, 'Acceleration_Count': 48, 'Max_Acceleration': 4.9, 'Deceleration_Count': 35, 'Max_Deceleration': 5.2, 'Max_Speed': 22.4, 'Injury_Illness': 'No', 'Injury_Type': ''},
    {'Date': '01/22/2023', 'Name': 'Filip', 'Position': 'Midfielder', 'Session_Type': 'Game', 'Sleep_Duration': 4.2, 'Sleep_Score': 65, 'Sleep_Quality': 5, 'Soreness': 9, 'Stress': 10, 'RPE': 10, 'Distance': 11800, 'Acceleration_Count': 95, 'Max_Acceleration': 8.1, 'Deceleration_Count': 85, 'Max_Deceleration': 8.3, 'Max_Speed': 33.1, 'Injury_Illness': 'Yes', 'Injury_Type': 'Sprained ankle'},
    {'Date': '01/23/2023', 'Name': 'Aleks', 'Position': 'Forward', 'Session_Type': 'Game', 'Sleep_Duration': 5.2, 'Sleep_Score': 70, 'Sleep_Quality': 6, 'Soreness': 8, 'Stress': 9, 'RPE': 9, 'Distance': 11400, 'Acceleration_Count': 88, 'Max_Acceleration': 7.7, 'Deceleration_Count': 80, 'Max_Deceleration': 8.1, 'Max_Speed': 33.8, 'Injury_Illness': 'Yes', 'Injury_Type': 'Thigh strain'},
    {'Date': '01/24/2023', 'Name': 'Goran', 'Position': 'Outside Back', 'Session_Type': 'Practice', 'Sleep_Duration': 8.4, 'Sleep_Score': 92, 'Sleep_Quality': 9, 'Soreness': 3, 'Stress': 4, 'RPE': 4, 'Distance': 7800, 'Acceleration_Count': 60, 'Max_Acceleration': 5.4, 'Deceleration_Count': 50, 'Max_Deceleration': 5.5, 'Max_Speed': 28.4, 'Injury_Illness': 'No', 'Injury_Type': ''},
    {'Date': '01/25/2023', 'Name': 'Marko', 'Position': 'Right Wing', 'Session_Type': 'Practice', 'Sleep_Duration': 7.1, 'Sleep_Score': 86, 'Sleep_Quality': 8, 'Soreness': 4, 'Stress': 5, 'RPE': 5, 'Distance': 7600, 'Acceleration_Count': 65, 'Max_Acceleration': 5.8, 'Deceleration_Count': 55, 'Max_Deceleration': 6.0, 'Max_Speed': 29.9, 'Injury_Illness': 'No', 'Injury_Type': ''},
    {'Date': '01/26/2023', 'Name': 'Darko', 'Position': 'Left Wing', 'Session_Type': 'Game', 'Sleep_Duration': 6.2, 'Sleep_Score': 74, 'Sleep_Quality': 6, 'Soreness': 5, 'Stress': 6, 'RPE': 7, 'Distance': 10800, 'Acceleration_Count': 78, 'Max_Acceleration': 6.6, 'Deceleration_Count': 72, 'Max_Deceleration': 7.0, 'Max_Speed': 32.0, 'Injury_Illness': 'No', 'Injury_Type': ''},
    {'Date': '01/27/2023', 'Name': 'Simo', 'Position': 'Central Midfielder', 'Session_Type': 'Practice', 'Sleep_Duration': 5.5, 'Sleep_Score': 69, 'Sleep_Quality': 5, 'Soreness': 7, 'Stress': 8, 'RPE': 8, 'Distance': 9400, 'Acceleration_Count': 72, 'Max_Acceleration': 6.2, 'Deceleration_Count': 64, 'Max_Deceleration': 6.8, 'Max_Speed': 30.0, 'Injury_Illness': 'No', 'Injury_Type': ''},
    {'Date': '01/28/2023', 'Name': 'Dario', 'Position': 'Defender', 'Session_Type': 'Practice', 'Sleep_Duration': 8.9, 'Sleep_Score': 91, 'Sleep_Quality': 9, 'Soreness': 3, 'Stress': 3, 'RPE': 4, 'Distance': 8200, 'Acceleration_Count': 63, 'Max_Acceleration': 5.0, 'Deceleration_Count': 56, 'Max_Deceleration': 5.1, 'Max_Speed': 27.7, 'Injury_Illness': 'No', 'Injury_Type': ''},
    {'Date': '01/29/2023', 'Name': 'Todor', 'Position': 'Outside Back', 'Session_Type': 'Game', 'Sleep_Duration': 4.8, 'Sleep_Score': 64, 'Sleep_Quality': 5, 'Soreness': 8, 'Stress': 9, 'RPE': 9, 'Distance': 11700, 'Acceleration_Count': 92, 'Max_Acceleration': 8.0, 'Deceleration_Count': 86, 'Max_Deceleration': 8.2, 'Max_Speed': 33.5, 'Injury_Illness': 'Yes', 'Injury_Type': 'Groin tightness'},
]
extra_df = pd.DataFrame(extra)
for col in ['Sleep_Duration', 'Sleep_Score', 'Sleep_Quality', 'Soreness', 'Stress', 'RPE', 'Distance', 'Acceleration_Count', 'Max_Acceleration', 'Deceleration_Count', 'Max_Deceleration', 'Max_Speed']:
    extra_df[col] = pd.to_numeric(extra_df[col], errors='coerce')
extra_df['Injury_Illness'] = extra_df['Injury_Illness'].astype(str).fillna('No').replace({'nan': 'No', '': 'No'})
extra_sleep = extra_df['Sleep_Duration'].clip(0, 10)
extra_sleep_quality = extra_df['Sleep_Quality'].clip(0, 10)
extra_stress = extra_df['Stress'].clip(1, 10)
extra_rpe = extra_df['RPE'].clip(1, 10)
extra_soreness = extra_df['Soreness'].clip(1, 10)
extra_fatigue = ((10 - extra_sleep) * 4) + ((10 - extra_sleep_quality) * 3) + extra_stress * 4 + extra_rpe * 3 + extra_soreness * 2
extra_fatigue = extra_fatigue.clip(0, 100)
extra_df['Fatigue'] = extra_fatigue.round(1)
extra_df['FatigueScore'] = extra_df['Fatigue']
extra_df['RiskLevel'] = pd.cut(extra_df['Fatigue'], bins=bins, labels=labels, include_lowest=True).astype(str)

out = pd.concat([df, extra_df], ignore_index=True)
out.to_csv(path, index=False)
print('Saved updated CSV with', len(out), 'rows')
