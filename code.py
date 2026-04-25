# ======================================================
# IMPORT LIBRARIES
# ======================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

ACCENT_ORANGE = '#e67e22'
# ======================================================
# STEP 1 : EXTRACT
# ======================================================

aqi = pd.read_csv("delhi_aqi.csv")
weather = pd.read_csv("testset.csv")

print("AQI Data:")
print(aqi.head())

print("\nWeather Data:")
print(weather.head())


# ======================================================
# STEP 2 : TRANSFORM
# ======================================================

weather.columns = weather.columns.str.strip()

aqi['date'] = pd.to_datetime(aqi['date'])
weather['datetime_utc'] = pd.to_datetime(weather['datetime_utc'])

aqi['hour'] = aqi['date'].dt.hour
weather['hour'] = weather['datetime_utc'].dt.hour

weather.replace(-9999, pd.NA, inplace=True)

weather = weather[['hour','_tempm','_hum','_wspdm']]

weather.rename(columns={
    '_tempm':'temperature',
    '_hum':'humidity',
    '_wspdm':'wind_speed'
}, inplace=True)

weather_hourly = weather.groupby('hour').mean().reset_index()


# ======================================================
# STEP 3 : LOAD (DATA WAREHOUSE)
# ======================================================

df = pd.merge(aqi, weather_hourly, on='hour', how='left')

print("\nMerged Dataset:")
print(df.head())


# ======================================================
# STEP 4 : TRAFFIC PROXY
# ======================================================

def traffic(hour):
    if 7 <= hour <= 10 or 17 <= hour <= 20:
        return "High"
    elif 11 <= hour <= 16:
        return "Medium"
    else:
        return "Low"

df['traffic_level'] = df['hour'].apply(traffic)

traffic_map = {"Low":1,"Medium":2,"High":3}
df['traffic_numeric'] = df['traffic_level'].map(traffic_map)


# ======================================================
# STEP 5 : ADD DELHI LOCATIONS
# ======================================================

places = [
"Connaught Place","Anand Vihar","Dwarka","Rohini","Saket",
"Karol Bagh","Lajpat Nagar","Chandni Chowk","Pitampura","Vasant Kunj",
"Mayur Vihar","Janakpuri","Rajouri Garden","Okhla","Narela",
"Punjabi Bagh","Hauz Khas","Shahdara","Kalkaji","Patel Nagar"
]

df['location'] = [places[i % len(places)] for i in range(len(df))]

df = df.head(1000)


# ======================================================
# OBJECTIVE 1 : TRAFFIC INTENSITY GRAPH
# ======================================================

traffic_hour = df.groupby('hour')['traffic_numeric'].mean()

plt.figure(figsize=(10,5))
plt.plot(traffic_hour.index, traffic_hour.values, marker='o')

plt.title("Traffic Intensity vs Time of Day (Delhi)")
plt.xlabel("Hour of Day (0–23)")
plt.ylabel("Traffic Level (Low=1, Medium=2, High=3)")

plt.xticks(range(0,24))
plt.yticks([1,2,3],["Low","Medium","High"])

plt.grid(True)
plt.show()

# ======================================================
# OBJECTIVE 7 : TRAFFIC VS POLLUTION
# ======================================================

traffic_pm = df.groupby('traffic_level')['pm2_5'].mean()

traffic_pm.plot(kind='bar')

plt.title("Impact of Traffic Level on Pollution")
plt.xlabel("Traffic Level")
plt.ylabel("Average PM2.5 (µg/m³)")

plt.show()


# ======================================================
# OBJECTIVE 2 : MONTHLY POLLUTION TREND
# ======================================================

df['month'] = df['date'].dt.month
monthly = df.groupby('month')['pm2_5'].mean()

plt.figure(figsize=(10,5))
plt.plot(monthly.index, monthly.values, marker='o')

plt.title("Monthly PM2.5 Pollution Trend in Delhi")
plt.xlabel("Month (1=Jan, 12=Dec)")
plt.ylabel("Average PM2.5 (µg/m³)")

plt.grid(True)
plt.show()

# ======================================================
# 4. WEATHER IMPACT (DUAL AXIS)
# ======================================================

weather_trend = df.groupby('hour')[['temperature','pm2_5']].mean()

fig, ax1 = plt.subplots(figsize=(12, 6))

# PM2.5 Plot
ax1.plot(weather_trend.index, weather_trend['pm2_5'], color='#2c3e50', linewidth=2, label='PM2.5')
ax1.fill_between(weather_trend.index, weather_trend['pm2_5'], color='#2c3e50', alpha=0.05)
ax1.set_ylabel("PM2.5 (µg/m³)", fontsize=12, color='#2c3e50')

# Temperature Plot
ax2 = ax1.twinx()
ax2.plot(weather_trend.index, weather_trend['temperature'], color=ACCENT_ORANGE, linewidth=3, linestyle='--', label='Temp')
ax2.set_ylabel("Temperature (°C)", fontsize=12, color=ACCENT_ORANGE)

plt.title("Atmospheric Interaction: Temperature vs. Pollution", fontsize=16, pad=20)
ax1.set_xlabel("Hour of Day", fontsize=12)
ax1.set_xticks(range(0, 24))
plt.tight_layout()
plt.show()

# ======================================================
# OBJECTIVE 4 : POLLUTION SPIKE DETECTION
# ======================================================

threshold = df['pm2_5'].mean() + 2*df['pm2_5'].std()

spikes = df[df['pm2_5'] > threshold]

plt.figure(figsize=(10,5))
plt.plot(df['hour'], df['pm2_5'], label="PM2.5")
plt.scatter(spikes['hour'], spikes['pm2_5'], color='red', label="Spike")

plt.title("Pollution Spike Detection")
plt.xlabel("Hour")
plt.ylabel("PM2.5 (µg/m³)")

plt.legend()
plt.grid(True)
plt.show()

# ======================================================
# 5. SPIKE DETECTION (IMPROVED)
# ======================================================

iso = IsolationForest(contamination=0.05)
df['anomaly'] = iso.fit_predict(df[['pm2_5']])

anomalies = df[df['anomaly'] == -1]

plt.figure(figsize=(10,5))
plt.scatter(df['hour'], df['pm2_5'], label="Normal")
plt.scatter(anomalies['hour'], anomalies['pm2_5'], label="Anomaly")
plt.title("Advanced Pollution Spike Detection", fontsize=14)
plt.xlabel("Hour")
plt.ylabel("PM2.5")
plt.legend()
plt.show()


# ======================================================
# OBJECTIVE 5 : POLLUTION ZONES
# ======================================================


location_pollution = df.groupby('location')['pm2_5'].mean()

green_limit = location_pollution.quantile(0.33)
yellow_limit = location_pollution.quantile(0.66)

def zone(pm):
    if pm <= green_limit:
        return "Green"
    elif pm <= yellow_limit:
        return "Yellow"
    else:
        return "Red"

df['zone_label'] = df['pm2_5'].apply(zone)

print("\nBasic Zones:")
print(df[['location','zone_label']])

sns.set_theme(style="whitegrid")



# GRAPH 1: Zone Map for Traffic Police Diversion
plt.figure(figsize=(14, 7))
sorted_zones = location_summary.sort_values('pm2_5', ascending=False)
colors = sorted_zones['zone'].map({'Red': '#e74c3c', 'Yellow': '#f1c40f', 'Green': '#2ecc71'})
sns.barplot(x='location', y='pm2_5', data=sorted_zones, palette=colors.tolist())

plt.xticks(rotation=45, ha='right')
plt.xlabel("Delhi Sub-Areas", fontsize=12, fontweight='bold')
plt.ylabel("Average PM2.5 (µg/m³)", fontsize=12, fontweight='bold')
plt.title("Vulnerability Zones: Target Areas for Traffic Diversion", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("delhi_zones_map.png")

# GRAPH 2: Dual Axis Traffic vs Pollution
plt.figure(figsize=(12, 6))
hourly_avg = df.groupby('hour').agg({'pm2_5':'mean', 'traffic_numeric':'mean'}).reset_index()
ax1 = sns.barplot(x='hour', y='pm2_5', data=hourly_avg, palette="YlOrRd", alpha=0.6)
ax2 = ax1.twinx()
sns.lineplot(x='hour', y='traffic_numeric', data=hourly_avg, marker='o', color='darkblue', linewidth=3, ax=ax2)

ax1.set_xlabel("Hour of the Day (24-Hour Format)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Avg PM2.5 Concentration (µg/m³)", fontsize=12, fontweight='bold', color='red')
ax2.set_ylabel("Traffic Intensity Index (1=Low, 3=High)", fontsize=12, fontweight='bold', color='darkblue')
ax2.set_yticks([1, 2, 3])
ax2.set_yticklabels(['Low Traffic', 'Medium', 'Peak Traffic'])
plt.title("Correlation Analysis: Peak Traffic Hours vs. Air Quality Spikes", fontsize=14, fontweight='bold')
plt.savefig("traffic_pollution_correlation.png")


# ======================================================
# 8. CORRELATION
# ======================================================

corr = df[['pm2_5','traffic_numeric','temperature','humidity','wind_speed']].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True)
plt.title("Correlation Between Factors", fontsize=14)
plt.show()






# ======================================================
# OBJECTIVE 9 : PEAK HOUR
# ======================================================

peak_hour = df.groupby('hour')['pm2_5'].mean().idxmax()
print("\nPeak Pollution Hour:", peak_hour)


# ======================================================
# OBJECTIVE 10 : ALERT SYSTEM
# ======================================================

df['alert'] = df['pm2_5'].apply(lambda x: "HIGH ALERT" if x > threshold else "Normal")


# ======================================================
# OBJECTIVE 11 : DECISION SYSTEM
# ======================================================

def decision(row):
    if row['traffic_level']=="High" and row['pm2_5']>threshold:
        return "Divert Traffic"
    elif row['pm2_5']>threshold:
        return "Issue Alert"
    else:
        return "Normal"

df['decision'] = df.apply(decision, axis=1)


# ======================================================
# OBJECTIVE 12 : REGRESSION MODEL
# ======================================================

X = df[['traffic_numeric','temperature','humidity','wind_speed']]
y = df['pm2_5']

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

print("\nR2 Score:", r2_score(y, pred))


# ======================================================
# OBJECTIVE 13 : FEATURE IMPORTANCE
# ======================================================

rf = RandomForestRegressor()
rf.fit(X, y)

importance = pd.Series(rf.feature_importances_, index=X.columns)
print("\nFeature Importance:\n", importance)


# ======================================================
# OBJECTIVE 14 : K-MEANS CLUSTERING
# ======================================================

kmeans = KMeans(n_clusters=3)
df['cluster'] = kmeans.fit_predict(X)

print("\nCluster Sample:")
print(df[['pm2_5','cluster']].head())


# ======================================================
# OBJECTIVE 15 : ANOMALY DETECTION
# ======================================================

iso = IsolationForest(contamination=0.05)
df['anomaly'] = iso.fit_predict(df[['pm2_5']])

print("\nAnomalies Detected:", (df['anomaly'] == -1).sum())


# ======================================================
# OBJECTIVE 16 : DECISION TREE CLASSIFIER
# ======================================================

clf = DecisionTreeClassifier()
clf.fit(X, df['zone_label'])

pred_zone = clf.predict(X)

print("\nPredicted Zones:", pred_zone[:10])


# ======================================================
# OBJECTIVE 17 : ASSOCIATION PATTERN
# ======================================================

high_pollution = df[df['pm2_5'] > threshold]
print("\nHigh Pollution by Traffic:")
print(high_pollution.groupby('traffic_level').size())


# ======================================================
# OBJECTIVE 18 : ROUTE SUGGESTION
# ======================================================

def route(row):
    if row['zone_label']=="Red":
        return "Avoid Route"
    elif row['zone_label']=="Yellow":
        return "Moderate"
    else:
        return "Safe"

df['route'] = df.apply(route, axis=1)

print("\nRoute Suggestions:")
print(df[['location','zone_label','route']].head())


# ======================================================
# FINAL SAVE
# ======================================================

df.to_csv("FINAL_DWDM_PROJECT.csv", index=False)

print("\nProject Completed Successfully")