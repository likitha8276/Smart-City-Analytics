# ======================================================
# IMPORT LIBRARIES
# ======================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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
# STEP 3 : LOAD (MERGE DATA)
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
plt.xlabel("Hour")
plt.ylabel("Traffic Level")
plt.grid(True)
plt.show()


# ======================================================
# OBJECTIVE 2 : MONTHLY POLLUTION TREND
# ======================================================

df['month'] = df['date'].dt.month
monthly = df.groupby('month')['pm2_5'].mean()

plt.figure(figsize=(10,5))
plt.plot(monthly.index, monthly.values, marker='o')
plt.title("Monthly PM2.5 Trend")
plt.xlabel("Month")
plt.ylabel("PM2.5")
plt.grid(True)
plt.show()


# ======================================================
# OBJECTIVE 3 : WEATHER IMPACT
# ======================================================

weather_trend = df.groupby('hour')[['temperature','pm2_5']].mean()

fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(weather_trend.index, weather_trend['pm2_5'])
ax2 = ax1.twinx()
ax2.plot(weather_trend.index, weather_trend['temperature'])
plt.title("Temperature vs Pollution")
plt.show()


# ======================================================
# OBJECTIVE 4 : SPIKE DETECTION
# ======================================================

threshold = df['pm2_5'].mean() + 2*df['pm2_5'].std()

spikes = df[df['pm2_5'] > threshold]

plt.figure(figsize=(10,5))
plt.plot(df['hour'], df['pm2_5'])
plt.scatter(spikes['hour'], spikes['pm2_5'])
plt.title("Pollution Spikes")
plt.show()


# ======================================================
# OBJECTIVE 5 : CORRELATION
# ======================================================

corr = df[['pm2_5','traffic_numeric','temperature','humidity','wind_speed']].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True)
plt.show()


# ======================================================
# OBJECTIVE 6 : ML MODEL
# ======================================================

X = df[['traffic_numeric','temperature','humidity','wind_speed']]
y = df['pm2_5']

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

print("R2 Score:", r2_score(y, pred))


# ======================================================
# SAVE FINAL MERGED DATA
# ======================================================

df.to_csv("final_dw_project_output.csv", index=False)


# ======================================================
# ================= DATA WAREHOUSE =====================
# ======================================================

print("\nCreating Data Warehouse Tables...")

# ================================
# DIMENSION TABLES
# ================================

# DATE DIM
dim_date = df[['date']].drop_duplicates().reset_index(drop=True)
dim_date['date_id'] = dim_date.index + 1

# TIME DIM
dim_time = df[['hour']].drop_duplicates().reset_index(drop=True)
dim_time['time_id'] = dim_time.index + 1

# LOCATION DIM
dim_location = df[['location']].drop_duplicates().reset_index(drop=True)
dim_location['location_id'] = dim_location.index + 1

# WEATHER DIM
dim_weather = df[['temperature','humidity','wind_speed']].drop_duplicates().reset_index(drop=True)
dim_weather['weather_id'] = dim_weather.index + 1

# TRAFFIC DIM
dim_traffic = df[['traffic_level','traffic_numeric']].drop_duplicates().reset_index(drop=True)
dim_traffic['traffic_id'] = dim_traffic.index + 1


# ================================
# FACT TABLE
# ================================

fact = df.merge(dim_date, on='date') \
         .merge(dim_time, on='hour') \
         .merge(dim_location, on='location') \
         .merge(dim_weather, on=['temperature','humidity','wind_speed']) \
         .merge(dim_traffic, on=['traffic_level','traffic_numeric'])

fact = fact[['date_id','time_id','location_id','weather_id','traffic_id','pm2_5']]


# ================================
# SAVE DATA WAREHOUSE TABLES
# ================================

dim_date.to_csv("dim_date.csv", index=False)
dim_time.to_csv("dim_time.csv", index=False)
dim_location.to_csv("dim_location.csv", index=False)
dim_weather.to_csv("dim_weather.csv", index=False)
dim_traffic.to_csv("dim_traffic.csv", index=False)
fact.to_csv("fact_pollution.csv", index=False)

print("✅ Data Warehouse Created Successfully!")