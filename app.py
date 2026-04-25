# ======================================================
# IMPORT LIBRARIES
# ======================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium


# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(page_title="Smart Pollution Navigation", layout="wide")

st.title("🌍 Smart Pollution-Aware Navigation System (Delhi)")

# ======================================================
# LOAD DATA
# ======================================================

df = pd.read_csv("final_dw_project_output.csv")

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.header("User Input")

locations = df['location'].unique()
user_location = st.sidebar.selectbox("Select Your Location", locations)

# ======================================================
# ZONE CLASSIFICATION
# ======================================================

location_pollution = df.groupby('location')['pm2_5'].mean()

green_limit = location_pollution.quantile(0.33)
yellow_limit = location_pollution.quantile(0.66)

def zone(pm):
    if pm <= green_limit:
        return "green"
    elif pm <= yellow_limit:
        return "yellow"
    else:
        return "red"

zone_map = {loc: zone(pm) for loc, pm in location_pollution.items()}

# ======================================================
# ALERT SYSTEM
# ======================================================

st.subheader("📍 Location Status")

if zone_map[user_location] == "red":
    st.error(f"⚠ {user_location} is a HIGH POLLUTION ZONE")
elif zone_map[user_location] == "yellow":
    st.warning(f"⚠ {user_location} is a MODERATE POLLUTION ZONE")
else:
    st.success(f"✅ {user_location} is a LOW POLLUTION ZONE")

# ======================================================
# MAP SECTION
# ======================================================

st.subheader("🗺️ Delhi Pollution Map")

# Delhi coordinates (approx)
coordinates = {
    "Connaught Place": [28.6315, 77.2167],
    "Anand Vihar": [28.646, 77.315],
    "Dwarka": [28.5921, 77.046],
    "Rohini": [28.7041, 77.1025],
    "Saket": [28.5245, 77.2066],
    "Karol Bagh": [28.6516, 77.1909],
    "Lajpat Nagar": [28.5677, 77.2433],
    "Chandni Chowk": [28.6562, 77.2303],
    "Pitampura": [28.7037, 77.1313],
    "Vasant Kunj": [28.5204, 77.1552],
    "Mayur Vihar": [28.6002, 77.3237],
    "Janakpuri": [28.6219, 77.0878],
    "Rajouri Garden": [28.6505, 77.1194],
    "Okhla": [28.5355, 77.2732],
    "Narela": [28.852, 77.092],
    "Punjabi Bagh": [28.668, 77.132],
    "Hauz Khas": [28.5494, 77.2001],
    "Shahdara": [28.6735, 77.289],
    "Kalkaji": [28.549, 77.258],
    "Patel Nagar": [28.651, 77.158],
    "AIIMS": [28.5672, 77.2100],
    "Dhaula Kuan": [28.5916, 77.1613],
    "ITO": [28.6289, 77.2410],
    "Ashok Vihar": [28.6920, 77.1760],
    "Model Town": [28.7049, 77.1930]
}

m = folium.Map(location=[28.6139, 77.2090], zoom_start=11)

# Add markers
for loc, coord in coordinates.items():
    color = zone_map.get(loc, "green")
    
    folium.CircleMarker(
        location=coord,
        radius=8,
        color=color,
        fill=True,
        fill_color=color,
        popup=f"{loc} ({color.upper()})"
    ).add_to(m)

from math import radians, cos, sin, sqrt, atan2

def calculate_distance(coord1, coord2):
    R = 6371
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c

#==================
#New Routing
#=====================
user_coord = coordinates[user_location]

# SAFE ZONES = yellow + green
safe_locations = [
    loc for loc in coordinates 
    if zone_map.get(loc, "green") in ["green", "yellow"] and loc != user_location
]

nearest_location = None
min_distance = float('inf')

for loc in safe_locations:
    dist = calculate_distance(user_coord, coordinates[loc])

    if dist < min_distance:
        min_distance = dist
        nearest_location = loc

best_route = nearest_location
best_distance = min_distance










#=============================
# MULTIPLE ROUTE LOGIC
# ======================================================
user_coord = coordinates[user_location]


routes = []

max_dist = 20
max_pollution = location_pollution.max()

for loc in safe_locations:
    if loc == user_location:
        continue

    dist = calculate_distance(user_coord, coordinates[loc])
    pollution = location_pollution.get(loc, df['pm2_5'].mean())

    norm_dist = dist / max_dist
    norm_poll = pollution / max_pollution

    eco_score = norm_poll
    fast_score = norm_dist
    balanced_score = (0.6 * norm_dist) + (0.4 * norm_poll)

    routes.append({
        "location": loc,
        "distance": dist,
        "pollution": pollution,
        "eco_score": eco_score,
        "fast_score": fast_score,
        "balanced_score": balanced_score
    })



# SORT ROUTES
eco_route = min(routes, key=lambda x: x['eco_score'])
fast_route = min(routes, key=lambda x: x['distance'])
balanced_route = min(routes, key=lambda x: x['balanced_score'])




  



#============================
# DISPLAY MULTIPLE ROUTES (UI)
#==============================


if zone_map[user_location] == "red":
    st.subheader("🚦 Suggested Safe Route")

    st.success(f"➡ Nearest safer zone: {best_route}")
    st.write(f"📏 Distance: {round(best_distance,2)} km")

    time = int(best_distance * 3)
    st.write(f"⏱ Estimated Time: {time} mins")
else:
    st.info("You are already in a safe (Green/Yellow) zone")

if zone_map[user_location] == "red":
    st.subheader("🚦 Route Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("🌿 Eco Route")
        st.write(f"📍 {eco_route['location']}")
        st.write(f"📏 {round(eco_route['distance'],2)} km")

    with col2:
        st.info("⚡ Fastest Route")
        st.write(f"📍 {fast_route['location']}")
        st.write(f"📏 {round(fast_route['distance'],2)} km")

    with col3:
        st.warning("⚖ Balanced Route")
        st.write(f"📍 {balanced_route['location']}")
        st.write(f"📏 {round(balanced_route['distance'],2)} km")

#=================
# DRAW ALL ROUTES ON MAP
#=========================
if zone_map[user_location] == "red":

    # MAIN ROUTE
    folium.PolyLine(
        [coordinates[user_location], coordinates[best_route]],
        color="blue", weight=8, tooltip="Main Safe Route"
    ).add_to(m)

    # ECO ROUTE
    if eco_route['location'] != best_route:
        folium.PolyLine(
            [coordinates[user_location], coordinates[eco_route['location']]],
            color="green", weight=5, tooltip="Eco Route"
        ).add_to(m)

    # BALANCED ROUTE
    if balanced_route['location'] != best_route:
        folium.PolyLine(
            [coordinates[user_location], coordinates[balanced_route['location']]],
            color="orange", weight=5, tooltip="Balanced Route"
        ).add_to(m)

# Show map
st_folium(m, width=900)
st.markdown("""
### 🗺️ Legend:
- 🔴 Red → High Pollution  
- 🟡 Yellow → Moderate  
- 🟢 Green → Safe  
- 🔵 Blue Line → Main Route  
- 🟢 Green Line → Eco Route  
- 🟠 Orange Line → Balanced Route  
""")

# ======================================================
# DASHBOARD (GRAPHS)
# ======================================================

st.subheader("📊 Data Insights Dashboard")

col1, col2 = st.columns(2)

# Traffic vs Pollution
with col1:
    traffic_pm = df.groupby('traffic_level')['pm2_5'].mean()
    fig, ax = plt.subplots()
    traffic_pm.plot(kind='bar', ax=ax)
    ax.set_title("Traffic vs Pollution")
    ax.set_ylabel("PM2.5")
    st.pyplot(fig)

# Correlation Heatmap
with col2:
    corr = df[['pm2_5','traffic_numeric','temperature','humidity','wind_speed']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")
st.markdown("Smart City DWDM Project | Pollution-Aware Navigation System")