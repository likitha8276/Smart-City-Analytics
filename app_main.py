# ======================================================
# IMPORT LIBRARIES
# ======================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from math import radians, cos, sin, sqrt, atan2

# ======================================================
# PAGE CONFIG + PREMIUM UI
# ======================================================

st.set_page_config(page_title="AQI Intelligence Pro", layout="wide")

st.markdown("""
<style>
html, body {
    background: radial-gradient(circle at top left, #0e1525, #010409);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #00d4ff, #0052d4);
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🌍 AQI Intelligence + Smart Pollution Navigation System")

# ======================================================
# LOAD DATA
# ======================================================

@st.cache_data
def load_data():
    try:
        return pd.read_csv("final_dw_project_output.csv")
    except:
        st.warning("Using demo data")
        return pd.DataFrame({
            'hour': list(range(24)) * 4,
            'pm2_5': [30,45,80,120]*24,
            'location': ['Connaught Place','Dwarka','Saket','Rohini']*24,
            'traffic_level': ['Low','Medium','High','High']*24,
            'alert': ['NORMAL','NORMAL','HIGH ALERT','HIGH ALERT']*24,
            'traffic_numeric': [1,2,3,3]*24,
            'temperature':[25]*96,
            'humidity':[50]*96,
            'wind_speed':[5]*96
        })

df = load_data()

# ======================================================
# FULL DELHI COORDINATES
# ======================================================

coordinates = {
    "Connaught Place":[28.6315,77.2167],"Anand Vihar":[28.646,77.315],
    "Dwarka":[28.5921,77.046],"Rohini":[28.7041,77.1025],
    "Saket":[28.5245,77.2066],"Karol Bagh":[28.6516,77.1909],
    "Lajpat Nagar":[28.5677,77.2433],"Chandni Chowk":[28.6562,77.2303],
    "Pitampura":[28.7037,77.1313],"Vasant Kunj":[28.5204,77.1552],
    "Mayur Vihar":[28.6002,77.3237],"Janakpuri":[28.6219,77.0878],
    "Rajouri Garden":[28.6505,77.1194],"Okhla":[28.5355,77.2732],
    "Narela":[28.852,77.092],"Punjabi Bagh":[28.668,77.132],
    "Hauz Khas":[28.5494,77.2001],"Shahdara":[28.6735,77.289],
    "Kalkaji":[28.549,77.258],"Patel Nagar":[28.651,77.158],
    "AIIMS":[28.5672,77.2100],"Dhaula Kuan":[28.5916,77.1613],
    "ITO":[28.6289,77.2410],"Ashok Vihar":[28.6920,77.1760],
    "Model Town":[28.7049,77.1930]
}

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("⚙ Control Panel")

user_location = st.sidebar.selectbox("📍 Location", list(coordinates.keys()))
traffic_filter = st.sidebar.selectbox("🚗 Traffic Level", df['traffic_level'].unique())

st.sidebar.markdown("---")

temp = st.sidebar.slider("🌡 Temperature",0,50,25)
humidity = st.sidebar.slider("💧 Humidity",0,100,50)
wind = st.sidebar.slider("🌬 Wind Speed",0,30,10)
traffic_input = st.sidebar.selectbox("🚦 Prediction Traffic",["Low","Medium","High"])

# ======================================================
# FILTER DATA
# ======================================================

filtered = df[(df['location']==user_location) & (df['traffic_level']==traffic_filter)]

# ======================================================
# KPI DASHBOARD
# ======================================================

st.subheader("📊 AQI Overview")

c1,c2,c3 = st.columns(3)
c1.metric("Average AQI", round(df['pm2_5'].mean(),2))
c2.metric("Max AQI", round(df['pm2_5'].max(),2))
c3.metric("Alerts", df[df['alert']=="HIGH ALERT"].shape[0])

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

zone_map = {loc: zone(location_pollution.get(loc, df['pm2_5'].mean())) for loc in coordinates}

st.subheader("📍 Location Status")

if zone_map[user_location] == "red":
    st.error(f"🔴 {user_location} is HIGH POLLUTION")
elif zone_map[user_location] == "yellow":
    st.warning(f"🟡 {user_location} is MODERATE POLLUTION")
else:
    st.success(f"🟢 {user_location} is SAFE")

# ======================================================
# AI PREDICTION
# ======================================================

model = LinearRegression()
X = df[['traffic_numeric','temperature','humidity','wind_speed']]
y = df['pm2_5']
model.fit(X,y)

traffic_map={"Low":1,"Medium":2,"High":3}
pred = model.predict([[traffic_map[traffic_input],temp,humidity,wind]])

st.subheader("🤖 AI Prediction")
st.info(f"Predicted AQI: {round(pred[0],2)}")

# ======================================================
# MAP + ROUTING SYSTEM (FULL)
# ======================================================

st.subheader("🗺 Smart Pollution Map & Routing")

m = folium.Map(location=[28.6139,77.2090],zoom_start=11)

# markers
for loc,coord in coordinates.items():
    color = zone_map[loc]
    folium.CircleMarker(
        location=coord,
        radius=8,
        color=color,
        fill=True,
        fill_color=color,
        popup=f"{loc} ({color.upper()})"
    ).add_to(m)

# distance function
def calculate_distance(c1,c2):
    R=6371
    lat1,lon1=c1
    lat2,lon2=c2
    dlat=radians(lat2-lat1)
    dlon=radians(lon2-lon1)
    a=sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*atan2(sqrt(a),sqrt(1-a))

user_coord = coordinates[user_location]

safe_locations = [
    loc for loc in coordinates
    if zone_map[loc] in ["green","yellow"] and loc != user_location
]

routes = []
max_dist = 20
max_poll = location_pollution.max()

for loc in safe_locations:
    dist = calculate_distance(user_coord, coordinates[loc])
    poll = location_pollution.get(loc, df['pm2_5'].mean())

    routes.append({
        "loc":loc,
        "dist":dist,
        "poll":poll,
        "eco":poll/max_poll,
        "fast":dist/max_dist,
        "balanced":0.6*(dist/max_dist)+0.4*(poll/max_poll)
    })

# best routes
eco = min(routes, key=lambda x:x['eco'])
fast = min(routes, key=lambda x:x['dist'])
balanced = min(routes, key=lambda x:x['balanced'])

best = min(routes, key=lambda x:x['dist'])

# ======================================================
# ROUTE DISPLAY
# ======================================================

if zone_map[user_location] == "red":

    st.subheader("🚦 Suggested Safe Route")

    best_distance = best['dist']
    time = int(best_distance * 3)

    st.success(f"➡ Nearest safer zone: {best['loc']}")
    st.write(f"📏 Distance: {round(best_distance,2)} km")
    st.write(f"⏱ Estimated Time: {time} mins")

    st.subheader("🚦 Route Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("🌿 Eco Route")
        st.write(f"{eco['loc']}")
        st.write(f"{round(eco['dist'],2)} km")

    with col2:
        st.info("⚡ Fast Route")
        st.write(f"{fast['loc']}")
        st.write(f"{round(fast['dist'],2)} km")

    with col3:
        st.warning("⚖ Balanced Route")
        st.write(f"{balanced['loc']}")
        st.write(f"{round(balanced['dist'],2)} km")

else:
    st.info("You are already in a safe (Green/Yellow) zone")

# draw routes
if zone_map[user_location] == "red":
    folium.PolyLine([user_coord,coordinates[best['loc']]],color="blue",weight=8).add_to(m)
    folium.PolyLine([user_coord,coordinates[eco['loc']]],color="green",weight=5).add_to(m)
    folium.PolyLine([user_coord,coordinates[balanced['loc']]],color="orange",weight=5).add_to(m)

st_folium(m,width=900)

# ======================================================
# LEGEND
# ======================================================

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
# FOOTER
# ======================================================

st.markdown("---")
st.markdown("🚀 Smart City")