import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AQI Intelligence Pro", layout="wide", initial_sidebar_state="expanded")

# =========================
# CUSTOM CSS (PREMIUM DESIGN)
# =========================
st.markdown("""
<style>
    /* Main Background & Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: radial-gradient(circle at top left, #0e1525, #010409);
        color: #ffffff;
    }

    /* Glassmorphism KPI Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #00d4ff;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 15, 30, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00d4ff 0%, #0052d4 100%);
        color: white;
        border: none;
        padding: 12px;
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 1px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0px 0px 20px rgba(0, 212, 255, 0.5);
    }

    /* Prediction Result Box */
    .prediction-box {
        padding: 20px;
        background: rgba(0, 212, 255, 0.1);
        border-radius: 15px;
        border: 1px solid #00d4ff;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# DATA LOAD (MOCK DATA FOR DEMO)
# =========================
# Replace this with: df = pd.read_csv("final_dw_project_output.csv")
@st.cache_data
def load_data():
    # Placeholder for your actual CSV loading
    try:
        return pd.read_csv("final_dw_project_output.csv")
    except:
        # Fallback dummy data if file is missing
        data = {
            'hour': list(range(24)) * 4,
            'pm2_5': [30, 45, 80, 120, 150, 60, 40, 35] * 12,
            'location': ['Downtown', 'Suburbs', 'Industrial', 'Parks'] * 24,
            'traffic_level': ['Low', 'Medium', 'High', 'High'] * 24,
            'alert': ['NORMAL', 'NORMAL', 'HIGH ALERT', 'HIGH ALERT'] * 24,
            'traffic_numeric': [1, 2, 3, 3] * 24,
            'temperature': [25] * 96, 'humidity': [50] * 96, 'wind_speed': [5] * 96
        }
        return pd.DataFrame(data)

df = load_data()

# =========================
# SIDEBAR FILTERS
# =========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1684/1684375.png", width=80)
    st.title("Control Center")
    st.markdown("Adjust parameters to filter dashboard view.")
    
    location = st.selectbox("📍 Select Location", df['location'].unique())
    traffic = st.radio("🚗 Traffic Level", df['traffic_level'].unique(), horizontal=True)
    
    st.divider()
    st.caption("v2.1.0 Premium Access")

# =========================
# MAIN DASHBOARD
# =========================
filtered = df[(df['location'] == location) & (df['traffic_level'] == traffic)]

st.markdown("<h1 style='color:#00d4ff;'>🌍 AQI Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("### Strategic Environmental Insights")

# KPI SECTION
col1, col2, col3 = st.columns(3)
col1.metric("Average AQI", f"{round(df['pm2_5'].mean(), 1)} µg/m³", delta="-2.5%")
col2.metric("Max Recorded", f"{round(df['pm2_5'].max(), 1)}", delta="High Risk", delta_color="inverse")
col3.metric("Active Alerts", df[df['alert'] == "HIGH ALERT"].shape[0], delta="In past 24h")

st.write("")

# CHARTS SECTION
left, right = st.columns([3, 2])

with left:
    st.subheader("📈 Temporal AQI Concentration")
    fig = px.area(filtered, x='hour', y='pm2_5', 
                  color_discrete_sequence=['#00d4ff'],
                  template="plotly_dark")
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        margin=dict(l=0, r=0, t=20, b=0),
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("🚗 Traffic Correlation")
    traffic_avg = df.groupby('traffic_level')['pm2_5'].mean().reset_index()
    fig2 = px.bar(traffic_avg, x='traffic_level', y='pm2_5',
                  color='pm2_5', color_continuous_scale='Blues')
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        height=350,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# =========================
# AI PREDICTION ENGINE
# =========================
st.markdown("## 🔮 ML Predictive Forecasting")
st.markdown("Configure environmental factors to simulate future Air Quality Index levels.")

# Model Training (Keep simple for UI focus)
model = LinearRegression()
X = df[['traffic_numeric', 'temperature', 'humidity', 'wind_speed']]
y = df['pm2_5']
model.fit(X, y)

with st.container():
    p1, p2, p3, p4 = st.columns(4)
    temp = p1.slider("🌡 Temperature (°C)", 0, 50, 25)
    humidity = p2.slider("💧 Humidity (%)", 0, 100, 54)
    wind = p3.slider("🌬 Wind Speed (km/h)", 0, 30, 12)
    traffic_input = p4.selectbox("🚦 Traffic Density", ["Low", "Medium", "High"])

    traffic_map = {"Low": 1, "Medium": 2, "High": 3}
    
    st.write("")
    if st.button("🚀 Run AI Simulation"):
        pred = model.predict([[traffic_map[traffic_input], temp, humidity, wind]])
        
        st.markdown(f"""
            <div class="prediction-box">
                <h2 style="margin:0; color:#00d4ff;">Predicted AQI: {round(pred[0], 2)}</h2>
                <p style="margin:0; opacity:0.8;">Classification: {'Moderate' if pred[0] < 100 else 'Unhealthy'}</p>
            </div>
        """, unsafe_allow_html=True)

# FOOTER
st.write("")
st.markdown("<p style='text-align:center; opacity:0.5;'>System Status: Operational | Data Sync: Real-time</p>", unsafe_allow_html=True)