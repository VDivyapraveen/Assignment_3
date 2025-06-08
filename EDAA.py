import streamlit as st

# --- MUST be first Streamlit command ---
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    /* Sidebar background color */
    section[data-testid="stSidebar"] {
        background-color: #d8b26e !important;
    }

    /* Sidebar card style */
    .block-container .stSidebarContent {
        padding-top: 2rem;
    }

    .sidebar-container {
        background-color: #acc6e0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }

    /* Menu header */
    .sidebar-title {
        font-size: 26px;
        font-weight: bold;
        color: black;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }

    .sidebar-title:before {
        content: "\\1F5A5";  /* 🖥 icon */
        margin-right: 10px;
    }

    /* Custom radio styling */
    .sidebar-radio > div > label {
        background-color: #b36920 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
    }

    .sidebar-radio > div > label:hover {
        background-color: #8f4e16 !important;
    }

    .sidebar-radio > div > label > div:first-child {
        margin-right: 10px;
    }
    </style>
""", unsafe_allow_html=True)





# --- Background Image Setup ---
import base64

def set_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set your local image path here
set_bg_from_local("D:/MyProjectsDuplicates/project3/pic.jpg")

# --- Other Imports ---
import pandas as pd
import plotly.express as px
import mysql.connector
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# --- MySQL Connection ---
mydb = mysql.connector.connect(
    host="localhost",
    database="project3",
    user="root",
    password=""
)
mycursor = mydb.cursor()

import streamlit as st

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');

    * {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    html, body, [class*="css"], [class^="st-"], [class*="st-"] {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    .stMarkdown, .stDataFrame, .stTable, .stTextInput, .stSelectbox, .stSlider, .stNumberInput,
    .stButton, .stRadio, .stMultiSelect, .stDateInput, .stCheckbox {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)
with st.sidebar:
    # st.markdown('<div class="sidebar-container">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">MAIN MENU</div><hr>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)




# --- Streamlit UI ---
# st.sidebar.title("CROP PRODUCTION EXPLORER")
page = st.sidebar.radio("Navigate:", ["Introduction","Top producers analysis", "Trend analysis", "Geographical analysis", "Production predictor", "End page"])

# --- Fetch crop options ---
mycursor.execute("SELECT DISTINCT Item FROM data")
crops = sorted([row[0] for row in mycursor.fetchall()])
selected_crop = st.sidebar.selectbox("Choose a Crop", crops)

# --- Page: Introduction ---
if page == "Introduction":
    st.markdown("""
        <div style='background-color: #fff3cd; padding: 30px; border-radius: 10px; border-left: 6px solid #ffecb5;'>
            <h1 style='color: #795548;'>🌾 Crop Production Explorer</h1>
            <p style='font-size: 18px; color: #333;'>
                Welcome to the <b>Crop Production Explorer</b> — a comprehensive interactive tool for analyzing and predicting crop production trends using global data.
            </p>
            <ul style='font-size: 17px; color: #333;'>
                <li><b>Top Producers:</b> Visualize which countries lead in crop production.</li>
                <li><b>Trend Analysis:</b> Explore production patterns over the years across regions.</li>
                <li><b>Geographical Analysis:</b> Examine crop output distribution on the world map.</li>
                <li><b>Production Predictor:</b> Estimate future production based on area and yield inputs.</li>
            </ul>
            <p style='font-size: 18px; color: #333;'>
                🚀 Let's dive into data-driven agriculture and uncover insights that matter!
            </p>
        </div>
    """, unsafe_allow_html=True)




# --- Page: Top Producers ---
if page == "Top producers analysis":
    mycursor.execute("SELECT DISTINCT Area FROM data WHERE Item = %s", (selected_crop,))
    areas = sorted([row[0] for row in mycursor.fetchall()])
    selected_area = st.sidebar.selectbox("Choose a Region (Area)", areas)

    mycursor.execute("SELECT DISTINCT Year FROM data WHERE Item = %s AND Area = %s", (selected_crop, selected_area))
    years = sorted([int(row[0]) for row in mycursor.fetchall()])
    selected_year = st.sidebar.selectbox("Choose a Year", years)

    top_n = st.sidebar.slider("Top N Producers", min_value=1, max_value=20, value=10)

    query_all = """
        SELECT Area, Item, Year, `Production(tonnes)`
        FROM data
        WHERE Item = %s AND Year = %s
    """
    mycursor.execute(query_all, (selected_crop, selected_year))
    df_all = pd.DataFrame(mycursor.fetchall(), columns=['Area', 'Item', 'Year', 'Production(tonnes)'])
    df_all['Production(tonnes)'] = pd.to_numeric(df_all['Production(tonnes)'], errors='coerce').dropna()

    query_specific = """
        SELECT Area, Item, Year, `Production(tonnes)`
        FROM data
        WHERE Item = %s AND Area = %s AND Year = %s
    """
    mycursor.execute(query_specific, (selected_crop, selected_area, selected_year))
    df_specific = pd.DataFrame(mycursor.fetchall(), columns=['Area', 'Item', 'Year', 'Production(tonnes)'])
    df_specific['Production(tonnes)'] = pd.to_numeric(df_specific['Production(tonnes)'], errors='coerce')

    # st.title("🌾 Top Producers Analysis")
    st.subheader(f" Production of {selected_crop} in {selected_area} ({selected_year})")
    st.dataframe(df_specific)

    top_df = df_all.groupby('Area')['Production(tonnes)'].sum().sort_values(ascending=False).head(top_n).reset_index()

    fig = px.bar(
        top_df,
        x='Production(tonnes)',
        y='Area',
        orientation='h',
        title=f"Top {top_n} {selected_crop} Producing Regions ({selected_year})",
        labels={'Production(tonnes)': 'Production (Tonnes)', 'Area': 'Region'},
        color='Production(tonnes)',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family='IBM Plex Sans',
            size=16,
            color='black'
        ),
        title_font=dict(
            size=25,
            color='black',
            family='IBM Plex Sans'
        )
    )
    fig.update_xaxes(title_font=dict(size=16, color='black', family='IBM Plex Sans'), tickfont=dict(size=14, color='black', family='IBM Plex Sans'))
    fig.update_yaxes(title_font=dict(size=16, color='black', family='IBM Plex Sans'), tickfont=dict(size=14, color='black', family='IBM Plex Sans'))
    st.plotly_chart(fig)

# --- Page: Trend Analysis ---
elif page == "Trend analysis":
    # st.title("📈 Yearly Production Trend")
    mycursor.execute("SELECT DISTINCT Area FROM data WHERE Item = %s", (selected_crop,))
    all_areas = sorted([row[0] for row in mycursor.fetchall()])
    selected_areas = st.multiselect("Select Regions", all_areas[:30], default=all_areas[:3])

    mycursor.execute("SELECT DISTINCT Year FROM data WHERE Item = %s", (selected_crop,))
    all_years = sorted([int(row[0]) for row in mycursor.fetchall()])
    year_range = st.slider("Select Year Range", min_value=min(all_years), max_value=max(all_years), value=(min(all_years), max(all_years)))

    if selected_areas:
        format_strings = ','.join(['%s'] * len(selected_areas))
        query = f"""
            SELECT Area, Year, `Production(tonnes)`
            FROM data
            WHERE Item = %s AND Area IN ({format_strings}) AND Year BETWEEN %s AND %s
        """
        mycursor.execute(query, tuple([selected_crop] + selected_areas + list(year_range)))
        df_trend = pd.DataFrame(mycursor.fetchall(), columns=['Area', 'Year', 'Production(tonnes)'])
        df_trend['Year'] = pd.to_numeric(df_trend['Year'], errors='coerce')
        df_trend['Production(tonnes)'] = pd.to_numeric(df_trend['Production(tonnes)'], errors='coerce')

        fig_trend = px.line(
            df_trend,
            x='Year',
            y='Production(tonnes)',
            color='Area',
            markers=True,
            title=f"Yearly Production Trend of {selected_crop}",
            labels={'Production(tonnes)': 'Production (Tonnes)', 'Year': 'Year'}
        )
        fig_trend.update_layout(
            xaxis=dict(dtick=1),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='IBM Plex Sans', size=16, color='black'),
            title_font=dict(size=25, color='black', family='IBM Plex Sans')
        )
        fig_trend.update_xaxes(title_font=dict(size=16, color='black', family='IBM Plex Sans'), tickfont=dict(size=14, color='black', family='Arial'))
        fig_trend.update_yaxes(title_font=dict(size=16, color='black', family='IBM Plex Sans'), tickfont=dict(size=14, color='black', family='Arial'))
        st.plotly_chart(fig_trend)
        # st.dataframe(df_trend)

# --- Page: Production Map ---
elif page == "Geographical analysis":
    # st.title("Geographical Map")
    mycursor.execute("SELECT DISTINCT Year FROM data WHERE Item = %s", (selected_crop,))
    all_years = sorted([int(row[0]) for row in mycursor.fetchall()])
    selected_year = st.sidebar.selectbox("Select a Year", all_years)

    mycursor.execute("""
        SELECT Area, SUM(`Production(tonnes)`) AS Value
        FROM data
        WHERE Item = %s AND Year = %s
        GROUP BY Area
    """, (selected_crop, selected_year))
    df_map = pd.DataFrame(mycursor.fetchall(), columns=['Area', 'Value'])

    fig_map = px.choropleth(
        df_map,
        locations="Area",
        locationmode="country names",
        color="Value",
        hover_name="Area",
        color_continuous_scale="YlOrBr",
        title=f"{selected_crop} Production by Country in {selected_year}"
    )
    fig_map.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans', size=16, color='black'),
        title_font=dict(size=20, color='black', family='IBM Plex Sans')
    )
    fig_map.update_geos(
        showcountries=True, countrycolor="black"
    )
    st.plotly_chart(fig_map)

# --- Page: Predict Production ---
elif page == "Production predictor":
    # st.title("🌾 Crop Production Predictor")

    df = pd.read_sql("SELECT * FROM data", con=mydb)
    df = df[(df['Area_harvested(ha)'] > 0) & (df['Yield(kg/ha)'] > 0) & (df['Production(tonnes)'] > 0)]

    df['Log_Area'] = np.log1p(df['Area_harvested(ha)'])
    df['Log_Yield'] = np.log1p(df['Yield(kg/ha)'])
    df['Log_Production'] = np.log1p(df['Production(tonnes)'])
    df_encoded = pd.get_dummies(df, columns=['Area', 'Item'], drop_first=True)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(df_encoded[['Log_Area', 'Year', 'Log_Yield']])
    dummy_columns = df_encoded.drop(columns=[
        'Area_harvested(ha)', 'Yield(kg/ha)', 'Production(tonnes)',
        'Log_Area', 'Log_Yield', 'Log_Production', 'Year'
    ]).columns.tolist()
    X_final = np.hstack((X_poly, df_encoded[dummy_columns].values))
    y = df_encoded['Log_Production']

    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LinearRegression().fit(X_train_scaled, y_train)

    st.subheader("📅 Enter Input Values")
    area_name = st.selectbox("Select Area", sorted(df['Area'].unique()))
    filtered_items = df[df['Area'] == area_name]['Item'].unique()
    item_name = st.selectbox("Select Crop", sorted(filtered_items))
    year = st.number_input("Enter Year (e.g., 2025)", min_value=1900, max_value=2100, value=2025)
    area_ha = st.number_input("Area Harvested (ha)", min_value=1.0, value=10000.0)
    yield_kg_ha = st.number_input("Yield (kg/ha)", min_value=1.0, value=2000.0)

    if st.button("Predict Production"):
        log_area = np.log1p(area_ha)
        log_yield = np.log1p(yield_kg_ha)
        input_df = pd.DataFrame([[log_area, year, log_yield]], columns=['Log_Area', 'Year', 'Log_Yield'])
        X_poly_input = poly.transform(input_df)

        dummy_row = pd.DataFrame([[0]*len(dummy_columns)], columns=dummy_columns)
        if f'Area_{area_name}' in dummy_row.columns:
            dummy_row[f'Area_{area_name}'] = 1
        if f'Item_{item_name}' in dummy_row.columns:
            dummy_row[f'Item_{item_name}'] = 1

        X_input = np.hstack((X_poly_input, dummy_row.values))
        X_input_scaled = scaler.transform(X_input)
        log_pred = model.predict(X_input_scaled)
        pred = np.expm1(log_pred[0])
        # st.success(f"🌾 Predicted Production: **{pred:.2f} tonnes**")
        st.markdown(f"""
        <div style='padding:20px; background-color:#ffffffaa; border-radius:10px; text-align:center;'>
        <h2 style='color:black; font-size:28px;'>🌾 Predicted Production: <b>{pred:.2f} tonnes</b></h2>
         </div>
        """, unsafe_allow_html=True)


elif page == "End page":


    st.markdown("""
        <div style='background-color: #fff3cd; padding: 30px; border-radius: 10px; border-left: 6px solid #ffecb5;'>
            <h1 style='color: #795548;'>
            <p style='font-size: 18px; color: #333;'>
                You've reached the end of your journey through the Crop Production Explorer.We hope this platform helped you gain meaningful insights into the agricultural world.
            </p>
            <ul style='font-size: 17px; color: #333;'>
                  Built With ❤️ Using:
                </p>
                           1) Streamlit for an interactive and user-friendly frontend
                </p>
                2) Pandas & MySQL for efficient data management and retrieval
                </p>
                3) Scikit-learn for building and deploying machine learning models
            </ul>
            <p style='font-size: 18px; color: #333;'>
                 🌱 Grow with Knowledge. Harvest with Insight.
            </p> 
            <p style='font-size: 25px; color: #607c3c; text-align: center; font-weight: bold;'>  
             Thank you for visiting!
                
            
        </div>
    """, unsafe_allow_html=True)
