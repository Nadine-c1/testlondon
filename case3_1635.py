import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


###########################################################################################################################

@st.cache_data
def load_weather_data():
    df_weer = pd.read_csv('weather_london.csv') 
    df_weer.rename(columns={'Unnamed: 0': 'Datum'}, inplace=True)
    df_weer['Datum'] = pd.to_datetime(df_weer['Datum'])
    df_weer.drop(columns=['tsun'])
    return df_weer

###########################################################################################################################

@st.cache_data
def load_metro_data():
    metro = pd.read_csv('AnnualisedEntryExit_2019.csv', sep=';')
    stations = gpd.read_file("London stations.json")
    metro_stations = metro.merge(stations, how='left', left_on='Station', right_on='name')
    metro_stations = metro_stations.dropna(subset=['geometry'])
    metro_stations['Annualised_en/ex'] = metro_stations['Annualised_en/ex'].str.replace('.', '').astype('float')
    return gpd.GeoDataFrame(metro_stations, geometry="geometry")

###########################################################################################################################

@st.cache_data
def load_fiets_data():
    F4_dec = pd.read_csv('Koud redelijk droog 191JourneyDataExtract04Dec2019-10Dec2019.csv')
    F4_dec['week_soort'] = 'Koud & droog 4-12 10-12'
    F4_dec['regenval'] = 'Droog'
    F6_mrt = pd.read_csv('Nat gemiddeld 152JourneyDataExtract06Mar2019-12Mar2019.csv')
    F6_mrt['week_soort'] = "Gemiddeld & nat 6-3 12-3"
    F6_mrt['regenval'] = 'Nat'
    F30_okt = pd.read_csv('Nat gemiddeld 186JourneyDataExtract30Oct2019-05Nov2019.csv')
    F30_okt['week_soort'] = 'Gemiddeld & nat 30-10 5-11'
    F30_okt['regenval'] = 'Nat'
    F2_jan = pd.read_csv('Koud droog 143JourneyDataExtract02Jan2019-08Jan2019.csv')
    F2_jan['week_soort'] = 'Koud & droog 2-1 8-1'
    F2_jan['regenval'] = 'Droog'
    F26_jun = pd.read_csv('Warm droog 168JourneyDataExtract26Jun2019-02Jul2019.csv')
    F26_jun['week_soort'] = 'Warm & droog 26-6 2-7'
    F26_jun['regenval'] = 'Droog'
    F21_aug = pd.read_csv('Warm droog 176JourneyDataExtract21Aug2019-27Aug2019.csv')
    F21_aug['week_soort'] = 'Warm & droog 21-8 27-8'
    F21_aug['regenval'] = 'Droog'
    F11_dec = pd.read_csv('Nat koud 192JourneyDataExtract11Dec2019-17Dec2019.csv')
    F11_dec['week_soort'] = 'Koud & nat 11-12 17-12'
    F11_dec['regenval'] = 'Nat'
    F30_jan = pd.read_csv('Nat koud 147JourneyDataExtract30Jan2019-05Feb2019.csv')
    F30_jan['week_soort'] = 'Koud & nat 30-1 5-2'
    F30_jan['regenval'] = 'Nat'
    F08_aug = pd.read_csv('Nat warm 174JourneyDataExtract07Aug2019-13Aug2019.csv')
    F08_aug['week_soort'] = 'Warm & nat 08-8 13-8'
    F08_aug['regenval'] = 'Nat'
    F12_jun = pd.read_csv('Nat warm 166JourneyDataExtract12Jun2019-18Jun2019.csv')
    F12_jun['week_soort'] = 'Warm & nat 12-6 18-6'
    F12_jun['regenval'] = 'Nat'
    F20_mrt = pd.read_csv('Droog gemiddeld 154JourneyDataExtract20Mar2019-26Mar2019.csv')
    F20_mrt['week_soort'] = "Gemiddeld & droog 20-3 26-3"
    F20_mrt['regenval'] = 'Droog'
    F27_feb = pd.read_csv('Droog gemiddeld 151JourneyDataExtract27Feb2019-05Mar2019.csv')
    F27_feb['week_soort'] = 'Gemiddeld & droog 27-2 5-3'
    F27_feb['regenval'] = 'Droog'

    fiets_df = pd.concat([F4_dec, F6_mrt, F30_okt, F2_jan, F26_jun, F21_aug,F11_dec,F30_jan,F08_aug,F12_jun,F20_mrt,F27_feb])
    fiets_df.drop(columns=['Rental Id', 'Bike Id'], inplace=True)
    fiets_df['End Date']=pd.to_datetime(fiets_df['End Date'], format='%d/%m/%Y %H:%M')
    fiets_df['End'] = fiets_df['End Date'].dt.date
    fiets_df['Start Date']=pd.to_datetime(fiets_df['Start Date'], format='%d/%m/%Y %H:%M')
    fiets_df['Start'] = fiets_df['Start Date'].dt.date
 
    parkeer = pd.read_csv('cycle_stations.csv')
    parkeer['id'] = parkeer['id'].replace({865: 32, 857: 174, 867: 512, 868: 270, 870: 347, 864: 120, 869: 100, 872: 339, 852: 639, 866: 777})
 
    fiets_start = fiets_df.groupby(['Start', 'StartStation Name', 'StartStation Id','week_soort','regenval']).size().reset_index(name="Vertrek_Count")
    fiets_eind = fiets_df.groupby(['End', 'EndStation Name', 'EndStation Id','week_soort', 'regenval']).size().reset_index(name="Aankomst_Count")
    fiets_count = fiets_start.merge(fiets_eind, left_on=['Start', 'StartStation Name'], right_on=['End', 'EndStation Name'], how='outer').fillna(0)
    fiets_count.rename(columns={'Start': 'datum', 'StartStation Name': 'Station', 'StartStation Id': 'Id','week_soort_x':'Week_soort', 'regenval_x':'regenval'}, inplace=True)
    fiets_count['datum'] = pd.to_datetime(fiets_count['datum'])
    fiets_count['total_users'] = fiets_count['Vertrek_Count'] + fiets_count['Aankomst_Count']
    fiets_count['dag_van_de_week'] = pd.to_datetime(fiets_count['datum']).dt.dayofweek
    weekelijkse_gebruikers = fiets_count.groupby(['Station', 'Week_soort'])['total_users'].sum().reset_index()
    weekelijkse_gebruikers.rename(columns={'total_users': 'weekly_total_users'}, inplace=True)
    fiets_count = fiets_count.merge(weekelijkse_gebruikers, on=['Station', 'Week_soort'], how='left')
    fiets_count = fiets_count.merge(parkeer[['id', 'lat', 'long']], how='left', left_on='Id', right_on='id')
    fiets_count = fiets_count.dropna(subset=['lat'])
    fiets_count["geometry"] = fiets_count.apply(lambda row: Point(row["long"], row["lat"]), axis=1)
    return gpd.GeoDataFrame(fiets_count, geometry="geometry")


@st.cache_data
def load_ritten_per_dag():
    df = pd.read_csv('ritten_per_dag.csv') 
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    return df

@st.cache_data
def load_ritten_per_dag_janfeb_2020():
    df = pd.read_csv('ritten_per_dag_jan_feb_2020.csv') 
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    return df
###########################################################################################################################

# Data laden
weather = load_weather_data()
metro_stations_geo = load_metro_data()
fiets_count_geo = load_fiets_data()
ritten_per_dag_fiets = load_ritten_per_dag()
ritten_per_dag_fiets_janfeb_2020 = load_ritten_per_dag_janfeb_2020()

jaren = sorted(weather['Datum'].dt.year.unique())

###########################################################################################################################

# Sidebar voor paginaselectie
pagina = st.sidebar.radio(
    "Selecteer een pagina:",
    ('Data verkenning', 'Analyse fietstochten', 'Voorspellend model', 'Conclusie')
)

###########################################################################################################################

# Pagina: Weerdata
if pagina == 'Data verkenning':
    
    # Streamlit App
    st.title("🚲 De correlatie tussen het aantal fietsritten en de weerfactoren op een dag")

    st.title('Weer in London')
    jaar = st.selectbox('Selecteer het jaar', jaren, index=19)
    df_jaar = weather[weather['Datum'].dt.year == jaar]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=('Temperatuur (°C) & Windsnelheid (km/h)', 'Neerslag (mm)'),
                        row_heights=[0.7, 0.3],
                        specs=[[{"secondary_y": True}], [{}]])

    fig.add_trace(go.Scatter(x=df_jaar['Datum'], y=df_jaar['tavg'], mode='lines', name='Gem. temperatuur', line=dict(color='goldenrod', width=1)), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df_jaar['Datum'], y=df_jaar['tmax'], mode='lines', name='Max. temperatuur', line=dict(color='firebrick', width=1)), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df_jaar['Datum'], y=df_jaar['tmin'], mode='lines', name='Min. temperatuur', line=dict(color='gainsboro', width=1)), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df_jaar['Datum'], y=df_jaar['wspd'], mode='lines', name='Windsnelheid', line=dict(color='sandybrown', width=1), visible='legendonly'),
                  row=1, col=1, secondary_y=True)

    fig.add_trace(go.Bar(x=df_jaar['Datum'], y=df_jaar['prcp'], name='Regenval', marker_color='royalblue'), row=2, col=1)
    fig.add_trace(go.Bar(x=df_jaar['Datum'], y=df_jaar['snow'], name='Sneeuwval', marker_color='white'), row=2, col=1)

    fig.update_layout(template='plotly_dark', height=600, width=1500, barmode='overlay', bargap=0.1, showlegend=True)
    fig.update_xaxes(title_text="Datum", tickmode='linear', dtick='M1', row=2, col=1)
    fig.update_yaxes(title_text="Temperatuur (°C)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Windsnelheid (km/h)", row=1, col=1, secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

#############################################################################

    # Ensure Start Date is in datetime format
    ritten_per_dag_fiets["Start Date"] = pd.to_datetime(ritten_per_dag_fiets["Start Date"])
    
    # Merge bike trip counts per day with weather data
    df_merged = weather.merge(ritten_per_dag_fiets, left_on="Datum", right_on="Start Date", how="inner")
    
    # Drop non-numeric columns
    df_corr = df_merged.drop(columns=["Datum", "Start Date"])  # Remove date columns
    
    # Compute correlation matrix
    correlatie_matrix = df_corr.corr()
    
    # Sorteer op absolute correlatie met 'Aantal Ritten'
    correlatie_bike = correlatie_matrix[['Aantal Ritten']].dropna()
    correlatie_bike['abs_corr'] = correlatie_bike['Aantal Ritten'].abs()  # Voeg absolute waarden toe
    
    # Classificeer de correlatie volgens de tabel
    def classificatie(r):
        if abs(r) > 0.5:
            return "Sterk"
        elif abs(r) > 0.3:
            return "Matig"
        elif abs(r) > 0:
            return "Zwak"
        else:
            return "Geen"
    
    # Voeg de classificatie toe
    correlatie_bike['Sterkte'] = correlatie_bike['Aantal Ritten'].apply(classificatie)
    
    # Sorteer en verwijder helper kolom
    correlatie_bike = correlatie_bike.sort_values(by='abs_corr', ascending=False).drop(columns=['abs_corr'])
    
    # Maak aangepaste annotaties met zowel de correlatie als de classificatie
    annotaties = correlatie_bike.apply(lambda row: f"{row['Aantal Ritten']:.2f}\n({row['Sterkte']})", axis=1)

    
    st.subheader("📊 Correlatie Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        correlatie_bike[['Aantal Ritten']], 
        annot=annotaties.values.reshape(-1,1),  # Gebruik aangepaste annotaties
        cmap="coolwarm", fmt="", linewidths=0.5, center=0, ax=ax
    )
    ax.set_title("Correlatie tussen aantal fietsritten en weerdata")
    
    # Display plot in Streamlit
    st.pyplot(fig)

###############################################################################

    # 📌 Create subplots with shared y-axis
    fig = make_subplots(rows=1, cols=3, subplot_titles=["🔴 tmax", "🟢 tavg", "🔵 tmin"], shared_yaxes=True)
    
    # Add boxplots with custom colors
    fig.add_trace(go.Box(y=weather["tmax"], name="tmax", marker_color="red"), row=1, col=1)
    fig.add_trace(go.Box(y=weather["tavg"], name="tavg", marker_color="green"), row=1, col=2)
    fig.add_trace(go.Box(y=weather["tmin"], name="tmin", marker_color="blue"), row=1, col=3)
    
    # Update layout
    fig.update_layout(
        title_text="📊 Temperature Distributions (tmax, tavg, tmin)",
        showlegend=False,
        height=500,
        width=900,
        yaxis_title="Temperature (°C)"  # Common y-axis label
    )
    
    # Display the boxplots in Streamlit
    st.subheader("📊 Boxplots of Temperature Variables")
    st.plotly_chart(fig)

#######################################################################################################################################


# Pagina: Extra Analyses

elif pagina == 'Analyse fietstochten':
    st.title('Analyse fietstochten')
    st.subheader('Heatmap fietstochten in warme, koude, natte, droge weken')

    # Voorbereiden dataframes
    fiets_plot = fiets_count_geo.copy()
    fiets_plot['lat'] = fiets_plot.geometry.y
    fiets_plot['lon'] = fiets_plot.geometry.x

    metro_plot = metro_stations_geo.copy()
    metro_plot['lat'] = metro_plot.geometry.y
    metro_plot['lon'] = metro_plot.geometry.x

    # Locaties voor dropdown
    map_center_dropdown = {
        'Londen Centrum 🏙️': {"lat": 51.5074, "lon": -0.1278, "zoom": 11},
        'The British Museum 🏛️': {"lat": 51.5194, "lon": -0.1270, "zoom": 14},
        'The Tower of London 🏰': {"lat": 51.5081, "lon": -0.0759, "zoom": 14},
        'Buckingham Palace 👑': {"lat": 51.5014, "lon": -0.1419, "zoom": 14},
        'The Houses of Parliament & Big Ben 🏛️': {"lat": 51.5007, "lon": -0.1246, "zoom": 14},
        'The London Eye 🎡': {"lat": 51.5033, "lon": -0.1195, "zoom": 14}
    }

    location = st.selectbox('Kies een locatie om in te zoomen:', options=list(map_center_dropdown.keys()), index=0)
    selected_location = map_center_dropdown[location]

    gekozen_week = st.selectbox('Kies een week:', sorted(fiets_plot['Week_soort'].unique()))
    filtered_fiets_plot = fiets_plot[fiets_plot['Week_soort'] == gekozen_week]

    # Alleen minimale slider
    min_fietsritten = st.slider('Minimaal aantal fietsritten', 0, int(filtered_fiets_plot['weekly_total_users'].max()), 0)
    filtered_fiets_plot = filtered_fiets_plot[filtered_fiets_plot['weekly_total_users'] >= min_fietsritten]

    # Maak interactieve kaart zonder kleurbar
    fig = px.scatter_mapbox(
        filtered_fiets_plot,
        lat="lat", lon="lon",
        color_discrete_sequence=["green"],
        size='weekly_total_users',
        hover_name="Station",
        size_max=25,
        labels={"weekly_total_users": "Aantal fietsritten"}
    )

    fig.update_traces(name="Fietsritten", showlegend=True)

    # Heatmap laag toevoegen zonder colourbar
    fig.add_trace(go.Densitymapbox(
        lat=filtered_fiets_plot['lat'],
        lon=filtered_fiets_plot['lon'],
        z=filtered_fiets_plot['weekly_total_users'],
        radius=15,
        colorscale="Viridis",
        opacity=0.5,
        showscale=False,
        name='Heatmap fietsritten'
    ))

    # Metrodata altijd toevoegen
    metro_trace = px.scatter_mapbox(
        metro_plot,
        lat="lat", lon="lon",
        color_discrete_sequence=["purple"],
        size='Annualised_en/ex',
        hover_name="name",
        size_max=25,
        labels={'Annualised_en/ex':'Metroritten'}
    ).data[0]

    metro_trace.name = "Metroritten"
    metro_trace.showlegend = True
    fig.add_trace(metro_trace)

    # Bezienswaardigheden altijd toevoegen
    markers_data = [
        {"name": "The British Museum 🏛️", "lat": 51.5194, "lon": -0.1270},
        {"name": "The Tower of London 🏰", "lat": 51.5081, "lon": -0.0759},
        {"name": "Buckingham Palace 👑", "lat": 51.5014, "lon": -0.1419},
        {"name": "The Houses of Parliament 🏛️", "lat": 51.5007, "lon": -0.1246},
        {"name": "The London Eye 🎡", "lat": 51.5033, "lon": -0.1195}
    ]

    fig.add_trace(go.Scattermapbox(
        lat=[m['lat'] for m in markers_data],
        lon=[m['lon'] for m in markers_data],
        mode='markers+text',
        text=[m['name'] for m in markers_data],
        marker=dict(size=14, color='red'),
        textposition='top right',
        name='Bezienswaardigheden'
    ))

    center_lat = np.mean([m['lat'] for m in markers_data])
    center_lon = np.mean([m['lon'] for m in markers_data])+0.009
    max_radius = max(np.sqrt((np.array([m['lat'] for m in markers_data]) - center_lat)**2 +
                             (np.array([m['lon'] for m in markers_data]) - center_lon)**2))

    theta = np.linspace(0, 2*np.pi, 100)
    circle_lat = center_lat + max_radius * np.cos(theta)
    circle_lon = center_lon + max_radius * np.sin(theta)

    fig.add_trace(go.Scattermapbox(
        lat=circle_lat,
        lon=circle_lon,
        mode='lines',
        line=dict(color='orange', width=2),
        name='Cirkel rondom bezienswaardigheden'
    ))

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        height=400,
        mapbox=dict(center={"lat": selected_location["lat"], "lon": selected_location["lon"]}, zoom=selected_location["zoom"]),
        legend=dict(y=0.99, x=0.01, bgcolor="rgba(0,0,0,0.6)", font=dict(color="white")),
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    st.plotly_chart(fig, use_container_width=True)


###########################################################################################################################

# barplot
    # Zorg voor weekend en doordeweeks
    fiets_count_geo['is_weekend'] = fiets_count_geo['dag_van_de_week'].isin([5, 6])  # 5=Saturday, 6=Sunday
 
    # Groeperen per week, is_weekend en het type dag
    week_data = fiets_count_geo.groupby(['Week_soort', 'is_weekend']).agg({'total_users': 'sum'}).reset_index()
 
    # Voor barchart willen we een kolom per week per type dag (doordeweeks vs weekend)
    week_data_pivot = week_data.pivot_table(index='Week_soort', columns='is_weekend', values='total_users', aggfunc='sum')
 
    # Gemiddelde per dag berekenen voor week- en weekenddagen
    week_data_pivot['Gemiddelde doordeweekse dag'] = week_data_pivot[False] / 5
    week_data_pivot['Gemiddelde weekenddag'] = week_data_pivot[True] / 2

    # Maak barplot met correcte gemiddelden
    fig = px.bar(
        week_data_pivot,
        x=week_data_pivot.index,
        y=['Gemiddelde doordeweekse dag', 'Gemiddelde weekenddag'],
        labels={'value': 'Gemiddeld aantal gebruikers per dag', 'Week_soort': 'Week_soort', 'variable': 'Type dag'},
        title='Gemiddeld aantal ritten per dag per week (doordeweeks vs weekend)', 
        barmode='group',
        height=400
    )
    # Toon de plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Pagina: Overzicht
elif pagina == 'Voorspellend model':

    # Load Data
    jaar_data = pd.read_csv('ritten_per_dag.csv')
    jaar_data['Start Date'] = pd.to_datetime(jaar_data['Start Date'])

    df_weather = pd.read_csv('weather_london.csv')
    df_weather = df_weather.rename(columns={'Unnamed: 0': 'Date'})
    df_weather['Date'] = pd.to_datetime(df_weather['Date'], errors='coerce')

    # Filter only for the year 2019
    weather_2019 = df_weather[df_weather['Date'].dt.year == 2019]

    # Count unique tmax values and their occurrences in 2019
    tmax_counts_2019 = weather_2019['tmax'].value_counts().reset_index()
    tmax_counts_2019.columns = ['tmax', 'count']
    tmax_counts_2019 = tmax_counts_2019.sort_values(by='tmax', ascending=True)

    # Merge the weather and bike data on the date column
    merged_df = pd.merge(jaar_data, df_weather, left_on='Start Date', right_on='Date', how='inner')

    # Group by 'tmax' and calculate statistics
    summary_table = merged_df.groupby('tmax')['Aantal Ritten'].agg(
        mean='mean',
        median='median',
        mode=lambda x: stats.mode(x, keepdims=True)[0][0]  # Get the first mode
    ).reset_index()

    # Merge counts
    new_df = pd.merge(summary_table, tmax_counts_2019)

    # Kleinste kwadraten lineaire regressie
    coeff = np.polyfit(new_df['tmax'], new_df['mean'], 1)
    poly1d_fn = np.poly1d(coeff)

    # Formule van de regressielijn
    regressie_formule = f'y = {coeff[0]:.2f}x + {coeff[1]:.2f}'

    # Streamlit App Title
    st.title("Aantal fietsritten per dag t.o.v. maximale temperatuur (˚C)")

    # Create interactive scatter plot using Plotly
    fig = px.scatter(
        new_df,  
        x="tmax",
        y="mean",
        color="tmax",  
        color_continuous_scale="RdBu_R",
        size="count",
        hover_data={"tmax": True, "mean": True, "count": True},
        labels={"tmax": "Maximale temperatuur(°C)", "mean": "Gemiddeld aantal fietsritten", "count": "Dagen met temperatuur"},
        title="Aantal fietsritten per dag t.o.v. maximale temperatuur ˚C"
    )

    fig.add_trace(
        go.Scatter(
            x=new_df['tmax'],
            y=poly1d_fn(new_df['tmax']),
            mode='lines',
            name=f'Lineaire regressie: {regressie_formule}',
            line=dict(color='gold', width=2)
        )
    )

    fig.update_traces(marker=dict(line=dict(width=1, color="black")))  
    fig.update_layout(coloraxis_colorbar=dict(title="Temperatuur (°C)"))
    st.plotly_chart(fig)

    # **Train Model Using `tmax` Instead of `tavg`**
    X_train = new_df[['tmax']]  
    y_train = new_df['mean']  

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Select weather data for Jan & Feb 2020
    weer_2020 = df_weather[(df_weather['Date'].dt.year == 2020) & (df_weather['Date'].dt.month.isin([1, 2]))]

    # Compute daily max temperature in Jan & Feb 2020
    weer_2020_avg = weer_2020.groupby('Date')['tmax'].max().reset_index()

    # Predict bike rides based on max temperature
    weer_2020_avg['Voorspelde Fietsritten'] = model.predict(weer_2020_avg[['tmax']])

    # Display Prediction Results
    st.subheader("📊 Voorspelde Fietsritten per Dag in Jan & Feb 2020")
    st.write("Deze voorspelling is gebaseerd op de lineaire relatie tussen maximale temperatuur en fietsritten.")

    # **Subplots: Maximum Temperature & Predicted Bike Rides**
    fig_pred = make_subplots(
        rows=2, cols=1,  
        shared_xaxes=True,  
        vertical_spacing=0.2,  
        subplot_titles=("Maximale Temperatuur", "Voorspelde Fietsritten")
    )

    # Add bars for predicted bike rides (lower plot)
    fig_pred.add_trace(go.Bar(
        x=weer_2020_avg['Date'], 
        y=weer_2020_avg['Voorspelde Fietsritten'], 
        name="Voorspelde Fietsritten",
        marker=dict(color="blue", opacity=0.7)  
    ), row=2, col=1)

    # Add max temperature line (upper plot)
    fig_pred.add_trace(go.Scatter(
        x=weer_2020_avg['Date'], 
        y=weer_2020_avg['tmax'], 
        name="Maximale Temperatuur (°C)",
        mode="lines",
        line=dict(color="red", dash="dash")  
    ), row=1, col=1)

    # Update layout for larger plots
    fig_pred.update_layout(
        title_text="Voorspelling van Fietsritten en Maximale Temperatuur in Jan & Feb 2020",
        xaxis=dict(title="Datum"),
        yaxis=dict(title="Maximale Temperatuur (°C)"),
        yaxis2=dict(title="Voorspelde Fietsritten"),
        showlegend=True,
        height=800,  
        width=1200,  
    )

    # Show plots
    st.plotly_chart(fig_pred)


    #########################################################################################################################
    #Daadwerkelijke data van fietsritten 2020 jan/feb in grafiek

    # Subset weer maken

    df_weather_20_janfeb = weather[(weather['Datum'].dt.year == 2020) & (weather['Datum'].dt.month <= 2)]

    fig = go.Figure()

    # Plot regenval
    fig.add_trace(go.Scatter(
        x=df_weather_20_janfeb['Datum'],
        y=df_weather_20_janfeb['prcp'],
        mode='lines',
        name='regenval in mm',
        line=dict(color='royalblue')
    ))

    # Plot gem. temperatuur
    fig.add_trace(go.Scatter(
        x=df_weather_20_janfeb['Datum'],
        y=df_weather_20_janfeb['tavg'],
        mode='lines',
        name='gem. temperatuur',
        line=dict(color='goldenrod'),
        visible='legendonly'  # Standaard verborgen
    ))

    # Plot max. temperatuur
    fig.add_trace(go.Scatter(
        x=df_weather_20_janfeb['Datum'],
        y=df_weather_20_janfeb['tmax'],
        mode='lines',
        name='max. temperatuur',
        line=dict(color='firebrick')
    ))

    # Plot min. temperatuur
    fig.add_trace(go.Scatter(
        x=df_weather_20_janfeb['Datum'],
        y=df_weather_20_janfeb['tmin'],
        mode='lines',
        name='min. temperatuur',
        line=dict(color='gainsboro'),
        visible='legendonly'  # Standaard verborgen
    ))

    # Plot fietsritten per dag als barplot op de tweede y-as
    fig.add_trace(go.Bar(
        x=ritten_per_dag_fiets_janfeb_2020['Start Date'],
        y=ritten_per_dag_fiets_janfeb_2020['Aantal Ritten'],
        name='fietsritten per dag',
        marker=dict(color='pink', opacity=0.15),
        yaxis='y2'  # Koppel deze bars aan de tweede y-as
    ))

    # Layout met tweede y-as
    fig.update_layout(
        title='Daadwerkelijke Fietsritten Jan/Feb 2020',
        xaxis_title='Datum',
        yaxis=dict(
            title='Temperatuur in graden Celsius',
            side='left'
        ),
        yaxis2=dict(
            title='Aantal fietsritten per dag',
            overlaying='y',
            side='right',
            showgrid=False  # Voorkomt overlappende gridlijnen
        ),
        barmode='overlay',  # Zorgt ervoor dat de bars niet de lijnen verbergen
        template='plotly_dark',
        showlegend=True
    )

    # x-as aanpassen
    fig.update_xaxes(
        tickmode='linear',
        dtick='M1'  # Elke maand een tick (bij datums)
    )

    st.plotly_chart(fig)

###########################################################################################################################
elif pagina == 'Conclusie':
    st.title('Conclusie')
    st.subheader('Belangrijkste bevindingen:')

    st.markdown("""
    - Het weer speelt een belangrijke rol in het fietsgebruik in Londen.
    - Hoe hoger de maximale temperatuur, hoe meer fietsritten worden gemaakt.
    - Regen heeft een negatieve invloed op het aantal fietsritten.
    - De gevonden relatie kan worden gebruikt om het aantal fietsritten te voorspellen aan de hand van weersvoorspellingen.
    """)
    st.markdown("Deze inzichten helpen stadsplanners en beleidsmakers bij het verbeteren van fietsinfrastructuur en het stimuleren van fietsgebruik.")