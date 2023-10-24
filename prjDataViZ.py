pip install matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
import bokeh as bk
import streamlit as st
import geopandas as gpd
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import dodge
from shapely.geometry import Point

def hist_distrib(df):
    prix_nom_counts = df['prix_nom'].value_counts()
    prix_noms = prix_nom_counts.index
    counts = prix_nom_counts.values

    plt.figure(figsize=(12, 6))
    plt.bar(prix_noms, counts)
    plt.xlabel('Prix Nom')
    plt.ylabel('Nombre de lignes')
    plt.title('Histogramme du nombre de lignes par valeur de Prix Nom')
    plt.xticks(rotation=45)
    plt.show()

def boxplot_distribution(df):
    sns.set(style="whitegrid")  
    plt.figure(figsize=(12, 6))  
    sns.boxplot(x="prix_nom", y="prix_valeur", data=df)
    plt.title("Repartition des prix par type d'essence")
    plt.xticks(rotation=45)
    plt.show()

def loc(df):
    long = df['longitude']
    lat = df['latitude']
    prix_nom = df['prix_nom']
    prix_valeur = df['prix_valeur']
    data = pd.DataFrame({'Longitude': long, 'Latitude': lat, 'Prix Nom': prix_nom, 'Prix Val': prix_valeur})

    alt.data_transformers.enable('default', max_rows=None)
    y_min, y_max = 41, 51.5  #approxtimativement les limites de la France 
    scatter_plot = alt.Chart(data).mark_circle(size=60).encode(
        x=alt.X('Latitude:Q', axis=alt.Axis(title='Latitude')),  
        y=alt.Y('Longitude:Q', axis=alt.Axis(title='Longitude'), scale=alt.Scale(domain=[y_min, y_max])),
        color='Prix Nom:N',
        tooltip=['Latitude:Q', 'Longitude:Q', 'Prix Nom:N', 'Prix Val:Q']
    ).properties(
        width=600,
        height=600,
        title='Points de vente de carburant répertoriés en France'
    )
    scatter_plot

def cartedep(df): 
    url = 'https://www.data.gouv.fr/fr/datasets/r/90b9341a-e1f7-4d75-a73c-bbc010c7feeb'
    departements = gpd.read_file(url)
    average_by_department = df.groupby('dep_code')['prix_valeur'].mean().reset_index() #moyenne par département, à lier avec la carte

    fig = px.choropleth(average_by_department,
                        geojson=departements,  
                        locations='dep_code',
                        featureidkey="properties.code",  
                        color='prix_valeur',
                        color_continuous_scale="Viridis",
                        title="Moyenne de prix_valeur par département en France")
    fig.update_geos(projection_type="mercator")
    fig.update_geos(
        projection_scale=20,  
        center={"lon": 2.0, "lat": 47.0}  #centrer sur la France
    )
    st.plotly_chart(fig)

def carte2(df):
    france_map = gpd.read_file("https://www.data.gouv.fr/fr/datasets/r/90b9341a-e1f7-4d75-a73c-bbc010c7feeb")
    geometry = [Point(xy) for xy in zip(df['latitude'], df['longitude'])] 
    point_gdf = gpd.GeoDataFrame(df, geometry=geometry)
    minx, miny, maxx, maxy = point_gdf.total_bounds #limites
    minx -= 0.5
    maxx += 0.5
    miny -= 0.5
    maxy += 0.5

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    france_map.boundary.plot(ax=ax, linewidth=1, color='blue', alpha=0.5) #on affiche la carte
    point_gdf.plot(ax=ax, color='red', markersize=20) #on ajoute nos points
    st.pyplot(plt)

def afficher_info_essence(code_postal, type_essence, df):
    st.write(f"Code Postal: {code_postal}")
    st.write(f"Type d'essence: {type_essence}")
    code_dep = code_postal[:2]

    # Fiiltrer le DataFrame par le code département qui nous intéresse
    df_filtered = df[(df['cp'].str[:2] == code_dep)]

    if not df_filtered.empty:
        #Calcul de la moyenne des prix dans le département grace au code postal de l'user
        moyenne_prix_departement = df[df['cp'].str[:2] == code_dep]['prix_valeur'].mean()
        st.write(f"Moyenne des prix de l'essence ({type_essence}) dans le département {code_dep}: {moyenne_prix_departement:.2f} €/L")

        #Sparer les 3 stations les moins chères
        top_3_endroits = df_filtered.nsmallest(3, 'prix_valeur')
        st.write("Les 3 endroits les moins chers :")
       

        for index, row in top_3_endroits.iterrows():
            rank = "première" if index == top_3_endroits.index[0] else "deuxième" if index == top_3_endroits.index[1] else "troisième"
            station_info = f"La {rank} station la moins chère est à {row['ville']} - {row['adresse']}, où le litre de {row['prix_nom']} est à {row['prix_valeur']:.2f} €"
            st.write(station_info)
        st.header("Carte des stations les moins chères de votre département")
        carte2(top_3_endroits)

        st.header("Comparaison des prix")
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        ax = sns.barplot(x='adresse', y='prix_valeur', data=top_3_endroits, palette="Blues")
        ax.axhline(moyenne_prix_departement, color='red', label="Moyenne du département")
        ax.set(xlabel="Adresse de la station", ylabel=f"Prix de l'essence ({type_essence}) en €/L")
        ax.set_title("Prix de l'essence par Station")
        ax.legend()
        plt.xticks(rotation=30, ha="right")
        st.pyplot(plt)
    else:
        st.write("Aucune donnée disponible pour ce code postal et ce type d'essence.")

def augmentation(df, code_postal):
    df['prix_maj'] = pd.to_datetime(df['prix_maj'], format='%Y-%m-%d %H:%M:%S')
    code_dep = code_postal[:2]
    df = df[df['cp'].str[:2] == code_dep]
    df['Mois'] = df['prix_maj'].dt.strftime('%Y-%m')
    df = df.groupby(['Mois', 'prix_nom'])['prix_valeur'].mean().reset_index()
    df = df.sort_values(by='Mois')#trier les valeurs par mois

    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    ax = sns.lineplot(x='Mois', y='prix_valeur', hue='prix_nom', data=df, ci=None)
    ax.set(xlabel="Mois", ylabel="Moyenne de prix_valeur", title="Évolution moyenne par mois du prix de l'essence selectionée ")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

def services(df):
    df['Nombre de Services'] = df['services_service'].apply(lambda x: len(x.split('//'))) #separer les services et compter
    df_grouped = df.groupby('Nombre de Services')['prix_valeur'].mean().reset_index() #moyenne par nombre de services

    chart = alt.Chart(df_grouped).mark_line().encode(
        x=alt.X('Nombre de Services:O', title='Nombre de Services'),
        y=alt.Y('prix_valeur:Q', title='Prix moyen'),
    ).properties(
        width=600,
        height=300,
        title='Prix moyen en fonction du nombre de services proposés'
    )
    chart

def autoroute(df, type_essence): 
    df_filtered = df[df['prix_nom'] == type_essence]
    
    if not df_filtered.empty:
        moyenne_prix_par_pop = df_filtered.groupby('pop')['prix_valeur'].mean().reset_index()
    
        st.write(f"Comparaison des prix de l'essence par type de route (A = Autoroute, R= Route)")
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        ax = sns.barplot(x='pop', y='prix_valeur', data=moyenne_prix_par_pop, palette="Blues")
        ax.set(xlabel="Type de route", ylabel="Prix de l'essence en €/L")
        ax.set_title("Prix par type de route")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(plt)


with st.sidebar:
    st.title('Pierre Taithe, M1 BIA')
    st.header('#datavz2023efrei')

st.sidebar.markdown("[Accueil](#top-section)")
st.sidebar.markdown("[Histogramme](#histogram-section)")
st.sidebar.markdown("[Boxplot](#boxplot-section)")
st.sidebar.markdown("[Carte des Points de Vente](#carte-section)")
st.sidebar.markdown("[Prix moyen par département](#cartedep-section)")
st.sidebar.markdown("[Une station proche de chez moi? ](#info-section)")
st.sidebar.markdown("[Évolution des valeurs en fonction de la date](#evolution-section)")
st.sidebar.markdown("[Prix moyen en fonction du nombre de services proposés](#services-section)")

st.markdown('<a name="top-section"></a>',  unsafe_allow_html=True)
st.title('Mon site de visualisation de Données')
st.header("Avec l'augmentation des prix du carburant, Comment rouler à petit prix en France ?")

st.subheader("Nos Données : ")
st.write("Données actualisées par le gouvernement français, disponibles à cette URL : [Data.gouv.fr](https://www.data.gouv.fr/fr/datasets/prix-des-carburants-en-france-flux-instantane/#/resources/64e02cff-9e53-4cb2-adfd-5fcc88b2dc09)")


url = 'https://www.data.gouv.fr/fr/datasets/r/64e02cff-9e53-4cb2-adfd-5fcc88b2dc09'
df = pd.read_csv(url, delimiter=';')
st.write("On utilise ces données, elles possedent de nombreuses colonnes utiles, mais nous allons nous concentrer particulier sur prix_val")

df = df.dropna(subset=['prix_valeur', 'dep_code'])
df['services_service'] = df['services_service'].fillna('non renseigné')
df = df.drop(['epci_name', 'epci_code'], axis=1)
df['horaires'] = df['horaires'].fillna('non renseigné')
df['cp'] = df['cp'].apply(lambda x: str(int(x)).rjust(5, '0'))
df['prix_maj'] = pd.to_datetime(df['prix_maj'], utc=True)

df = df.reset_index(drop=True)

df['geom'] = df['geom'].str.split(',')  
df[['longitude', 'latitude']] = pd.DataFrame(df['geom'].to_list(), columns=['longitude', 'latitude'])
df['longitude'] = df['longitude'].astype(float)
df['latitude'] = df['latitude'].astype(float)
df.drop(columns=['geom'], inplace=True)


df_sp98 = df[df['prix_nom'] == 'SP98']
df_e10 = df[df['prix_nom'] == 'E10']
df_e85 = df[df['prix_nom'] == 'E85']
df_gazole = df[df['prix_nom'] == 'Gazole']
df_sp95 = df[df['prix_nom'] == 'SP95']
df_gplc = df[df['prix_nom'] == 'GPLc']

categories = ['SP98', 'E10', 'E85', 'GPLc', 'Gazole', 'SP95']

st.write(df.head(80))
st.write("Ici notre Dataframe, nettoyé et prêt à être utilisé.")


st.markdown('<a name="histogram-section"></a>',  unsafe_allow_html=True)
st.title("Histogramme du nombre de lignes par type d'essence")
ah = hist_distrib(df)
st.pyplot(ah)
st.write("On remarque que le Gazole est le plus représenté, suivi du SP98 et du E10, ce qui est logique car ce sont les carburants les plus utilisés en France.")


st.markdown('<a name="boxplot-section"></a>',  unsafe_allow_html=True)
res = boxplot_distribution(df)
st.title("Diagramme en boîte des prix par litre par type d'essence")
st.write("Maintenant, on cherche à visualiser la repartition des prix par type d'essence, pour cela on utilise un boxplot.")
st.pyplot(res)
st.write("On remarque deux groupes, les carburants conventionels, qui tournent autour de 2euros/L, et les carburants alternatifs, qui tournent autour de 1euros/L.")


st.markdown('<a name="carte-section"></a>',  unsafe_allow_html=True)
st.title('Carte de France des points de vente de carburant')
res2 = loc(df)
st.write("On remarque que notre dataset est bien rempli, la france entiere est representée, avec une concentration de points dans les grandes villes.")

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('<a name="cartedep-section"></a>',  unsafe_allow_html=True)
st.title('Prix moyen de chaque essence par département')
selected_category = st.selectbox("Sélectionnez un type d'essence", categories)

if selected_category == 'SP98':
    cartedep(df_sp98)
elif selected_category == 'E10':
    cartedep(df_e10)
elif selected_category == 'E85':
    cartedep(df_e85)
elif selected_category == 'GPLc':
    cartedep(df_gplc)
elif selected_category == 'Gazole':
    cartedep(df_gazole)
elif selected_category == 'SP95':
    cartedep(df_sp95)


st.markdown('<a name="info-section"></a>',  unsafe_allow_html=True)
st.title('Trouvez les stations les moins chères près de chez vous')
code_postal = st.text_input("Entrez un code postal à 5 chiffres :", "93400")
type_essence = st.selectbox("Sélectionnez un type d'essence :", categories)

if code_postal and type_essence:
    if type_essence == 'SP98':
        afficher_info_essence(code_postal, 'SP98', df_sp98)
    elif type_essence == 'E10':
        afficher_info_essence(code_postal, 'E10', df_e10)
    elif type_essence == 'E85':
        afficher_info_essence(code_postal, 'E85', df_e85)
    elif type_essence == 'GPLc':
        afficher_info_essence(code_postal, 'GPLc', df_gplc)
    elif type_essence == 'Gazole':
        afficher_info_essence(code_postal, 'Gazole', df_gazole)
    elif type_essence == 'SP95':
        afficher_info_essence(code_postal, 'SP95', df_sp95)

st.markdown('<a name="evolution-section"></a>',  unsafe_allow_html=True)
st.title('Évolution de la moyenne du département en fonction de la date')
code_postal = st.text_input("Entrez un code postal à 5 chiffres :", key="code_postal_input", value= "93400")

if code_postal:
    augmentation(df, code_postal)

st.markdown('<a name="services-section"></a>',  unsafe_allow_html=True)
st.title('Mieux vaut-il aller dans une station avec beaucoup de services ?')
type_essence = st.selectbox("Sélectionnez un type d'essence :", categories, key="serviceType")
if  type_essence:
    if type_essence == 'SP98':
        services( df_sp98)
    elif type_essence == 'E10':
        services(df_e10)
    elif type_essence == 'E85':
        services(df_e85)
    elif type_essence == 'GPLc':
        services(df_gplc)
    elif type_essence == 'Gazole':
        services(df_gazole)
    elif type_essence == 'SP95':
        services(df_sp95)

st.header("Une baisse liée au type de route ?")
type_essence2 = st.selectbox("Sélectionnez un type d'essence :",categories, key="autorouteType")
if type_essence2:
    autoroute(df, type_essence2)
