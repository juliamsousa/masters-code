import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch, Sbopen
import networkx as nx
import seaborn as sns

@st.fragment
def load_matches(): 
    parser = Sbopen()
    df_competition = parser.competition()
    df_female_competitions = df_competition.loc[df_competition['competition_gender'] == "female"]
    df_matches = parser.match(competition_id = 72, season_id = 107)

    return df_matches

@st.fragment
def get_events(match_id, match_home_team):
    parser = Sbopen()
    df_events, related, freeze, tatics = parser.event(match_id)
    
    substitutions = df_events.loc[df_events["type_name"] == "Substitution"]
    home_team_subs = substitutions.loc[substitutions["team_name"] == match_home_team]

    ht_first_sub_index = home_team_subs.iloc[0]["index"]
    
    match_passes = df_events.loc[df_events["type_name"] == "Pass"]
    home_team_passes = match_passes.loc[match_passes["team_name"] == match_home_team]

    home_team_passes = home_team_passes.loc[match_passes.sub_type_name != "Throw-in"]
    
    ht_succesful_passes = home_team_passes.loc[home_team_passes["outcome_name"].isnull()]
    ht_first_sub_passes = ht_succesful_passes.loc[ht_succesful_passes["index"] < ht_first_sub_index]
    ht_first_sub_passes = ht_first_sub_passes[['x', 'y', 'end_x', 'end_y', "player_name", "pass_recipient_name"]]
    
    ht_first_sub_passes["player_name"] = ht_first_sub_passes["player_name"].apply(lambda x: str(x).split()[-1])
    ht_first_sub_passes["pass_recipient_name"] = ht_first_sub_passes["pass_recipient_name"].apply(lambda x: str(x).split()[-1])

    return ht_first_sub_passes

@st.fragment
def calculate_average_position(ht_first_sub_passes):
    df_scatter = pd.DataFrame()

    for i, name in enumerate(ht_first_sub_passes["player_name"].unique()):
        passer_x = ht_first_sub_passes.loc[ht_first_sub_passes["player_name"] == name]["x"].to_numpy()
        passer_y = ht_first_sub_passes.loc[ht_first_sub_passes["player_name"] == name]["y"].to_numpy()

        receiver_x = ht_first_sub_passes.loc[ht_first_sub_passes["pass_recipient_name"] == name]["end_x"].to_numpy()
        receiver_y = ht_first_sub_passes.loc[ht_first_sub_passes["pass_recipient_name"] == name]["end_y"].to_numpy()

        df_scatter.at[i, "player_name"] = name

        df_scatter.at[i, "x"] = np.mean(np.concatenate([passer_x, receiver_x]))
        df_scatter.at[i, "y"] = np.mean(np.concatenate([passer_y, receiver_y]))
        
        df_scatter.at[i, "no_passes"] = ht_first_sub_passes.loc[ht_first_sub_passes["player_name"] == name].count().iloc[0]
    
    df_scatter["marker_size"] = (df_scatter["no_passes"] / df_scatter["no_passes"].max() * 1500)

    return df_scatter

@st.fragment
def calculate_passes(ht_first_sub_passes):
    df_lines = ht_first_sub_passes.groupby(["player_name", "pass_recipient_name"]).x.count().reset_index()
    df_lines.rename({"x": "pass_count"}, axis="columns", inplace=True)
    df_lines = df_lines[df_lines["pass_count"]>0]
    
    return df_lines

@st.fragment
def plot_pitch(df_scatter, df_lines):
    pitch = Pitch(pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False, endnote_height=0.04, title_space=0, endnote_space=0)

    pitch.scatter(
        df_scatter.x, 
        df_scatter.y, 
        s=df_scatter.marker_size,
        color="red",
        edgecolors="grey",
        linewidth=1, 
        alpha=1, 
        ax=ax["pitch"], 
        zorder = 3
    )

    for i, row in df_scatter.iterrows():
        pitch.annotate(
            row.player_name, 
            xy=(row.x, row.y), 
            c='black', 
            va='center', 
            ha='center', 
            weight = "bold", 
            size=16, 
            ax=ax["pitch"], 
            zorder = 4
        )

    for i, row in df_lines.iterrows():
            player1 = row["player_name"]
            player2 = row["pass_recipient_name"]
        
            player1_x = df_scatter.loc[df_scatter["player_name"] == player1]['x'].iloc[0]
            player1_y = df_scatter.loc[df_scatter["player_name"] == player1]['y'].iloc[0]
            
            player2_x = df_scatter.loc[df_scatter["player_name"] == player2]['x'].iloc[0]
            player2_y = df_scatter.loc[df_scatter["player_name"] == player2]['y'].iloc[0]

            num_passes = row["pass_count"]
            line_width = (num_passes / df_lines['pass_count'].max() * 10)
        
            pitch.lines(
                player1_x, 
                player1_y, 
                player2_x, 
                player2_y,
                alpha=1, 
                lw=line_width, 
                zorder=2, 
                color="red", 
                ax = ax["pitch"]
            )

    # st.pyplot(plt.gcf())
    st.pyplot(fig)

@st.fragment
def calculate_team_centralization(ht_first_sub_passes):
    number_of_passes = ht_first_sub_passes.groupby(["player_name"]).x.count().reset_index()
    number_of_passes.rename({'x': 'pass_count'}, axis='columns', inplace=True)

    max_number_of_passes = number_of_passes["pass_count"].max()

    denominator = 10*number_of_passes["pass_count"].sum()
    nominator = (max_number_of_passes - number_of_passes["pass_count"]).sum()

    centralisation_index = nominator/denominator

    return centralisation_index

@st.fragment
def generate_isomorphic_graph(df_lines):
    pass_graph = df_lines.apply(tuple, axis=1).tolist()
    print(pass_graph)
    ISO_Graph = nx.DiGraph()

    for i in range(len(pass_graph)):
        ISO_Graph.add_edge(pass_graph[i][0], pass_graph[i][1], weight=pass_graph[i][2])

    ISO_edges = ISO_Graph.edges()
    ISO_weights = [ISO_Graph[u][v]['weight'] for u, v in ISO_edges]

    fig = plt.figure()

    nx.draw(ISO_Graph, node_size=800, with_labels= True, node_color='white', width=ISO_weights)
    plt.gca().collections[0].set_edgecolor('black')
    
    st.pyplot(fig)

    return ISO_Graph

@st.fragment
def calculate_degree(ISO_Graph):
    degrees_ISO = dict(nx.degree(ISO_Graph))
    df_degrees_ISO = pd.DataFrame.from_dict(list(degrees_ISO.items()))

    df_degrees_ISO.rename(columns = {
        0: "Nome da Jogadora",
        1: "Grau do Nó"
        },
        inplace=True
    )     

    fig = plt.figure()
    X = list(degrees_ISO.keys())
    Y = list(degrees_ISO.values())

    sns.barplot(x=Y, y=X, palette="magma", legend=False)

    plt.xticks(range(0, max(Y)+5, 2))
    plt.xlabel("Grau")
    plt.ylabel("Jogadora")

    st.pyplot(fig)

@st.fragment
def calculate_in_degree(ISO_Graph):
    degrees_ISO = dict(ISO_Graph.in_degree())
    df_degrees_ISO = pd.DataFrame.from_dict(list(degrees_ISO.items()))

    df_degrees_ISO.rename(columns = {
        0: "Nome da Jogadora",
        1: "Grau do Nó"
        },
        inplace=True
    )     

    fig = plt.figure()
    X = list(degrees_ISO.keys())
    Y = list(degrees_ISO.values())

    sns.barplot(x=Y, y=X, palette="magma", legend=False)

    plt.xticks(range(0, max(Y)+5, 2))
    plt.xlabel("Grau")
    plt.ylabel("Jogadora")

    st.pyplot(fig)

@st.fragment
def calculate_out_degree(ISO_Graph):
    degrees_ISO = dict(ISO_Graph.out_degree())
    df_degrees_ISO = pd.DataFrame.from_dict(list(degrees_ISO.items()))

    df_degrees_ISO.rename(columns = {
        0: "Nome da Jogadora",
        1: "Grau do Nó"
        },
        inplace=True
    )     

    fig = plt.figure()
    X = list(degrees_ISO.keys())
    Y = list(degrees_ISO.values())

    sns.barplot(x=Y, y=X, palette="magma", legend=False)

    plt.xticks(range(0, max(Y)+5, 2))
    plt.xlabel("Grau")
    plt.ylabel("Jogadora")

    st.pyplot(fig)

@st.fragment
def calculate_player_metrics(ISO_Graph):
    excentricity_ISO = nx.eccentricity(ISO_Graph, v=None, weight='weight')
    avg_excentricity_ISO = sum(list(excentricity_ISO.values()))/len(excentricity_ISO)
    
    clustering_ISO = nx.clustering(ISO_Graph, weight='weight')
    avg_clustering_ISO = nx.average_clustering(ISO_Graph, weight='weight')

    betweenness_ISO = nx.betweenness_centrality(ISO_Graph, weight='weight')

    closeness_ISO = nx.closeness_centrality(ISO_Graph)

    ds =[excentricity_ISO, clustering_ISO, betweenness_ISO, closeness_ISO]
    df_players = pd.DataFrame(ds)
    df_players = df_players.transpose() 
    df_players.rename(columns={0: 'Excentricidade', 1: 'Clustering', 2: 'Betweeness', 3: 'Closeness'}, inplace=True)

    return df_players, avg_excentricity_ISO, avg_clustering_ISO
# =====================================================================================================

# Lendo as partidas totais da competição do CSV
df_matches = load_matches()
# Criando uma lista para armazenar o nome das partidas
list_matches = []
teams = []

# Adicionando os itens formatados na lista "Time de casa" vs "Time de fora"
for index, df_current_match in df_matches.iterrows():
    list_matches.append(df_current_match["home_team_name"].replace("Women's","") + " vs " + df_current_match["away_team_name"].replace("Women's",""))

home_teams = df_matches["home_team_name"].unique()
# away_teams = df_matches["away_team_name"].unique()

# # Unindo a lista de times
# for team in away_teams:
#     teams.append(team)
    
teams = sorted(home_teams)  
list_matches = sorted(list_matches)

# Definindo o filtro de partidas
teams_choice = st.sidebar.selectbox('Escolha o time:', teams)
df_matches = df_matches[df_matches['home_team_name'] == teams_choice]
# df_team_away = df_matches[df_matches['away_team_name'] == teams_choice]
# dfs=[df_team_home, df_team_away]
# df_matches = pd.concat(dfs)

parser = Sbopen()

# Definindo o título da dashboard
st.title("Análise Baseada em Redes Complexas")
st.subheader("Essa dashboard contém uma análise baseada em Redes Complexas da Copa do Mundo de Futebol Feminino de 2023")
st.image("./pages/cover2.jpg")
st.markdown("--------------------------------------------------")

# =====================================================================================================

st.header(teams_choice.replace("Women's",""))
st.text("As estatísticas abaixo contemplam todos os jogos de " + teams_choice.replace("Women's",""))
for index, df_current_match in df_matches.iterrows():
    st.subheader(df_current_match['home_team_name'].replace("Women's","") + " vs " + df_current_match['away_team_name'].replace("Women's",""))
    df_passes=get_events(match_id = df_current_match['match_id'], match_home_team = df_current_match['home_team_name'])
    df_scatter = calculate_average_position(ht_first_sub_passes=df_passes)
    df_lines = calculate_passes(ht_first_sub_passes=df_passes)
    plot_pitch(df_scatter, df_lines)

# =====================================================================================================

    st.markdown("#### Métricas das Jogadoras")
    
    ISO_Graph = generate_isomorphic_graph(df_lines=df_lines)

# =====================================================================================================

    st.markdown("#### Graus das Jogadoras - Total")

    calculate_degree(ISO_Graph=ISO_Graph)

# =====================================================================================================

    st.markdown("#### Graus das Jogadoras - Entrada")

    calculate_in_degree(ISO_Graph=ISO_Graph)

# =====================================================================================================

    st.markdown("#### Graus das Jogadoras - Saída")

    calculate_out_degree(ISO_Graph=ISO_Graph)

# =====================================================================================================
    st.markdown("#### Centralidade das Jogadoras")

    df_players, avg_excentricity_ISO, avg_clustering_ISO = calculate_player_metrics(ISO_Graph)

    st.dataframe(df_players)

# =====================================================================================================

    st.markdown("#### Métricas do Time")
    
    team_centralization = calculate_team_centralization(ht_first_sub_passes=df_passes)
    st.text("A centralização do time foi de " + "%.2f" % team_centralization)
    st.text("A excentricidade média do time foi de " + "%.2f" % avg_excentricity_ISO)   
    st.text("O coeficiente de clustering médio do time foi de " + "%.2f" % avg_clustering_ISO)   
# =====================================================================================================

    st.markdown("--------------------------------------------------")

    

# df_passes = get_events(df_current_match["match_id"])
# st.dataframe(df_passes)
# df = df[df['position'].isin(position_choice)]
# df = df[df['team'].isin(teams_choice)]
# table = pd.DataFrame({})
# st.table(table)
# st.dataframe(table)
# É possível utilizar markdown também
# st.image("image")
# st.markdown("")
# Bom para fórmulas matemáticas
# st.latex("")
# st.json()
# 
# code="""
# 
# """
# st.code(code, language="python")
# st.metric()