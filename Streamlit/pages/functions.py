# Importando as bibliotecas necessárias
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch, Sbopen
import pandas as pd
import networkx as nx
import seaborn as sns

# Definindo o título da dashboard
st.title("Implementação 🐍")
st.subheader("Essa seção contém o código desenvolvido na análise da rede de passes.")

st.markdown("#### Extraindo os dados StatsBomb")
code="""
    # Importando as bibliotecas necessárias
    import matplotlib.pyplot as plt
    import numpy as np
    from mplsoccer import Pitch, Sbopen
    import pandas as pd
    import networkx as nx
    import seaborn as sns

    # Instanciando um objeto parser para importar dados abertos da StatsBomb
    parser = Sbopen()

    # Extraindo os dados de competições disponíveis no StatsBomb em um DataFrame
    df_competition = parser.competition()

    # Verificando quais dessas competições são de futebol feminino
    df_female_competitions = df_competition.loc[df_competition['competition_gender'] == "female"]

    ## 
    # Extraindo os dados de todas as partidas da competição Women's World Cup - 2023
    # competition_id = 72 e season_id = 107
    ## 
    df_matches = parser.match(competition_id = 72, season_id = 107)

    # Extraindo o data frame contendo os dados da partida específica
    df_current_match = df_matches.loc[df_matches['match_id'] == 3904629]

    # Extraindo o nome e id do time da casa
    match_home_team = df_current_match.iloc[0]["home_team_name"]
    match_home_team_id = df_current_match.iloc[0]["home_team_id"]

    print(str(match_home_team))
    print(str(match_home_team_id))

    # Extraindo o nome e id do time de fora
    match_away_team = df_current_match.iloc[0]["away_team_name"]
    match_away_team_id = df_current_match.iloc[0]["away_team_id"]

    print(str(match_away_team))
    print(str(match_away_team_id))

    # Extrai os eventos de uma partida fornecendo seu Id para o parser
    df_events, related, freeze, tatics = parser.event(3904629)
"""
st.code(code, language="python")

st.markdown("#### Extraindo os dados dos passes")
code="""
    # Procurando por eventos de substituição no jogo - (Time da Casa)
    substitutions = df_events.loc[df_events["type_name"] == "Substitution"]
    home_team_subs = substitutions.loc[substitutions["team_name"] == match_home_team]

    # Extrai o índice do primeiro evento de substituição - (Time da Casa)
    ht_first_sub_index = home_team_subs.iloc[0]["index"]

    # Extraindo todos os eventos de passe da partida
    match_passes = df_events.loc[df_events["type_name"] == "Pass"]
    # Extraindo os eventos de passe - (Time da Casa)
    home_team_passes = match_passes.loc[match_passes["team_name"] == match_home_team]

    # Removendo eventos de Lateral
    home_team_passes = home_team_passes.loc[match_passes.sub_type_name != "Throw-in"]

    # Os passes bem sucedidos são aqueles nos quais o outcome_name é nulo
    # Extraindo os passes bem sucedidos - (Time da Casa)
    ht_succesful_passes = home_team_passes.loc[home_team_passes["outcome_name"].isnull()]

    ## 
    # Os passes que aconteceram antes da primeira substituição têm id menor que o do evento de substituição. 
    # Os ids dos eventos são sequenciais
    ## 
    ht_first_sub_passes = ht_succesful_passes.loc[ht_succesful_passes["index"] < ht_first_sub_index]

    ## Extraindo os dados necessários para a rede de passes
    # x: coordenada x de início do passe
    # y: coordenada y de início do passe
    # end_x: coordenada x de fim do passe
    # end_y: coordenada y de fim do passe
    # player_name: jogador que inicia o passe
    # pass_recipient_name: jogador que recebe o passe
    ##
    ht_first_sub_passes = ht_first_sub_passes[['x', 'y', 'end_x', 'end_y', "player_name", "pass_recipient_name"]]

    # Formatando o nome das jogadoras para que apenas o sobrenome delas seja exibido
    ht_first_sub_passes["player_name"] = ht_first_sub_passes["player_name"].apply(lambda x: str(x).split()[-1])
    ht_first_sub_passes["pass_recipient_name"] = ht_first_sub_passes["pass_recipient_name"].apply(lambda x: str(x).split()[-1])

    # Obs.: no exemplo do soccermatics um filtro é utilizado para simplificar o código, porém a extração dos dados foi feita passo
    # a passo para melhor entendimento dos dados e como manipulá-los
"""
st.code(code, language="python")

st.markdown("#### Calculando o tamanho e localização dos nós")
code="""
    # Criando um DataFrame vazio para armazenar as informações
    df_scatter = pd.DataFrame()

    # Percorre os eventos de cada jogador por nome
    for i, name in enumerate(ht_first_sub_passes["player_name"].unique()):
        # Extrai as coordenadas x,y (início) do passe - Todos os passes que foram feitos pelo jogador, em que ele foi o passador
        # Os dados dos passes são convertidos para um array numpy
        passer_x = ht_first_sub_passes.loc[ht_first_sub_passes["player_name"] == name]["x"].to_numpy()
        passer_y = ht_first_sub_passes.loc[ht_first_sub_passes["player_name"] == name]["y"].to_numpy()

        # Extrai as coordenas end_x, end_y (fim) do passe - Todos os passes que foram recebidos pelo jogador, em que ele foi o receptor
        # Os dados dos passes são convertidos para um array numpy
        receiver_x = ht_first_sub_passes.loc[ht_first_sub_passes["pass_recipient_name"] == name]["end_x"].to_numpy()
        receiver_y = ht_first_sub_passes.loc[ht_first_sub_passes["pass_recipient_name"] == name]["end_y"].to_numpy()

        # Preenche no DataFrame o nome do jogador sendo analisado no momento
        df_scatter.at[i, "player_name"] = name

        # Calcula a média das coordenadas do jogador para determinar sua posição "média" no campo
        # Média da coordenada x do jogador no campo durante a partida, tanto como passador quanto receptor
        df_scatter.at[i, "x"] = np.mean(np.concatenate([passer_x, receiver_x]))
        # Média da coordenada y do jogador no campo durante a partida, tanto como passador quanto receptor
        df_scatter.at[i, "y"] = np.mean(np.concatenate([passer_y, receiver_y]))
        
        # Preenche no DataFrame a quantidade de passes do jogador sendo analisado no momento
        df_scatter.at[i, "no_passes"] = ht_first_sub_passes.loc[ht_first_sub_passes["player_name"] == name].count().iloc[0]

    # Define o tamanho do marcador do vértice de acordo com a quantidade de passes feitos pelo jogador
    df_scatter["marker_size"] = (df_scatter["no_passes"] / df_scatter["no_passes"].max() * 1500)
"""
st.code(code, language="python")

st.markdown("#### Calculando a espessura das arestas")
code="""
    # Calculando a quantidade de passes entre jogadores
    df_lines = ht_first_sub_passes.groupby(["player_name", "pass_recipient_name"]).x.count().reset_index()

    # Renomeando a coluna para contagem de passes
    df_lines.rename({"x": "pass_count"}, axis="columns", inplace=True)

    # Definindo um limiar para quantidade de passes entre jogadores
    # Nesse caso o limiar está definido a 0, de modo que todos os passes entre jogadores serão considerados
    # É possível avaliar diferentes limiares para quantidade de passes
    df_lines = df_lines[df_lines["pass_count"]>0]   


"""
st.code(code, language="python")

st.markdown("#### Plotando os nós")
code="""
    # Instanciando um campo - Verde com linhas brancas
    pitch = Pitch(pitch_color='grass', line_color='white', stripe=True)
    # Especificações da figura para plotagem
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False, endnote_height=0.04, title_space=0, endnote_space=0)

    # Dados de espalhamento dos jogadores no campo
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

    # Anota os nomes dos jogadores nos nós, assim como suas coordenadas no campo
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

    # Percorre a rede de passes para cada par de jogadores
    for i, row in df_lines.iterrows():
            # Extrai os jogadores
            player1 = row["player_name"]
            player2 = row["pass_recipient_name"]
        
            # Exrai as coordenadas x, y do primeiro jogador
            player1_x = df_scatter.loc[df_scatter["player_name"] == player1]['x'].iloc[0]
            player1_y = df_scatter.loc[df_scatter["player_name"] == player1]['y'].iloc[0]
            
            # Exrai as coordenadas x, y do segundo jogador
            player2_x = df_scatter.loc[df_scatter["player_name"] == player2]['x'].iloc[0]
            player2_y = df_scatter.loc[df_scatter["player_name"] == player2]['y'].iloc[0]

            # Extrai a quantidade de passes trocados pelos jogadores
            num_passes = row["pass_count"]
        
            # Ajusta a espessura da linha do link, quanto mais passes feitos maior será a espessura da linha
            line_width = (num_passes / df_lines['pass_count'].max() * 10)
        
            # Configuração do plot dos links
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

    # Título da partida com time da casa vs. time de fora
    subtitle = "Rede de passes de: " + match_home_team + " contra " + match_away_team

    # Determinando o título da partida
    fig.suptitle(subtitle)

    # Mostrando o campo com os jogadores plotados
    plt.show()
"""
st.code(code, language="python")

st.markdown("#### Cálculo da Centralização")
code="""
    # Calculando o número de passes para cada jogador
    number_of_passes = ht_first_sub_passes.groupby(["player_name"]).x.count().reset_index()
    number_of_passes.rename({'x': 'pass_count'}, axis='columns', inplace=True)

    # Encontrando o jogador com maior número de passes
    max_number_of_passes = number_of_passes["pass_count"].max()

    # Realizando o cálculo da equação de centralização
    # Calculando o denominador
    denominator = 10*number_of_passes["pass_count"].sum()
    # Calculando o numerador
    nominator = (max_number_of_passes - number_of_passes["pass_count"]).sum()

    # Calculando o índice de centralização
    centralisation_index = nominator/denominator

    print("O ídice de centralização do time " + match_home_team + " foi de: %.2f'" %centralisation_index)
"""
st.code(code, language="python")

st.markdown("#### Gerando um Grafo Isomorfo")
code="""
    # O DataFrame de passes entre jogadores é convertido para o formato tupla para ser utilizado como grafo do networkx
    pass_graph = df_lines.apply(tuple, axis=1).tolist()

    # Criando um grafo isomorfo ao grafo de passes
    AUS_Graph = nx.DiGraph()

    # Percorrendo a lista de tuplas do grafo de passes
    for i in range(len(pass_graph)):
        # Criando as arestas entre os nós (passes entre jogadores)
        # pass_graph[i][0]: representa o jogador que originou o passe
        # pass_graph[i][1]: representa o jogador que recebeu o passe
        # pass_graph[i][2]: representa a quantidade de passes entre a dupla de jogadores
        AUS_Graph.add_edge(pass_graph[i][0], pass_graph[i][1], weight=pass_graph[i][2])

    # Extraindo as arestas do grafo criado
    AUS_edges = AUS_Graph.edges()
    # Extraindo os pesos para determinar a espessura das arestas
    AUS_weights = [AUS_Graph[u][v]['weight'] for u, v in AUS_edges]

    # Plotando o grafo isomorfo gerado
    nx.draw(AUS_Graph, node_size=800, with_labels= True, node_color='yellow', width=AUS_weights)
    # Colocando borda nos nós
    plt.gca().collections[0].set_edgecolor('black')
    # Definindo o título do plot
    plt.title("Rede de passes de: " + match_home_team + " contra " + match_away_team, size=30)
    plt.show()
"""
st.code(code, language="python")

st.markdown("#### Cálculo do Grau Total")
code="""
    # Preparando um dicionário com o nome das jogadoras
    degrees_AUS = dict(nx.degree(AUS_Graph))

    # Criando um DataFrame a partir do dicionário de graus das jogadoras
    df_degrees_AUS = pd.DataFrame.from_dict(list(degrees_AUS.items()))

    # Renomeando as colunas para melhor entendimento
    df_degrees_AUS.rename(columns = {
        0: "Nome da Jogadora",
        1: "Grau do Nó"
        },
        inplace=True
    )     

    # Plotando o grau de passes para cada jogadora
    # Extraindo os atributos para plot
    X = list(degrees_AUS.keys())
    Y = list(degrees_AUS.values())

    sns.barplot(x=Y, y=X, palette="magma", legend=False)

    plt.xticks(range(0, max(Y)+5, 2))
    plt.xlabel("Grau")
    plt.ylabel("Jogadora")
    plt.title("Rede de passes de: " + match_home_team + " contra " + match_away_team)

    plt.show()
"""
st.code(code, language="python")

st.markdown("#### Cálculo do Grau de Entrada")
code="""
    # Preparando um dicionário com o nome das jogadoras
    degrees_AUS = dict(AUS_Graph.in_degree())

    # Criando um DataFrame a partir do dicionário de graus das jogadoras
    df_degrees_AUS = pd.DataFrame.from_dict(list(degrees_AUS.items()))

    # Renomeando as colunas para melhor entendimento
    df_degrees_AUS.rename(columns = {
        0: "Nome da Jogadora",
        1: "Grau do Nó"
        },
        inplace=True
    )     

    # Plotando o grau de passes para cada jogadora
    # Extraindo os atributos para plot
    X = list(degrees_AUS.keys())
    Y = list(degrees_AUS.values())

    sns.barplot(x=Y, y=X, palette="magma", legend=False)

    plt.xticks(range(0, max(Y)+5, 2))
    plt.xlabel("Grau")
    plt.ylabel("Jogadora")
    plt.title("Quantidade de passes recebidos por jogadora")

    plt.show()
"""
st.code(code, language="python")

st.markdown("#### Cálculo do Grau de Saída")
code="""
    # Preparando um dicionário com o nome das jogadoras
    degrees_AUS = dict(AUS_Graph.out_degree())

    # Criando um DataFrame a partir do dicionário de graus das jogadoras
    df_degrees_AUS = pd.DataFrame.from_dict(list(degrees_AUS.items()))

    # Renomeando as colunas para melhor entendimento
    df_degrees_AUS.rename(columns = {
        0: "Nome da Jogadora",
        1: "Grau do Nó"
        },
        inplace=True
    )     

    # Plotando o grau de passes para cada jogadora
    # Extraindo os atributos para plot
    X = list(degrees_AUS.keys())
    Y = list(degrees_AUS.values())

    sns.barplot(x=Y, y=X, palette="magma", legend=False)

    plt.xticks(range(0, max(Y)+5, 2))
    plt.xlabel("Grau")
    plt.ylabel("Jogadora")
    plt.title("Quantidade de passes originados por jogadora")

    plt.show()
"""
st.code(code, language="python")

st.markdown("#### Cálculo da Excentricicade")
code="""
    # Calculando a excentricidade das jogadoras
    excentricity_AUS = nx.eccentricity(AUS_Graph, v=None, weight='weight')
    avg_excentricity_AUS = sum(list(excentricity_AUS.values()))/len(excentricity_AUS)
    print("Excentricidade média do time: " + str(avg_excentricity_AUS))
"""
st.code(code, language="python")

st.markdown("#### Cálculo do Coeficiente de Clustering")
code="""
    clustering_AUS = nx.clustering(AUS_Graph, weight='weight')
    avg_clustering_AUS = nx.average_clustering(AUS_Graph, weight='weight')
"""
st.code(code, language="python")

st.markdown("#### Cálculo de Betweenness")
code="""
    betweenness_AUS = nx.betweenness_centrality(AUS_Graph, weight='weight')
"""
st.code(code, language="python")

st.markdown("#### Cálculo de Closeness")
code="""
    closeness_AUS = nx.closeness_centrality(AUS_Graph)
"""
st.code(code, language="python")