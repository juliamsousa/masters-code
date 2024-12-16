# Importando as bibliotecas necessárias
import streamlit as st

# Definindo o título da dashboard
st.title("Métricas Utilizadas 📈")
st.subheader("Essa seção contém uma descrição das métricas utilizadas para avaliação das jogadoras e dos times.")
st.markdown("--------------------------------------------------")

st.image(".\pages\covermetrics.jpg")

st.markdown("## Métricas Jogadoras 🏃‍♀️")

st.subheader("Grau")

st.text("O grau de um nó em um grafo direcionado consiste na soma dos seus graus de entrada e saída. No contexto do futebol o grau de um nó representa quantidade de passes que uma jogadora participou tanto como passadora quanto como receptora.")

st.subheader("Grau de Entrada")

st.text("O grau de entrada de um nó em um grafo direcionado consiste em todas as arestas que chegam no nó. No contexto do futebol, o grau de entrada de um nó representa a quantidade de passes que uma jogadora participou como receptora. Essa medida também pode demonstrar o prestígio que uma jogadora tem no time, uma grande quantidade de passes recebidos pode indicar uma preferência entre as demais.")

st.subheader("Grau de Saída")

st.text("O grau de saída de um nó em um grafo direcionado consiste em todas as arestas que partem do nó. No contexto do futebol o grau de saída de um nó representa a quantidade de passes que uma jogadora participou como passadora.")


st.subheader("Clusterização")

st.text("O coeficiente de clustering representa a frequência da interação na rede de jogadas e permite discriminar o nível de conexão de diferentes jogadoras enquanto o time esta com a posse de bola.")

st.subheader("Centralidade de Proximidade - Closeness")

st.text("A medida de closeness indica o grau de proximidade do nó jogador com a rede, demonstrando sua capacidade de conectar com o restante dos jogadores. Pode ser entendido como a habilidade de um jogador acessar ou mandar informações entre outros nós da rede.")

st.subheader("Centralidade de Intermediação - Betweeness")

st.text("A medida de betweenness indica o grau de proximidade do nó jogador com a rede, demonstrando sua capacidade de atuar como uma ponte no menor caminho entre dois nós. Essa métrica pode ser interpretada como a possibilidade de um jogador intermediar ações geradas entre dois jogadores.")

st.subheader("Excentricidade")

st.text("A medida de excentricidade indica o grau de proximidade do nó jogador com a rede, demonstrando a qual distância o jogador mais longe desse nó está posicionado na rede.")

st.markdown("--------------------------------------------------")

st.markdown("##  Métricas Time 🥅")

st.subheader("Centralização")

st.text("A medida de centralização do time demonstra o quanto seu modo de jogar foca em um ou mais jogadores. Em um grupo em que todos nós são centrais - o grupo é decentralizado - a medida de centralização é mais baixa. Times decentralizados geralmente apresentam maior sucesso.")


