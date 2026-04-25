# Descubra o melhor percurso com Osmnx e Networkx

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OSMnx](https://img.shields.io/badge/OSMnx-1.9+-green.svg)
![Folium](https://img.shields.io/badge/Folium-0.15+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

### Introdução

Este projeto aplica conceitos de **Análise de redes** e **Otimização de rotas** para calcular o trajeto mais curto para percorrer os quarenta bares participantes do concurso **Comida di Buteco 2026**, em Juiz de Fora/MG. 

Embora tenha propósito didático, demonstra a aplicação prática do **Problema do Caixeiro Viajante (TSP - Traveling Salesman Problem)** usando dados reais da rede viária do *OpenStreetMap*.


### Objetivo

Calcular e visualizar a rota mais curta para percorrer todos os bares participantes, utilizando a rede viária real e gerar um mapa interativo com as distâncias e tempos estimados por trecho, marcadores personalizados e interface responsiva.


### Importamos as bibliotecas


```python
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import folium
from folium.plugins import Fullscreen
from sklearn.neighbors import NearestNeighbors
import matplotlib.colors as mcolors
from matplotlib import colormaps
import warnings

warnings.filterwarnings('ignore')
```

### Carregamos os dados em um DataFrame


```python
gdf = pd.read_csv("lista_bares.csv")
```

### Inspecionamos o DataFrame


```python
gdf[:5]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>Endereço</th>
      <th>Petisco</th>
      <th>Contato</th>
      <th>Instagram</th>
      <th>Descrição</th>
      <th>Funcionamento</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ADEGA BAR</td>
      <td>-43.298967</td>
      <td>-21.782000</td>
      <td>Av. Dr. Francisco Álvares de Assis, 490 – Retiro</td>
      <td>Caruru Mineiro</td>
      <td>(32) 98893-1082</td>
      <td>adegawinebar</td>
      <td>Uma fusão da Bahia com Minas: caruru de taioba...</td>
      <td>Terça a sexta – 18h às 23h | Sábado – 12h às 0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BAR DIAS</td>
      <td>-43.360996</td>
      <td>-21.736599</td>
      <td>Rua Luís Rocha, 2 – Santa Terezinha</td>
      <td>Sabor de Buteco</td>
      <td>(32) 3224-9914</td>
      <td>bardiasjf</td>
      <td>Nhoque de agrião acompanhado de ragu de rabada...</td>
      <td>Segunda a Sábado – 10h às 23h | Domingo – 10h ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BAR DO ABILIO</td>
      <td>-43.347222</td>
      <td>-21.758611</td>
      <td>Rua Fonseca Hermes, 180 – Centro</td>
      <td>Brigadeiro de buteco</td>
      <td>(32) 3215-6216</td>
      <td>bardoabilio</td>
      <td>Bolinho de queijo mussarela e bacon, recheado ...</td>
      <td>Segunda a Sexta – 11h às 22h | Sábado – 11h às...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BAR DO ANTONIO</td>
      <td>-43.372311</td>
      <td>-21.766567</td>
      <td>Rua José Lourenço, 1262 – São Pedro</td>
      <td>Mostarda, mas não falha</td>
      <td>(32) 98709-0353</td>
      <td>bar.do.antonio</td>
      <td>Cama de angu da roça temperado, rodeado de mos...</td>
      <td>Terça a Quinta – 17h às 23h30 | Sexta – 16h às...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BAR DO BENE</td>
      <td>-43.378489</td>
      <td>-21.775617</td>
      <td>Rua João Batista Sampaio, 150 – São Pedro</td>
      <td>Deu cupim no agrião</td>
      <td>(32) 98813-6345</td>
      <td>bardobeneoficial</td>
      <td>Bolinho de agrião recheado com cupim ao vinho ...</td>
      <td>Terça – 16h às 22h | Quarta a Sexta – 16h às 2...</td>
    </tr>
  </tbody>
</table>
</div>




```python
gdf.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 40 entries, 0 to 39
    Data columns (total 9 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Name           40 non-null     object 
     1   longitude      40 non-null     float64
     2   latitude       40 non-null     float64
     3   Endereço       40 non-null     object 
     4   Petisco        40 non-null     object 
     5   Contato        40 non-null     object 
     6   Instagram      40 non-null     object 
     7   Descrição      40 non-null     object 
     8   Funcionamento  40 non-null     object 
    dtypes: float64(2), object(7)
    memory usage: 2.9+ KB


### Calculamos a rota mais curta

A classe *NearestNeighbors* do módulo *sklearn.neighbors*, junto com o
algoritmo *ball_tree*, fornece uma solução robusta para problemas de
busca por proximidade, como o do roteamento entre pontos geográficos.

Foi definido um ponto de início (Ponto 0 - ADEGA BAR) e um ponto de término específico (Ponto 5 - BAR DO BREJO). O algoritmo principal (*while*) constrói a rota, adicionando o vizinho mais próximo ainda não visitado até atingir o ponto de destino ou visitar todos os pontos.

A saída mostra a sequência de bares a serem visitados na rota calculada.


```python
X = np.array(gdf[['latitude', 'longitude']])
bares = gdf['Name'].tolist()

# Criar modelo de vizinhos mais próximos
nbrs = NearestNeighbors(n_neighbors=len(X), algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

# Algoritmo para construir a rota
visited = np.zeros(len(X), dtype=bool)
visited[0] = True  # Iniciar pelo primeiro bar (ADEGA BAR)
tour = [0]
current = 0

# Construir rota visitando o vizinho mais próximo não visitado
while len(tour) < len(X):
    neighbors = indices[current]
    unvisited_mask = ~visited[neighbors]
    
    if np.any(unvisited_mask):
        nearest = neighbors[unvisited_mask][0]
    else:
        # Fallback: pegar o mais próximo global se não houver vizinhos
        unvisited_all = np.where(~visited)[0]
        if len(unvisited_all) == 0:
            break
        dists = distances[current, unvisited_all]
        nearest = unvisited_all[np.argmin(dists)]
    
    tour.append(nearest)
    visited[nearest] = True
    current = nearest

# print(f"✅ Roteiro calculado com {len(tour)} bares (deveria ser 40).")

```

### Definimos o grafo

Usamos o pacote **OSMnx** para baixar um grafo da rede viária para
veículos automotivos, centrado no primeiro ponto da rota ordenada, com um raio de 25 km (dist=25000).


```python

center = (X[:, 0].mean(), X[:, 1].mean())
G = ox.graph_from_point(center, dist=25000, network_type='drive')
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
nodes, edges = ox.graph_to_gdfs(G)

# Gerar cores distintas para cada trecho
cmap = colormaps['tab10'].resampled(max(len(tour)-1, 2))
hex_colors = [mcolors.rgb2hex(cmap(i)) for i in range(len(tour)-1)]

```

### Definimos o mapa e calculamos os trechos

O mapa **Folium** é inicializado centralizado no primeiro ponto da rota.
Marcadores são adicionados para cada bar na ordem da rota, com ícones diferentes para início (verde), fim (vermelho) e demais (azul com ícone de cerveja).

Em seguida, itera pelos pares de coordenadas consecutivas na rota ordenada.
Para cada par (origem e destino), encontra os nós mais próximos no grafo **OSM** e calcula o caminho mais curto (*ox.shortest_path*) considerando o tempo ou a distância.

As distâncias e o tempo de viagem de cada segmento e do caminho completo são acumuladas.
Os nós de todos os segmentos são concatenados, evitando duplicação do nó inicial de um novo segmento se ele for igual ao nó final do anterior.

A rota completa calculada é desenhada como uma linha poligonal colorida no mapa. Um rótulo HTML fixo é adicionado ao mapa mostrando a distância total e o número de paradas. São adicionadas extensões úteis como *MeasureControl* e *Fullscreen*.



```python

m = folium.Map(location=center, zoom_start=13, scrollWheelZoom=True)

# Adicionar controle de tela cheia
Fullscreen().add_to(m)

# Adicionar controle de medição (opcional, mas útil)
MeasureControl(
    position='bottomleft',
    primary_length_unit='kilometers',
    secondary_length_unit='meters',
    primary_area_unit='sqmeters',
    secondary_area_unit='hectares'
).add_to(m)

# Inicializar acumulador de distância
distancia_total_m = 0.0
tempo_total_sec = 0.0

# Adicionar os trechos coloridos com popup de distância/tempo
for i in range(len(tour) - 1):
    idx_start, idx_end = tour[i], tour[i+1]
    lat1, lon1 = X[idx_start]
    lat2, lon2 = X[idx_end]
    
    # Encontrar nós mais próximos no grafo
    node1 = ox.distance.nearest_nodes(G, lon1, lat1)
    node2 = ox.distance.nearest_nodes(G, lon2, lat2)
    
    # Calcular caminho mais curto (prioriza tempo, fallback para distância)
    try:
        path = nx.shortest_path(G, node1, node2, weight='travel_time')
    except nx.NetworkXNoPath:
        path = nx.shortest_path(G, node1, node2, weight='length')
    
    dist_m, time_sec = 0.0, 0.0
    path_coords = []
    
    # Extrair geometria real das vias (inclui curvas)
    for u, v in zip(path[:-1], path[1:]):
        edge_data = G.edges[u, v, 0]
        dist_m += edge_data.get('length', 0)
        time_sec += edge_data.get('travel_time', 0)
        
        # Usar geometria real se disponível
        if 'geometry' in edge_data:
            for lon, lat in edge_data['geometry'].coords:
                path_coords.append((lat, lon))
        else:
            # Fallback: linha reta entre nós
            path_coords.append((nodes.loc[u, 'y'], nodes.loc[u, 'x']))
            path_coords.append((nodes.loc[v, 'y'], nodes.loc[v, 'x']))
    
    # Calcular tempo estimado se não disponível
    if time_sec == 0:
        time_sec = (dist_m / 1000) / 40 * 3600  # Estimativa: 40 km/h
    
    dist_km = round(dist_m / 1000, 2)
    time_min = round(time_sec / 60, 1)
    
    # Acumular distância e tempo totais
    distancia_total_m += dist_m
    tempo_total_sec += time_sec
    
    # Criar popup HTML
    popup_html = f"""
    <div style='font-family:sans-serif; font-size:12px; padding:5px;'>
        <b style='color:#ff6b6b; font-size:13px;'>{bares[idx_start]} ➝ {bares[idx_end]}</b><br>
        <hr style='margin:5px 0; border:none; border-top:1px solid #ddd;'>
        📍 Distância: <b>{dist_km} km</b><br>
        ⏱️ Tempo estimado: <b>{time_min} min</b>
    </div>"""
    
    # Adicionar linha ao mapa
    folium.PolyLine(
        locations=path_coords,
        color=hex_colors[i],
        weight=4,
        opacity=0.85,
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(m)

for i, idx in enumerate(tour):
    lat, lon = X[idx, 0], X[idx, 1]
    bar_name = bares[idx]
    
    # Definir ícone conforme posição na rota
    if i == 0:
        icon_color, icon_name = 'green', 'play'
        popup_text = f"<b style='color:green; font-size:15px;'>🟢 {bar_name}</b><br>🏁 Início do percurso<br>Parada 1 de {len(tour)}"
    elif i == len(tour) - 1:
        icon_color, icon_name = 'red', 'stop'
        popup_text = f"<b style='color:red; font-size:15px;'>🔴 {bar_name}</b><br>🏁 Fim do percurso<br>Parada {i+1} de {len(tour)}"
    else:
        icon_color, icon_name = 'blue', 'beer'
        popup_text = f"<b style='color:blue;'>🔵 {bar_name}</b><br>Parada {i+1} de {len(tour)}"
    
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_text, max_width=250),
        tooltip=f"{i+1}. {bar_name}",  # Aparece ao passar o mouse
        icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa'),
        draggable=False  # Marcadores fixos
    ).add_to(m)

distancia_total_km = distancia_total_m / 1000.0
tempo_total_horas = tempo_total_sec / 3600.0

label_html = f"""
<div style="
    position: fixed; 
    top: 10px; 
    right: 10px; 
    z-index: 1000; 
    background-color: white; 
    padding: 15px 18px; 
    border-radius: 10px; 
    border: 3px solid #ff6b6b; 
    font-family: 'Segoe UI', Arial, sans-serif; 
    font-size: 14px; 
    font-weight: bold; 
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    text-align: center;
    min-width: 280px;
">
    <div style="color: #ff6b6b; font-size: 18px; margin-bottom: 8px; border-bottom: 2px solid #ff6b6b; padding-bottom: 5px;">
        🍺 Percurso Comida di Buteco 2026
    </div>
    <div style="margin: 8px 0;">
        <div>📍 Total de paradas: <span style="color: #2c3e50;">{len(tour)} bares</span></div>
        <div style="margin-top: 5px;">🚗 Distância total: <span style="color: #ff6b6b; font-size: 16px;">{distancia_total_km:.2f} km</span></div>
        <div style="margin-top: 5px;">⏱️ Tempo estimado: <span style="color: #27ae60;">{tempo_total_horas:.1f} horas</span></div>
    </div>
    <div style="
        font-size: 11px; 
        color: #666; 
        margin-top: 10px; 
        padding-top: 8px; 
        border-top: 1px solid #ddd;
        display: flex; 
        justify-content: space-around;
    ">
        <span>🟢 Início</span>
        <span>🔴 Fim</span>
        <span>🔵 Demais</span>
    </div>
    <div style="
        font-size: 10px; 
        color: #999; 
        margin-top: 8px;
        font-style: italic;
    ">
        💡 Clique nos trechos para ver detalhes
    </div>
</div>
"""

m.get_root().html.add_child(folium.Element(label_html))
```

```python
display(m)
```
![](percurso_cb.png)

**Considerações finais:**

O cenário proposto é um exercício prático e não uma sugestão realista de itinerário, dado que o propósito pedagógico por trás da análise é o aprendizado e aplicação de conceitos e ferramentas para resolver um problema complexo como o **TSP** (*Traveling Salesman Problem*), embora, nesse caso, sem a garantia de solução ideal. Por fim, o projeto conecta a atividade prática realizada, que consiste em calcular uma rota entre pontos, ao conceito teórico da **TSP** e à construção da rota final.

**Referências**

Boeing, G. (2025). Modeling and Analyzing Urban Networks and Amenities
with OSMnx. Geographical Analysis, published online ahead of print.
<doi:10.1111/gean.70009>

SCIKIT-LEARN. User Guide: Nearest Neighbors. 2025. Disponível em:
<https://scikit-learn.org/stable/modules/neighbors.html>. Acesso em: 18
JUN 2025.

