# 🍺 Roteamento de Bares - Comida di Buteco 2026 (Juiz de Fora/MG)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OSMnx](https://img.shields.io/badge/OSMnx-1.9+-green.svg)
![Folium](https://img.shields.io/badge/Folium-0.15+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Sobre o Projeto

Este projeto aplica conceitos de **Análise de Redes** e **Otimização de Rotas** para calcular o trajeto mais curto visitando 40 bares participantes do concurso **Comida di Buteco 2026** em Juiz de Fora/MG. 

Embora seja um exercício acadêmico, ele demonstra a aplicação prática do **Problema do Caixeiro Viajante (TSP - Traveling Salesman Problem)** usando dados reais da rede viária do OpenStreetMap.

## 🎯 Objetivo

- Calcular e visualizar a rota mais curta para percorrer todos os bares participantes
- Utilizar a rede viária real (não apenas linhas retas)
- Gerar um mapa interativo com:
  - Distâncias e tempos estimados por trecho
  - Marcadores personalizados (início, meio e fim)
  - Interface responsiva para desktop e mobile

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Função |
|------------|--------|
| **OSMnx** | Download e análise de redes viárias do OpenStreetMap |
| **NetworkX** | Cálculo de caminhos mais curtos (algoritmo de Dijkstra) |
| **Folium** | Visualização de mapas interativos |
| **scikit-learn** | Algoritmo NearestNeighbors (ball_tree) para busca de proximidade |
| **Matplotlib** | Geração de paleta de cores distinta para cada trecho |
| **Pandas/NumPy** | Manipulação e análise de dados |

## 📊 Metodologia

### 1. Coleta e Preparação dos Dados
- Carregamento das coordenadas GPS dos 40 bares a partir de arquivo CSV
- Criação de dicionário de referência: `{nome_do_bar: (lat, lon)}`

### 2. Algoritmo de Roteamento (Heurística do Vizinho Mais Próximo)

```python
# Inicialização
visited[0] = True  # Inicia pelo ADEGA BAR
tour = [0]
current = 0

# Loop principal
while len(tour) < len(X):
    # Encontra vizinhos mais próximos não visitados
    neighbors = indices[current]
    unvisited_mask = ~visited[neighbors]
    
    if np.any(unvisited_mask):
        nearest = neighbors[unvisited_mask][0]
    else:
        # Fallback: busca o mais próximo globalmente
        unvisited_all = np.where(~visited)[0]
        dists = distances[current, unvisited_all]
        nearest = unvisited_all[np.argmin(dists)]
    
    tour.append(nearest)
    visited[nearest] = True
    current = nearest