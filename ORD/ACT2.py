import networkx as nx
import matplotlib.pyplot as plt

# Crear una red aleatoria utilizando el modelo Erdős-Rényi
n = 20  # Número de nodos
p = 0.2  # Probabilidad de conexión entre nodos
G = nx.erdos_renyi_graph(n, p)

# Visualizar la red
nx.draw(G, with_labels=True)
plt.title("Red Aleatoria (Erdős-Rényi)")
plt.show()

# Calcular algunas métricas de la red
print("Número de nodos:", G.number_of_nodes())
print("Número de enlaces:", G.number_of_edges())
print("Coeficiente de clustering promedio:", nx.average_clustering(G))
print("Grado promedio:", sum(dict(G.degree()).values()) / G.number_of_nodes())

