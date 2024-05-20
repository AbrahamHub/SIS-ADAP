import numpy as np

def construir_grafo(conexiones):
    grafo = {}
    for amigo1, amigo2 in conexiones:
        if amigo1 not in grafo:
            grafo[amigo1] = []
        if amigo2 not in grafo:
            grafo[amigo2] = []
        grafo[amigo1].append(amigo2)
        grafo[amigo2].append(amigo1)
    return grafo

def generar_matriz_adyacencias(grafo):
    amigos = sorted(list(grafo.keys()))
    matriz = np.zeros((len(amigos), len(amigos)), dtype=int)
    for i, amigo1 in enumerate(amigos):
        for amigo2 in grafo[amigo1]:
            j = amigos.index(amigo2)
            matriz[i][j] = 1
    return matriz

conexiones = [("Amigo1", "Amigo2"), ("Amigo1", "Amigo3"), ("Amigo2", "Amigo3"), ("Amigo3", "Amigo4")]

grafo = construir_grafo(conexiones)
matriz_adyacencias = generar_matriz_adyacencias(grafo)

print("Grafo:")
for amigo, conexiones in grafo.items():
    print(f"{amigo}: {conexiones}")
print("\nMatriz de Adyacencias:")
print(matriz_adyacencias)
