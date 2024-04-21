import numpy as np
import matplotlib.pyplot as plt
from simanneal import Annealer

# Función de costo de ejemplo
def costo(x):
    return (x - 3) ** 2 + 5

# Definir el problema de optimización
class Optimizacion(Annealer):
    def __init__(self, state):
        super(Optimizacion, self).__init__(state)
    
    def move(self):
        self.state += np.random.uniform(-0.5, 0.5)
        self.state = max(0, min(10, self.state))  # Limitar el estado entre 0 y 10
    
    def energy(self):
        return costo(self.state)

# Estado inicial aleatorio
state = np.random.uniform(0, 10)

# Inicializar la simulación de auto-ajuste de parámetros
opt = Optimizacion(state)
opt.set_schedule(opt.auto(minutes=0.2))

# Ejecutar la simulación
state, e = opt.anneal()

print("Parámetro óptimo encontrado:", state)
print("Valor mínimo de la función de costo:", e)

# Visualización de la función de costo
x = np.linspace(0, 10, 100)
y = costo(x)

plt.plot(x, y)
plt.scatter(state, costo(state), color='red', label='Parámetro óptimo encontrado')
plt.legend()
plt.xlabel('Parámetro')
plt.ylabel('Función de costo')
plt.title('Simulación de auto-ajuste de parámetros')
plt.show()
