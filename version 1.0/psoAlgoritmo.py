import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CLASES DEL ALGORITMO PSO
# ==============================================================================

class Particula:
    def __init__(self, dimensiones, min_val, max_val):
        # Inicializar posición y velocidad de forma aleatoria
        self.posicion = np.random.uniform(min_val, max_val, size=dimensiones)
        self.velocidad = np.random.uniform(-1, 1, size=dimensiones)
        
        # Histórico individual (pbest)
        self.mejor_posicion_individual = np.copy(self.posicion)
        self.mejor_score_individual = float('inf')

class OptimizadoPSO:
    def __init__(self, funcion_aptitud, dimensiones, num_particulas, max_iter, min_val, max_val):
        self.funcion_aptitud = funcion_aptitud
        self.dimensiones = dimensiones
        self.num_particulas = num_particulas
        self.max_iter = max_iter
        self.min_val = min_val
        self.max_val = max_val
        
        # Hiperparámetros del PSO
        self.w = 0.7   # Inercia
        self.c1 = 1.5  # Coeficiente cognitivo
        self.c2 = 1.5  # Coeficiente social
        
        self.enjambre = [Particula(dimensiones, min_val, max_val) for _ in range(num_particulas)]
        self.mejor_posicion_global = np.zeros(dimensiones)
        self.mejor_score_global = float('inf')
        
        # Lista para almacenar el rendimiento a lo largo del tiempo
        self.historial_scores = []

    def ejecutar(self):
        for iteracion in range(self.max_iter):
            for particula in self.enjambre:
                score_actual = self.funcion_aptitud(particula.posicion)
                
                # Actualizar el histórico individual (pbest)
                if score_actual < particula.mejor_score_individual:
                    particula.mejor_score_individual = score_actual
                    particula.mejor_posicion_individual = np.copy(particula.posicion)
                
                # Actualizar el histórico global (gbest)
                if score_actual < self.mejor_score_global:
                    self.mejor_score_global = score_actual
                    self.mejor_posicion_global = np.copy(particula.posicion)
            
            # Actualizar velocidad y posición
            for particula in self.enjambre:
                r1 = np.random.rand(self.dimensiones)
                r2 = np.random.rand(self.dimensiones)
                
                cognitivo = self.c1 * r1 * (particula.mejor_posicion_individual - particula.posicion)
                social = self.c2 * r2 * (self.mejor_posicion_global - particula.posicion)
                
                # Calcular nueva velocidad y posición
                particula.velocidad = (self.w * particula.velocidad) + cognitivo + social
                particula.posicion = particula.posicion + particula.velocidad
                
                # Restringir las posiciones a los límites válidos
                particula.posicion = np.clip(particula.posicion, self.min_val, self.max_val)
                
            # Guardar el mejor score de esta iteración para la gráfica
            self.historial_scores.append(self.mejor_score_global)
            print(f"Iteración {iteracion + 1}/{self.max_iter} -> Mejor tiempo global (Makespan): {self.mejor_score_global:.2f} ms")
            
        return self.mejor_posicion_global, self.mejor_score_global, self.historial_scores

# ==============================================================================
# ESCENARIO: BALANCEO DE CARGA EN CLOUD
# ==============================================================================

NUM_TAREAS = 10       
NUM_SERVIDORES = 3    

# Pesos simulados de cada tarea (ej. millones de instrucciones)
PESO_TAREAS = np.array([15, 30, 45, 10, 20, 50, 25, 35, 15, 40])  

# Capacidad o factor de coste de los servidores (menor es más rápido)
# Server 0 = 1.0x, Server 1 = 1.5x, Server 2 = 2.0x
CAPACIDAD_SERVIDORES = np.array([1.0, 1.5, 2.0]) 

def evaluar_balanceo_carga(posicion_continua):
    """
    Mapea el espacio continuo del PSO a asignaciones discretas de servidores
    y calcula el Makespan (tiempo del servidor más saturado).
    """
    asignaciones = np.clip(np.round(posicion_continua), 0, NUM_SERVIDORES - 1).astype(int)
    tiempo_servidores = np.zeros(NUM_SERVIDORES)
    
    for tarea_id, servidor_id in enumerate(asignaciones):
        tiempo_servidores[servidor_id] += PESO_TAREAS[tarea_id] * CAPACIDAD_SERVIDORES[servidor_id]
        
    return np.max(tiempo_servidores)

# ==============================================================================
# BLOQUE DE EJECUCIÓN PRINCIPAL Y GRAFICACIÓN
# ==============================================================================

if __name__ == "__main__":
    limite_inferior = 0
    limite_superior = NUM_SERVIDORES - 1
    
    # Instanciar el algoritmo
    pso = OptimizadoPSO(
        funcion_aptitud=evaluar_balanceo_carga,
        dimensiones=NUM_TAREAS,
        num_particulas=20,
        max_iter=30,
        min_val=limite_inferior,
        max_val=limite_superior
    )
    
    # Ejecutar la optimización
    mejor_configuracion, mejor_tiempo, historial = pso.ejecutar()
    
    # Imprimir resultados en consola
    servidores_finales = np.clip(np.round(mejor_configuracion), 0, NUM_SERVIDORES - 1).astype(int)
    print("\n" + "="*50)
    print(" RESULTADO FINAL DE LA OPTIMIZACIÓN ")
    print("="*50)
    print(f"Asignación óptima de tareas a servidores:\n{servidores_finales}")
    print(f"Tiempo máximo de procesamiento (Makespan): {mejor_tiempo:.2f} ms\n")

    # Generar la gráfica de convergencia
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(historial) + 1), historial, marker='o', linestyle='-', color='#2ca02c', linewidth=2)
    
    plt.title('Curva de Convergencia: Optimización por Enjambre de Partículas (PSO)', fontsize=14, fontweight='bold')
    plt.xlabel('Número de Iteración', fontsize=12)
    plt.ylabel('Makespan (ms) [Menor es mejor]', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Anotar el punto óptimo final
    plt.annotate(f'Óptimo: {mejor_tiempo:.2f} ms', 
                 xy=(len(historial), historial[-1]), 
                 xytext=(len(historial) - 5, historial[-1] + (max(historial)*0.1)), # Ajuste dinámico de texto
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11)
                 
    plt.tight_layout()
    plt.show()