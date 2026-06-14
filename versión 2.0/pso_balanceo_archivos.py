# ==========================================
# Preparación de herramientas (Librerías)
# ==========================================
# Importamos numpy, una herramienta que nos ayuda a hacer cálculos matemáticos muy rápido.
import numpy as np
# Importamos matplotlib, que es como un estuche de dibujo para crear gráficas visuales.
import matplotlib.pyplot as plt
# Importamos csv, para poder leer y escribir archivos con formato de tabla (como Excel).
import csv
# Importamos json, para guardar resultados en un formato ordenado que otros programas pueden leer fácilmente.
import json
# Importamos os, que nos permite hablar con el sistema operativo (por ejemplo, para ver si un archivo ya existe).
import os

# ==========================================
# Creación de los exploradores (Partículas)
# ==========================================
# Imagina que estamos buscando la mejor forma de repartir tareas.
# Cada "Partícula" es como un explorador ciego que intenta encontrar la mejor solución en un mapa.
class Particula:
    # Esta función '__init__' es lo que ocurre cuando "nace" un nuevo explorador.
    def __init__(self, dimensiones, min_val, max_val):
        # Le damos al explorador una posición aleatoria de inicio en el mapa.
        self.posicion = np.random.uniform(min_val, max_val, size=dimensiones)
        
        # Le damos una velocidad inicial aleatoria (hacia dónde se va a mover y qué tan rápido).
        self.velocidad = np.random.uniform(-1, 1, size=dimensiones)
        
        # El explorador tiene un cuaderno donde anota el mejor lugar en el que ÉL ha estado.
        # Como acaba de empezar, su mejor lugar es donde está ahora mismo.
        self.mejor_posicion_individual = np.copy(self.posicion)
        
        # Anota la "calificación" de ese lugar. Empieza en infinito porque queremos encontrar 
        # el número más bajo (el menor tiempo de trabajo), así que cualquier número será mejor que infinito.
        self.mejor_score_individual = float('inf')

# ==========================================
# El Jefe del grupo (El Optimizador)
# ==========================================
# Esta clase es como el director que controla a todos los exploradores.
class OptimizadoPSO:
    # Aquí configuramos las reglas del juego antes de empezar.
    def __init__(self, funcion_aptitud, dimensiones, num_particulas, max_iter, min_val, max_val):
        self.funcion_aptitud = funcion_aptitud # La forma en que calificaremos si una solución es buena o mala.
        self.dimensiones = dimensiones         # Cuántas tareas hay que repartir.
        self.num_particulas = num_particulas   # Cuántos exploradores vamos a enviar.
        self.max_iter = max_iter               # Cuántos pasos o rondas les vamos a dar para buscar.
        self.min_val = min_val                 # Límite izquierdo del mapa (Servidor 0).
        self.max_val = max_val                 # Límite derecho del mapa (El último servidor).
        
        # Estos tres números son la "personalidad" de los exploradores:
        self.w = 0.7   # Inercia: Qué tan tercos son para seguir la dirección que ya llevaban.
        self.c1 = 1.5  # Memoria propia: Qué tanto caso le hacen a su propia experiencia pasada.
        self.c2 = 1.5  # Presión social: Qué tanto caso le hacen al explorador más exitoso del grupo.
        
        # Creamos nuestro grupo (enjambre) lleno de exploradores listos para empezar.
        self.enjambre = [Particula(dimensiones, min_val, max_val) for _ in range(num_particulas)]
        
        # El director tiene un pizarrón donde anota la mejor solución encontrada por CUALQUIERA del grupo.
        self.mejor_posicion_global = np.zeros(dimensiones)
        self.mejor_score_global = float('inf') # De nuevo, empezamos en infinito.
        self.historial_scores = [] # Aquí iremos guardando el progreso para hacer una gráfica al final.

    # Esta es la acción principal: 
    def ejecutar(self):
        # Repetimos el proceso por el número de rondas (iteraciones) que definimos al inicio.
        for iteracion in range(self.max_iter):
            
            # Paso 1: Calificar a todos los exploradores
            for particula in self.enjambre:
                # Vemos qué tan buena es la posición actual de este explorador
                score_actual = self.funcion_aptitud(particula.posicion)
                
                # Si esta posición es mejor (menor tiempo) que su propio récord personal, lo actualiza.
                if score_actual < particula.mejor_score_individual:
                    particula.mejor_score_individual = score_actual
                    particula.mejor_posicion_individual = np.copy(particula.posicion)
                
                # Si esta posición es la mejor que ha visto TODO el grupo, el director lo anota en el pizarrón.
                if score_actual < self.mejor_score_global:
                    self.mejor_score_global = score_actual
                    self.mejor_posicion_global = np.copy(particula.posicion)
            
            # Paso 2: Mover a los exploradores a su siguiente posición
            for particula in self.enjambre:
                # Generamos un poco de aleatoriedad (suerte) para que no todos hagan exactamente lo mismo.
                r1 = np.random.rand(self.dimensiones)
                r2 = np.random.rand(self.dimensiones)
                
                # Calculamos el impulso hacia su mejor recuerdo personal.
                cognitivo = self.c1 * r1 * (particula.mejor_posicion_individual - particula.posicion)
                # Calculamos el impulso hacia el mejor recuerdo de todo el grupo.
                social = self.c2 * r2 * (self.mejor_posicion_global - particula.posicion)
                
                # Sumamos su impulso anterior más la influencia personal y social para obtener la nueva velocidad.
                particula.velocidad = (self.w * particula.velocidad) + cognitivo + social
                
                # Movemos al explorador sumando la velocidad a su posición actual.
                particula.posicion = particula.posicion + particula.velocidad
                
                # Nos aseguramos de que el explorador no se salga de los límites del mapa.
                particula.posicion = np.clip(particula.posicion, self.min_val, self.max_val)
                
            # Guardamos la mejor calificación de esta ronda para la gráfica
            self.historial_scores.append(self.mejor_score_global)
            # Imprimimos en pantalla cómo vamos, para no aburrirnos mientras trabaja.
            print(f"Iteración {iteracion + 1}/{self.max_iter} -> Makespan: {self.mejor_score_global:.2f} ms")
            
        # Cuando terminan todas las rondas, devolvemos el mejor resultado encontrado.
        return self.mejor_posicion_global, self.mejor_score_global, self.historial_scores


# ==========================================
# Funciones para leer y guardar archivos
# ==========================================

# Esta función es como un creador de tareas de emergencia.
# Si no existe el archivo de tareas, crea uno pequeño con 10 tareas de ejemplo.
def crear_csv_ejemplo(filename="tareas_input.csv"):
    # Revisa si el archivo NO existe en tu computadora.
    if not os.path.exists(filename):
        # Lista de tareas inventadas con su nivel de esfuerzo (peso).
        tareas_ejemplo = [
            {"id_tarea": 0, "peso": 15, "descripcion": "Autenticacion de Usuario"},
            {"id_tarea": 1, "peso": 30, "descripcion": "Procesamiento de Imagen"},
            {"id_tarea": 2, "peso": 45, "descripcion": "Consulta Reporte Anual"},
            {"id_tarea": 3, "peso": 10, "descripcion": "Envio Email Notificacion"},
            {"id_tarea": 4, "peso": 20, "descripcion": "Validacion de Token"},
            {"id_tarea": 5, "peso": 50, "descripcion": "Renderizado de Video Short"},
            {"id_tarea": 6, "peso": 25, "descripcion": "Actualizacion Base Datos"},
            {"id_tarea": 7, "peso": 35, "descripcion": "Compresion de Archivos Log"},
            {"id_tarea": 8, "peso": 15, "descripcion": "Verificacion de Pago"},
            {"id_tarea": 9, "peso": 40, "descripcion": "Sincronizacion de Backup"}
        ]
        # Abre un archivo nuevo y guarda la tabla con formato especial.
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["id_tarea", "peso", "descripcion"])
            writer.writeheader()      # Escribe los títulos de las columnas.
            writer.writerows(tareas_ejemplo) # Escribe los datos de las tareas.

# Esta función se encarga de leer el archivo de tareas (como leer la receta).
def cargar_tareas_desde_csv(filename="tareas_input.csv"):
    pesos = []          # Aquí guardaremos qué tan pesada es cada tarea.
    descripciones = []  # Aquí guardaremos el nombre de cada tarea.
    
    # Abrimos el archivo en modo lectura ('r').
    with open(filename, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Vamos fila por fila del archivo leyendo los datos
        for row in reader:
            pesos.append(float(row["peso"])) # Guardamos el peso como un número con decimales.
            descripciones.append(row["descripcion"]) # Guardamos el texto.
            
    # Devolvemos las listas listas para usarse.
    return np.array(pesos), descripciones

# Esta función guarda el resultado final ordenado para que podamos revisarlo después.
def guardar_resultados_json(filename="resultados_optimizacion.json", asignaciones=None, tiempo_final=None, pesos_tareas=None):
    # Creamos un diccionario (como un formulario rellenado) con la información.
    resultado = {
        "metrica_global": {
            "makespan_optimo_ms": round(tiempo_final, 2), # El tiempo que tardaron en total.
            "total_tareas_procesadas": len(asignaciones)  # Cuántas tareas acomodamos.
        },
        "asignaciones_detalladas": []
    }
    
    # Recorremos cada tarea para anotar a qué servidor se le asignó.
    for tarea_id, servidor_id in enumerate(asignaciones):
        resultado["asignaciones_detalladas"].append({
            "id_tarea": tarea_id,
            "peso_tarea": float(pesos_tareas[tarea_id]),
            "servidor_asignado": int(servidor_id)
        })
        
    # Escribimos este "formulario" en un archivo en nuestra computadora.
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, indent=4, ensure_ascii=False)


# ==========================================
# Configuración del entorno de trabajo
# ==========================================
NUM_SERVIDORES = 3 # Cuántas computadoras (servidores) tenemos disponibles para hacer el trabajo.
# Qué tan rápido es cada servidor (El tercero es el más rápido, con 2.0).
CAPACIDAD_SERVIDORES = np.array([1.0, 1.5, 2.0])

# Variables vacías que llenaremos más tarde con los datos del archivo.
PESO_TAREAS = np.array([])
DESCRIPCION_TAREAS = []


# ==========================================
# Fitness (La función que califica)
# ==========================================
# Esta función es súper importante: revisa qué tan buena es una solución.
def evaluar_balanceo_carga(posicion_continua):
    # Los exploradores nos dan posiciones con decimales, pero los servidores son enteros (0, 1 o 2).
    # Así que redondeamos y nos aseguramos de no salirnos de ese rango.
    asignaciones = np.clip(np.round(posicion_continua), 0, NUM_SERVIDORES - 1).astype(int)
    
    # Creamos contadores de tiempo para cada servidor, iniciando en cero.
    tiempo_servidores = np.zeros(NUM_SERVIDORES)
    
    # Repartimos las tareas y sumamos el tiempo que le tomará a cada servidor.
    for tarea_id, servidor_id in enumerate(asignaciones):
        # El tiempo es el peso de la tarea por la rapidez de ese servidor.
        tiempo_servidores[servidor_id] += PESO_TAREAS[tarea_id] * CAPACIDAD_SERVIDORES[servidor_id]
        
    # Devolvemos el tiempo del servidor que va a tardar MÁS (el cuello de botella).
    # Nuestro objetivo como programa es que este número máximo sea lo más pequeño posible.
    return np.max(tiempo_servidores)


# ==========================================
# El Motor Principal (Donde arranca todo)
# ==========================================
# Esto le dice a Python: "Si alguien ejecuta este archivo directamente, haz lo que está aquí abajo".
if __name__ == "__main__":
    
    # 1. Asegurarnos de tener un archivo de pruebas por si acaso.
    crear_csv_ejemplo("tareas_input.csv")
    
    # 2. Leemos las tareas reales del archivo de Big Data (¡AQUÍ SE CAMBIA EL ARCHIVO DE ENTRADA!).
    PESO_TAREAS, DESCRIPCION_TAREAS = cargar_tareas_desde_csv("tareas_bigData.csv")
    
    # Contamos cuántas tareas tenemos en total.
    num_tareas_cargadas = len(PESO_TAREAS)
    
    # 3. Configuramos los límites para nuestros exploradores (del Servidor 0 al Servidor 2).
    limite_inferior = 0
    limite_superior = NUM_SERVIDORES - 1
    
    # 4.
    pso = OptimizadoPSO(
        funcion_aptitud=evaluar_balanceo_carga, # fitness
        dimensiones=num_tareas_cargadas,        # Una dimensión por tarea.
        num_particulas=25,                      # Mandaremos 25 exploradores a buscar.
        max_iter=30,                            # Les daremos 30 rondas para encontrar la respuesta.
        min_val=limite_inferior,
        max_val=limite_superior
    )
    
    # ¡A trabajar! Obtenemos el mejor resultado después de que terminan las rondas.
    mejor_configuracion, mejor_tiempo, historial = pso.ejecutar()
    
    # Convertimos los resultados (que tienen decimales) a números enteros limpios (0, 1 o 2).
    servidores_finales = np.clip(np.round(mejor_configuracion), 0, NUM_SERVIDORES - 1).astype(int)
    
    # Guardamos los resultados finales en un archivo para no perderlos.
    guardar_resultados_json("resultados_optimizacion.json", servidores_finales, mejor_tiempo, PESO_TAREAS)
    
    # 5. Dibujar la gráfica del progreso
    # Preparamos un lienzo de 10x6 pulgadas para dibujar.
    plt.figure(figsize=(10, 6))
    
    # Dibujamos una línea azul conectando el mejor tiempo de cada ronda.
    plt.plot(range(1, len(historial) + 1), historial, marker='s', linestyle='-', color='#1f77b4', linewidth=2)
    
    # Le ponemos títulos bonitos y etiquetas a nuestro dibujo.
    plt.title('PSO Cloud Scheduling - Convergencia Basada en Datos de Entrada', fontsize=12, fontweight='bold')
    plt.xlabel('Iteración del Algoritmo (Número de rondas)')
    plt.ylabel('Makespan Mínimo (ms) (Tiempo del servidor más lento)')
    
    # Ponemos una cuadrícula de fondo para que sea más fácil leer los valores.
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Ponemos una flecha roja apuntando al mejor tiempo final que logramos conseguir.
    # SOLUCIÓN: Usar 'offset points' asegura que el texto de la flecha 
    # siempre se dibuje a 30 píxeles por encima del punto, sin importar los datos.
    plt.annotate(f'Resultado Guardado: {mejor_tiempo:.2f} ms', 
                 xy=(len(historial), historial[-1]), 
                 xytext=(-100, 30), textcoords='offset points',
                 arrowprops=dict(facecolor='red', shrink=0.08, width=1, headwidth=6))
                 
    # Ajustamos los bordes de la imagen para que todo quepa perfectamente.
    plt.subplots_adjust(bottom=0.15, top=0.85, left=0.1, right=0.95)
 
    # Avisamos al usuario qué debe hacer para terminar de correr el programa.
    print("[INFO] Cerrar la gráfica para finalizar el script.")
    
    # Finalmente, mostramos el dibujo en la pantalla.
    plt.show()