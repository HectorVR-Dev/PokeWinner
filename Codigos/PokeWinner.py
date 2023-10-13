from logic import *
import pandas as pd
import termcolor
from pomegranate import *
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


atk = Node(DiscreteDistribution({
    0: 0.5,
    1: 0.5
}), name="atk")

spatk = Node(DiscreteDistribution({
    0: 0.5,
    1: 0.5
}), name="spatk")

spdef = Node(DiscreteDistribution({
    0: 0.5,
    1: 0.5
}), name="spdef")

defensa = Node(DiscreteDistribution({
    0: 0.5,
    1: 0.5
}), name="def")

speed = Node(DiscreteDistribution({
    0: 0.5,
    1: 0.5
}), name="speed")

total = Node(DiscreteDistribution({
    0: 0.5,
    1: 0.5
}), name="total")

# Nodo de Mantenimiento esta condicionado por la lluvia
atk_T = Node(ConditionalProbabilityTable([

    [1,1,1, 0.95],
    [1,1,0, 0.05],

    [1,0,1, 0.7],
    [1,0,0, 0.3],

    [0,1,1, 0.3],
    [0,1,0, 0.7],

    [0,0,1, 0.05],
    [0,0,0, 0.95]

], [atk.distribution, spatk.distribution]), name="atk Total")

def_T = Node(ConditionalProbabilityTable([

    [1,1,1, 0.95],
    [1,1,0, 0.05],

    [1,0,1, 0.7],
    [1,0,0, 0.3],

    [0,1,1, 0.3],
    [0,1,0, 0.7],

    [0,0,1, 0.05],
    [0,0,0, 0.95]

], [defensa.distribution, spdef.distribution]), name="def Total")

plus = Node(ConditionalProbabilityTable([

    [1,1,1, 0.95],
    [1,1,0, 0.05],

    [1,0,1, 0.6],
    [1,0,0, 0.4],

    [0,1,1, 0.4],
    [0,1,0, 0.6],

    [0,0,1, 0.05],
    [0,0,0, 0.95],

], [speed.distribution, total.distribution]), name="plus")

win = Node(ConditionalProbabilityTable([

    [1,1,1,1, 0.9],
    [1,1,1,0, 0.1],

    [1,1,0,1, 0.7],
    [1,1,0,0, 0.3],

    [1,0,1,1, 0.7],
    [1,0,1,0, 0.3],

    [1,0,0,1, 0.2],
    [1,0,0,0, 0.8],

    [0,1,1,1, 0.7],
    [0,1,1,0, 0.3],

    [0,1,0,1, 0.3],
    [0,1,0,0, 0.7],

    [0,0,1,1, 0.2],
    [0,0,1,0, 0.8],
    
    [0,0,0,1, 0.1],
    [0,0,0,0, 0.9]

], [atk_T.distribution, def_T.distribution, plus.distribution]), name="WIN")

# Creamos una Red Bayesiana y añadimos estados
modelo = BayesianNetwork()
modelo.add_states(atk, spatk, defensa, spdef, speed, total, atk_T, def_T, plus, win)

# Añadimos bordes que conecten nodos
modelo.add_edge(atk, atk_T)
modelo.add_edge(spatk, atk_T)

modelo.add_edge(defensa, def_T)
modelo.add_edge(spdef, def_T)

modelo.add_edge(speed, plus)
modelo.add_edge(total, plus)

modelo.add_edge(atk_T, win)
modelo.add_edge(def_T, win)
modelo.add_edge(plus, win)
# Modelo Final
modelo.bake()

def mejores(candidatos, P_candidatos, n):
    best = np.array([candidatos[0] , P_candidatos[0]])

    for N_candidato in range(1,len(P_candidatos)):
        temp = np.array([candidatos[N_candidato] , P_candidatos[N_candidato]])

        if len(best) < n:
            best = np.vstack((best, temp))
        else: 
            minimo = np.argmin(best[:, -1])
            if best[minimo][1] < temp[1]:
                best[minimo] = temp
        
    return best 
    

def inferencia(comparativa):
    predit = []
    def_candidatos = []
    for i in range(len(comparativa)):
        def_candidatos.append({'atk': comparativa[i][0], 'def': comparativa[i][1], 'spatk': comparativa[i][2], 'spdef': comparativa[i][3], 'speed': comparativa[i][4], 'total': comparativa[i][5]})
    for i in def_candidatos:
        temp = modelo.predict_proba(i)[-1].parameters[0][1]
        predit.append(temp)
        
    return predit

def print_poke(poke):
    print(f"\nEl pokemon a derrotar es ",end='') 
    termcolor.cprint(f'{poke[1]} ','green',end='')

    print('de tipo',end='') 
    termcolor.cprint(f' {poke[3]} ', 'green',end='')

    if pd.isna(poke[4]):
        print()
    else:
        print('y ',end='')
        termcolor.cprint(f'{poke[4]}', 'green')

class pokemon():
    def __init__(self, pokemon):
        self.id = pokemon[0]
        self.tipo1 = Symbol(pokemon[3])
        if pokemon[4]:
            self.tipo2 = None
        else:
            self.tipo2 = Symbol(pokemon[4])

class Frontera():
    def __init__(self):
        self.frontera = []

    def empty(self):
        return (len(self.frontera) == 0)

    def add(self, nodo):
        self.frontera.append(nodo)

class Cola(Frontera):
    def eliminar(self):
        if self.empty():
            raise Exception("Frontera vacia")
        else:
            nodo = self.frontera[0]

            self.frontera = self.frontera[1:]
            return nodo
        
class poke_winner():
    def __init__(self, file):
        self.df = pd.read_csv(file)

    def vecinos(self, id):
        resultados = []
        if id != 1 and id != 1017:
            resultados.append(id-1)
            resultados.append(id+1)
        elif id == 1:
            resultados.append(id+1)
        elif id == 1017:
            resultados.append(id-1)
        return resultados
    
    def buscar(self, poke_objetivo, tipos, n_candidatos):
        
        self.num_explorados = 0
        candidatos = []
        inicio = pokemon( pokemon = poke_objetivo)
        frontera = Cola()
        frontera.add(inicio)

        self.explorado = set()

        while True:

            # Si nada queda en la frontera, entonces no hya mas camino
            if frontera.empty():
                print(f"Solo se enconraron {len(candidatos)}")
                return candidatos

            # Escogemos un nodo de la frontera
            nodo = frontera.eliminar()
            self.num_explorados += 1
            # Si el nodo es el objetivo, entonces tenemos una solución
            if nodo.tipo1 in tipos and nodo.tipo2 in tipos:
                candidatos.append(nodo.id)
                if len(candidatos) == n_candidatos:
                    return candidatos 
            elif nodo.tipo1 in tipos and nodo.tipo2 == None:
                candidatos.append(nodo.id)
                if len(candidatos) == n_candidatos:
                    return candidatos

            # Marcamos el nodo como exploado
            self.explorado.add(nodo.id)

            # Agregamos vecinos a la frontera
            for id_poke in self.vecinos(nodo.id):
                if id_poke not in self.explorado:
                    poke = pokemon(pokemon = self.df[self.df['ID'] == id_poke].values.tolist()[0])
                    frontera.add(poke)

    def com_est(self, poke_obje, candidatos ):
        est_objetivo = poke_obje[5:12]
        est_candidatos = []
        comparativa = []

        for id_poke in candidatos:
            temp = self.df[self.df['ID'] == id_poke].values.tolist()[0][5:12]
            est_candidatos.append(temp)
        
        for est in est_candidatos:
            temp = [0,0,0,0,0,0]
            if (est_objetivo[0]+est_objetivo[2])/2 > est[1]: #compara el promedio de la hp y def del objetivo con el atk del candidato
                temp[0] = 1
            else:
                temp[0] = 0

            if (est[0]+est[2])/2 > est_objetivo[1]: #compara el promedio de la hp y def del candidato con el atk del objetivo
                temp[1] = 1
            else:
                temp[1] = 0

            if est[3] > est_objetivo[4]: #compara el spatk del candidato con la spdef del objetivo
                temp[2] = 1
            else:
                temp[2] = 0

            if est[4] > est_objetivo[3]: #compara el spdef del candidato con la spatk del objetivo
                temp[3] = 1
            else:
                temp[3] = 0

            if est[5] > est_objetivo[5]: #compara el speed del candidato con la speed del objetivo
                temp[4] = 1
            else:
                temp[4] = 0

            if est[6] > est_objetivo[6]: #compara el total del candidato con la total del objetivo
                temp[5] = 1
            else:
                temp[5] = 0

            comparativa.append(temp)
        return comparativa
        

class poke_logic():
    def __init__(self):

        self.fuego = Symbol('fuego')
        self.agua = Symbol('agua')
        self.acero = Symbol('acero')
        self.bicho = Symbol('bicho')
        self.dragon = Symbol('dragon')
        self.electrico = Symbol('electrico')
        self.fantasma = Symbol('fantasma')
        self.hada = Symbol('hada')
        self.hielo = Symbol('hielo')
        self.lucha = Symbol('lucha')
        self.normal = Symbol('normal')
        self.planta = Symbol('planta')
        self.psiquico = Symbol('psiquico')
        self.roca = Symbol('roca')
        self.siniestro = Symbol('siniestro')
        self.veneno = Symbol('veneno')
        self.volador = Symbol('volador')
        self.tierra = Symbol('tierra')

        self.tipos = (self.fuego,self.agua,self.acero,self.bicho,self.dragon,
                      self.electrico,self.fantasma,self.hada,self.hielo,self.lucha,
                      self.normal,self.planta,self.psiquico,self.roca,self.siniestro,
                      self.veneno,self.volador,self.tierra)
        
        self.knowledge = And(Or(self.fuego,self.agua,self.acero,self.bicho,self.dragon,
                      self.electrico,self.fantasma,self.hada,self.hielo,self.lucha,
                      self.normal,self.planta,self.psiquico,self.roca,self.siniestro,
                      self.veneno,self.volador,self.tierra))
        
    def tipos_winner(self,poke_objetivo):

        if poke_objetivo[3] == 'acero' or poke_objetivo[4] == 'acero':
            self.knowledge.add(self.tierra)
            self.knowledge.add(self.fuego)
            self.knowledge.add(self.lucha)

        if poke_objetivo[3] == 'agua' or poke_objetivo[4] == 'agua':
            self.knowledge.add(self.electrico)
            self.knowledge.add(self.planta)

        if poke_objetivo[3] == 'bicho' or poke_objetivo[4] == 'bicho':
            self.knowledge.add(self.fuego)
            self.knowledge.add(self.volador)
            self.knowledge.add(self.roca)
             
        if poke_objetivo[3] == 'dragon' or poke_objetivo[4] == 'dragon':
            self.knowledge.add(self.hielo)
            self.knowledge.add(self.hada)
            self.knowledge.add(self.dragon)

        if poke_objetivo[3] == 'electrico' or poke_objetivo[4] == 'electrico':
            self.knowledge.add(self.tierra)

        if poke_objetivo[3] == 'fantasma' or poke_objetivo[4] == 'fantasma':
            self.knowledge.add(self.fantasma)
            self.knowledge.add(self.siniestro)

        if poke_objetivo[3] == 'fuego' or poke_objetivo[4] == 'fuego':
            self.knowledge.add(self.agua)
            self.knowledge.add(self.tierra)
            self.knowledge.add(self.roca)

        if poke_objetivo[3] == 'hada' or poke_objetivo[4] == 'hada':
            self.knowledge.add(self.acero)
            self.knowledge.add(self.veneno)

        if poke_objetivo[3] == 'hielo' or poke_objetivo[4] == 'hielo':
            self.knowledge.add(self.acero)
            self.knowledge.add(self.fuego)
            self.knowledge.add(self.lucha)
            self.knowledge.add(self.roca)

        if poke_objetivo[3] == 'lucha' or poke_objetivo[4] == 'lucha':
            self.knowledge.add(self.psiquico)
            self.knowledge.add(self.volador)
            self.knowledge.add(self.hada)

        if poke_objetivo[3] == 'normal' or poke_objetivo[4] == 'normal':
            self.knowledge.add(self.lucha)

        if poke_objetivo[3] == 'planta' or poke_objetivo[4] == 'planta':
            self.knowledge.add(self.hielo)
            self.knowledge.add(self.bicho)
            self.knowledge.add(self.fuego)
            self.knowledge.add(self.veneno)
            self.knowledge.add(self.volador)

        if poke_objetivo[3] == 'psiquico' or poke_objetivo[4] == 'psiquico':
            self.knowledge.add(self.fantasma)
            self.knowledge.add(self.siniestro)
            self.knowledge.add(self.bicho)
        
        if poke_objetivo[3] == 'roca' or poke_objetivo[4] == 'roca':
            self.knowledge.add(self.agua)
            self.knowledge.add(self.acero)
            self.knowledge.add(self.planta)
            self.knowledge.add(self.lucha)
            self.knowledge.add(self.tierra)

        if poke_objetivo[3] == 'siniestro' or poke_objetivo[4] == 'siniestro':
            self.knowledge.add(self.hada)
            self.knowledge.add(self.bicho)
            self.knowledge.add(self.lucha)
  
        if poke_objetivo[3] == 'tierra' or poke_objetivo[4] == 'tierra':
            self.knowledge.add(self.agua)
            self.knowledge.add(self.planta)
            self.knowledge.add(self.hielo)

        if poke_objetivo[3] == 'veneno' or poke_objetivo[4] == 'veneno':
            self.knowledge.add(self.psiquico)
            self.knowledge.add(self.tierra)

        if poke_objetivo[3] == 'volador' or poke_objetivo[4] == 'volador':
            self.knowledge.add(self.hielo)
            self.knowledge.add(self.electrico)
            self.knowledge.add(self.roca)
        
        candidatos = []
        for tipo in self.tipos:
            if model_check(self.knowledge, tipo):
                candidatos.append(tipo)

        if poke_objetivo[4] is not None:
            if poke_objetivo[3] == 'acero' or poke_objetivo[4] == 'acero':
                if self.hada in candidatos:
                    candidatos.remove(self.hada)
                if self.hielo in candidatos:
                    candidatos.remove(self.hielo)
                if self.roca in candidatos:
                    candidatos.remove(self.roca)
                if self.acero in candidatos:
                    candidatos.remove(self.acero)

            if poke_objetivo[3] == 'agua' or poke_objetivo[4] == 'agua':
                if self.agua in candidatos:
                    candidatos.remove(self.agua)
                if self.fuego in candidatos:
                    candidatos.remove(self.fuego)
                if self.roca in candidatos:
                    candidatos.remove(self.roca)
                if self.tierra in candidatos:
                    candidatos.remove(self.tierra)

            if poke_objetivo[3] == 'bicho' or poke_objetivo[4] == 'bicho':
                if self.bicho in candidatos:
                    candidatos.remove(self.bicho)
                if self.planta in candidatos:
                    candidatos.remove(self.planta)
                if self.psiquico in candidatos:
                    candidatos.remove(self.psiquico)
                if self.siniestro in candidatos:
                    candidatos.remove(self.siniestro)

            if poke_objetivo[3] == 'electrico' or poke_objetivo[4] == 'electrico':
                if self.electrico in candidatos:
                    candidatos.remove(self.electrico)
                if self.agua in candidatos:
                    candidatos.remove(self.agua)
                if self.volador in candidatos:
                    candidatos.remove(self.volador)

            if poke_objetivo[3] == 'fantasma' or poke_objetivo[4] == 'fantasma':
                if self.psiquico in candidatos:
                    candidatos.remove(self.psiquico)

            if poke_objetivo[3] == 'fuego' or poke_objetivo[4] == 'fuego':
                if self.fuego in candidatos:
                    candidatos.remove(self.fuego)
                if self.acero in candidatos:
                    candidatos.remove(self.acero)
                if self.bicho in candidatos:
                    candidatos.remove(self.bicho)
                if self.hielo in candidatos:
                    candidatos.remove(self.hielo)
                if self.planta in candidatos:
                    candidatos.remove(self.planta)

            if poke_objetivo[3] == 'hada' or poke_objetivo[4] == 'hada':
                if self.hada in candidatos:
                    candidatos.remove(self.hada)
                if self.dragon in candidatos:
                    candidatos.remove(self.dragon)
                if self.lucha in candidatos:
                    candidatos.remove(self.lucha)
                if self.siniestro in candidatos:
                    candidatos.remove(self.siniestro)

            if poke_objetivo[3] == 'hielo' or poke_objetivo[4] == 'hielo':
                if self.hielo in candidatos:
                    candidatos.remove(self.hielo)
                if self.dragon in candidatos:
                    candidatos.remove(self.dragon)
                if self.planta in candidatos:
                    candidatos.remove(self.planta)
                if self.tierra in candidatos:
                    candidatos.remove(self.tierra)
                if self.volador in candidatos:
                    candidatos.remove(self.volador)

            if poke_objetivo[3] == 'lucha' or poke_objetivo[4] == 'lucha':
                if self.lucha in candidatos:
                    candidatos.remove(self.lucha)
                if self.normal in candidatos:
                    candidatos.remove(self.normal)
                if self.siniestro in candidatos:
                    candidatos.remove(self.siniestro)
                if self.roca in candidatos:
                    candidatos.remove(self.roca)
                if self.hielo in candidatos:
                    candidatos.remove(self.hielo)
                if self.acero in candidatos:
                    candidatos.remove(self.acero)

            if poke_objetivo[3] == 'planta' or poke_objetivo[4] == 'planta':
                if self.agua in candidatos:
                    candidatos.remove(self.agua)
                if self.roca in candidatos:
                    candidatos.remove(self.roca)
                if self.tierra in candidatos:
                    candidatos.remove(self.tierra)
                if self.planta in candidatos:
                    candidatos.remove(self.planta)

            if poke_objetivo[3] == 'psiquico' or poke_objetivo[4] == 'psiquico':
                if self.psiquico in candidatos:
                    candidatos.remove(self.psiquico)
                if self.lucha in candidatos:
                    candidatos.remove(self.lucha)
                if self.veneno in candidatos:
                    candidatos.remove(self.veneno)
            
            if poke_objetivo[3] == 'roca' or poke_objetivo[4] == 'roca':
                if self.roca in candidatos:
                    candidatos.remove(self.roca)
                if self.bicho in candidatos:
                    candidatos.remove(self.bicho)
                if self.fuego in candidatos:
                    candidatos.remove(self.fuego)
                if self.hielo in candidatos:
                    candidatos.remove(self.hielo)
                if self.volador in candidatos:
                    candidatos.remove(self.volador)


            if poke_objetivo[3] == 'siniestro' or poke_objetivo[4] == 'siniestro':
                if self.siniestro in candidatos:
                    candidatos.remove(self.siniestro)
                if self.fantasma in candidatos:
                    candidatos.remove(self.fantasma)
                if self.psiquico in candidatos:
                    candidatos.remove(self.psiquico)
    
            if poke_objetivo[3] == 'tierra' or poke_objetivo[4] == 'tierra':
                if self.tierra in candidatos:
                    candidatos.remove(self.tierra)
                if self.acero in candidatos:
                    candidatos.remove(self.acero)
                if self.electrico in candidatos:
                    candidatos.remove(self.electrico)
                if self.fuego in candidatos:
                    candidatos.remove(self.fuego)
                if self.roca in candidatos:
                    candidatos.remove(self.roca)
                if self.veneno in candidatos:
                    candidatos.remove(self.veneno)

            if poke_objetivo[3] == 'veneno' or poke_objetivo[4] == 'veneno':
                if self.veneno in candidatos:
                    candidatos.remove(self.veneno)
                if self.hada in candidatos:
                    candidatos.remove(self.hada)
                if self.planta in candidatos:
                    candidatos.remove(self.planta)

            if poke_objetivo[3] == 'volador' or poke_objetivo[4] == 'volador':
                if self.volador in candidatos:
                    candidatos.remove(self.volador)
                if self.bicho in candidatos:
                    candidatos.remove(self.bicho)
                if self.lucha in candidatos:
                    candidatos.remove(self.lucha)
                if self.planta in candidatos:
                    candidatos.remove(self.planta)
            candidatos.append(None)
        return candidatos

if __name__ == "__main__":
    # Crear una representación de la Red Bayesiana como un objeto DiGraph
    red_bayesiana = nx.DiGraph()

    # Agregamos nodos a la representación de la red
    for node in modelo.states:
        red_bayesiana.add_node(node.name)

    # Agregamos bordes a la representación de la red
    for edge in modelo.edges:
        red_bayesiana.add_edge(edge[0].name, edge[1].name)

    # Visualizar la estructura de la red
    pos = nx.spring_layout(red_bayesiana, seed=42)
    labels = {node: node for node in red_bayesiana.nodes()}
    nx.draw(red_bayesiana, pos, labels=labels, with_labels=True,
            node_size=5000, node_color="cyan")
    plt.title("Estructura de la Red Bayesiana")
    plt.show()
