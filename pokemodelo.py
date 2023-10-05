from pomegranate import *
import networkx as nx
import matplotlib.pyplot as plt


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

    [1,1,1, 0.9],
    [1,1,0, 0.1],

    [1,0,1, 0.6],
    [1,0,0, 0.4],

    [0,1,1, 0.4],
    [0,1,0, 0.6],

    [0,0,1, 0.1],
    [0,0,0, 0.9],

], [atk.distribution, spatk.distribution]), name="atk Total")

def_T = Node(ConditionalProbabilityTable([

    [1,1,1, 0.9],
    [1,1,0, 0.1],

    [1,0,1, 0.6],
    [1,0,0, 0.4],

    [0,1,1, 0.4],
    [0,1,0, 0.6],

    [0,0,1, 0.1],
    [0,0,0, 0.9],

], [defensa.distribution, spdef.distribution]), name="def Total")

plus = Node(ConditionalProbabilityTable([

    [1,1,1, 0.95],
    [1,1,0, 0.05],

    [1,0,1, 0.2],
    [1,0,0, 0.8],

    [0,1,1, 0.6],
    [0,1,0, 0.4],

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

def inferencia(comparativa):

    C1 = {'atk': comparativa[0][0], 'def': comparativa[0][1], 'spatk': comparativa[0][2], 'spdef': comparativa[0][3], 'speed': comparativa[0][4], 'total': comparativa[0][5]}
    C2 = {'atk': comparativa[1][0], 'def': comparativa[1][1], 'spatk': comparativa[1][2], 'spdef': comparativa[1][3], 'speed': comparativa[1][4], 'total': comparativa[1][5]}
    C3 = {'atk': comparativa[2][0], 'def': comparativa[2][1], 'spatk': comparativa[2][2], 'spdef': comparativa[2][3], 'speed': comparativa[2][4], 'total': comparativa[2][5]}

    predit_R1 = modelo.predict_proba(C1)
    predit_R2 = modelo.predict_proba(C2)
    predit_R3 = modelo.predict_proba(C3)

    p1 = predit_R1[-1].parameters[0][1]
    p2 = predit_R2[-1].parameters[0][1]
    p3 = predit_R3[-1].parameters[0][1]

    return [p1,p2,p3]


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
