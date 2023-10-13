import pandas as pd
from PokeWinner import *

# Cargar la base de datos de Pokémon desde un archivo CSV
name_dateset = "ProyectoFinal//dataset_pokemons.csv"
df_pokemon = pd.read_csv(name_dateset)

# Especificar el ID del Pokémon que deseas obtener
id_pokemon_deseado = 12 #int(input('Ingrese ID de Pokedex:')) # Por ejemplo, Pikachu es 25
n_candidatos = 10 #int(input('Ingrese ID de Pokedex:'))
# Comprobar si se encontró el Pokémon
poke_objetivo = df_pokemon[df_pokemon['ID'] == id_pokemon_deseado].values.tolist()[0]


print_poke(poke_objetivo)

filtrado_logico = poke_logic()
tipos = filtrado_logico.tipos_winner(poke_objetivo)
print(f'\nEl PokeWinner debe ser de almenos uno los siguientes tipos: \n{tipos[:-1]}\n')

poke_winner = poke_winner(name_dateset)
candidatos = poke_winner.buscar(poke_objetivo, tipos, n_candidatos)


columnas = ['ID', 'Nombre', 'Tipo1', 'Tipo2']
df_temp = pd.DataFrame(columns=columnas)
for ID in candidatos:
    PokeWinner = df_pokemon[df_pokemon['ID'] == ID].values.tolist()[0]
    nueva_fila = {'ID': PokeWinner[0], 'Nombre': PokeWinner[1], 'Tipo1': PokeWinner[3], 'Tipo2': PokeWinner[4]}
    df_temp = df_temp._append(nueva_fila, ignore_index=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df_temp)

P_candidatos = inferencia(poke_winner.com_est(poke_objetivo, candidatos))
best = mejores(candidatos, P_candidatos, 3)
print()
for opciones in best:
    PokeWinner_WIN = [df_pokemon[df_pokemon['ID'] == int(opciones[0])].values.tolist()[0][1] , float(opciones[1])]
    print(f"El PokeWinner elegido es {PokeWinner_WIN[0]} con una probabilidad de {PokeWinner_WIN[1]}")
