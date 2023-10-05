import pandas as pd
from PokeWinner import *
from pokemodelo import inferencia

# Cargar la base de datos de Pokémon desde un archivo CSV
name_dateset = "ProyectoFinal//dataset_pokemons.csv"
df_pokemon = pd.read_csv(name_dateset)

# Especificar el ID del Pokémon que deseas obtener
id_pokemon_deseado = 25#int(input('Ingrese ID de Pokedex:')) # Por ejemplo, Pikachu es 25

# Comprobar si se encontró el Pokémon
poke_objetivo = df_pokemon[df_pokemon['ID'] == id_pokemon_deseado].values.tolist()[0]



print(f"El pokemon a derrotar es {poke_objetivo[1]}")
filtrado_logico = poke_logic()
tipos = filtrado_logico.tipos_winner(poke_objetivo)
print(f'El PokeWinner debe ser de almenos los siguientes tipos:\n\n{tipos[:-1]}\n')

poke_winner = poke_winner(name_dateset)
candidatos = poke_winner.buscar(poke_objetivo, tipos)
cont = 1
for ID in candidatos:
    PokeWinner = df_pokemon[df_pokemon['ID'] == ID].values.tolist()[0]
    print(f'{cont}. {PokeWinner[1]} de tipo {PokeWinner[3]} y {PokeWinner[4]}')
    cont+= 1
print()
pro = poke_winner.com_est(poke_objetivo, candidatos)

probabilidades = inferencia(pro)

# Obtiene la ruta que tiene la mayor probabilidad de llegar a tiempo
PokeWinner_WIN = [df_pokemon[df_pokemon['ID'] == candidatos[probabilidades.index(max(probabilidades))]].values.tolist()[0][1] , max(probabilidades)]
#print(probabilidades)
print(f"El PokeWinner elegido es {PokeWinner_WIN[0]} con una probabilidad de {PokeWinner_WIN[1]}")


