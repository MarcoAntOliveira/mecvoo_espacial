from Orbitas import *
from Manobras import *

interseccoes = encontrar_intersecoes_xy(Orbita_Intermediaria, Orbita_Detrito, 10000)
plot_distances_from_intersecoes(interseccoes)