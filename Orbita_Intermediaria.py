from Manobras import *
from Orbitas import *

#Este código serve para descobrir qual a orbita que conecta a orbita inicial a final. Não define quais as manobras necessáiras,
# Mas determina os parametros orbitais. Se mantem o RAAN e a Inclinação da inicial, mudando apenas o argumento do perigeu,
# excentricidade e 

if __name__ == "__main__":
        #Obtenção da orbita intermediaria
    DA = angulo_manobra(Orbita_Nave,Orbita_Detrito, 0)
    Ra = posicao_orbita(Orbita_Detrito,angulo_manobra(Orbita_Detrito,Orbita_Nave, 0))
    Rp = posicao_orbita(Orbita_Nave,angulo_manobra(Orbita_Nave,Orbita_Detrito))
    print(Ra)
    e = (Ra - Rp)/(Ra + Rp)
    Periapse_angle = DA + Orbita_Nave[4]
    if(Periapse_angle > 2*np.pi):
        Periapse_angle -=2*np.pi
    a = Rp/(1-e)

    print(f"Exc {e}, Semi-Eixo-Maior {a}, Argumento do perigeu {Periapse_angle}")
