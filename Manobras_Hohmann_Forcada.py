from Orbitas import *
from Manobras import *

#Calculo dos dVs e angulos para sair da orbita inicial e chegar na final

# A Manobra é planar, e, portanto, pode ser calculada pelo método da rotação de linha de apse, ou vetorialmente
DV = apseline_rotation(Orbita_Nave, Orbita_Intermediaria2)
print()
print(f"Módulo do impulso necessário e angulo em relação a orbita: {DV[0]:.4f}[km/s] {DV[1]:.4f}[rad]")
print()

DV = velocidade_ponto(Orbita_Intermediaria2, 0) - velocidade_ponto(Orbita_Nave, angulo_manobra(Orbita_Nave,Orbita_Detrito))
print(f"Vetor Impulso necessário para troca entre as orbitas: {DV[0][0]:.4f}, {DV[1][0]:.4f}, {DV[2][0]:.4f} [km/s]")
print()

#print(angulo_manobra(Orbita_Intermediaria2, Orbita_Detrito, 1), angulo_manobra(Orbita_Detrito, Orbita_Intermediaria2, 0))
DV2 = plane_rotation(Orbita_Intermediaria2, Orbita_Detrito, 1, 0)
#print(velocidade_ponto(Orbita_Detrito, angulo_manobra(Orbita_Detrito, Orbita_Intermediaria2, 0)))
print(f"Vetor impulso necessário para troca de plano da orbita, com ajuste para órbita final: {DV2[0][0]:.4f}, {DV2[1][0]:.4f}, {DV2[2][0]:.4f} [km/s]")
print()

T = periodo(Orbita_Detrito)
Phasing = phasing_manuever(Orbita_Detrito, T/2, 17)
print(f"A manobra de phasing é dada por: {Phasing[0][0][0]}, {Phasing[0][1][0]}, {Phasing[0][2][0]} [km/s],  {Phasing[1]} [km/s]")
print()
v1 = velocidade_ponto(Orbita_Detrito, np.pi)
v2 = v1 + Phasing[0]

QD = Orbita_Detrito[5]* v1 * 1000
QV = Orbita_Nave[5] * v2 * 1000

V = (QD+QV)/(Orbita_Detrito[5] + Orbita_Nave[5])/1000
P = posicao_xyz(Orbita_Detrito, np.pi)



print(f"Posição de impacto: {P[0][0]:.4f}, {P[0][1]:.4f}, {P[0][2]:.4f} [km]")
print()
print(f"velocidade pós impacto {V[0][0]:.4f}, {V[1][0]:.4f}, {V[2][0]:.4f} [km/s]")
VP = velocidade_ponto(Orbita_Detrito, np.pi)
print(f"velocidade pré impacto {VP[0][0]:.4F}, {VP[1][0]:.4F}, {VP[2][0]:.4F} [km/s]")

OrbitaFinal = Conversao_XYZ_Orbita(V, P)[1]
print(f"""Parâmetros da órbita final após o impacto:
        a (semi-eixo maior): {OrbitaFinal[0]},
        e (excentricidade): {OrbitaFinal[1]},
        i (inclinação) [deg]: {OrbitaFinal[3]},
        Ω (longitude do nodo) [deg]: {OrbitaFinal[2]},
        ω (arg. do periastro) [deg]: {OrbitaFinal[4]}""")

#Adicionar um comparativo entre o método atual: Mudança de plano seguida de mudança de fase
# método do prof: Mudança de fase + mudança de plano simultaneamente.\

#Colocar uma observação para trabalhos posteriores da variação dos elementos orbitais por conta da deriva e um possível rendezvous