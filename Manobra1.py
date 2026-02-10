from Orbitas import *
from Manobras import *

a = np.pi - angulo_manobra(Orbita_Nave,Orbita_Intermediaria,1)
b = np.pi - angulo_manobra(Orbita_Intermediaria,Orbita_Nave, 1)
print()
print(f"Anomalia verdadeira de realização da manobra: {a:.4f} (rad), Anomalia verdadeira em que chegará na orbita intermediária: {b:.4f} (rad)")
print()
v1 =velocidade_ponto(Orbita_Nave, a)
v2 =velocidade_ponto(Orbita_Intermediaria, b)
dV = v2 - v1

print(f"Vetor Impulso necessário para trocar da órbita da nave para a intermediaria: {dV[0][0]:.4f}, {dV[1][0]:.4f}, {dV[2][0]:.4f} (km/s)") #Velocidade de impulso necessária para troca de orbita
print()

#print(plane_rotation(Orbita_Nave,Orbita_Intermediaria)) #Faz uma análise vetorial da manobra, trocando da orbita inicial pra final.
t =  42 #segundos
erro = 1 #kms
R = randevouz(Orbita_Intermediaria, Orbita_Detrito, erro, t)
Angulo = R[3]
print(f"Para um rendezvous em {t} segundos, assumindo uma margem de {erro} kms para a aproximação:")
print(f"Vetor Impulso necessário: {R[1]} km/s")
print()

V_Detrito_Imp = np.ravel(velocidade_ponto(Orbita_Detrito, Angulo)) * 1000
V_Nave_Imp = V_Detrito_Imp + np.ravel(R[1]) * 1000


# Quantia de movimento = Massa * velocidade * 1000
QD = Orbita_Detrito[5]* V_Detrito_Imp
QV = Orbita_Nave[5] * V_Nave_Imp

V = ((QD+QV)/(Orbita_Detrito[5] + Orbita_Nave[5]))/1000 # Deixando em km/s
P = posicao_xyz(Orbita_Detrito, Angulo)
print(P)
OrbitaFinal = Conversao_XYZ_Orbita(V, P)[1]
print(f"""Parâmetros da órbita final após o impacto:
        a (semi-eixo maior): {OrbitaFinal[0]},
        e (excentricidade): {OrbitaFinal[1]},
        i (inclinação) [deg]: {OrbitaFinal[3]},
        Ω (longitude do nodo) [deg]: {OrbitaFinal[2]},
        ω (arg. do periastro) [deg]: {OrbitaFinal[4]}""")