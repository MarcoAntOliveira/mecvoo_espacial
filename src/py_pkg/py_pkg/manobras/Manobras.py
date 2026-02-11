import numpy as np
import math
import matplotlib.pyplot as plt


def posicao_xyz(Orbita, f):
    """
    Dado os inputs: Orbita e Anomalia Verdadeira
    Obtemos: Posição espacial do satélite em sua órbita
    """

    a = Orbita[0]
    e = Orbita[1]
    RAAN = Orbita[2]
    Inclinacao = Orbita[3]
    argumentoperiapse = Orbita[4]

    r = posicao_orbita(Orbita, f)

    M1 = [[math.cos(RAAN), -math.sin(RAAN), 0],
          [math.sin(RAAN), math.cos(RAAN), 0],
          [0, 0, 1]]
    
    M2 = [[1, 0, 0],
          [0, math.cos(Inclinacao), -math.sin(Inclinacao)],
          [0, math.sin(Inclinacao), math.cos(Inclinacao)]]
    
    M3 = [[math.cos(argumentoperiapse), -math.sin(argumentoperiapse), 0],
          [math.sin(argumentoperiapse), math.cos(argumentoperiapse), 0],
          [0, 0, 1]]
    
    M4 = [[r * math.cos(f)],
          [r * math.sin(f)],
          [0]]
    
    pos = np.matmul(np.matmul(np.matmul(M1,M2),M3),M4)
    pos_out = np.transpose(pos)
    return pos_out

def posicao_orbita(Orbita, f):

    """
    Dado os inputs: Orbita e Anomalia Verdadeira
    Obtemos: A altitude do corpo na sua órbita para a posição pedida
    """

    a = Orbita[0]
    e = Orbita[1]

    r = a*(1-e**2)/(1+e*math.cos(f))
    return r

def vetor_normal(Orbita):


    """
    Dado os inputs: Orbita de interesse
    Obtemos: O vetor normal a orbita, útil para descobrir outros vetores que podem ser usados
    """


    a = Orbita[0]
    e = Orbita[1]
    RAAN = Orbita[2]
    Inclinacao = Orbita[3]
    argumentoperiapse = Orbita[4]
    V1 = posicao_xyz(Orbita, 0)
    V2 = posicao_xyz(Orbita, math.pi/2)

    N = np.cross(V1,V2)
    n = N / np.linalg.norm(N)

    return n

def vetor_intersect(Orbita1, Orbita2):


    """
    Dado os inputs: Orbitas 1 e 2
    Obtemos: Um vetor de intersecção entre os planos das órbitas. Usado para saber quando realizar alguma manobra.
    """


    n = vetor_normal(Orbita1)
    e = vetor_normal(Orbita2)
    V = np.cross(n, e)
    v1 = V / np.linalg.norm(V)
    v2 = -v1
    Output = [v1,v2]
    return Output


#Precisa corrigir esta função
def angulo_manobra(Orbita1, Orbita2, Ponto = 0):


    """
    Dado os inputs: Orbita Inicial, Orbita Final e em qual ponto de intersecção dos planos deseja realizar a manobra 
    Obtemos: Qual o angulo, em cada orbita, que esta manobra deve ser realizada. Descobrindo as anomalias verdadeiras do impulso.
    """


    vetor_perigeu = posicao_xyz(Orbita1, 0)
    vetor_perigeu_normalizado = vetor_perigeu/np.linalg.norm(vetor_perigeu)
    vetor_intersecao_temp = vetor_intersect(Orbita1, Orbita2)
    vetor_intersecao = vetor_intersecao_temp[Ponto]

    normal = np.cross(vetor_normal(Orbita1)[0], vetor_normal(Orbita2)[0])

    seno = np.dot(normal,(np.cross(vetor_perigeu_normalizado[0],vetor_intersecao[0])))
    angulo_cos  = np.acos(np.dot(vetor_perigeu_normalizado[0],vetor_intersecao[0]))
    angulo = angulo_cos
    #angulo = np.arctan2(angulo_sin,np.dot(vetor_perigeu_normalizado[0],vetor_intersecao[0]))

    if(seno < 0):
        if(angulo_cos > np.pi/2):
            angulo = 2*np.pi - angulo_cos
        else:
            angulo = - angulo_cos
        
    if(Orbita1[4] < np.pi):
        angulo = angulo - np.pi #Coloquei um teste para sempre rodar
    return angulo

def h(Orbita):


    """
    Dado os inputs: Orbita
    Obtemos: Momento Angular
    """


    r = posicao_orbita(Orbita, 0)
    h = math.sqrt(r*Mi*(1+Orbita[1]*math.cos(0)))
    return h

def apseline_rotation(Orbita1, Orbita2):


    """
    Dado os inputs: Orbita inicial e Orbita Finak
    Obtemos: O Impulso necessário e onde ele deve ser realizado na orbita inicial.
    """


    Eta = Orbita2[4] - Orbita1[4]
    #Eta = 25*math.pi/180
    h1 = h(Orbita1)
    h2 = h(Orbita2)
    a = Orbita1[1]*(h2**2) - Orbita2[1]*(h1**2)*math.cos(Eta)
    b = -Orbita2[1]*(h1**2)*math.sin(Eta)
    c = h1**2 - h2**2
    phi = math.atan(b/a)
    theta1 = phi + math.acos(c*math.cos(phi)/a)
    theta2 = phi - math.acos(c*math.cos(phi)/a)
    theta = [theta1, theta2]

    
    r = posicao_orbita(Orbita1, theta1)

    vt1 = h1/r
    vr1 = Mi/h1*Orbita1[1]*math.sin(theta1)
    gamma1 = math.atan(vr1/vt1)
    v1 = math.sqrt(vr1**2 + vt1**2)

    vt2 = h2/r
    vr2 = Mi/h2 *Orbita2[1]*math.sin(theta1-Eta)
    gamma2 = math.atan(vr2/vt2)
    v2 = math.sqrt(vr2**2 + vt2**2)

    dV = math.sqrt(v1**2 + v2**2 - 2*v1*v2*math.cos(gamma2-gamma1))
    angle = math.atan((vr2-vr1)/(vt2-vt1))
    Manuever = [dV, angle]

    return Manuever

def velocidade_ponto(Orbita, f):


    """
    Dado os inputs: Orbita e Anomalia Veradeira
    Obtemos: Velocidade em coordenadas carterianas do satélite neste ponto
    """


    a = Orbita[0]
    e = Orbita[1]
    RAAN = Orbita[2]
    Inclinacao = Orbita[3]
    argumentoperiapse = Orbita[4]

    Rot = [[math.cos(RAAN)*math.cos(argumentoperiapse) - math.sin(RAAN)*math.cos(Inclinacao)*math.sin(argumentoperiapse), -math.sin(RAAN)*math.cos(Inclinacao)*math.cos(argumentoperiapse) - math.cos(RAAN)*math.sin(argumentoperiapse), math.sin(RAAN)*math.sin(Inclinacao)],
           [math.sin(RAAN)*math.cos(argumentoperiapse) + math.cos(RAAN)*math.cos(Inclinacao)*math.sin(argumentoperiapse), math.cos(RAAN)*math.cos(Inclinacao)*math.cos(argumentoperiapse) - math.sin(RAAN)*math.sin(argumentoperiapse), -math.cos(RAAN)*math.sin(Inclinacao)],
           [math.sin(Inclinacao)*math.sin(argumentoperiapse), math.sin(Inclinacao)*math.cos(argumentoperiapse), math.cos(Inclinacao)]]
    
    Vel = [[-math.sin(f)],
           [e + math.cos(f)],
           [0]]
    
    Resultado = np.matmul(Rot,Vel) * Mi / h(Orbita)
    return Resultado

def encontrar_intersecoes_xy(orbita1, orbita2, tolerancia=500,  passo=0.5):
    """
    Encontra interseções espaciais entre duas órbitas (mesmo plano), mesmo com diferentes orientações.

    Retorna:
        Lista de tuplas: (x, y, theta1, theta2)
    """
    intersecoes = []

    # angulos1 = np.radians(np.arange(170, 200, passo))
    # angulos20 = np.radians(np.arange(0, 0, passo))
    # angulos230 = np.radians(np.arange(350, 365, passo))
    # angulos2 = np.append(angulos20,angulos230)
    angulos1 = np.radians(np.arange(0, 360, passo))
    angulos20 = np.radians(np.arange(0, 0, passo))
    angulos230 = np.radians(np.arange(180, 540, passo))
    angulos2 = np.append(angulos20,angulos230)


    for theta1 in angulos1:
        p1 = posicao_xyz(orbita1, theta1)

        for theta2 in angulos2:
            p2 = posicao_xyz(orbita2, theta2)

            dist = np.linalg.norm(p1 - p2), theta1, theta2
            if dist[0] < tolerancia:
                intersecoes.append((p1[0,0], p1[0,1], np.degrees(theta1), np.degrees(theta2), np.linalg.norm(p1 - p2)))
    return intersecoes

def plot_distances_from_intersecoes(intersecoes):
    """
    Gera:
    - Heatmap 2D com curvas de nível
    - Plot distância vs. θ₁
    - Plot distância vs. θ₂

    Cada item de `intersecoes` deve ser (x, y, theta1, theta2, dR).
    `step` é o passo em graus para reconstrução da malha.
    """
    theta1_list = [i[2] for i in intersecoes]
    theta2_list = [i[3] for i in intersecoes]
    distances = [i[4] for i in intersecoes]

    # Encontrar ponto de menor distância
    min_idx = np.argmin(distances)
    min_theta1 = theta1_list[min_idx]
    min_theta2 = theta2_list[min_idx]
    min_dist = distances[min_idx]

    # ---- Plot 1: Heatmap com curvas de nível ----
    theta1_unique = np.unique(np.round(theta1_list, 8))
    theta2_unique = np.unique(np.round(theta2_list, 8))

    if len(theta1_unique) == 0 or len(theta2_unique) == 0:
        print("Erro: lista de ângulos vazia. Verifique se há interseções suficientes.")
        return

    grid_shape = (len(theta2_unique), len(theta1_unique))
    dist_grid = np.full(grid_shape, np.nan)

    for t1, t2, d in zip(theta1_list, theta2_list, distances):
        j = np.searchsorted(theta1_unique, round(t1, 8))
        i = np.searchsorted(theta2_unique, round(t2, 8))
        if 0 <= i < grid_shape[0] and 0 <= j < grid_shape[1]:
            dist_grid[i, j] = d

    T1, T2 = np.meshgrid(theta1_unique, theta2_unique)

    plt.figure(figsize=(10, 7))
    im = plt.imshow(dist_grid,
                    extent=[theta1_unique[0], theta1_unique[-1], theta2_unique[0], theta2_unique[-1]],
                    origin='lower', cmap='plasma', aspect='auto')

    contour = plt.contour(T1, T2, dist_grid, levels=10, colors='black', linewidths=0.2)
    plt.clabel(contour, inline=True, fontsize=8, fmt="%.1f")

    # Marcador de distância mínima
    plt.plot(min_theta1, min_theta2, 'bo', label='Distância mínima')
    plt.legend()

    plt.xlabel("θ₁ (graus)")
    plt.ylabel("θ₂ (graus)")
    plt.title("Distância entre pontos das órbitas")
    cbar = plt.colorbar(im)
    cbar.set_label("Distância (km)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---- Plot 2: Distância vs. θ₁ ----
    plt.figure(figsize=(8, 4))
    plt.scatter(theta1_list, distances, s=10, alpha=0.7, label='Pontos')
    plt.plot(min_theta1, min_dist, 'ro', label='Distância mínima')
    plt.xlabel("θ₁ (graus)")
    plt.ylabel("Distância (km)")
    plt.title("Distância vs. θ₁")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Plot 3: Distância vs. θ₂ ----
    plt.figure(figsize=(8, 4))
    plt.scatter(theta2_list, distances, s=10, alpha=0.7, label='Pontos')
    plt.plot(min_theta2, min_dist, 'ro', label='Distância mínima')
    plt.xlabel("θ₂ (graus)")
    plt.ylabel("Distância (km)")
    plt.title("Distância vs. θ₂")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plane_rotation(Orbita1, Orbita2, Ponto1 = 0, Ponto2 = 0):

    
    """
    Dado os inputs: Orbita inicial, Orbita Final, ponto de manobra na orbita inicial e ponto de chegada na orbita final
    Obtemos: O Impulso necessário para realização da manobra.
    """


    f1 = angulo_manobra(Orbita1, Orbita2, Ponto1)
    f2 = angulo_manobra(Orbita2, Orbita1, Ponto2)
    v1 = velocidade_ponto(Orbita1, f1)
    v2 = velocidade_ponto(Orbita2, f2)
    manuever = v2 - v1
    
    return manuever

def angulo_voo(Orbita, f):


    """
    Dado os inputs: Orbita e anomalia verdadeira
    Obtemos: O Angulo de voo neste ponto.
    """


    angulo = np.atan((Orbita[1]*np.sin(f))/(1+(Orbita[1]*np.cos(f))))
    return angulo

def randevouz(Orbita1, Orbita2, Distancia, t): #Só vale para circulares, porém como a excentricidade é pequena, e o delta Angulo é pequeno, podemos assumir que as variações são pequenas.
    """
    Obtém a velocidade para rendezvous a partir das orbitas,
    Descobre o ponto de intersecção por iteração e utiliza ele para posição e velocidade de impacto
    Retorna os valores de:
    - Velocidade de impulso em módulo
    - Vetor Velocidade d Impulso
    - Anomalia Verdadeira para impacto na orbita 1
    - Anomalia Verdadeira para impacto na orbita 2
    """

    intersecao = encontrar_intersecoes_xy(Orbita1, Orbita2, Distancia)
    intersecao = intersecao[0]
    #print(intersecao)
    f1 = intersecao[2] * np.pi/180
    f2 = intersecao[3] * np.pi/180
    #print(f"theta1 {f1}, theta2 {f2}")
    dr = intersecao[4]


    #Posicoes no espaço geocentrico dos corpos de estudo
    XYZ1 = posicao_xyz(Orbita1, f1)
    XYZ2 = posicao_xyz(Orbita2, f2)
    dXYZ = XYZ1 - XYZ2
    dXYZmod = np.linalg.norm(dXYZ)

    #Velocidades dos corpos no espaço geocentrico
    V1 = np.transpose(velocidade_ponto(Orbita1, f1))
    V2 = np.transpose(velocidade_ponto(Orbita2, f2))
    dV = V1-V2
    #Criação dos vetores diretores do referencial detritocentrico
    vH1 = np.cross(XYZ1[0], V1[0])
    vH2 = np.cross(XYZ2[0], V2[0])
    k1 = vH1 / np.linalg.norm(vH1)
    k2 = vH2 / np.linalg.norm(vH2)

    i1 = XYZ1 / np.linalg.norm(XYZ1)
    i2 = XYZ2 / np.linalg.norm(XYZ2)

    J1 = np.cross(k1, i1)
    J2 = np.cross(k2, i2)

    j1 = J1 / np.linalg.norm(J1)
    j2 = J2 / np.linalg.norm(J2)

    #Distancia entre corpos no referencial detritocentrico
    dR = np.zeros(3)
    dR[0] = np.dot(dXYZ[0], i2[0])
    dR[1] = np.dot(dXYZ[0], j2[0])
    dR[2] = np.dot(dXYZ[0], k2)

    Omega2 = vH2 / ((np.linalg.norm(XYZ2))**2) 

    Vrel = V1-V2-np.cross(Omega2,dXYZ)

    Q = np.zeros((3,3))
    Q[0] = i2
    Q[1] = j2
    Q[2] = k2
    
    rx0 = np.matmul(Q, dXYZ[0])
    vx0 = np.matmul(Q, dV[0])
    vx0_mod = np.linalg.norm(vx0)


    n = np.linalg.norm(V2) / np.linalg.norm(XYZ2)


    #Matrizes para solução
    phirr = [[4-3*np.cos(n*t), 0, 0],
            [6*(np.sin(n*t)-n*t), 1, 0],
            [0, 0, 1],]
    
    phirv = [[1/n*np.sin(n*t), 2*(1-(np.cos(n*t)))/n, 0],
             [2/n*(np.cos(n*t)-1), 1/n*(4*np.sin(n*t)-3*n*t), 0],
             [0, 0, 1/n*np.sin(n*t)]]

    phivr = [[3*n*np.sin(n*t), 0, 0],
             [6*n*(np.cos(n*t)-1), 0, 0],
             [0, 0, -n*np.sin(n*t)]]
    
    phivv = [[np.cos(n*t), 2*np.cos(n*t), 0],
             [-2*np.sin(n*t), 4*np.cos(n*t)-3, 0],
             [0, 0, np.cos(n*t)]]
    
    phirv_inversa = np.linalg.inv(phirv)

    rt = [0,0,0]
    mint = np.matmul(phirr, rx0)
    v0_manuever = np.matmul(phirv_inversa, rt) - np.matmul(phirv_inversa, np.matmul(phirr, rx0))
    dVmanuever = v0_manuever - vx0

    vt = np.matmul(phivr,rx0) + np.matmul(phivv, v0_manuever)
    vt = np.array(vt)

    return np.linalg.norm(dVmanuever), vt, f1, f2

def phasing_manuever(Orbita1, Tempo, N_Orbits = 1,pos = np.pi):
    """
    Retorna a velocidade de impacto para uma correção de fase entre dois objetos na mesma órbita
    """
    T1 = periodo(Orbita1)
    T2 = T1 - Tempo/N_Orbits
    a = float((T2 * np.sqrt(Mi) / (2* np.pi))**float((2/3)))

    ra = posicao_orbita(Orbita1, np.pi)
    rp = 2*a - ra
    e = ra/a -1
    Orbita2 = Orbita1.copy()
    Orbita2[0] = a
    Orbita2[1] = e
    v1 = velocidade_ponto(Orbita1, pos)
    v2 = velocidade_ponto(Orbita2, pos)
    V_Impulso = v2 - v1 #Velocidade de impulso é identica a velocidade de impacto
    V_Abs = np.linalg.norm(V_Impulso)

    return V_Impulso, V_Abs

def periodo(Orbita1):

    
    """
    Dado os inputs: Orbita
    Obtemos: O Período da órbita
    """

    a = Orbita1[0]

    T = 2*np.pi*(a**(3/2))/np.sqrt(Mi)

    return T

def Conversao_XYZ_Orbita(V, R):
    """
    Recebe os valores de velocidade e posição, respectivamente
    Retorna os parâmetros da órbita resultante após o impacto dos corpos
    Faz a conversão dos paramêtros geocêntricos para o referencial perifocal
    """

    R = np.ravel(R)
    V = np.ravel(V)

    Abs_V = np.linalg.norm(V)
    Abs_R = np.linalg.norm(R)
    r = R/Abs_R
    v = V/Abs_V
    H = np.cross(R,V)
    h = np.cross(r, v) 

    k = np.array([0,0,1])

    Nodo = np.cross(k, H)
    Abs_Nodo = np.linalg.norm(Nodo)

    # nodo = np.cross(k, h)
    # Abs_nodo = np.linalg.norm(nodo)


    e_vec = 1/Mi * ((Abs_V**2 - Mi/Abs_R) * R - np.dot(R, V)* V)
    e = np.linalg.norm(e_vec)

    Energia = (Abs_V**2)/2 - Mi/Abs_R
    a = -Mi/(2*Energia)

    I = np.arccos(h[2] / np.linalg.norm(h))

    #Calculo RAAN
    if Abs_Nodo != 0:
        Raan = np.arccos(Nodo[0] / Abs_Nodo)
        if Nodo[1] < 0:
            Raan = 2*np.pi - Raan
    else:
        Raan = 0

    #Calculo ArgPeri
    if Abs_Nodo != 0 and e > 1e-8:
        ArgPeri = np.arccos(np.dot(Nodo, e_vec) / (Abs_Nodo * e))
        if e_vec[2] < 0:
            ArgPeri = 2*np.pi - ArgPeri
    else:
        ArgPeri = 0

    rad2deg = 180/np.pi
    OrbitaFinal = [(a, e, Raan, I, ArgPeri), (a, e, Raan* rad2deg, I*rad2deg, ArgPeri*rad2deg)]

    return OrbitaFinal



Mi = 398600
