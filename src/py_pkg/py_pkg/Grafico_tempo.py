from Manobras import *
from Orbitas import * 

#Tudo aqui pra frente é sobre o teste de valores de T

t_values = np.arange(20, 80, 1)
delta_v_magnitudes = []
l = 0.0

#Podemos alterar os valores de tempo de modo a procurar o tempo ótimo

for t in t_values:
    delta_v = randevouz(Orbita_Intermediaria, Orbita_Detrito, 1, t)[0]
    delta_v = np.array(delta_v)  # Garante que é array
    norm_dv = np.linalg.norm(delta_v)
    delta_v_magnitudes.append(norm_dv)
    l += 1
    print(l/np.size(t_values))

Posicao_temporal = np.argmin(delta_v_magnitudes)
print(randevouz(Orbita_Intermediaria, Orbita_Detrito, 1, t_values[Posicao_temporal]))
print(f" index {Posicao_temporal}, dV {delta_v_magnitudes[Posicao_temporal]}, tempo {t_values[Posicao_temporal]}")
plt.figure(figsize=(10, 5))
plt.plot(t_values, delta_v_magnitudes, label="||ΔV||", color='blue')
plt.xlabel("Tempo (s)")
plt.ylabel("||ΔV|| (km/s)")
plt.title("Variação do módulo de ΔV ao longo do tempo")
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()