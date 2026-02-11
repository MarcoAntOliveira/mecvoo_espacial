from Orbitas import *
from Manobras import *


T = periodo(Orbita_Detrito)
valores = []
x = 50
for i in range(1,x+1, 1):
    resultado = phasing_manuever(Orbita_Detrito, T/2, i)
    valores.append(resultado[1])
    
# Plotando
plt.figure(figsize=(10, 5))
plt.plot(range(x), valores, label='phasing_manuever()[1]')
plt.xlabel('i')
plt.ylabel('Resultado da função')
plt.title('Resultado de phasing_manuever em função de i')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()