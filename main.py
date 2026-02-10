import pygame
import numpy as np

from Orbitas import *
from desenho_orbitas import *
theta = 0
Orbita_Nave = [6820, 0.0002, 5.051777, 0.7504916, 3.8017826, 50000]

current_orbit = Orbita_Nave
phase = 0  # 0 inicial → 1 inter1 → 2 inter2 → 3 final

Orbita_Inter1  = [6988.96, 0.02405, 5.16617, 1.23918, 0.22492]   # Manobra 1
Orbita_Inter2  = [6989.24, 0.02403, 5.05177, 0.75049, 6.51127]   # Manobra 2

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        # tecla SPACE faz manobra
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                phase += 1
                if phase == 1:
                    current_orbit = Orbita_Inter1
                elif phase == 2:
                    current_orbit = Orbita_Inter2
                elif phase == 3:
                    current_orbit = Orbita_Detrito

    screen.fill((0,0,0))

    # desenhar todas orbitas
    draw_orbit(Orbita_Nave, (0,0,255))
    draw_orbit(Orbita_Inter1, (255,255,0))
    draw_orbit(Orbita_Inter2, (255,0,255))
    draw_orbit(Orbita_Detrito, (255,0,0))

    # posição nave
    theta += 0.002
    x,y = orbit_position(current_orbit, theta)
    pygame.draw.circle(screen, (255,255,255), (int(x),int(y)), 5)

    txt = font.render(f"Phase {phase}", True, (255,255,255))
    screen.blit(txt, (10,10))

    pygame.display.flip()
    clock.tick(60)
