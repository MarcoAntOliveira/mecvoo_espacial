import pygame
import numpy as np

from Orbitas import *

# ================= PYGAME ==================
pygame.init()
WIDTH, HEIGHT = 900, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Orbital Simulation")

CENTER = np.array([WIDTH//2, HEIGHT//2])
SCALE = 0.05  # Ajuste para caber na tela

# ================= FUNÇÃO PARA DESENHAR ORBITA ==================
def draw_orbit(orbit, color):
    a, e, RAAN, inc, w = orbit[0], orbit[1], orbit[2], orbit[3], orbit[4]
    
    points = []
    for theta in np.linspace(0, 2*np.pi, 500):
        r = a*(1-e**2)/(1 + e*np.cos(theta))
        
        # posição no plano orbital
        x = r * np.cos(theta + w)
        y = r * np.sin(theta + w)
        
        # escala e centro
        pos = CENTER + SCALE * np.array([x, -y])
        points.append(pos)

    pygame.draw.lines(screen, color, True, points, 2)

# ================= LOOP ==================
running = True
while running:
    screen.fill((0,0,0))

    # Terra
    pygame.draw.circle(screen, (0,100,255), CENTER, 10)

    # Orbitas
    draw_orbit(Orbita_Detrito, (255,0,0))
    draw_orbit(Orbita_Intermediaria, (0,255,0))
    draw_orbit(Orbita_Intermediaria2, (255,255,0))
    draw_orbit(Orbita_Nave, (0,255,255))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()

pygame.quit()
