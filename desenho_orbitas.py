import pygame
import numpy as np
from Orbitas import *


# ---------------- CONFIG -----------------
WIDTH, HEIGHT = 1000, 1000
CENTER = np.array([WIDTH//2, HEIGHT//2])
SCALE = 0.05  # km → pixels (ajuste!)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulador de Manobras Orbitais")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)


def draw_orbit(orbit, color):
    a, e, RAAN, inc, w, *_ = orbit

    
    points = []
    for theta in np.linspace(0, 2*np.pi, 500):
        r = a*(1-e**2)/(1+e*np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # rotação pelo argumento do perigeu
        R = np.array([
            [np.cos(w), -np.sin(w)],
            [np.sin(w),  np.cos(w)]
        ])
        x, y = R @ np.array([x, y])

        px = CENTER[0] + x * SCALE
        py = CENTER[1] + y * SCALE
        points.append((px, py))

    pygame.draw.lines(screen, color, True, points, 2)



def orbit_position(orbit, theta):
    a, e, RAAN, inc, w, *_ = orbit

    r = a*(1-e**2)/(1+e*np.cos(theta))

    x = r*np.cos(theta)
    y = r*np.sin(theta)

    R = np.array([
        [np.cos(w), -np.sin(w)],
        [np.sin(w),  np.cos(w)]
    ])
    x, y = R @ np.array([x, y])

    return CENTER[0] + x*SCALE, CENTER[1] + y*SCALE
