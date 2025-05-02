import cProfile
import pygame
from game.game_environment import Car
from game.track import get_walls

def benchmark_cast():
    pygame.init()
    pygame.display.set_mode((1, 1))
    # 1) Walls laden
    walls = get_walls()
    # 2) Car an einer sinnvollen Startposition erzeugen
    car = Car(600, 790)

    # 3) Profiling starten
    pr = cProfile.Profile()
    pr.enable()

    # 4) 1000x cast aufrufen
    for _ in range(1000):
        _ = car.cast(walls)

    # 5) Profiling beenden und ausgeben
    pr.disable()
    pr.print_stats(sort="tottime")


if __name__ == "__main__":
    benchmark_cast()