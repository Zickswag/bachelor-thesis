import pygame
from game.game_environment import GameEnvironment
from pygame.locals import QUIT, K_UP, K_DOWN, K_LEFT, K_RIGHT

def main():
    pygame.init()
    env = GameEnvironment()
    clock = pygame.time.Clock()
    running = True

    while running:
        # 1) Input-Events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        # 2) Tasten abfragen
        keys = pygame.key.get_pressed()

        # 3) Mapping auf die 0–8-Aktionen
        if keys[K_DOWN]:
            action = 1  
        elif keys[K_LEFT]:
            action = 2   
        elif keys[K_RIGHT]:
            action = 3   
        elif keys[K_UP]:
            action = 4   
        else:
            action = 0   
           

        # 4) Einen Schritt im Env ausführen
        state, reward, done = env.step(action)
        
        # 5) Zeichnen
        env.render(action)

        # 6) Neustart, falls Crash oder letztes Checkpoint
        if done:
            env.reset()

        # 7) Frame-Limit
        clock.tick(env.fps)  

    env.close()
    pygame.quit()

if __name__ == '__main__':
    main()
