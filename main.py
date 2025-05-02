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

        # 3) Mapping auf die 0–8-Aktionen, mit vertauschtem Vorwärts/Rückwärts
        #    Original: 1=vorwärts, 4=rückwärts
        #    Hier: Down => vorwärts (1), Up => rückwärts (4)
        if keys[K_DOWN] and keys[K_LEFT]:
            action = 7   # vorwärts + links
        elif keys[K_DOWN] and keys[K_RIGHT]:
            action = 8   # vorwärts + rechts
        elif keys[K_DOWN]:
            action = 1   # nur vorwärts
        elif keys[K_UP] and keys[K_LEFT]:
            action = 5   # rückwärts + links
        elif keys[K_UP] and keys[K_RIGHT]:
            action = 6   # rückwärts + rechts
        elif keys[K_UP]:
            action = 4   # nur rückwärts
        elif keys[K_LEFT]:
            action = 2   # nur links drehen
        elif keys[K_RIGHT]:
            action = 3   # nur rechts drehen
        else:
            action = 0   # keine Eingabe

        # 4) Einen Schritt im Env ausführen
        state, reward, done = env.step(action)

        # 5) Zeichnen
        env.render(action)

        # 6) Neustart, falls Crash oder letztes Checkpoint
        if done:
            env.reset()

        # 7) Frame-Limit
        clock.tick(env.fps)  # hält das Spiel bei 120 FPS

    env.close()
    pygame.quit()

if __name__ == '__main__':
    main()
