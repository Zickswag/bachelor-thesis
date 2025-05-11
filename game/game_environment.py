import pygame
import math
import numpy as np
from math import hypot
from game.track import get_walls
from game.checkpoints import get_checkpoints

# Fenster & Framerate 
WINDOW_SIZE         = (1200, 900)
FPS                 = 120

# Belohnungen & Strafen
CHECKPOINT_REWARD   = 1
LIFE_REWARD         = 0
CRASH_PENALTY       = -1

#Fahrzeugparameter
CAR_WIDTH           = 14
CAR_HEIGHT          = 30
THRESHOLD           = int(math.hypot(CAR_WIDTH, CAR_HEIGHT)/2)  # ≈16 px
MAX_SPEED           = 15
ACCELERATION_FACTOR = 1.05     
TURN_ANGLE_RAD      = math.radians(15)

# Sensor (Ray-Casting)
SENSOR_RANGE        = 1000     

def is_checkpoint_passed(px, py, x1, y1, x2, y2, thresh):
    # Vektoren A→B und A→P
    AB = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    AP = np.array([px - x1, py - y1], dtype=np.float32)
    # Projektion von AP auf AB: t in ℝ
    denom = np.dot(AB, AB)
    if denom == 0:
        return False  # Segment der Länge 0
    t = np.dot(AP, AB) / denom
    # nur innerhalb des Segments [A,B] interessiert
    if t < 0.0 or t > 1.0:
        return False
    # nächster Punkt Q auf AB
    Q = np.array([x1, y1], dtype=np.float32) + t * AB
    # quadratischer Abstand P↔Q
    dist2 = (px - Q[0])**2 + (py - Q[1])**2
    return dist2 <= thresh * thresh

def rotate_rect(cx, cy, width, height, angle):
    sa = math.sin(angle)
    ca = math.cos(angle)
    hx, hy = width/2, height/2
    dx2, dy2 = ca*hx, sa*hx
    dx1, dy1 = sa*hy, -ca*hy
    p1 = (cx - dx2 + dx1, cy - dy2 + dy1)
    p2 = (cx + dx2 + dx1, cy + dy2 + dy1)
    p3 = (cx + dx2 - dx1, cy + dy2 - dy1)
    p4 = (cx - dx2 - dx1, cy - dy2 - dy1)
    return p1, p2, p3, p4

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class Line:
    def __init__(self, pt1, pt2):
        self.pt1 = Point(pt1.x, pt1.y)
        self.pt2 = Point(pt2.x, pt2.y)

class Ray:
    __slots__ = ("x", "y", "dx", "dy")
    def __init__(self,x,y,angle):
        self.x = x
        self.y = y
        self.dx = math.sin(angle) * SENSOR_RANGE
        self.dy = -math.cos(angle) * SENSOR_RANGE

    def cast(self, wall):
        x1, y1, x2, y2 = wall.x1, wall.y1, wall.x2, wall.y2
        x3, y3 = self.x, self.y
        x4, y4 = x3 + self.dx, y3 + self.dy

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            
        if(den == 0):
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if 0 < t < 1 and 0 < u < 1:
            return Point(math.floor(x1 + t * (x2 - x1)),
                           math.floor(y1 + t * (y2 - y1)))
        return None

class Car:
    def __init__(self, x, y):
        self.pt = Point(x, y)
        self.x = x
        self.y = y
        self.width = CAR_WIDTH
        self.height = CAR_HEIGHT
        self.points = 0
        self.original_image = pygame.image.load("assets/car.png").convert()
        self.image = self.original_image  
        self.image.set_colorkey((0,0,0))
        self.rect = self.image.get_rect().move(self.x, self.y)
        self.angle = math.radians(90)
        self.target_angle = self.angle
        self.dvel = 1
        self.vel = 0

        # Eckpunkte
        self.pt1 = Point(self.pt.x - self.width/2, self.pt.y - self.height/2)
        self.pt2 = Point(self.pt.x + self.width/2, self.pt.y - self.height/2)
        self.pt3 = Point(self.pt.x + self.width/2, self.pt.y + self.height/2)
        self.pt4 = Point(self.pt.x - self.width/2, self.pt.y + self.height/2)
        self.p1 = self.pt1; self.p2 = self.pt2; self.p3 = self.pt3; self.p4 = self.pt4

        # Die vier Linien-Seiten (precomputed) als Tupel (x3,y3,x4,y4)
        self.collision_lines = [
            (self.p1.x, self.p1.y, self.p2.x, self.p2.y),
            (self.p2.x, self.p2.y, self.p3.x, self.p3.y),
            (self.p3.x, self.p3.y, self.p4.x, self.p4.y),
            (self.p4.x, self.p4.y, self.p1.x, self.p1.y),
        ]
        

    def action(self, choice):
        if choice == 1:
            self.accelerate(self.dvel)
        elif choice == 8:
            self.accelerate(self.dvel); self.turn(1)
        elif choice == 7:
            self.accelerate(self.dvel); self.turn(-1)
        elif choice == 4:
            self.accelerate(-self.dvel)
        elif choice == 5:
            self.accelerate(-self.dvel); self.turn(1)
        elif choice == 6:
            self.accelerate(-self.dvel); self.turn(-1)
        elif choice == 3:
            self.turn(1)
        elif choice == 2:
            self.turn(-1)
    
    def accelerate(self,dvel):
        dvel = dvel * ACCELERATION_FACTOR
        self.vel = self.vel + dvel
        if self.vel > MAX_SPEED:
            self.vel = MAX_SPEED
        if self.vel < -MAX_SPEED:
            self.vel = -MAX_SPEED
        
    def turn(self, dir):
        self.target_angle = self.target_angle + dir * TURN_ANGLE_RAD
    
    def update(self):
        # 1) Winkel und Geschwindigkeit
        self.angle = self.target_angle
        sa = math.sin(self.angle)
        ca = math.cos(self.angle)
        self.velX = -sa * self.vel
        self.velY = ca * self.vel
        self.x += self.velX; self.y += self.velY

        # 2) Eckpunkte verschieben
        self.pt1 = Point(self.pt1.x + self.velX, self.pt1.y + self.velY)
        self.pt2 = Point(self.pt2.x + self.velX, self.pt2.y + self.velY)
        self.pt3 = Point(self.pt3.x + self.velX, self.pt3.y + self.velY)
        self.pt4 = Point(self.pt4.x + self.velX, self.pt4.y + self.velY)

        # 3) Rotieren
        coords = rotate_rect(
            self.x, self.y,
            self.width, self.height,
            self.target_angle
        )
        # Koordinaten als 4 Tupel: ((x1,y1), …, (x4,y4))
        self.p1 = Point(*coords[0])
        self.p2 = Point(*coords[1])
        self.p3 = Point(*coords[2])
        self.p4 = Point(*coords[3])

        # 4) Bild aktualisieren
        self.image = pygame.transform.rotate(self.original_image, 90 - math.degrees(self.target_angle))
        self.rect = self.image.get_rect(center=(self.x, self.y))
        
        # 5) collision_lines erneuern
        self.collision_lines = [
            (self.p1.x, self.p1.y, self.p2.x, self.p2.y),
            (self.p2.x, self.p2.y, self.p3.x, self.p3.y),
            (self.p3.x, self.p3.y, self.p4.x, self.p4.y),
            (self.p4.x, self.p4.y, self.p1.x, self.p1.y),
        ]

    def cast(self, walls):
        self.rays = []
        self.rays.append(Ray(self.x, self.y, self.target_angle))
        self.rays.append(Ray(self.x, self.y, self.target_angle - math.radians(30)))
        self.rays.append(Ray(self.x, self.y, self.target_angle + math.radians(30)))
        self.rays.append(Ray(self.x, self.y, self.target_angle + math.radians(45)))
        self.rays.append(Ray(self.x, self.y, self.target_angle - math.radians(45)))
        self.rays.append(Ray(self.x, self.y, self.target_angle + math.radians(90)))
        self.rays.append(Ray(self.x, self.y, self.target_angle - math.radians(90)))
        self.rays.append(Ray(self.x, self.y, self.target_angle + math.radians(180)))
        self.rays.append(Ray(self.x, self.y, self.target_angle + math.radians(10)))
        self.rays.append(Ray(self.x, self.y, self.target_angle - math.radians(10)))
        self.rays.append(Ray(self.x, self.y, self.target_angle + math.radians(135)))
        self.rays.append(Ray(self.x, self.y, self.target_angle - math.radians(135)))
        self.rays.append(Ray(self.x, self.y, self.target_angle + math.radians(20)))
        self.rays.append(Ray(self.x, self.y, self.target_angle - math.radians(20)))
        self.rays.append(Ray(self.p1.x, self.p1.y, self.target_angle + math.radians(90)))
        self.rays.append(Ray(self.p2.x, self.p2.y, self.target_angle - math.radians(90)))

        observations = []
        self.closestRays = []

        for ray in self.rays:
            closest = None #myPoint(0,0)
            record = math.inf
            dir_x, dir_y = ray.dx, ray.dy

            for wall in walls:
                if dir_x >= 0:
                    if wall.xmax < self.x:  # komplett links
                        continue
                else:
                    if wall.xmin > self.x:  # komplett rechts
                        continue

                if dir_y >= 0:
                    if wall.ymax < self.y:  # komplett oberhalb
                        continue
                else:
                    if wall.ymin > self.y:  # komplett unterhalb
                        continue

                pt = ray.cast(wall)
                if pt:
                    # Hypot schnell da C optimiert
                    dx = self.x - pt.x
                    dy = self.y - pt.y
                    d = hypot(dx, dy)
                    if d < record:
                        record, closest = d, pt

            if closest: 
                # append distance for current ray 
                self.closestRays.append(closest)
                observations.append(record)
               
            else:
                observations.append(SENSOR_RANGE)

        for i in range(len(observations)):
            # invert observation values 0 is far away 1 is close
            observations[i] = ((SENSOR_RANGE - observations[i]) / SENSOR_RANGE)

        observations.append(self.vel / MAX_SPEED)
        return observations

    def collision(self, wall):
        x1, y1, x2, y2 = wall.x1, wall.y1, wall.x2, wall.y2

        for x3, y3, x4, y4 in self.collision_lines:
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if den == 0:
                continue  # parallel, kein Schnitt
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
            if 0 < t < 1 and 0 < u < 1:
                return True
        return False
    
    def score(self, checkpoint, thresh):
        if is_checkpoint_passed(self.x, self.y,checkpoint.x1, checkpoint.y1,checkpoint.x2, checkpoint.y2, thresh):
            self.points += CHECKPOINT_REWARD
            return True
        return False

    def reset(self):
        self.x = 50
        self.y = 300
        self.velX = 0
        self.velY = 0
        self.vel = 0
        self.angle = math.radians(90)
        self.target_angle = self.angle
        self.points = 0
        self.pt1 = Point(self.pt.x - self.width / 2, self.pt.y - self.height / 2)
        self.pt2 = Point(self.pt.x + self.width / 2, self.pt.y - self.height / 2)
        self.pt3 = Point(self.pt.x + self.width / 2, self.pt.y + self.height / 2)
        self.pt4 = Point(self.pt.x - self.width / 2, self.pt.y + self.height / 2)
        self.p1 = self.pt1
        self.p2 = self.pt2
        self.p3 = self.pt3
        self.p4 = self.pt4

    def draw(self, win):
        win.blit(self.image, self.rect)

class GameEnvironment:
    def __init__(self):
        pygame.init()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 36)
        self.fps = FPS
        self.width, self.height = WINDOW_SIZE
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RACE")
        self.back_image = pygame.image.load("assets/background.png").convert()
        self.back_rect = self.back_image.get_rect().move(0, 0)
        self.action_space = None
        self.observation_space = None
        self.game_reward = 0
        self.score = 0
        self.reset()

    def reset(self):
        self.car = Car(600, 790)
        self.walls = get_walls()
        self.checkpoints = get_checkpoints()
        self.game_reward = 0

    def step(self, action):
        done = False
        self.car.action(action)
        self.car.update()
        reward = LIFE_REWARD

        # 1) Check if car passes active Goal and scores
        for checkpoint_index, checkpoint in enumerate(self.checkpoints):
            if not checkpoint.isactiv:
                continue

            # benutze jetzt die neue score-Methode (Punkt-Segment-Abstand)
            if self.car.score(checkpoint, THRESHOLD):
                # deactivate current, activate next (zyklisch)
                checkpoint.isactiv = False
                prev_checkpoint_index = (checkpoint_index - 1) % len(self.checkpoints)
                self.checkpoints[prev_checkpoint_index].isactiv = True

                reward += CHECKPOINT_REWARD
            break  # nur ein aktiver Checkpoint existiert zu jeder Zeit

        #check if car crashed in the wall
        for wall in self.walls:
            if self.car.collision(wall):
                reward += CRASH_PENALTY
                done = True
                break

        state = None if done else self.car.cast(self.walls)
        return state, reward, done

    def render(self, action):
        DRAW_WALLS = False
        DRAW_CHECKPOINTS = False
        DRAW_RAYS = False
        self.clock = pygame.time.Clock()
        self.screen.blit(self.back_image, self.back_rect)

        if DRAW_WALLS:
            for wall in self.walls:
                wall.draw(self.screen)
        
        if DRAW_CHECKPOINTS:
            for checkpoint in self.checkpoints:
                checkpoint.draw(self.screen)
                if checkpoint.isactiv:
                    checkpoint.draw(self.screen)
        
        self.car.draw(self.screen)

        if DRAW_RAYS:
            i = 0
            for pt in self.car.closestRays:
                pygame.draw.circle(self.screen, (120,0,255), (pt.x, pt.y), 5)
                i += 1
                if i < 15:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.x, self.car.y), (pt.x, pt.y), 1)
                elif i >=15 and i < 17:
                    pygame.draw.line(self.screen, (255,255,255), ((self.car.p1.x + self.car.p2.x)/2, (self.car.p1.y + self.car.p2.y)/2), (pt.x, pt.y), 1)
                elif i == 17:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.p1.x , self.car.p1.y ), (pt.x, pt.y), 1)
                else:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.p2.x, self.car.p2.y), (pt.x, pt.y), 1)

        #render controll
        #pygame.draw.rect(self.screen,(255,255,255),(800, 100, 40, 40),2)
        #pygame.draw.rect(self.screen,(255,255,255),(850, 100, 40, 40),2)
        #pygame.draw.rect(self.screen,(255,255,255),(900, 100, 40, 40),2)
        #pygame.draw.rect(self.screen,(255,255,255),(850, 50, 40, 40),2)
        #
        #if action == 4:
        #    pygame.draw.rect(self.screen,(0,255,0),(850, 50, 40, 40)) 
        #elif action == 6:
        #    pygame.draw.rect(self.screen,(0,255,0),(850, 50, 40, 40))
        #    pygame.draw.rect(self.screen,(0,255,0),(800, 100, 40, 40))
        #elif action == 5:
        #    pygame.draw.rect(self.screen,(0,255,0),(850, 50, 40, 40))
        #    pygame.draw.rect(self.screen,(0,255,0),(900, 100, 40, 40))
        #elif action == 1:
        #    pygame.draw.rect(self.screen,(0,255,0),(850, 100, 40, 40)) 
        #elif action == 8:
        #    pygame.draw.rect(self.screen,(0,255,0),(850, 100, 40, 40))
        #    pygame.draw.rect(self.screen,(0,255,0),(800, 100, 40, 40))
        #elif action == 7:
        #    pygame.draw.rect(self.screen,(0,255,0),(850, 100, 40, 40))
        #    pygame.draw.rect(self.screen,(0,255,0),(900, 100, 40, 40))
        #elif action == 2:
        #    pygame.draw.rect(self.screen,(0,255,0),(800, 100, 40, 40))
        #elif action == 3:
        #    pygame.draw.rect(self.screen,(0,255,0),(900, 100, 40, 40))

        # score
        text_surface = self.font.render(f'Punkte {self.car.points}', True, pygame.Color('red'))
        self.screen.blit(text_surface, dest=(0, 0))

        self.clock.tick(FPS)
        pygame.display.update()

    def close(self):
        pygame.quit()


