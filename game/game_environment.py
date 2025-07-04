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

# Fahrzeugparameter
CAR_WIDTH           = 14
CAR_HEIGHT          = 30
THRESHOLD           = int(math.hypot(CAR_WIDTH, CAR_HEIGHT)/2) 
MAX_SPEED           = 15
ACCELERATION_FACTOR = 1.05
DRIFT_FACTOR        = 0.1    
TURN_ANGLE_RAD      = math.radians(15)

# Sensor (Ray-Casting)
SENSOR_RANGE        = 1000

# Render
DRAW_WALLS          = False
DRAW_CHECKPOINTS    = False
DRAW_RAYS           = False

# Hilfsfunktion: Normalisiere Winkel auf [-π, π]
def _normalize_angle(delta):
    while delta > math.pi:
        delta -= 2*math.pi
    while delta < -math.pi:
        delta += 2*math.pi
    return delta

# Prüft, ob ein Punkt nah genug an einem Checkpoint-Segment liegt
def is_checkpoint_passed(px, py, x1, y1, x2, y2, thresh):
    # Vektor von A → B (Checkpoint)
    AB = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    # Vektor von A → P (Position des Autos)
    AP = np.array([px - x1, py - y1], dtype=np.float32)

    # Projektion: berechne Parameter t für die Position auf AB
    denom = np.dot(AB, AB)
    if denom == 0:
        return False  # Ungültiges Segment
    t = np.dot(AP, AB) / denom
    
    # Nur dann relevant, wenn Punkt auf dem Segment liegt (0 ≤ t ≤ 1)
    if t < 0.0 or t > 1.0:
        return False
    
    # Berechne nächsten Punkt Q auf AB
    Q = np.array([x1, y1], dtype=np.float32) + t * AB

    # Abstand zwischen Punkt P und Q quadrieren
    dist2 = (px - Q[0])**2 + (py - Q[1])**2

    # Ist der Abstand kleiner als der Schwellwert?
    return dist2 <= thresh * thresh

# Rotiert ein Rechteck
def rotate_rect(cx, cy, width, height, angle):
    sa = math.sin(angle)
    ca = math.cos(angle)
    hx, hy = width/2, height/2

    # Eckpunktverschiebungen entlang der Rotationsachsen
    dx2, dy2 = ca*hx, sa*hx
    dx1, dy1 = sa*hy, -ca*hy

    # Rückgabe: vier Punkte
    p1 = (cx - dx2 + dx1, cy - dy2 + dy1)
    p2 = (cx + dx2 + dx1, cy + dy2 + dy1)
    p3 = (cx + dx2 - dx1, cy + dy2 - dy1)
    p4 = (cx - dx2 - dx1, cy - dy2 - dy1)
    return p1, p2, p3, p4

# Hilfsklasse: Punkt (2D-Koordinaten)
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Hilfsklasse: Gerade zwischen zwei Punkten (als Liniensegment) 
class Line:
    def __init__(self, pt1, pt2):
        self.pt1 = Point(pt1.x, pt1.y)
        self.pt2 = Point(pt2.x, pt2.y)

# Klasse für einen Ray (Sensorstrahl)
class Ray:
    __slots__ = ("x", "y", "dx", "dy") # Speicheroptimierung

    def __init__(self,x,y,angle):
        self.x = x
        self.y = y
        self.dx = math.sin(angle) * SENSOR_RANGE
        self.dy = -math.cos(angle) * SENSOR_RANGE

    def cast(self, wall):
        # Wand-Endpunkte
        x1, y1, x2, y2 = wall.x1, wall.y1, wall.x2, wall.y2
        # Ray-Endpunkte
        x3, y3 = self.x, self.y
        x4, y4 = x3 + self.dx, y3 + self.dy

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            
        if(den == 0):
            return None # Parallele Linien, kein Schnitt
        
        # Parameter t und u für den Schnittpunkt
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        # Schnittpunkt muss innerhalb der Segmente liegen
        if 0 < t < 1 and 0 < u < 1:
            return Point(math.floor(x1 + t * (x2 - x1)),math.floor(y1 + t * (y2 - y1)))
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
        self.travel_angle = self.angle
        self.angular_velocity = 0.0   
        self.dvel = 1
        self.vel = 0
        self.velX = 0
        self.velY = 0

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
        
    # Aktionsauswahl
    def action(self, choice):
        if choice == 1:
            self.accelerate(self.dvel)
        elif choice == 2:
            self.turn(-1)
        elif choice == 3:
            self.turn(1)
        elif choice == 4:
            self.accelerate(-self.dvel)
    
    # Beschleunigung
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
        # 1) Sofortige Ausrichtung übernehmen
        self.angle = self.target_angle

        # 2) Fahrtrichtung (travel_angle) schwenkt nur teilweise zur neuen Ausrichtung
        delta = _normalize_angle(self.angle - self.travel_angle)
        # Winkelgeschwindigkeit ist die Änderung des travel_angle pro Frame
        self.angular_velocity = delta * DRIFT_FACTOR
        self.travel_angle += self.angular_velocity

        # 3) Neue Geschwindigkeitskomponenten aus travel_angle berechnen
        sa = math.sin(self.travel_angle)
        ca = math.cos(self.travel_angle)
        self.velX = -sa * self.vel
        self.velY =  ca * self.vel

        # 4) Position updaten
        self.x += self.velX
        self.y += self.velY

        # 5) Eckpunkte verschieben
        for pt in (self.pt1, self.pt2, self.pt3, self.pt4):
            pt.x += self.velX
            pt.y += self.velY

        # 6) Eckpunkte rotieren an der Karosserie-Ausrichtung
        coords = rotate_rect(
            self.x, self.y,
            self.width, self.height,
            self.target_angle
        )
        self.p1 = Point(*coords[0])
        self.p2 = Point(*coords[1])
        self.p3 = Point(*coords[2])
        self.p4 = Point(*coords[3])

        # 7) Bild weiterhin nach Karosserie-Ausrichtung drehen
        self.image = pygame.transform.rotate(self.original_image, 90 - math.degrees(self.target_angle))
        self.rect = self.image.get_rect(center=(self.x, self.y))

        # 8) Collision-Linien updaten
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
            closest = None
            record = math.inf
            dir_x, dir_y = ray.dx, ray.dy

            for wall in walls:
                if dir_x >= 0:
                    if wall.xmax < self.x:  
                        continue
                else:
                    if wall.xmin > self.x:  
                        continue

                if dir_y >= 0:
                    if wall.ymax < self.y:  
                        continue
                else:
                    if wall.ymin > self.y: 
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
                self.closestRays.append(closest)
                observations.append(record)
            else:
                observations.append(SENSOR_RANGE)

        observations = np.array(observations, dtype=np.float32)
        # NaN und Inf abfangen
        observations = np.nan_to_num(observations, nan=0.0, posinf=SENSOR_RANGE, neginf=0.0)
        # Vektorisiertes Normalisieren auf [0,1]
        observations = (SENSOR_RANGE - observations) / SENSOR_RANGE
        # Reste außerhalb [0,1] abschneiden
        observations = np.clip(observations, 0.0, 1.0)

        max_angular_vel = TURN_ANGLE_RAD * DRIFT_FACTOR
        normalized_angular_vel = np.clip(self.angular_velocity / max_angular_vel, -1.0, 1.0)

        additional_features = np.array([
            self.vel / MAX_SPEED,
            normalized_angular_vel
        ], dtype=np.float32)

        # Geschwindigkeit anhängen (np.append → np.concatenate, um float32 zu behalten)
        observations = np.concatenate([observations, additional_features]).astype(np.float32)

        return observations


    def collision(self, wall):
        x1, y1, x2, y2 = wall.x1, wall.y1, wall.x2, wall.y2

        for x3, y3, x4, y4 in self.collision_lines:
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if den == 0:
                continue  # Parallel, kein Schnitt
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
        self.travel_angle = self.angle
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

        # Check ob Auto einen Checkpoint passiert hat
        for checkpoint_index, checkpoint in enumerate(self.checkpoints):
            if not checkpoint.isactiv:
                continue

            # Benutze jetzt die neue score-Methode (Punkt-Segment-Abstand)
            if self.car.score(checkpoint, THRESHOLD):
                # Aktueller Checkpoint deaktivieren, nächster aktivieren
                checkpoint.isactiv = False
                prev_checkpoint_index = (checkpoint_index - 1) % len(self.checkpoints)
                self.checkpoints[prev_checkpoint_index].isactiv = True

                reward += CHECKPOINT_REWARD
            break  # Nur ein aktiver Checkpoint existiert zu jeder Zeit

        # Crash Erkennung
        for wall in self.walls:
            if self.car.collision(wall):
                reward += CRASH_PENALTY
                done = True
                break

        state = None if done else self.car.cast(self.walls)
        return state, reward, done

    def render(self, action, episode, global_steps, max_q, episode_steps):
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



        WHITE = (255, 255, 255)
        hud_font = pygame.font.SysFont("Courier New", 24)

        # Positionen
        label_x = 30
        value_x = 190  # Feste X-Position für die Zahlen
        start_y = 30
        line_spacing = 30

        # Daten vorbereiten
        lines = [
            ("Episode:",                f"{episode}"),
            ("Länge:",                  f"{episode_steps}"),
            ("Schritte:",               f"{global_steps}"),
            ("Score:",                  f"{self.car.points}"),
            ("Max Q:",                  f"{max_q:.2f}" if max_q is not None else "-"),
        ]

        # Anzeige untereinander
        for i, (label, value) in enumerate(lines):
            y = start_y + i * line_spacing
            label_surface = hud_font.render(label, True, WHITE)
            value_surface = hud_font.render(value, True, WHITE)
            self.screen.blit(label_surface, (label_x, y))
            self.screen.blit(value_surface, (value_x, y))

        self.clock.tick(FPS)
        pygame.display.update()

    def close(self):
        pygame.quit()


