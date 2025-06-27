import pygame

class Checkpoint:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.isactiv = False
    def draw(self, win):
        pygame.draw.line(win, (0,0,0), (self.x1, self.y1), (self.x2, self.y2), 2)
        if self.isactiv:
            pygame.draw.line(win, (255,0,0), (self.x1, self.y1), (self.x2, self.y2), 2)

def get_checkpoints():
    checkpoints = []
    checkpoints.append(Checkpoint(500,700,500,880))
    checkpoints.append(Checkpoint(400,700,400,880))
    checkpoints.append(Checkpoint(300,700,300,880))
    checkpoints.append(Checkpoint(150,850,240,690))
    checkpoints.append(Checkpoint(50,765,200,650))
    checkpoints.append(Checkpoint(12,640,180,592))
    checkpoints.append(Checkpoint(10,535,180,535))
    checkpoints.append(Checkpoint(27,400,185,485))
    checkpoints.append(Checkpoint(113,290,200,450))
    checkpoints.append(Checkpoint(250,260,230,440))
    checkpoints.append(Checkpoint(370,320,270,460))
    checkpoints.append(Checkpoint(437,390,312,513))
    checkpoints.append(Checkpoint(480,430,390,600))
    checkpoints.append(Checkpoint(530,463,486,642))
    checkpoints.append(Checkpoint(600,465,600,650))
    checkpoints.append(Checkpoint(700,465,700,650))
    checkpoints.append(Checkpoint(800,465,800,650))
    checkpoints.append(Checkpoint(817,467,920,630))
    checkpoints.append(Checkpoint(840,460,1000,544))
    checkpoints.append(Checkpoint(845,442,1010,442))
    checkpoints.append(Checkpoint(840,423,1000,331))
    checkpoints.append(Checkpoint(822,414,910,242))
    checkpoints.append(Checkpoint(800,235,800,410))
    checkpoints.append(Checkpoint(700,235,700,410))
    checkpoints.append(Checkpoint(600,235,600,410))
    checkpoints.append(Checkpoint(567,230,480,400))
    checkpoints.append(Checkpoint(555,231,390,334))
    checkpoints.append(Checkpoint(350,220,550,220))
    checkpoints.append(Checkpoint(552,213,382,108))
    checkpoints.append(Checkpoint(563,200,487,40))
    checkpoints.append(Checkpoint(600,200,600,30))
    checkpoints.append(Checkpoint(700,200,700,30))
    checkpoints.append(Checkpoint(800,200,800,30))
    checkpoints.append(Checkpoint(900,200,900,30))
    checkpoints.append(Checkpoint(940,197,997,20))
    checkpoints.append(Checkpoint(977,209,1137,72))
    checkpoints.append(Checkpoint(1000,236,1190,192))
    checkpoints.append(Checkpoint(1000,300,1190,300))
    checkpoints.append(Checkpoint(1000,400,1190,400))
    checkpoints.append(Checkpoint(1000,500,1190,500))
    checkpoints.append(Checkpoint(1000,600,1190,600))
    checkpoints.append(Checkpoint(980,630,1170,700))
    checkpoints.append(Checkpoint(970,650,1140,770))
    checkpoints.append(Checkpoint(950,666,1100,815))
    checkpoints.append(Checkpoint(930,690,1020,850))
    checkpoints.append(Checkpoint(900,700,900,880))
    checkpoints.append(Checkpoint(800,700,800,880))
    checkpoints.append(Checkpoint(700,700,700,880))
    checkpoints[len(checkpoints)-1].isactiv = True
    return(checkpoints)
