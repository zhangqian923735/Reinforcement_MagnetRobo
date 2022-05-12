import pygame
import MagnetEnv

pygame.init()
r = 0
FPS = 120
clock = pygame.time.Clock()
size = (1200, 600)
Env = MagnetEnv.MagnetEnv(size)
Env.engine.change_force(10000000, 0, 0.05)
screen = pygame.display.set_mode(size)

def main_loop():
    global r
    """主循环"""
    while True:
        act = event_handler()
        _, r ,_, _ = Env.step(act, FPS)
        render()
        clock.tick(FPS)

def event_handler():
    """事件管理"""
    act = 4
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                act = 3
            elif event.key == pygame.K_DOWN:
                act = 2
            elif event.key == pygame.K_LEFT:
                act = 1
            elif event.key == pygame.K_RIGHT:
                act = 0
        elif event.type == pygame.MOUSEBUTTONDOWN:
            Env.engine.target_pos = event.pos
    return act


def render():
    screen.blit(Env.engine.Suface, (0,0))
    pygame.display.flip()
    pygame.display.set_caption(str(Env.engine.get_target_dis())+"-"+str(r))

def main():
   main_loop()

if __name__ == "__main__":
    main()