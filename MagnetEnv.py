import numpy as np
import MagnetEngine
import pygame

class MagnetEnv():
    max_steps  = 300
    step_penalty   = 0
    screen = None
    
    def __init__(self, size=(600,600)) -> None:
        self.engine = MagnetEngine.magnetEngine(size)
        self.engine.add_a_ball((0, 0))
        self.engine.set_magnet_pos((600, 600))
        self.engine.target_pos = (0, 0)
        self.now_steps = 0

    def step(self, action:int, FPS:int=10)->tuple:
        """
        X速度 +10,
        X速度 -10,
        Y速度 +10,
        Y速度 -10,
        挂机
        """
        done = None
        self.now_steps += 1
        actions = [(10, 0), (-10, 0), (0, 10), (0, -10), (0, 0)]
        now_act = actions[action]
        now_speed = [i + k for i, k in zip(self.engine.magnet_speed, now_act)]
        self.engine.set_megnet_speed(tuple(now_speed))
        self.engine.step(1/FPS)
        s = np.array(pygame.surfarray.pixels3d(self.engine.render()))
        reward_distance = self.engine.get_target_dis()
        if reward_distance < 50 or self.now_steps >= self.max_steps:
            done = True
            reward = - np.log(reward_distance * 0.001 + 1e-6)
        else:
            done = False
            reward = self.step_penalty
        return s, reward, done, (reward_distance)


    def reset(self, pos_ball=(0,0), pos_magnet=(0,0), pos_target=(0,0)):
        self.engine.delete_all_ball()
        self.engine.add_a_ball(pos_ball)
        self.engine.set_magnet_pos(pos_magnet)
        self.engine.target_pos = pos_target
        self.now_steps = 0
        s = np.array(pygame.surfarray.pixels3d(self.engine.render()))
        return s
    
    def render(self)->None:
        """渲染窗口"""
        screen = pygame.display.set_mode((600, 600)) if self.screen is None else screen
        screen.blit(self.engine.Suface, (0,0))
        pygame.display.flip()
        pygame.display.set_caption(f"step:{self.now_steps}/{self.max_steps}")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
    
    def close(self):
        """关闭窗口"""
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None

if __name__ == "__main__":
    pass