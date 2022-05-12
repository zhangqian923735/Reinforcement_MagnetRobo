import numpy as np
import MagnetEngine
import pygame
import pymunk

class MagnetEnv():
    engine = MagnetEngine.magnetEngine()
    def __init__(self) -> None:
        self.engine.add_a_ball((0, 0))
        self.engine.set_magnet_pos((600, 600))
        self.target_pos = (0, 0)
        self.max_steps  = 300
        self.now_steps = 0
        self.penalty   = -0.1

    def step(self, action:int, FPS:int=10)->tuple:
        """
        X速度 +10
        X速度 -10
        Y速度 +10
        Y速度 -10
        挂机
        """
        done = None
        self.now_steps += 1
        actions = [(10, 0), (-10, 0), (0, 10), (0, -10), (0, 0)]
        now_act = actions[action]
        now_speed = [i + k if abs(i + k) < 50 else 50 * int((i + k)/abs(i + k)) 
                        for i, k in zip(self.engine.magnet_speed, now_act)]
        self.engine.set_megnet_speed(tuple(now_speed))
        self.engine.step(1/FPS)
        s = np.array(pygame.surfarray.pixels3d(self.engine.render()))
        reward_distance = self.engine.body_index[0].position.get_distance((self.magnet_pos))
        if reward_distance < 5 or self.now_steps > self.max_steps:
            done = True
            reward = - np.log(reward_distance * 0.001)
        else:
            done = False
            reward = self.penalty
        return s, reward, done, (reward_distance)

    def reset()->array:
        pass
    
    def render()->None:
        pass


if __name__ == "__main__":
    pass