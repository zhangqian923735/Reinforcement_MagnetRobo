import pygame
import pymunk
from pygame.color import THECOLORS
import pymunk.constraints
import pymunk.pygame_util
import numpy as np
from requests import delete
pygame.init()

class magnetEngine():
    target_pos = (0, 0)
    magnet_pos = (0, 0)
    magnet_speed = (0, 0)
    drop_pos   = (0, 0)
    clock = pygame.time.Clock()
    
    def __init__(self, size:tuple=(600, 600)) -> None:
        self.size = size
        self.strength, self.dis_coeff, self.viscous_coeff = 0, 0, 0
        self.change_force()
        self.space = pymunk.Space()
        self.body_index = []
        self.poly = []
        self.Suface = pygame.surface.Surface(size)
        self.print_options = pymunk.pygame_util.DrawOptions(self.Suface)
        self.print_options.flags = (
        self.print_options.flags ^ pymunk.pygame_util.DrawOptions.DRAW_COLLISION_POINTS
        )

    def add_a_line(self, pos_0, pos_1):
        """添加一堵障碍墙"""
        static_body = self.space.static_body
        line = pymunk.Segment(static_body, pos_0, pos_1, 3.0)
        line.elasticity = 0  # 弹性系数 0-1
        line.friction = 0  # 摩擦系数 0-1
        self.space.add(line)

    def set_magnet_pos(self, pos=(0,0)):
        """设置磁铁的位置"""
        posx = pos[0] if pos[0] < self.size[0] else self.size[0]
        posy = pos[1] if pos[1] < self.size[1] else self.size[1]

        posx = 0 if posx < 0 else posx
        posy = 0 if posy < 0 else posy

        self.magnet_pos = (posx, posy)

    def add_a_ball(self, pos=(0,0), radium=5, elasticity=0.5, friction=0.5):
        """添加一个小磁球"""
        body = pymunk.Body()
        body.position = pos
        body.r = radium
        poly = pymunk.Circle(body, radium)
        poly.mass = 10
        poly.elasticity=elasticity  # 弹性系数 0-1
        poly.friction=friction  # 摩擦系数 0-1
        self.body_index.append(body)
        self.poly.append(poly)
        self.space.add(body, poly)

    def delete_the_ball(self):
        for key, ball in enumerate(self.body_index):
            self.space.remove(self.body_index[key], self.poly[key])      
            self.body_index.pop(key)
            self.poly.pop(key)

    def apply_magnet_force(self):
        for key, ball in enumerate(self.body_index):
            dis = ball.position.get_distance((self.magnet_pos))
            dir = ((self.magnet_pos) - ball.position).normalized()
            mag_force = self.strength / (self.dis_coeff*dis + 60)
            mag_force = mag_force * dir 
            ball.apply_force_at_world_point(mag_force, ball.position)
            #施加粘滞阻力
            try:
                dir = - ball.velocity.normalized()
            except:
                raise BaseException(key, *ball.velocity, *ball.force)  
            v = ball.velocity.length
            damp_force = (((self.viscous_coeff * mag_force.length +  1 * v) * v)) * dir 
            ball.apply_force_at_world_point(damp_force, ball.position)
    
    def change_force(self, strength=7000, dis_coeff=1.5, viscous_coeff=0.1):
        """磁场强度，距离衰减系数，粘滞力大小"""
        self.strength, self.dis_coeff, self.viscous_coeff = strength, dis_coeff, viscous_coeff

    def set_megnet_speed(self, speed=(0, 0)):
        self.magnet_speed = speed
    
    def step(self, time=1, exact=1000):
        """时间（秒）， 仿真精度"""
        for _ in range(int(exact * time)):
            """仿真次数 = 精度（次/秒）* 时间 (秒)"""
            new_pos = ( self.magnet_pos[0] + self.magnet_speed[0] * (1 / exact), 
                        self.magnet_pos[1] + self.magnet_speed[1] * (1 / exact) )
            self.set_magnet_pos(new_pos)
            self.apply_magnet_force()
            self.space.step(1 / exact)

    def render(self):
        self.Suface.fill(THECOLORS["white"])  # 清空屏幕
        self.space.debug_draw(self.print_options)
        w, h = 40, 40
        rect = (self.magnet_pos[0] - w/2, self.magnet_pos[1] - h/2, w, h)
        pygame.draw.rect(self.Suface, (0, 0, 0), rect, 5)
        return self.Suface
    

if __name__ == "__main__":
    size = (600, 600)
    Engine = magnetEngine(size)
    clock = pygame.time.Clock()
    
    screen = pygame.display.set_mode(size)
    Engine.add_a_ball((0, 0))
    # Engine.add_a_line((300, 300), (600, 300))
    Engine.set_magnet_pos((600, 600))
    Engine.change_force(100000, 0.5, 0.0001)
    Engine.set_megnet_speed((-50, -50))
    while True:
        pygame.event.get()
        Engine.step(1/60)
        screen.blit(Engine.render(), (0,0))
        pygame.display.flip()
        clock.tick(120)
        

        