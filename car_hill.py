import math
import time
from typing import Optional
import numpy as np
import gym
from gym import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled


class carEnvironment(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 1000,
    }

    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0):

        self.min_position = -2.4
        self.max_position = 3.6
        self.max_speed = 0.07
        self.goal_position = 3.55
        self.goal_velocity = goal_velocity
        self.straight_line_position=-0.6
        self.fuel_level=60

        self.force = 0.07
        self.gravity = 0.0025

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 800
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def step(self, action: int):
        a=1
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        position, velocity = self.state

        if position < -1:
            velocity = (action-1)*(self.force)+0.5-0.24199399728
            if action==1:
                position+=0
                a=0
            if action!=1:
                position+=velocity

        elif -1 <= position < 0:
            velocity = (action-1)*0.06 +(math.cos(3 * position)*0.8 + 0.55 + 0.5 )* (-self.gravity)
            position+=velocity

        elif 0 < position <= 1:
            velocity += (action-1)*0.06  +(math.cos(3 * position)*0.8 + 0.55 + 0.5 )* (self.gravity)
            k=(math.cos(3 * position)*0.8 + 0.55 + 0.5 )* (self.gravity)
            if action==0:
                position-=k
            if action==1:
                position+=velocity
                a=0
            if action==2:
                position+=velocity

        elif position >= 1:
            velocity += (action-1)*(self.force+0.5-0.241993997280)
            if action==0:
                position-=velocity
            if action==1:
                position+=velocity
                a=0
            if action==2:
                position+=velocity

        if action==2 or action==0:
            self.fuel_level-=1

        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0
        terminated = bool(
            position >= self.goal_position or (self.fuel_level == 0 and position < 0)
        )
        reward = 75*((position+2.4)**1.5) - 4*(60-self.fuel_level)*a
        if position >= self.goal_position and self.fuel_level >= 20 : reward = 1000

        self.state = (position, velocity)
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {'fuel':self.fuel_level}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,

    ):
        super().reset(seed=seed)
        low, high = utils.maybe_parse_reset_bounds(options, -2.3, -2.25 )
        self.state=np.array([self.np_random.uniform(low=low, high=high), 0])
        self.fuel_level = 60
        return np.array(self.state, dtype=np.float32), {}

    def _height(self, xs):

        less_than_minus_one = xs[xs <= -1]
        between_minus_one_and_one = xs[(xs >= -1) & (xs <= 1)]
        greater_than_one = xs[xs >= 1]
        return np.concatenate((np.full((less_than_minus_one.size,), 0.5 - 0.24199399728).astype(float),
        (np.cos(3 * between_minus_one_and_one) * 0.8 + 0.5 + 0.55).astype(float),np.full((greater_than_one.size,),
         0.5 - 0.24199399728).astype(float)))

        #return np.cos(3 * xs)* 0.45 + 0.55
    def height2(self,xs):
        return np.cos(3 * xs)* 0.8 + 0.55 + 0.5

    def render(self, travel = False, position = None):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        font = pygame.font.SysFont('monospace', 12)
        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        carwidth = 40
        carheight = 15

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        if (not travel):
            pos = self.state[0]
        else:
            if (position is None):
                print("Error")
                exit()
            else:
                pos = position


        xs = np.linspace(self.min_position, self.max_position, 1000)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))
        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))
        less_than_minus_one = xs[xs <=-1]
        between_minus_one_and_zero = xs[(xs >= -1) & (xs <= 0)]
        between_zero_and_one = xs[(xs >= 0) & (xs <= 1)]
        greater_than_one = xs[xs >= 1]


        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        #torso of the car
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            if pos < -1:
                c=pygame.math.Vector2(c).rotate_rad(0)

                coords.append(
                    (
                    c[0]+(pos-self.min_position)*scale,
                    c[1]+float((0.5 - 0.24199399728)*scale+10),
                    )
                )

            elif -1<=pos<=-0.34:
                c=pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos)*0.8+1.05)
                coords.append(
                    (
                    c[0] + (pos - self.min_position) * scale,
                    float(c[1] + clearance + self.height2(pos) * scale),
                    )
                )
            elif -0.34<=pos<=0:
                c=pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos)*0.8)
                coords.append(
                    (
                    c[0] + (pos - self.min_position) * scale,
                    float(c[1] + clearance + self.height2(pos) * scale),
                    )
                )
            elif 0<=pos<=0.34:
                c=pygame.math.Vector2(c).rotate_rad(-math.cos(3 * pos)*0.8)
                coords.append(
                    (
                    c[0] + (pos - self.min_position) * scale,
                    float(c[1] + clearance + self.height2(pos) * scale),
                    )
                )
            elif 0.34<=pos<=1:
                c=pygame.math.Vector2(c).rotate_rad(-(math.cos(3 * pos)*0.8+1.05))
                coords.append(
                    (c[0] + (pos - self.min_position) * scale,
                    float(c[1] + clearance + self.height2(pos) * scale),
                )
                )
            elif pos > 1:
                c=pygame.math.Vector2(c).rotate_rad(0)
                coords.append((
                    c[0]+(pos-self.min_position)*scale,
                    c[1]+float((0.5 - 0.24199399728)*scale+10),

                ))
        # print(self.state)
        # print(coords)
        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))
        #wheel of the car
        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            if pos < -1:
                c = pygame.math.Vector2(c).rotate_rad(0)
                wheel = (
                    (
                    int(c[0] + (pos - self.min_position) * scale),
                    math.ceil((0.5 - 0.24199399728)*scale+clearance),
                    )
                )
            elif -1 <= pos <= -0.34:
                c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos)*0.8 + 1.05)
                wheel = (
                    int(c[0] + (pos - self.min_position) * scale),
                    int(c[1] + clearance + self.height2(pos) * scale),
                )
            elif -0.34 <= pos <= 0:
                c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos)*0.8 )
                wheel = (
                    int(c[0] + (pos - self.min_position) * scale),
                    int(c[1] + clearance + self.height2(pos) * scale),
                )
            elif 0<= pos <= 0.34:
                c = pygame.math.Vector2(c).rotate_rad(-math.cos(3 * pos)*0.8 )
                wheel = (
                    int(c[0] + (pos - self.min_position) * scale),
                    int(c[1] + clearance + self.height2(pos) * scale),
                )
            elif 0.34 <= pos <= 1:
                c = pygame.math.Vector2(c).rotate_rad(-(math.cos(3 * pos)*0.8 + 1.05))
                wheel = (
                    int(c[0] + (pos - self.min_position) * scale),
                    int(c[1] + clearance + self.height2(pos) * scale),
                )
            elif pos > 1:
                c = pygame.math.Vector2(c).rotate_rad(0)
                wheel = (
                    (
                    int(c[0] + (pos - self.min_position) * scale),
                    math.ceil((0.5 - 0.24199399728)*scale+clearance),
                    )
                )


            gfxdraw.aacircle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )



        position = font.render("Position: {:.2f}".format(pos), 1, (0,0,0))
        fuel = font.render("Fuel Level: {:.0f}".format(self.fuel_level), 1, (0,0,0))
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        self.screen.blit(position, (680, 10))
        self.screen.blit(fuel, (680, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        time.sleep(1)

    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def travel(self):
        # f = np.linspace(-2.4,-1,150)
        # s = np.linspace(-1,1,450)
        # t = np.linspace(1,3.6,250)

        tot = np.linspace(-2.4, 3.6, 1000)
        # xs = np.concatenate([f,s,t])#t[::-1],s[::-1],f[::-1]])
        for i in tot:
            self.render(travel = True, position = i)
            # time.sleep(1)