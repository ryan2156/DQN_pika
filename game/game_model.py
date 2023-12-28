import sys
import pygame
from pygame.locals import QUIT, KEYDOWN, KEYUP
import time
import math
import random

import constants as const

# 遊戲常數
# 視窗
WIDTH = const.WIDTH
HEIGHT = const.HEIGHT

# 地板高度
FLOOR_HEIGHT = const.FLOOR_HEIGHT

# 角色大小
CHARACTER_SIZE = const.CHARACTER_SIZE

# 球的係數
BALL_SIZE = const.BALL_SIZE
BALL_MAX_SPEED = const.BALL_MAX_SPEED

class Character:
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.vx = 0
        
        self.radius = radius
        self.color = color

        # 建立 Pygame 的畫布，並將其初始化為指定顏色
        self.rect = pygame.Surface((radius * 2, radius * 2)).fill(color)
        
    def draw(self, surface):
        # 將畫布貼上到視窗上
        pygame.draw.rect(surface, self.color, (self.x - self.radius, self.y - self.radius, self.radius, self.radius))
    def update(self):
        # 檢查是否碰撞邊界
        if self.x < CHARACTER_SIZE:
            self.vx = 1
        elif self.x > WIDTH:
            self.vx = -1
        # 更新角色的座標
        self.x += self.vx

class Ball:
    def __init__(self, x, y, radius, color, vx=0, vy=0):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.vx = vx  # 初始化 vx 為 0
        self.vy = vy  # 初始化 vy 為 0

        # Rect屬性
        self.rect = pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)

        # 新增 character 屬性
        self.character = None

    def draw(self, surface):
        # 使用 pygame.draw.circle() 方法繪製圓形
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.radius)

    def update(self):
        # 更新球的座標
        # 計算重力加速度
        gravity = 0.4

        # 更新球的速度
        self.vy += gravity

        
        # 球速上限
        if(self.vx > 0):
            if(self.vx > BALL_MAX_SPEED):
                self.vx = BALL_MAX_SPEED
        elif(self.vx < 0):
            if(abs(self.vx > BALL_MAX_SPEED)):
                self.vx = - BALL_MAX_SPEED
        if(self.vy > 0):
            if(self.vy > BALL_MAX_SPEED):
                self.vy = BALL_MAX_SPEED
        elif(self.vy < 0):
            if(abs(self.vy > BALL_MAX_SPEED)):
                self.vx = - BALL_MAX_SPEED
        
        # 更新球的位置
        self.x += self.vx
        self.y += self.vy

        # 檢查是否碰撞邊界
        if self.x < 0 or self.x > WIDTH:
            self.vx = -self.vx
        if self.y < 0 or self.y > HEIGHT:
            self.vy = -self.vy
            

        # # 檢查是否碰撞角色
        # if self.collide(self.character):  # 檢查 character 是否存在
        #     # 反彈球的方向
        #     self.vx = -self.vx
            
    def angleCalculate(self, character: Character):
        del_x = (character.x-self.x)
        del_y = (character.y-self.y)
        
        if(not del_x or not del_y):
            if(not del_x):
                return [0, 1]
            if(not del_y):
                return [1, 0]
        
        hypo = (del_x**2 + del_y**2)**0.5
        
        sin, cos = del_x/hypo, del_y/hypo
        
        
        return [sin, cos]
        
    def collide(self, character: Character):
        # 檢查球與角色是否相交
        if(self.character):
            # 計算球與角色的中心距離
            distance = math.sqrt((self.x - character.x)**2 + (self.y - character.y)**2)

            # 檢查中心距離是否小於球與角色的半徑之和
            if(distance <= self.radius + character.radius):
                # 球與角色相交
                return True
            else:
                # 球與角色不相交
                return False
        else:
            # 沒有角色
            return False

class ScoreArea:
    def __init__(self):
        self.size = 50 * 2
        self.score = 0
        self.randomize_position()

    def randomize_position(self):
        self.x = random.randint(100, 700)
        self.y = random.randint(30, 200)

    def draw(self, surface):
        pygame.draw.rect(surface, (255, 255, 255), (self.x, self.y, self.size, self.size))

    def check_collision(self, ball: Ball):
        xBool = (self.x < ball.x+25 and ball.x+25 < self.x + self.size) or (self.x < ball.x-25 and ball.x-25 < self.x + self.size)
        yBool = (self.y < ball.y+25 and ball.y+25 < self.y + self.size) or (self.y < ball.y-25 and ball.y-25 < self.y + self.size)
        return xBool and yBool

    def increase_score(self):
        self.score += 1
        self.randomize_position()

class Floor:
    def __init__(self, width, height, color):
        self.width = width
        self.height = height
        self.color = color

    def draw(self, surface, y):
        pygame.draw.rect(surface, self.color, (0, y, self.width, self.height))

class Controller:
    def __init__(self, character: Character, speed=5):
        self.character = character
        self.speed = speed

    def handle_event(self, event):
        if event.type == KEYDOWN:
            if event.key == pygame.K_LEFT:
                if(self.character.x < CHARACTER_SIZE):
                    self.stop()
                else:
                    self.move_left()
            elif event.key == pygame.K_RIGHT:
                if(self.character.x > WIDTH):
                    self.stop()
                else:
                    self.move_right()
        elif event.type == KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                self.stop()

    def move_left(self):
        self.character.vx = -self.speed*1.5

    def move_right(self):
        self.character.vx = self.speed*1.5

    def stop(self):
        self.character.vx = 0

    def update(self):
        self.character.update()

        if self.character.x < 0 or self.character.x > WIDTH:
            self.character.vx = 0
            self.character.x = max(0, min(self.character.x, WIDTH))
            
class Window:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT

        self.surface = pygame.display.set_mode((self.width, self.height))
        self.floor = Floor(WIDTH, FLOOR_HEIGHT, (0, 0, 255))  # 新增地板物件
        
    def draw(self, character: Character, ball: Ball):
        self.surface.fill((0, 0, 0))

        self.floor.draw(self.surface, HEIGHT-FLOOR_HEIGHT)  # 繪製地板
        character.draw(self.surface)
        ball.draw(self.surface)

        pygame.display.update()