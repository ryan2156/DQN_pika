import sys
import pygame
from pygame.locals import QUIT, KEYDOWN, KEYUP
import time
import torch
import math
import random

WIDTH = 800
HEIGHT = 600

BALL_MAX_SPEED = 12

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
        if self.x < 100:
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
                if(self.character.x < 100):
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
        self.width = 800
        self.height = 600

        self.surface = pygame.display.set_mode((self.width, self.height))
        self.floor = Floor(800, 150, (0, 0, 255))  # 新增地板物件
        
    def draw(self, character: Character, ball: Ball):
        self.surface.fill((0, 0, 0))

        self.floor.draw(self.surface, 450)  # 繪製地板
        character.draw(self.surface)
        ball.draw(self.surface)

        pygame.display.update()

def check_collide(ball: Ball, character: Character):
    # 檢查球是否碰撞角色
    return ball.collide(character)

def reset_game(character: Character, ball: Ball, score: ScoreArea):
    # 重新開始遊戲，將球的位置和速度重新設定為初始值
    ball.x = 400
    ball.y = 100
    ball.vx = 0
    ball.vy = 0
    # 重新開始遊戲，將角色的位置和速度重新設定為初始值
    character.x = 450
    character.vx = 0
    score.score = 0
    
def main():
    addmode = 0
    pygame.init()  # 初始化 Pygame

    window = Window()  # 建立視窗物件

    ball = Ball(400, 100, 50, (0, 255, 0))  # 建立球物件
    character = Character(450, 450, 100, (255, 255, 0))  # 建立角色物件
    controller = Controller(character)
    
    score_area = ScoreArea()
    ball.character = character
    while True:  # 遊戲迴圈
        time.sleep(0.016)
        for event in pygame.event.get():  # 處理事件
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == pygame.K_r:  # 按下 'r' 鍵重置遊戲
                    reset_game(character, ball)
            controller.handle_event(event)

        ball.update()  # 更新球的狀態
        character.update()  # 更新角色的狀態（如果有）
        
        #print(ball.x, character.x)

        # 檢查球是否碰撞角色
        if check_collide(ball, character):
            # 反彈球的方向
            ball.vx += -character.vx * 0.05
            ball.vy = -ball.vy * 1.18

        window.draw(character, ball)  # 繪製畫面
        
        if score_area.check_collision(ball) and addmode == 0:
            addmode = 1
            score_area.increase_score()
            print(score_area.score)
            #score_area.draw(window.surface)  # Draw the score area after updating the score
        elif score_area.check_collision(ball) and addmode == 1:
            pass
        else:
            addmode = 0
        # Draw the score table (draw outside the collision check)
        score_area.draw(window.surface)
        pygame.display.update()  # 更新視窗顯示
        
        # 檢查是否需要重新開始遊戲
        if ball.y > HEIGHT - 150:
            reset_game(character, ball, score_area)
            print(score_area.score)

if __name__ == "__main__":
    main()