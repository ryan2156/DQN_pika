import sys
import pygame
from pygame.locals import QUIT, KEYDOWN, KEYUP

WIDTH = 800
HEIGHT = 600


class Ball:
    def __init__(self, x, y, radius, color, vx=0, vy=0):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.vx = vx  # 初始化 vx 為 0
        self.vy = vy  # 初始化 vy 為 0

    def update(self):
        # 更新球的座標
        self.x += self.vx
        self.y += self.vy

        # 檢查是否碰撞邊界
        if self.x < 0 or self.x > WIDTH:
            self.vx = -self.vx
        if self.y < 0 or self.y > HEIGHT:
            self.vy = -self.vy
    def draw(self, surface):
        # 建立 Pygame 的畫布
        self.surface = pygame.Surface((self.radius * 2, self.radius * 2))
        self.surface.fill(self.color)

        # 將畫布貼上到視窗上
        surface.blit(self.surface, (self.x - self.radius, self.y - self.radius))
class Controller:
    def __init__(self, speed=5):
        # 控制器的初始位置
        self.x = 400
        self.y = 200
        #移動速度
        self.speed = speed
        # 新增殺球的相關屬性
        self.ball_x = self.x
        self.ball_y = self.y - 20  # 在角色頭上的位置
        self.is_hitting = False  # 是否正在殺球
        self.jump_height = 0      # 跳躍高度

    #根據按下或釋放的按鍵觸發相應的移動或停止操作。
    def handle_event(self, event):
        if event.type == KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.move_left()
            elif event.key == pygame.K_RIGHT: 
                self.move_right()
            elif event.key == pygame.K_RETURN: #殺球
                self.hit_ball()
            elif event.key == pygame.K_UP: 
                self.jump()
            elif event.key == pygame.K_DOWN:
                self.hit_down()

        elif event.type == KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                self.stop()
    # 移動控制
    def move_left(self):
        self.x -= self.speed

    def move_right(self):
        self.x += self.speed

    def stop(self):
        self.x = 0

    # 殺球控制
    def hit_ball(self):
        self.is_hitting = True
        self.ball_vx = self.x - self.ball_x
        self.ball_vy = self.y - self.ball_y

    def hit_down(self):
        self.jump_height = 200

    # 跳躍控制
    def update(self):
        # 更新角色的座標
        if self.jump_height > 0:
            self.y -= self.jump_height * 0.01
            self.jump_height -= 1

        # 檢查是否碰撞邊界
        if self.x < 0 or self.x > WIDTH:
            self.x = 0
        if self.y < 0 or self.y > HEIGHT:
            self.y = 0
class Character:
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.radius = radius
        self.color = color

        # 建立 Pygame 的畫布
        self.surface = pygame.Surface((radius * 2, radius * 2))
        self.surface.fill(color)

    def draw(self, surface):
        # 將畫布貼上到視窗上
        surface.blit(self.surface, (self.x - self.radius, self.y - self.radius))
    def update(self):
        # 更新角色的座標
        self.x += self.vx
        self.y += self.vy

        # 檢查是否碰撞邊界
        if self.x < 0 or self.x > WIDTH:
            self.vx = -self.vx
        if self.y < 0 or self.y > HEIGHT:
            self.vy = -self.vy
class Window:
    def __init__(self):
        self.width = 800
        self.height = 600

        self.surface = pygame.display.set_mode((self.width, self.height))

    def draw(self, character, ball):
        self.surface.fill((0, 0, 0))

        character.draw(self.surface)
        ball.draw(self.surface)

        pygame.display.update()

def main():
    pygame.init()  # 初始化 Pygame

    window = Window()  # 建立視窗物件

    ball = Ball(400, 200, 50, (0, 255, 0))  # 建立球物件
    character = Character(400, 200, 50, (255, 0, 0))  # 建立角色物件

    while True:  # 遊戲迴圈
        for event in pygame.event.get():  # 處理事件
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        ball.update()  # 更新球的狀態
        character.update()  # 更新角色的狀態（如果有）

        window.draw(character, ball)  # 繪製畫面

        pygame.display.update()  # 更新視窗顯示

if __name__ == "__main__":
    main()