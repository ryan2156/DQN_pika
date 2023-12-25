import sys
import pygame
from pygame.locals import QUIT, KEYDOWN, KEYUP
import time

WIDTH = 800
HEIGHT = 600


class Character:
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.vx = 0
        
        self.radius = radius
        self.color = color

        # 建立 Pygame 的畫布
        self.surface = pygame.Surface((radius * 2, radius * 2))
        self.surface.fill(color)
        
        self.rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)

    def draw(self, surface):
        # 將畫布貼上到視窗上
        # 使用 pygame.draw.ellipse() 方法繪製橢圓形
        pygame.draw.ellipse(surface, self.color, (self.x - self.radius, self.y - self.radius, self.radius*1.5, 2 * self.radius))
    def update(self):
        # 更新角色的座標
        self.x += self.vx

        # 檢查是否碰撞邊界
        if self.x < 0 or self.x > WIDTH:
            self.vx = -self.vx


class Ball:
    def __init__(self, x, y, radius, color, vx=0, vy=0):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.vx = vx  # 初始化 vx 為 0
        self.vy = vy  # 初始化 vy 為 0

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

        # 更新球的位置
        self.x += self.vx
        self.y += self.vy
        
        # print(self.x,self.vx,self.y,self.vy)

        # 檢查是否碰撞邊界
        if self.x < 0 or self.x > WIDTH:
            self.vx = -self.vx
        if self.y < 0 or self.y > HEIGHT:
            self.vy = -self.vy

        # 檢查是否碰撞角色
        if self.character and self.collide(self.character):  # 檢查 character 是否存在
            # 反彈球的方向
            self.vx = -self.vx

    def collide(self, other: Character):
        # 檢查兩個物件的中心點是否相距小於兩個物件的半徑之和
        d = ((self.x - other.x)**2 + (self.y - other.y)**2) ** 0.5
        r = self.radius + other.radius

        # 檢查碰撞點是否在 character 物件的範圍內
        if d <= r:
            # 反彈球的方向
            self.vx = -1.0069*self.vx 
            self.vy = -1.0069*self.vy
            
            # 限制球速
            max_speed = 15.6
            if self.vx < 0:
                self.vx = -min(abs(self.vx), max_speed)
            else:
                self.vx = min(abs(self.vx), max_speed)

            if self.vy < 0:
                self.vy = -min(abs(self.vy), max_speed)
            else:
                self.vy = min(abs(self.vy), max_speed)

            # 檢查碰撞點是否在 character 物件的範圍內
            return pygame.Rect.collidepoint(other.rect, (self.x, self.y))
        else:
            return False

        


class Floor:
    def __init__(self, width, height, color):
        self.width = width
        self.height = height
        self.color = color

    def draw(self, surface, y):
        pygame.draw.rect(surface, self.color, (0, y, self.width, self.height))


class Controller:
    def __init__(self, character, speed=5):
        self.character = character
        self.speed = speed

    def handle_event(self, event):
        if event.type == KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.move_left()
            elif event.key == pygame.K_RIGHT:
                self.move_right()
        elif event.type == KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                self.stop()

    def move_left(self):
        self.character.vx = -self.speed

    def move_right(self):
        self.character.vx = self.speed

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

def reset_game(character, ball):
    # 重新開始遊戲，將球的位置和速度重新設定為初始值
    ball.x = 400
    ball.y = 100
    ball.vx = 0
    ball.vy = 0
    # 重新開始遊戲，將角色的位置和速度重新設定為初始值
    character.x = 475
    character.vx = 0
    
def main():
    pygame.init()  # 初始化 Pygame

    window = Window()  # 建立視窗物件

    ball = Ball(400, 100, 50, (0, 255, 0))  # 建立球物件
    character = Character(475, 375, 75, (255, 0, 0))  # 建立角色物件
    controller = Controller(character)
    
    # 將 character 物件傳遞給 ball 物件
    ball.character = character
    while True:  # 遊戲迴圈
        time.sleep(0.016)
        for event in pygame.event.get():  # 處理事件
            print(event)
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == pygame.K_r:  # 按下 'r' 鍵重置遊戲
                    reset_game(character, ball)
            controller.handle_event(event)

        ball.update()  # 更新球的狀態
        character.update()  # 更新角色的狀態（如果有）

        # 檢查球是否碰撞角色
        if check_collide(ball, character):
            # 反彈球的方向
            ball.vx = -ball.vx
            ball.vy = -ball.vy

        window.draw(character, ball)  # 繪製畫面

        pygame.display.update()  # 更新視窗顯示
        # 檢查是否需要重新開始遊戲
        if ball.y > HEIGHT:
            reset_game(character, ball)
if __name__ == "__main__":
    main()