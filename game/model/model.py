from turtle import window_width, window_height
import pygame
import turtle

class Character:
    def __init__(self):
        # 創建 turtle 物件
        self.turtle = turtle.Turtle()
        # 設置 turtle 顏色為黃色
        self.turtle.color("yellow")
        # 隱藏 turtle 箭頭
        self.turtle.hideturtle()
        # 設置橢圓形的寬和長
        self.turtle.shapesize(stretch_wid=5, stretch_len=2)
        # 設置 turtle 形狀為橢圓形
        self.turtle.shape("circle")

    def draw_ellipse(self):
        # 繪製橢圓形
        self.turtle.begin_fill()
        self.turtle.end_fill()

# 創建 Character 物件
my_character = Character()

# 繪製黃色橢圓形
my_character.draw_ellipse()

# 顯示視窗
turtle.mainloop()


class Ball:
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color

    def update(self):
        # 更新球的座標
        self.x += self.vx
        self.y += self.vy

        # 檢查是否碰撞邊界
        if self.x < 0 or self.x > window_width:
            self.vx = -self.vx
        if self.y < 0 or self.y > window_height:
            self.vy = -self.vy

    def draw(self, surface):
        # 繪製球
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.radius)