class Character:
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color

        # 建立 Pygame 的畫布
        self.surface = pygame.Surface((radius * 2, radius * 2))
        self.surface.fill(color)

    def draw(self, surface):
        # 將畫布貼上到視窗上
        surface.blit(self.surface, (self.x - self.radius, self.y - self.radius))
    # 球的反彈效果
    def update(self):
        # 更新球的座標
        self.x += self.vx
        self.y += self.vy

        # 檢查是否碰撞邊界
        if self.x < 0 or self.x > Window.width:
            self.vx = -self.vx
        if self.y < 0 or self.y > Window.height:
            self.vy = -self.vy

        # 檢查是否碰撞角色
        if self.x >= character.x - character.radius and self.x <= character.x + character.radius and \
                self.y >= character.y - character.radius and self.y <= character.y + character.radius:
            # 判斷球是否在角色的頭上
            if self.y <= character.y + 20:
                # 殺球
                self.ball_vx = self.x - self.ball_x
                self.ball_vy = self.y - self.ball_y
            else:
                # 反彈
                self.vx = -self.vx
                self.vy = -self.vy