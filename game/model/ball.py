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
        if self.x < 0 or self.x > Window.width:
            self.vx = -self.vx
        if self.y < 0 or self.y > Window.height:
            self.vy = -self.vy
        # 檢查球是否在角色的頭上
        if self.y <= character.y + 20:
            # 殺球
            self.ball_vx = self.x - self.ball_x
            self.ball_vy = self.y - self.ball_y
        else:
            # 反彈
            self.vx = -self.vx
            self.vy = -self.vy
