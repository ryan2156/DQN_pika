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
        if self.x < 0 or self.x > Window.width:
            self.x = 0
        if self.y < 0 or self.y > Window.height:
            self.y = 0
