# 機器學習
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

# 遊戲的model
import constants as const
from game_model import *
import random

# 繪圖
import matplotlib.pyplot as plt

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

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128) # 輸入層
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size) # 輸出層

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = torch.tensor(state, dtype=torch.float32).view(1, -1)
            q_values = self.model(state).detach().numpy()
            return np.argmax(q_values)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # 隨機抽取一批樣本
        minibatch = random.sample(self.memory, batch_size)

        # 獲取下一個狀態的 Q 值
        state_batch = torch.tensor([x[0] for x in minibatch], dtype=torch.float32)
        next_state_batch = torch.tensor([x[3] for x in minibatch], dtype=torch.float32)
        next_q_values = self.model(next_state_batch).detach().numpy()

        # 計算目標 Q 值
        for i in range(batch_size):
            action = minibatch[i][1]
            reward = minibatch[i][2]  # 從minibatch中獲取獎勵
            target = reward
            if not minibatch[i][4]:  # 檢查遊戲是否結束
                # 如果遊戲沒有結束，則計算目標Q值
                target = reward + self.gamma * np.max(next_q_values[i])

            # 更新Q表
            self.model.update_parameters(state_batch[i], action, target)

def check_collide(ball: Ball, character: Character):
    # 檢查球是否碰撞角色
    return ball.collide(character)

def reset_game(character: Character, ball: Ball, score: ScoreArea):
    # 重新開始遊戲，將球的位置和速度重新設定為初始值
    ball.x = WIDTH//2
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

    ball = Ball(400, 100, BALL_SIZE, (0, 255, 0))  # 建立球物件
    character = Character(450, 450, CHARACTER_SIZE, (255, 255, 0))  # 建立角色物件
    controller = Controller(character) # 控制
    score_area = ScoreArea() # 得分區域
    ball.character = character # 球要根據角色判斷
    
    # 每局得分列表
    scores = []
    
    state_size = 6  # 修改為你的狀態大小
    action_size = 3  # 修改為你的動作大小
    agent = DQNAgent(state_size, action_size)
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

        # 使用DQNAgent選擇動作
        state = [character.x, character.vx, ball.x, ball.vx, ball.y, ball.vy]  # 修改為你的狀態表示
        action = agent.act(state)
        
        # 將動作應用到遊戲中
        if action == 0:
            controller.move_left()
        elif action == 1:
            controller.move_right()
        # 如果 action == 2，則表示停止，無需額外處理

        ball.update()  # 更新球的狀態
        character.update()  # 更新角色的狀態（如果有）
        
        #print(ball.x, character.x)

        # 檢查球是否碰撞角色
        if check_collide(ball, character):
            # 反彈球的方向
            ball.vx += -character.vx * 0.05
            ball.vy = -ball.vy * 1.18

        # 將當前狀態、動作、獎勵和下一個狀態儲存到DQNAgent的記憶體中
        next_state = [character.x, character.vx, ball.x, ball.vx]  # 修改為你的狀態表示
        reward = 1 if check_collide(ball, character) else 0  # 設定獎勵
        done = ball.y > HEIGHT
        agent.remember(state, action, reward, next_state, done)

        # 使用DQNAgent進行回放學習
        agent.replay(32)

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
            scores.append(score_area.score)
            # 繪製圖表
            plt.plot(scores)
            plt.xlabel("Game number")
            plt.ylabel("Score")
            plt.show()
            reset_game(character, ball, score_area)
        
        

if __name__ == "__main__":
    main()