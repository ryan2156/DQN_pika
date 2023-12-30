# 機器學習
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import torch.nn.functional as F

# 遊戲的model
import constants as const
from game_model import *
import random

# 繪圖
import matplotlib.pyplot as plt

# 紀錄檔
import pickle

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
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 128, dtype=torch.float) # 輸入層
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size) # 輸出層

    def forward(self, x):
        #print("0Input size:", x.size())
        x = x.view(-1, self.input_size) 
        x = F.elu(self.fc1(x))
        #print("1Input size:", x.size())
        # print(x)
        x = F.elu(self.fc2(x))
        #print("2Input size:", x.size())
        #print(x)
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
        self.times = 0 # 紀錄次數
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32) # 將list 轉為tensor
        state = state.view(1, -1)
        action = self.model.forward(state)
        
        self.epsilon = self.epsilon * self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

        if random.uniform(0, 0.5) < self.epsilon:
            action = random.choice(range(self.action_size))
        else:
            state = torch.tensor(state, dtype=torch.float32).view(1, -1)
            q_values = self.model(state).detach().numpy()
            return np.argmax(q_values[0])
        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
    
        # 隨機抽取一批樣本
        minibatch = random.sample(self.memory, batch_size)

        # 獲取下一個狀態的 Q 值
        next_state_batch = [x[3] for x in minibatch]
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
        next_state_batch = next_state_batch.view(-1, 6)

        # 計算目標 Q 值
        for i in range(batch_size):
            action = minibatch[i][1]
            reward = minibatch[i][2] 
            done = minibatch[i][4] 
            

            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.model(next_state_batch[i]).detach().numpy())

            current_q_values = self.model(minibatch[i][0])
            target_q_values = current_q_values.clone().detach()
            target_q_values[0, action] = target
            loss = nn.MSELoss()(current_q_values, target_q_values)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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
    

    state_size = 6  # 修改為你的狀態大小
    action_size = 3  # 修改為你的動作大小
    agent = DQNAgent(state_size, action_size)
    finished_list = [] # 每完成50個訓練就加一筆
    epochs = 10
    
    for epoch in range(epochs):
        reset_game(character, ball, score_area)
        print("epoch: ", epoch+1)
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
            state = [character.x, character.vx, ball.x, ball.vx, ball.y, ball.vy]
            state = state = torch.tensor(state, dtype=torch.float32).view(1, -1)
            action = agent.act(state)
            agent.times += 1
            
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
            next_state = [character.x, character.vx, ball.x, ball.vx, ball.y, ball.vy]  # 修改為你的狀態表示
            done = 0
            # reward 設定
            if(check_collide(ball, character)):  # 球與角色碰撞
                reward = 1
            elif(score_area.check_collision(ball)): # 球進到得分區域
                reward = 5
                score_area.score += 1
            elif(ball.y > HEIGHT-150): # 球落地
                reward = -1
                done = ball.y > HEIGHT-150
            else:
                reward = 0
            
            agent.remember(state, action, reward, next_state, done)

            # 使用DQNAgent進行回放學習
            agent.replay(32)
            agent.times += 1
            
            window.draw(character, ball)  # 繪製畫面
            
            if score_area.check_collision(ball) and addmode == 0:
                addmode = 1
                score_area.increase_score()
                # print(score_area.score)
                #score_area.draw(window.surface)  # Draw the score area after updating the score
            elif score_area.check_collision(ball) and addmode == 1:
                pass
            else:
                addmode = 0
            # Draw the score table (draw outside the collision check)
            score_area.draw(window.surface)
            pygame.display.update()  # 更新視窗顯示
            
            # 檢查是否觸地
            if done:
                break
        if(not epoch % 50):
            finished_list.append(score_area.score)
        print("探索率：" , agent.epsilon, "損失：", agent.gamma)
        print()
    
    torch.save({
        "model": agent.model.state_dict(),
        "times": agent.times,
    }, "model/model.pth")
    
    x_axis = [i for i in range(0, len(finished_list))]
    
    plt.plot(x_axis, finished_list, label="score")

    plt.title("the score of Agent play")
    plt.xlabel("times(50)")
    plt.ylabel("score")
    
    plt.show()
        

if __name__ == "__main__":
    main()