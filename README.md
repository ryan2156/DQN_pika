## 建置環境：
### 1. 安裝虛擬環境 
```
pip install virtualenv
```
```
virtualenv venv
```
### 2. 啟動虛擬環境

MacOS:
```
source venv/bin/activate
```
Windows:
```
.\venv\Scripts\activate
```
### 3. 下載相關套件
MacOS:
```
pip install -r requirements.txt
```
## 啟動：
```
python DQN/main.py
```
### 其他：
    模型紀錄存在 DQN/model/
