import gym
import numpy as np
import cv2

class MyEnv(gym.Env):
    def __init__(self):
        self.FIELD_SIZE = 10 #画面サイズの決定 正方形
        self.POSITION_SIZE = 8 #初期値のサイズの決定
        self.WINDOW_SIZE = 600
        self.GOAL_RANGE = 0.25 #ゴールの範囲設定(直径)
        self.TS = 0.2 #制御周期[s]
        
        # アクション定義
        self.INPUT_DIM = 2 #入力数
        self.ACTION_HIGH = np.array([0.5,np.pi*30/180])
        self.ACTION_LOW = np.array([-0.5,-np.pi*30/180])
        self.action_space = gym.spaces.Box(low=np.float32(self.ACTION_LOW),high=np.float32(self.ACTION_HIGH))

        # 状態の範囲を定義
        self.OUTPUT_DIM = 6 #出力数
        self.STATE_LOW = np.array([0.,0.])
        self.STATE_HIGH = np.array([10.,10.])
        self.observation_space = gym.spaces.Box(low=np.float32(self.STATE_LOW), high=np.float32(self.STATE_HIGH))

        # 状態方程式
        # self.A = np.array([[1,0,0],[0,1,0],[0,0,1]])
        # self.B = np.array([[1,0],[0,1],[0,1]])
        
        # 重み行列
        self.Q = np.eye(2)*0.5
        self.R = np.eye(2)*0.5
        self.S = 0.5
        
        # カートのパラメータ設定
        self.vehicle_d = 0.1
        self.vehicle_width = 0.1
        self.vehicle_height = 0.2
        
        # 経過ステップ数を定義
        self.time_step = 0
        
        # 最大経過ステップ数
        self.max_step = 2500
        
        self.reset()

    def reset(self):
        # カートとゴールの位置をランダムで初期化
        vehicle_center = np.array([np.random.rand(), np.random.rand()]) * self.POSITION_SIZE + 1
        self.goal_position = np.array([np.random.rand(), np.random.rand()]) * self.POSITION_SIZE + 1

        # カートの初期角度
        vehicle_theta = (np.random.rand() * 2 - 1) * np.pi # [-pi:pi]
        
        # 制御点
        control_point = vehicle_center  + np.array([self.vehicle_d*np.cos(vehicle_theta),self.vehicle_d*np.sin(vehicle_theta)])
        
        # カートの状態
        self.vehicle_state = np.hstack((control_point,vehicle_theta))
        
        # モデルに渡す状態の作成
        vec = self.goal_position - control_point
        rad = self._phase(np.arctan2(vec[1],vec[0]) - self.vehicle_state[2]) # 相対角度
        observation = np.hstack((vec, rad, self.vehicle_state))
        self.time_step = 0
        
        return observation

    def step(self, action):
        
        # -1:1 -> min:max に変換
        action = self._min_max_decode(action,self.ACTION_HIGH,self.ACTION_LOW)
        
        # 時変の状態方程式
        A, B = self._getStateMatrix(self.vehicle_state)
        
        #前の相対距離と相対角度
        # before_vec = self.goal_position - self.vehicle_state[0:2]
        # before_rad = self._phase(np.arctan2(before_vec[1],before_vec[0]) - self.vehicle_state[2])

        # 1ステップ進める
        self.vehicle_state = A @ self.vehicle_state + B @ (action * self.TS)
        self.vehicle_state[2] = self._phase(self.vehicle_state[2])
        # 制御点
        control_point = self.vehicle_state[0:2]
        # vehicle_theta = self.vehicle_state[2]
        
        # カートの中心
        # vehicle_center = control_point - np.array([self.vehicle_d*np.cos(vehicle_theta),self.vehicle_d*np.sin(vehicle_theta)] )
    
        # 状態の作成
        vec = self.goal_position - control_point # 相対距離
        rad = self._phase(np.arctan2(vec[1],vec[0]) - self.vehicle_state[2]) # 相対角度
        observation = np.hstack((vec, rad, self.vehicle_state)) # 相対距離、グローバル座標(x,y, theta)
        
        # 報酬の計算
        distance = np.linalg.norm(vec)  # 距離の計算
        # reward = (self.before_distance - distance) # どれだけゴールに近づいたか
        # reward = (before_vec.T @ self.Q @ before_vec) - (vec.T @ self.Q @ vec) + (before_rad**2 - rad**2) *150
        reward = - (vec.T @ self.Q @ vec) - rad**2 * self.S
        # 時間を進める
        self.time_step += 1
        
        # 終了判定 false -> continue, true -> done
        done = False
        if distance < self.GOAL_RANGE:
            done = True
            reward += 150000
        if  self.max_step <= self.time_step:
            done = True
            reward += -100000
        if np.any(control_point < np.array([0,0])) or np.any(control_point > np.array([self.FIELD_SIZE,self.FIELD_SIZE])):
            done = True
            reward += -100000
        # self.before_distance = distance

        return observation, reward, done, {}
    
    def _getStateMatrix(self,state):
        # 時変の状態方程式
        A = np.array([[1,0,0],
                      [0,1,0],
                      [0,0,1]])
        B = np.array([[np.cos(state[2]),-self.vehicle_d*np.sin(state[2])],
                      [np.sin(state[2]),self.vehicle_d*np.cos(state[2])],
                      [0,1]])
        return A, B

    def _min_max_decode(self, x_norm, source_min, source_max):
            return ((x_norm * source_max) - (x_norm * source_min) + source_max + source_min) * 0.5
        
    def _phase(self,theta):
        if theta >= np.pi:
            theta = theta - 2 * np.pi
        elif theta <= -np.pi:
            theta = theta + 2 * np.pi
        return theta 
    
    def render(self,mode="human"):        
        scale = self.WINDOW_SIZE / self.FIELD_SIZE
        
        image = np.zeros((self.WINDOW_SIZE, self.WINDOW_SIZE, 3))
        for i in range(1,self.FIELD_SIZE):
            w  = np.int32(np.round(i*scale))
            cv2.line(image,(w,0),(w,self.WINDOW_SIZE),color=(255,255,255),thickness=1)
            cv2.line(image,(0,w),(self.WINDOW_SIZE,w),color=(255,255,255),thickness=1)
        cv2.circle(image, tuple(np.int32(np.round(self.goal_position*scale))), 5, (0, 255, 0), thickness=-1) #ゴールの描画 (B,G,R)
        cv2.circle(image, tuple(np.int32(np.round(self.goal_position*scale))), np.int32(np.round((self.GOAL_RANGE)*scale)), color=(255,255,0), thickness=2) #ゴールの範囲の描画

        # カートを描画
        cv2.fillConvexPoly(image, self._rotatedRectangle(self.vehicle_state,scale), color=(255,0,255))
        #制御点の描画 (B,G,R)
        cv2.circle(image, tuple(np.int32(np.round(self.vehicle_state[:2]*scale))), 5, (0, 0, 255), thickness=-1) 
        
        # 座標系を数学にする
        image = np.flipud(image)
        
        if mode == 'human':
            cv2.imshow('image', image)
            cv2.waitKey(5)
            return image
        return image
    
    def _rotatedRectangle(self,state,scale):
        angle = state[2] - np.pi/2
        center = state[:2] - np.array([self.vehicle_d*np.cos(state[2]),self.vehicle_d*np.sin(state[2])])
        x = center[0] * scale
        y = center[1] * scale
        
        width = self.vehicle_width * scale
        height = self.vehicle_height * scale
        
        # 回転する前の矩形の頂点
        before_pts = np.array([
            [(x + width / 2), (y + height / 2), 1],
            [(x + width / 2), (y - height / 2), 1],
            [(x - width / 2), (y - height / 2), 1],
            [(x - width / 2), (y + height / 2), 1]])
    
        # 変換行列
        t = np.array([[np.cos(angle),   -np.sin(angle), x-x*np.cos(angle)+y*np.sin(angle)],
                    [np.sin(angle), np.cos(angle),  y-x*np.sin(angle)-y*np.cos(angle)],
                    [0,             0,              1]])
        
        rorated_pts = t @ before_pts.T
        
        points = rorated_pts.T[:,:2]
        # condition = points < 0
        # points[condition] = 0
        # condition = points > self.WINDOW_SIZE
        # points[condition] = self.WINDOW_SIZE
        # print(points)
        return np.int32(np.round(points))