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
        
        # 予測ステップ数
        self.PREDICTION_HORIZON = 5
        
        # 状態方程式
        self.A = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.B = np.array([[1,0],[0,1],[0,1]])
        
        # 入力次元
        self.INPUT_DIM = self.B.shape[1]
        self.STATE_DIM = self.B.shape[0]
        
        # アクション定義
        self.ACTION_DIM = self.INPUT_DIM * self.PREDICTION_HORIZON #入力数
        self.ACTION_HIGH = np.array([0.5,np.pi*30/180])
        self.ACTION_LOW = np.array([-0.5,-np.pi*30/180])
        self.action_space = gym.spaces.Box(low=np.float32(self.ACTION_LOW),high=np.float32(self.ACTION_HIGH))

        # 状態の範囲を定義
        self.OUTPUT_DIM = 6 #出力数
        self.STATE_LOW = np.array([0.,0.])
        self.STATE_HIGH = np.array([10.,10.])
        self.observation_space = gym.spaces.Box(low=np.float32(self.STATE_LOW), high=np.float32(self.STATE_HIGH))
        
        # 重み行列
        self.Q = np.eye(2)*0.1
        self.R = np.eye(2)*0.1
        self.S = 0.1
        
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
        
        self.predict_states = np.tile(self.vehicle_state,(self.PREDICTION_HORIZON,1))
        
        return observation

    def step(self, action):
        
        # 時変の状態方程式
        # A ,B = self._getStateMatrix(self.vehicle_state)

        # before_vec = np.tile(self.goal_position,(self.PREDICTION_HORIZON,1)) - self.predict_states[:,0:2]
        # before_rad = np.array([self._phase(np.arctan2(bv[1],bv[0]) - ps[2]) for ps, bv in zip(self.predict_states,before_vec)])
        
        # 入力の整形
        input_list = self._makeAction(action)
        
        # 状態の予測
        self.predict_states = self._prediction(self.vehicle_state,input_list)
        
        # 1ステップ進める
        self.vehicle_state = self.predict_states[0]

        # 制御点
        control_point = self.vehicle_state[0:2]
        # vehicle_theta = self.vehicle_state[2]
        
        # カートの中心
        # vehicle_center = control_point - np.array([self.vehicle_d*np.cos(vehicle_theta),self.vehicle_d*np.sin(vehicle_theta)] )
    
        # 状態の作成
        vec = self.goal_position - control_point # 相対距離
        rad = self._phase(np.arctan2(vec[1],vec[0]) - self.vehicle_state[2]) # 相対角度
        observation = np.hstack((vec,rad,self.vehicle_state)) # 相対距離、グローバル座標(x,y, theta)
        
        # 距離の計算
        distance = np.linalg.norm(vec)
        
        # 報酬の計算
        # reward = - sum([(self.goal_position - predict_state[0:2]).T @ self.Q @ (self.goal_position - predict_state[0:2]) + input.T @ self.R @ input for predict_state,input in zip(self.predict_states, input_list)])
        
        # reward = sum([(bv.T @ self.Q @ bv + br * self.S * br) - ((self.goal_position - predict_state[0:2]).T @ self.Q @ (self.goal_position - predict_state[0:2]) + predict_state[2] * self.S * predict_state[2]) for bv,br,predict_state in zip(before_vec,before_rad,self.predict_states)])
        # reward = - (vec.T @ self.Q @ vec + action.T @ self.R @ action)
        reward = sum([ - (self.goal_position - predict_state[0:2]).T @ self.Q @ (self.goal_position - predict_state[0:2]) - predict_state[2] * self.S * predict_state[2] for predict_state in self.predict_states])
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
        if np.any(self.predict_states[:,:2] < np.tile(np.array([0,0]),(self.PREDICTION_HORIZON,1))) or np.any(self.predict_states[:,:2] > np.tile(np.array([self.FIELD_SIZE,self.FIELD_SIZE]),(self.PREDICTION_HORIZON,1))):
            done = True
            reward += -100000
        # self.before_distance = distance

        return np.atleast_2d(observation), reward, done, {}
    
    def _getStateMatrix(self, state):
        theta = state[2]
        # 時変の状態方程式
        A = np.array([[1,0,0],
                      [0,1,0],
                      [0,0,1]])
        B = np.array([[np.cos(theta),-self.vehicle_d*np.sin(theta)],
                      [np.sin(theta),self.vehicle_d*np.cos(theta)],
                      [0,1]])
        return A,B

    def _makeAction(self,action):
        action = action.reshape((self.PREDICTION_HORIZON,self.INPUT_DIM))
        # -1:1 -> min:max に変換
        source_max = np.tile(self.ACTION_HIGH,(self.PREDICTION_HORIZON,1))
        source_min = np.tile(self.ACTION_LOW,(self.PREDICTION_HORIZON,1))
        input = ((action * source_max) - (action * source_min) + source_max + source_min) * 0.5
        return input

    def _prediction(self,state,input_list):
        tmp_list = []
        for input in input_list:
            A,B = self._getStateMatrix(state)
            state = A @ state + B @ (input * self.TS)
            state[2] = self._phase(state[2])
            tmp_list.append(state)
        state_list = np.asarray(tmp_list)
        return state_list
    
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
        cv2.fillConvexPoly(image, self._rotatedRectangle(self.vehicle_state, scale), color=(255,0,255))
        # 制御点の描画 (B,G,R)
        cv2.circle(image, tuple(np.int32(np.round(self.vehicle_state[:2]*scale))), 5, (0, 0, 255), thickness=-1) 
    
        # 予測状態の描画
        if  self.predict_states is not None:
            for pred_state in self.predict_states[1:]:
                cv2.circle(image,tuple(np.int32(np.round(pred_state[0:2]*scale))), 4, (0, 255, 255), thickness=1) 
        
        # 座標系を数学にする
        image = np.flipud(image)
        
        if mode == 'human':
            cv2.imshow('image', image)
            cv2.waitKey(5)
            return image
        return image
    
    def _rotatedRectangle(self,state, scale):
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