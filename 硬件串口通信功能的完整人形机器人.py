import numpy as np
import heapq
import time
import cv2
import serial
import serial.tools.list_ports

# ===================== 1. 硬件串口配置与通信模块 =====================
class SerialComm:
    def __init__(self, baudrate=115200, timeout=0.1):
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        # 自动查找可用串口（优先COM3、/dev/ttyUSB0等常见端口）
        self.port = self.find_available_port()

    def find_available_port(self):
        """自动搜索可用串口"""
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            # 适配Windows(COMx)和Linux(/dev/ttyUSBx)
            if "COM" in port.device or "ttyUSB" in port.device:
                print(f"找到可用串口: {port.device}")
                return port.device
        print("未找到可用串口！请检查硬件连接")
        return None

    def open_serial(self):
        """打开串口"""
        if self.port is None:
            return False
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(0.5)  # 等待串口稳定
            return True
        except Exception as e:
            print(f"串口打开失败: {e}")
            return False

    def send_joint_cmd(self, theta1_left, theta2_left, theta1_right, theta2_right):
        """
        发送关节角度指令给下位机
        指令格式: [L1,L2,R1,R2]\n  （角度单位：度，便于单片机处理）
        """
        if self.ser is None or not self.ser.is_open:
            return False
        # 弧度转角度，保留1位小数
        cmd = f"[{theta1_left*180/np.pi:.1f},{theta2_left*180/np.pi:.1f},{theta1_right*180/np.pi:.1f},{theta2_right*180/np.pi:.1f}]\n"
        try:
            self.ser.write(cmd.encode('utf-8'))
            # 可选：读取下位机响应
            # resp = self.ser.readline().decode('utf-8').strip()
            # print(f"下位机响应: {resp}")
            return True
        except Exception as e:
            print(f"指令发送失败: {e}")
            return False

    def close_serial(self):
        """关闭串口"""
        if self.ser is not None and self.ser.is_open:
            self.ser.close()

# ===================== 2. 常量配置（根据硬件修改） =====================
# 关节角度限位 (rad)
THETA1_MIN = -np.pi/2    # 髋关节最小角
THETA1_MAX = np.pi/2     # 髋关节最大角
THETA2_MIN = -np.pi      # 膝关节最小角
THETA2_MAX = 0.0         # 膝关节最大角（弯曲为负）
# 腿部尺寸 (m)
THIGH_LENGTH = 0.2
CALF_LENGTH = 0.2
# 步态参数
STEP_HEIGHT = 0.05       # 步高
STEP_LENGTH = 0.1        # 步长
GAIT_FREQUENCY = 10      # 步态频率(Hz)
# 安全参数
SAFE_DISTANCE = 20       # 碰撞安全距离(像素)
MOTOR_SPEED_SET = 30.0   # 电机目标速度(rpm)

# ===================== 3. 运动学模块（正+逆解+关节限位） =====================
def dh_transform(theta, d, a, alpha):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

def leg_forward_kinematics(theta1, theta2, is_left=True):
    """单腿正运动学：输入髋角/膝角，输出足端坐标"""
    side = 1 if is_left else -1
    dh_params = [
        [theta1, 0.15, THIGH_LENGTH, 0],  # 髋关节
        [theta2, 0, CALF_LENGTH, 0]      # 膝关节
    ]
    T = np.eye(4)
    for param in dh_params:
        T = T @ dh_transform(*param)
    x, y, z = T[:3, 3]
    return np.array([x, side * y, z])  # 左右腿y轴镜像

def leg_inverse_kinematics(target_pos, is_left=True):
    """单腿逆运动学：输入足端坐标，输出关节角（带限位）"""
    side = 1 if is_left else -1
    x, y, z = target_pos
    y = side * y  # 左右腿镜像转换

    r = np.sqrt(x**2 + y**2)
    # 余弦定理求解关节角
    cos_theta2 = (r**2 - THIGH_LENGTH**2 - CALF_LENGTH**2) / (2 * THIGH_LENGTH * CALF_LENGTH)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = -np.arccos(cos_theta2)

    alpha = np.arctan2(y, x)
    beta = np.arccos((THIGH_LENGTH**2 + r**2 - CALF_LENGTH**2) / (2 * THIGH_LENGTH * r))
    theta1 = alpha - beta

    # 关节限位检查
    if not (THETA1_MIN <= theta1 <= THETA1_MAX and THETA2_MIN <= theta2 <= THETA2_MAX):
        return None, None
    return theta1, theta2

# ===================== 4. 双腿步态规划模块 =====================
class GaitPlanner:
    def __init__(self):
        self.phase = 0  # 步态相位: 0-1
        self.dt = 1 / GAIT_FREQUENCY

    def generate_gait(self, current_pos, is_left=True):
        """生成双腿交替步态轨迹：支撑相+摆动相"""
        x0, y0, z0 = current_pos
        # 步态相位更新
        self.phase = (self.phase + self.dt) % 1.0
        phase = self.phase if is_left else (self.phase + 0.5) % 1.0

        if phase < 0.5:
            # 摆动相：抬腿迈步
            x = x0 + STEP_LENGTH * (phase - 0.25)
            z = z0 + STEP_HEIGHT * np.sin(2 * np.pi * phase)
        else:
            # 支撑相：落地支撑
            x = x0 + STEP_LENGTH * (0.75 - phase)
            z = z0
        return np.array([x, y0, z])

# ===================== 5. PID 电机速度控制模块 =====================
class PIDController:
    def __init__(self, kp=2.0, ki=0.5, kd=0.1, set_speed=MOTOR_SPEED_SET):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.set_speed = set_speed
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def calculate(self, current_speed):
        current_time = time.time()
        dt = current_time - self.last_time if current_time > self.last_time else 0.01
        error = self.set_speed - current_speed
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = np.clip(output, -100, 100)  # 控制量限幅
        self.last_error = error
        self.last_time = current_time
        return output

# ===================== 6. A* 路径规划模块 =====================
class AStarPlanner:
    def __init__(self, grid_map, start, goal):
        self.grid = grid_map
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.open_list = []
        self.closed_list = set()
        self.g_cost = {self.start: 0}
        self.f_cost = {self.start: self.heuristic(self.start)}
        self.parent = {}

    def heuristic(self, pos):
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    def get_neighbors(self, pos):
        neighbors = []
        dirs = [(-1,0), (1,0), (0,-1), (0,1)]
        for dx, dy in dirs:
            nx, ny = pos[0]+dx, pos[1]+dy
            if 0<=nx<len(self.grid) and 0<=ny<len(self.grid[0]) and self.grid[nx][ny]==0:
                neighbors.append((nx, ny))
        return neighbors

    def plan(self):
        heapq.heappush(self.open_list, (self.f_cost[self.start], self.start))
        while self.open_list:
            _, current = heapq.heappop(self.open_list)
            if current == self.goal:
                path = []
                while current in self.parent:
                    path.append(current)
                    current = self.parent[current]
                return path[::-1]
            self.closed_list.add(current)
            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_list:
                    continue
                tentative_g = self.g_cost[current] + 1
                if tentative_g < self.g_cost.get(neighbor, float('inf')):
                    self.parent[neighbor] = current
                    self.g_cost[neighbor] = tentative_g
                    self.f_cost[neighbor] = tentative_g + self.heuristic(neighbor)
                    heapq.heappush(self.open_list, (self.f_cost[neighbor], neighbor))
        return []

# ===================== 7. 视觉检测与碰撞检测模块 =====================
def detect_obstacle(frame):
    """检测红色障碍物，返回帧、障碍物状态、位置、轮廓"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    has_obs = False
    obs_pos = (0,0)
    obs_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            obs_pos = (x + w//2, y + h//2)
            has_obs = True
            obs_contours.append(cnt)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, "Obstacle", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame, has_obs, obs_pos, obs_contours

def collision_check(foot_pixel, obs_contours, safe_dist=SAFE_DISTANCE):
    """足端与障碍物碰撞检测"""
    foot_point = np.array([foot_pixel], dtype=np.int32)
    for cnt in obs_contours:
        dist = cv2.pointPolygonTest(cnt, (foot_pixel[0], foot_pixel[1]), True)
        if dist > 0 or abs(dist) < safe_dist:
            return True
    return False

# ===================== 8. 机器人主控制逻辑 =====================
def humanoid_robot_main():
    # 初始化串口通信
    serial_comm = SerialComm(baudrate=115200)
    if not serial_comm.open_serial():
        print("串口未连接，进入仿真模式")
        is_hardware_mode = False
    else:
        is_hardware_mode = True
        print("串口连接成功，进入硬件控制模式")

    # 初始化核心模块
    gait_planner = GaitPlanner()
    pid_left = PIDController()
    pid_right = PIDController()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    # 初始状态
    current_grid_pos = [0, 0]
    goal_grid_pos = [4, 4]
    grid_map = np.zeros((5,5), dtype=int)
    left_foot_pos = np.array([0.0, 0.1, 0.15])
    right_foot_pos = np.array([0.0, -0.1, 0.15])
    current_speed_left = 0.0
    current_speed_right = 0.0
    collision_risk = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 步骤1：视觉障碍物检测 + 栅格地图更新
        frame, has_obs, obs_pixel, obs_contours = detect_obstacle(frame)
        if has_obs:
            obs_grid_x = min(obs_pixel[0] // (w//5), 4)
            obs_grid_y = min(obs_pixel[1] // (h//5), 4)
            grid_map[obs_grid_x][obs_grid_y] = 1
            print(f"障碍物栅格位置: ({obs_grid_x},{obs_grid_y})")

        # 步骤2：A* 全局路径规划
        planner = AStarPlanner(grid_map, current_grid_pos, goal_grid_pos)
        path = planner.plan()
        if not path:
            cv2.putText(frame, "NO PATH!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            cv2.putText(frame, f"Path: {path}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        # 步骤3：步态生成 + 逆运动学解算 + 串口下发
        if len(path) > 1 and not collision_risk:
            # 生成左右腿步态轨迹
            target_left = gait_planner.generate_gait(left_foot_pos, is_left=True)
            target_right = gait_planner.generate_gait(right_foot_pos, is_left=False)

            # 逆解关节角度
            theta1_left, theta2_left = leg_inverse_kinematics(target_left, is_left=True)
            theta1_right, theta2_right = leg_inverse_kinematics(target_right, is_left=False)

            # 关节角度有效性检查
            if None in [theta1_left, theta2_left, theta1_right, theta2_right]:
                print("关节超程！跳过本次步态")
                continue

            # 步骤4：足端像素映射 + 碰撞检测
            foot_pixel_left = (int(target_left[0]*(w/0.5)), int(target_left[1]*(h/0.5)))
            foot_pixel_right = (int(target_right[0]*(w/0.5)), int(target_right[1]*(h/0.5)))
            cv2.circle(frame, foot_pixel_left, 5, (255,0,0), -1)
            cv2.circle(frame, foot_pixel_right, 5, (0,0,255), -1)

            if has_obs:
                collision_risk = collision_check(foot_pixel_left, obs_contours) or collision_check(foot_pixel_right, obs_contours)
                if collision_risk:
                    cv2.putText(frame, "COLLISION RISK!", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    print("碰撞风险！暂停行走")
                    continue

            # 步骤5：PID 速度控制 + 串口下发关节指令
            current_speed_left += pid_left.calculate(current_speed_left) * 0.02
            current_speed_right += pid_right.calculate(current_speed_right) * 0.02

            if is_hardware_mode:
                serial_comm.send_joint_cmd(theta1_left, theta2_left, theta1_right, theta2_right)

            # 打印运动状态
            print(f"左髋角:{theta1_left:.2f}rad | 左膝角:{theta2_left:.2f}rad | 右髋角:{theta1_right:.2f}rad | 右膝角:{theta2_right:.2f}rad")
            print(f"左电机速度:{current_speed_left:.2f} | 右电机速度:{current_speed_right:.2f}")

            # 更新足端位置与栅格位置
            left_foot_pos = target_left
            right_foot_pos = target_right
            current_grid_pos = path[1]
            collision_risk = False

        # 步骤6：可视化与退出条件
        cv2.imshow("Humanoid Robot Autonomous Walking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or current_grid_pos == goal_grid_pos:
            break

    # 资源释放
    cap.release()
    cv2.destroyAllWindows()
    serial_comm.close_serial()
    print("任务完成！人形机器人到达目标位置")

if __name__ == "__main__":
    humanoid_robot_main()
