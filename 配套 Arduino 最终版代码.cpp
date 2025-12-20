#include <Servo.h>
#include <Wire.h>

// 硬件配置
const int MPU6050_ADDR = 0x68;
const int PIN_HL = 2, PIN_KL = 3, PIN_HR = 4, PIN_KR = 5;
// 全局变量
Servo hip_left, knee_left, hip_right, knee_right;
char cmd_buf[32];
int buf_idx = 0;
float angles[4] = {0.0, 0.0, 0.0, 0.0};

void setup() {
  // 串口初始化
  Serial.begin(115200);
  // MPU6050 初始化
  Wire.begin();
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);
  // 舵机初始化
  hip_left.attach(PIN_HL);
  knee_left.attach(PIN_KL);
  hip_right.attach(PIN_HR);
  knee_right.attach(PIN_KR);
  // 初始位置归零
  hip_left.write(90);
  knee_left.write(90);
  hip_right.write(90);
  knee_right.write(90);
  delay(1000);
}

void loop() {
  // 读取MPU6050数据（可选上传至上位机）
  read_mpu6050_data();

  // 解析上位机串口指令
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '[' && buf_idx == 0) {
      buf_idx = 0;
    } else if (c == ']') {
      cmd_buf[buf_idx] = '\0';
      parse_joint_cmd(cmd_buf);
      buf_idx = 0;
    } else if (buf_idx < 31) {
      cmd_buf[buf_idx++] = c;
    }
  }
}

void read_mpu6050_data() {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, 6, true);
  int16_t ax = Wire.read() << 8 | Wire.read();
  int16_t ay = Wire.read() << 8 | Wire.read();
  int16_t az = Wire.read() << 8 | Wire.read();
  // 可选：将数据发送至上位机
  // Serial.printf("ACC: %d,%d,%d\n", ax, ay, az);
}

void parse_joint_cmd(char* buf) {
  char* token = strtok(buf, ",");
  int i = 0;
  while (token != NULL && i < 4) {
    angles[i++] = atof(token);
    token = strtok(NULL, ",");
  }
  // 舵机角度映射：[-90°,90°] -> [0°,180°]
  hip_left.write(constrain(angles[0] + 90, 0, 180));
  knee_left.write(constrain(angles[1] + 90, 0, 180));
  hip_right.write(constrain(angles[2] + 90, 0, 180));
  knee_right.write(constrain(angles[3] + 90, 0, 180));
  // 发送响应
  Serial.println("CMD_EXECUTED");
}
