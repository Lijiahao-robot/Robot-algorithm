#include <Servo.h>
#include <Wire.h>

// MPU6050 地址
const int MPU_ADDR = 0x68;
// 关节舵机
Servo hip_left, knee_left, hip_right, knee_right;
const int PIN_HL = 2, PIN_KL = 3, PIN_HR = 4, PIN_KR = 5;
// 指令缓存
char cmd_buf[32];
int buf_idx = 0;
float angles[4] = {0,0,0,0};

void setup() {
  Serial.begin(115200);
  Wire.begin();
  // 唤醒 MPU6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);
  // 舵机初始化
  hip_left.attach(PIN_HL);
  knee_left.attach(PIN_KL);
  hip_right.attach(PIN_HR);
  knee_right.attach(PIN_KR);
}

void loop() {
  // 读取 MPU6050 数据（可选：上传姿态数据到上位机）
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);
  int16_t ax = Wire.read()<<8 | Wire.read();
  int16_t ay = Wire.read()<<8 | Wire.read();
  int16_t az = Wire.read()<<8 | Wire.read();
  
  // 解析上位机关节指令
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '[' && buf_idx == 0) {
      buf_idx = 0;
    } else if (c == ']') {
      cmd_buf[buf_idx] = '\0';
      parse_cmd(cmd_buf);
      buf_idx = 0;
    } else if (buf_idx < 31) {
      cmd_buf[buf_idx++] = c;
    }
  }
}

void parse_cmd(char* buf) {
  char* token = strtok(buf, ",");
  int i = 0;
  while (token != NULL && i < 4) {
    angles[i++] = atof(token);
    token = strtok(NULL, ",");
  }
  // 舵机角度映射
  hip_left.write(constrain(angles[0] + 90, 0, 180));
  knee_left.write(constrain(angles[1] + 90, 0, 180));
  hip_right.write(constrain(angles[2] + 90, 0, 180));
  knee_right.write(constrain(angles[2] + 90, 0, 180));
  Serial.println("CMD_OK");
}
