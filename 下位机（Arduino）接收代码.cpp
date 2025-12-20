#include <Servo.h>

// 定义关节舵机
Servo hip_left, knee_left, hip_right, knee_right;
// 舵机引脚（根据硬件修改）
const int PIN_HL = 2, PIN_KL = 3, PIN_HR = 4, PIN_KR = 5;

// 指令解析缓存
char cmd_buf[32];
int buf_idx = 0;
float angles[4] = {0,0,0,0}; // [L1,L2,R1,R2]

void setup() {
  Serial.begin(115200);
  // 舵机初始化
  hip_left.attach(PIN_HL);
  knee_left.attach(PIN_KL);
  hip_right.attach(PIN_HR);
  knee_right.attach(PIN_KR);
}

void loop() {
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
  // 解析格式: L1,L2,R1,R2
  char* token = strtok(buf, ",");
  int i = 0;
  while (token != NULL && i < 4) {
    angles[i++] = atof(token);
    token = strtok(NULL, ",");
  }
  // 驱动舵机（角度映射：根据舵机实际范围调整）
  hip_left.write(constrain(angles[0] + 90, 0, 180));
  knee_left.write(constrain(angles[1] + 90, 0, 180));
  hip_right.write(constrain(angles[2] + 90, 0, 180));
  knee_right.write(constrain(angles[3] + 90, 0, 180));
  // 发送响应
  Serial.println("CMD_RECV_OK");
}
