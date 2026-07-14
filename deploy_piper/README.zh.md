[English](README.md) | **中文**

# Deploy — 在 Piper 机械臂上运行策略

一个简单的客户端—服务端部署框架：

- **服务端**（`deploy/server.py`）：在模型环境中加载策略适配器。
- **客户端**（`deploy/client.py`）：运行在连接 Piper 机械臂和相机的机器上，
  向服务端发送观测并执行返回的动作序列。

要部署模型，只需按 `deploy/adapters/base.py` 的接口实现适配器；示例见
`deploy/adapters/dummy.py`。客户端、数据传输和动作执行逻辑保持不变。

## 快速开始

从仓库根目录使用两个终端：

4090 工作站的 `piper_sdk` 已经装好 Piper SDK、支持 Piper 的 lerobot 和全部
客户端依赖；无需安装，只需激活环境。

```bash
# 终端 1：模型/GPU 机器上的服务端
conda activate <model-env>
python -m deploy.server --config <model-config>

# 终端 2：机械臂客户端
conda activate piper_sdk
python -m deploy.client --config <model-config> --task="your task" --duration_s=60
```

把双臂 Piper 移向 home 的硬件测试：

```bash
python -m deploy.server --config home
python -m deploy.client --config home --task=home --duration_s=5
```

## 配置

一个配置（`deploy/configs/<name>.json`）记录适配器参数、端口、机械臂、相机和
`camera_map`。启动前激活对应模型环境；配置不管理环境。额外命令行参数会覆盖
配置中的值。

- `example` — 只启动 dummy 服务端，不连接机械臂
- `home` — dummy 策略，把机械臂移向 home
- `pi05` — 真实模型；机器相关的配置说明在 `_notes` 里

- `server.adapter` 和 `server.args` — checkpoint、device、RTC、compile 等模型参数；
- `server.port` — 客户端连接的端口。

### 远程服务端

策略服务端可以运行在远程 GPU 机器上，客户端运行在连接机械臂的本地机器上。
在两台机器上都复制本仓库，然后在配置中设置：

```json
"server": { "host": "0.0.0.0", ... }
"client": { "server": "http://<GPU 机器 IP>:<port>", ... }
```

GPU 机器运行服务端，机械臂端机器使用相同配置运行客户端。
`--server=http://<ip>:<port>` 可覆盖服务端地址。服务端没有鉴权，只应暴露在
可信网络中。

## 添加策略

最简单的接入方式是实现 `PolicyAdapter`，共用 HTTP 服务端负责传输。
`server.args` 中的构造参数会以字符串传入适配器。实现方式和字段定义见英文
README 的完整示例以及 `deploy/adapters/dummy.py`。

1. 把适配器放进模型环境可以导入的 Python 模块。
2. 复制 `deploy/configs/example.json` 作为新配置，把 `server.adapter` 设为
   类的导入路径，例如 `my_policy.deploy:MyPolicyAdapter`。
3. 添加 `client` 配置：机械臂、相机和 `camera_map`。
4. 按快速开始的方式分别启动服务端和客户端。

## 机械臂环境

运行前，四个 Piper CAN 接口（`left_leader`、`left_follower`、
`right_leader`、`right_follower`）都必须先找到并激活。

```bash
# 查找机械臂
bash utils/find_all_can_port.sh
```

![查找所有 Piper CAN 接口](docs/find_can.png)

```bash
# 激活机械臂
bash utils/activate_all_can.sh
```

![激活所有 Piper CAN 接口](docs/activate_can.png)

```bash
# 运行后软失能
python utils/disable_arms.py                           # 四个机械臂
python utils/disable_arms.py left_follower --no-home   # 单个机械臂，原地释放
```

## 通信协议

### 客户端输入和模型输出

`predict_chunk(images, state, task, consumed, delay_ticks)` 的输入：

- `images`：`{policy_image_key: image}`；每张图是 HWC `uint8` RGB。
  `client.camera_map` 把机械臂端相机名映射到模型输入名，例如
  `{"top": "camera1"}`。
- `state`：shape 为 `(state_dim,)` 的 float32，严格按照
  `robot.action_features` 的电机顺序。适配器负责转换成训练时使用的单位并归一化。
- `task`：命令行中 `--task` 的原始字符串。
- `consumed`：上一 action chunk 已执行的行数；第一次请求为 `-1`。
- `delay_ticks`：以控制周期计的预测推理延迟。非 RTC 策略可忽略后两项。

适配器必须返回 shape 为 `(chunk_size, action_dim)` 的有限 float32 绝对电机目标。
列顺序和单位必须与 `robot.action_features` 相同；各行以 `fps` 顺序执行。这些是
绝对目标，不是 delta。

### HTTP 接口

使用共享 `deploy.server` 时只需实现适配器。独立服务端必须实现：

- `GET /info`：JSON 包含 `protocol_version`、`name`、`image_keys`、
  `state_dim`、`action_dim`、`chunk_size`、`fps`、`checkpoint`。
- `POST /predict`：请求是压缩 NumPy `.npz`，包含 `img_<key>`、`state`、
  `task` 及可选的 `consumed`/`delay_ticks`；返回 float32 action chunk 的原始
  NumPy `.npy`。
- `POST /reset`：清空 episode state 并返回 HTTP 200。

建议直接使用 `deploy.protocol.decode_observation()` 和 `encode_chunk()`，避免重复
实现二进制格式。NumPy 始终以 `allow_pickle=False` 加载。
