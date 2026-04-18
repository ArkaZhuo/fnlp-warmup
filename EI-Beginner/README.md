# EI-Beginner 运行说明

本文档整理了当前仓库中 Task1 到 Task6 的运行方式。以下命令默认在仓库根目录执行：

---

## Task1：基于传统运动学的机械臂物体抓取

脚本：

```bash
scripts/task1_pybullet_kinematic_pick_place.py
```

### 1. 自动抓取，最快版本

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task1_pybullet_kinematic_pick_place.py --speed ultrafast
```

### 2. GUI 自动抓取演示版

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task1_pybullet_kinematic_pick_place.py --gui --realtime
```

### 3. 键盘控制版

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task1_pybullet_kinematic_pick_place.py --gui --interactive keyboard
```

### 4. 滑块控制版

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task1_pybullet_kinematic_pick_place.py --gui --interactive slider
```

### 5. 键盘 + 滑块同时开启

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task1_pybullet_kinematic_pick_place.py --gui --interactive both
```

### 6. 指定输出文件

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task1_pybullet_kinematic_pick_place.py --speed ultrafast --output results/task1_pybullet_result.json
```

### 参数说明

- `--gui`：打开 PyBullet 图形界面
- `--realtime`：放慢仿真速度，便于观察
- `--speed`：自动抓取速度，可选 `normal`、`fast`、`ultrafast`
- `--interactive`：交互模式，可选 `none`、`keyboard`、`slider`、`both`
- `--output`：结果 JSON 输出路径

---

## Task2：基于强化学习的机械臂物体抓取

Task2 分成两个子任务脚本：

- `task2_gym_qlearning.py`：Gym/Gymnasium 基础强化学习
- `task2_pybullet_qlearning_pick.py`：PyBullet 机械臂抓取强化学习

### Task2-1：Gym 基础强化学习

脚本：

```bash
scripts/task2_gym_qlearning.py
```

运行命令：

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task2_gym_qlearning.py --seed 42 --output results/task2_gym_qlearning_result.json
```

默认参数直接运行：

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task2_gym_qlearning.py
```

### Task2-2：PyBullet 机械臂强化学习抓取

脚本：

```bash
scripts/task2_pybullet_qlearning_pick.py
```

标准运行：

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task2_pybullet_qlearning_pick.py --seed 42 --train_episodes 500 --eval_episodes 120 --output results/task2_pybullet_rl_result.json
```

如果想打开 GUI：

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task2_pybullet_qlearning_pick.py --seed 42 --train_episodes 500 --eval_episodes 120 --gui --output results/task2_pybullet_rl_result.json
```

### 参数说明

`task2_gym_qlearning.py` 支持：

- `--seed`
- `--output`

`task2_pybullet_qlearning_pick.py` 支持：

- `--train_episodes`
- `--eval_episodes`
- `--seed`
- `--gui`
- `--output`

---

## Task3：基于模仿学习的机械臂物体抓取

脚本：

```bash
scripts/task3_imitation_diffusion_policy.py
```

标准运行：

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task3_imitation_diffusion_policy.py --seed 42 --demo_episodes 220 --eval_episodes 80 --output results/task3_imitation_result.json
```

按默认参数运行：

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task3_imitation_diffusion_policy.py
```

### 参数说明

- `--seed`
- `--demo_episodes`
- `--eval_episodes`
- `--output`

---

## Task4：基于 VLA 大模型的机械臂物体抓取

脚本：

```bash
scripts/task4_vla_mini_pipeline.py
```

标准运行：

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task4_vla_mini_pipeline.py --seed 42 --train_size 1800 --test_size 500 --output results/task4_vla_result.json
```

按默认参数运行：

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task4_vla_mini_pipeline.py
```

### 参数说明

- `--seed`
- `--train_size`
- `--test_size`
- `--output`

---

## Task5：基于 LLM/VLM 大模型的任务规划

脚本：

```bash
scripts/task5_llm_vlm_planning.py
```

标准运行：

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task5_llm_vlm_planning.py --seed 42 --output results/task5_planning_result.json
```

按默认参数运行：

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task5_llm_vlm_planning.py
```

### 参数说明

- `--seed`
- `--output`

---

## Task6：基于强化学习的人形机器人运动控制

脚本：

```bash
scripts/task6_humanoid_rl_imitation.py
```

### 1. 训练并输出结果

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task6_humanoid_rl_imitation.py --seed 42 --demo_episodes 70 --rl_episodes 220 --output results/task6_humanoid_result.json
```

### 2. GUI 演示：固定底座 teleop 策略

这个版本更稳，适合录屏和展示。

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task6_humanoid_rl_imitation.py --gui --realtime --demo_policy teleop --demo_episodes 10 --rl_episodes 20 --gui_demo_episodes 3
```

### 3. GUI 演示：RL 策略

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task6_humanoid_rl_imitation.py --gui --realtime --demo_policy rl --demo_episodes 10 --rl_episodes 20 --gui_demo_episodes 10
```

### 4. GUI 演示：自由底座尝试版

这个版本位置会变化，但稳定性相对差一些。

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/task6_humanoid_rl_imitation.py --gui --realtime --free_base_demo --demo_policy teleop --demo_episodes 10 --rl_episodes 20 --gui_demo_episodes 3
```

### 参数说明

- `--seed`
- `--demo_episodes`
- `--rl_episodes`
- `--gui`
- `--realtime`
- `--free_base_demo`
- `--demo_policy`：可选 `teleop`、`imitation`、`rl`
- `--gui_demo_episodes`
- `--output`

---



## 默认结果文件

常见结果文件包括：

- `results/task1_pybullet_result.json`
- `results/task2_gym_qlearning_result.json`
- `results/task2_pybullet_rl_result.json`
- `results/task3_imitation_result.json`
- `results/task4_vla_result.json`
- `results/task5_planning_result.json`
- `results/task6_humanoid_result.json`

如果需要查看某个脚本的完整参数，也可以直接执行：

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 scripts/对应脚本.py --help
```
