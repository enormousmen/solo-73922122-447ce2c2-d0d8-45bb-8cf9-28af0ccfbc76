# Resumable Training API（可恢复单作业训练服务）

FastAPI + CPU 版 PyTorch 实现的**单作业训练服务**。训练工程师在作业被抢占（暂停、
进程强退、容器被杀）后，可以从最近成功的检查点继续训练；在**同环境、同种子、同
配置**下，恢复后的逐批损失与最终参数与**不中断训练逐位一致**。

- 模型：两层全连接网络 `Linear → ReLU → Dropout → Linear`
- 优化器：Adam；每轮（epoch）打乱全部样本后按批训练
- 数据：仓库内置固定分类数据集 `data/classification.csv`（600 样本 / 2 特征 / 3 类）
- 默认配置：`config/default.json`（可在启动时按字段覆盖）

## 目录结构

```
app/
  main.py        # FastAPI 入口（/docs 自动生成）
  trainer.py     # 单作业状态机 + 确定性训练循环 + 暂停/续训
  checkpoint.py  # 原子检查点读写（tmp + fsync + rename），支持透明加解密
  crypto.py      # AES-256-GCM 检查点静态加密（环境变量密钥、帧格式、校验）
  config.py      # 默认配置加载与校验
  dataset.py     # 固定 CSV 数据集加载
  model.py       # 两层 FC + Dropout
config/default.json      # 默认训练配置
data/classification.csv  # 固定分类数据集（scripts/generate_data.py 生成）
scripts/generate_data.py # 数据集生成脚本（固定种子，仅作溯源）
scripts/verify_repro.py  # 端到端验证：不中断对照 / 暂停续训 / 强退恢复 等
scripts/verify_encryption.py # 加密专项端到端验证（密钥缺失/错误、明文点拒绝等）
Dockerfile, docker-compose.yml, requirements.txt
```

## 快速开始

### Docker Compose（推荐）

```bash
docker compose up --build
# 服务: http://localhost:8000  交互式文档: http://localhost:8000/docs
```

检查点保存在命名卷 `trainer-runs` 中，容器被杀或重启后数据不丢。

### 本地运行（开发）

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 环境变量: RUNS_DIR(默认 runs/)  CONFIG_PATH(默认 config/default.json)
```

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 服务信息 |
| GET | `/config` | 默认配置与运行时信息 |
| POST | `/train/start` | 启动新训练。Body: `{"force": false, "config": {覆盖字段}}` |
| POST | `/train/pause` | 暂停。**训练停止且完整检查点持久化后才返回 200** |
| POST | `/train/resume` | 从暂停态/最近检查点续训（并发只放行一个训练者） |
| GET | `/train/status` | 状态与进度（epoch/batch/百分比/检查点健康度） |
| GET | `/train/losses` | 逐批损失（支持 `since_step`、`limit`） |
| GET | `/train/parameters` | 下载当前模型参数（.pt，取自最近持久化检查点） |
| GET | `/train/checkpoint` | 下载完整检查点（模型+优化器+RNG+历史） |

交互式文档（Swagger UI）在 `/docs`，可直接在浏览器里调通上述接口。

### curl 示例

```bash
# 启动（覆盖部分配置；已有运行需 force=true 或先续训）
curl -X POST localhost:8000/train/start -H 'Content-Type: application/json' \
     -d '{"config": {"epochs": 8, "batch_size": 32, "lr": 0.01}}'

# 查询进度
curl -s localhost:8000/train/status | jq '.state, .completed_batches, .total_batches'

# 逐批损失
curl -s 'localhost:8000/train/losses?limit=5' | jq '.losses[]'

# 暂停（阻塞至训练停止且检查点落盘才返回）
curl -X POST localhost:8000/train/pause

# 续训
curl -X POST localhost:8000/train/resume

# 下载参数 / 完整检查点
curl -OJ localhost:8000/train/parameters     # model_parameters.pt
curl -OJ localhost:8000/train/checkpoint     # checkpoint.pt
```

状态机：`idle → running ⇄ pausing → paused → running … → finished`（异常为
`failed`）。`paused` 表示训练已停、可续训；`checkpoint_dirty=true` 表示内存进度
领先于磁盘（上次保存失败），此时重启进程会回退到磁盘上的旧检查点。

## 检查点静态加密（AES-256-GCM，可选）

共享卷里的 `checkpoint.pt` 可能被备份系统或其他容器直接读取。服务支持对**完整
检查点**（模型参数、Adam 优化器状态、RNG、损失历史、配置）做静态加密，明文只存在
于进程内存中，磁盘上的最终文件与写入过程中的临时文件都**只有密文**。

### 密钥配置（本地）

密钥从环境变量 `CHECKPOINT_ENCRYPTION_KEY` 读取，内容是 **32 字节随机密钥的
Base64 编码**（解码后必须恰好 32 字节，否则进程启动即失败，不会回退为明文）：

```bash
# 1) 生成密钥
python3 -c "import os, base64; print(base64.b64encode(os.urandom(32)).decode())"

# 2a) 本地直接运行：导出环境变量后启动
export CHECKPOINT_ENCRYPTION_KEY='<上一步输出的 base64>'
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2b) Docker Compose：写入本地 .env（不要提交到版本库），compose 会自动加载
printf 'CHECKPOINT_ENCRYPTION_KEY=%s\n' '<base64>' > .env
chmod 600 .env
docker compose up --build
# 或者直接在 shell 中 export 同名变量，docker-compose.yml 中的
# ${CHECKPOINT_ENCRYPTION_KEY:-} 会透传给容器；留空/不设置 = 不启用加密。
```

启用后 `GET /config` 的 `checkpoint_encryption.enabled=true`、
`algorithm="AES-256-GCM"`；`GET /train/status` 增加 `encryption_enabled`、
`checkpoint_encrypted` 与 `recovery_error` 字段。

### 磁盘格式与安全语义

- 帧格式：`magic "RPCKENC1"(8B) + nonce(12B) + GCM tag(16B) + 密文`，magic 同时
  作为 GCM 的 AAD，并用于区分旧版明文点（torch ZIP 以 `PK\x03\x04` 开头）。
- **每次保存都用 `os.urandom` 生成新 nonce**，同一现场重复保存的密文也不同。
- 加密写入与明文模式共用同一套 tmp → fsync → `os.replace` → 目录 fsync 的原子
  流程；加密/写入失败只清理临时文件，**旧检查点原样保留**，可重试。
- 强退/断电后重启，磁盘上只会是完整的旧密文点或完整的新密文点，启动时清理残留
  临时文件；恢复、暂停后续训、参数/检查点下载仍针对**最近成功保存的同一现场**，
  同配置续训的逐批损失与参数逐位一致（加密只改变持久化编码，不改变内存现场）。
- 明确拒绝（HTTP 409，`state=failed` 且 `status.recovery_error` 给出原因）且
  **绝不覆盖原文件**的情形：
  - 密文点存在但未配置密钥（"no key is set"）；
  - 密钥错误 / 文件被篡改（GCM 认证失败，"failed to decrypt and authenticate"）；
  - 启用加密时发现旧的**明文**检查点（"plaintext ... refusing to load or
    overwrite"，`force=true` 同样拒绝）。
  - 处置方式：用正确密钥重启即可恢复；确认明文点不再需要时，由人工把
    `checkpoint.pt` 移出 `RUNS_DIR` 后再启动新加密作业（服务本身不提供删除
    接口）。
- **不启用加密时**（环境变量未设置或为空），旧明文目录的行为与之前完全一致，
  可直接恢复与续训。
- `GET /train/checkpoint` 下载的是磁盘上的原始密文帧（备份语义不变，不含明文）；
  `GET /train/parameters` 在进程内存中解密后返回普通 `.pt` 参数字典。
- 密钥是静态机密：请通过 `.env`（权限 600）/宿主机环境变量/密钥管理系统注入，
  不要写进镜像或提交到版本库；丢失密钥等于检查点不可恢复。

### 加密专项自测

```bash
python3 scripts/verify_encryption.py   # 36 项检查，覆盖下列 I1–I13 场景
```

覆盖：启用状态上报、磁盘与临时文件无明文、每次保存 nonce 唯一、暂停/强退恢复后
与不中断加密基线逐位一致、缺密钥/错密钥/明文点全部 409 拒绝且文件不被覆盖、
换回正确密钥后可恢复、未启用时旧明文目录可用、密文/明文下载语义、畸形密钥启动
失败、加密保存失败保留旧点。

## 可复现性设计（为什么恢复后逐位一致）

1. **单线程 + 确定性算子**：`torch.set_num_threads(1)` 并开启
   `torch.use_deterministic_algorithms(True)`，同一构建下浮点结果逐位确定。
2. **单一 RNG 流**：新运行先 `torch.manual_seed(seed)`，模型初始化与 Dropout
   都从该流取数；检查点保存 `torch.get_rng_state()`，恢复时原样写回。
3. **每轮打乱可重推**：第 e 轮的样本排列是 `(seed, e)` 的纯函数
   （`random.Random(f"{seed}:{epoch}")`），重启后无需存储即可重建当轮顺序。
4. **批边界检查点**：每个检查点记录 `epoch`、`next_batch`、模型与 Adam 状态、
   RNG 状态、完整损失历史、解析后的配置与 `completed` 标志——即"准备执行第 k
   批"的完整现场。恢复 = 还原现场后继续同一确定性序列。
5. **原子写**：先写 `.checkpoint.pt.tmp.<uuid>` → fsync → `os.replace` 原子改名
   → 目录 fsync。保存中被强退，磁盘上只会是**完整的旧点或完整的新点**；
   启动时清理残留的临时文件。保存失败不触碰旧文件，明确报错，可重试。

> 注意：逐位一致的前提是**同环境**（同一 torch 构建/CPU 指令集、单线程）。跨
> 不同 CPU 架构或 torch 版本不保证逐位一致，但同机同镜像（如本 compose）成立。

## 不中断对照与退出重启示例

### 自动化端到端验证（推荐）

```bash
python3 scripts/verify_repro.py
```

脚本会真实拉起 uvicorn 服务进程，依次验证（详见脚本 docstring）：
A 不中断基线；B 暂停+续训；C 强退+重启+续训；D 并发续训只放行一个；
E1 保存失败→报错→旧点保留→重试成功；E2 保存失败后强退→回退旧点恢复；
F 慢保存不挡查询；G 保存进行中强退→恢复完整旧/新点；H 基础接口与 /docs。
B–E、G 的逐批损失与最终参数均与各自不中断基线**逐位一致**才判通过。
加密模式另有 `scripts/verify_encryption.py`（36 项，见上节）。

> 本环境的 seccomp 策略禁止向其他进程发信号，脚本用 `ENABLE_TEST_CRASH_ENDPOINT=1`
> 开启的测试专用 `POST /test/crash`（`os._exit`，立即死亡、无任何清理）模拟强退；
> 恢复语义与 SIGKILL 等价。该端点默认关闭，切勿在生产开启。

### Docker 手动退出重启示例

```bash
docker compose up -d --build
curl -X POST localhost:8000/train/start -H 'Content-Type: application/json' \
     -d '{"config": {"epochs": 8}}'

docker kill resumable-trainer          # 模拟抢占/强退（真实 SIGKILL）
docker compose start                   # 重启容器（检查点在命名卷中）
curl -s localhost:8000/train/status    # state=paused, recovered_after_restart=true
curl -X POST localhost:8000/train/resume
curl -s localhost:8000/train/status    # 直至 state=finished
```

另开一个终端可对同一数据集做一次不中断对照（`force=true` 重跑同配置），用
`GET /train/losses` 与 `GET /train/parameters` 对比两边逐批损失与参数完全一致。

## 实测结果

（以下为 `python3 scripts/verify_repro.py` 在本仓库环境下的真实输出摘要）

```
    workdir: /tmp/verify-repro-ndor_el1
    
    == H: service basics (idle server) ==
      [PASS] H1 GET /
      [PASS] H2 GET /docs (Swagger UI)
      [PASS] H3 parameters before any checkpoint -> 404
      [PASS] H4 resume with no run -> 404
      [PASS] H5 pause while idle -> 409
    
    == A: uninterrupted baseline ==
      [PASS] A baseline finished - 95 batches, final loss 0.027771
    
    == B: pause + resume ==
      [PASS] B1 pause returns only after checkpoint persisted - pause latency 0.03s, stopped at batch 14
      [PASS] B2 resume accepted
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] B3 pause+resume == uninterrupted
    
    == D: concurrent resume starts exactly one trainer ==
      [PASS] D1 concurrent resumes -> one 200, one 409 - got [200, 409]
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] D2 resumed run == uninterrupted
    
    == E1: checkpoint save failure keeps old checkpoint, retry works ==
      [PASS] E1 pause with failing save -> 500 - training is stopped but the checkpoint could not be persisted; the previous checkpoint is intact on 
      [PASS] E1 status shows dirty checkpoint + error
      [PASS] E1 old checkpoint untouched
      [PASS] E1 retried pause -> 200, checkpoint persisted
      [PASS] E1 resume after retry
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] E1 run after save-failure == uninterrupted
    
    == F: slow checkpoint saves do not block queries ==
      [PASS] F1 queries respond fast while a 1s save is in flight - worst status latency 1ms
      [PASS] F2 pause during slow saves
      [PASS] F3 slow-save run finishes
    
    == C: hard crash + restart recovers from latest checkpoint ==
      [PASS] C1 server crashed mid-run
      [PASS] C2 restart recovers paused state from checkpoint - recovered at batch 10/95
      [PASS] C3 resume after restart
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] C4 kill+restart == uninterrupted
    
    == E2: save failure, then hard crash -> old checkpoint used ==
      [PASS] E2 pause with failing save -> 500
      [PASS] E2 restart falls back to old complete checkpoint - recovered at batch 6/95
      [PASS] E2 resume after restart
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] E2 save-fail+kill == uninterrupted
    
    == G: hard crash in the middle of a checkpoint save ==
      (baseline: identical config, uninterrupted)
      [PASS] G1 server crashed during save
      [PASS] G2 no torn checkpoint / stale temp cleaned - leftovers: none
      [PASS] G3 restart recovers a complete checkpoint (old or new) - recovered at batch 1/19
      [PASS] G3b resume after restart
        losses: 19 entries, exact match
        params: all tensors bit-identical (sha256 1353a7415c7cae90 == 1353a7415c7cae90)
      [PASS] G4 kill-mid-save == uninterrupted
    
    ================ SUMMARY ================
      33/33 checks passed（逐项明细同上，全部 PASS）

```

## 限制与未完成事项

- 单作业、单进程、单 worker：多副本/多 worker 部署会破坏单训练者语义（需引入
  外部锁与共享存储，未实现）。
- 逐位一致仅保证同环境同构建（见上节）；未做跨机器/跨 torch 版本对齐。
- 检查点只保留最新一份（原子替换保证新旧二选一）；未做历史版本保留/滚动清理。
- `force=true` 启动新运行会**立即丢弃**旧检查点：若新运行在写出第一个检查点
  前崩溃，则无点可恢复（状态回到 `idle`），只能重新开始——这是刻意语义，
  避免"已丢弃的旧运行"在重启后被悄悄恢复。
- `GET /train/parameters` 取自最近**已持久化**检查点，运行中可能落后最多
  `checkpoint_every` 批；未提供"当前内存参数"下载（避免与训练线程竞争）。
- 本环境无 docker，镜像构建与 compose 未实际执行；Dockerfile/compose 与本地实
  测使用完全相同的启动命令与依赖版本，但未在真实 Docker 中跑过验证脚本。
- 暂停等待上限为 `pause_timeout_seconds`（默认 30s），超时返回 504，训练仍会
  在批边界停下，可轮询状态确认。
- 加密限制：单一对称密钥，未做密钥轮换 / 多密钥 / KMS 集成；服务不提供删除或
  "明文→密文"迁移接口，模式不匹配时只拒绝并保留原文件，需人工处置；密钥丢失
  后密文检查点在密码学上不可恢复。

## 本次加密改动的实测结果

在本仓库环境实际执行（torch 2.15.0+cpu / fastapi 0.163.1 / cryptography 44.0.3）：

- `python3 scripts/verify_repro.py`：**33/33 通过**（明文模式行为无回归）。
- `python3 scripts/verify_encryption.py`：**36/36 通过**，覆盖：
  - 畸形密钥（非法 Base64、长度≠32 字节）进程启动即失败；
  - 启用后 `/config`、`/train/status` 正确上报加密状态；
  - 磁盘 `checkpoint.pt` 与所有临时文件均为 `RPCKENC1` 密文帧，无 torch ZIP
    头/明文张量标记泄漏；每次保存 nonce 不同；
  - 加密下暂停→续训、强退→同密钥重启→续训，逐批损失与最终参数相对不中断
    加密基线**逐位一致**；
  - 密文点缺密钥 / 错密钥（GCM 认证失败）/ 启用加密遇明文点（含 paused 未完成
    点）：加载、start（含 `force=true`）、resume、参数下载全部明确 409 拒绝，
    原文件哈希不变；换回正确密钥后立即恢复；
  - 未启用加密时旧明文目录正常恢复与下载；
  - `/train/checkpoint` 下载与磁盘密文逐字节相同，`/train/parameters` 内存解密
    返回可加载的 `.pt`；加密保存失败时旧密文点完整保留、重试成功。
