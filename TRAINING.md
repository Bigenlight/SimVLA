# SimVLA — Docker Training + LIBERO-Plus Fine-Tune Guide

> 이 문서는 SimVLA를 **Docker 기반 2-컨테이너 (inference + training)** 로 돌리기 위한 상세 설정 가이드다.
> Inference는 `bigenlight/simvla-http:latest` (FastAPI + unified VLA protocol, 포트 8700)을 쓰고,
> Training은 `bigenlight/simvla-train:latest` (conda `simvla` + `libero` 2개 env)을 쓴다.
> 기존 conda 원본 워크플로우(`readme.md` 하단)는 그대로 보존된다 — 이 문서는 **병렬 경로**다.

---

## 📋 목차

1. [전체 아키텍처](#1-전체-아키텍처)
2. [Pre-requisites (서버 세팅)](#2-pre-requisites-서버-세팅)
3. [학습 컨테이너 이미지](#3-학습-컨테이너-이미지)
4. [데이터 파이프라인 (LIBERO-Plus 2 task)](#4-데이터-파이프라인-libero-plus-2-task)
5. [Fine-tune 실행](#5-fine-tune-실행)
6. [Eval 실행 (Task A + Task B 한정)](#6-eval-실행-task-a--task-b-한정)
7. [하이퍼파라미터 민감도 (주의사항)](#7-하이퍼파라미터-민감도-주의사항)
8. [실측 결과 (2026-04-21~22)](#8-실측-결과-2026-04-2122)
9. [파일 레퍼런스](#9-파일-레퍼런스)
10. [Known issues + Gotchas](#10-known-issues--gotchas)

---

## 1. 전체 아키텍처

```
┌─────────────────────────────┐   HTTP :8700    ┌──────────────────────────────┐
│  simvla-http  (inference)   │ ◀──────────────▶│  simvla-train (libero env)   │
│  FastAPI + /health /act     │                 │  LIBERO-Plus drop-in +       │
│  GPU 0 (~3.5 GB VRAM)       │                 │  scripts/eval_taskAB_*.py    │
└─────────────────────────────┘                 │  GPU 1 (sim rendering)       │
         ▲                                      └──────────────────────────────┘
         │ bind mount                                         ▲
         │                                                    │ bind mount
┌────────┴─────────────────────┐                              │
│  simvla-train (simvla env)   │                              │
│  accelerate train_smolvlm.py │ ┌────────────────────────────┤
│  4× A6000                    │ │  LIBERO-plus/              │
│                              │ │    libero/                 │
└──────────────────────────────┘ │    data/libero_plus_lerobot│
         ▲                       │    data/libero_plus_hdf5   │
         │ bind mount            │    (변환된 HDF5, 22GB)     │
         │                       └────────────────────────────┘
┌────────┴──────────────────────┐
│  SimVLA/                      │
│    SimVLA-LIBERO/    (3GB)    │  ← 출발 체크포인트
│    runs/simvla_libero_plus_ft │  ← fine-tuned 출력
│    scripts/ datasets/ ...     │
└───────────────────────────────┘
```

**핵심 설계 원칙**:
- Docker 이미지 = **의존성 환경만** (모델/데이터/체크포인트 전부 bind mount)
- `simvla` conda env: transformers 4.57.3, peft 0.17.1, torch 2.5.1+cu124, flash-attn 2.5.6
- `libero` conda env: transformers 4.21.1, robosuite 1.4.0, gym 0.25.2 (SimVLA와 pin 충돌하므로 분리)
- 통신은 HTTP (포트 8700, VLA_COMMUNICATION_PROTOCOL.md 규약)

---

## 2. Pre-requisites (서버 세팅)

### 하드웨어
본 가이드는 **NVIDIA RTX A6000 × 4 (48GB each), CUDA 12.4 driver 550+** 에서 검증됐다.
다른 GPU에서도 VRAM만 충분하면 동작 (A100 40GB × 4도 가능하되 batch size 조절 필요).

### 호스트 S/W
- Docker 27.4+ + NVIDIA Container Toolkit
- 디스크 여유공간 ≥ 100 GB (이미지 19GB + 데이터 22GB HDF5 + 체크포인트 3GB × 3 등)
- `/dev/shm` 크기 Docker에서 `--shm-size=32g` 로 올려야 함 (DataLoader worker)

### 호스트 디렉토리 배치
```bash
~/workspace/
├── SimVLA/                  # 이 리포
│   ├── SimVLA-LIBERO/       # HF 체크포인트 clone (3GB, 1회)
│   ├── runs/                # 학습/eval 산출물
│   └── ...
├── LIBERO-plus/             # 별도 clone
│   ├── libero/libero/
│   │   └── assets/          # 원본 LIBERO에서 복사 (405MB)
│   └── data/
│       ├── libero_plus_lerobot/  # HF 데이터 (~1.4GB, Task A/B만)
│       └── libero_plus_hdf5/     # 변환된 HDF5 (~22GB)
└── Libero-pro_benchmark/    # (선택) vla_client.py 재사용용
    └── LIBERO/              # 원본 LIBERO (assets 복사용)
```

### 일회성 Setup 커맨드

```bash
# 1) 리포 clone
cd ~/workspace
git clone https://github.com/Bigenlight/SimVLA.git
git clone https://github.com/sylvestf/LIBERO-plus.git
# (선택) 기존 libero-pro 벤치마크 재사용용:
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git \
    Libero-pro_benchmark/LIBERO

# 2) SimVLA-LIBERO 체크포인트 clone (3.1GB, LFS)
cd ~/workspace/SimVLA
git lfs install
git clone https://huggingface.co/YuankaiLuo/SimVLA-LIBERO SimVLA-LIBERO
cd SimVLA-LIBERO && git lfs checkout   # LFS smudge가 안 됐다면
cd ..

# 3) LIBERO-Plus에 assets 복사 (405MB, repo엔 누락)
cp -r ~/workspace/Libero-pro_benchmark/LIBERO/libero/libero/assets \
      ~/workspace/LIBERO-plus/libero/libero/
```

---

## 3. 학습 컨테이너 이미지

### 3.1. Dockerfile 개요
`scripts/docker/train_simvla.Dockerfile` — `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` 베이스,
**miniforge** (conda-forge 채널, Anaconda TOS 우회) 설치, 두 conda env 분리:

| env | Python | 핵심 deps | 용도 |
|---|---|---|---|
| `simvla` | 3.10 | torch 2.5.1+cu124, transformers 4.57.3, peft 0.17.1, accelerate 1.2.1, flash-attn 2.5.6 | 학습 (`train_smolvlm.py`) |
| `libero` | 3.8.13 | numpy 1.22.4, transformers 4.21.1, robosuite 1.4.0, bddl 1.0.1, gym 0.25.2, robomimic 0.2.0 | 시뮬 + eval |

### 3.2. PIP_CONSTRAINT trick
`simvla` env는 torch 2.5.1을 `PIP_CONSTRAINT=/etc/pip.constraints.simvla` 로 전역 고정 — 그렇게 안 하면 peft/transformers가 torch nightly로 업그레이드 해버림. `libero` env 블록에 들어가기 전에 `ENV PIP_CONSTRAINT=` 로 해제해야 torch 1.x 계열이 막 안 올라감.

### 3.3. flash-attn non-fatal 빌드
~10분 컴파일 (A6000용 sm_86 + H100용 sm_90). 실패 시에도 SDPA로 폴백 가능하도록 `|| echo warn` 래핑. 빌드할 때만 `PIP_CONSTRAINT=` 로 해제 (build backend가 fresh env에서 torch 재해석).

### 3.4. Entrypoint (`scripts/docker/train_simvla_entrypoint.sh`)
컨테이너 start 시 다음을 **idempotent** 하게 수행:
1. `/libero_plus`가 bind mount 되어 있으면 `libero` env에 LIBERO-Plus를 editable install (`pip install -e /libero_plus`)
2. `/root/.libero/config.yaml` 프리셋 (LIBERO의 첫 import interactive prompt 스킵)
3. `exec "$@"` — 사용자 CMD 실행

### 3.5. 이미지 빌드
```bash
cd ~/workspace/SimVLA
docker build -t bigenlight/simvla-train:latest \
    -f scripts/docker/train_simvla.Dockerfile .
# 초회 ~25분 (flash-attn 컴파일), 이후 rebuild는 layer cache로 수초
```

---

## 4. 데이터 파이프라인 (LIBERO-Plus 2 task)

### 4.1. Target tasks (확정)
| Key | Task 이름 | LIBERO-Plus task_index | demos |
|---|---|---|---|
| **A** | `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 14 | **390** |
| **B** | `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 27 | **245** |

### 4.2. 다운로드 (Sylvest/libero_plus_lerobot, Task A/B만 ~1.4GB)
```bash
docker run --rm --network host \
    -v $(pwd):/app \
    -v $(pwd)/../LIBERO-plus:/libero_plus \
    -v $HOME/.cache/huggingface:/hf_cache \
    -e HF_HOME=/hf_cache \
    --entrypoint python bigenlight/simvla-http:latest \
    /app/scripts/download_libero_plus_taskAB.py --max-workers 4
```

- **왜 `simvla-http`?** 이미 `huggingface_hub` + `pyarrow` 가 깔려있어서. 호스트에 pip 설치 불필요.
- **workers=4**: HF xet-read-token rate limit (429)에 안 걸리는 상한. 병렬 16으로 올리면 429 폭발.
- **자동 재시도**: exponential backoff (4s→8s→…→60s) 내장.
- **누락 파일 resume**: `scripts/retry_missing_libero_plus.py` 단일-스레드 재시도.

### 4.3. Schema 실측 (inspect_libero_plus_sample.py)
- `observation.state` (8D float32): `[eef_pos(3), axis_angle(3), gripper_qpos(2)]` — **axis-angle** (rotvec magnitude ≈ π)
- `action` (7D float32): `[delta_xyz(3), delta_axisangle(3), gripper(1)]`, gripper는 **±1 바이너리**
- 이미지: parquet에 없고 **별도 AV1 mp4** (256×256 yuv420p, 20 fps, front + wrist)

### 4.4. LeRobot → SimVLA HDF5 변환 (`lerobot_to_libero_hdf5.py`)
SimVLA의 native `LiberoHDF5Handler`가 Euler 순서 `obs/ee_ori` 를 기대하고 내부에서 axis-angle로 변환 (`libero_hdf5.py:206`). 따라서 변환기는:
1. parquet → state/action 읽기
2. state[:, 3:6] **axis-angle → Euler (xyz)** 수동 변환 (Rodrigues + matrix→euler)
3. mp4 → pyav/libdav1d로 프레임별 디코드 → `[T, H, W, 3]` uint8
4. `data/demo_X/{actions, obs/{agentview_rgb, eye_in_hand_rgb, ee_pos, ee_ori, gripper_states}}` HDF5에 저장

**라운드트립**: axis-angle → Euler → (SimVLA loader) → axis-angle → 모델. gimbal lock 위험은 LIBERO 테이블탑에서는 실용적으로 무시 가능.

**실행**:
```bash
docker run --rm --network host \
    -v $(pwd):/app \
    -v $(pwd)/../LIBERO-plus:/libero_plus \
    --entrypoint python bigenlight/simvla-http:latest \
    /app/scripts/lerobot_to_libero_hdf5.py --max-workers 8
# 635 eps × ~3s/ep (병렬 8) ≈ 30분, 출력 22GB
```

### 4.5. Meta JSON + norm stats
```bash
# (simvla-train 이미지 안에서) Meta JSON 생성
python /app/scripts/make_libero_plus_meta.py
# → /app/datasets/metas/libero_plus_taskAB.json (635 demos, 2 tasks)

# Norm stats 재계산 (task A/B 분포 기반)
mkdir -p /libero_plus/data/libero_plus_hdf5_norm/libero_plus_AB
ln -sf /libero_plus/data/libero_plus_hdf5/*.hdf5 \
       /libero_plus/data/libero_plus_hdf5_norm/libero_plus_AB/
python /app/compute_libero_norm_stats.py \
    --data_dir /libero_plus/data/libero_plus_hdf5_norm \
    --subsets libero_plus_AB \
    --output /app/norm_stats/libero_plus_norm.json
```

### 4.6. DATA_WEIGHTS 레지스트리 패치
`datasets/domain_config.py` 에 두 새 subset 키를 추가함 (본 fork에 이미 반영):
```python
DATA_WEIGHTS = {
    ...기존..,
    "libero_plus_taskA": 1.0,
    "libero_plus_taskB": 1.0,
}
DATA_DOMAIN_ID = {..., "libero_plus_taskA": 0, "libero_plus_taskB": 0}
```

---

## 5. Fine-tune 실행

### 5.1. 런처 (`scripts/train_simvla_libero_plus.sh`)
모든 환경 변수는 override 가능:
```bash
NGPU=4            # 사용할 GPU 수
BS_PER_GPU=1      # per-GPU batch size (Large arch @ 384 → bs=1 약 9-12GB VRAM)
GRAD_ACCUM=2      # gradient accumulation (effective batch = NGPU × BS × GRAD_ACCUM = 8)
ITERS=3000        # total optimizer steps
SAVE_EVERY=1000
LR=5e-5           # paper 2e-4 대비 낮음 (fine-tune, 이미 trained ckpt에서 출발)
LR_COEF=0.1       # VLM LR multiplier — 절대 1.0으로 올리지 말 것! (paper ablation 44% collapse)
RESUME_CKPT=/app/SimVLA-LIBERO
NORM_STATS=/app/norm_stats/libero_plus_norm.json
```

### 5.2. 실행 커맨드
```bash
docker run -d --name simvla-train --network host \
    --gpus all --shm-size=32g \
    -v $(pwd):/app \
    -v $(pwd)/../LIBERO-plus:/libero_plus \
    -v $HOME/.cache/huggingface:/hf_cache \
    -e HF_HOME=/hf_cache \
    -e NGPU=4 -e BS_PER_GPU=1 -e GRAD_ACCUM=2 -e ITERS=3000 -e SAVE_EVERY=1000 \
    -e NORM_STATS=/app/norm_stats/libero_plus_norm.json \
    bigenlight/simvla-train:latest \
    bash /app/scripts/train_simvla_libero_plus.sh

docker logs -f simvla-train   # 진행 상황
```

### 5.3. 중요 플래그 설명
| 플래그 | 값 | 이유 |
|---|---|---|
| `--models` | `/app/SimVLA-LIBERO` | **pretrained ckpt 로드** |
| (없음: `--resume`) | 미지정 | `state.json`의 `global_step=150000` 이 복원되면 ITERS=3000에 즉시 종료됨 — 가중치만 로드하고 step은 0부터 |
| `--hidden_size 1024 --depth 24 --num_heads 16` | Large 고정 | 출고 ckpt가 Large arch. Small로 돌리면 shape 불일치로 에러 |
| `--action_mode libero_joint` | LIBERO 환경 | 기본 `galaxea_joint` 에서 이 값으로 덮어야 함 |
| `--freeze_steps 0 --warmup_steps 0` | fine-tune | 기본 1000은 VLM을 처음 1000 step 재동결 — pretrained ckpt에는 불필요 |
| `--num_actions 10` | LIBERO 최적 | paper: H=20/30은 LIBERO에서 열화 |
| `--max_grad_norm 1.0` | 안정성 | |
| `accelerate launch --mixed_precision bf16` | | A6000 tensor core 효율 |

### 5.4. Docker shm 필수
`--shm-size=32g` 없으면 DataLoader worker가 `Bus error` → `No space left on device` 로 죽음. `/dev/shm` 기본 64MB는 PyTorch shared tensor에 턱없이 부족.

---

## 6. Eval 실행 (Task A + Task B 한정)

### 6.1. 전제
- `simvla-http` 서버가 fine-tuned 체크포인트로 떠 있음 (포트 8700)
- `simvla-train` 이미지를 재활용해서 `libero` env 로 시뮬 돌림

### 6.2. 서버 기동 (fine-tuned ckpt)
```bash
docker rm -f simvla-http 2>/dev/null
docker run -d --name simvla-http --network host \
    --gpus '"device=0"' \
    -v $(pwd):/app \
    -v $(pwd)/runs/simvla_libero_plus_ft/ckpt-3000:/checkpoint:ro \
    -v $HOME/.cache/huggingface:/hf_cache \
    -e HF_HOME=/hf_cache \
    bigenlight/simvla-http:latest
# /health가 ok일 때까지 대기 (~1분, SmolVLM 베이스 재다운로드 없음)
```

### 6.3. Eval 드라이버 (`scripts/eval_taskAB_libero_plus.py`)
- Task A + Task B 원본 BDDL만 돌림 (LIBERO-Plus의 `_add_N` perturbation variants는 스킵)
- `OffScreenRenderEnv` 를 BDDL 경로로 직접 생성 (benchmark dict 의존성 제거)
- `Libero-pro_benchmark/scripts/vla_client.py` 를 sys.path로 재활용 (중복 코드 없음)

### 6.4. 실행 커맨드
```bash
docker run --rm --name simvla-eval --network host \
    --gpus '"device=1"' --shm-size=8g \
    -v $(pwd):/app \
    -v $(pwd)/../LIBERO-plus:/libero_plus \
    -v $(pwd)/../Libero-pro_benchmark:/libero_pro_benchmark:ro \
    -e VLA_SERVER_URL=http://localhost:8700 \
    bigenlight/simvla-train:latest \
    bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate libero && \
      python /app/scripts/eval_taskAB_libero_plus.py \
        --tasks A,B --num-trials 3 --max-steps 520 \
        --output-dir /app/runs/eval_taskAB_ckpt3000"
```

결과: `runs/eval_taskAB_ckpt3000/eval_taskAB_<timestamp>/{summary.json, videos/task*_t*_{success,failure}.mp4}`

---

## 7. 하이퍼파라미터 민감도 (주의사항)

**SimVLA는 하이퍼파라미터에 매우 민감** (논문 §4.1.1, Table 6 ablation). 다음은 **절대** 건들면 안 되는 것들:

| HP | 기본값 | 잘못 바꿀 때 |
|---|---|---|
| `--learning_coef` (VLM LR multiplier) | **0.1** | 1.0 → LIBERO avg **44%** (정상 98%) |
| `--learning_rate` | 5e-5 (fine-tune) / 2e-4 (pretrain) | 5e-4 → **72%**로 붕괴 |
| Data shuffling | on (기본) | off → **9.9%** 대참사 |
| Action normalization | on (norm_stats 필수) | off → **12.3%** 붕괴 |
| `--num_actions` | 10 (LIBERO) | 20/30 → 성능 하락 |
| VLM freeze_steps | 0 (resume 시) | 1000 남기면 첫 1000 step 무용지물 |

### 추천 fine-tune 레시피
- LR 5e-5 (pretrain LR의 ~1/4)
- VLM LR multiplier 0.1 고정
- freeze_steps=0, warmup_steps=0 (이미 pretrained에서 출발)
- batch 8-32 effective (우리는 8)
- 3000-10000 iter (scratch 150k와 비교 불필요, pretrain에서 fine-tune이라)

---

## 8. 실측 결과 (2026-04-21~22)

### 8.1. 환경
- 서버: RTX A6000 × 4 (48GB each)
- 데이터: Task A (390 demos) + Task B (245 demos) = 635 demos
- 체크포인트: `SimVLA-LIBERO` (Large, 150k pretrain) 에서 출발

### 8.2. 학습 성능
- **Step 시간: 0.61 s/iter** (4 GPU, bs=1, grad_accum=2, bf16)
- **3000 iter wall-clock: 약 31분**
- **VRAM**: GPU당 ~12 GB
- **Loss**: 초기 1.88 → 마지막 ~0.05-0.20 (flow matching noise)

### 8.3. Eval (ckpt-3000, 3 trials per task, max-steps 520)
| Task | Trial 0 | Trial 1 | Trial 2 | 성공률 | 평균 latency |
|---|:-:|:-:|:-:|:-:|:-:|
| A: turn on stove + moka pot | ❌ (520) | ❌ (520) | ❌ (520) | **0/3** | 176 ms/call |
| B: bowl in drawer + close | ✅ (415) | ✅ (212) | ✅ (339) | **3/3** | 178 ms/call |
| **Total** | | | | **3/6 = 50.0%** | |

### 8.4. 해석
- Task B 3/3: 파이프라인 end-to-end 검증 성공
- Task A 0/3 원인 후보:
  1. Task A가 내재적으로 더 어려움 (stove 켜기 + pick/place 4단계)
  2. `env.reset()` 랜덤 init이 학습 분포와 mismatch (`set_init_state` 미사용)
  3. 3000 iter가 부족 (10k+ 권장)
  4. max-steps 520이 빠듯 (longest demo 약 500+)

### 8.5. 학습 시간 스케일 참고
| iter 수 | 예상 wall-clock (4×A6000) |
|---|---|
| 1000 | ~10분 |
| 3000 | **~31분** (실측) |
| 10000 | ~1시간 45분 |
| 30000 | ~5시간 15분 |
| 150000 (scratch pretrain) | ~26시간 |

---

## 9. 파일 레퍼런스

### 신규 작성된 파일 (이 fork)
```
scripts/
├── download_libero_plus_taskAB.py      # HF Task A/B 다운로드 (retry + backoff)
├── retry_missing_libero_plus.py        # 누락 파일 재시도
├── inspect_libero_plus_sample.py       # parquet + mp4 schema 실측
├── lerobot_to_libero_hdf5.py           # 변환기 (axis-angle → Euler, mp4 디코드)
├── make_libero_plus_meta.py            # 학습 meta JSON 생성
├── train_simvla_libero_plus.sh         # fine-tune 런처 (accelerate launch)
├── eval_taskAB_libero_plus.py          # Task A/B 전용 eval 드라이버
├── serve_simvla_http.py                # (이미 있음) FastAPI inference 서버
└── docker/
    ├── train_simvla.Dockerfile         # 학습 이미지 (conda 2-env)
    ├── train_simvla_entrypoint.sh      # LIBERO-Plus editable install + config 프리셋
    ├── serve_simvla_http.Dockerfile    # (이미 있음) inference 이미지
    ├── simvla_http_compose.yml
    └── simvla_http_entrypoint.sh
datasets/
├── metas/libero_plus_taskAB.json       # 변환된 HDF5를 가리키는 meta
└── domain_config.py                    # DATA_WEIGHTS에 libero_plus_task{A,B} 추가
norm_stats/
└── libero_plus_norm.json               # Task A/B 분포 기반 재계산된 norm stats
```

### 변경/추가되지 않는 원본 SimVLA 파일
- `train_smolvlm.py` — 건드리지 않고 그대로 사용 (`--models` + `--hidden_size 1024 --depth 24 --num_heads 16` 조합)
- `datasets/dataset_smolvlm.py`, `datasets/domain_handler/libero_hdf5.py` — 그대로
- `models/modeling_smolvlm_vla.py` — 그대로

### 외부 레퍼런스
- LIBERO-Plus: https://github.com/sylvestf/LIBERO-plus
- LeRobot 데이터: https://huggingface.co/datasets/Sylvest/libero_plus_lerobot
- 체크포인트: https://huggingface.co/YuankaiLuo/SimVLA-LIBERO
- SimVLA 논문: arxiv 2602.18224
- LIBERO-Plus 논문: arxiv 2510.13626

---

## 10. Known issues + Gotchas

### 빌드 단계
- **Anaconda TOS**: `conda defaults` 채널은 TOS 동의 없이 설치 불가 → **miniforge + conda-forge** 사용
- **PIP_CONSTRAINT leak**: simvla env의 torch 2.5.1 핀이 libero env로 전파되어 torch 1.11 설치 실패 → `ENV PIP_CONSTRAINT=` 로 해제
- **flash-attn 빌드 실패 시**: `--attn-impl sdpa` 로 폴백 가능, 성능 10-20% 손실

### 런타임 (training)
- **`Bus error`/`No space left on device`**: `--shm-size=32g` 없으면 DataLoader 죽음
- **`Resuming from step: 150000`**: `--resume` 플래그가 state.json의 step을 복원 → fine-tune에서는 **제거 필수**
- **Shape mismatch**: 출고 ckpt는 **Large** 이므로 `--hidden_size 1024 --depth 24 --num_heads 16` 필수

### 런타임 (eval)
- **LIBERO `EOFError: EOF when reading a line`**: 첫 import가 interactive Y/N prompt → entrypoint가 `/root/.libero/config.yaml` 프리셋으로 우회
- **`FileNotFoundError: libero_kitchen_tabletop_base_style.xml`**: LIBERO-Plus repo에 `assets/` 없음 → 원본 LIBERO에서 **물리 복사** (심볼릭 링크는 호스트 경로라 컨테이너에서 깨짐)
- **`ModuleNotFoundError: termcolor / matplotlib`**: robosuite 1.4 transitive deps 누락 → Dockerfile에 baked in (또는 런타임 `pip install` 해도 됨)
- **`robosuite.macros_private` 없음**: `python -m robosuite.scripts.setup_macros` 필요 — Dockerfile에 포함

### 데이터
- **HF 429 rate limit**: xet-read-token에 IP-global 제한. workers 4 이하 + exponential backoff 권장
- **state[3:6] 해석**: LIBERO-Plus LeRobot은 **axis-angle**이지만 SimVLA HDF5 loader는 **Euler** 기대 → 변환 때 axis-angle → Euler 수동 변환 필수

---

**문서 개정**: 2026-04-22. 원본 conda 워크플로우는 `readme.md` 하단을 참고.
