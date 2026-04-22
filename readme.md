# SimVLA: A Simple VLA Baseline for Robotic Manipulation

| **Paper** | **Website** | **Model & Data** |
| :------------------: | :-----------------------: | :---------------------: |
| [![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.18224) | [![Website](https://img.shields.io/badge/Project%20Page-181717?style=for-the-badge&logo=githubpages&logoColor=white)](https://frontierrobo.github.io/SimVLA/) | [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFBA00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/collections/YuankaiLuo/simvla) |

A simple and efficient Vision-Language-Action (VLA) model for robot manipulation tasks.

<img width="506" height="796" alt="image" src="https://github.com/user-attachments/assets/7ffb8969-aa4f-4bcc-8c38-33d5e7da4b25" />

# 결론: RTX3060(12GB) 만으로는 트레이닝이 어려움. (매우 오래 걸리고 배치 사이즈를 낮춰야하기에 성공률을 장담하지 못함.)

---

# 🐳 Docker 기반 세팅 (bigenlight 서버 기준, unified VLA protocol)

> 이 섹션은 아래의 conda 기반 "완전 정복 가이드"와는 별도의 흐름이다.
> 기존 가이드는 **conda 2-env (simvla + libero) + WebSocket+msgpack** 경로고,
> 이 섹션은 **Docker 2-컨테이너 + FastAPI HTTP (unified VLA protocol)** 경로다.
> 하나를 고르면 된다 — Pi0.5 / OpenVLA / X-VLA 등 다른 VLA 모델을 이미 돌리고 있고 동일한 벤치마크(Libero-pro)로 공정 비교하고 싶을 때 이 쪽을 쓴다.
>
> **👉 Docker 기반 학습(fine-tune) + LIBERO-Plus Task A/B eval 상세 가이드는 [`TRAINING.md`](TRAINING.md) 참조.** 데이터 변환 파이프라인, 학습 컨테이너 구성, 민감 하이퍼파라미터, 실측 결과(3000 iter / 31분 / Task B 3/3 성공) 까지 포함.

## 1. 아키텍처

```
┌───────────────────────────────┐        ┌─────────────────────────────┐
│  simvla-http  (GPU)           │        │  libero-pro  (GPU)          │
│  FastAPI :8700                │  HTTP  │  robosuite / MuJoCo render  │
│  /health /reset /act          │◀──────▶│  scripts/libero_vla_eval.py │
│  - SmolVLM + action tf        │        │  - 에피소드 loop            │
│  - 180° rotate + 224 resize   │        │  - action chunk 소진        │
└───────────────────────────────┘        └─────────────────────────────┘
              ▲                                         ▲
              │ bind mount                              │ bind mount
              │   /app    ← SimVLA (이 리포)            │   /workspace/LIBERO-PRO
              │   /checkpoint ← ./SimVLA-LIBERO (3GB)   │   /workspace/LIBERO
              │   /hf_cache ← ~/.cache/huggingface      │   ood_data, test_outputs
```

컨테이너 한 개 = VLA 모델 한 개, 컨테이너 한 개 = 벤치마크 한 개.
컨테이너는 `--network host` 로 띄워서 `localhost:8700` 으로 통신한다.
포트 관례: **OpenVLA 8600 / Pi0.5 8400 / SimVLA 8700**.

통신 규약은 `VLA_COMMUNICATION_PROTOCOL.md` (이 워크스페이스 루트) — `observation.images.{static,wrist}` (base64 PNG HWC uint8) + `observation.state.*` + `task` → `action.eef_pos / action.eef_euler / action.gripper` (`[10, D]` chunk, `action_type="relative"`).

## 2. Pre-requisites (호스트)

```bash
# 1) SimVLA 소스 (= 이 리포)
cd ~/workspace          # 또는 원하는 루트
git clone <이 리포 URL> SimVLA

# 2) 체크포인트 (~3 GB, LFS) — SimVLA 리포 안에 둔다
cd SimVLA
git clone https://huggingface.co/YuankaiLuo/SimVLA-LIBERO SimVLA-LIBERO
# LFS 파일이 자동으로 materialize 안되면:  cd SimVLA-LIBERO && git lfs checkout
cd ..

# 3) LIBERO / LIBERO-PRO 소스 (벤치마크 컨테이너가 bind mount 함)
cd ~/workspace/Libero-pro_benchmark       # 이 레포는 별도 clone
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git LIBERO-PRO
```

호스트에 Docker + NVIDIA Container Toolkit + GPU 드라이버 ≥ 550 (CUDA 12.4) 필요. 본 세팅은 **RTX A6000 × 4 (48GB each)** 에서 검증됐다 (SimVLA 서버 단독 ~3.4 GB VRAM).

## 3. 이미지 빌드 or 풀

**Docker Hub 에서 풀** (권장, 빌드 12GB/~15분 생략):
```bash
docker pull bigenlight/simvla-http:latest
```

**로컬 빌드** (수정 반영이 필요할 때):
```bash
cd ~/workspace/SimVLA
docker build -t bigenlight/simvla-http:latest \
    -f scripts/docker/serve_simvla_http.Dockerfile .
```
초기 빌드 시 flash-attn CUDA 커널 컴파일이 ~10분, 총 ~15분 소요. 이후 rebuild는 캐시 덕에 30초대.

이미지 구성: CUDA 12.4.1-cudnn-devel-ubuntu22.04 + Python 3.10 + torch 2.5.1+cu124 + transformers 4.57.3 + peft 0.17.1 + flash-attn 2.5.6 + FastAPI/uvicorn. 소스는 이미지에 굽지 않고 **`/app` 에 bind mount** 하므로 SimVLA 쪽 코드 수정은 재빌드 없이 즉시 반영된다.

## 4. 서버 띄우기

```bash
cd ~/workspace/SimVLA

docker run -d --name simvla-http --network host \
    --gpus '"device=0"' \
    -v "$(pwd):/app" \
    -v "$(pwd)/SimVLA-LIBERO:/checkpoint:ro" \
    -v "$HOME/.cache/huggingface:/hf_cache" \
    -e HF_HOME=/hf_cache \
    bigenlight/simvla-http:latest
```

또는 docker compose:
```bash
docker compose -f scripts/docker/simvla_http_compose.yml up -d
```

첫 기동 시 SmolVLM-500M base weights (~1GB) 가 `~/.cache/huggingface` 로 다운로드되고 이후 재기동은 즉시. 서버 ready 확인:
```bash
curl -s http://localhost:8700/health
# => {"status":"ok","model":"simvla","action_type":"relative",
#     "action_keys":["action.eef_pos","action.eef_euler","action.gripper"],
#     "n_action_steps":10}
```

**--dummy 스모크 테스트** (체크포인트 없이 프로토콜 배선만 검증):
```bash
docker run --rm --network host -v $(pwd):/app \
    -e SIMVLA_HTTP_ARGS="--dummy" bigenlight/simvla-http:latest
curl -s http://localhost:8700/health   # model = "simvla_dummy"
```

## 5. LIBERO Evaluation (libero-pro 컨테이너에서)

`Libero-pro_benchmark` 리포의 `run.sh` 가 `bigenlight/libero-pro:latest` 를 띄우고 `scripts/libero_vla_eval.py` 를 실행한다. 별도 SimVLA 쪽 코드 변경 불필요.

```bash
cd ~/workspace/Libero-pro_benchmark

# 1 task × 1 trial 스모크 테스트 (약 1분)
./run.sh --vla-eval libero_spatial \
    --vla-url http://localhost:8700 \
    --vla-num-tasks 1 --vla-num-trials 1

# full libero_spatial (10 task × 10 trial) — A6000 기준 ~30분
./run.sh --vla-eval libero_spatial \
    --vla-url http://localhost:8700 \
    --vla-num-tasks 10 --vla-num-trials 10
```

결과: `test_outputs/eval/libero_spatial_<timestamp>/summary.json` + `videos/*.mp4`.

### 검증 결과 (2026-04-21, RTX A6000)
- `libero_spatial` task 0 ("Pick the akita black bowl ...") 1 trial → **성공 (73 step)**, 평균 latency 191ms/call
- `/health` `/act` 응답 shape: `action.eef_pos (10,3)` / `action.eef_euler (10,3)` / `action.gripper (10,1)` — 프로토콜 준수
- 서버 단독 VRAM 사용량: ~3.4 GB

## 6. 프로토콜 매핑 요약 (디버깅용)

native SimVLA (WebSocket+msgpack, `evaluation/libero/serve_smolvlm_libero.py`) 와 이 FastAPI 서버의 매핑:

| 차원 | native (SimVLA 원본) | Docker HTTP (이 리포 `scripts/serve_simvla_http.py`) |
|---|---|---|
| Transport | WebSocket + msgpack_numpy | FastAPI + JSON |
| 포트 | 8102 (예) | **8700** |
| 이미지 수신 | 224×224 uint8 HWC (클라이언트가 이미 180° 회전 + resize_with_pad) | 256×256 raw (서버가 180° 회전 + 224 resize 수행) |
| state | `observation/state` 8D array | `observation.state.{eef_pos,eef_quat,gripper_qpos}` → 내부에서 8D 조립 (`eef_quat` → axis-angle 변환) |
| language | `prompt` | `task` |
| action | `{"actions": [10,7]}` | `action.eef_pos[10,3]` + `action.eef_euler[10,3]` (axis-angle이 euler 슬롯) + `action.gripper[10,1]` |
| chunk 재구성 | client가 `[10,7]` 그대로 popleft | Libero-pro 쪽 `_assemble_action_from_subkeys` 가 `[pos,rot,grip]` concat → 7D OSC_POSE |

전처리/상태 조립 코드는 전부 서버 안에 들어있으므로 벤치마크 쪽은 SimVLA 를 전혀 모른 채 동작한다.

## 7. 파일 레퍼런스

```
scripts/
├── serve_simvla_http.py                    # FastAPI 서버
└── docker/
    ├── serve_simvla_http.Dockerfile        # 이미지 빌드
    ├── simvla_http_compose.yml             # compose 헬퍼
    └── simvla_http_entrypoint.sh           # 컨테이너 런타임
```

---

# 🚀 SimVLA + LIBERO 완전 정복 가이드 (Ubuntu 24.04 / 12GB VRAM 기준) — conda 기반 원본 워크플로우

본 가이드는 단일 소비자용 GPU(예: RTX 3060 12GB) 환경에서 SimVLA 모델을 세팅하고, 평가(Evaluation) 및 훈련(Training)까지 진행하기 위한 최적화된 워크플로우를 담고 있습니다.

> **참고**: 위의 Docker 섹션과 이 conda 섹션은 **병렬 경로**입니다. 훈련(Training)은 이 conda 가이드를 따라가세요 — Docker 이미지는 inference 전용입니다.


## 1. 소스 코드 및 모델 체크포인트 다운로드

작업을 진행할 홈 디렉토리(`~`)에서 코드를 클론하고 허깅페이스 모델을 다운로드합니다.

```bash
cd ~

# 1. SimVLA 공식 코드 클론 (본인의 Fork 저장소가 있다면 해당 주소 사용)
git clone https://github.com/LUOyk1999/SimVLA.git

# 2. 허깅페이스 사전 학습 모델 다운로드 (Git LFS 필요) — SimVLA 리포 안에 둔다
cd SimVLA
git clone https://huggingface.co/YuankaiLuo/SimVLA-LIBERO SimVLA-LIBERO
```

## 2. 가상환경 세팅
SimVLA는 서버(모델 추론/학습)와 클라이언트(LIBERO 시뮬레이션) 환경을 분리하여 구동해야 합니다.

### 2.1. 서버 가상환경 (simvla)
모델을 메모리에 올리고 추론/훈련을 담당할 환경입니다. 최신 라이브러리의 이중 로드 에러를 막기 위해 특정 버전을 지정하여 설치합니다.

```bash
conda create -n simvla python=3.10 -y
conda activate simvla

# PyTorch (CUDA 12.4 기준)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

# 검증된 버전의 Hugging Face 패키지 설치
pip install transformers==4.57.3 accelerate==1.2.1 peft==0.17.1 safetensors==0.4.5 tokenizers==0.22.1 huggingface-hub==0.36.0

# 기타 필수 패키지 설치
pip install fastapi tensorboard uvicorn json_numpy scipy einops timm mmengine pyarrow h5py mediapy num2words av wandb websockets msgpack_numpy
pip install flash-attn==2.5.6 --no-build-isolation
pip install tensorflow tensorflow-datasets
```

### 2.2. 클라이언트 가상환경 (libero)
로봇 시뮬레이션(MuJoCo)을 띄우고 통신할 환경입니다.

```bash
conda create -n libero python=3.8.13 -y
conda activate libero

# LIBERO 공식 환경 설치
cd ~
git clone [https://github.com/Lifelong-Robot-Learning/LIBERO.git](https://github.com/Lifelong-Robot-Learning/LIBERO.git)
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url [https://download.pytorch.org/whl/cu113](https://download.pytorch.org/whl/cu113)
pip install -e .

# 클라이언트 통신 및 비디오 저장을 위한 추가 패키지
pip install json_numpy imageio requests tqdm openpi-client
```

## 3. 코드 최적화

- 메모리 절약을 위한 그래디언트 누적(Gradient Accumulation) 추가

- ~/SimVLA/train_smolvlm.py 수정:

  - get_args_parser()에 --gradient_accumulation_steps 인자 추가.

  - Accelerator 객체 초기화 시 gradient_accumulation_steps=args.gradient_accumulation_steps 추가.

  - 훈련 루프(for batch in train_dataloader:) 내부를 with accelerator.accumulate(model):로 감싸고, accelerator.sync_gradients 조건 추가.

- ~/SimVLA/train_smolvlm_small.sh 수정:

  - BATCH_SIZE=1로 변경

  - accelerate launch의 --num_processes=1로 변경

  - ARGS에 --gradient_accumulation_steps 8 추가

## 4. Evaluation (허깅페이스 사전 학습 모델 테스트)

터미널 2개를 열어 서버와 클라이언트를 각각 구동합니다.

[터미널 1: 서버 실행]

```bash
conda activate simvla
cd ~/SimVLA

CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
    --checkpoint ./SimVLA-LIBERO \
    --norm_stats ./norm_stats/libero_norm.json \
    --port 8102
```

[터미널 2: 클라이언트 실행]

```bash
conda activate libero
cd ~/SimVLA

CUDA_VISIBLE_DEVICES=0 python evaluation/libero/libero_client.py \
    --host 127.0.0.1 \
    --port 8102 \
    --client_type websocket \
    --task_suite libero_spatial \
    --num_trials 10 \
    --video_out "./eval_simvla"
```

## 5. 훈련을 위한 LIBERO 데이터셋 다운로드 및 연결

훈련(Training)을 진행하려면 전체 데모 데이터가 필요합니다.

1) 데이터셋 다운로드

```bash
conda activate libero
cd ~/LIBERO
python benchmark_scripts/download_libero_datasets.py
```

2) SimVLA 폴더에 데이터 심볼릭 링크 연결

```bash
cd ~/SimVLA
mkdir -p ./datasets/metas

ln -sf ~/LIBERO/libero/datasets/libero_10 ./datasets/metas/
ln -sf ~/LIBERO/libero/datasets/libero_goal ./datasets/metas/
ln -sf ~/LIBERO/libero/datasets/libero_object ./datasets/metas/
ln -sf ~/LIBERO/libero/datasets/libero_spatial ./datasets/metas/
ln -sf ~/LIBERO/libero/datasets/libero_90 ./datasets/metas/
```

3) 훈련 메타데이터 생성

```bash
conda activate simvla
python create_libero_meta.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial libero_90 \
    --output ./datasets/metas/libero_train.json
```

## 6. 로컬 훈련 (Training) 시작

위의 "3. 코드 최적화 수정"이 완료된 상태에서 훈련 스크립트를 실행합니다. (12GB 환경에서 약 17시간 소요)

```bash
conda activate simvla
cd ~/SimVLA

# 훈련 시작
bash train_smolvlm_small.sh
```

체크포인트는 ~/SimVLA/runs/simvla_libero_small/ 폴더에 ckpt-50000 등의 형태로 저장됩니다.

## 7. 로컬 훈련 모델 Evaluation

우리가 직접 학습시킨 체크포인트를 올려서 성능을 평가해 봅니다.

[터미널 1: 로컬 모델 서빙]

```bash
conda activate simvla
cd ~/SimVLA

CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
    --checkpoint ./runs/simvla_libero_small/ckpt-50000 \
    --norm_stats ./norm_stats/libero_norm.json \
    --port 8102
```

*체크포인트는 ~/SimVLA/runs/simvla_libero_small/ 폴더에 ckpt-50000 등의 형태로 저장됩니다.*

[터미널 2: 평가 진행 및 비디오 저장]

```bash
conda activate libero
cd ~/SimVLA

CUDA_VISIBLE_DEVICES=0 python evaluation/libero/libero_client.py \
    --host 127.0.0.1 \
    --port 8102 \
    --client_type websocket \
    --task_suite libero_spatial \
    --num_trials 10 \
    --video_out "./eval_my_simvla"
```

*완료 후 ./eval_my_simvla 폴더에서 로봇의 실제 조작 영상(.mp4)을 확인가능.*

## Model Architecture

- **Vision-Language Backbone**: SmolVLM-500M-Instruct (576 hidden dim)
- **Action Transformer**: Configurable depth and width
  - Small: 768 hidden, 12 layers, 12 heads
  - Large: 1024 hidden, 24 layers, 16 heads
  
## Reference

If you find our codes useful, please consider citing our work

```
@article{luo2026simvla,
  title={SimVLA: A Simple VLA Baseline for Robotic Manipulation},
  author={Luo, Yuankai and Chen, Woping and Liang, Tong and Wang, Baiqiao and Li, Zhenguo},
  journal={arXiv preprint arXiv:2602.18224},
  year={2026}
}
```


