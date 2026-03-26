# SimVLA: A Simple VLA Baseline for Robotic Manipulation

| **Paper** | **Website** | **Model & Data** |
| :------------------: | :-----------------------: | :---------------------: |
| [![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.18224) | [![Website](https://img.shields.io/badge/Project%20Page-181717?style=for-the-badge&logo=githubpages&logoColor=white)](https://frontierrobo.github.io/SimVLA/) | [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFBA00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/collections/YuankaiLuo/simvla) |

A simple and efficient Vision-Language-Action (VLA) model for robot manipulation tasks.

<img width="506" height="796" alt="image" src="https://github.com/user-attachments/assets/7ffb8969-aa4f-4bcc-8c38-33d5e7da4b25" />

# 결론: RTX3060(12GB) 만으로는 트레이닝이 어려움. (매우 오래 걸리고 배치 사이즈를 낮춰야하기에 성공률을 장담하지 못함.)

# 🚀 SimVLA + LIBERO 완전 정복 가이드 (Ubuntu 24.04 / 12GB VRAM 기준)

본 가이드는 단일 소비자용 GPU(예: RTX 3060 12GB) 환경에서 SimVLA 모델을 세팅하고, 평가(Evaluation) 및 훈련(Training)까지 진행하기 위한 최적화된 워크플로우를 담고 있습니다.


## 1. 소스 코드 및 모델 체크포인트 다운로드

작업을 진행할 홈 디렉토리(`~`)에서 코드를 클론하고 허깅페이스 모델을 다운로드합니다.

```bash
cd ~

# 1. SimVLA 공식 코드 클론 (본인의 Fork 저장소가 있다면 해당 주소 사용)
git clone [https://github.com/LUOyk1999/SimVLA.git](https://github.com/LUOyk1999/SimVLA.git)

# 2. 허깅페이스 사전 학습 모델 다운로드 (Git LFS 필요)
git clone [https://huggingface.co/YuankaiLuo/SimVLA-LIBERO](https://huggingface.co/YuankaiLuo/SimVLA-LIBERO) ~/SimVLA-LIBERO
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
    --checkpoint ~/SimVLA-LIBERO \
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


