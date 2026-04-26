# Huong dan train 3 cai tien TransReID tren Market-1501

Source nay duoc phat trien tu TransReID va them cac bien the train cho Market-1501:

- `configs/Market/vit_transreid_stride_sem_align.yml`: TransReID + semantic alignment bang mask phan vung nguoi.
- `configs/Market/vit_transreid_stride_local_reliability.yml`: TransReID + reliability-aware local modeling voi anh bi che tong hop trong luc train.
- `configs/Market/vit_transreid_stride_sem_align_reliability.yml`: ket hop semantic alignment va local reliability.

Tat ca lenh ben duoi chay tu thu muc `transreid_colab`. Cac lenh nhieu dong dung cu phap Bash/Colab voi ky tu `\`; neu chay trong Windows PowerShell, hay thay `\` bang backtick `` ` `` hoac viet thanh mot dong.

## 1. Chuan bi moi truong

Tao virtual environment:

```bash
cd transreid_colab
python -m venv .venv
```

Kich hoat moi truong:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS / Colab
source .venv/bin/activate
```

Cai dependency:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Neu can tu sinh semantic mask bang model human parsing:

```bash
pip install transformers accelerate
```

## 2. Chuan bi pretrained ViT

Tao thu muc pretrained va tai ViT-Base ImageNet checkpoint:

```bash
mkdir -p ../pretrained
wget -O ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
```

Tren Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force ..\pretrained
Invoke-WebRequest `
  -Uri "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth" `
  -OutFile "..\pretrained\jx_vit_base_p16_224-80ecf9dd.pth"
```

## 3. Chuan bi Market-1501

Dat dataset theo cau truc sau, tinh tu `transreid_colab`:

```text
../data/market1501/
  bounding_box_train/
  query/
  bounding_box_test/
```

Kiem tra nhanh:

```bash
python - <<'PY'
from pathlib import Path
root = Path("../data/market1501")
for name in ["bounding_box_train", "query", "bounding_box_test"]:
    p = root / name
    print(name, p.exists(), len(list(p.glob("*.jpg"))) if p.exists() else 0)
PY
```

Neu dataset dat o vi tri khac, thay `DATASETS.ROOT_DIR ../data` trong cac lenh train/test bang root dang chua folder `market1501`.

## 4. Chuan bi semantic mask

Buoc nay bat buoc cho:

- `vit_transreid_stride_sem_align.yml`
- `vit_transreid_stride_sem_align_reliability.yml`

Khong bat buoc cho:

- `vit_transreid_stride_local_reliability.yml`

Code se tim mask theo dung relative path cua anh:

```text
../data/market1501/semantic_groups/bounding_box_train/0002_c1s1_000451_03.png
```

Tuong ung voi anh:

```text
../data/market1501/bounding_box_train/0002_c1s1_000451_03.jpg
```

### Cach A: sinh mask bang tool co san

Tool mac dinh dung model `fashn-ai/fashn-human-parser` va preset `fashn6`:

```bash
python tools/prepare_semantic_maps.py \
  --dataset-root ../data/market1501 \
  --output-root ../data/market1501/semantic_groups \
  --preset fashn6 \
  --batch-size 8 \
  --device cuda
```

Voi preset `fashn6`, cac group khong phai cap trai/phai nhu LIP, vi vay khi train semantic config can override:

```bash
MODEL.SEM_ALIGN.FLIP_LABEL_PAIRS []
```

### Cach B: da co raw parsing mask

Neu da co raw label map tu parser khac, convert sang 6 nhom:

```bash
python tools/build_semantic_group_masks.py \
  --input-root /path/to/raw_masks \
  --output-root ../data/market1501/semantic_groups \
  --preset lip6
```

Voi preset `lip6`, co the giu nguyen `FLIP_LABEL_PAIRS` trong config: `[[3, 4], [5, 6]]`.

## 5. Train tu dau

Nhung override nen giu trong moi lenh:

- `MODEL.DEVICE_ID "('0')"`: chon GPU 0.
- `MODEL.PRETRAIN_PATH ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth`: checkpoint ViT ImageNet.
- `DATASETS.ROOT_DIR ../data`: root dataset.
- `OUTPUT_DIR ...`: thu muc log va checkpoint rieng cho tung thuc nghiem.

### 5.1. Semantic alignment

Neu dung semantic mask preset `fashn6` sinh bang tool trong repo:

```bash
python train.py \
  --config_file configs/Market/vit_transreid_stride_sem_align.yml \
  MODEL.DEVICE_ID "('0')" \
  MODEL.PRETRAIN_PATH ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR ../data \
  MODEL.SEM_ALIGN.FLIP_LABEL_PAIRS [] \
  OUTPUT_DIR ../logs/market_sem_align
```

Neu dung mask `lip6`, bo dong `MODEL.SEM_ALIGN.FLIP_LABEL_PAIRS []`.

### 5.2. Local reliability

Bien the nay khong can semantic mask:

```bash
python train.py \
  --config_file configs/Market/vit_transreid_stride_local_reliability.yml \
  MODEL.DEVICE_ID "('0')" \
  MODEL.PRETRAIN_PATH ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR ../data \
  OUTPUT_DIR ../logs/market_local_reliability
```

### 5.3. Semantic alignment + reliability

Neu dung semantic mask preset `fashn6`:

```bash
python train.py \
  --config_file configs/Market/vit_transreid_stride_sem_align_reliability.yml \
  MODEL.DEVICE_ID "('0')" \
  MODEL.PRETRAIN_PATH ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR ../data \
  MODEL.SEM_ALIGN.FLIP_LABEL_PAIRS [] \
  OUTPUT_DIR ../logs/market_sem_align_reliability
```

Neu dung mask `lip6`, bo dong `MODEL.SEM_ALIGN.FLIP_LABEL_PAIRS []`.

Mac dinh moi config train `120` epoch, validate moi `10` epoch, save checkpoint moi `5` epoch. Checkpoint duoc luu thanh:

```text
../logs/<ten_thuc_nghiem>/transformer_5.pth
../logs/<ten_thuc_nghiem>/transformer_10.pth
...
../logs/<ten_thuc_nghiem>/transformer_120.pth
../logs/<ten_thuc_nghiem>/transformer_best.pth
```

## 6. Resume train

Vi du tiep tuc semantic alignment tu epoch 60 den 120:

```bash
python train.py \
  --config_file configs/Market/vit_transreid_stride_sem_align.yml \
  MODEL.DEVICE_ID "('0')" \
  MODEL.PRETRAIN_CHOICE self \
  MODEL.PRETRAIN_PATH ../logs/market_sem_align/transformer_60.pth \
  DATASETS.ROOT_DIR ../data \
  MODEL.SEM_ALIGN.FLIP_LABEL_PAIRS [] \
  SOLVER.START_EPOCH 60 \
  SOLVER.MAX_EPOCHS 120 \
  OUTPUT_DIR ../logs/market_sem_align
```

Doi config, checkpoint path va `OUTPUT_DIR` tuong ung cho hai bien the con lai. Neu resume config dung mask `lip6`, bo override `MODEL.SEM_ALIGN.FLIP_LABEL_PAIRS []`.

## 7. Evaluate checkpoint

Dung `transformer_best.pth` de bao cao ket qua chinh. Vi du:

```bash
python test.py \
  --config_file configs/Market/vit_transreid_stride_sem_align.yml \
  MODEL.DEVICE_ID "('0')" \
  DATASETS.ROOT_DIR ../data \
  MODEL.SEM_ALIGN.FLIP_LABEL_PAIRS [] \
  TEST.WEIGHT ../logs/market_sem_align/transformer_best.pth \
  OUTPUT_DIR ../logs/market_sem_align_eval
```

Local reliability:

```bash
python test.py \
  --config_file configs/Market/vit_transreid_stride_local_reliability.yml \
  MODEL.DEVICE_ID "('0')" \
  DATASETS.ROOT_DIR ../data \
  TEST.WEIGHT ../logs/market_local_reliability/transformer_best.pth \
  OUTPUT_DIR ../logs/market_local_reliability_eval
```

Semantic alignment + reliability:

```bash
python test.py \
  --config_file configs/Market/vit_transreid_stride_sem_align_reliability.yml \
  MODEL.DEVICE_ID "('0')" \
  DATASETS.ROOT_DIR ../data \
  MODEL.SEM_ALIGN.FLIP_LABEL_PAIRS [] \
  TEST.WEIGHT ../logs/market_sem_align_reliability/transformer_best.pth \
  OUTPUT_DIR ../logs/market_sem_align_reliability_eval
```

Ket qua `mAP`, `Rank-1`, `Rank-5`, `Rank-10` nam trong `test_log.txt` cua `OUTPUT_DIR`. Trong qua trinh train, log nam trong `train_log.txt`.

## 8. Ghi chu cau hinh

- Baseline chung la TransReID ViT-Base voi `STRIDE_SIZE [12, 12]`, `JPM True`, `SIE_CAMERA True`, input `256x128`.
- Semantic alignment doc mask chi trong train loader. Evaluation khong can mask, nhung khi test voi config semantic van nen giu override `FLIP_LABEL_PAIRS` nhat quan voi luc train.
- Reliability pipeline tao anh bi che online bang cac tham so `INPUT.REL_OCC_*`, khong can file mask rieng.
- Neu GPU thieu VRAM, giam `SOLVER.IMS_PER_BATCH`, vi du `SOLVER.IMS_PER_BATCH 32`. Co the giam `TEST.IMS_PER_BATCH` khi evaluate.
- Neu chay tren Windows va gap loi multiprocessing DataLoader, thu override `DATALOADER.NUM_WORKERS 0`.

## 9. Loi thuong gap

`RuntimeError: '<path>/market1501' is not available`

Kiem tra lai `DATASETS.ROOT_DIR`. Gia tri nay phai tro den thu muc cha cua `market1501`, khong phai vao thang `market1501`.

`FileNotFoundError` voi semantic mask

Kiem tra mask co cung relative path va doi extension sang `.png`, vi du anh trong `bounding_box_train` thi mask phai nam trong `semantic_groups/bounding_box_train`.

Ket qua semantic alignment bat thuong sau khi horizontal flip

Kiem tra preset mask. Neu dung `fashn6`, train voi `MODEL.SEM_ALIGN.FLIP_LABEL_PAIRS []`. Neu dung `lip6`, giu cap swap mac dinh `[[3, 4], [5, 6]]`.

CUDA out of memory

Giam batch size:

```bash
SOLVER.IMS_PER_BATCH 32 TEST.IMS_PER_BATCH 128
```

them cac override nay vao cuoi lenh train/test.
