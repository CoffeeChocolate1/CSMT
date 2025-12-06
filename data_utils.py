
import os
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split

# ---------- 数据集根目录 ----------
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'HSI')   # 原来 '..', 'HSI'
import os
import scipy.io as sio
import numpy as np

# 统一根目录：与 data_utils.py 同目录下的 HSI 文件夹
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'HSI')

def LoadHSIData(method):
    """
    返回：HSI 立方体, GT 标签图, 类别数, 类别名称列表
    method ∈ {'PU', 'IP', 'SA', 'HO'}
    """
    if method == 'PU':          # PaviaU
        sub = 'PU'
        HSI = sio.loadmat(os.path.join(DATA_ROOT, sub,'PaviaU.mat'))['paviaU']
        GT  = sio.loadmat(os.path.join(DATA_ROOT, sub,'PaviaU_gt.mat'))['paviaU_gt']
        Num_Classes = 9
        target_names = ['Asphalt','Meadows','Gravel','Trees','Painted',
                        'Soil','Bitumen','Bricks','Shadows']

    elif method == 'IP':        # Indian Pines
        sub = 'IP'
        HSI = sio.loadmat(os.path.join(DATA_ROOT, sub,'Indian_pines.mat'))['indian_pines']
        GT  = sio.loadmat(os.path.join(DATA_ROOT, sub, 'Indian_pines_gt.mat'))['indian_pines_gt']
        Num_Classes = 16
        target_names = ['Alfalfa','Corn-notill','Corn-mintill','Corn',
                        'Grass-pasture','Grass-trees','Grass-pasture-mowed',
                        'Hay-windrowed','Oats','Soybean-notill','Soybean-mintill',
                        'Soybean-clean','Wheat','Woods','Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']

    elif method == 'SA':        # Salinas Valley
        sub = 'SA'  # ← 新增
        HSI = sio.loadmat(os.path.join(DATA_ROOT, sub, 'Salinas.mat'))['salinas']
        GT = sio.loadmat(os.path.join(DATA_ROOT, sub, 'Salinas_gt.mat'))['salinas_gt']
        Num_Classes = 16
        target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow',
                        'Fallow_rough_plow','Fallow_smooth','Stubble','Celery',
                        'Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk',
                        'Lettuce_romaine_7wk','Vinyard_untrained','Vinyard_vertical_trellis']

    elif method == 'HO':        # Houston 2013
        sub = 'HO'
        HSI = sio.loadmat(os.path.join(DATA_ROOT,sub, 'Houston2013.mat'))['houston2013']
        GT  = sio.loadmat(os.path.join(DATA_ROOT,sub, 'Houston2013_gt.mat'))['houston2013_gt']
        Num_Classes = 15
        target_names = ['Grass_healthy','Grass_stressed','Grass_synthetic','Tree',
                        'Soil','Water','Residential','Commercial','Road','Highway',
                        'Railway','Parking_lot1','Parking_lot2','Tennis_court','Running_track']

    else:
        raise ValueError(f'Unsupported dataset: {method}')

    return HSI, GT, Num_Classes, target_names

# ---------- 降维 ----------
def DLMethod(method, HSI, NC=15):
    RHSI = np.reshape(HSI, (-1, HSI.shape[2]))
    if method == 'PCA':
        pca = PCA(n_components=NC, whiten=True)
        RHSI = pca.fit_transform(RHSI)
    elif method == 'iPCA':
        inc_pca = IncrementalPCA(n_components=NC)
        for X_batch in np.array_split(RHSI, 256):
            inc_pca.partial_fit(X_batch)
        RHSI = inc_pca.transform(RHSI)
    else:
        raise ValueError('unknown DLMethod')
    return np.reshape(RHSI, (HSI.shape[0], HSI.shape[1], NC))

def ImageCubes(HSI, GT, WS=12, removeZeroLabels=True):
    h, w, b = HSI.shape
    margin = WS // 2
    pad = np.pad(HSI, ((margin, margin), (margin, margin), (0, 0)), 'constant')
    cubes = np.zeros((h * w, WS, WS, b))      # 固定 WS×WS
    labels = np.zeros(h * w)
    idx = 0
    for r in range(margin, h + margin):
        for c in range(margin, w + margin):
            # 🔑 关键：用 WS 而不是 WS+1
            cube = pad[r - margin:r - margin + WS, c - margin:c - margin + WS, :]
            cubes[idx] = cube
            labels[idx] = GT[r - margin, c - margin]
            idx += 1
    if removeZeroLabels:
        cubes  = cubes[labels > 0]
        labels = labels[labels > 0] - 1
    return cubes, labels.astype(int)

# ---------- 训练/验证/测试划分 ----------
def TrTeSplit(HSI, GT, trRatio, vrRatio, teRatio, randomState=345):
    Tr, Te, TrC, TeC = train_test_split(HSI, GT, test_size=teRatio,
                                        random_state=randomState, stratify=GT)
    totalTrRatio = trRatio + vrRatio
    Tr, Va, TrC, VaC = train_test_split(Tr, TrC, test_size=vrRatio/totalTrRatio,
                                        random_state=randomState, stratify=TrC)
    return Tr, Va, Te, TrC, VaC, TeC


def ImageCubes_by_coords(HSI, GT, coords, WS=12):
    """
    根据像素坐标 coords 生成模型输入的 WS×WS patch
    coords: N×2 的 (row, col)
    """
    margin = WS // 2
    H, W, B = HSI.shape

    # pad
    pad = np.pad(HSI, ((margin, margin), (margin, margin), (0, 0)), mode="constant")

    cubes = []
    labels = []

    for (r, c) in coords:
        r2 = r + margin
        c2 = c + margin
        patch = pad[r2-margin:r2+margin, c2-margin:c2+margin, :]
        cubes.append(patch)
        labels.append(GT[r, c]-1)  # 转为 0-based

    return np.array(cubes), np.array(labels)





import numpy as np
from sklearn.model_selection import KFold

def patch_kfold_split(gt_map, patch_h=22, patch_w=10, K=5):
    """
    gt_map: H×W 的整幅地物标签矩阵
    返回 folds: [
        (train_pixel_coords, test_pixel_coords),
        ...
    ]
    """

    H, W = gt_map.shape

    # -------- 1. 收集所有 patch 左上角 --------
    patch_coords = []
    for r in range(0, H - patch_h + 1, patch_h):
        for c in range(0, W - patch_w + 1, patch_w):
            patch_coords.append((r, c))

    patch_coords = np.array(patch_coords)

    # -------- 2. 打乱 patch 顺序 --------
    np.random.shuffle(patch_coords)

    # -------- 3. K 折划分 --------
    kf = KFold(n_splits=K, shuffle=False)
    folds = []

    for fold_id, (train_patch_idx, test_patch_idx) in enumerate(kf.split(patch_coords)):

        train_pixels = []
        test_pixels = []

        # --- 训练 patch ---
        for idx in train_patch_idx:
            r, c = patch_coords[idx]
            patch = gt_map[r:r+patch_h, c:c+patch_w]
            coords = np.argwhere(patch > 0) + np.array([r, c])    # 仅使用非背景像素
            train_pixels.append(coords)

        # --- 测试 patch ---
        for idx in test_patch_idx:
            r, c = patch_coords[idx]
            patch = gt_map[r:r+patch_h, c:c+patch_w]
            coords = np.argwhere(patch > 0) + np.array([r, c])
            test_pixels.append(coords)

        # 合并
        train_pixels = np.vstack(train_pixels)
        test_pixels  = np.vstack(test_pixels)

        print(f"[Fold {fold_id+1}] train_pixels={len(train_pixels)}, test_pixels={len(test_pixels)}")

        folds.append((train_pixels, test_pixels))

    return folds












