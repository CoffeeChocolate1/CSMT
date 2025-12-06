
import time, os, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix
from operator import truediv
from dl_model import CSMT
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score  # 新增
from datetime import datetime

import numpy as np
from tensorflow.keras.layers import Conv3D, Dense


# 全局时间戳（在程序启动时生成一次）
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ---------- 评估 ----------
def ClassificationReports(TeC, Te_Pred, target_names):
    y_true = np.argmax(TeC, axis=1)
    y_pred = np.argmax(Te_Pred, axis=1)


    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,      # ✅ 修复 UndefinedMetricWarning
        output_dict=False     # ✅ 确保返回字符串
    )

    oa = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    each = np.nan_to_num(np.diag(cm) / np.sum(cm, axis=1))
    aa = np.mean(each)
    kappa = cohen_kappa_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weight_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return report, cm, oa * 100, each * 100, aa * 100, kappa * 100, macro_f1 * 100, weight_f1 * 100


# ---------- 写 CSV ----------
import csv

def CSVResults(file_name, classification, Confusion, Parameters,
               Flops, Tr_time, Te_Time, Kappa, OA, AA, Per_Class,
               Macro_F1, Weighted_F1):
    try:
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Training Time (s)', f'{Tr_time:.2f}'])
            writer.writerow(['Inference Time (s)', f'{Te_Time:.2f}'])
            writer.writerow(['FLOPs', Flops])
            writer.writerow(['Parameters', Parameters])
            writer.writerow(['Kappa (%)', f'{Kappa:.2f}'])
            writer.writerow(['Overall Accuracy (%)', f'{OA:.2f}'])
            writer.writerow(['Average Accuracy (%)', f'{AA:.2f}'])
            writer.writerow(['Macro-F1 (%)', f'{Macro_F1:.2f}'])
            writer.writerow(['Weighted-F1 (%)', f'{Weighted_F1:.2f}'])
            writer.writerow(['Per-Class Accuracy', Per_Class.tolist()])
            writer.writerow([])

            writer.writerow(['Classification Report'])
            f.write(str(classification) + '\n')   # ✅ 防止类型问题
            writer.writerow([])

            writer.writerow(['Confusion Matrix'])
            np.savetxt(f, Confusion, fmt='%d', delimiter=',')  # ✅ 明确二维矩阵写入

        print(f"[INFO] 已成功保存 CSV 文件: {file_name}")

    except Exception as e:
        print(f"[ERROR] 写入 CSV 文件失败: {e}")




def GT_Plot(HSI, GT, model, WS, k):


    h, w, c = HSI.shape
    margin = WS // 2

    pad = np.pad(HSI, ((margin, margin), (margin, margin), (0, 0)), 'constant')

    patches = []
    coords = []

    for r in range(margin, h + margin):
        for c0 in range(margin, w + margin):
            cube = pad[r - margin:r - margin + WS, c0 - margin:c0 - margin + WS, :]
            patches.append(cube)
            coords.append((r - margin, c0 - margin))

    patches = np.array(patches)    # (N, WS, WS, C)

    probs = model.predict(patches, batch_size=256, verbose=0)
    pred_labels = np.argmax(probs, axis=1)

    Pred = np.zeros((h, w), dtype=np.int32)

    for (r, c0), label in zip(coords, pred_labels):
        Pred[r, c0] = label + 1   # +1 因为标签在图中是从1开始

    return Pred





def train_and_evaluate(WS, k, Num_Classes, target_names,
                       CRDHSI, CGT, Tr, TrC, Va, VaC, Te, TeC,
                       epochs, batch_size, adam_lr, adam_decay, HSID, trRatio, GT):
    import tensorflow as tf
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from dl_model import CSMT, SST
    from tensorflow.keras import Model

    # ---------- 1. 构建模型 ----------
    model = CSMT(WS, k, Num_Classes)
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=adam_lr, decay=adam_decay)

    # ---------- 2. 取最后一层注意力 ----------
    attn_layers = [l for l in model.layers if isinstance(l, SST)]
    if not attn_layers:
        raise ValueError("未找到 Transformer 层 (SST)")
    attn_output_model = Model(inputs=model.input, outputs=attn_layers[-1].output[1])

    # ---------- 3. 自定义 loss（整合 focus loss + L2 正则） ----------
    def custom_loss(y_true, y_pred, lambda_l2=1e-4):


        # ---------- 分类交叉熵 ----------
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # ---------- focus loss ----------
        attn_tensor = attn_output_model(Tr[:batch_size], training=False)
        attn_mean = tf.reduce_mean(attn_tensor, axis=1)[:, 0, :]  # (B, L)

        attn_len = tf.shape(attn_mean)[1]

        mask_resized = tf.image.resize(
            tf.cast(GT[tf.newaxis, ..., tf.newaxis], tf.float32),
            [1, attn_len - 1],
            method='nearest'
        )[0, 0, :]

        mask_resized = tf.squeeze(mask_resized, axis=-1)
        cls_mask = tf.ones((1,), dtype=tf.float32)
        mask_aligned = tf.concat([cls_mask, mask_resized], axis=0)
        mask_aligned = tf.expand_dims(mask_aligned, axis=0)

        focus_loss = tf.reduce_mean(attn_mean * (1 - mask_aligned))

        # ---------- L2 正则项（模拟 AdamW 权重衰减） ----------
        l2_loss = tf.add_n([
            tf.nn.l2_loss(v)
            for v in model.trainable_variables
            if 'bias' not in v.name.lower()  # 不对 bias 做 L2
               and 'norm' not in v.name.lower()  # 不对 LayerNorm 做 L2
        ])


        return ce + 0.1 * focus_loss + lambda_l2 * l2_loss


    model.compile(optimizer=adam, loss=custom_loss, metrics=['accuracy'])



    t0 = time.time()
    hist = model.fit(Tr, TrC, batch_size=batch_size, epochs=epochs, validation_data=(Va, VaC))
    tr_time = time.time() - t0


    flops, macs, params = get_flops(model,
                                    input_shape=(Tr.shape[1], Tr.shape[2], Tr.shape[3]),
                                    num_classes=Num_Classes,
                                    dataset_name='pavia')


    def human_fmt(num):
        if num >= 1e9:
            return f"{num / 1e9:.2f} G"
        if num >= 1e6:
            return f"{num / 1e6:.2f} M"
        if num >= 1e3:
            return f"{num / 1e3:.2f} K"
        return str(int(num))

    params_fmt = human_fmt(params)
    flops_fmt = human_fmt(flops)


    print(f"Parameters : {params_fmt}")
    print(f"FLOPs      : {flops_fmt}")
    print("=============================\n")


    # params_M = params / 1e6
    # flops_M = flops / 1e6


    t0 = time.time()
    Te_Pre = model.predict(Te)
    te_time = time.time() - t0


    report, cm, oa, each, aa, kappa, macro_f1, weight_f1 = ClassificationReports(TeC, Te_Pre, target_names)




    print(report)
    print(f"\n===== [模型训练结果汇总] =====")
    print(f'Macro-F1   : {macro_f1:.2f}%')
    print(f'Weighted-F1: {weight_f1:.2f}%')
    print(f'OA         : {oa:.2f}%')
    print(f'AA         : {aa:.2f}%')
    print(f'Kappa      : {kappa:.2f}%')
    print(f'Training Time  : {tr_time:.2f}s')
    print(f'Inference Time : {te_time:.2f}s')
    print(f"=============================\n")


    # ---------- 6. 保存分类结果 ----------
    # ---------- 创建独立结果文件夹 ----------
    from datetime import datetime
    import os

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results", timestamp)  # 所有结果都保存在 results 目录下
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] 所有输出文件将保存到: {save_dir}")

    csv_name = os.path.join(save_dir, f'4_1_{HSID}_{trRatio}_{WS}_Classification_Report_{RUN_TIMESTAMP}.csv')
    CSVResults(
        csv_name,
        report,
        cm,
        params_fmt,
        flops_fmt,
        tr_time,
        te_time,
        kappa,
        oa,
        aa,
        each,
        macro_f1,
        weight_f1
    )

    # ---------- 7. Ground Truth 可视化 ----------
    map_name = os.path.join(save_dir, f'4_1_{HSID}_{trRatio}_{WS}_Ground_Truths_{RUN_TIMESTAMP}.png')
    outputs = GT_Plot(CRDHSI, GT, model, WS, k)

    # ---------- 8. 注意力热图 ----------
    attn_map_name = os.path.join(save_dir, f'4_1_{HSID}_{trRatio}_{WS}_attn_map_{RUN_TIMESTAMP}.png')
    AttentionHeatMap(model, CRDHSI, CGT, WS, k,
                     sample_idx=0,
                     save_path=attn_map_name)

    # ---------- 9. 保存伪彩图 ----------
    plt.figure(figsize=(8, 8))
    plt.imshow(outputs, cmap='nipy_spectral')
    plt.axis('off')
    plt.savefig(map_name, dpi=500, bbox_inches='tight', pad_inches=0)
    print(f"[INFO] 所有结果已保存，时间戳: {RUN_TIMESTAMP}")
    print('========== 完成，结果已保存 ==========')


    return {
        "OA": oa,
        "AA": aa,
        "Kappa": kappa,
        "Macro_F1": macro_f1,
        "Weighted_F1": weight_f1
    }



def AttentionHeatMap(model, HSI, GT, WS, k, sample_idx=0, save_path=None):
    """
    修复版本：
    保留原功能（多头注意力 + Overlay），但不再使用 CRDHSI。
    改为从原始 HSI（H×W×C）中重新裁 patch。
    """

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from tensorflow.keras import Model
    from dl_model import SST

    print("[INFO] 提取 Transformer 注意力热图...")
    h, w, c = HSI.shape
    margin = WS // 2

      coords = np.argwhere(GT > 0)
    if len(coords) == 0:
        print("[WARN] GT 中无标注像素，无法生成注意力图。")
        return

    if sample_idx >= len(coords):
        sample_idx = 0

    r, c0 = coords[sample_idx]

    # pad HSI
    pad = np.pad(HSI, ((margin, margin), (margin, margin), (0, 0)), mode='constant')

    # 提取 WS×WS patch
    cube = pad[r:r + WS, c0:c0 + WS, :]  # shape = (WS, WS, C)
    cube = np.expand_dims(cube, axis=0)  # shape = (1, WS, WS, C)


    attn_layers = [l for l in model.layers if isinstance(l, SST)]
    if not attn_layers:
        print("[WARN] 模型中未找到 Transformer 层 (SST)")
        return

    attn_output_model = Model(inputs=model.input, outputs=attn_layers[-1].output[1])
    attn_tensor = attn_output_model(cube, training=False)


    attn_out = attn_tensor[0].numpy()
    if attn_out.ndim == 3:
        # (H, L, L)
        attn_mean = attn_out
        num_heads = attn_out.shape[0]
    elif attn_out.ndim == 2:
        attn_mean = attn_out[np.newaxis, ...]
        num_heads = 1
    else:
        raise ValueError(f"注意力张量维度异常: {attn_out.shape}")

    print(f"[INFO] 注意力张量形状: {attn_mean.shape}")

        fig, axs = plt.subplots(1, num_heads, figsize=(3*num_heads, 3))
    if num_heads == 1:
        axs = [axs]

    for i in range(num_heads):
        axs[i].imshow(attn_mean[i], cmap='viridis')
        axs[i].set_title(f'Head {i+1}')
        axs[i].axis('off')

    if save_path:
        save_path_heads = save_path.replace('.png', f'_{RUN_TIMESTAMP}.png')
        plt.savefig(save_path_heads, dpi=300, bbox_inches='tight')
        print(f"[INFO] 注意力热图已保存: {save_path_heads}")
    else:
        plt.show()

   

    print("[INFO] 正在生成空间注意力叠加图...")

    cls_attn = attn_mean[0, 0, 1:]      # cls 对所有 patch 的注意力
    num_tokens = cls_attn.shape[0]

    patch_size = int(np.sqrt(num_tokens))
    cls_attn_map = cls_attn.reshape((patch_size, patch_size))


    cls_attn_up = cv2.resize(cls_attn_map, (WS, WS), interpolation=cv2.INTER_CUBIC)
    cls_attn_up = (cls_attn_up - cls_attn_up.min()) / (cls_attn_up.max() - cls_attn_up.min() + 1e-8)


    img_rgb = np.repeat(cube[0, :, :, 0:1], 3, axis=-1)

    overlay = cv2.applyColorMap(np.uint8(255 * cls_attn_up), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.uint8(255 * img_rgb / img_rgb.max()), 0.6, overlay, 0.6, 0)

    plt.figure(figsize=(4, 4))
    plt.imshow(overlay[..., ::-1])   # BGR → RGB
    plt.title("Transformer Attention Overlay")
    plt.axis('off')

    if save_path:
        overlay_path = save_path.replace('.png', '_overlay.png')
        plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] 空间注意力叠加图已保存: {overlay_path}")
    else:
        plt.show()
