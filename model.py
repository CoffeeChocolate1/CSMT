
import numpy as np
from tensorflow.keras.utils import to_categorical
from config import *
from data_utils import LoadHSIData, DLMethod, ImageCubes, TrTeSplit
from train_utils import train_and_evaluate

def main():

    HSI, GT, Num_Classes, target_names = LoadHSIData(HSID)
    RDHSI = DLMethod(DLM, HSI, NC=k)
    from data_utils import patch_kfold_split, ImageCubes_by_coords
    folds = patch_kfold_split(GT, patch_h=22, patch_w=10, K=K)
    fold_results = []
    for fold_id, (train_coords, test_coords) in enumerate(folds):
        print(f"\n=========== Patch Fold {fold_id+1}/{K} ===========")
        Tr, TrC = ImageCubes_by_coords(RDHSI, GT, train_coords, WS)
        TrC = to_categorical(TrC)
        Te, TeC = ImageCubes_by_coords(RDHSI, GT, test_coords, WS)
        TeC = to_categorical(TeC)
        Va, VaC = Tr[:200], TrC[:200]     # 如果模型要求有验证集，你这样做
        result = train_and_evaluate(
            WS, k, Num_Classes, target_names,
            RDHSI, GT,
            Tr, TrC, Va, VaC, Te, TeC,
            epochs, batch_size, adam_lr, adam_decay,
            HSID, trRatio, GT
        )

        fold_results.append(result)




        metrics = ["OA", "AA", "Kappa", "Macro_F1", "Weighted_F1"]

        for i, r in enumerate(fold_results):
            print(f"Fold {i + 1}: { {k: f'{v:.2f}%' for k, v in r.items()} }")


        for m in metrics:
            values = [r[m] for r in fold_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{m}: {mean_val:.2f} ± {std_val:.2f}")




if __name__ == '__main__':
    main()