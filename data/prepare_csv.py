import pandas as pd
import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from configs.config import Config, resolve_path
cfg = Config()


def prepare_vindr_csv(
    breast_annotations_path: str,   # breast_level_annotations.csv
    images_dir: str,             
    output_csv_path: str,
    image_ext: str
):
    """
    breast_level_annotations.csv columns:
        study_id, laterality, view_position,
        breast_birads, breast_density
    """
    ann_path = resolve_path(breast_annotations_path)
    images_root = resolve_path(images_dir)

    print(f"Annotations : {ann_path}")
    print(f"Images dir  : {images_root}")

    df = pd.read_csv(ann_path)
    print(f"Raw annotations: {len(df)} rows, {df['study_id'].nunique()} studies")
    df = df[df["study_id"] != "dbca9d28baa3207b3187c4d07dc81a80"]

    # Rename study_id → patient_id 
    df = df.rename(columns={"study_id": "patient_id"})

    # Uppercase 
    df["laterality"]     = df["laterality"].str.strip().str.upper()   # L / R
    df["view_position"]  = df["view_position"].str.strip().str.upper() # MLO / CC
    df['finding_birads'] = df['finding_birads'].str.strip().str[-1]

    # Tạo đường dẫn ảnh — LƯU vào CSV dạng tương đối (gọn, portable),
    # nhưng KIỂM TRA tồn tại bằng đường dẫn tuyệt đối.
    rel_root = Path(images_dir).as_posix().rstrip("/")

    df["image_path"] = (rel_root + "/" + df["patient_id"].astype(str)
                        + "/" + df["image_id"].astype(str) + image_ext)
    df["file_exists"] = [(images_root / f"{p}/{i}{image_ext}").exists()
                         for p, i in zip(df["patient_id"], df["image_id"])]

    n_found = int(df["file_exists"].sum())
    print(f"Tìm thấy: {n_found}/{len(df)} ảnh")

    if n_found < len(df):
        missing = df[~df["file_exists"]]
        print(f"⚠️  {len(missing)} ảnh không tìm thấy (bỏ qua). 5 ví dụ:")
        print(missing[["patient_id", "image_id", "image_path"]].head(5).to_string(index=False))

    df = df[df["file_exists"]].drop(columns=["file_exists"])

    # Đảm bảo đủ 4 views mỗi bệnh nhân
    df["_view_key"] = df["laterality"] + "_" + df["view_position"]
    view_counts = df.groupby("patient_id")["_view_key"].nunique()
    df = df.drop(columns=["_view_key"])
    complete = view_counts[view_counts == 4].index
    print(f"Patients với đủ 4 views: {len(complete)} / {df['patient_id'].nunique()}")

    df_complete = df[df["patient_id"].isin(complete)].copy()
    

    # Save
    output_cols = [
        "patient_id", "image_id",
        "image_path", 'laterality', "view_position","finding_birads", "split"  # ← giữ cột split gốc
    ]
    out_path = resolve_path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_complete[output_cols].to_csv(out_path, index=False)
    print(f"\n✅ Saved: {out_path} ({len(df_complete)} rows)")
    print(f"\n   Bước tiếp theo:")
    print(f"     python data/resize_images.py --src {images_dir} --csv {output_csv_path}")

    return df_complete


if __name__ == "__main__":
    prepare_vindr_csv(
        breast_annotations_path=cfg.data.raw_annotations_csv,
        images_dir=cfg.data.raw_images_dir,
        output_csv_path=cfg.data.csv_raw,
        image_ext=cfg.data.image_ext,
    )
