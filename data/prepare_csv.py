"""
Script chuẩn bị CSV từ VinDr-Mammo raw annotations.

VinDr-Mammo structure (sau khi download):
  physionet.org/files/vindr-mammo/1.0.0/
  ├── finding_annotations.csv
  ├── breast_level_annotations.csv
  └── images/
      └── {study_id}/
          └── {image_id}.dicom  
"""

import pandas as pd
import os
from pathlib import Path


def prepare_vindr_csv(
    breast_annotations_path: str,   # breast_level_annotations.csv
    images_dir: str,                 # Thư mục chứa ảnh đã crop (.png)
    output_csv_path: str,
    image_ext: str = ".dicom",
):
    """
    breast_level_annotations.csv columns:
        study_id, laterality, view_position,
        breast_birads, breast_density
    """
    df = pd.read_csv(breast_annotations_path)
    print(f"Raw annotations: {len(df)} rows, {df['study_id'].nunique()} studies")

    # Rename study_id → patient_id cho nhất quán
    df = df.rename(columns={"study_id": "patient_id"})

    # Chuẩn hóa laterality và view_position
    df["laterality"]     = df["laterality"].str.strip().str.upper()   # L / R
    df["view_position"]  = df["view_position"].str.strip().str.upper() # MLO / CC

    # Tạo đường dẫn ảnh
    # Giả sử bạn đã lưu ảnh theo: {images_dir}/{patient_id}/{image_id}.png
    def build_path(row):
        path = Path(images_dir) / row["patient_id"] / f"{row['image_id']}{image_ext}"
        return str(path.as_posix())

    df["image_path"] = df.apply(build_path, axis=1)

    # Kiểm tra file tồn tại
    df["file_exists"] = df["image_path"].apply(os.path.exists)
    missing = df[~df["file_exists"]]
    if len(missing) > 0:
        print(f"⚠️  WARNING: {len(missing)} ảnh không tìm thấy!")
        print(missing[["patient_id", "image_id", "image_path"]].head(5))

    df = df[df["file_exists"]].drop(columns=["file_exists"])

    # Đảm bảo đủ 4 views mỗi bệnh nhân
    view_counts = df.groupby("patient_id").apply(
        lambda g: len(set(zip(g["laterality"], g["view_position"])))
    )
    complete = view_counts[view_counts == 4].index
    print(f"Patients với đủ 4 views: {len(complete)} / {df['patient_id'].nunique()}")

    df_complete = df[df["patient_id"].isin(complete)].copy()

    # Parse BI-RADS
    df_complete["birads_int"] = df_complete["breast_birads"].apply(
        lambda x: int(str(x).replace("BI-RADS", "").strip())
    )

    # Label
    df_complete["label"] = df_complete["birads_int"].apply(
        lambda x: 1 if x in [4, 5] else 0
    )

    # Class distribution
    patient_labels = df_complete.groupby("patient_id")["label"].max()
    pos = patient_labels.sum()
    neg = len(patient_labels) - pos
    print(f"\nClass distribution (patient-level):")
    print(f"  Positive (BI-RADS 4,5): {pos} ({pos/len(patient_labels)*100:.1f}%)")
    print(f"  Negative (BI-RADS 1-3): {neg} ({neg/len(patient_labels)*100:.1f}%)")

    # Save
    output_cols = [
        "patient_id", "image_id", "laterality", "view_position",
        "image_path", "breast_birads", "birads_int", "label", "split"  # ← giữ cột split gốc
    ]
    df_complete[output_cols].to_csv(output_csv_path, index=False)
    print(f"\n✅ Saved: {output_csv_path} ({len(df_complete)} rows)")

    return df_complete


if __name__ == "__main__":
    prepare_vindr_csv(
        breast_annotations_path="finding_annotations.csv",
        images_dir="images_cropped",
        output_csv_path="labels.csv",
        image_ext=".dicom",
    )
