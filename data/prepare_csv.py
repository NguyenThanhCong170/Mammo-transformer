import pandas as pd
import os
from pathlib import Path

from configs.config import Config
cfg = Config()


def prepare_vindr_csv(
    breast_annotations_path: str,   # breast_level_annotations.csv
    images_dir: str,                 # Thư mục chứa ảnh đã crop (.png)
    output_csv_path: str,
    image_ext: str,
    label_mapping
):
    """
    breast_level_annotations.csv columns:
        study_id, laterality, view_position,
        breast_birads, breast_density
    """
    df = pd.read_csv(breast_annotations_path)
    print(f"Raw annotations: {len(df)} rows, {df['study_id'].nunique()} studies")
    df = df[df["study_id"] != "dbca9d28baa3207b3187c4d07dc81a80"]
    # Rename study_id → patient_id cho nhất quán
    df = df.rename(columns={"study_id": "patient_id"})

    # Chuẩn hóa laterality và view_position
    df["laterality"]     = df["laterality"].str.strip().str.upper()   # L / R
    df["view_position"]  = df["view_position"].str.strip().str.upper() # MLO / CC

    # Tạo đường dẫn ảnh
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
    # (dùng nunique trên cột ghép — tránh DeprecationWarning của groupby.apply ở pandas 2.2+)
    df["_view_key"] = df["laterality"] + "_" + df["view_position"]
    view_counts = df.groupby("patient_id")["_view_key"].nunique()
    df = df.drop(columns=["_view_key"])
    complete = view_counts[view_counts == 4].index
    print(f"Patients với đủ 4 views: {len(complete)} / {df['patient_id'].nunique()}")

    df_complete = df[df["patient_id"].isin(complete)].copy()

    # Parse BI-RADS
    df_complete["target"] = df_complete["finding_categories"]


    # Label
    def text_to_label_id(category_text):
        """
        Hàm chuyển đổi category string thành integer label.
        """
        labels = []
        if "Asymmetry" in category_text or "Global Asymmetry" in category_text or "Focal Asymmetry" in category_text:
            labels.append(label_mapping["Asymmetry"])
        
        if "Mass" in category_text:
            labels.append(label_mapping["Mass"])
            
        if "Suspicious Calcification" in category_text:
            labels.append(label_mapping["Suspicious Calcification"])
            
        if len(labels) == 0:
            labels.append(label_mapping["no finding"])
        return labels
    
    df_complete["target"] = df_complete["finding_categories"].apply(text_to_label_id)
        

    # Class distribution
    print("Thống kê số lượng mỗi class:")
    print(df_complete["target"].value_counts())

    # Save
    output_cols = [
        "patient_id", "image_id",
        "image_path", 'laterality', "view_position", "target", "split"  # ← giữ cột split gốc
    ]
    df_complete[output_cols].to_csv(output_csv_path, index=False)
    print(f"\n✅ Saved: {output_csv_path} ({len(df_complete)} rows)")

    return df_complete


if __name__ == "__main__":
    # Input = file annotation GỐC của VinDr; output = labels.csv mà dataset.py đọc.
    prepare_vindr_csv(
        breast_annotations_path=cfg.data.raw_annotations_csv,
        images_dir=cfg.data.data_root,
        output_csv_path=cfg.data.csv_path,
        image_ext=cfg.data.image_ext,
        label_mapping=cfg.data.label_mapping,
    )
