import pandas as pd
import os
import sys
from pathlib import Path

from configs.config import Config, resolve_path
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
    # Đường dẫn tương đối tính từ PROJECT_ROOT, không phụ thuộc cwd
    # → `python -m data.prepare_csv` chạy được từ bất kỳ đâu.
    ann_path = resolve_path(breast_annotations_path)
    images_root = resolve_path(images_dir)

    print(f"Annotations : {ann_path}")
    print(f"Images dir  : {images_root}")
    if not ann_path.exists():
        sys.exit(f"\n❌ Không thấy file annotation: {ann_path}")
    if not images_root.is_dir():
        sys.exit(f"\n❌ Không thấy thư mục ảnh: {images_root}\n"
                 f"   Sửa cfg.data.raw_images_dir cho đúng, hoặc truyền images_dir=...")

    df = pd.read_csv(ann_path)
    print(f"Raw annotations: {len(df)} rows, {df['study_id'].nunique()} studies")
    df = df[df["study_id"] != "dbca9d28baa3207b3187c4d07dc81a80"]
    # Rename study_id → patient_id cho nhất quán
    df = df.rename(columns={"study_id": "patient_id"})

    # Chuẩn hóa laterality và view_position
    df["laterality"]     = df["laterality"].str.strip().str.upper()   # L / R
    df["view_position"]  = df["view_position"].str.strip().str.upper() # MLO / CC

    # Tạo đường dẫn ảnh — LƯU vào CSV dạng tương đối (gọn, portable),
    # nhưng KIỂM TRA tồn tại bằng đường dẫn tuyệt đối.
    rel_root = Path(images_dir).as_posix().rstrip("/")

    df["image_path"] = (rel_root + "/" + df["patient_id"].astype(str)
                        + "/" + df["image_id"].astype(str) + image_ext)
    df["file_exists"] = [(images_root / f"{p}/{i}{image_ext}").exists()
                         for p, i in zip(df["patient_id"], df["image_id"])]

    n_found = int(df["file_exists"].sum())
    print(f"Tìm thấy: {n_found}/{len(df)} ảnh")

    if n_found == 0:
        # Chẩn đoán: so đường dẫn MONG ĐỢI với file THỰC SỰ có trên đĩa
        on_disk = list(images_root.rglob(f"*{image_ext}"))[:3]
        r = df.iloc[0]
        print("\n" + "=" * 62)
        print("❌ KHÔNG khớp được ảnh nào. So sánh:")
        print(f"\n  Mong đợi : {images_root / str(r['patient_id']) / (str(r['image_id']) + image_ext)}")
        if on_disk:
            print(f"  Trên đĩa : {on_disk[0]}")
            print(f"             (tổng {len(list(images_root.rglob('*' + image_ext)))} file {image_ext})")
        else:
            print(f"  Trên đĩa : KHÔNG có file {image_ext} nào trong {images_root}")
        print("\n  Nguyên nhân hay gặp:")
        print("    • raw_images_dir trỏ nhầm sang thư mục ĐÃ RESIZE (data_root)")
        print("    • ảnh nằm phẳng, không có thư mục con theo study_id")
        print("    • phần mở rộng khác (.jpg/.jpeg) → sửa cfg.data.image_ext")
        print("=" * 62)
        sys.exit(1)

    if n_found < len(df):
        missing = df[~df["file_exists"]]
        print(f"⚠️  {len(missing)} ảnh không tìm thấy (bỏ qua). 5 ví dụ:")
        print(missing[["patient_id", "image_id", "image_path"]].head(5).to_string(index=False))

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
    out_path = resolve_path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_complete[output_cols].to_csv(out_path, index=False)
    print(f"\n✅ Saved: {out_path} ({len(df_complete)} rows)")
    print(f"\n   Bước tiếp theo:")
    print(f"     python data/resize_images.py --src {images_dir} --csv {output_csv_path}")

    return df_complete


if __name__ == "__main__":
    # Quét raw_images_dir (ảnh GỐC sau crop), KHÔNG phải data_root (ảnh đã resize).
    # Output = csv_raw; sau đó chạy data/resize_images.py để sinh csv_path.
    prepare_vindr_csv(
        breast_annotations_path=cfg.data.raw_annotations_csv,
        images_dir=cfg.data.raw_images_dir,
        output_csv_path=cfg.data.csv_raw,
        image_ext=cfg.data.image_ext,
        label_mapping=cfg.data.label_mapping,
    )
