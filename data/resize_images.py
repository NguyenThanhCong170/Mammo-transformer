"""
Resize toàn bộ ảnh PNG sang kích thước mới, giữ nguyên cấu trúc thư mục.

    # Xem trước, không ghi gì
    python data/resize_images.py --dry-run

    # Chạy thật (mặc định: images_cropped → images_928x352, đồng thời sinh CSV mới)
    python data/resize_images.py

    # Dataset đầy đủ nằm chỗ khác
    python data/resize_images.py --src D:/data/images_cropped --dst D:/data/images_928x352 \
                                 --csv labels.csv --workers 12

Đặc điểm:
  - Resume được: file đã tồn tại đúng size ở đích sẽ bị bỏ qua (--force để ép làm lại).
  - Đa tiến trình (mặc định = số CPU core).
  - Chuyển RGB → L (ảnh mammo là ảnh xám, lưu 3 kênh phí 3x dung lượng).
  - Sinh CSV mới với image_path đã trỏ sang thư mục đích.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

RESAMPLE = {
    "lanczos": Image.LANCZOS,   # sắc nét nhất, giữ đốm sáng nhỏ tốt hơn (mặc định)
    "box": Image.BOX,           # trung bình vùng, không ringing, mượt hơn
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
}


def resize_one(args):
    src, dst, width, height, resample, to_gray, force = args
    try:
        dst = Path(dst)
        if dst.exists() and not force:
            try:
                with Image.open(dst) as im:
                    if im.size == (width, height):
                        return ("skip", str(src), None)
            except Exception:
                pass  # file hỏng → làm lại

        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            if to_gray and im.mode != "L":
                im = im.convert("L")
            im = im.resize((width, height), resample)
            im.save(dst, format="PNG", optimize=True)
        return ("ok", str(src), None)
    except Exception as e:
        return ("err", str(src), f"{type(e).__name__}: {e}")


def rewrite_csv(csv_in: Path, csv_out: Path, src_root: str, dst_root: str):
    import pandas as pd
    df = pd.read_csv(csv_in)
    if "image_path" not in df.columns:
        print(f"  ⚠ {csv_in} không có cột image_path — bỏ qua.")
        return
    src_n, dst_n = src_root.replace("\\", "/").rstrip("/"), dst_root.replace("\\", "/").rstrip("/")
    df["image_path"] = (df["image_path"].astype(str).str.replace("\\", "/", regex=False)
                        .str.replace(src_n, dst_n, regex=False))
    df.to_csv(csv_out, index=False)
    print(f"  ✔ CSV mới: {csv_out}  ({len(df)} dòng)")
    print(f"    ví dụ: {df['image_path'].iloc[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="images_cropped")
    ap.add_argument("--dst", default="images_928x352")
    ap.add_argument("--height", type=int, default=928, help="chiều cao đích (H)")
    ap.add_argument("--width", type=int, default=352, help="chiều rộng đích (W)")
    ap.add_argument("--resample", default="lanczos", choices=list(RESAMPLE))
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--csv", default="labels.csv", help="CSV cần sinh bản mới ('' để bỏ qua)")
    ap.add_argument("--keep-rgb", action="store_true", help="giữ 3 kênh thay vì chuyển sang L")
    ap.add_argument("--force", action="store_true", help="làm lại cả file đã có")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_root, dst_root = Path(args.src), Path(args.dst)
    if not src_root.is_dir():
        sys.exit(f"Không thấy thư mục nguồn: {src_root.resolve()}")

    files = sorted(src_root.rglob("*.png"))
    if not files:
        sys.exit(f"Không có file .png nào trong {src_root.resolve()}")

    # Kiểm tra tỉ lệ khung hình
    with Image.open(files[0]) as im:
        sw, sh = im.size
    ratio_src, ratio_dst = sh / sw, args.height / args.width
    print(f"Nguồn : {src_root}  ({len(files)} ảnh, {sw}x{sh}, tỉ lệ {ratio_src:.4f})")
    print(f"Đích  : {dst_root}  ({args.width}x{args.height}, tỉ lệ {ratio_dst:.4f})")
    print(f"Resample: {args.resample} | workers: {args.workers} | "
          f"grayscale: {not args.keep_rgb}")
    if abs(ratio_src - ratio_dst) > 0.01:
        print(f"  ⚠ Tỉ lệ lệch {abs(ratio_src-ratio_dst):.3f} → ảnh sẽ bị méo!")
    else:
        print(f"  ✔ Giữ nguyên tỉ lệ (scale {args.width/sw:.3f}x)")

    if args.dry_run:
        print("\n[dry-run] 5 file đầu sẽ được ghi thành:")
        for f in files[:5]:
            print(f"  {f}  →  {dst_root / f.relative_to(src_root)}")
        print(f"\n[dry-run] Không ghi gì. Bỏ --dry-run để chạy thật.")
        return

    tasks = [(str(f), str(dst_root / f.relative_to(src_root)), args.width, args.height,
              RESAMPLE[args.resample], not args.keep_rgb, args.force) for f in files]

    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "err": 0}
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(resize_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            status, path, err = fut.result()
            counts[status] += 1
            if status == "err":
                errors.append((path, err))
            if i % 200 == 0 or i == len(futures):
                rate = i / max(1e-6, time.time() - t0)
                eta = (len(futures) - i) / max(1e-6, rate)
                print(f"  {i}/{len(futures)}  ({rate:.0f} ảnh/s, còn ~{eta/60:.1f} phút)")

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"  Xong sau {elapsed:.1f}s — resize {counts['ok']}, bỏ qua {counts['skip']}, lỗi {counts['err']}")
    if errors:
        print(f"\n  {len(errors)} lỗi đầu tiên:")
        for p, e in errors[:10]:
            print(f"    {p}\n      {e}")

    def folder_mb(p):
        return sum(f.stat().st_size for f in Path(p).rglob("*.png")) / 1024**2
    print(f"\n  Dung lượng: {folder_mb(src_root):.0f} MB  →  {folder_mb(dst_root):.0f} MB")

    if args.csv:
        csv_in = Path(args.csv)
        if csv_in.exists():
            csv_out = csv_in.with_name(f"{csv_in.stem}_{args.width}x{args.height}{csv_in.suffix}")
            print()
            rewrite_csv(csv_in, csv_out, str(src_root), str(dst_root))
            print(f"\n  → Cập nhật configs/config.py:")
            print(f"      data_root  = \"{dst_root}\"")
            print(f"      csv_path   = \"{csv_out}\"")
            print(f"      image_size = ({args.height}, {args.width})")
        else:
            print(f"\n  ⚠ Không thấy {csv_in} — bỏ qua bước sinh CSV.")
    print("=" * 55)


if __name__ == "__main__":
    main()
