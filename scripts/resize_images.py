import cv2
import os

input_root = "image"
output_root = "resized_images"
os.makedirs(output_root, exist_ok=True)

image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

count = 0
for root, dirs, files in os.walk(input_root):
    # Compute the relative path and the new output folder
    rel_path = os.path.relpath(root, input_root)
    output_folder = os.path.join(output_root, rel_path)
    os.makedirs(output_folder, exist_ok=True)
    for filename in files:
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] Cannot read: {img_path} (check format or corruption)")
                continue
            try:
                resized_img = cv2.resize(img, (299, 299))
                out_path = os.path.join(output_folder, filename)
                cv2.imwrite(out_path, resized_img)
                count += 1
                print(f"Resized and saved: {out_path}")
            except Exception as e:
                print(f"[ERROR] Failed resize {img_path}: {e}")

print(f"\nDone! Total resized images: {count}")
