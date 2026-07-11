import os
from PIL import Image, ImageChops

def get_content_bbox(img_path):
    img = Image.open(img_path).convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    return diff.getbbox(), img.size

def main():
    images = [
        "category_gossip_recommendation.png",
        "demo_mode_visible.png",
        "fedkg_recommendation.png",
        "lock_screen.png",
        "settings.png"
    ]
    
    valid_images = [img for img in images if os.path.exists(img)]
    if not valid_images:
        print("Keine Bilder gefunden.")
        return

    bboxes = {}
    max_w = 0
    max_h = 0
    
    for img_name in valid_images:
        bbox, size = get_content_bbox(img_name)
        if bbox:
            l, u, r, d = bbox
            w = r - l
            h = d - u
            max_w = max(max_w, w)
            max_h = max(max_h, h)
            bboxes[img_name] = (bbox, size)
        else:
            print(f"{img_name}: Kein Inhalt.")

    if not bboxes:
        return

    margin_x = 40
    margin_y = 20
    
    target_w = max_w + 2 * margin_x
    target_h = max_h + 2 * margin_y
    
    for img_name in valid_images:
        if img_name not in bboxes:
            continue
            
        bbox, original_size = bboxes[img_name]
        l, u, r, d = bbox
        
        # Wenn ein Bild am Rand abgeschnitten war (z.B. fehlender Schatten rechts bei fedkg),
        # ist die Breite (r - l) kleiner als die maximale Breite max_w.
        # In diesem Fall orientieren wir uns an der intakten Seite (z.B. linker Rand),
        # damit das Handy exakt auf der gleichen Position liegt wie bei den anderen Bildern.
        dist_left = l
        dist_right = original_size[0] - r
        
        if (r - l) < max_w - 5: 
            if dist_right < dist_left:
                # Rechts abgeschnitten -> Nutze linken Rand als Anker
                cx = l + max_w / 2.0
            else:
                # Links abgeschnitten -> Nutze rechten Rand als Anker
                cx = r - max_w / 2.0
        else:
            # Bild intakt -> normale Zentrierung
            cx = (l + r) / 2.0
            
        cy = (u + d) / 2.0
        
        crop_left = int(cx - target_w / 2.0)
        crop_upper = int(cy - target_h / 2.0)
        crop_right = crop_left + target_w
        crop_lower = crop_upper + target_h
        
        # WICHTIG: Konvertiere zu "RGB", damit transparente Pixel nicht schwarz werden!
        img = Image.open(img_name).convert("RGB")
        new_img = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        
        src_left = max(0, crop_left)
        src_upper = max(0, crop_upper)
        src_right = min(img.width, crop_right)
        src_lower = min(img.height, crop_lower)
        
        dst_left = src_left - crop_left
        dst_upper = src_upper - crop_upper
        
        if src_right > src_left and src_lower > src_upper:
            cropped_region = img.crop((src_left, src_upper, src_right, src_lower))
            new_img.paste(cropped_region, (dst_left, dst_upper))
        
        new_img.save(img_name)
        print(f"Perfekt zentriert und repariert: {img_name}")

if __name__ == "__main__":
    main()
