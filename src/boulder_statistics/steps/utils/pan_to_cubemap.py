import cv2
import numpy as np


class PANToCubemap():
    @staticmethod
    def sample_face_roi(e_img, face: str, face_w: int,
                        x0: int, y0: int, w: int, h: int):
        H, W = e_img.shape[:2]

        # pixel centers for ROI in face space → [-1, 1]
        xs = (np.arange(x0, x0 + w, dtype=np.float32) + 0.5) / face_w * 2.0 - 1.0
        ys = (np.arange(y0, y0 + h, dtype=np.float32) + 0.5) / face_w * 2.0 - 1.0
        ys = ys[::-1]  # make top row +V
        U, V = np.meshgrid(xs, ys)

        # cube directions
        if face == "posx":
            x, y, z = np.ones_like(U), V, -U
        elif face == "negx":
            x, y, z = -np.ones_like(U), V, U
        elif face == "posy":
            x, y, z = U, np.ones_like(U), -V
        elif face == "negy":
            x, y, z = U, -np.ones_like(U), V
        elif face == "posz":
            x, y, z = U, V, np.ones_like(U)
        elif face == "negz":
            x, y, z = -U, V, -np.ones_like(U)
        else:
            raise ValueError("face must be posx/negx/posy/negy/posz/negz")

        # normalize & map to equirect (lon/lat → pixels)
        L = np.maximum(np.sqrt(x * x + y * y + z * z), 1e-8)
        x, y, z = x / L, y / L, z / L
        lon = np.arctan2(x, z)
        lat = np.arcsin(np.clip(y, -1.0, 1.0))

        mapx = (lon / (2 * np.pi) + 0.5) * W
        mapy = (0.5 - (lat / np.pi)) * H
        mapx = (mapx % W).astype(np.float32)         # wrap horizontally
        mapy = np.clip(mapy, 0, H - 1).astype(np.float32)  # clamp vertically

        # sample just the ROI
        tile = cv2.remap(
            e_img,
            mapx,
            mapy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP)
        return tile  # shape (h, w, C)

# # --- example: render a 512x512 patch from the +Z face ---
# os.makedirs("simple_extract", exist_ok=True)
# roi = (1024, 768, 512, 512)  # (x0, y0, w, h) in that face's pixel space
# patch = sample_face_roi(img, "posz", face_w, *roi)
# iio.imwrite(f"simple_extract/posz_{roi[0]}_{roi[1]}_{roi[2]}x{roi[3]}.png", patch)
