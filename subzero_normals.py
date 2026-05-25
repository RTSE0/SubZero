import cv2
import cv2.ximgproc
import numpy as np
import torch
import os
import argparse

from PIL import Image
from diffusers import MarigoldNormalsPipeline

from subzero_core import (
    FrameSource, prepare_frame, composite_on_gray,
    _download_bisenet_weights, _build_bisenet,
    parse_face_regions, temporal_blend_regional, write_frame,
    get_face_confidence, get_temporal_scale, get_head_rotation_matrix,
    build_face_landmarker,
)

LABEL = "SubZero-Normals"

# ---- Eye spherical dome normal ----

def generate_eye_dome_normal(eye_mask: np.ndarray, normal_bgr_shape: tuple, rotation_matrix = None) -> np.ndarray:
    """
    This will generate a smooth dome normal map for the eye region.

    Why? You may ask. Eyes are wet specular spheres. Marigold reads shading cues to infer
    geometry, but the pupil.iris has no diffuse shading - it's specular and
    reflective. The model hallucinates normals there, producing noisy garbage.
    Best way to tackle this is to smooth eyes to a dome approximation
    a dome normal is more physically accurate and looks cleaner

    1. Find ellipse of eye mask region
    2. For every pixel inside the ellipse, compute where it sits on a unit hemisphere (x,y --> z via sqrt(1 - x² - y²))
    3. Encode XYZ normal vector as RGB (same convention as Marigold's visualisation: R=X, G=Y, B=Z, remapped from [-1,1] to [0,255])
    4. Return the dome normal image -- blend it over model output using eye_mask as the blend weight
    """
    h,w = normal_bgr_shape[:2]
    dome = np.zeros((h, w, 3), dtype=np.float32)

    # Find connected components in eye mask (left eye + right eye = 2 blobs)
    binary = (eye_mask > 64).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary)

    for lid in range(1, num_labels):
        blob = (labels==lid).astype(np.uint8)

        #Fit ellipse to the blob for clean dome centre + radius
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or len(contours[0]) < 5:
            continue # You need at least 5 points to fit ellipse

        ellipse = cv2.fitEllipse(contours[0])
        cx, cy = ellipse[0] # center
        radius   = max(ellipse[1][0], ellipse[1][1]) / 2.0

        if radius < 2:
            continue

        # Pixel coordinate grid for bounding box of eye
        x1, x2 = max(0, int(cx-radius)-2), min(w,int(cx+radius)+2)
        y1, y2 = max(0, int(cy-radius)-2), min(h,int(cy+radius)+2)
        xv, yv = np.meshgrid(np.arange(x1,x2),np.arange(y1,y2))

        nx = (xv-cx)/radius
        ny = (yv-cy)/radius
        r2 = nx**2 + ny**2
        nz = np.where(r2 <= 1.0, np.sqrt(np.clip(1.0 - r2, 0.0, 1.0)), 0.0)

        if rotation_matrix is not None:
            # Stack into (N ,3) array,rotate,unstack
            flat = np.stack([nx.flatten(), (-ny).flatten(), nz.flatten()], axis=1)
            rotated = (rotation_matrix @ flat.T).T
            nx = rotated[:, 0].reshape(nx.shape)
            ny = -rotated[:, 1].reshape(ny.shape) #flip Y back
            nz = rotated[:, 2].reshape(nz.shape)

        # Marigold normal visualisation: RB = (X+1)/2, (Y+1)/2, (Z+1)/2
        # X points right (+R), Y points up (+G), Z points to the camera (+B)
        #ny is flipped because image Y increases downward
        r_chan = np.clip((nx+1.0)/2.0 * 255.0,0,255)
        g_chan = np.clip((-ny + 1.0)/2.0 *255.0,0,255)
        b_chan = np.clip((nz+1.0)/2.0 *255.0,0,255)
        
        # Write dome normals only where the ellipse blob is
        blob_crop = blob[y1:y2, x1:x2].astype(bool)
        dome[y1:y2, x1:x2, 0] = np.where(blob_crop, r_chan, dome[y1:y2, x1:x2, 0])
        dome[y1:y2, x1:x2, 1] = np.where(blob_crop, g_chan, dome[y1:y2, x1:x2, 1])
        dome[y1:y2, x1:x2, 2] = np.where(blob_crop, b_chan, dome[y1:y2, x1:x2, 2])
    
    return cv2.cvtColor(np.clip(dome,0,255).astype(np.uint8),cv2.COLOR_RGB2BGR)

#---- Pre-region filtering ----

def apply_region_filtering(
    normal_bgr: np.ndarray,
    guide_rgb: np.ndarray,
    masks: dict,
    confidence:float=1.0,
    rotation_matrix=None,
) -> np.ndarray:
    """
    All filtering happens at native model resolution before upscaling.

    Face: Gentle guided filter to preserve fine surface geometry
    Eyes: Spherical dome normal - replace the hallucinated model output entirely
    Hair: aggressive guided + Gaussian - smooth to volume envelope
    Clothing: moderate guided filter
    Fallback: single global guided filter (BiSeNet unavailable)
    """
    guide = cv2.cvtColor(guide_rgb, cv2.COLOR_RGB2BGR).astype(np.uint8)

    face_mask = masks['face']
    eye_mask = masks['eye']
    hair_mask = masks['hair']
    cloth_mask = masks['cloth']
    mouth_mask = masks['mouth']

    all_zero = all(m.max() == 0 for m in [face_mask, eye_mask, hair_mask, cloth_mask])
    if all_zero:
        return cv2.ximgproc.guidedFilter(guide = guide, src=normal_bgr, radius = 6, eps=120)
    
    face_filtered = cv2.ximgproc.guidedFilter(guide = guide, src = normal_bgr, radius = 4, eps = 50)
    mouth_filtered = cv2.ximgproc.guidedFilter(guide = guide, src = normal_bgr, radius = 4, eps = 50)
    hair_filtered = cv2.ximgproc.guidedFilter(guide = guide, src = normal_bgr, radius = 18, eps = 500)
    hair_filtered = cv2.GaussianBlur(hair_filtered, (21,21), 0)
    cloth_filtered = cv2.ximgproc.guidedFilter(guide = guide,src=normal_bgr, radius = 10, eps=200)

    def _w(m): return np.stack([m.astype(np.float32)/255.0]*3, axis=-1)

    # Layered compositing - eye wins over face wins over hair wins over cloth
    result = normal_bgr.astype(np.float32)
    result = result * (1.0 - _w(cloth_mask)) + cloth_filtered.astype(np.float32) * _w(cloth_mask)
    result = result * (1.0 - _w(hair_mask)) + hair_filtered.astype(np.float32) * _w(hair_mask)
    result = result * (1.0 - _w(face_mask)) + face_filtered.astype(np.float32) * _w(face_mask)
    result = result * (1.0 - _w(mouth_mask)) + mouth_filtered.astype(np.float32) * _w(mouth_mask)

    if confidence > 0.05:
        eye_dome = generate_eye_dome_normal(eye_mask, normal_bgr.shape, rotation_matrix)
        result = result * (1.0 - _w(eye_mask)) + eye_dome.astype(np.float32) * _w(eye_mask)

    return np.clip(result, 0, 255).astype(np.uint8)

# ---- Main Pipeline ----

def generate_normals(
    input_path: str,
    output_dir: str,
    steps: int = 10,
    ensemble_size: int = 12,
    processing_res: int = 1024,
    temporal: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SubZero] Engine: {device.upper()}")
    print(f"[SubZero] ensemble = {ensemble_size} | res = {processing_res} | steps = {steps}\n")
    os.makedirs(output_dir, exist_ok=True)

    pipe = MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-v1-1",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    print("[SubZero] Marigold loaded.")

    _download_bisenet_weights()
    bisenet = _build_bisenet(device)
    landmarker = build_face_landmarker()
    
    source = FrameSource(input_path, label = LABEL)
    total = source.total_frames
    w_out = source.width
    h_out = source.height
    print(f"[SubZero] {w_out}x{h_out} | {total} frames\n")

    prev_bgr = None
    frame_idx = 0
    prev_raw_frame = None
    for frame_bgr, alpha in source:
        # Composite subject on neutral gray
        isolated, alpha_out = prepare_frame(frame_bgr, alpha, label = LABEL)

        #Marigold inference
        pil_image = Image.fromarray(cv2.cvtColor(isolated, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            generator = torch.Generator(device = device).manual_seed(42)
            output = pipe(
                pil_image,
                num_inference_steps = steps,
                ensemble_size = ensemble_size,
                generator = generator,
                processing_resolution = processing_res,
            )
            normal_pil = pipe.image_processor.visualize_normals(output.prediction)[0]
        
        # Native model output
        native_bgr = cv2.cvtColor(np.array(normal_pil), cv2.COLOR_RGB2BGR)
        native_w, native_h = native_bgr.shape[1], native_bgr.shape[0]

        # Downscale guide to native resolution 
        guide_native = cv2.resize(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
            (native_w, native_h),
            interpolation = cv2.INTER_LINEAR
        )

        # Parse face regions at native resolution (eyes now seperated)
        masks = parse_face_regions(
            frame_rgb = guide_native,
            bisenet = bisenet,
            device = device,
            target_size = (native_w, native_h)
        )

        conf = get_face_confidence(masks)
        scale = get_temporal_scale(conf, prev_raw_frame, frame_bgr)
        
        rotation_matrix = get_head_rotation_matrix(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), landmarker
        ) if conf > 0.05 else None

        # Per-region filtering + eye dome replacement at native resolution
        native_filtered = apply_region_filtering(native_bgr, guide_native, masks, conf, rotation_matrix)

        # Upscale clean data
        if native_filtered.shape[1] != w_out or native_filtered.shape[0] != h_out:
            final_bgr = cv2.resize(native_filtered, (w_out, h_out), interpolation = cv2.INTER_LINEAR)
        else:
            final_bgr = native_filtered
        
        # Temporal smoothing
        if temporal and prev_bgr is not None:
            masks_full = {
                k: cv2.resize(v, (w_out, h_out), interpolation = cv2.INTER_LINEAR)
                for k,v in masks.items()
            }
            final_bgr = temporal_blend_regional(final_bgr, prev_bgr, masks_full, scale)
        
        prev_bgr = final_bgr.copy()
        write_frame(final_bgr, alpha_out, output_dir, "normal", frame_idx, total)

        prev_raw_frame = frame_bgr.copy()

        frame_idx += 1
        if frame_idx % 5 == 0 or frame_idx == total:
            print(f"[{LABEL}] {frame_idx}/{total} frames...", end="\r")
    
    print(f"\n[{LABEL}] ✓ Done — PNG sequence saved to: {output_dir}/")

#---- CLI ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SubZero Normals v1")
    parser.add_argument("--input", required = True, help="Path to input video or folder")
    parser.add_argument("--output", required = True, help = "Path to save output maps")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--ensemble", type=int, default=12)
    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument("--preview", action="store_true",
                        help="Fast preview: steps=4, ensemble=4, res=512")
    args = parser.parse_args()
 
    if args.preview:
        args.steps, args.ensemble, args.res = 4, 4, 512
        print(f"[{LABEL}] Preview mode\n")
 
    generate_normals(
        input_path = args.input,
        output_dir = args.output,
        steps = args.steps,
        ensemble_size = args.ensemble,
        processing_res = args.res,
        temporal = not args.no_temporal,
    )