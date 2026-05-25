import cv2
import cv2.ximgproc
import numpy as np
import torch
import os
import glob
import importlib.util
import sys
import urllib.request

# ---- BiSeNet label map ----
FACE_LABELS = {1, 2, 3, 6, 7, 8, 9, 10} # Each number is a segment on a persons face
EYE_LABELS = {4, 5} # I will seperate the eyes to avoid having the normals have too much detail in them
MOUTH_LABELS = {11, 12, 13}
HAIR_LABELS = {14, 17} # Hair causes too much shadows, best handle them seperately
CLOTH_LABELS = {18, 19}

BISENET_MODEL_URL = (
    "https://github.com/zllrunning/face-parsing.PyTorch/"
    "releases/download/v1.0/79999_iter.pth"
)
BISENET_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bisenet_face_parsing.pth"
)

# How much HISTORY to keep per region, Higher is more stable but ghosting, Lower is flickery but tracks motion cleanly
TEMPORAL_WEIGHTS = {
    'forehead': 0.65,
    'hair': 0.7,
    'face': 0.55,
    'cloth': 0.6,
    'mouth': 0.15,
    'eye': 0.15,
    'default': 0.5
}

# ---- Input: Frame Source ----
class FrameSource:
    '''
    Unified frame iterator yielding (frame_bgr, alpha_uint8 | None).

    Accepted inputs (in priority order):
    1. Folder / PNG sequences with alpha
    2. Folder / EXR sequences with alpha
    3. MP4 / MOV
    '''
    def __init__(self, input_path: str, label: str = "SubZero"):
        self.frames = []
        self.mode = None # 'png' | 'mp4' | 'video'
        self.cap = None
        self.label = label
        self._detect(input_path)
    
    def _detect(self, path: str):
        if os.path.isdir(path):
            pngs = sorted(glob.glob(os.path.join(path, "*.png")))
            exrs = sorted(glob.glob(os.path.join(path, "*.exr")))

            if pngs:
                self.frames, self.mode = pngs, "png"
                print(f"[LOG] Detected PNG sequence")
            elif exrs:
                self.frames, self.mode = exrs, "exr"
                print(f"[LOG] Detected EXR sequence")
            else:
                raise FileNotFoundError(f"[ERR] No .png or .exr files found in: {path}")
        elif "*" in path:
            files = sorted(glob.glob(path))
            if not files:
                raise FileNotFoundError(f"[ERR] Glob matched no files: {path}")
            self.mode = 'png' if os.path.splitext(files[0])[1].lower() == '.png' else 'exr'
            self.frames = files
        elif path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            self.mode = 'video'
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise FileNotFoundError(f"[ERR] Could not open video: {path}")

            print(f"\n[{self.label}] WARNING: VIDEO INPUT - no alpha channel present.")
            print(f"[{self.label}] For best results export a pre-keyed PNG sequence from Davinci or any masking software.")
            print(f"[{self.label}]    Falling back to rembg — hair edges will be lower quality.\n")
        else:
            raise ValueError(f"Unrecognised input: {path}")
        
        if self.mode in ('png', 'exr'):
            print(f"[{self.label}] Input: {self.mode.upper()} sequence — {len(self.frames)} frames. ✓")
            print(f"[{self.label}] Alpha will be read directly from frames.\n")
    
    @property
    def total_frames(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.mode == 'video' else len(self.frames)

    @property
    def width(self) -> int:
        if self.mode == 'video':
            return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return cv2.imread(self.frames[0], cv2.IMREAD_UNCHANGED).shape[1]
        
    @property
    def height(self) -> int:
        if self.mode == 'video':
            return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cv2.imread(self.frames[0], cv2.IMREAD_UNCHANGED).shape[0]

    def _read_image_frame(self, path: str):
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError(f"Could not read: {path}")
        if path.lower().endswith('.exr'):
            if raw.dtype != np.uint8:
                raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)
                if raw.max() > 1.0:
                    p99 = np.percentile(raw[raw > 0], 99)
                    raw = raw / p99
                raw = np.clip(raw, 0.0, 1.0)
                raw = (raw * 255.0).astype(np.uint8)
        # IF ITS 16-BIT
        if raw.dtype == np.uint16:
            raw = (raw>>8).astype(np.uint8)
        # UNIVERSAL HIGH BIT DEPTH PROTECTION
        #This only runs for weird/unexpected formats that aren't 8-bit, 16-bit, or EXR
        elif raw.dtype != np.uint8:
            print(f"[{self.label}] Normalizing high bit-depth frame ({raw.dtype}): {os.path.basename(path)}")
            raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)

            if raw.max() > 1.0:
                p99 = np.percentile(raw[raw > 0], 99) if raw.max() > 0 else 1.0
                if p99 == 0: p99 = 1.0
                raw = raw / p99
            
            raw = np.clip(raw, 0.0, 1.0)
            raw = (raw * 255.0).astype(np.uint8)

        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        if raw.shape[2] == 4:
            return raw[:, :, :3], raw[:, :, 3]
        return raw, None
    
    def __iter__(self):
        if self.mode == 'video':
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                yield frame, None
            self.cap.release()
        else:
            for path in self.frames:
                yield self._read_image_frame(path)

# ---- Subject Isolation ----

def composite_on_gray(frame_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Composite subject on neutral gray #808080
    """
    a = alpha.astype(np.float32)[:,:,np.newaxis] / 255.0
    fg = frame_bgr.astype(np.float32)
    bg = np.full_like(fg, 128.0)
    return np.clip(fg*a+bg*(1.0-a),0,255).astype(np.uint8)

def prepare_frame(frame_bgr: np.ndarray, alpha, label: str = "SubZero"):
    if alpha is not None:
        return composite_on_gray(frame_bgr, alpha), alpha
    try:
        from rembg import remove
        from PIL import Image as PILImage
        pil_in = PILImage.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        pil_out = remove(pil_in)
        rgba = np.array(pil_out)
        al = rgba[:,:,3]
        bgr = cv2.cvtColor(rgba[:,:,:3], cv2.COLOR_RGB2BGR)
        return composite_on_gray(bgr, al), al
    except ImportError:
        print(f"[{label}] rembg not installed — pip install rembg for video fallback.")
        h, w = frame_bgr.shape[:2]
        return frame_bgr, np.full((h, w), 255, dtype=np.uint8)

# ---- Face Parsing ----

def _download_bisenet_weights():
    if not os.path.exists(BISENET_MODEL_PATH):
        import urllib.request
        print("[SubZero] Downloading BiSeNet weights (~50 MB)...")
        urllib.request.urlretrieve(BISENET_MODEL_URL, BISENET_MODEL_PATH)
        print("[SubZero] BiSeNet weights downloaded.")

def _build_bisenet(device: str):
    for candidate in [
        "./face-parsing.PyTorch",
        "../face-parsing.PyTorch",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "face-parsing.PyTorch"),
    ]:
        model_file = os.path.join(candidate, "model.py")
        if os.path.isfile(model_file):
            bisenet_dir = os.path.dirname(model_file)
            if bisenet_dir not in sys.path:
                sys.path.insert(0, bisenet_dir)     
            spec = importlib.util.spec_from_file_location("bisenet_model", model_file)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["bisenet_model"] = mod
            try:
                spec.loader.exec_module(mod)
                net = mod.BiSeNet(n_classes=19)
                net.load_state_dict(torch.load(BISENET_MODEL_PATH, map_location='cpu'))
                net = net.float()
                net.to(device).eval()
                print("[SubZero] BiSeNet face parser loaded.")
                return net
            except Exception as e:
                print(f"[SubZero] Warning: Found BiSeNet files but failed to initialize model: {e}")
                return None

    print("[SubZero] BiSeNet repo not found — clone it for best results:")
    print("[SubZero]   git clone https://github.com/zllrunning/face-parsing.PyTorch")
    print("[SubZero] Continuing with global guided filter fallback.\n")
    return None

def parse_face_regions(frame_rgb: np.ndarray, bisenet, device: str, target_size: tuple):
    """
    returns face mask, eye mask, hair mask, cloth mask at target size
    eyes are now seperated from face so they get their own treatment.
    falls back to all-face / empty mask if bisenet unavailable.
    """
    tw, th = target_size
    zeros = np.zeros((th, tw), dtype=np.uint8)

    if bisenet is None:
        return {
            'face': np.full((th, tw), 255, dtype=np.uint8),
            'eye': zeros.copy(),
            'mouth': zeros.copy(),
            'hair': zeros.copy(),
            'cloth': zeros.copy(),
            'forehead': zeros.copy(),
        }
    
    resized = cv2.resize(frame_rgb, (512,512)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalised = (resized-mean)/std
    tensor = (torch.from_numpy(normalised.transpose(2,0,1))
    .unsqueeze(0).float().to(device))

    with torch.no_grad():
        out = bisenet(tensor)[0]
    seg = out.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
    
    def _mask(labels, blur):
        m = np.isin(seg, list(labels)).astype(np.uint8) * 255
        m = cv2.resize(m, (tw, th), interpolation = cv2.INTER_NEAREST)
        return cv2.GaussianBlur(m, (blur, blur), 0)

    face_mask = _mask(FACE_LABELS, 11)
    eye_mask = _mask(EYE_LABELS, 5)
    mouth_mask = _mask(MOUTH_LABELS, 7)
    hair_mask = _mask(HAIR_LABELS, 11)
    cloth_mask = _mask(CLOTH_LABELS, 11)

    """
    Forehead apporximation:
    Find topmost row of face mask with significant coverage
    then take the band between that and the eyebrow line (top ~30% of the face)
    This gives us the stable, flat forehead region for aggressive smoothing
    """
    face_binary = (face_mask > 64).astype(np.uint8)
    forehead_mask = np.zeros((th,tw), dtype=np.uint8)
    rows_with_face = np.where(face_binary.sum(axis=1) > tw* 0.1)[0]
    if len(rows_with_face) > 4:
        top_row = rows_with_face[0]
        face_height = rows_with_face[-1] - top_row
        forehead_bottom = top_row + int(face_height * 0.28) # top 28% of face span
        forehead_mask[top_row:forehead_bottom,:] = face_binary[top_row:forehead_bottom,:]*255
        forehead_mask = cv2.GaussianBlur(forehead_mask, (15,15), 0)

    return {
        'face':face_mask,
        'eye':eye_mask,
        'mouth':mouth_mask,
        'hair':hair_mask,
        'cloth':cloth_mask,
        'forehead':forehead_mask,
    }

# ---- Build Face Landmarker ----

def build_face_landmarker():
    import mediapipe as mp
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
    if not os.path.exists(model_path):
        import urllib.request
        print("[SubZero] Downloading Mediapipe face landmarker model (~6MB)...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            model_path            
        )
        print("[SubZero] Mediapipe model downloaded.")
    
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode = VisionRunningMode.IMAGE,
        num_faces=1
    )
    return FaceLandmarker.create_from_options(options)


# ---- Get head pose ----

def get_head_rotation_matrix(frame_rgb: np.ndarray, landmarker) -> np.ndarray:
    """
    Use Mediapipe face mesh to estimate head rotation.
    returns a 3x3 rotation matrix, or None if no face is detected
    Uses new MediaPipe 0.10.X API. 
    """
    import mediapipe as mp

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None
    
    landmarks = result.face_landmarks[0]
    h, w = frame_rgb.shape[:2]

    # Use 6 stable anchor points for pose estimation 
    # these are well known stable landmarks for solvePnP
    image_points = np.array([
        [landmarks[1].x * w, landmarks[1].y * h], # nose tip
        [landmarks[152].x * w, landmarks[152].y * h], # chin 
        [landmarks[263].x * w, landmarks[263].y * h], # left eye corner
        [landmarks[33].x * w, landmarks[33].y * h], # right eye corner
        [landmarks[287].x * w, landmarks[287].y * h], # left mouth corner
        [landmarks[57].x * w, landmarks[57].y * h], # right mouth corner
    ],dtype=np.float64)

    # Generic 3D model points for a normalised face
    model_points = np.array([
        [0.0,0.0,0.0], # nose tip
        [0.0,-63.6,-12.5], # chin
        [-43.4,32.7,-26.0], # left eye corner
        [43.4,32.7,-26.0], # right eye corner
        [-28.9,-28.9,-24.1], # left mouth corner
        [28.9,-28.9,-24.1], # right mouth corner
    ],dtype=np.float64)

    #approximate camera intrinsics (focal length = image width)
    focal = w
    camera_matrix = np.array([
        [focal, 0, w/2],
        [0, focal, h/2],
        [0,0,1],
    ],dtype=np.float64)

    dist_coeffs = np.zeros((4,1))

    success, rotation_vector, _ = cv2.solvePnP(
        model_points,image_points,camera_matrix,dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    return rotation_matrix

# ---- Face Confidence ----

def get_face_confidence(masks: dict) -> float:
    """
    Returns 0.0 to 1.0 - how much of the frame is a forward facing face
    0.0 means no face
    0.08+ full forward face
    """
    face_pixels = (masks['face'] > 64).sum()
    total_pixels = masks['face'].size
    return face_pixels / total_pixels

def get_temporal_scale(
    confidence: float,
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
) -> float:
    """
    Combines the confidence + motion into one scale factor (0.0 to 1.0)
    multiplied against all temporal_weights before blender
    near 1.0 is a stable forward face and use full history weights
    near 0.0 is turning/back to camera which basically discables temporal smoothing
    """
    if prev_frame is None:
        return 1.0

    # Motion detection - how different are the two frames
    diff = cv2.absdiff(
        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    )
    motion = diff.mean()
    motion_scale = np.clip(1.0-(motion/30.0), 0.2, 1.0)
    confidence_scale = np.clip(confidence/0.08,0.0,1.0)

    return float(min(motion_scale, confidence_scale))
# ---- Regional Aware Temporal Smoothing ----

def temporal_blend_regional(
    current: np.ndarray,
    previous: np.ndarray,
    masks: dict,
    scale: float=1.0,
) -> np.ndarray:
    if previous is None:
        return current
    
    h, w = current.shape[:2]

    # Start with default weight everywhere
    weight_map = np.full((h,w), TEMPORAL_WEIGHTS['default'], dtype=np.float32)

    # Apply region weights from most stable to least stable, the order matters: 
    # later assignments override earlier ones. So fast moving regions (written last) win over stable ones
    region_order = [
        ('hair', TEMPORAL_WEIGHTS['hair']),
        ('cloth', TEMPORAL_WEIGHTS['cloth']),
        ('forehead', TEMPORAL_WEIGHTS['forehead']),
        ('face', TEMPORAL_WEIGHTS['face']),
        ('mouth', TEMPORAL_WEIGHTS['mouth']),
        ('eye', TEMPORAL_WEIGHTS['eye']),
    ]
    
    for region_name, history_weight in region_order:
        mask = masks.get(region_name)
        if mask is None:
            continue
        
        #Soft mask: Pixels at mask = 255 fully use this weight,
        # pixels at mask = 0 keep whatever weight they already have
        mask_f = mask.astype(np.float32)/255.0
        weight_map = weight_map * (1.0 - mask_f) + (history_weight * scale) * mask_f

    # Expand to 3 channels for per-pixel blend
    weight_3ch = np.stack([weight_map]*3, axis=-1)

    blended = (
        previous.astype(np.float32) * weight_3ch
        +
        current.astype(np.float32) * (1.0 - weight_3ch)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)

#---- Output ----
    
def write_frame(
    img_bgr: np.ndarray,
    alpha: np.ndarray,
    output_dir: str,
    prefix: str,
    frame_idx: int,
    total: int,
):
    pad = max(4, len(str(total)))
    fpath = os.path.join(output_dir, f"{prefix}_{str(frame_idx).zfill(pad)}.png")
    if alpha is not None:
        bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        alpha_resized = cv2.resize(alpha, (img_bgr.shape[1], img_bgr.shape[0]))
        bgra[:, :, 3] = alpha_resized
        cv2.imwrite(fpath, bgra)
    else:
        cv2.imwrite(fpath, img_bgr)