import os
import sys
import pickle
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import binary_dilation, generate_binary_structure

import vtk
from vtk.util import numpy_support

# ── Reproductibilité ──────────────────────────────────────────────────────────
np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CONFIGURATION adapter ces chemins

BASE_DIR = Path(".")
RESULTS_DIR = BASE_DIR / "outputs/benchmark/jerman/ircad_enhancer_jerman_2026-07-11_20-39-32/results"
IMAGES_DIR = BASE_DIR / "data/ircad/3d-échantillonnées/images"
LABELS_DIR = BASE_DIR / "data/ircad/3d-échantillonnées/labels"
LIVER_DIR = BASE_DIR / "data/ircad/3d-échantillonnées/masks"
OUTPUT_DIR = BASE_DIR / "outputs/benchmark/jerman/ircad_enhancer_jerman_2026-07-11_20-39-32/overview/visualisation_3D"

OPERATORS = ['default', 'gaussian', 'farid', 'cubic', 'trigonometric',
             'catmull', 'bspline', 'bezier','scharr']

PATIENTS = [f"{i:02d}"for i in range(1, 34)]

# LABELS DE LA MATRICE DE CONFUSION
# 0 = background (hors foie)
# 1 = TN vrai négatif foie non vasculaire, non prédit
# 2 = TP vrai positif GT ∩ Pred
# 3 = FP faux positif Pred \ GT
# 4 = FN faux négatif GT \ Pred

LABEL_BG = 0
LABEL_TN = 1
LABEL_TP = 2
LABEL_FP = 3
LABEL_FN = 4

# Couleurs RGB [0,1] pour chaque label
LABEL_COLORS_RGB = {
    LABEL_BG: np.array([0.00, 0.00, 0.00]), # noir background
    LABEL_TN: np.array([0.35, 0.35, 0.35]), # gris TN (foie)
    LABEL_TP: np.array([0.20, 0.85, 0.20]), # vert TP
    LABEL_FP: np.array([0.90, 0.15, 0.15]), # rouge FP
    LABEL_FN: np.array([1.00, 0.65, 0.00]), # orange FN
}

# Couleurs VTK (0–255) identiques pour le rendu 3D
LABEL_COLORS_VTK = {k: tuple(int(c * 255) for c in v)
                    for k, v in LABEL_COLORS_RGB.items()}

# Opacité VTK par label (TN semi-transparent comme le foie)
LABEL_OPACITY_VTK = {
    LABEL_BG: 0.00,
    LABEL_TN: 0.18, # foie translucide
    LABEL_TP: 1.00,
    LABEL_FP: 1.00,
    LABEL_FN: 1.00,
}

# Paramètres de rendu VTK
BG_COLOR = (0.10, 0.13, 0.20) # fond sombre marine
SMOOTH_ITER = 25
ISO_VALUE = 0.5
IMAGE_SIZE = (1400, 1000)
CAMERA_ELEV = 20
CAMERA_AZIM = -40

# UTILITAIRES NIfTI

def load_bin(path: Path) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Charge un NIfTI et renvoie (masque binaire uint8, affine, spacing)."""
    img = nib.load(str(path))
    data = (img.get_fdata(dtype=np.float32) > 0.5).astype(np.uint8)
    affine = img.affine
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return data, affine, spacing


def build_confusion_label(gt_b: np.ndarray, pred_b: np.ndarray,
                          liver_b: np.ndarray) -> np.ndarray:
    """
    Construit le label map de confusion spatiale (uint8).
    Priorité de masque : FP > FN > TP > TN > BG.
    """
    label = np.zeros(gt_b.shape, dtype=np.uint8) # BG partout

    in_liver = liver_b > 0

    label[in_liver & ~gt_b & ~pred_b] = LABEL_TN # TN
    label[in_liver & gt_b & pred_b] = LABEL_TP # TP
    label[in_liver & ~gt_b & pred_b] = LABEL_FP # FP (priorité haute)
    label[in_liver & gt_b & ~pred_b] = LABEL_FN # FN (priorité haute)

    return label


def label_to_rgb(label: np.ndarray) -> np.ndarray:
    
    rgb = np.zeros((*label.shape, 4), dtype=np.float32)
    for lbl, color in LABEL_COLORS_RGB.items():
        mask = label == lbl
        rgb[mask, :3] = color
        rgb[mask, 3] = 0.0 if lbl == LABEL_BG else 1.0
    return rgb


def save_nii_label(label: np.ndarray, affine: np.ndarray, path: Path):
    """Sauvegarde le label map entier (uint8)."""
    img = nib.Nifti1Image(label.astype(np.uint8), affine)
    img.header.set_data_dtype(np.uint8)
    nib.save(img, str(path))


def save_nii_rgb(rgb: np.ndarray, affine: np.ndarray, path: Path):
    """
    Sauvegarde le volume RGBA (X,Y,Z,4) en NIfTI float32.
    ImageJ/Fiji: File > Import > Bio-Formats → choisir "Split channels".
    """
    img = nib.Nifti1Image(rgb, affine)
    img.header.set_data_dtype(np.float32)
    nib.save(img, str(path))


def save_imagej_lut(path: Path):
    """
    Génère le fichier LUT binaire ImageJ (768 octets = 256×RGB)
    pour le label map (0=BG,1=TN,2=TP,3=FP,4=FN).
    Charger dans ImageJ : Image > Color > Load LUT.
    """
    r = np.zeros(256, dtype=np.uint8)
    g = np.zeros(256, dtype=np.uint8)
    b = np.zeros(256, dtype=np.uint8)

    color_map = {
        LABEL_BG: (0, 0, 0),
        LABEL_TN: (90, 90, 90),
        LABEL_TP: (51, 217, 51),
        LABEL_FP: (230, 38, 38),
        LABEL_FN: (255, 166, 0),
    }
    for label_val, (rv, gv, bv) in color_map.items():
        r[label_val] = rv
        g[label_val] = gv
        b[label_val] = bv

    lut_bytes = np.concatenate([r, g, b]).tobytes()
    path.write_bytes(lut_bytes)


# UTILITAIRES VTK

def numpy_to_vtk_image(arr: np.ndarray, spacing: tuple) -> vtk.vtkImageData:
    """Tableau numpy float32 → vtkImageData."""
    arr_f = arr.astype(np.float32)
    vtk_data = numpy_support.numpy_to_vtk(arr_f.ravel(order='F'), deep=True,
                                          array_type=vtk.VTK_FLOAT)
    image = vtk.vtkImageData()
    image.SetDimensions(arr.shape)
    image.SetSpacing(spacing)
    image.SetOrigin(0, 0, 0)
    image.GetPointData().SetScalars(vtk_data)
    return image


def make_surface_actor(vtk_image: vtk.vtkImageData,
                       color_rgb: tuple,
                       opacity: float,
                       smooth_iter: int = SMOOTH_ITER) -> vtk.vtkActor | None:
    """Marching Cubes + lissage → vtkActor. Retourne None si surface vide."""
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_image)
    mc.SetValue(0, ISO_VALUE)
    mc.ComputeNormalsOn()
    mc.Update()

    if mc.GetOutput().GetNumberOfPoints() == 0:
        return None

    if smooth_iter > 0:
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(mc.GetOutputPort())
        smoother.SetNumberOfIterations(smooth_iter)
        smoother.SetRelaxationFactor(0.15)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOn()
        smoother.Update()
        poly_port = smoother.GetOutputPort()
    else:
        poly_port = mc.GetOutputPort()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(poly_port)
    normals.SetFeatureAngle(60.0)
    normals.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    r, g, b = (c / 255.0 for c in color_rgb)
    actor.GetProperty().SetColor(r, g, b)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetSpecular(0.25)
    actor.GetProperty().SetSpecularPower(25)
    actor.GetProperty().SetAmbient(0.25)
    actor.GetProperty().SetDiffuse(0.75)
    return actor


def setup_renderer() -> vtk.vtkRenderer:
    """Renderer avec fond sombre et deux lumières."""
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(*BG_COLOR)

    for pos, intensity, color in [
        ((250, 250, 350), 0.85, (1.0, 1.0, 1.0)),
        ((-200, -150, 150), 0.40, (0.7, 0.8, 1.0)),
    ]:
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(*pos)
        light.SetFocalPoint(0, 0, 0)
        light.SetIntensity(intensity)
        light.SetColor(*color)
        renderer.AddLight(light)

    return renderer


def add_legend_actors(renderer: vtk.vtkRenderer):
    """
    Ajoute une légende textuelle TP/FP/FN/TN dans le coin supérieur gauche
    via vtkTextActor (rendu en 2D overlay sur la scène 3D).
    """
    entries = [
        ("TP : vrai positif", LABEL_COLORS_VTK[LABEL_TP]),
        ("FP : faux positif", LABEL_COLORS_VTK[LABEL_FP]),
        ("FN : faux négatif", LABEL_COLORS_VTK[LABEL_FN]),
        ("TN : foie (fond)", LABEL_COLORS_VTK[LABEL_TN]),
    ]
    y_start = IMAGE_SIZE[1] - 40
    for i, (label_text, (r, g, b)) in enumerate(entries):
        actor = vtk.vtkTextActor()
        actor.SetInput(f"■ {label_text}")
        actor.GetTextProperty().SetFontSize(20)
        actor.GetTextProperty().SetColor(r / 255, g / 255, b / 255)
        actor.GetTextProperty().SetBold(1)
        actor.GetTextProperty().ShadowOn()
        actor.GetTextProperty().SetFontFamilyToArial()
        actor.SetPosition(20, y_start - i * 30)
        renderer.AddActor2D(actor)


def render_to_png(renderer: vtk.vtkRenderer, filepath: Path):
    """Rendu offscreen → PNG."""
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetSize(*IMAGE_SIZE)
    win.AddRenderer(renderer)

    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam.Elevation(CAMERA_ELEV)
    cam.Azimuth(CAMERA_AZIM)
    cam.Dolly(1.15)
    renderer.ResetCameraClippingRange()
    win.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(win)
    w2i.ReadFrontBufferOff()
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(filepath))
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    win.Finalize()


# PIPELINE PATIENT × OPÉRATEUR

def load_patient(patient_id: str):
    """
    Charge GT, masque foie et résultats pickle pour un patient.
    Retourne (gt_bin, affine, spacing, liver_bin, results_dict).
    """
    label_file = LABELS_DIR / f"patient_{patient_id}_label.nii.gz"
    gt_bin, affine, spacing = load_bin(label_file)

    liver_file = LIVER_DIR / f"patient_{patient_id}_liver.nii.gz"
    if liver_file.exists():
        liver_bin, _, _ = load_bin(liver_file)
    else:
        struct = generate_binary_structure(3, 1)
        liver_bin = binary_dilation(gt_bin > 0, structure=struct,
                                    iterations=30).astype(np.uint8)

    result_file = RESULTS_DIR / f"results_patient_{patient_id}_images.nii"
    with open(result_file, 'rb') as f:
        results = pickle.load(f)

    return gt_bin, affine, spacing, liver_bin, results


def get_pred_bin(results: dict, op_name: str) -> np.ndarray | None:
    """Extrait la segmentation binaire d'un opérateur depuis le pickle."""
    derivators = results.get('derivator', results)
    if op_name not in derivators:
        return None
    op_data = derivators[op_name]
    seg = getattr(op_data, 'data_segmented',
          getattr(op_data, 'segmented', None))
    return None if seg is None else (seg > 0.5).astype(np.uint8)


def process_operator(patient_id: str, op_name: str,
                     gt_bin: np.ndarray, pred_bin: np.ndarray,
                     liver_bin: np.ndarray,
                     affine: np.ndarray, spacing: tuple,
                     out_dir: Path):
    """
    Pour un patient × opérateur :
    1. Calcule le label map de confusion (TP/FP/FN/TN)
    2. Exporte NIfTI label + NIfTI RGB (ImageJ)
    3. Rendu VTK 3D PNG avec légende
    """
    prefix = out_dir / f"patient_{patient_id}_{op_name}"

    # ── Label map ────────────────────────────────────────────────────────────
    label = build_confusion_label(gt_bin > 0, pred_bin > 0, liver_bin > 0)

    save_nii_label(label, affine, Path(str(prefix) + "_confusion_label.nii.gz"))
    save_nii_rgb(label_to_rgb(label), affine,
                 Path(str(prefix) + "_confusion_rgb.nii.gz"))

    # ── Statistiques rapides ─────────────────────────────────────────────────
    tp = int((label == LABEL_TP).sum())
    fp = int((label == LABEL_FP).sum())
    fn = int((label == LABEL_FN).sum())
    tn = int((label == LABEL_TN).sum())
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    print(f"TP={tp:>7,} FP={fp:>7,} FN={fn:>7,} TN={tn:>8,}"
          f"Dice={dice:.4f}")

    # ── Rendu VTK ────────────────────────────────────────────────────────────
    renderer = setup_renderer()

    # Ajouter une surface par label (sauf background)
    for lbl in [LABEL_TN, LABEL_TP, LABEL_FP, LABEL_FN]:
        mask = (label == lbl).astype(np.float32)
        if mask.sum() == 0:
            continue
        vtk_img = numpy_to_vtk_image(mask, spacing)
        actor = make_surface_actor(
            vtk_img,
            color_rgb=LABEL_COLORS_VTK[lbl],
            opacity=LABEL_OPACITY_VTK[lbl],
        )
        if actor is not None:
            renderer.AddActor(actor)

    add_legend_actors(renderer)

    # Titre opérateur (bas de l'image)
    title_actor = vtk.vtkTextActor()
    title_actor.SetInput(f"Patient {patient_id} - {op_name} "
                         f"Dice={dice:.4f} TP={tp:,} FP={fp:,} FN={fn:,}")
    title_actor.GetTextProperty().SetFontSize(18)
    title_actor.GetTextProperty().SetColor(0.85, 0.90, 1.0)
    title_actor.GetTextProperty().SetBold(0)
    title_actor.GetTextProperty().ShadowOn()
    title_actor.SetPosition(20, 12)
    renderer.AddActor2D(title_actor)

    png_path = Path(str(prefix) + "_confusion_overlay.png")
    render_to_png(renderer, png_path)
    print(f"PNG → {png_path.name}")


def process_patient(patient_id: str):
    print(f"\n{'─'*60}")
    print(f"Patient {patient_id}")
    print(f"{'─'*60}")

    out_dir = OUTPUT_DIR / f"patient_{patient_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_bin, affine, spacing, liver_bin, results = load_patient(patient_id)
    print(f"Shape={gt_bin.shape} Spacing={tuple(f'{s:.2f}'for s in spacing)} mm"
          f"GT={int(gt_bin.sum()):,} voxels")

    for op_name in OPERATORS:
        pred_bin = get_pred_bin(results, op_name)
        if pred_bin is None:
            print(f"[{op_name}] données manquantes : ignoré")
            continue
        # Alignement dimensionnel (sécurité)
        s = tuple(min(a, b) for a, b in zip(gt_bin.shape, pred_bin.shape))
        print(f"[{op_name}]")
        process_operator(
            patient_id, op_name,
            gt_bin[:s[0], :s[1], :s[2]],
            pred_bin[:s[0], :s[1], :s[2]],
            liver_bin[:s[0], :s[1], :s[2]],
            affine, spacing, out_dir,
        )


# LUT IMAGEJ générée une seule fois

def ensure_lut():
    """Génère la LUT ImageJ dans le dossier de sortie si absente."""
    lut_path = OUTPUT_DIR / "confusion_cm.lut"
    if not lut_path.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_imagej_lut(lut_path)
        print(f"LUT ImageJ générée : {lut_path}")
        print("→ ImageJ : Image > Color > Load LUT → sélectionner confusion_cm.lut")


# POINT D'ENTRÉE

if __name__ == "__main__":
    patient_list = sys.argv[1:] if len(sys.argv) > 1 else PATIENTS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="* 60)
    print("CONFUSION 3D - TP / FP / FN / TN par opérateur")
    print("="* 60)
    print(f"Patients : {patient_list}")
    print(f"Opérateurs : {OPERATORS}")
    print(f"Sortie : {OUTPUT_DIR}")
    print("="* 60)

    ensure_lut()

    ok, err = 0, 0
    for pid in patient_list:
        try:
            process_patient(pid)
            ok += 1
        except Exception as e:
            import traceback
            print(f"\n ERREUR patient {pid}: {e}")
            traceback.print_exc()
            err += 1

    print("\n"+ "="* 60)
    print(f"Succès : {ok} Erreurs : {err}")
    print(f"Résultats : {OUTPUT_DIR.resolve()}")
    print()
    print("Ouvrir dans ImageJ/Fiji :")
    print("• _confusion_label.nii.gz → Image > Color > Load LUT → confusion_cm.lut")
    print("• _confusion_rgb.nii.gz → Plugins > Bio-Formats > Import (RGB stack)")
    print("="* 60)