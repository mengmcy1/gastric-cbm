from pathlib import Path
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent
PAPERS_DIR = ROOT / "papers"
PAPERS_DIR.mkdir(exist_ok=True)

items = [
    ("NeRF_Representing_Scenes_as_Neural_Radiance_Fields_for_View_Synthesis.pdf", "https://arxiv.org/pdf/2003.08934"),
    ("SLIDE_Single_Image_3D_Photo_with_Soft_Layering.pdf", "https://arxiv.org/pdf/2109.01068"),
    ("MINE_Towards_Continuous_Depth_MPI_with_NeRF.pdf", "https://arxiv.org/pdf/2103.14910"),
    ("Depth_Anything_V2.pdf", "https://arxiv.org/pdf/2406.09414"),
    ("Depth_Pro_Sharp_Monocular_Metric_Depth.pdf", "https://arxiv.org/pdf/2410.02073"),
    ("SHARP_Sharp_Monocular_View_Synthesis.pdf", "https://arxiv.org/pdf/2512.10685"),
    ("pixelSplat_3D_Gaussian_Splats_from_Image_Pairs.pdf", "https://arxiv.org/pdf/2312.12337"),
    ("AnySplat_Feed_forward_3D_Gaussian_Splatting.pdf", "https://arxiv.org/pdf/2505.23716"),
    ("AAA_Gaussians_Anti_Aliased_and_Artifact_Free.pdf", "https://arxiv.org/pdf/2504.12811"),
    ("EVER_Exact_Volumetric_Ellipsoid_Rendering.pdf", "https://arxiv.org/pdf/2410.01804"),
    ("StochasticSplats_Sorting_Free_3DGS.pdf", "https://arxiv.org/pdf/2503.24366"),
    ("3DGS_LM_Faster_Gaussian_Splatting_Optimization.pdf", "https://arxiv.org/pdf/2409.12892"),
    ("RayGaussX_Accelerating_Gaussian_Based_Ray_Marching.pdf", "https://arxiv.org/pdf/2509.07782"),
    ("RegGS_Unposed_Sparse_Views_Gaussian_Splatting.pdf", "https://arxiv.org/pdf/2507.08136"),
    ("ResGS_Residual_Densification_of_3D_Gaussian.pdf", "https://arxiv.org/pdf/2412.07494"),
]

opener = urllib.request.build_opener()
opener.addheaders = [
    ("User-Agent", "Mozilla/5.0"),
    ("Accept", "application/pdf,*/*"),
]

report_lines = []
for filename, url in items:
    target = PAPERS_DIR / filename
    if target.exists() and target.stat().st_size > 50_000:
        msg = f"EXISTS\t{filename}\t{target.stat().st_size}"
        print(msg, flush=True)
        report_lines.append(msg)
        continue
    try:
        with opener.open(url, timeout=20) as resp:
            data = resp.read()
        if len(data) < 50_000:
            msg = f"FAIL_SMALL\t{filename}\t{len(data)}\t{url}"
            print(msg, flush=True)
            report_lines.append(msg)
            continue
        target.write_bytes(data)
        msg = f"OK\t{filename}\t{len(data)}"
        print(msg, flush=True)
        report_lines.append(msg)
    except Exception as e:
        msg = f"FAIL\t{filename}\t{type(e).__name__}: {e}\t{url}"
        print(msg, flush=True)
        report_lines.append(msg)

(ROOT / "papers_download_report_remaining.txt").write_text("\n".join(report_lines), encoding="utf-8")
