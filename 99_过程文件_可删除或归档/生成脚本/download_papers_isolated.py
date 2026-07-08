from pathlib import Path
import multiprocessing as mp
import urllib.request

ROOT = Path(__file__).resolve().parent
PAPERS_DIR = ROOT / "papers"
PAPERS_DIR.mkdir(exist_ok=True)

items = [
    ("EVER_Exact_Volumetric_Ellipsoid_Rendering.pdf", "https://arxiv.org/pdf/2410.01804"),
    ("StochasticSplats_Sorting_Free_3DGS.pdf", "https://arxiv.org/pdf/2503.24366"),
    ("3DGS_LM_Faster_Gaussian_Splatting_Optimization.pdf", "https://arxiv.org/pdf/2409.12892"),
    ("RayGaussX_Accelerating_Gaussian_Based_Ray_Marching.pdf", "https://arxiv.org/pdf/2509.07782"),
    ("RegGS_Unposed_Sparse_Views_Gaussian_Splatting.pdf", "https://arxiv.org/pdf/2507.08136"),
    ("ResGS_Residual_Densification_of_3D_Gaussian.pdf", "https://arxiv.org/pdf/2412.07494"),
]


def worker(filename, url, queue):
    target = PAPERS_DIR / filename
    try:
        if target.exists() and target.stat().st_size > 50_000:
            queue.put(("EXISTS", filename, target.stat().st_size, str(target)))
            return
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0"), ("Accept", "application/pdf,*/*")]
        with opener.open(url, timeout=20) as resp:
            data = resp.read()
        if len(data) < 50_000:
            queue.put(("FAIL_SMALL", filename, len(data), url))
            return
        target.write_bytes(data)
        queue.put(("OK", filename, len(data), str(target)))
    except Exception as e:
        queue.put(("FAIL", filename, f"{type(e).__name__}: {e}", url))


def main():
    lines = []
    for filename, url in items:
        q = mp.Queue()
        p = mp.Process(target=worker, args=(filename, url, q))
        p.start()
        p.join(45)
        if p.is_alive():
            p.terminate()
            p.join(5)
            row = ("TIMEOUT", filename, "超过45秒", url)
        else:
            row = q.get() if not q.empty() else ("FAIL", filename, "无返回", url)
        line = "\t".join(map(str, row))
        print(line, flush=True)
        lines.append(line)
    (ROOT / "papers_download_report_isolated.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
