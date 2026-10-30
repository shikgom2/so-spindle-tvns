# -*- coding: utf-8 -*-
"""Python port of the dataviz six-checks palette validator.

Checks, per the skill:
  1 lightness band      OKLab L within [0.45, 0.75] for light-mode categorical
  2 chroma floor        OKLab C >= 0.06
  3 CVD separation      adjacent-pair dE >= 8 (OKLab x100) under deut/prot/trit
  4 normal-vision floor adjacent-pair dE >= 15 in normal vision  (hard fail)
  5 contrast            >= 3.0 : 1 against the chart surface
Usage: python validate_palette.py "#aaa,#bbb" [--surface #FFFFFF] [--pairs all]
"""
import sys
import numpy as np

# ----------------------------------------------------------------- sRGB <-> OKLab
def hex2rgb(h):
    h = h.strip().lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def srgb2lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
               [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]])
M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
               [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]])


def lin2oklab(rgb_lin):
    lms = M1 @ rgb_lin
    return M2 @ np.cbrt(lms)


def oklab(hexstr):
    return lin2oklab(srgb2lin(hex2rgb(hexstr)))


# ------------------------------------------------------- CVD (Viénot 1999, LMS)
RGB2LMS = np.array([[17.8824, 43.5161, 4.11935],
                    [3.45565, 27.1554, 3.86714],
                    [0.0299566, 0.184309, 1.46709]])
LMS2RGB = np.linalg.inv(RGB2LMS)
SIM = {
    "deuteranopia": np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]]),
    "protanopia":   np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]]),
    "tritanopia":   np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]]),
}


def simulate(hexstr, kind):
    lin = srgb2lin(hex2rgb(hexstr))
    lms = RGB2LMS @ lin
    out = LMS2RGB @ (SIM[kind] @ lms)
    return lin2oklab(np.clip(out, 0, 1))


def de(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) * 100)


def rel_lum(hexstr):
    r, g, b = srgb2lin(hex2rgb(hexstr))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(h1, h2):
    a, b = sorted([rel_lum(h1), rel_lum(h2)], reverse=True)
    return (a + 0.05) / (b + 0.05)


def main():
    pal = [c.strip() for c in sys.argv[1].split(",") if c.strip()]
    surface = "#FFFFFF"
    all_pairs = "--pairs" in sys.argv and "all" in sys.argv
    if "--surface" in sys.argv:
        surface = sys.argv[sys.argv.index("--surface") + 1]
    if "--mode" in sys.argv and sys.argv[sys.argv.index("--mode") + 1] == "dark":
        surface = "#14181D"

    fails = 0
    print(f"palette: {pal}   surface {surface}\n")
    print(f"{'hex':<9}{'L':>7}{'C':>7}{'contrast':>10}  verdict")
    for h in pal:
        L, a, b = oklab(h)
        C = float(np.hypot(a, b))
        cr = contrast(h, surface)
        bad = []
        if not (0.45 <= L <= 0.75):
            bad.append("L band")
        if C < 0.06:
            bad.append("chroma")
        if cr < 3.0:
            bad.append("contrast")
        if bad:
            fails += 1
        print(f"{h:<9}{L:7.3f}{C:7.3f}{cr:10.2f}  "
              f"{'FAIL: ' + ', '.join(bad) if bad else 'pass'}")

    idx = ([(i, j) for i in range(len(pal)) for j in range(i + 1, len(pal))]
           if all_pairs else [(i, i + 1) for i in range(len(pal) - 1)])
    print(f"\n{'pair':<21}{'normal':>8}{'deut':>8}{'prot':>8}{'trit':>8}  verdict")
    for i, j in idx:
        n = de(oklab(pal[i]), oklab(pal[j]))
        cv = {k: de(simulate(pal[i], k), simulate(pal[j], k)) for k in SIM}
        bad = []
        if n < 15:
            bad.append("normal<15 HARD FAIL")
        worst = min(cv.values())
        if worst < 6:
            bad.append("CVD<6 FAIL")
        elif worst < 8:
            bad.append("CVD 6-8 needs 2nd encoding")
        if bad:
            fails += 1
        print(f"{pal[i]}/{pal[j]:<10}{n:8.1f}{cv['deuteranopia']:8.1f}"
              f"{cv['protanopia']:8.1f}{cv['tritanopia']:8.1f}  "
              f"{'; '.join(bad) if bad else 'pass'}")

    print(f"\n==> {'ALL PASS' if fails == 0 else str(fails) + ' PROBLEM(S)'}")


if __name__ == "__main__":
    main()
