"""Geometrie und Stil der Paper-Abbildungen (ESANN, esannV2.cls; Festlegung F27).

Bilder werden in Endgröße erzeugt: Breite = Anteil der Textbreite, Höhe frei in cm, Schrift 8 pt Serif im
Computer-Modern-Stil (cmr10 + mathtext cm), Linien ≈ 1 pt. Ausgabe PDF (Vektor, Schriften eingebettet) + PNG 300 dpi.
"""
import io
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[2]
CLS = ROOT / "papers" / "p2_overleaf" / "esannV2.cls"
TEXTWIDTH_CM = 12.2          # esannV2.cls: \textwidth 12.2 cm
TEXTHEIGHT_CM = 19.2         # esannV2.cls: \textheight 19.2 cm
A4_CM = (21.0, 29.7)
CM = 1 / 2.54
OUT = ROOT / "results" / "public" / "paper_figs"
MIN_FONT_PT = 7

# Farben zentral (G/CW/Identität fest; Orakel schwarz gestrichelt)
COL = {"G": "#2a6fdb", "CW": "#e8642c", "ident": "#6b6b6b", "oracle": "#000000", "ink": "#1c2836",
       "mute": "#6b6b6b", "grid": "#e4e3dd", "start": "#dcd9cf", "proto": "#8a9aa6", "harm": "#c0392b",
       "band": "#2a9d6f"}
# Lernraten als Abstufung der Ebenenfarbe (hell = kleines eta); abnehmender Plan gestrichelt in Tinte
ETA_COL = {"CW": ["#f6c39f", "#ee9a63", "#e8642c", "#b04a18"], "G": ["#a9c4f2", "#6d9be8", "#2a6fdb", "#1a4a9c"]}
DECAY_COL = COL["ink"]


def read_cls_geometry(path=CLS):
    """Liest \\textwidth und \\textheight aus esannV2.cls (cm); None, wenn die Datei fehlt."""
    path = Path(path)
    if not path.exists():
        return None
    s = path.read_text(encoding="latin-1")
    get = lambda key: float(re.search(rf"\\{key}\s+([0-9.]+)\s*cm", s).group(1))
    return dict(textwidth_cm=get("textwidth"), textheight_cm=get("textheight"))


def check_geometry():
    """Prüft die Konstanten gegen esannV2.cls; gibt eine Zeile aus."""
    g = read_cls_geometry()
    if g is None:
        print(f"esannV2.cls nicht gefunden ({CLS}); Konstanten {TEXTWIDTH_CM} × {TEXTHEIGHT_CM} cm ungeprüft")
        return False
    ok = abs(g["textwidth_cm"] - TEXTWIDTH_CM) < 1e-9 and abs(g["textheight_cm"] - TEXTHEIGHT_CM) < 1e-9
    print(f"esannV2.cls: Textbreite {g['textwidth_cm']} cm, Texthöhe {g['textheight_cm']} cm → "
          f"{'gleich den Konstanten' if ok else 'ABWEICHUNG von den Konstanten'}")
    return ok


def paper_style(font_pt=8, lang="en"):
    """rcParams für Paper-Bilder: Serif (cmr10), mathtext cm, dünne Achsen, keine oberen/rechten Achsen."""
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "axes.unicode_minus": True,
        "font.size": font_pt, "axes.labelsize": font_pt, "axes.titlesize": font_pt, "legend.fontsize": font_pt,
        "xtick.labelsize": font_pt, "ytick.labelsize": font_pt,
        "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.5, "patch.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.labelpad": 2, "axes.titlepad": 3,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#333333",
        "axes.labelcolor": COL["ink"], "xtick.color": COL["ink"], "ytick.color": COL["ink"],
        "axes.grid": False, "grid.color": COL["grid"], "grid.linewidth": 0.4,
        "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2, "legend.borderaxespad": 0.2,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 100, "savefig.dpi": 300,
        "hatch.linewidth": 0.6,
    })
    if lang == "de":                                       # cmr10 hat keine Umlaute -> fuer Folien DejaVu Serif
        matplotlib.rcParams["font.serif"] = ["DejaVu Serif"]
    return dict(font_pt=font_pt, lang=lang)


def figsize_cm(width_frac=1.0, height_cm=5.0):
    """(Breite, Höhe) in Zoll; Breite als Anteil der Textbreite."""
    return (TEXTWIDTH_CM * width_frac * CM, height_cm * CM)


def _fonts(fig):
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    return min(sizes) if sizes else None


def size_info(fig):
    w, h = fig.get_size_inches()
    return dict(breite_cm=round(w / CM, 2), hoehe_cm=round(h / CM, 2), anteil_textbreite=round(w / CM / TEXTWIDTH_CM, 3),
                kleinste_schrift_pt=_fonts(fig))


def save(fig, name, out=None, quiet=False, formats=("pdf", "png")):
    """PDF (Vektor, Schriften eingebettet) + PNG 300 dpi nach out (Standard results/public/paper_figs/); zusätzlich
    'eps' möglich (falls Overleaf mit dvips kompiliert; EPS kennt keine Transparenz).
    Gibt Breite × Höhe in cm und kleinste Schrift aus; Warnung unter 7 pt."""
    out = Path(out) if out is not None else OUT
    out.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out / f"{name}.{fmt}", **({"dpi": 300} if fmt == "png" else {}))
    info = size_info(fig)
    if not quiet:
        warn = (f"  WARNUNG: kleinste Schrift {info['kleinste_schrift_pt']} pt < {MIN_FONT_PT} pt"
                if info["kleinste_schrift_pt"] and info["kleinste_schrift_pt"] < MIN_FONT_PT else "")
        print(f"{name}: {info['breite_cm']} × {info['hoehe_cm']} cm ({info['anteil_textbreite']:.0%} der Textbreite), "
              f"kleinste Schrift {info['kleinste_schrift_pt']} pt → {out / name}.{'/.'.join(formats)}{warn}")
    return info


def _png_bytes(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    return buf.getvalue()


def show_true_size(fig, screen_dpi=96):
    """PNG-Vorschau im Notebook in echter Größe (cm am Bildschirm ≈ cm im PDF bei 100 % Zoom; screen_dpi anpassen)."""
    w_in = fig.get_size_inches()[0]
    try:
        from IPython.display import Image, display
    except ImportError:                                    # ausserhalb von Jupyter (Testlauf per Skript)
        print(f"(Vorschau nur im Notebook; Breite {w_in / CM:.2f} cm)")
        return None
    img = Image(data=_png_bytes(fig), width=int(round(w_in * screen_dpi)))
    display(img)
    return img


def page_preview(figs, text_lines=None, gap_cm=0.6, name=None, out=None):
    """A4-Seite mit Satzspiegel 12.2 × 19.2 cm; die Bilder in ihrer Höhe von oben eingesetzt, Rest grau als Textfläche.
    figs: Liste von Figures (oder (Figure, Beschriftung)). text_lines: optionale Zeilen als Notiz unten.
    Gibt 'Bild belegt x % der Seite' (des Satzspiegels) aus und liefert die Vorschau-Figure."""
    from matplotlib.patches import Rectangle
    import matplotlib.image as mpimg
    pw, ph = A4_CM
    x0, y_top = (pw - TEXTWIDTH_CM) / 2, (ph + TEXTHEIGHT_CM) / 2
    pv = plt.figure(figsize=(pw * CM, ph * CM))
    ax = pv.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, pw); ax.set_ylim(0, ph); ax.axis("off")
    ax.add_patch(Rectangle((x0, y_top - TEXTHEIGHT_CM), TEXTWIDTH_CM, TEXTHEIGHT_CM, color="#d9d9d9", lw=0))
    y, used = y_top, 0.0
    for item in figs:
        f, lab = item if isinstance(item, tuple) else (item, None)
        w_cm, h_cm = (v / CM for v in f.get_size_inches())
        if y - h_cm < y_top - TEXTHEIGHT_CM:
            print(f"passt nicht mehr auf die Seite: {lab or 'Bild'} ({h_cm:.1f} cm)")
            break
        img = mpimg.imread(io.BytesIO(_png_bytes(f, dpi=150)), format="png")
        xl = x0 + (TEXTWIDTH_CM - w_cm) / 2
        ax.add_patch(Rectangle((x0, y - h_cm - gap_cm / 2), TEXTWIDTH_CM, h_cm + gap_cm, color="white", lw=0))
        ax.imshow(img, extent=(xl, xl + w_cm, y - h_cm, y), zorder=3)
        used += h_cm + gap_cm
        y -= h_cm + gap_cm
    ax.add_patch(Rectangle((x0, y_top - TEXTHEIGHT_CM), TEXTWIDTH_CM, TEXTHEIGHT_CM, fill=False, ec="#888888", lw=0.5,
                           zorder=4))
    share = used / TEXTHEIGHT_CM
    ax.text(pw / 2, 1.2, f"Bilder (inkl. {gap_cm} cm Abstand je Bild) belegen {used:.1f} cm = {share:.0%} der Texthöhe",
            ha="center", fontsize=8, family="DejaVu Sans")
    if text_lines:
        ax.text(x0, (ph - TEXTHEIGHT_CM) / 2 - 0.5, "\n".join(text_lines), fontsize=7, va="top",
                family="DejaVu Sans")
    print(f"Bild belegt {share:.0%} der Seite (Satzspiegel {TEXTWIDTH_CM} × {TEXTHEIGHT_CM} cm; {used:.1f} cm Höhe)")
    if name:
        save(pv, name, out=out, quiet=True)
    return pv, share
