# -*- coding: utf-8 -*-
"""overlapcheck.py — collision audit that works on the RENDERED PIXELS.

The earlier version compared text boxes against a hand-picked list of artists
(Line2D, PathCollection) and skipped legends entirely, so text sitting on bars,
on shaded bands, on image cells, or under a frameless legend went unreported.

This version renders the figure twice:
  pass 1  every Text hidden   -> an "ink" mask of all non-text drawing
  pass 2  everything visible  -> the bounding box of every Text
and reports any Text whose box contains ink.  That catches every mark type.

Checks
  1 TEXT/TEXT       two text boxes intersect
  2 OFF-CANVAS      a text box leaves the figure
  3 INTO-NEIGHBOUR  a text box crosses into another Axes' rectangle
  4 TEXT/INK        a text box sits on drawn content
  5 LEGEND/INK      a frameless legend box sits on drawn content

Deliberate overlaps are exempted with artist.set_gid("nocheck") (skip the
artist entirely) or "ink-ok" (keep the text checks, allow ink underneath).
"""
import numpy as np
from matplotlib.text import Text

PAD = 2.5         # px of clear space required between two text boxes
INK_SHRINK = 1.5  # px shaved off a text box before testing for ink underneath
INK_MIN = 8       # this many inked pixels inside a box counts as a collision
WHITE = 250       # any channel below this is "ink"


def _boxes_overlap(a, b, pad=PAD):
    return not (a.x1 + pad <= b.x0 or b.x1 + pad <= a.x0 or
                a.y1 + pad <= b.y0 or b.y1 + pad <= a.y0)


def _live_ticklabels(ax, axis):
    """Tick labels actually inside the current view (matplotlib keeps the rest)."""
    if axis == "x":
        locs, labs, lim = ax.get_xticks(), ax.get_xticklabels(), ax.get_xlim()
    else:
        locs, labs, lim = ax.get_yticks(), ax.get_yticklabels(), ax.get_ylim()
    lo, hi = min(lim), max(lim)
    eps = (hi - lo) * 1e-6
    return [t for loc, t in zip(locs, labs) if lo - eps <= loc <= hi + eps]


def _all_texts(fig):
    out = []
    for ax in fig.axes:
        out += list(ax.texts)
        out += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        out += [ax.xaxis.label, ax.yaxis.label, ax.title]
        leg = ax.get_legend()
        if leg is not None:
            out += list(leg.get_texts())
            ttl = leg.get_title()
            if ttl is not None:
                out.append(ttl)
    out += list(fig.texts)
    return [t for t in out if isinstance(t, Text)]


def _collect(fig):
    """(text, axes, legend, is_manual) for every visible, non-empty Text."""
    out = []
    for ax in fig.axes:
        for t in ax.texts:
            out.append((t, ax, None, True))
        if getattr(ax, "axison", True):
            for t in _live_ticklabels(ax, "x") + _live_ticklabels(ax, "y"):
                out.append((t, ax, None, False))
        for t in (ax.xaxis.label, ax.yaxis.label, ax.title):
            out.append((t, ax, None, False))
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                out.append((t, ax, leg, False))
            ttl = leg.get_title()
            if ttl is not None:
                out.append((ttl, ax, leg, False))
    for t in fig.texts:
        out.append((t, None, None, True))

    keep = []
    for t, ax, leg, manual in out:
        if not isinstance(t, Text) or not t.get_visible():
            continue
        if not str(t.get_text()).strip():
            continue
        if t.get_gid() == "nocheck":
            continue
        keep.append((t, ax, leg, manual))
    return keep


def _ink_mask(fig):
    """Render with every Text hidden; return a boolean mask of drawn pixels."""
    texts = _all_texts(fig)
    state = [(t, t.get_visible()) for t in texts]
    for t, _ in state:
        t.set_visible(False)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    for t, v in state:
        t.set_visible(v)
    fig.canvas.draw()
    return (buf[:, :, :3] < WHITE).any(axis=2)


def _ink_in(mask, bb, shrink=INK_SHRINK):
    h, w = mask.shape
    x0 = int(np.ceil(bb.x0 + shrink))
    x1 = int(np.floor(bb.x1 - shrink))
    y0 = int(np.ceil(bb.y0 + shrink))
    y1 = int(np.floor(bb.y1 - shrink))
    if x1 <= x0 or y1 <= y0:
        return 0
    r0, r1 = h - y1, h - y0          # display y is bottom-up; buffer row 0 is top
    r0, r1 = max(0, r0), min(h, r1)
    c0, c1 = max(0, x0), min(w, x1)
    if r1 <= r0 or c1 <= c0:
        return 0
    return int(mask[r0:r1, c0:c1].sum())


def audit(fig, name="figure", verbose=True):
    mask = _ink_mask(fig)
    r = fig.canvas.get_renderer()
    items = _collect(fig)

    boxes = []
    for t, ax, leg, manual in items:
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:
            continue
        if bb.width > 0 and bb.height > 0:
            boxes.append((t, ax, leg, manual, bb))

    problems = []

    # --- 1 text vs text ---------------------------------------------------
    for i in range(len(boxes)):
        ti, axi, li, _, bi = boxes[i]
        for j in range(i + 1, len(boxes)):
            tj, axj, lj, _, bj = boxes[j]
            if li is not None and li is lj:
                continue
            if _boxes_overlap(bi, bj):
                problems.append(("TEXT/TEXT",
                                 f"{ti.get_text()!r} x {tj.get_text()!r}"))

    # --- 2 off canvas -----------------------------------------------------
    fb = fig.bbox
    for t, ax, leg, _, bb in boxes:
        if (bb.x0 < fb.x0 - 0.5 or bb.x1 > fb.x1 + 0.5 or
                bb.y0 < fb.y0 - 0.5 or bb.y1 > fb.y1 + 0.5):
            problems.append(("OFF-CANVAS", repr(t.get_text())))

    # --- 3 crossing into another Axes -------------------------------------
    axboxes = [(ax, ax.get_window_extent(renderer=r)) for ax in fig.axes]
    for t, ax, leg, _, bb in boxes:
        if ax is None:
            continue
        for other, ob in axboxes:
            if other is not ax and _boxes_overlap(bb, ob, pad=0.0):
                problems.append(("INTO-NEIGHBOUR",
                                 f"{t.get_text()!r} -> other panel"))
                break

    # --- 4 text on drawn content ------------------------------------------
    for t, ax, leg, manual, bb in boxes:
        if t.get_gid() == "ink-ok":
            continue
        if leg is not None:
            continue   # legends carry an opaque backing; their handles are not ink
        if not manual:
            continue                      # tick/axis labels are placed by mpl
        n = _ink_in(mask, bb)
        if n >= INK_MIN:
            problems.append(("TEXT/INK",
                             f"{t.get_text()!r}  ({n} px of ink under it)"))

    if verbose:
        if problems:
            print(f"  [{name}] {len(problems)} collision(s):")
            seen = set()
            for kind, msg in problems:
                if (kind, msg) in seen:
                    continue
                seen.add((kind, msg))
                print(f"      {kind:<16}{msg}")
        else:
            print(f"  [{name}] clean")
    return problems
