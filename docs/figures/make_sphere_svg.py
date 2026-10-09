#!/usr/bin/env python3
"""Generate the two sphere figures of docs/explicacao.html by real orthographic projection.

Figures 1 (ERP cell vs spherical cell) and 4 (tangent plane / exp / log) used to be
hand-drawn SVG: the "spherical" patches were arbitrary arcs and the tangent plane was a
fixed parallelogram that did not actually touch the sphere. Here every point is a unit
vector projected with an orthographic camera, so the patches lie on the sphere and the
plane is tangent by construction.

Run:  python3 docs/figures/make_sphere_svg.py
It rewrites the two <svg> blocks in docs/explicacao.html in place (matched by aria-label)
and also dumps standalone copies next to this script.
"""

import re
from pathlib import Path

import numpy as np

D2R = np.pi / 180.0
HERE = Path(__file__).resolve().parent
HTML = HERE.parent / "explicacao.html"


# ---------------------------------------------------------------- geometry --


def sph2vec(lon, lat):
    """(lon, lat) in degrees -> unit vectors, z up, x at lon=0."""
    lon = np.asarray(lon, float) * D2R
    lat = np.asarray(lat, float) * D2R
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1
    )


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class Ortho:
    """Orthographic camera looking at the sphere origin from direction `d`."""

    def __init__(self, cam_lon, cam_lat, cx, cy, R):
        self.d = sph2vec(cam_lon, cam_lat)
        right = np.cross(np.array([0.0, 0.0, 1.0]), self.d)
        n = np.linalg.norm(right)
        self.right = right / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        self.up = np.cross(self.d, self.right)
        self.cx, self.cy, self.R = cx, cy, R

    def xy(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        # SVG y grows downward, hence the sign flip on `up`.
        return np.stack(
            [self.cx + self.R * (P @ self.right), self.cy - self.R * (P @ self.up)], -1
        )

    def depth(self, P):
        return np.atleast_2d(np.asarray(P, float)) @ self.d


# ------------------------------------------------------------------- paths --


def n(v):
    s = f"{float(v):.2f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def poly(pts, close=False):
    d = "M" + " L".join(f"{n(x)} {n(y)}" for x, y in pts)
    return d + " Z" if close else d


def front_path(cam, P):
    """Polyline on the sphere, clipped to the visible hemisphere at the terminator."""
    P = np.asarray(P, float)
    s = cam.depth(P)
    runs, cur = [], []
    for i in range(len(P)):
        if s[i] > 0:
            cur.append(P[i])
            if i + 1 < len(P) and s[i + 1] <= 0:  # exit crossing
                t = s[i] / (s[i] - s[i + 1])
                cur.append(unit(P[i] + t * (P[i + 1] - P[i])))
                runs.append(cur)
                cur = []
        else:
            if i + 1 < len(P) and s[i + 1] > 0:  # entry crossing
                t = s[i] / (s[i] - s[i + 1])
                cur = [unit(P[i] + t * (P[i + 1] - P[i]))]
    if cur:
        runs.append(cur)
    return " ".join(poly(cam.xy(np.array(r))) for r in runs if len(r) > 1)


def cell_edge(lon0, lon1, lat0, lat1, k=28):
    """Boundary of a lon/lat cell sampled along its four edges (true spherical quad)."""
    t = np.linspace(0, 1, k)
    lons = np.concatenate(
        [
            lon0 + t * (lon1 - lon0),
            np.full(k, lon1),
            lon1 + t * (lon0 - lon1),
            np.full(k, lon0),
        ]
    )
    lats = np.concatenate(
        [
            np.full(k, lat0),
            lat0 + t * (lat1 - lat0),
            np.full(k, lat1),
            lat1 + t * (lat0 - lat1),
        ]
    )
    return sph2vec(lons, lats)


def inside(pts, ring):
    """Even-odd point-in-polygon, vectorised over `pts`."""
    x, y = pts[:, 0], pts[:, 1]
    ax, ay = ring[:, 0], ring[:, 1]
    bx, by = np.roll(ax, -1), np.roll(ay, -1)
    hit = np.zeros(len(pts), bool)
    for i in range(len(ring)):
        crosses = (ay[i] > y) != (by[i] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (bx[i] - ax[i]) * (y - ay[i]) / (by[i] - ay[i]) + ax[i]
        hit ^= crosses & (x < xint)
    return hit


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, cls="tx-s", anchor=None, style=None):
    a = f' text-anchor="{anchor}"' if anchor else ""
    st = f' style="{style}"' if style else ""
    return f'<text class="{cls}" x="{n(x)}" y="{n(y)}"{a}{st}>{esc(s)}</text>'


# ------------------------------------------------------------------ fig. 1 --


def figure1():
    W, H = 720, 288
    X0, Y0, BW, BH = 8, 28, 352, 176  # ERP panel
    CX, CY, R = 534, 118, 92  # sphere
    LON0, LON1 = 0.0, 60.0
    EQ = (0.0, 30.0)  # equatorial cell latitude band
    PO = (60.0, 90.0)  # polar cell latitude band
    cam = Ortho(cam_lon=30.0, cam_lat=30.0, cx=CX, cy=CY, R=R)

    def ex(lon):
        return X0 + (lon + 180.0) / 360.0 * BW

    def ey(lat):
        return Y0 + (90.0 - lat) / 180.0 * BH

    # solid angle of a lon/lat cell, as a fraction of the whole sphere
    def frac(lat0, lat1, dlon=LON1 - LON0):
        return (
            dlon
            / 360.0
            * (np.sin(lat1 * D2R) - np.sin(lat0 * D2R))
            / 2.0
        )

    f_eq, f_po = frac(*EQ), frac(*PO)
    ratio = f_eq / f_po

    o = [
        txt(8, 16, "Projeção equirretangular", "tx-b"),
        txt(430, 16, "As mesmas duas células na esfera", "tx-b"),
        f'<rect class="box" x="{X0}" y="{Y0}" width="{BW}" height="{BH}" rx="3"/>',
    ]

    # ERP graticule: same lines as the sphere graticule below
    g = []
    for lat in range(-60, 90, 30):
        g.append(
            f'<line x1="{X0}" y1="{n(ey(lat))}" x2="{X0 + BW}" y2="{n(ey(lat))}"/>'
        )
    for lon in range(-120, 180, 60):
        g.append(
            f'<line x1="{n(ex(lon))}" y1="{Y0}" x2="{n(ex(lon))}" y2="{Y0 + BH}"/>'
        )
    o.append('<g class="gl">' + "".join(g) + "</g>")

    for (lat0, lat1), cls in ((EQ, "f1"), (PO, "f2")):
        o.append(
            f'<rect class="{cls}" x="{n(ex(LON0))}" y="{n(ey(lat1))}" '
            f'width="{n(ex(LON1) - ex(LON0))}" height="{n(ey(lat0) - ey(lat1))}" opacity=".85"/>'
        )

    o += [
        txt(12, 42, "+90°"),
        txt(12, ey(0) + 4, "0°"),
        txt(12, 200, "−90°"),
    ]

    # sphere: outline, graticule, then the two projected cells
    o.append(f'<circle cx="{CX}" cy="{CY}" r="{R}" class="box"/>')
    g = []
    for lat in range(-60, 90, 30):
        P = sph2vec(np.linspace(-180, 180, 361), np.full(361, float(lat)))
        d = front_path(cam, P)
        if d:
            g.append(f'<path d="{d}"/>')
    for lon in range(-180, 180, 60):
        P = sph2vec(np.full(181, float(lon)), np.linspace(-90, 90, 181))
        d = front_path(cam, P)
        if d:
            g.append(f'<path d="{d}"/>')
    o.append('<g class="gl">' + "".join(g) + "</g>")

    polys = {}
    for (lat0, lat1), cls in ((EQ, "f1"), (PO, "f2")):
        P = cell_edge(LON0, LON1, lat0, lat1)
        assert (cam.depth(P) > 0).all(), "cell falls off the visible hemisphere"
        xy = cam.xy(P)
        o.append(f'<path d="{poly(xy, close=True)}" class="{cls}" opacity=".85"/>')
        polys[cls] = xy

    # leader lines: ERP cell -> its image on the sphere, stopped at the patch outline
    for (lat0, lat1), cls in ((EQ, "f1"), (PO, "f2")):
        x1, y1 = ex(LON1) + 6, (ey(lat0) + ey(lat1)) / 2
        x2, y2 = polys[cls].mean(axis=0)
        t = np.linspace(0, 1, 400)
        seg = np.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], -1)
        hit = np.argmax(inside(seg, polys[cls]))
        L = float(np.hypot(x2 - x1, y2 - y1))
        t_end = max(0.0, t[hit] - 7.0 / L)
        o.append(
            f'<line class="arw" x1="{n(x1)}" y1="{n(y1)}" '
            f'x2="{n(x1 + t_end * (x2 - x1))}" y2="{n(y1 + t_end * (y2 - y1))}" '
            f'stroke-dasharray="3 4"/>'
        )

    # legend
    o += [
        f'<rect class="f1" x="8" y="222" width="11" height="11" rx="2"/>',
        txt(
            26,
            232,
            f"célula equatorial (0–30°) — {f_eq * 100:.2f}% do ângulo sólido da esfera".replace(
                ".", ","
            ),
        ),
        f'<rect class="f2" x="8" y="244" width="11" height="11" rx="2"/>',
        txt(
            26,
            254,
            f"célula polar (60–90°) — {f_po * 100:.2f}% do ângulo sólido".replace(
                ".", ","
            ),
        ),
        txt(8, 276, "Ambas ocupam 1/36 = 2,8% do retângulo — a mesma contagem de pixels."),
        txt(430, 232, "Mesma contagem de pixels;", "tx-v"),
        txt(
            430,
            254,
            f"a célula polar cobre {ratio:.1f}× menos ângulo sólido.".replace(".", ","),
            "tx-b",
        ),
    ]

    label = "Comparação entre uma célula equatorial e uma célula polar no ERP e na esfera"
    return svg_wrap(W, H, label, o), ratio


# ------------------------------------------------------------------ fig. 4 --


def figure4():
    W, H = 720, 248
    CX, CY, R = 212, 132, 104
    cam = Ortho(cam_lon=0.0, cam_lat=12.0, cx=CX, cy=CY, R=R)

    # p sits left of the camera axis and f points back across it, so the whole
    # p -> e displacement stays near the centre of the disk and is barely foreshortened.
    plon, plat = -46.0, 18.0
    p = sph2vec(plon, plat)
    east = np.array([-np.sin(plon * D2R), np.cos(plon * D2R), 0.0])
    north = np.array(
        [
            -np.sin(plat * D2R) * np.cos(plon * D2R),
            -np.sin(plat * D2R) * np.sin(plon * D2R),
            np.cos(plat * D2R),
        ]
    )

    fmag = 0.88  # radians of geodesic displacement
    fdir = np.cos(48 * D2R) * east + np.sin(48 * D2R) * north
    tip = p + fmag * fdir  # tip of the tangent vector, off the sphere
    e = np.cos(fmag) * p + np.sin(fmag) * fdir  # exp_p(f), back on the sphere

    def plane(a, b):
        return p + a * east + b * north

    # tangent plane, drawn as the true projection of a planar parallelogram
    A0, A1, B0, B1 = -0.32, 0.74, -0.30, 0.82
    quad = np.array([plane(A0, B0), plane(A1, B0), plane(A1, B1), plane(A0, B1)])

    o = [f'<circle cx="{CX}" cy="{CY}" r="{R}" class="box"/>']

    g = []
    for lat in (-40.0, 0.0, 40.0, 70.0):
        d = front_path(cam, sph2vec(np.linspace(-180, 180, 361), np.full(361, lat)))
        if d:
            g.append(f'<path d="{d}"/>')
    for lon in (-50.0, 0.0, 50.0, 100.0):
        d = front_path(cam, sph2vec(np.full(181, lon), np.linspace(-90, 90, 181)))
        if d:
            g.append(f'<path d="{d}"/>')
    o.append('<g class="gl">' + "".join(g) + "</g>")

    # the plane physically occludes the sphere near p, so it is drawn over it
    o.append(
        f'<path d="{poly(cam.xy(quad), close=True)}" fill="var(--accent-wash)" '
        f'stroke="var(--s1)" stroke-width="1.2"/>'
    )
    # lattice inside the plane, so it reads as a tilted surface and not as a flat badge
    lat_lines = []
    for a in np.arange(A0 + 0.212, A1 - 0.01, 0.212):
        lat_lines.append(poly(cam.xy(np.array([plane(a, B0), plane(a, B1)]))))
    for b in np.arange(B0 + 0.224, B1 - 0.01, 0.224):
        lat_lines.append(poly(cam.xy(np.array([plane(A0, b), plane(A1, b)]))))
    o.append(
        '<g stroke="var(--s1)" stroke-width=".6" fill="none" opacity=".22">'
        + "".join(f'<path d="{d}"/>' for d in lat_lines)
        + "</g>"
    )

    # geodesic p -> e on the sphere; kept on top of the translucent plane to stay legible
    t = np.linspace(0, 1, 60)[:, None]
    arc = unit(np.cos(fmag * t) * p + np.sin(fmag * t) * fdir)
    o.append(
        f'<path d="{poly(cam.xy(arc))}" stroke="var(--s3)" stroke-width="2.4" '
        f'fill="none" stroke-linecap="round"/>'
    )

    (px, py) = cam.xy(p)[0]
    (ex_, ey_) = cam.xy(e)[0]
    (tx, ty) = cam.xy(tip)[0]
    e_ax = cam.xy(plane(0.58, 0.0))[0]
    n_ax = cam.xy(plane(0.0, 0.60))[0]
    # offset the "f" label perpendicular to the arrow so it never sits on it
    dx, dy = tx - px, ty - py
    L = float(np.hypot(dx, dy)) or 1.0
    fx, fy = (px + tx) / 2 - 13 * dy / L, (py + ty) / 2 + 13 * dx / L

    o += [
        f'<line class="ax" x1="{n(px)}" y1="{n(py)}" x2="{n(e_ax[0])}" y2="{n(e_ax[1])}"/>',
        txt(e_ax[0] + 6, e_ax[1] + 13, "leste"),
        f'<line class="ax" x1="{n(px)}" y1="{n(py)}" x2="{n(n_ax[0])}" y2="{n(n_ax[1])}"/>',
        txt(n_ax[0] - 2, n_ax[1] - 8, "norte", anchor="middle"),
        f'<line class="arw" x1="{n(tx)}" y1="{n(ty)}" x2="{n(ex_)}" y2="{n(ey_)}" '
        f'stroke-dasharray="2 3"/>',
        f'<line x1="{n(px)}" y1="{n(py)}" x2="{n(tx)}" y2="{n(ty)}" stroke="var(--s2)" '
        f'stroke-width="2.4" marker-end="url(#ah)"/>',
        txt(fx, fy, "f", "tx-b", anchor="middle"),
        f'<circle cx="{n(px)}" cy="{n(py)}" r="4.5" class="f1"/>',
        txt(px - 9, py + 6, "p", "tx-b", anchor="end"),
        f'<circle cx="{n(ex_)}" cy="{n(ey_)}" r="4.5" class="f3"/>',
        txt(ex_ + 8, ey_ + 6, "e", "tx-b"),
        txt(
            CX,
            H - 10,
            "o plano é tangente em p · |f| é o comprimento do arco p → e",
            anchor="middle",
        ),
        txt(452, 60, "e = exp(p, f)", "tx-b"),
        txt(452, 80, "do vetor tangente para o ponto de chegada"),
        txt(452, 112, "f = log(p, e)", "tx-b"),
        txt(452, 132, "do ponto de chegada para o vetor tangente"),
        txt(452, 164, "d(a, b) = 2·arcsin(‖a − b‖ / 2)", "tx-b"),
        txt(452, 184, "a distância geodésica da perda e das métricas"),
        txt(452, 204, "(forma haversine — exata também perto de zero)"),
        '<defs><marker id="ah" markerUnits="userSpaceOnUse" markerWidth="10" '
        'markerHeight="8" refX="9.5" refY="4" orient="auto">'
        '<path d="M0 0 L10 4 L0 8 z" fill="var(--s2)"/></marker></defs>',
    ]

    label = "Diagrama do plano tangente, vetor de fluxo e ponto de chegada sobre a esfera"
    q = cam.xy(quad)
    diag = dict(
        f_deg=np.degrees(fmag),
        p=(round(px, 1), round(py, 1)),
        e=(round(ex_, 1), round(ey_, 1)),
        p_to_e_px=round(float(np.hypot(ex_ - px, ey_ - py)), 1),
        tip_to_e_px=round(float(np.hypot(ex_ - tx, ey_ - ty)), 1),
        plane_x=(round(q[:, 0].min(), 1), round(q[:, 0].max(), 1)),
        plane_y=(round(q[:, 1].min(), 1), round(q[:, 1].max(), 1)),
    )
    return svg_wrap(W, H, label, o), diag


# -------------------------------------------------------------------- glue --


def svg_wrap(w, h, label, body):
    head = (
        f'<svg class="chart-svg" viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="{esc(label)}">'
    )
    return head + "\n    " + "\n    ".join(body) + "\n  </svg>"


def splice(html, label, block):
    pat = re.compile(
        r'<svg class="chart-svg"[^>]*aria-label="' + re.escape(label) + r'">.*?</svg>',
        re.S,
    )
    out, k = pat.subn(lambda _: block, html)
    assert k == 1, f"expected 1 match for {label!r}, got {k}"
    return out


def main():
    f1, ratio = figure1()
    f4, diag = figure4()
    (HERE / "fig1_erp_vs_sphere.svg").write_text(f1 + "\n", encoding="utf-8")
    (HERE / "fig4_tangent_plane.svg").write_text(f4 + "\n", encoding="utf-8")

    html = HTML.read_text(encoding="utf-8")
    html = splice(
        html,
        "Comparação entre uma célula equatorial e uma célula polar no ERP e na esfera",
        f1,
    )
    html = splice(
        html,
        "Diagrama do plano tangente, vetor de fluxo e ponto de chegada sobre a esfera",
        f4,
    )
    HTML.write_text(html, encoding="utf-8")
    print(f"fig1 solid-angle ratio = {ratio:.3f}x")
    print("fig4 " + "  ".join(f"{k}={v}" for k, v in diag.items()))
    print(f"spliced into {HTML}")


if __name__ == "__main__":
    main()
