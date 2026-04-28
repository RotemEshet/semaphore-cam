"""
Semaphore Automata Cam Generator

Walls: one continuous quad mesh per wall, column by column.
Entry slots: elliptical bores at 0deg, baked into wall mesh from the start.
Bottom cap: Delaunay triangulation with exact rim points.
"""

import numpy as np
import struct
import sys

# ── Lookup tables ──────────────────────────────────────────────────────────
RIGHT_Z = {'BL':21.5,'B':18.5,'BR':15.5,'R':12.5,'UR':9.5,'U':6.5,'UL':3.5}
LEFT_Z  = {'BR':21.5,'B':18.5,'BL':15.5,'L':12.5,'UL':9.5,'U':6.5,'UR':3.5}

SEMAPHORE = {
    'A':('BR','B'),  'B':('R', 'B'),  'C':('UR','B'),  'D':('U', 'B'),
    'E':('B', 'UL'), 'F':('B', 'L'),  'G':('B', 'BL'), 'H':('R', 'BR'),
    'I':('UR','BR'), 'J':('U', 'L'),  'K':('BR','U'),  'L':('BR','UL'),
    'M':('BR','L'),  'N':('BR','BL'), 'O':('R', 'UR'), 'P':('R', 'U'),
    'Q':('R', 'UL'), 'R':('R', 'L'),  'S':('R', 'BL'), 'T':('UR','U'),
    'U':('UR','UL'), 'V':('U', 'BL'), 'W':('UL','L'),  'X':('UL','BL'),
    'Y':('UR','L'),  'Z':('BL','L'),  ' ':('B', 'B'),
}

# Number mode signals
SEMAPHORE['NUM'] = ('U', 'UL')   # numbers follow signal
# J is already in SEMAPHORE as letters follow signal: ('U', 'L')
# Digit mapping: 1-9 = A-I, 0 = K
DIGIT_TO_LETTER = {'1':'A','2':'B','3':'C','4':'D','5':'E',
                   '6':'F','7':'G','8':'H','9':'I','0':'K'}

def expand_input(text):
    """
    Convert raw text (may contain digits) into a sequence of semaphore
    characters, inserting NUM and J signals automatically.
    """
    text = text.upper()
    result = []
    in_number_mode = False
    for i, ch in enumerate(text):
        if ch.isdigit():
            if not in_number_mode:
                result.append('NUM')
                in_number_mode = True
            result.append(DIGIT_TO_LETTER[ch])
        elif ch == ' ':
            result.append(' ')
        else:
            if in_number_mode:
                remaining = text[i:]
                if any(c.isalpha() for c in remaining):
                    result.append('J')
                in_number_mode = False
            result.append(ch)
    return result

def letter_to_zs(ch):
    r, l = SEMAPHORE[ch]
    return RIGHT_Z[r], LEFT_Z[l]

# ── Constants ──────────────────────────────────────────────────────────────
N_LETTERS        = 5      # default word length — overridden dynamically
ARC_PER_SEGMENT  = 15.7   # mm of arc per anchor point — increase if slope too steep

N_ANCHORS     = N_LETTERS + 1                        # includes rest position
R_INNER       = (ARC_PER_SEGMENT * N_ANCHORS) / (2 * np.pi)
R_OUTER       = R_INNER + 15.0                       # wall thickness = 15mm
HOLE_PCD_R    = R_INNER + 7.5                        # midwall

CAM_H         = 25.0
GROOVE_DEPTH  = 3.2
GROOVE_SEMI_H = 2.0
HOLE_R        = 3.0
HOLE_DEPTH    = 10.0
REST_Z        = 18.5

N_CIRC        = 360
N_PROF        = 32
N_ROWS        = 80
N_HOLE_SEG    = 48

# ── Triangle list ──────────────────────────────────────────────────────────
tris = []

def tri(a, b, c):
    tris.append((np.asarray(a,float), np.asarray(b,float), np.asarray(c,float)))

def quad(a, b, c, d):
    tri(a, b, c); tri(a, c, d)

# ── Spline ─────────────────────────────────────────────────────────────────
def hermite(t, p0, p1):
    t2, t3 = t*t, t*t*t
    return (2*t3 - 3*t2 + 1)*p0 + (-2*t3 + 3*t2)*p1

def eval_spline(anchor_zs, angles):
    seg = 2 * np.pi / N_ANCHORS
    result = np.zeros(len(angles))
    for i, a in enumerate(angles):
        idx = int(a / seg) % N_ANCHORS
        nxt = (idx + 1) % N_ANCHORS
        t   = (a - idx * seg) / seg
        result[i] = hermite(t, anchor_zs[idx], anchor_zs[nxt])
    return result

# ── Slot geometry helpers ──────────────────────────────────────────────────
def build_wall(anchor_zs, surface_r, inward):
    angles  = np.linspace(0, 2*np.pi, N_CIRC, endpoint=False)
    zc_all  = eval_spline(anchor_zs, angles)

    def wall_r(r_off):
        return (surface_r - r_off) if inward else (surface_r + r_off)

    def pt(angle, r, z):
        return (r*np.cos(angle), r*np.sin(angle), z)

    def column_profile(zc):
        """(r_offset, z) pairs from bottom to top for one wall column."""
        z_bot = zc - GROOVE_SEMI_H
        z_top = zc + GROOVE_SEMI_H
        pts = []
        n_below = max(2, int(N_ROWS * z_bot / CAM_H))
        for z in np.linspace(0, z_bot, n_below):
            pts.append((0.0, z))
        ts = np.linspace(np.pi, 0, N_PROF)
        for t in ts:
            pts.append((GROOVE_DEPTH * np.sin(t),
                        np.clip(zc + GROOVE_SEMI_H * np.cos(t), 0, CAM_H)))
        n_above = max(2, int(N_ROWS * (CAM_H - z_top) / CAM_H))
        for z in np.linspace(z_top, CAM_H, n_above):
            pts.append((0.0, z))
        return pts

    def resample(col, n):
        t_old = np.linspace(0, 1, len(col))
        t_new = np.linspace(0, 1, n)
        rs = np.interp(t_new, t_old, [p[0] for p in col])
        zs = np.interp(t_new, t_old, [p[1] for p in col])
        return list(zip(rs, zs))

    cols = [resample(column_profile(zc), N_ROWS) for zc in zc_all]

    for k in range(N_CIRC):
        k1   = (k+1) % N_CIRC
        a0, a1 = angles[k], angles[k1]
        col0, col1 = cols[k], cols[k1]

        for j in range(N_ROWS - 1):
            r0,z0   = col0[j];  r1,z1   = col0[j+1]
            r0b,z0b = col1[j];  r1b,z1b = col1[j+1]

            v00 = pt(a0, wall_r(r0),  z0)
            v10 = pt(a1, wall_r(r0b), z0b)
            v01 = pt(a0, wall_r(r1),  z1)
            v11 = pt(a1, wall_r(r1b), z1b)
            if inward:
                quad(v00, v10, v11, v01)
            else:
                quad(v00, v01, v11, v10)



# ── Top cap ────────────────────────────────────────────────────────────────
def build_top_cap():
    angles = np.linspace(0, 2*np.pi, N_CIRC, endpoint=False)
    for i in range(N_CIRC):
        a0, a1 = angles[i], angles[(i+1)%N_CIRC]
        vi0 = (R_INNER*np.cos(a0), R_INNER*np.sin(a0), CAM_H)
        vi1 = (R_INNER*np.cos(a1), R_INNER*np.sin(a1), CAM_H)
        vo0 = (R_OUTER*np.cos(a0), R_OUTER*np.sin(a0), CAM_H)
        vo1 = (R_OUTER*np.cos(a1), R_OUTER*np.sin(a1), CAM_H)
        quad(vi0, vo0, vo1, vi1)

def build_bottom_cap():
    from scipy.spatial import Delaunay

    hole_centres = [(HOLE_PCD_R*np.cos(i*2*np.pi/3),
                     HOLE_PCD_R*np.sin(i*2*np.pi/3)) for i in range(3)]

    def in_any_hole(x, y):
        return any((x-cx)**2+(y-cy)**2 < HOLE_R**2 for cx,cy in hole_centres)

    def in_annulus(x, y):
        r = np.sqrt(x**2+y**2)
        return R_INNER <= r <= R_OUTER

    wall_angles = np.linspace(0, 2*np.pi, N_CIRC, endpoint=False)
    hsegs = np.linspace(0, 2*np.pi, N_HOLE_SEG, endpoint=False)

    pts = []
    for a in wall_angles:
        pts.append([R_OUTER*np.cos(a), R_OUTER*np.sin(a)])
    for a in wall_angles:
        pts.append([R_INNER*np.cos(a), R_INNER*np.sin(a)])
    N_RAD = 8
    for r in np.linspace(R_INNER, R_OUTER, N_RAD)[1:-1]:
        for a in wall_angles[::3]:
            pts.append([r*np.cos(a), r*np.sin(a)])

    hole_rim_starts = []
    for cx, cy in hole_centres:
        hole_rim_starts.append(len(pts))
        for a in hsegs:
            pts.append([cx + HOLE_R*np.cos(a), cy + HOLE_R*np.sin(a)])

    pts = np.array(pts)
    d = Delaunay(pts)

    for t in d.simplices:
        centroid = pts[t].mean(axis=0)
        if in_annulus(*centroid) and not in_any_hole(*centroid):
            v0 = (pts[t[0]][0], pts[t[0]][1], 0.0)
            v1 = (pts[t[1]][0], pts[t[1]][1], 0.0)
            v2 = (pts[t[2]][0], pts[t[2]][1], 0.0)
            tri(v0, v2, v1)

    for hi, (cx, cy) in enumerate(hole_centres):
        rs = hole_rim_starts[hi]
        for j in range(N_HOLE_SEG):
            j1 = (j+1) % N_HOLE_SEG
            b0 = (pts[rs+j][0],  pts[rs+j][1],  0.0)
            b1 = (pts[rs+j1][0], pts[rs+j1][1], 0.0)
            t0 = (pts[rs+j][0],  pts[rs+j][1],  HOLE_DEPTH)
            t1 = (pts[rs+j1][0], pts[rs+j1][1], HOLE_DEPTH)
            quad(b0, t0, t1, b1)
        ctr = (cx, cy, HOLE_DEPTH)
        for j in range(N_HOLE_SEG):
            j1 = (j+1) % N_HOLE_SEG
            v0 = (pts[rs+j][0],  pts[rs+j][1],  HOLE_DEPTH)
            v1 = (pts[rs+j1][0], pts[rs+j1][1], HOLE_DEPTH)
            tri(ctr, v1, v0)

# ── STL writer ─────────────────────────────────────────────────────────────
def write_stl(path):
    hdr = b'Semaphore Cam' + b'\x00' * 67
    with open(path, 'wb') as f:
        f.write(hdr)
        f.write(struct.pack('<I', len(tris)))
        for v0,v1,v2 in tris:
            n  = np.cross(v1-v0, v2-v0)
            nl = np.linalg.norm(n)
            if nl > 0: n /= nl
            f.write(struct.pack('<fff', *n))
            f.write(struct.pack('<fff', *v0))
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            f.write(struct.pack('<H', 0))
    print(f"Written {len(tris)} triangles to {path}")

# ── Generate ───────────────────────────────────────────────────────────────
def generate_cam(word, output_path):
    word = word.upper().strip()
    if len(word) != N_LETTERS:
        raise ValueError(f"Word must be {N_LETTERS} letters, got '{word}' ({len(word)})")
    expanded = expand_input(word)
    bad = [ch for ch in expanded if ch not in SEMAPHORE]
    if bad:
        raise ValueError(f"Characters not supported: {bad}")

    sequence = [' '] + expanded
    inner_zs = [letter_to_zs(ch)[0] for ch in sequence]
    outer_zs = [letter_to_zs(ch)[1] for ch in sequence]

    print(f"Word: {word}")
    for i, ch in enumerate(sequence):
        print(f"  [{i*60:3d}deg] {'REST' if i==0 else ch}  inner={inner_zs[i]:.1f}  outer={outer_zs[i]:.1f}")

    global tris; tris = []
    print("Building outer wall..."); build_wall(outer_zs, R_OUTER, inward=True)
    print("Building inner wall..."); build_wall(inner_zs, R_INNER, inward=False)
    print("Building top cap...");    build_top_cap()
    print("Building bottom cap..."); build_bottom_cap()
    write_stl(output_path)

def generate_test(output_path):
    global tris; tris = []
    inner_zs = [18.5, 3.5, 21.5, 12.5, 21.5,  3.5]
    outer_zs = [18.5, 21.5,  3.5, 21.5, 12.5, 12.5]
    print("TEST CAM")
    build_wall(outer_zs, R_OUTER, inward=True)
    build_wall(inner_zs, R_INNER, inward=False)
    build_top_cap()
    build_bottom_cap()
    write_stl(output_path)

def generate_sentence(sentence, output_path):
    """Generate a cam for an arbitrary length sentence."""
    global N_ANCHORS, R_INNER, R_OUTER, HOLE_PCD_R

    sentence = sentence.upper().strip()
    expanded = expand_input(sentence)
    bad = [ch for ch in expanded if ch not in SEMAPHORE]
    if bad:
        raise ValueError(f"Characters not supported: {bad}")

    sequence = [' '] + expanded
    N_ANCHORS = len(sequence)
    R_INNER   = (ARC_PER_SEGMENT * N_ANCHORS) / (2 * np.pi)
    R_OUTER   = R_INNER + 15.0
    HOLE_PCD_R = R_INNER + 7.5

    inner_zs = [letter_to_zs(ch)[0] for ch in sequence]
    outer_zs = [letter_to_zs(ch)[1] for ch in sequence]

    print(f"Sentence: {sentence}")
    print(f"Anchor points: {N_ANCHORS}  R_INNER: {R_INNER:.1f}mm  R_OUTER: {R_OUTER:.1f}mm  OD: {R_OUTER*2:.1f}mm")
    for i, ch in enumerate(sequence):
        label = 'REST' if i == 0 else ch
        print(f"  [{i*360//N_ANCHORS:3d}deg] {label!r}  inner={inner_zs[i]:.1f}  outer={outer_zs[i]:.1f}")

    global tris; tris = []
    print("Building outer wall..."); build_wall(outer_zs, R_OUTER, inward=True)
    print("Building inner wall..."); build_wall(inner_zs, R_INNER, inward=False)
    print("Building top cap...");    build_top_cap()
    print("Building bottom cap..."); build_bottom_cap()
    write_stl(output_path)

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    out  = sys.argv[2] if len(sys.argv) > 2 else "cam.stl"
    if mode == "test":
        generate_test(out)
    elif len(mode) > N_LETTERS:
        generate_sentence(mode, out)
    else:
        generate_cam(mode, out)
