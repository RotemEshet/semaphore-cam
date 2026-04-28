import streamlit as st
import tempfile, os, sys
import numpy as np

# Import generator
sys.path.insert(0, os.path.dirname(__file__))
import cam_generator as cg

st.set_page_config(page_title="Semaphore Cam Generator", page_icon="🚩", layout="centered")

st.title("🚩 Semaphore Cam Generator")
st.caption("Enter any word or sentence to generate a 3D-printable cylindrical cam that encodes it in flag semaphore.")

# ── Input ──────────────────────────────────────────────────────────────────
sentence = st.text_input("Word or sentence", placeholder="e.g. your name or a short word").upper().strip()

if sentence:
    # Validate
    # Expand digits to semaphore signals first
    expanded = cg.expand_input(sentence)
    bad = [c for c in expanded if c not in cg.SEMAPHORE]
    if bad:
        st.error(f"Characters not in semaphore alphabet: {', '.join(set(bad))}")
        st.stop()

    # Calculate dimensions
    sequence  = [' '] + list(sentence)
    n_anchors = len(sequence)
    r_inner   = (cg.ARC_PER_SEGMENT * n_anchors) / (2 * np.pi)
    r_outer   = r_inner + 15.0

    # Info
    col1, col2, col3 = st.columns(3)
    col1.metric("Characters", len(sentence))
    col2.metric("Inner diameter", f"{r_inner*2:.1f} mm")
    col3.metric("Outer diameter", f"{r_outer*2:.1f} mm")

    st.markdown("---")

    # ── Semaphore preview ──────────────────────────────────────────────────
    st.markdown("### Semaphore positions")

    import math

    def draw_semaphore_svg(r_key, l_key, size=72):
        RIGHT_ANG = {'BL':45,'B':90,'BR':135,'R':180,'UR':225,'U':270,'UL':315}
        LEFT_ANG  = {'BR':135,'B':90,'BL':45,'L':0,'UL':315,'U':270,'UR':225}
        pad = 10
        cx = cy = size // 2
        r  = size // 2 - pad

        def arm(deg, color):
            rad = math.radians(deg)
            ex = cx + r * math.cos(rad)
            ey = cy + r * math.sin(rad)
            fx = cx + (r + 6) * math.cos(rad)
            fy = cy + (r + 6) * math.sin(rad)
            return (f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" '
                    f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
                    f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="5" fill="{color}"/>')

        ra = RIGHT_ANG.get(r_key, 90)
        la = LEFT_ANG.get(l_key, 90)
        return (f'<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">'
                f'<circle cx="{cx}" cy="{cy}" r="4" fill="#555"/>'
                f'{arm(ra, "#c0392b")}{arm(la, "#2980b9")}'
                f'</svg>')

    # Show in rows of 8
    cols_per_row = 8
    for row_start in range(0, len(sequence), cols_per_row):
        row = sequence[row_start:row_start+cols_per_row]
        cols = st.columns(len(row))
        for i, ch in enumerate(row):
            abs_i = row_start + i
            if ch not in cg.SEMAPHORE:
                continue
            r_key, l_key = cg.SEMAPHORE[ch]
            if abs_i == 0:
                label = 'REST'
            elif ch == ' ':
                label = '_'
            elif ch == 'NUM':
                label = '#'
            elif ch == 'J':
                label = 'ABC'
            else:
                label = ch
            with cols[i]:
                st.markdown(f"<div style='text-align:center'>"
                           f"<b>{label}</b><br>"
                           f"{draw_semaphore_svg(r_key, l_key, size=72)}<br>"
                           f"<small style='color:#888'>{r_key}/{l_key}</small>"
                           f"</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Generate ───────────────────────────────────────────────────────────
    st.markdown("### Generate STL")
    st.caption(f"Cylinder: OD {r_outer*2:.1f}mm · ID {r_inner*2:.1f}mm · Height 25mm · {n_anchors} anchor points")

    if st.button("⬇ Generate & Download STL", type="primary", use_container_width=True):
        with st.spinner("Generating STL..."):
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
                tmp = f.name
            try:
                if len(sentence) == cg.N_LETTERS:
                    cg.generate_cam(sentence, tmp)
                else:
                    cg.generate_sentence(sentence, tmp)
                with open(tmp, 'rb') as f:
                    stl_bytes = f.read()
                os.unlink(tmp)
                fname = f"cam_{'_'.join(sentence.split())}.stl"
                st.success(f"✓ {len(stl_bytes)//1024} KB — ready to download")
                st.download_button(
                    label=f"💾 Download {fname}",
                    data=stl_bytes,
                    file_name=fname,
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Error: {e}")

