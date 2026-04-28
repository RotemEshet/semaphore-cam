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
sentence = st.text_input("Word or sentence", placeholder="e.g. HELLO or ONCE UPON A TIME").upper().strip()

if sentence:
    # Validate
    bad = [c for c in sentence if c not in cg.SEMAPHORE]
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

    def draw_semaphore_svg(r_key, l_key, size=48):
        RIGHT_ANG = {'BL':135,'B':90,'BR':45,'R':0,'UR':315,'U':270,'UL':225}
        LEFT_ANG  = {'BR':45, 'B':90,'BL':135,'L':180,'UL':225,'U':270,'UR':315}
        cx = cy = size // 2
        r  = size // 2 - 6

        def arm(deg, color):
            rad = math.radians(deg)
            ex = cx + r * math.cos(rad)
            ey = cy + r * math.sin(rad)
            fx = cx + (r+5) * math.cos(rad)
            fy = cy + (r+5) * math.sin(rad)
            return (f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" '
                    f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
                    f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="4" fill="{color}"/>')

        ra = RIGHT_ANG.get(r_key, 90)
        la = LEFT_ANG.get(l_key, 90)
        return (f'<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">'
                f'<circle cx="{cx}" cy="{cy}" r="4" fill="#555"/>'
                f'{arm(ra, "#c0392b")}{arm(la, "#2980b9")}'
                f'</svg>')

    # Show in rows of 10
    cols_per_row = 10
    for row_start in range(0, len(sequence), cols_per_row):
        row = sequence[row_start:row_start+cols_per_row]
        cols = st.columns(len(row))
        for i, ch in enumerate(row):
            r_key, l_key = cg.SEMAPHORE[ch]
            label = 'REST' if (row_start + i == 0) else (ch if ch != ' ' else '·')
            with cols[i]:
                st.markdown(f"<div style='text-align:center'>"
                           f"<b>{label}</b><br>"
                           f"{draw_semaphore_svg(r_key, l_key)}<br>"
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

with st.expander("About"):
    st.markdown("""
Two helical grooves cut into a cylindrical cam encode a word or sentence in
[flag semaphore](https://en.wikipedia.org/wiki/Flag_semaphore).
As the cam rotates, two followers drive flag arms to spell out the message.

- 🔴 Red arm = right arm (inner groove)
- 🔵 Blue arm = left arm (outer groove)
- Wall thickness: 15mm
- Groove: half-ellipse, 3.2mm deep × 4mm tall
- Arc per letter: 15.7mm (adjustable in code)
    """)
