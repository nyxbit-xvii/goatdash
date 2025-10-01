# --- MP4 generation: robust writer with fallbacks, low memory ---
import imageio.v3 as iio
import os
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import tempfile

# Try a basic font if default draw() fails silently on some hosts
def _get_font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None

def _ffmpeg_available():
    try:
        # This is enough to confirm imageio-ffmpeg bundled binary is usable
        _ = iio.imopen("<nocreate>", "w", plugin="FFMPEG")
        return True
    except Exception:
        return False

def render_mp4_safe(interp: pd.DataFrame, center_lat, center_lon,
                    outfile="bonus_week.mp4", zoom=STATIC_ZOOM,
                    w=STATIC_W, h=STATIC_H, step=12, fps=10):
    """
    Streams frames directly to FFmpeg, keeps memory tiny.
    - step=12 -> about every 12 minutes (change as you like)
    - falls back to a simpler codec if libx264 is unavailable
    - writes to a temp dir then returns final path
    """
    # 1) fetch single background + icon
    bg = fetch_static_map(center_lat, center_lon, w, h, zoom)
    icon = fetch_icon().resize((64, 64), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
    font = _get_font()

    # 2) precompute pixel coords
    coords = [latlon_to_pixel(float(r.lat), float(r.lon), center_lat, center_lon, zoom, w, h)
              for r in interp.itertuples(index=False)]

    # 3) temp path
    tmpdir = tempfile.mkdtemp()
    out_path = os.path.join(tmpdir, outfile)

    # 4) pick a codec (try libx264, then mpeg4)
    codecs = [
        dict(codec="libx264", pix_fmt="yuv420p"),
        dict(codec="mpeg4",  pix_fmt="yuv420p"),
    ]

    last_err = None
    for enc in codecs:
        try:
            with iio.imopen(out_path, "w", plugin="FFMPEG", fps=fps, **enc) as writer:
                # Stream frames; short trail to lighten draw()
                for i in range(0, len(interp), step):
                    frame = bg.copy()
                    draw = ImageDraw.Draw(frame, "RGBA")

                    # Trail: ~80 steps at 'step' spacing
                    if i > 1:
                        start_idx = max(0, i - 80)
                        draw.line(coords[start_idx:i+1], fill=(0, 200, 255, 200), width=4)

                    # Icon
                    x, y = coords[i]
                    frame.alpha_composite(icon, (x - icon.width//2, y - icon.height))

                    # Timestamp label (safe even without a font)
                    ts = interp.iloc[i]["timestamp"]
                    label = f"Bonus — {ts.strftime('%Y-%m-%d %H:%M UTC')}"
                    draw.rectangle([(10, h - 40), (360, h - 10)], fill=(0, 0, 0, 140))
                    if font:
                        draw.text((16, h - 34), label, fill=(255, 255, 255, 230), font=font)
                    else:
                        draw.text((16, h - 34), label, fill=(255, 255, 255, 230))

                    writer.write(np.array(frame))

            # success with this codec
            return out_path
        except Exception as e:
            last_err = e
            continue

    # If both codecs fail, raise the last error with helpful context
    raise RuntimeError(f"FFmpeg failed (checked codecs libx264/mpeg4). "
                       f"Ensure 'imageio-ffmpeg' is installed. Last error: {last_err}")

# ----------------- MP4 UI block (replace your current MP4 section) -----------------
st.subheader("3) Weekly Timelapse (downloadable MP4)")
col_mp4a, col_mp4b = st.columns([1,3])
with col_mp4a:
    make_mp4 = st.button("🎬 Generate MP4")
with col_mp4b:
    st.caption("Tip: We downsample automatically to keep it fast and stable on Streamlit Cloud.")

if make_mp4:
    # Sanity checks up front with clear messages
    if not _ffmpeg_available():
        st.error("FFmpeg backend not available. Make sure 'imageio-ffmpeg' is in requirements.txt.")
    else:
        try:
            with st.spinner("Rendering timelapse video…"):
                # Lighter defaults for Streamlit Cloud:
                # - step=12 (~every 12 minutes)
                # - 800x600 (already set via STATIC_W/H)
                # - fps=10
                mp4_path = render_mp4_safe(interp, mid_lat, mid_lon, step=12, w=STATIC_W, h=STATIC_H, fps=10)

            # Read and show
            with open(mp4_path, "rb") as f:
                data = f.read()
                st.video(data)
                st.download_button("⬇️ Download bonus_week.mp4", data=data,
                                   file_name="bonus_week.mp4", mime="video/mp4")
        except Exception as e:
            st.error(f"Video rendering failed: {e}")










