import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from scipy.spatial import ConvexHull

st.set_page_config(layout="centered")
st.title("👣 Foot Pressure Distribution Visualizer")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    # Check if the CSV has L1, L2... R1, R2... format and convert to left_0, left_1... right_0, right_1... format
    if 'L1' in data.columns and 'R1' in data.columns:
        # Create new dataframe with expected column names
        new_data = pd.DataFrame()
        
        # Copy left foot data (L1-L40 to left_0-left_39)
        for i in range(1, 41):  # L1 to L40
            if f'L{i}' in data.columns:
                new_data[f'left_{i-1}'] = data[f'L{i}']
        
        # Copy right foot data (R1-R40 to right_0-right_39)
        for i in range(1, 41):  # R1 to R40
            if f'R{i}' in data.columns:
                new_data[f'right_{i-1}'] = data[f'R{i}']
        
        data = new_data
        st.success(f"✅ Converted CSV format: L1-L40, R1-R40 → left_0-left_39, right_0-right_39")
        
else:
    # Sample data
    rows = 10
    samples = []
    for i in range(rows):
        row = {"id": i}
        for j in range(40):
            row[f"left_{j}"] = np.random.rand() * 100
        for j in range(40):
            row[f"right_{j}"] = np.random.rand() * 100
        samples.append(row)
    data = pd.DataFrame(samples)

# --- Helper Functions ---
def get_foot_coordinates(is_left=True):
    coords = [
        [(0, 25)],
        [(-15, 40), (0, 40), (15, 40), (30, 45)],
        [(-20, 55), (-5, 55), (10, 55), (25, 55)],
        [(-25, 70), (-10, 70), (5, 70), (20, 70)],
        [(-30, 85), (-15, 85), (0, 85), (15, 85)],
        [(-30, 100), (-15, 100), (0, 100)],
        [(-30, 115), (-15, 115), (0, 115)],
        [(-30, 130), (-15, 130), (0, 130)],
        [(-25, 145), (-10, 145), (5, 145)],
        [(-20, 160), (-5, 160), (10, 160)],
        [(-15, 175), (0, 175), (15, 175)],
        [(-5, 190), (5, 190)]
    ]
    points = []
    for row in coords:
        for (dx, dy) in row:
            points.append({"x": 75 + dx, "y": dy})
    if not is_left:
        return [{"x": 150 - p["x"], "y": p["y"]} for p in points]
    return points

def get_pressure_color(value):
    if pd.isna(value):
        return "#f0f0f0"
    normalized = min(max(value / 100, 0), 1)
    if normalized < 0.2: return "#f0f0f0"
    if normalized < 0.4: return "#ffff99"
    if normalized < 0.6: return "#ffcc00"
    if normalized < 0.8: return "#ff6600"
    return "#cc0000"

def _foot_outline_path_from_coords(is_left=True):
    coords = get_foot_coordinates(is_left)
    points = np.array([[c["x"], c["y"]] for c in coords])
    hull = ConvexHull(points)
    outline_points = points[hull.vertices]
    verts = np.vstack([outline_points, outline_points[0]])
    codes = [Path.MOVETO] + [Path.LINETO]*(len(verts)-2) + [Path.CLOSEPOLY]
    return Path(verts, codes)

def draw_foot_outline(ax, is_left=True):
    path = _foot_outline_path_from_coords(is_left)
    outline = patches.PathPatch(path, fill=False, edgecolor="gray", linewidth=2, alpha=0.9)
    ax.add_patch(outline)

def create_squares(ax, is_left=True, show_values=False):
    coords = get_foot_coordinates(is_left)
    rects, dots, labels = [], [], []
    for coord in coords:
        rect = patches.Rectangle((coord["x"]-8, coord["y"]-8), 16, 16,
                                linewidth=1, edgecolor="black", facecolor="#f0f0f0", alpha=0.9)
        ax.add_patch(rect); rects.append(rect)
        dot, = ax.plot(coord["x"], coord["y"], "o", color="black", markersize=2, visible=(not show_values))
        dots.append(dot)
        label = ax.text(coord["x"], coord["y"], "", ha="center", va="center", fontsize=6,
                        color="black", visible=show_values)
        labels.append(label)
    return rects, dots, labels

def update_squares(rects, dots, labels, data_row, prefix, show_values):
    for idx, rect in enumerate(rects):
        value = data_row.get(f"{prefix}{idx}", None)
        rect.set_facecolor(get_pressure_color(value))
        if show_values and value is not None:
            labels[idx].set_text(f"{int(value)}")
        else:
            labels[idx].set_text("")
        labels[idx].set_visible(show_values)
        dots[idx].set_visible(not show_values)

def create_screenshot_figure(data_row, frame_num, total_frames, show_values):
    """Create a standalone figure for screenshot"""
    fig_snap, axes_snap = plt.subplots(1, 2, figsize=(8, 6))
    ax_l, ax_r = axes_snap
    for ax in (ax_l, ax_r):
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 220)
        ax.set_aspect("equal")
        ax.axis("off")
    ax_l.set_title("Left Foot")
    ax_r.set_title("Right Foot")
    draw_foot_outline(ax_l, True)
    draw_foot_outline(ax_r, False)
    l_rects, l_dots, l_labels = create_squares(ax_l, True, show_values)
    r_rects, r_dots, r_labels = create_squares(ax_r, False, show_values)
    update_squares(l_rects, l_dots, l_labels, data_row, "left_", show_values)
    update_squares(r_rects, r_dots, r_labels, data_row, "right_", show_values)
    fig_snap.text(0.5, 0.95, f"Frame: {frame_num}/{total_frames}", ha="center", va="center", fontsize=12)
    return fig_snap

# --- Initialize session state ---
if "frame" not in st.session_state:
    st.session_state.frame = 0
if "playing" not in st.session_state:
    st.session_state.playing = False
if "last_show_values" not in st.session_state:
    st.session_state.last_show_values = False
if "plot_placeholder" not in st.session_state:
    st.session_state.plot_placeholder = None

# --- Controls ---
play_col, pause_col = st.columns(2)
if play_col.button("▶ Play", key="playb"):
    st.session_state.playing = True
if pause_col.button("⏸ Pause", key="pauseb"):
    st.session_state.playing = False

speed = st.sidebar.slider("Playback Speed (sec per frame)", 0.05, 1.5, 0.35, 0.05)
show_values = st.sidebar.checkbox("Show Pressure Values", value=False)

# Only show slider when not playing
if not st.session_state.playing:
    st.session_state.frame = st.slider("Frame", 0, len(data) - 1, st.session_state.frame)

# --- Initialize or recreate figure when needed ---
need_new_fig = ("fig" not in st.session_state or 
                st.session_state.last_show_values != show_values)

if need_new_fig:
    # Close existing figure if it exists
    if "fig" in st.session_state:
        plt.close(st.session_state.fig)
    
    # Clear any existing plot placeholder
    if st.session_state.plot_placeholder is not None:
        st.session_state.plot_placeholder.empty()
    
    # Create new figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    ax_left, ax_right = axes
    for ax in (ax_left, ax_right):
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 220)
        ax.set_aspect("equal")
        ax.axis("off")
    ax_left.set_title("Left Foot")
    ax_right.set_title("Right Foot")
    
    # Draw static elements
    draw_foot_outline(ax_left, True)
    draw_foot_outline(ax_right, False)
    
    # Create interactive elements
    left_rects, left_dots, left_labels = create_squares(ax_left, True, show_values)
    right_rects, right_dots, right_labels = create_squares(ax_right, False, show_values)
    frame_text = fig.text(0.5, 0.95, "", ha="center", va="center", fontsize=12)
    
    # Store in session state
    st.session_state.fig = fig
    st.session_state.ax_left = ax_left
    st.session_state.ax_right = ax_right
    st.session_state.left_rects = left_rects
    st.session_state.left_dots = left_dots
    st.session_state.left_labels = left_labels
    st.session_state.right_rects = right_rects
    st.session_state.right_dots = right_dots
    st.session_state.right_labels = right_labels
    st.session_state.frame_text = frame_text
    st.session_state.last_show_values = show_values

# --- Render current frame ---
def render_frame(frame_idx, show_values):
    row = data.iloc[frame_idx]
    update_squares(st.session_state.left_rects, st.session_state.left_dots, 
                  st.session_state.left_labels, row, "left_", show_values)
    update_squares(st.session_state.right_rects, st.session_state.right_dots, 
                  st.session_state.right_labels, row, "right_", show_values)
    st.session_state.frame_text.set_text(f"Frame: {frame_idx + 1}/{len(data)}")

# Create placeholder for the main plot and render current frame
if st.session_state.plot_placeholder is None:
    st.session_state.plot_placeholder = st.empty()

render_frame(st.session_state.frame, show_values)
st.session_state.plot_placeholder.pyplot(st.session_state.fig, clear_figure=False, use_container_width=True)

# --- Auto-advance logic for playing ---
if st.session_state.playing:
    # Use a small delay and rerun
    time.sleep(speed)
    st.session_state.frame = (st.session_state.frame + 1) % len(data)
    st.rerun()

# --- Screenshot feature (always available) ---
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("📸 Screenshot", key="screenshot_btn"):
        current_frame = st.session_state.frame
        row = data.iloc[current_frame]
        fig_snap = create_screenshot_figure(row, current_frame + 1, len(data), show_values)
        
        # Save to buffer
        buf = io.BytesIO()
        fig_snap.savefig(buf, format="png", dpi=180, bbox_inches="tight")
        plt.close(fig_snap)
        
        # Trigger immediate download
        st.download_button(
            "⬇️ Download Now",
            data=buf.getvalue(),
            file_name=f"foot_pressure_frame_{current_frame + 1}.png",
            mime="image/png",
            key=f"download_{current_frame}_{int(time.time())}"  # Unique key
        )

# --- Legend ---
st.markdown("### Pressure Scale")
legend_colors = ["#f0f0f0", "#ffff99", "#ffcc00", "#ff6600", "#cc0000"]
legend_labels = ["Low", "↔", "↔", "↔", "High"]
cols = st.columns(len(legend_colors))
for col, color, label in zip(cols, legend_colors, legend_labels):
    col.markdown(f"<div style='width:20px;height:20px;background:{color};border:1px solid #00000022'></div>", unsafe_allow_html=True)
    col.write(label)

if uploaded_file:
    st.success(f"📊 Loaded {len(data)} data rows with {len([c for c in data.columns if c.startswith('left_')])} left + {len([c for c in data.columns if c.startswith('right_')])} right sensors")
    
    # Show a preview of the data range
    left_cols = [c for c in data.columns if c.startswith("left_")]
    right_cols = [c for c in data.columns if c.startswith("right_")]
    if left_cols and right_cols:
        left_max = data[left_cols].max().max()
        right_max = data[right_cols].max().max()
        st.info(f"📈 Pressure range: Left foot max = {left_max:.1f}, Right foot max = {right_max:.1f}")
else:
    st.info("Showing sample data – upload your CSV for real data")

st.markdown("---")
st.subheader("📊 Foot Pressure Analysis Report")

def make_summary(df):
    left_cols = [c for c in df.columns if c.startswith("left_")]
    right_cols = [c for c in df.columns if c.startswith("right_")]
    
    # Basic statistics
    summary = pd.DataFrame({
        "frame": np.arange(len(df)) + 1,
        "left_mean": df[left_cols].mean(axis=1),
        "left_max": df[left_cols].max(axis=1),
        "right_mean": df[right_cols].mean(axis=1),
        "right_max": df[right_cols].max(axis=1),
    })
    
    # Calculate total pressure for each frame
    summary["total_pressure"] = summary["left_mean"] + summary["right_mean"]
    
    return summary, left_cols, right_cols

summary_df, left_cols, right_cols = make_summary(data)

# Analysis metrics
col1, col2, col3 = st.columns(3)

with col1:
    max_pressure_frame = summary_df.loc[summary_df["total_pressure"].idxmax()]
    st.metric("🚨 Peak Pressure Frame", 
              f"Frame {int(max_pressure_frame['frame'])}", 
              f"{max_pressure_frame['total_pressure']:.1f}")

with col2:
    left_dominant_frames = len(summary_df[summary_df["left_mean"] > summary_df["right_mean"]])
    st.metric("⬅️ Left Foot Dominant", 
              f"{left_dominant_frames} frames", 
              f"{left_dominant_frames/len(summary_df)*100:.1f}%")

with col3:
    right_dominant_frames = len(summary_df[summary_df["right_mean"] > summary_df["left_mean"]])
    st.metric("➡️ Right Foot Dominant", 
              f"{right_dominant_frames} frames", 
              f"{right_dominant_frames/len(summary_df)*100:.1f}%")

# Detailed analysis
st.markdown("#### 🔍 Detailed Analysis")

# Find critical frames
high_pressure_threshold = summary_df["total_pressure"].quantile(0.8)
critical_frames = summary_df[summary_df["total_pressure"] > high_pressure_threshold]

if len(critical_frames) > 0:
    st.warning(f"⚠️ **High Pressure Alert**: {len(critical_frames)} frames show elevated pressure levels (above 80th percentile)")
    
    # Show top 5 critical frames
    top_critical = critical_frames.nlargest(5, "total_pressure")
    st.markdown("**Top 5 High-Pressure Frames:**")
    
    for _, frame_data in top_critical.iterrows():
        frame_num = int(frame_data["frame"])
        total_pressure = frame_data["total_pressure"]
        
        # Determine which foot has higher pressure
        if frame_data["left_mean"] > frame_data["right_mean"]:
            dominant_foot = "Left"
            pressure_diff = frame_data["left_mean"] - frame_data["right_mean"]
        else:
            dominant_foot = "Right"
            pressure_diff = frame_data["right_mean"] - frame_data["left_mean"]
        
        st.markdown(f"• **Frame {frame_num}**: Total pressure {total_pressure:.1f} | {dominant_foot} foot dominant (+{pressure_diff:.1f})")

else:
    st.success("✅ No critical pressure levels detected")

# Balance analysis
balance_scores = []
for _, row in summary_df.iterrows():
    if row["left_mean"] + row["right_mean"] > 0:
        balance = 1 - abs(row["left_mean"] - row["right_mean"]) / (row["left_mean"] + row["right_mean"])
        balance_scores.append(balance)
    else:
        balance_scores.append(1.0)

avg_balance = np.mean(balance_scores)
st.markdown(f"#### ⚖️ Balance Analysis")
st.progress(avg_balance)
st.markdown(f"**Average Balance Score**: {avg_balance:.2f} (1.0 = Perfect Balance)")

if avg_balance < 0.7:
    st.error("🚨 Significant imbalance detected - consider consulting a specialist")
elif avg_balance < 0.85:
    st.warning("⚠️ Moderate imbalance detected - monitor walking pattern")
else:
    st.success("✅ Good balance maintained throughout the walking sequence")

# Download enhanced summary
csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Detailed Analysis Report", 
                  data=csv_bytes, 
                  file_name="foot_pressure_analysis_report.csv", 
                  mime="text/csv")