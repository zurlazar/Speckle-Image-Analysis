import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import io
import os
import glob
import math
import cv2 # Required for HoughCircles, GaussianBlur, etc.
from tifffile import imread # Required for loading TIF files, though used generically here

# --- Particle Analysis Function ---

def analyze_particles_hough(img_array, param1, param2, minRadius, maxRadius):
    """
    Performs particle detection using the Hough Circle Transform with adjustable parameters.
    
    Returns: particle_count (int), color_img_with_detections (np.array), detection_data (list)
    """
    if img_array is None:
        return 0, None, [], []

    # 1. Convert to 8-bit grayscale for OpenCV compatibility
    # Scale to 0-255 range and convert to unsigned 8-bit integer
    img_8bit = np.uint8(cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX))

    # 2. Preprocessing (Blur and CLAHE from the original interactive script)
    blurred = cv2.GaussianBlur(img_8bit, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # 3. Detect circles using the current slider values
    # NOTE: cv2.HoughCircles only accepts 8-bit single-channel images.
    circles = cv2.HoughCircles(
        enhanced,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=6,
        param1=param1,      # Canny high threshold
        param2=param2,      # Accumulator threshold (Sensitivity)
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    # 4. Draw detections and calculate metrics
    color_img = cv2.cvtColor(img_8bit.copy(), cv2.COLOR_GRAY2BGR)
    xs, ys, rs = [], [], []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0, :]:
            # Draw the detected circle outline
            cv2.circle(color_img, (x, y), r, (255, 0, 0), 1)
            # Draw small red center point
            cv2.circle(color_img, (x, y), 2, (0, 0, 255), -1) 
            
            xs.append(x)
            ys.append(y)
            rs.append(r)

    particle_count = len(rs)
    total_area_px2 = np.sum([math.pi * (r ** 2) for r in rs])
    
    return particle_count, color_img, xs, ys, rs

# --- Analysis Functions (Speckle) ---

@st.cache_data
def load_image(file_content):
    """
    Load 12-bit PNG/TIF image and convert to numpy array.
    """
    try:
        if isinstance(file_content, io.BytesIO):
            # Handles Streamlit UploadedFile content
            # Try to read as TIF first if applicable, otherwise use PIL
            try:
                img_array = imread(file_content)
                if img_array.ndim > 2:
                    img_array = img_array[..., 0]
            except Exception:
                img = Image.open(file_content)
                img_array = np.array(img, dtype=np.float64)
        elif isinstance(file_content, str) and os.path.exists(file_content):
            # Handles local file paths (for pre-uploaded files)
            try:
                img_array = imread(file_content)
                if img_array.ndim > 2:
                    img_array = img_array[..., 0]
            except Exception:
                img = Image.open(file_content)
                img_array = np.array(img, dtype=np.float64)
        else:
            st.error("Invalid file content provided to load_image.")
            return None

        # Check for 16-bit container for 12-bit data (common in microscopy)
        if img_array.max() > 4095:
            img_array = img_array / 16.0 
            
        return img_array

    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# (Keep calculate_statistics, calculate_autocorrelation_fft, find_speckle_size, 
# plot_image, plot_psd, and plot_autocorr_cross_sections functions as they are)

def calculate_statistics(img_array):
    min_val = np.min(img_array)
    max_val = np.max(img_array)
    mean_val = np.mean(img_array)
    std_val = np.std(img_array)
    contrast = std_val / mean_val if mean_val != 0 else 0
    return min_val, max_val, mean_val, std_val, contrast

@st.cache_data
def calculate_autocorrelation_fft(img_array):
    img_norm = img_array - np.mean(img_array)
    f_img = np.fft.fft2(img_norm)
    power_spectrum_raw = np.abs(f_img) ** 2
    power_spectrum_shifted = np.fft.fftshift(power_spectrum_raw)
    power_spectrum_log = np.log10(power_spectrum_shifted + 1e-6)  
    autocorr = np.fft.ifft2(power_spectrum_raw).real
    autocorr = np.fft.fftshift(autocorr)
    autocorr = autocorr / autocorr.max()
    return autocorr, power_spectrum_log

def find_speckle_size(autocorr):
    center_y, center_x = np.array(autocorr.shape) // 2
    autocorr_x = autocorr[center_y, :]
    autocorr_y = autocorr[:, center_x]
    threshold = 0.5
    right_of_center = autocorr_x[center_x:]
    right_idx = np.where(right_of_center < threshold)[0]
    speckle_x = 2 * right_idx[0] if len(right_idx) > 0 else autocorr.shape[1] // 4
    down_of_center = autocorr_y[center_y:]
    down_idx = np.where(down_of_center < threshold)[0]
    speckle_y = 2 * down_idx[0] if len(down_idx) > 0 else autocorr.shape[0] // 4
    return speckle_x, speckle_y, autocorr_x, autocorr_y

def plot_image(img, min_val, max_val, mean_val, std_val, contrast, file_name):
    fig1 = go.Figure(data=go.Heatmap(z=img, colorscale='Gray', colorbar=dict(title='Intensity'), hoverongaps=False))
    fig1.update_layout(
        title={'text': f'**Original Speckle Image: {file_name}**<br><sup>Min: {min_val:.1f}, Max: {max_val:.1f}</sup>', 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis_title='X [pixels]', yaxis_title='Y [pixels]', yaxis=dict(scaleanchor="x", scaleratio=1), autosize=True, margin=dict(l=20, r=20, t=100, b=20)
    )
    return fig1

def plot_psd(power_spectrum_log, img_shape):
    fig_psd = go.Figure(data=go.Heatmap(z=power_spectrum_log, colorscale='Viridis', colorbar=dict(title='Log10(Power)'), hoverongaps=False))
    ny, nx = img_shape
    freq_x = np.fft.fftshift(np.fft.fftfreq(nx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(ny))
    fig_psd.update_layout(
        title={'text': '**Power Spectral Density (PSD)**<br><sup>Log-scaled, Zero Frequency at Center</sup>', 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis_title='Spatial Frequency Fx [cycles/pixel]', yaxis_title='Spatial Frequency Fy [cycles/pixel]',
        xaxis=dict(tickmode='array', tickvals=np.linspace(0, nx-1, 5), ticktext=[f"{f:.2f}" for f in np.linspace(freq_x.min(), freq_x.max(), 5)]),
        yaxis=dict(tickmode='array', tickvals=np.linspace(0, ny-1, 5), ticktext=[f"{f:.2f}" for f in np.linspace(freq_y.min(), freq_y.max(), 5)], scaleanchor="x", scaleratio=1),
        autosize=True, margin=dict(l=20, r=20, t=100, b=20)
    )
    return fig_psd

def plot_autocorr_cross_sections(autocorr_x, autocorr_y, speckle_x, speckle_y):
    fig2 = make_subplots(rows=2, cols=1, subplot_titles=('Autocorrelation - X Cross-section', 'Autocorrelation - Y Cross-section'))
    threshold = 0.5 
    x_axis = np.arange(len(autocorr_x)) - len(autocorr_x) // 2
    fig2.add_trace(go.Scatter(x=x_axis, y=autocorr_x, mode='lines', name='X Autocorr', line=dict(color='blue')), row=1, col=1)
    fig2.add_shape(type="line", x0=x_axis.min(), y0=threshold, x1=x_axis.max(), y1=threshold, line=dict(color="red", width=2, dash="dash"), row=1, col=1)
    fig2.add_vline(x=speckle_x/2, line_width=2, line_dash="dot", line_color="green", row=1, col=1)
    fig2.add_vline(x=-speckle_x/2, line_width=2, line_dash="dot", line_color="green", row=1, col=1)
    y_axis = np.arange(len(autocorr_y)) - len(autocorr_y) // 2
    fig2.add_trace(go.Scatter(x=y_axis, y=autocorr_y, mode='lines', name='Y Autocorr', line=dict(color='blue')), row=2, col=1)
    fig2.add_shape(type="line", x0=y_axis.min(), y0=threshold, x1=y_axis.max(), y1=threshold, line=dict(color="red", width=2, dash="dash"), row=2, col=1)
    fig2.add_vline(x=speckle_y/2, line_width=2, line_dash="dot", line_color="green", row=2, col=1)
    fig2.add_vline(x=-speckle_y/2, line_width=2, line_dash="dot", line_color="green", row=2, col=1)
    fig2.update_xaxes(title_text='Lag X [pixels]', row=1, col=1, range=[-len(autocorr_x)//4, len(autocorr_x)//4])
    fig2.update_yaxes(title_text='Autocorrelation', row=1, col=1)
    fig2.update_xaxes(title_text='Lag Y [pixels]', row=2, col=1, range=[-len(autocorr_y)//4, len(autocorr_y)//4])
    fig2.update_yaxes(title_text='Autocorrelation', row=2, col=1)
    fig2.update_layout(height=800, width=800, title_text="**Speckle Autocorrelation Cross-sections (FWHM)**", showlegend=False, margin=dict(l=20, r=20, t=80, b=20))
    return fig2

# --- Streamlit Application Layout ---

st.set_page_config(
    page_title="Speckle & Particle Image Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Interactive Speckle & Particle Image Analyzer")
st.markdown("Use the sidebar to load two separate images: one for Speckle Analysis and one for Particle Counting.")

# --- Image Selection Logic ---
image_files = sorted(glob.glob("*.png"))
selected_speckle_content = None
selected_speckle_name = None
uploaded_particle_file = None # New variable for particle image

# --- Sidebar Setup ---
st.sidebar.header("Speckle Image (Primary Analysis)")
source_option = st.sidebar.radio(
    "Load Speckle Image:",
    ("Upload a new file", "Select from deployed files"),
    index=0 if not image_files else 1
)

if source_option == "Select from deployed files":
    if image_files:
        selected_speckle_name = st.sidebar.selectbox("Choose a PNG file:", options=image_files, index=0)
        selected_speckle_content = selected_speckle_name
    else:
        st.sidebar.warning("No PNG files found in the app directory.")
        source_option = "Upload a new file"

if source_option == "Upload a new file":
    uploaded_speckle_file = st.sidebar.file_uploader(
        "Upload a new PNG/TIF speckle image",
        type=["png", "tif", "tiff"],
        help="Upload your 12-bit (stored as 16-bit) speckle image."
    )
    if uploaded_speckle_file is not None:
        selected_speckle_name = uploaded_speckle_file.name
        selected_speckle_content = io.BytesIO(uploaded_speckle_file.getvalue())

st.sidebar.markdown("---")
st.sidebar.header("Particle Image (Independent Upload)")
uploaded_particle_file = st.sidebar.file_uploader(
    "Upload Image for Particle Counting",
    type=["png", "tif", "tiff"],
    key="particle_uploader",
    help="Upload a separate image for particle detection using Hough Circles."
)

# --- Analysis Execution ---

# ----------------------------------------------------
# 1. Speckle Analysis Execution
# ----------------------------------------------------
if selected_speckle_content is not None:
    st.header(f"Analyzing Speckle Image: {selected_speckle_name}")
    try:
        img = load_image(selected_speckle_content)
        if img is None: st.error("Could not load speckle image data."); st.stop()
            
        # Calculate statistics
        min_val, max_val, mean_val, std_val, contrast = calculate_statistics(img)
        
        # Calculate autocorrelation and Power Spectral Density
        with st.spinner("Calculating 2D Autocorrelation and Power Spectral Density..."):
            autocorr, power_spectrum_log = calculate_autocorrelation_fft(img)
        
        # Calculate speckle size
        speckle_x, speckle_y, autocorr_x, autocorr_y = find_speckle_size(autocorr)
        
        st.subheader("Results and Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Min Pixel Value", f"{min_val:.1f}")
        col2.metric("Max Pixel Value", f"{max_val:.1f}")
        col3.metric("Mean Intensity (μ)", f"{mean_val:.2f}")
        col4.metric("Std. Deviation (σ)", f"{std_val:.2f}")
        col5.metric("Contrast (σ/μ)", f"{contrast:.4f}")
        
        st.markdown("---")
        
        col_x, col_y = st.columns(2)
        col_x.metric("Speckle Size (X)", f"{speckle_x:.1f} pixels")
        col_y.metric("Speckle Size (Y)", f"{speckle_y:.1f} pixels")
        
        st.markdown("---")
        
        # --- Display Interactive Figures ---
        st.subheader("1. Original Image and Pixel Map (Plotly Heatmap)")
        fig1 = plot_image(img, min_val, max_val, mean_val, std_val, contrast, selected_speckle_name)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("2. Power Spectral Density (PSD)")
        fig_psd = plot_psd(power_spectrum_log, img.shape)
        st.plotly_chart(fig_psd, use_container_width=True)

        st.subheader("3. Autocorrelation Cross-sections (FWHM)")
        fig2 = plot_autocorr_cross_sections(autocorr_x, autocorr_y, speckle_x, speckle_y)
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"An unexpected error occurred during speckle analysis: {e}")
        st.warning("Please check image format or try another file.")

# ----------------------------------------------------
# 2. Particle Counting Execution
# ----------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<hr style='border: 4px solid #007bff;'>", unsafe_allow_html=True)

st.header("⚪ Particle Counting Analysis")

if uploaded_particle_file is not None:
    particle_content = io.BytesIO(uploaded_particle_file.getvalue())
    particle_name = uploaded_particle_file.name
    
    st.subheader(f"Analyzing Particle Image: {particle_name}")
    
    try:
        # Load the particle image (assuming it's a TIF/PNG handled by load_image)
        particle_img = load_image(particle_content)
        if particle_img is None: st.error("Could not load particle image data."); st.stop()

        # --- Interactive Slider Setup ---
        st.markdown("Adjust the detection sensitivity using the sliders below:")
        col_p1, col_p2, col_minr, col_maxr = st.columns(4)
        
        param1 = col_p1.slider('Param1 (Canny Edge Thresh)', min_value=10, max_value=150, step=5, value=60, 
                               key='p1', help='High threshold for the Canny edge detector (higher = stricter edge detection).')
        param2 = col_p2.slider('Param2 (Accumulator Thresh)', min_value=5, max_value=30, step=1, value=14, 
                               key='p2', help='Accumulator threshold for circle detection (lower = more sensitive, more false positives).')
        minRadius = col_minr.slider('Min Radius (px)', min_value=1, max_value=20, step=1, value=5, 
                                     key='minr', help='Minimum particle size to detect.')
        maxRadius = col_maxr.slider('Max Radius (px)', min_value=5, max_value=30, step=1, value=10, 
                                     key='maxr', help='Maximum particle size to detect.')

        with st.spinner(f"Detecting circles with P1={param1}, P2={param2}..."):
            particle_count, color_img, xs, ys, rs = analyze_particles_hough(
                particle_img, param1, param2, minRadius, maxRadius
            )

        # --- Display Results ---
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("🔴 Detected Particle Count", f"{particle_count}", delta_color="off")
        
        # Calculate total area (optional)
        if particle_count > 0:
            total_area_px2 = np.sum([math.pi * (r ** 2) for r in rs])
            col_m2.metric("Total Particle Area (px²)", f"{total_area_px2:.1f}", delta_color="off")

        # Plotly Display
        st.subheader("Detected Particles Map (Hough Circles)")
        if color_img is not None:
            fig_particles = px.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB), 
                                     title=f"Detected Particles: {particle_count}")
            fig_particles.update_layout(
                xaxis=dict(visible=False), 
                yaxis=dict(visible=False, scaleanchor="x", scaleratio=1), 
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_particles, use_container_width=True)
            
            # Optional: Display histogram of detected radii
            if len(rs) > 0:
                fig_hist = px.histogram(rs, nbins=maxRadius-minRadius+1, 
                                        title="Particle Radius Distribution", 
                                        labels={'value': 'Radius (pixels)', 'count': 'Frequency'})
                st.plotly_chart(fig_hist, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during particle analysis: {e}")
        st.warning("Ensure the uploaded file is a valid image and all required libraries are installed.")

else:
    st.info("Please upload a file to the **Particle Image** section in the sidebar to start particle counting.")
