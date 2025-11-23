import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
import glob

# --- Existing Analysis Functions (Modified for Web/Streamlit) ---

@st.cache_data
def load_image(file_content):
    """
    Load 12-bit PNG image (stored as 16-bit) from file content (BytesIO or actual file path) 
    and convert to numpy array.
    """
    try:
        if isinstance(file_content, io.BytesIO):
            # This handles Streamlit's UploadedFile object content
            img = Image.open(file_content)
        elif isinstance(file_content, str) and os.path.exists(file_content):
            # This handles local file paths (for pre-uploaded files)
            img = Image.open(file_content)
        else:
            st.error("Invalid file content provided to load_image.")
            return None

        img_array = np.array(img, dtype=np.float64)

        # Check if values are shifted (16-bit container for 12-bit data)
        if img_array.max() > 4095:
            st.info("Detected bit-shifted image, converting from 16-bit to 12-bit (dividing by 16.0)...")
            img_array = img_array / 16.0  # Right shift by 4 bits
        
        return img_array

    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def calculate_statistics(img_array):
    """Calculate min, max, mean, std, and contrast of the image."""
    min_val = np.min(img_array)
    max_val = np.max(img_array)
    mean_val = np.mean(img_array)
    std_val = np.std(img_array)
    # Contrast definition: std/mean
    contrast = std_val / mean_val if mean_val != 0 else 0
    
    return min_val, max_val, mean_val, std_val, contrast

@st.cache_data
def calculate_autocorrelation_fft(img_array):
    """Calculate 2D autocorrelation using FFT method."""
    img_norm = img_array - np.mean(img_array)
    f_img = np.fft.fft2(img_norm)
    power_spectrum = np.abs(f_img) ** 2
    autocorr = np.fft.ifft2(power_spectrum).real
    autocorr = np.fft.fftshift(autocorr)
    autocorr = autocorr / autocorr.max()
    return autocorr

def find_speckle_size(autocorr):
    """Estimate speckle size from autocorrelation function."""
    center_y, center_x = np.array(autocorr.shape) // 2
    autocorr_x = autocorr[center_y, :]
    autocorr_y = autocorr[:, center_x]
    
    threshold = 1/np.e**2
    
    # X direction
    right_idx = np.where(autocorr_x[center_x:] < threshold)[0]
    speckle_x = 2 * right_idx[0] if len(right_idx) > 0 else autocorr.shape[1] // 4
    
    # Y direction
    down_idx = np.where(autocorr_y[center_y:] < threshold)[0]
    speckle_y = 2 * down_idx[0] if len(down_idx) > 0 else autocorr.shape[0] // 4
    
    return speckle_x, speckle_y, autocorr_x, autocorr_y

def plot_image(img, min_val, max_val, mean_val, std_val, contrast, file_name):
    """Generates the interactive Plotly figure for the original image."""
    fig1 = go.Figure(data=go.Heatmap(
        z=img,
        colorscale='Gray', 
        colorbar=dict(title='Intensity'),
        hoverongaps=False 
    ))
    
    fig1.update_layout(
        title={
            'text': f'**Original Speckle Image: {file_name}**<br><sup>Min: {min_val:.1f}, Max: {max_val:.1f}</sup>',
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        xaxis_title='X [pixels]',
        yaxis_title='Y [pixels]',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        autosize=True,
        margin=dict(l=20, r=20, t=100, b=20)
    )
    return fig1

def plot_autocorr_cross_sections(autocorr_x, autocorr_y, speckle_x, speckle_y):
    """Generates the interactive Plotly subplots for autocorrelation cross-sections."""
    fig2 = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Autocorrelation - X Cross-section', 'Autocorrelation - Y Cross-section')
    )
    
    threshold = 1/np.e**2
    
    # 1. X Cross-section (Row 1)
    x_axis = np.arange(len(autocorr_x)) - len(autocorr_x) // 2
    fig2.add_trace(go.Scatter(x=x_axis, y=autocorr_x, mode='lines', name='X Autocorr', line=dict(color='blue')), row=1, col=1)
    fig2.add_shape(type="line", x0=x_axis.min(), y0=threshold, x1=x_axis.max(), y1=threshold,
                    line=dict(color="red", width=2, dash="dash"), row=1, col=1)
    fig2.add_vline(x=speckle_x/2, line_width=2, line_dash="dot", line_color="green", row=1, col=1)
    fig2.add_vline(x=-speckle_x/2, line_width=2, line_dash="dot", line_color="green", row=1, col=1)
    
    # 2. Y Cross-section (Row 2)
    y_axis = np.arange(len(autocorr_y)) - len(autocorr_y) // 2
    fig2.add_trace(go.Scatter(x=y_axis, y=autocorr_y, mode='lines', name='Y Autocorr', line=dict(color='blue')), row=2, col=1)
    fig2.add_shape(type="line", x0=y_axis.min(), y0=threshold, x1=y_axis.max(), y1=threshold,
                    line=dict(color="red", width=2, dash="dash"), row=2, col=1)
    fig2.add_vline(x=speckle_y/2, line_width=2, line_dash="dot", line_color="green", row=2, col=1)
    fig2.add_vline(x=-speckle_y/2, line_width=2, line_dash="dot", line_color="green", row=2, col=1)
    
    # Update axes titles and ranges
    fig2.update_xaxes(title_text='Lag X [pixels]', row=1, col=1, range=[-len(autocorr_x)//4, len(autocorr_x)//4])
    fig2.update_yaxes(title_text='Autocorrelation', row=1, col=1)
    fig2.update_xaxes(title_text='Lag Y [pixels]', row=2, col=1, range=[-len(autocorr_y)//4, len(autocorr_y)//4])
    fig2.update_yaxes(title_text='Autocorrelation', row=2, col=1)
    
    fig2.update_layout(
        height=800, 
        width=800, 
        title_text="**Speckle Autocorrelation Cross-sections (Width at 1/e²)**",
        showlegend=False,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig2

# --- Streamlit Application Layout ---

st.set_page_config(
    page_title="Speckle Image Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Interactive Speckle Image Analyzer")
st.markdown("Upload a **12-bit PNG** image or select one from the app folder for analysis. The tool calculates image statistics and speckle size via 2D Autocorrelation.")

# --- Image Selection Logic ---
image_files = sorted(glob.glob("*.png"))
selected_file_content = None
selected_file_name = None

st.sidebar.header("Image Source Selection")
source_option = st.sidebar.radio(
    "How would you like to load the image?",
    ("Upload a new file", "Select from deployed files"),
    index=0 if not image_files else 1 # Default to selecting deployed if files exist
)

if source_option == "Select from deployed files":
    if image_files:
        selected_file_name = st.sidebar.selectbox(
            "Choose a PNG file from the app directory:",
            options=image_files,
            index=0
        )
        # For deployed files, we pass the file path. load_image will handle opening it.
        if selected_file_name:
            selected_file_content = selected_file_name
        st.sidebar.caption(f"Note: These {len(image_files)} files were committed to the repository.")
    else:
        st.sidebar.warning("No PNG files found in the app directory. Please upload one.")
        # If no files are found locally, switch to upload mode
        source_option = "Upload a new file" 


if source_option == "Upload a new file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a new PNG image file", 
        type="png", 
        help="Upload your 12-bit (stored as 16-bit) PNG speckle image."
    )
    if uploaded_file is not None:
        selected_file_name = uploaded_file.name
        # For uploaded files, we pass the file content buffer.
        selected_file_content = io.BytesIO(uploaded_file.getvalue())
    st.sidebar.caption("Uploaded files are processed securely but are not saved permanently.")


# --- Analysis Execution ---

if selected_file_content is not None:
    
    st.header(f"Analyzing: {selected_file_name}")
    
    # Use a try-except block for robust error handling
    try:
        # Load and process the image
        img = load_image(selected_file_content)
        
        if img is None:
            st.error("Could not load image data.")
            st.stop()
            
        # Calculate statistics
        min_val, max_val, mean_val, std_val, contrast = calculate_statistics(img)
        
        # Calculate autocorrelation
        with st.spinner("Calculating 2D Autocorrelation (this may take a moment)..."):
            autocorr = calculate_autocorrelation_fft(img)
        
        # Calculate speckle size and cross-sections
        speckle_x, speckle_y, autocorr_x, autocorr_y = find_speckle_size(autocorr)
        
        st.subheader("Results and Statistics")
        
        # Display results in a clear table/metric format
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Min Pixel Value", f"{min_val:.1f}")
        col2.metric("Max Pixel Value", f"{max_val:.1f}")
        col3.metric("Mean Intensity (μ)", f"{mean_val:.2f}")
        col4.metric("Std. Deviation (σ)", f"{std_val:.2f}")
        col5.metric("Contrast (σ/μ)", f"{contrast:.4f}")
        
        st.markdown("---")
        
        col_x, col_y = st.columns(2)
        col_x.metric("Speckle Size (X)", f"{speckle_x:.1f} pixels", help="Calculated as the full width at $1/e^2$ of the central peak.")
        col_y.metric("Speckle Size (Y)", f"{speckle_y:.1f} pixels", help="Calculated as the full width at $1/e^2$ of the central peak.")
        
        st.markdown("---")
        
        # --- Display Interactive Figures ---
        
        st.subheader("1. Original Image and Pixel Map (Plotly Heatmap)")
        fig1 = plot_image(img, min_val, max_val, mean_val, std_val, contrast, selected_file_name)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("2. Autocorrelation Cross-sections (Plotly Subplots)")
        fig2 = plot_autocorr_cross_sections(autocorr_x, autocorr_y, speckle_x, speckle_y)
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"An unexpected error occurred during analysis: {e}")
        st.warning("Please check the console for detailed error information or try another image.")

else:
    st.info("Please select an image source to begin the speckle analysis.")
