import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2
from io import BytesIO

# Sidebar parameters
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#FFFFFF")
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

def normalize_image(img):
    img = img / np.max(img)
    return (img * 255).astype('uint8')

def rgb_fft(image):
    fft_images = []
    fft_images_log = []
    for i in range(3):  # RGB Channels
        channel_fft = np.fft.fftshift(np.fft.fft2(image[:, :, i]))
        fft_images.append(channel_fft)
        fft_images_log.append(np.log(np.abs(channel_fft)))
    return fft_images, fft_images_log

def inverse_fourier(image):
    final_image = [np.abs(np.fft.ifft2(channel)) for channel in image]
    return np.dstack(final_image).astype('uint8')

def create_canvas(image_array, key, height, width):
    background_image = Image.fromarray(normalize_image(image_array).astype('uint8'))
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0)",  
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=background_image,
        update_streamlit=realtime_update,
        drawing_mode=drawing_mode,
        height=height,
        width=width,
        key=key,
    )
    return canvas_result

def apply_mask(image_fft, mask):
    _, mask_binary = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
    mask_bool = mask_binary.astype(bool)
    image_fft[mask_bool] = 1
    return image_fft

def process_fft_with_mask(fft_images, canvas_images):
    result_images = []
    for fft, canvas in zip(fft_images, canvas_images):
        mask = canvas[:, :, 3]
        result_images.append(apply_mask(fft, mask))
    return result_images

def main():
    st.header("Fourier Transformation with Editable Canvas")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpeg", "png", "jpg"])
    if uploaded_file:
        # Load and display the original image
        original = Image.open(uploaded_file)
        img = np.array(original)
        st.image(img, caption="Original Image", use_column_width=True)

        # Perform Fourier Transform
        fft_images, fft_images_log = rgb_fft(img)

        st.text("Red Channel in Frequency Domain")
        canvas_r = create_canvas(fft_images_log[0], "red", img.shape[0], img.shape[1])

        st.text("Green Channel in Frequency Domain")
        canvas_g = create_canvas(fft_images_log[1], "green", img.shape[0], img.shape[1])

        st.text("Blue Channel in Frequency Domain")
        canvas_b = create_canvas(fft_images_log[2], "blue", img.shape[0], img.shape[1])

        if st.button("Get Result"):
            # Extract masks from canvases
            canvas_data = [canvas_r.image_data, canvas_g.image_data, canvas_b.image_data]
            if None in canvas_data:
                st.error("Please draw on all canvases before proceeding.")
                return

            # Apply masks to FFT images
            processed_fft = process_fft_with_mask(fft_images, canvas_data)

            # Perform Inverse Fourier Transform
            transformed = inverse_fourier(processed_fft)
            transformed_clipped = np.clip(transformed, 0, 255).astype('uint8')

            st.text("Result After Inverse Fourier Transformation")
            st.image(transformed_clipped, caption="Transformed Image", use_column_width=True)

if __name__ == "__main__":
    main()
