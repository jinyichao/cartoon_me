import streamlit as st
from rembg import remove
from PIL import Image
from io import BytesIO

st.set_page_config(layout="wide", page_title="Image Background Remover")

st.write("Remove background from your image")
st.write(
    "Try uploading an image to watch the background magically removed. "
    "Full quality images can be downloaded from the sidebar."
)
st.sidebar.write("Upload and download")


# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def fix_image(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    fixed = remove(image)
    col2.write("Fixed Image :wrench:")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    fix_image(upload=my_upload)
else:
    fix_image("./zebra.jpg")
