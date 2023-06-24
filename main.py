import streamlit as st
from PIL import Image
from io import BytesIO
from time import time

from img2img import ImageConvert

st.set_page_config(layout="wide", page_title="Get your cartoon character")
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.title(f"_Get your cartoon character here_")
st.write("Just upload your portrait photo, and check the generated cartoon character.")
st.write("If this isn't what you want, just try rerun or re-upload another photo, have fun!")
st.sidebar.write("Upload and download")


@st.cache_resource
def get_model():
    return ImageConvert()


def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def generate_image(upload, model):
    image = Image.open(upload)
    col1.write("Original Image")
    col1.image(image)

    fixed = model.generate_image(image)
    col2.write("Generated Image")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button(
        "Download generated image",
        convert_image(fixed),
        f"{int(time()*100000)}.png",
        "image/png",
    )


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

imageConvertModel = get_model()
if my_upload is not None:
    generate_image(my_upload, imageConvertModel)
else:
    col1.write("Original Image")
    col1.image(Image.open("./test.png"))
    col2.write("Generated Image")
    col2.image(Image.open("./example.png"))
    st.sidebar.markdown("\n")

st.write(
    f"**Disclaimer:** we promise all the uploaded and generated images will be completely deleted "
    f"as soon as you close the browser."
)

# with st.expander(f"**_Not what you want? Let's make it better!_**"):
#     options_col, selection_col = st.columns(2)
