import streamlit as st
from PIL import Image
from io import BytesIO
from time import time
from rembg import remove
from img2img import ImageConvert

default_prompt = "((best quality)), (detailed), cartoon"
sessions = ["uploading", "rerun", "gender", "body", "style", "hair", "bg"]

st.set_page_config(layout="wide", page_title="Cartoonize Everything|万物皆可萌")
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.title(f"_Cartoonize Everything|万物皆可萌_")
st.sidebar.write("Upload and download")


@st.cache_resource
def get_model():
    return ImageConvert()


def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def generate_image(img, model):
    prompt = default_prompt
    for s in sessions:
        prompt += append_prompt(s)

    if st.session_state["bg"] is not None and st.session_state["bg"] != "default":
        img = remove(img)
    generated = model.generate_image(img, prompt)
    st.sidebar.markdown("\n")
    st.sidebar.download_button(
        "Download generated image",
        convert_image(generated),
        f"{int(time()*100000)}.png",
        "image/png",
    )
    return generated


def upload_trigger():
    st.session_state["uploading"] = True


def rerun_trigger():
    st.session_state["rerun"] = True


def init_sessions():
    for s in sessions:
        if s not in st.session_state:
            st.session_state[s] = None


def append_prompt(item, prefix="", suffix=""):
    res = ""
    value = st.session_state[item]
    if isinstance(value, str) and value != "default":
        res = f", {prefix}{value}{suffix}"
    return res


imageConvertModel = get_model()

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], on_change=upload_trigger)
col1.write("Original Image")
col2.write("Generated Image")

init_sessions()
if "generated_image" not in st.session_state:
    st.session_state["generated_image"] = Image.open("./example.png")
st.write(
    f"**Disclaimer:** we promise all the uploaded and generated images will be completely deleted "
    f"as soon as you close the browser."
)

initial_image = my_upload if my_upload else "./test.png"
col1.image(Image.open(initial_image))

if st.session_state["uploading"] or st.session_state["rerun"]:
    st.session_state["generated_image"] = generate_image(Image.open(initial_image), imageConvertModel)
    st.session_state["uploading"] = False
    st.session_state["rerun"] = False
col2.image(st.session_state["generated_image"])

with st.expander(f"**_Not what you want? Let's make it better!_**"):
    sub_col1, sub_col2 = st.columns(2)
    st.session_state["gender"] = sub_col1.selectbox("gender", ("default", "girl", "boy"))
    st.session_state["bg"] = sub_col2.selectbox("background", ("default", "seaside", "city landscape", "blue sky", "flowers"))
    st.session_state["style"] = sub_col1.selectbox("style", ("default", "vintage", "sci-fi", "realistic"))
    st.session_state["hair"] = sub_col2.selectbox("hair", ("default", "bangs hair", "mohawk", "ponytail", "long hair"))
    rerun = st.button("Re-generate", on_click=rerun_trigger)
