import locale
import cv2
from ultralytics import YOLOv10
import os
import numpy as np
import streamlit as st
from torchvision import transforms

IMG_SIZE = 640
BATCH_SIZE = 64
CONF_THRESHOLD = 0.3
script_dir = os.getcwd()
locale.getpreferredencoding = lambda: "UTF-8"


TRAINED_MODEL_PATH = os.path.join(
    script_dir, 'runs\\detect\\train\\weights\\best.pt')
model = YOLOv10(TRAINED_MODEL_PATH)

# IMAGE_URL = 'https://ips-dc.org/wp-content/uploads/2022/05/Black-Workers-Need-a-Bill-of-Rights.jpeg'

# results = model.predict(source=IMAGE_URL,
#                         imgsz=IMG_SIZE,
#                         conf=CONF_THRESHOLD)
# annotated_img = results[0].plot()
# window_name = 'image'

# cv2.imshow(window_name, annotated_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


st.set_page_config(layout="wide")
st.title("Helmet detecting App")

col1, col2 = st.columns(2)
with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])
with col2:
    st.header("Helmet Detection")
    if uploaded_file is not None:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([transforms.ToPILImage(),
                                        # Resize to the size required by your model
                                        transforms.Resize((640, 640)),
                                        transforms.ToTensor()])
        input_image = transform(image_rgb).unsqueeze(0)

        results = model.predict(source=input_image,
                                imgsz=IMG_SIZE, conf=CONF_THRESHOLD)
        annotated_img = results[0].plot()
        st.image(annotated_img, caption='Uploaded Image',
                 use_column_width=True)
