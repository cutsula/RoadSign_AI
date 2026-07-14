import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="RoadSign AI", layout="wide")
st.title("🚦 RoadSign AI (YOLOv12 SLE + RGP)")
st.write("Perbandingan performa model **YOLOv12 + RGP** dan **YOLOv12 SLE + RGP** secara real-time")

# --- LOAD KEDUA MODEL ---
@st.cache_resource
def load_models():
    m_c = YOLO("models/yolov12n_RGP_S1.pt")          # Model C
    m_d = YOLO("models/yolov12n_SLE_RGP_S1.pt")      # Model D
    return m_c, m_d

with st.spinner("🚀 Sedang memuat 4 model YOLOv12 ke memori..."):
    model_c, model_d = load_models()

# List bantuan untuk perulangan
daftar_model = [model_c, model_d]
nama_model = [
    "Model A: YOLOv12 RGP",
    "Model B: YOLOv12 SLE RGP"
]

# --- PILIHAN JENIS INPUT ---
st.subheader("Pengaturan Input")
jenis_input = st.radio(
    "Pilih jenis input:",
    ["Upload Gambar", "Kamera"],
    horizontal=True
)

# Parameter ambang batas keyakinan (Confidence Threshold)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.1, 0.05)


def predict_frame(model, frame, is_image_menu=False):
    import matplotlib.pyplot as plt
    import io

    t_start = time.time()

    # 1. Model Ultralytics mengharapkan array numpy dalam format BGR
    #    (sama seperti hasil cv2.imread). Semua sumber gambar di app ini
    #    (PIL.Image.open dari upload & st.camera_input) berformat RGB,
    #    jadi HARUS dikonversi ke BGR dulu sebelum masuk ke model.
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 2. Jalankan prediksi
    res = model.predict(source=frame_bgr, imgsz=640, conf=conf_threshold, verbose=False)
    fps = 1.0 / (time.time() - t_start)
    jumlah_rambu = len(res[0].boxes)

    # 3. Gambar ulang pakai Matplotlib + Annotate Panah (dipakai untuk gambar & kamera)
    if is_image_menu:
        # Matikan teks bawaan YOLO
        im_array = res[0].plot(labels=False, conf=False, line_width=2)
        img_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

        # Buat figure temporary Matplotlib
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(img_rgb)
        ax.axis('off')

        offsets = [(120, -60), (-120, -60), (120, 60), (-120, 60)]

        for idx, box in enumerate(res[0].boxes):
            coords = box.xyxy[0].tolist()
            cx = (coords[0] + coords[2]) / 2
            cy = (coords[1] + coords[3]) / 2

            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            conf_val = box.conf[0].item()

            label_text = f"{cls_name} ({conf_val:.2f})"
            offset_x, offset_y = offsets[idx % len(offsets)]

            ax.annotate(
                label_text, xy=(cx, cy), xytext=(cx + offset_x, cy + offset_y),
                fontsize=9, fontweight='bold', color='black', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#F39C12', edgecolor='none', alpha=0.9),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='#E74C3C', lw=2)
            )

        plt.tight_layout()

        # Mengubah hasil plot matplotlib menjadi gambar yang bisa dibaca Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plot_img = Image.open(buf)
        plt.close(fig)  # Hapus objek agar hemat memori
    else:
        # Untuk Video lokal, penggambaran standar YOLO agar FPS tetap kencang
        plot_img = cv2.cvtColor(res[0].plot(), cv2.COLOR_BGR2RGB)

    # Kumpulkan informasi teks untuk sidebar/keterangan bawah
    detail_teks = []
    for box in res[0].boxes:
        detail_teks.append(f"- {model.names[int(box.cls[0].item())]} ({box.conf[0].item():.2%})")

    return plot_img, fps, jumlah_rambu, detail_teks


def tampilkan_hasil_dua_model(img_array):
    """Jalankan kedua model pada satu gambar dan tampilkan berdampingan."""
    col1, col2 = st.columns(2)
    koloms = [col1, col2]

    for i, col in enumerate(koloms):
        with col:
            st.markdown(f"### {nama_model[i]}")
            plot_img, fps, jml, info = predict_frame(daftar_model[i], img_array, is_image_menu=True)
            st.image(plot_img, use_container_width=True)
            st.metric("Kecepatan Inferensi", f"{fps:.1f} FPS")
            st.write(f"Terdeteksi: **{jml} rambu**")
            for teks in info[:5]:  # Batasi tampilkan maksimal 5 objek teratas agar tidak kepanjangan
                st.caption(teks)


# --- LOGIKA BERDASARKAN JENIS INPUT ---

if jenis_input == "Upload Gambar":
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        st.image(image, caption="Gambar Asli", width=300)

        if st.button("Mulai Deteksi Model 🔍"):
            tampilkan_hasil_dua_model(img_array)

elif jenis_input == "Kamera":
    st.info(
        "📱 Tekan tombol di bawah untuk mengaktifkan kamera. "
    )

    foto = st.camera_input("Ambil foto rambu lalu lintas")

    if foto is not None:
        image = Image.open(foto).convert("RGB")
        img_array = np.array(image)

        st.markdown("---")
        st.subheader("Hasil Deteksi")
        tampilkan_hasil_dua_model(img_array)

       
