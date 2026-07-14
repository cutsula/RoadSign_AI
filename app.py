import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

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
    ["Upload Gambar", "Ambil Gambar", "Real-time"],
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

elif jenis_input == "Ambil Gambar":
    st.info(
        "📱 Tekan tombol di bawah untuk mengaktifkan kamera. "
        "Di HP, ini akan membuka kamera bawaan browser (bisa pilih kamera depan/belakang)."
    )

    foto = st.camera_input("Ambil foto rambu lalu lintas")

    if foto is not None:
        image = Image.open(foto).convert("RGB")
        img_array = np.array(image)

        st.markdown("---")
        st.subheader("Hasil Deteksi")
        tampilkan_hasil_dua_model(img_array)

        st.caption(
            "Tips: ambil ulang foto (tombol kamera di atas) untuk deteksi baru — "
            "st.camera_input akan otomatis refresh hasil setiap foto baru diambil."
        )

elif jenis_input == "Real-time":
    st.info(
        "🎥 Mode real-time memproses video langsung dari kamera. "
        "Karena keterbatasan CPU di server, hanya 1 model yang dijalankan "
        "sekaligus supaya video tetap lancar."
    )

    pilihan_model_rt = st.selectbox("Pilih model untuk real-time:", nama_model)
    idx_model_rt = nama_model.index(pilihan_model_rt)

    arah_kamera = st.radio(
        "Pilih kamera:",
        ["Belakang", "Depan"],
        horizontal=True,
        help="Kamera belakang biasanya lebih cocok untuk memotret rambu di jalan.",
    )
    facing_mode = "environment" if arah_kamera == "Belakang" else "user"

    # Simpan referensi model & confidence terkini di session_state supaya bisa
    # diakses dari dalam VideoProcessor (berjalan di thread terpisah) tanpa
    # perlu rebuild processor tiap kali pilihan berubah.
    st.session_state["_rt_model_idx"] = idx_model_rt
    st.session_state["_rt_conf"] = conf_threshold

    class YOLOVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img_bgr = frame.to_ndarray(format="bgr24")

            model_idx = st.session_state.get("_rt_model_idx", 0)
            conf = st.session_state.get("_rt_conf", 0.25)
            model = daftar_model[model_idx]

            res = model.predict(source=img_bgr, imgsz=640, conf=conf, verbose=False)
            annotated_bgr = res[0].plot()

            return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # key harus ikut berubah sesuai facing_mode, supaya widget di-restart
    # dengan constraint kamera yang baru saat user ganti Depan/Belakang.
    webrtc_streamer(
        key=f"realtime-yolo-{facing_mode}",
        video_processor_factory=YOLOVideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": {"facingMode": {"ideal": facing_mode}},
            "audio": False,
        },
        async_processing=True,
    )
