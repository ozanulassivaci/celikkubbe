#!/usr/bin/env python3
"""
Intel RealSense D435i - Tak-Calistir Test Betigi

Kamerayi USB 3.0 porta tak, sonra:

    cd ~/projeler/celikkubbe
    source .venv/bin/activate
    python kamera_test.py

Salt okunur: hicbir ayar degistirmez, firmware'e dokunmaz.
Tek yazdigi sey ~/projeler/celikkubbe/kamera_ornek.png (--kaydet ile).

Secenekler:
    --kaydet     Ornek RGB + derinlik goruntusu kaydet
    --sure N     Akis testi suresi (varsayilan 5 saniye)
"""

import sys
import time

KAYDET = "--kaydet" in sys.argv
SURE = 5
if "--sure" in sys.argv:
    try:
        SURE = int(sys.argv[sys.argv.index("--sure") + 1])
    except (IndexError, ValueError):
        pass

G, R, Y, D, X = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"
sonuclar = []


def rapor(ad, durum, detay=""):
    etiket = {True: f"{G}[PASS]{X}", False: f"{R}[FAIL]{X}", None: f"{Y}[BILGI]{X}"}[durum]
    print(f"{etiket} {ad}")
    for satir in str(detay).strip().splitlines():
        if satir.strip():
            print(f"       {D}{satir}{X}")
    sonuclar.append((ad, durum))


print(f"\n{'=' * 64}")
print("  INTEL REALSENSE D435i - TAK CALISTIR TESTI")
print(f"{'=' * 64}\n")

# --------------------------------------------------------------------
try:
    import numpy as np
    import pyrealsense2 as rs
except ImportError as e:
    print(f"{R}pyrealsense2 veya numpy yok: {e}{X}")
    print("Cozum: pip install pyrealsense2 numpy")
    sys.exit(1)

# --------------------------------------------------------------------
# 1. Cihaz bulundu mu
# --------------------------------------------------------------------
ctx = rs.context()
cihazlar = ctx.query_devices()

if len(cihazlar) == 0:
    rapor(
        "Cihaz algilandi",
        False,
        "Kamera bulunamadi. Kontrol listesi:\n"
        "  - USB kablosu takili mi (MAVI/USB 3.0 porta)\n"
        "  - lsusb | grep -i intel   -> gorunuyorsa udev sorunu\n"
        "  - groups                  -> plugdev listede mi\n"
        "  - Baska bir uygulama kamerayi tutuyor olabilir",
    )
    print(f"\n{R}Test durduruldu.{X}\n")
    sys.exit(1)

dev = cihazlar[0]


def bilgi(alan, varsayilan="?"):
    try:
        return dev.get_info(alan)
    except Exception:
        return varsayilan


ad = bilgi(rs.camera_info.name)
seri = bilgi(rs.camera_info.serial_number)
fw = bilgi(rs.camera_info.firmware_version)
usb = bilgi(rs.camera_info.usb_type_descriptor)

rapor("Cihaz algilandi", True, f"model    : {ad}\n" f"seri no  : {seri}\n" f"firmware : {fw}")

# --------------------------------------------------------------------
# 2. USB surumu - KRITIK
# --------------------------------------------------------------------
usb3 = usb.startswith("3")
rapor(
    "USB 3.x baglantisi",
    usb3,
    f"USB {usb}"
    + (
        ""
        if usb3
        else "\nDIKKAT: USB 2.1'de cozunurluk ve FPS ciddi dusuyor.\n"
        "Kabloyu MAVI porta tak veya kabloyu degistir.\n"
        "Uzatma kablosu kullaniyorsan AKTIF (repeater) olmali."
    ),
)

# --------------------------------------------------------------------
# 3. Sensorler
# --------------------------------------------------------------------
try:
    sensorler = dev.query_sensors()
    isimler = []
    for s in sensorler:
        try:
            isimler.append(s.get_info(rs.camera_info.name))
        except Exception:
            isimler.append("?")
    imu_var = any("Motion" in i for i in isimler)
    rapor("Sensorler listelendi", True, "\n".join(isimler))
    rapor(
        "IMU modulu (D435i)",
        imu_var,
        "Motion Module bulundu" if imu_var else "Motion Module YOK - bu D435 olabilir, D435i degil",
    )
except Exception as e:
    rapor("Sensorler listelendi", False, e)

# --------------------------------------------------------------------
# 4. Derinlik olcegi
# --------------------------------------------------------------------
try:
    depth_sensor = dev.first_depth_sensor()
    olcek = depth_sensor.get_depth_scale()
    rapor("Derinlik olcegi", True, f"{olcek} m/birim  (ham deger x {olcek} = metre)")
except Exception as e:
    rapor("Derinlik olcegi", False, e)

# --------------------------------------------------------------------
# 5. Akis testi: derinlik + renk
# --------------------------------------------------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_device(seri)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = None
son_renk = son_derinlik = None

try:
    profile = pipeline.start(config)

    # Otomatik pozlama otursun diye ilk kareleri at
    for _ in range(15):
        pipeline.wait_for_frames(5000)

    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < SURE:
        frames = pipeline.wait_for_frames(5000)
        d = frames.get_depth_frame()
        c = frames.get_color_frame()
        if d and c:
            n += 1
            son_derinlik, son_renk = d, c

    gecen = time.perf_counter() - t0
    fps = n / gecen
    rapor("Akis testi (640x480 @30)", fps > 25, f"{n} kare / {gecen:.1f} s  =  {fps:.1f} FPS")
except Exception as e:
    rapor("Akis testi", False, e)

# --------------------------------------------------------------------
# 6. Derinlik verisi anlamli mi
# --------------------------------------------------------------------
if son_derinlik:
    try:
        dizi = np.asanyarray(son_derinlik.get_data())
        h, w = dizi.shape
        merkez = son_derinlik.get_distance(w // 2, h // 2)
        gecerli = np.count_nonzero(dizi) / dizi.size * 100

        ok = gecerli > 30
        rapor(
            "Derinlik verisi",
            ok,
            f"cozunurluk    : {w}x{h}\n"
            f"merkez mesafe : {merkez:.3f} m\n"
            f"gecerli piksel: %{gecerli:.1f}"
            + ("" if ok else "\nDusuk oran: cok yakin/parlak yuzey veya lens kapali olabilir"),
        )
    except Exception as e:
        rapor("Derinlik verisi", False, e)

# --------------------------------------------------------------------
# 7. Metadata - DKMS gerekli mi sorusunun cevabi
# --------------------------------------------------------------------
if son_derinlik:
    try:
        alanlar = {
            "frame_timestamp": rs.frame_metadata_value.frame_timestamp,
            "sensor_timestamp": rs.frame_metadata_value.sensor_timestamp,
            "actual_exposure": rs.frame_metadata_value.actual_exposure,
            "gain_level": rs.frame_metadata_value.gain_level,
        }
        destekli = [k for k, v in alanlar.items() if son_derinlik.supports_frame_metadata(v)]

        if destekli:
            rapor("Donanim metadata", None, "Mevcut: " + ", ".join(destekli) + "\nDKMS gerekmiyor.")
        else:
            rapor(
                "Donanim metadata",
                None,
                "Yok - cekirdek yamasi (librealsense2-dkms) kurulmadigi icin normal.\n"
                "Kalman icin kare varis zamanini kullan.\n"
                "Donanim zaman damgasi gerekirse DKMS + MOK kaydi yapilabilir.",
            )
    except Exception as e:
        rapor("Donanim metadata", None, e)

# --------------------------------------------------------------------
# 8. Hizalama (derinlik -> renk)
# --------------------------------------------------------------------
if profile:
    try:
        align = rs.align(rs.stream.color)
        frames = pipeline.wait_for_frames(5000)
        hizali = align.process(frames)
        hd = hizali.get_depth_frame()
        hc = hizali.get_color_frame()

        ok = bool(hd and hc)
        detay = ""
        if ok:
            dd = np.asanyarray(hd.get_data())
            cc = np.asanyarray(hc.get_data())
            detay = f"derinlik {dd.shape[1]}x{dd.shape[0]}  =  renk {cc.shape[1]}x{cc.shape[0]}"
        rapor("Hizalama (align to color)", ok, detay)
    except Exception as e:
        rapor("Hizalama", False, e)

# --------------------------------------------------------------------
# 9. Ic parametreler (kalibrasyon)
# --------------------------------------------------------------------
if profile:
    try:
        cp = profile.get_stream(rs.stream.color).as_video_stream_profile()
        i = cp.get_intrinsics()
        rapor(
            "Kamera ic parametreleri",
            True,
            f"fx={i.fx:.1f}  fy={i.fy:.1f}\n"
            f"cx={i.ppx:.1f}  cy={i.ppy:.1f}\n"
            f"model: {i.model}",
        )
    except Exception as e:
        rapor("Kamera ic parametreleri", False, e)

# --------------------------------------------------------------------
# 10. Ornek goruntu kaydet
# --------------------------------------------------------------------
if KAYDET and son_renk and son_derinlik:
    try:
        import cv2

        renk = np.asanyarray(son_renk.get_data())
        derin = np.asanyarray(son_derinlik.get_data())
        renkli_derinlik = cv2.applyColorMap(
            cv2.convertScaleAbs(derin, alpha=0.03), cv2.COLORMAP_JET
        )
        birlesik = np.hstack((renk, renkli_derinlik))
        cv2.imwrite("kamera_ornek.png", birlesik)
        rapor("Ornek goruntu kaydedildi", True, "kamera_ornek.png (sol: RGB, sag: derinlik)")
    except Exception as e:
        rapor("Ornek goruntu", False, e)

if profile:
    pipeline.stop()

# --------------------------------------------------------------------
# 11. IMU - ayri pipeline
# --------------------------------------------------------------------
try:
    imu_pipe = rs.pipeline()
    imu_cfg = rs.config()
    imu_cfg.enable_device(seri)
    imu_cfg.enable_stream(rs.stream.accel)
    imu_cfg.enable_stream(rs.stream.gyro)
    imu_pipe.start(imu_cfg)

    accel = gyro = None
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.0 and (accel is None or gyro is None):
        f = imu_pipe.wait_for_frames(2000)
        for fr in f:
            m = fr.as_motion_frame()
            if not m:
                continue
            tip = m.get_profile().stream_type()
            if tip == rs.stream.accel:
                accel = m.get_motion_data()
            elif tip == rs.stream.gyro:
                gyro = m.get_motion_data()

    imu_pipe.stop()

    if accel and gyro:
        buyukluk = (accel.x**2 + accel.y**2 + accel.z**2) ** 0.5
        ivme = f"ivme: x={accel.x:+.2f} y={accel.y:+.2f} z={accel.z:+.2f} |g|={buyukluk:.2f} m/s2"
        jiro = f"jiro: x={gyro.x:+.3f} y={gyro.y:+.3f} z={gyro.z:+.3f} rad/s"
        rapor(
            "IMU verisi",
            8.0 < buyukluk < 11.5,
            f"{ivme}\n{jiro}\n(kamera sabitken |g| ~9.81 olmali)",
        )
    else:
        rapor("IMU verisi", False, "accel/gyro karesi alinamadi")
except Exception as e:
    rapor("IMU verisi", None, f"okunamadi: {e}")

# --------------------------------------------------------------------
# OZET
# --------------------------------------------------------------------
gecen_s = sum(1 for _, d in sonuclar if d is True)
kalan = sum(1 for _, d in sonuclar if d is False)
bilgi_s = sum(1 for _, d in sonuclar if d is None)

print(f"\n{'=' * 64}")
print(f"  OZET:  {G}{gecen_s} PASS{X}   {R}{kalan} FAIL{X}   {Y}{bilgi_s} BILGI{X}")
print(f"{'=' * 64}")

if kalan:
    print(f"\n{R}Basarisiz:{X}")
    for a, d in sonuclar:
        if d is False:
            print(f"  - {a}")
    print()
    sys.exit(1)

print(f"\n{G}D435i Celikkubbe icin hazir.{X}\n")
sys.exit(0)
