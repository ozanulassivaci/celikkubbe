#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Celikkubbe - Gelistirme Ortami Dogrulama Betigi

Salt okunur: hicbir sistem ayarini degistirmez, hicbir dosya yazmaz
(YOLO agirliklari haric - ilk calistirmada ~6 MB indirilir).

Kullanim:
    cd ~/projeler/celikkubbe
    source .venv/bin/activate
    python dogrulama.py

Secenekler:
    --hizli     YOLO ve webcam testlerini atla (~2 saniye surer)
"""

import os
import subprocess
import sys
import time

HIZLI = "--hizli" in sys.argv

G, R, Y, D, X = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"
sonuclar = []


def rapor(ad, durum, detay=""):
    """durum: True=PASS, False=FAIL, None=ATLA"""
    etiket = {True: f"{G}[PASS]{X}", False: f"{R}[FAIL]{X}", None: f"{Y}[ATLA]{X}"}[durum]
    print(f"{etiket} {ad}")
    for satir in str(detay).strip().splitlines():
        if satir.strip():
            print(f"       {D}{satir}{X}")
    sonuclar.append((ad, durum))


def sh(cmd, timeout=30):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


print(f"\n{'=' * 62}")
print("  CELIKKUBBE - GELISTIRME ORTAMI DOGRULAMA")
print(f"{'=' * 62}\n")

# --------------------------------------------------------------------
# 1. Cekirdek surumu
# --------------------------------------------------------------------
try:
    kernel = os.uname().release
    rapor("Cekirdek surumu (6.8 HWE bekleniyor)", kernel.startswith("6.8"), kernel)
except Exception as e:
    rapor("Cekirdek surumu", False, e)

# --------------------------------------------------------------------
# 2. NVIDIA surucusu
# --------------------------------------------------------------------
try:
    p = sh("nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader")
    if p.returncode == 0 and p.stdout.strip():
        rapor("nvidia-smi calisiyor", True, p.stdout.strip())
    else:
        rapor("nvidia-smi calisiyor", False, p.stderr.strip() or "cikti yok")
except Exception as e:
    rapor("nvidia-smi calisiyor", False, e)

# --------------------------------------------------------------------
# 3. Secure Boot
# --------------------------------------------------------------------
try:
    p = sh("mokutil --sb-state")
    rapor("Secure Boot durumu", p.returncode == 0, p.stdout.strip() or p.stderr.strip())
except Exception as e:
    rapor("Secure Boot durumu", None, e)

# --------------------------------------------------------------------
# 4. PRIME / Optimus
# --------------------------------------------------------------------
try:
    p = sh("prime-select query")
    mod = p.stdout.strip()
    rapor("PRIME modu (on-demand bekleniyor)", mod == "on-demand", mod)
except Exception as e:
    rapor("PRIME modu", None, e)

# --------------------------------------------------------------------
# 5. NumPy
# --------------------------------------------------------------------
try:
    import numpy as np
    rapor("NumPy iceri aktarildi", True, f"surum {np.__version__}")
except Exception as e:
    rapor("NumPy iceri aktarildi", False, e)

# --------------------------------------------------------------------
# 6. PyTorch + CUDA
# --------------------------------------------------------------------
torch_ok = False
try:
    import torch
    cuda = torch.cuda.is_available()
    detay = [f"torch {torch.__version__}", f"cuda_available: {cuda}"]
    if cuda:
        detay.append(f"GPU: {torch.cuda.get_device_name(0)}")
        detay.append(f"CUDA runtime: {torch.version.cuda}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        detay.append(f"VRAM: {vram:.1f} GB")
    torch_ok = cuda
    rapor("PyTorch CUDA erisimi", cuda, "\n".join(detay))
except Exception as e:
    rapor("PyTorch CUDA erisimi", False, e)

# --------------------------------------------------------------------
# 7. GPU uzerinde gercek hesaplama
# --------------------------------------------------------------------
if torch_ok:
    try:
        import torch
        a = torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            b = a @ a
        torch.cuda.synchronize()
        sure = (time.perf_counter() - t0) / 10
        tflops = (2 * 4096**3) / sure / 1e12
        rapor("GPU matris carpimi", True, f"4096x4096 matmul: {sure*1000:.1f} ms  (~{tflops:.1f} TFLOPS fp32)")
        del a, b
        torch.cuda.empty_cache()
    except Exception as e:
        rapor("GPU matris carpimi", False, e)
else:
    rapor("GPU matris carpimi", None, "CUDA yok, atlandi")

# --------------------------------------------------------------------
# 8. YOLOv8n cikarimi
# --------------------------------------------------------------------
if HIZLI:
    rapor("YOLOv8n cikarimi", None, "--hizli ile atlandi")
else:
    try:
        import numpy as np
        from ultralytics import YOLO
        import logging
        logging.getLogger("ultralytics").setLevel(logging.ERROR)

        model = YOLO("yolov8n.pt")
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        dev = 0 if torch_ok else "cpu"

        model.predict(img, device=dev, verbose=False)  # isinma

        sureler = []
        for _ in range(5):
            t0 = time.perf_counter()
            model.predict(img, device=dev, verbose=False)
            sureler.append(time.perf_counter() - t0)

        ort = sum(sureler) / len(sureler)
        rapor("YOLOv8n cikarimi", True,
              f"cihaz: {'GPU' if torch_ok else 'CPU'}\n"
              f"ortalama: {ort*1000:.1f} ms  ({1/ort:.0f} FPS)\n"
              f"en iyi:   {min(sureler)*1000:.1f} ms")
    except Exception as e:
        rapor("YOLOv8n cikarimi", False, e)

# --------------------------------------------------------------------
# 9. OpenCV + webcam
# --------------------------------------------------------------------
try:
    import cv2
    rapor("OpenCV iceri aktarildi", True, f"surum {cv2.__version__}")

    if HIZLI:
        rapor("Webcam kare yakalama", None, "--hizli ile atlandi")
    else:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ok, kare = cap.read()
            if ok and kare is not None:
                rapor("Webcam kare yakalama", True, f"cozunurluk: {kare.shape[1]}x{kare.shape[0]}")
            else:
                rapor("Webcam kare yakalama", False, "cihaz acildi ama kare okunamadi")
            cap.release()
        else:
            rapor("Webcam kare yakalama", False, "/dev/video0 acilamadi (video grubu? kamera baska uygulamada?)")
except Exception as e:
    rapor("OpenCV", False, e)

# --------------------------------------------------------------------
# 10. PyQt6
# --------------------------------------------------------------------
try:
    from PyQt6.QtWidgets import QApplication, QWidget
    from PyQt6.QtCore import QT_VERSION_STR
    app = QApplication.instance() or QApplication([])
    w = QWidget()
    w.resize(200, 100)
    platform_adi = app.platformName()
    rapor("PyQt6 pencere olusturma", True, f"Qt {QT_VERSION_STR}  |  platform eklentisi: {platform_adi}")
    w.deleteLater()
except Exception as e:
    rapor("PyQt6 pencere olusturma", False, e)

# --------------------------------------------------------------------
# 11. pyqtgraph
# --------------------------------------------------------------------
try:
    import pyqtgraph as pg
    rapor("pyqtgraph iceri aktarildi", True, f"surum {pg.__version__}")
except Exception as e:
    rapor("pyqtgraph iceri aktarildi", None, e)

# --------------------------------------------------------------------
# 12. filterpy (NumPy 2 uyumlulugu)
# --------------------------------------------------------------------
try:
    import numpy as np
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise

    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 0.1, 0], [0, 1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    kf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=1.0, block_size=2)
    kf.R = np.eye(2) * 5.0
    kf.P *= 100.0
    kf.x = np.array([0.0, 0.0, 1.0, 1.0])
    kf.predict()
    kf.update(np.array([0.12, 0.09]))
    rapor("filterpy Kalman (NumPy 2 uyumu)", True,
          f"predict+update calisti, x = {np.round(kf.x, 3)}")
except ImportError as e:
    rapor("filterpy Kalman", None, f"kurulu degil: {e}")
except Exception as e:
    rapor("filterpy Kalman (NumPy 2 uyumu)", False, f"{type(e).__name__}: {e}")

# --------------------------------------------------------------------
# 13. pyserial
# --------------------------------------------------------------------
try:
    import serial
    import serial.tools.list_ports as lp
    portlar = list(lp.comports())
    # /dev/ttyS0-31 cekirdegin ayirdigi eski UART yuvalari, gercek cihaz degil.
    # STM32 USB-RS422 ile baglaninca ttyUSB* veya ttyACM* olarak gorunecek.
    gercek = [p for p in portlar if not p.device.startswith("/dev/ttyS")]
    if gercek:
        detay = "\n".join(f"{p.device} - {p.description}" for p in gercek)
    else:
        detay = (f"gercek seri cihaz yok (STM32 takili degilse normal)\n"
                 f"{len(portlar) - len(gercek)} adet sanal ttyS yuvasi gizlendi")
    rapor("pyserial iceri aktarildi", True, f"surum {serial.__version__}\n{detay}")
except Exception as e:
    rapor("pyserial iceri aktarildi", False, e)

# --------------------------------------------------------------------
# 14. evdev
# --------------------------------------------------------------------
# Bos liste HATA DEGILDIR: /dev/input/event* cihazlari root:input 0660
# izinlidir. udev, yalnizca joystick/gamepad olarak isaretlenen cihazlara
# aktif oturum kullanicisi icin ACL verir. Gamepad takili degilken
# okunabilir cihaz olmamasi beklenen davranistir.
try:
    import evdev
    cihazlar = []
    for yol in evdev.list_devices():
        try:
            cihazlar.append(evdev.InputDevice(yol))
        except PermissionError:
            pass

    if cihazlar:
        detay = "\n".join(f"{d.path}: {d.name}" for d in cihazlar[:6])
        if len(cihazlar) > 6:
            detay += f"\n... ve {len(cihazlar) - 6} cihaz daha"
    else:
        detay = ("okunabilir cihaz yok - gamepad takili degilse normal.\n"
                 "F310 takildiginda udev ACL verecek ve burada gorunecek.")

    try:
        from importlib.metadata import version as _pkg_version
        surum = _pkg_version("evdev")
    except Exception:
        surum = "?"

    rapor("evdev calisiyor", True, f"surum {surum}\n{detay}")
except Exception as e:
    rapor("evdev calisiyor", False, f"{type(e).__name__}: {e}")

# --------------------------------------------------------------------
# 15. Grup uyelikleri
# --------------------------------------------------------------------
try:
    import grp
    gruplar = [grp.getgrgid(g).gr_name for g in os.getgroups()]
    gerekli = {"dialout": "RS422 / STM32 seri port", "video": "kamera, V4L2, RealSense"}
    for grup, aciklama in gerekli.items():
        rapor(f"Grup uyeligi: {grup}", grup in gruplar, aciklama)
    if "plugdev" in gruplar:
        rapor("Grup uyeligi: plugdev", True, "USB cihaz erisimi")
except Exception as e:
    rapor("Grup uyelikleri", False, e)

# --------------------------------------------------------------------
# 16. pyrealsense2
# --------------------------------------------------------------------
try:
    import pyrealsense2 as rs
    ctx = rs.context()
    n = len(ctx.query_devices())
    rapor("pyrealsense2 iceri aktarildi", True,
          f"bagli cihaz: {n}" + ("  (kamera takili degil)" if n == 0 else ""))
except ImportError:
    rapor("pyrealsense2", None, "kurulu degil - kamera geldiginde kurulacak")
except Exception as e:
    rapor("pyrealsense2", False, e)

# --------------------------------------------------------------------
# 17. GRUB'da Windows
# --------------------------------------------------------------------
try:
    with open("/boot/grub/grub.cfg", "r", errors="ignore") as f:
        cfg = f.read()
    bulundu = "Windows Boot Manager" in cfg
    rapor("GRUB menusunde Windows", bulundu,
          "os-prober girdiyi olusturmus" if bulundu
          else "girdi yok - 'sudo update-grub' calistir")
except PermissionError:
    rapor("GRUB menusunde Windows", None, "grub.cfg okunamadi (sudo gerekebilir)")
except Exception as e:
    rapor("GRUB menusunde Windows", False, e)

# --------------------------------------------------------------------
# 18. Windows ESP dokunulmamis mi
#
# NOT: NVMe cihaz adlari (/dev/nvme0n1, /dev/nvme1n1) acilislar arasinda
# yer degistirebilir. Bu yuzden kontrol UUID uzerinden yapiliyor.
# --------------------------------------------------------------------
UBUNTU_ESP_UUID = "77D5-1EE4"    # Samsung uzerindeki kendi ESP'miz
WINDOWS_ESP_UUID = "02CF-2EE6"   # Windows'un 100 MB ESP'si - ASLA baglanmamali

try:
    kaynak = sh("findmnt -no SOURCE /boot/efi").stdout.strip()
    esp_uuid = sh(f"lsblk -no UUID {kaynak}").stdout.strip() if kaynak else ""
    win_bagli = bool(sh(f"findmnt -rn -S UUID={WINDOWS_ESP_UUID}").stdout.strip())

    dogru = (esp_uuid.upper() == UBUNTU_ESP_UUID) and not win_bagli
    rapor("Windows ESP korunuyor", dogru,
          f"/boot/efi -> {kaynak}  (UUID {esp_uuid})\n"
          f"Windows ESP {WINDOWS_ESP_UUID}: "
          + ("BAGLI - DIKKAT, grub-install calistirma!" if win_bagli else "bagli degil (dogru)"))
except Exception as e:
    rapor("Windows ESP korunuyor", None, e)

# --------------------------------------------------------------------
# 18b. Kok ve paylasimli bolum UUID kontrolu
# --------------------------------------------------------------------
try:
    kok = sh("findmnt -no SOURCE /").stdout.strip()
    kok_uuid = sh(f"lsblk -no UUID {kok}").stdout.strip()
    bekleniyor = "78bb6950-2133-492b-b2f0-ba48b6b15e1d"
    rapor("Kok bolum dogru diskte", kok_uuid == bekleniyor,
          f"/ -> {kok}  (UUID {kok_uuid})")
except Exception as e:
    rapor("Kok bolum dogru diskte", None, e)

# --------------------------------------------------------------------
# 19. Paylasimli bolum
# --------------------------------------------------------------------
try:
    p = sh("mountpoint -q /mnt/shared")
    rapor("/mnt/shared bagli", p.returncode == 0,
          "D: paylasimli bolum erisilebilir" if p.returncode == 0
          else "bagli degil - CLAUDE.md symlink'i kirik olabilir")
except Exception as e:
    rapor("/mnt/shared bagli", None, e)

# --------------------------------------------------------------------
# OZET
# --------------------------------------------------------------------
gecen = sum(1 for _, d in sonuclar if d is True)
kalan = sum(1 for _, d in sonuclar if d is False)
atlanan = sum(1 for _, d in sonuclar if d is None)

print(f"\n{'=' * 62}")
print(f"  OZET:  {G}{gecen} PASS{X}   {R}{kalan} FAIL{X}   {Y}{atlanan} ATLA{X}")
print(f"{'=' * 62}")

if kalan:
    print(f"\n{R}Basarisiz testler:{X}")
    for ad, d in sonuclar:
        if d is False:
            print(f"  - {ad}")
    print()
    sys.exit(1)

print(f"\n{G}Ortam kullanima hazir.{X}\n")
sys.exit(0)
