"""Turkish UI strings.

The only place in this codebase where user-facing language lives. Nothing
in the decision logic imports this module; the outer UI layer maps a
``ReasonCode`` to text with ``REASON_CODE_TR``, or looks up a fixed label
with no associated code (an axis name, a badge, a banner prefix) in
``UI_LABEL_TR``, whenever it needs to display one. ``ui/`` widget code
must never inline a Turkish literal itself -- both dicts exist so every
string a competition operator sees can be found, and changed, in one
place, even the ones with no ``ReasonCode`` to key off.
"""

from __future__ import annotations

from celikkubbe.core.types import IFF, Layer, ReasonCode, Stage, TargetClass, TrackStatus

REASON_CODE_TR: dict[ReasonCode, str] = {
    ReasonCode.NOT_OPERATIONAL: "Sistem operasyonel modda değil",
    ReasonCode.NOT_ARMED: "Sistem silahlı değil",
    ReasonCode.ESTOP_ACTIVE: "Acil durdurma aktif",
    ReasonCode.POSITION_INVALID: "Konum geçersiz, yeniden sıfırlama gerekli",
    ReasonCode.TARGET_FRIENDLY: "Hedef dost olarak işaretlendi",
    ReasonCode.RANGE_UNKNOWN: "Menzil bilgisi yok",
    ReasonCode.RANGE_OUT_OF_BOUNDS: "Hedef menzil dışında",
    ReasonCode.LOW_CONFIDENCE: "Güven skoru düşük",
    ReasonCode.LIMIT_EXCEEDED: "Yazılım limiti aşıldı",
    ReasonCode.INFERENCE_SLOW: "Çıkarım süresi yavaş",
    ReasonCode.CAMERA_TIMEOUT: "Kameradan veri alınamıyor",
    ReasonCode.LINK_TIMEOUT: "STM32 bağlantısı zaman aşımına uğradı",
    ReasonCode.DEPTH_UNRELIABLE: "Derinlik verisi güvenilir değil",
    ReasonCode.OPERATOR_OVERRIDE: "Operatör müdahalesi",
    ReasonCode.SETPOINT_NOT_ACKED: "Komut edilen açı henüz doğrulanmadı",
    ReasonCode.NO_AIM_SOLUTION: "Hedef için nişan çözümü yok",
    ReasonCode.MOTION_IN_PROGRESS: "Hareket henüz tamamlanmadı",
    ReasonCode.DRIVER_ALARM: "Sürücü alarmı aktif",
    ReasonCode.IFF_UNKNOWN: "Dost/düşman bilgisi belirlenemedi",
    ReasonCode.NOT_HOMED: "Eksenler henüz sıfırlanmadı",
}


def describe(reason: ReasonCode) -> str:
    return REASON_CODE_TR[reason]


# Fixed UI labels with no ReasonCode of their own -- badges, axis names,
# banner text. Keyed by a short English slug rather than an enum, since
# these are presentation labels, not decision-logic outcomes; nothing
# here is looked up by anything in core/. Values with a "{...}" field are
# format templates, not literal display strings -- the caller supplies
# the dynamic part with .format(), keeping the Turkish text itself here
# rather than half in a widget's f-string.
UI_LABEL_TR: dict[str, str] = {
    "UNKNOWN_CLASS": "BİLİNMEYEN",
    # Suffix marking an operator-assigned class -- never omitted, so a
    # manual assignment is never visually indistinguishable from a
    # model-produced one. "EL" (hand) -- assigned by hand, not by a model.
    "CLASS_SOURCE_OPERATOR_TAG": "[EL]",
    "NOT_CALIBRATED": "KALİBRE DEĞİL",
    "TARGET_LOCKED": "HEDEF KİLİTLİ",
    "AXIS_PAN": "PAN",
    "AXIS_TILT": "EĞİM",
    "PASS": "BAŞARILI",
    "FAIL": "BAŞARISIZ",
    "SELF_TEST_TITLE": "KENDİ KENDİNİ TEST",
    "RETRY": "TEKRAR DENE",
    "SKIP_SELF_TEST": "GEÇ",
    "DEV_MODE_BANNER": "GELİŞTİRME MODU — EMNİYET KONTROLLERİ DEVRE DIŞI",
    "SAFE_TITLE": "M4 EMNİYET",
    "ACKNOWLEDGE": "ONAYLA",
    "HOMING_WARNING": "KONUM GEÇERSİZ — HOMING GEREKLİ",
    "SELF_TEST_PREFIX": "KENDİ KENDİNİ TEST: {detail}",
    "OPERATOR_ACTIVE": "OPERATÖR AKTİF",
    "TRACKING_AID_ON": "TAKİP YARDIMI ON",
    "L1_UNAVAILABLE": "L1:YOK",
    "AI_PANEL_TITLE": "YZ KARAR DESTEĞİ",
    "LIVE_INDICATOR": "CANLI",
    "THREAT_HIGH": "YÜKSEK TEHDİT",
    "THREAT_MEDIUM": "ORTA",
    "THREAT_LOW": "DÜŞÜK",
    "THREAT_NONE": "HEDEF YOK",
    "TARGET_LIST_TITLE": "HEDEF TAKİBİ",
    "RECOMMENDATION_SEARCHING": "Hedef aranıyor…",
    "RECOMMENDATION_TRACKING": "{cls} takip ediliyor",
    "RECOMMENDATION_FIRE": "{cls} İMHA ET — Güven eşiği aşıldı",
    "RECOMMENDATION_BLOCKED": "{cls} — {reason}",
    "RECOMMENDATION_ENGAGED": "{cls} ateşlendi, sonuç bekleniyor",
    "EXCLUDED_FRIENDLY": "DOST — otomatik hedef alınamaz",
    "DEFERRED_PREFIX": "ERTELENDİ ({seconds:.0f}s) — {reason}",
    "ESTOP_BUTTON": "ACİL STOP",
    "SAFETY_LOCK": "EMNİYET KİLİDİ",
    "FIRE_BUTTON": "ATIŞ",
    "SAFETY_WARNING_LOCKED": "EMNİYET KİLİDİ TAKILI",
    "SAFETY_WARNING_UNLOCKED": "EMNİYET KİLİDİ AÇIK — ATEŞLENEBİLİR",
    "MODE_CARDS_TITLE": "ALGILAMA KATMANI",
    "LOCKED_TARGET_TITLE": "KİLİTLİ HEDEF",
    "MANUAL_CONTROL_TITLE": "MANUEL KONTROL",
    "HOMING_TITLE": "SIFIRLAMA",
    "ZERO_PAN": "SIFIRLA PAN",
    "ZERO_TILT": "SIFIRLA TİLT",
    "OPEN_TUNING_WINDOW": "RENK AYARI",
    "TOOLS_MENU": "ARAÇLAR",
    "STOP_BUTTON": "DUR",
    "OPERATOR_CHOSEN": "OPERATÖR",
    "CASCADE_CHOSEN": "OTOMATİK",
    "UNAVAILABLE": "YOK",
    "SPEED_LABEL": "HIZ",
    "TUNING_TITLE": "HSV AYARLARI",
    "GLOBAL_SECTION_TITLE": "GENEL",
    "HOSTILE_CLASS_TITLE": "DÜŞMAN (KIRMIZI)",
    "FRIENDLY_CLASS_TITLE": "DOST (MAVİ)",
    "MORPH_KERNEL_LABEL": "MORFOLOJİ ÇEKİRDEĞİ",
    "MIN_AREA_LABEL": "MIN. ALAN",
    "MIN_AREA_AUTO_HINT": "0 = OTOMATİK (optikten hesaplanır)",
    "CIRCULARITY_MIN_LABEL": "MIN. DAİRESELLİK",
    "REQUIRE_CIRCULARITY_LABEL": "DAİRESELLİK ZORUNLU",
    "HUE_RANGE_1_LABEL": "TON ARALIĞI 1",
    "HUE_RANGE_2_LABEL": "TON ARALIĞI 2",
    "HUE_RANGE_LABEL": "TON ARALIĞI",
    "SAT_MIN_LABEL": "MIN. DOYGUNLUK",
    "VAL_MIN_LABEL": "MIN. PARLAKLIK",
    "PREVIEW_MODE_LABEL": "ÖNİZLEME",
    "PREVIEW_SOURCE": "KAYNAK",
    "PREVIEW_HSV_MASK": "HSV MASKE",
    "PREVIEW_MORPHED_MASK": "MORFOLOJİ",
    "PREVIEW_CONTOURS": "KONTUR",
    "PREVIEW_CLASS_LABEL": "SINIF",
    "TUNING_COUNTS": "KABUL: {accepted}   RED: {rejected}   SINIF LİMİTİ: {capped}",
    "PRESET_NAME_PLACEHOLDER": "ön ayar adı",
    "SAVE_PRESET": "KAYDET",
    "LOAD_PRESET": "YÜKLE",
    "RESET_DEFAULTS": "VARSAYILANA DÖN",
    "CLEAR_ROI": "ROI TEMİZLE",
    "CLOSE_BUTTON": "KAPAT",
    "HELP_TITLE": "KONTROLLER",
    "GAMEPAD_CONNECTED": "GAMEPAD BAĞLI",
    "GAMEPAD_DISCONNECTED": "GAMEPAD BAĞLI DEĞİL",
}

# (key label, description) pairs for the F1 help overlay -- a fixed
# reference list, not a ReasonCode lookup, so it lives as its own tuple
# rather than forcing single-line entries into UI_LABEL_TR.
HELP_CONTROLS_TR: tuple[tuple[str, str], ...] = (
    ("OK TUŞLARI", "Pan / eğim manuel kontrol"),
    ("BOŞLUK", "Ateşleme (basılı tutun)"),
    ("A", "Emniyet kilidini aç / kapat"),
    ("ESC", "Acil durdurma"),
    ("1 / 2 / 3", "Aşama seçimi"),
    ("H", "HSV ayarları penceresi"),
    ("F1", "Bu yardım ekranı"),
)

# Self-test item detail templates -- dynamic (interpolated), so kept
# separate from the fixed-string dicts above rather than crammed in.
SELF_TEST_DETAIL_TR: dict[str, str] = {
    "no_camera_signal": "kamera sinyali yok",
    "fps_too_low": "FPS çok düşük",
    "no_telemetry": "telemetri yok",
    "in_progress": "kontrol ediliyor",
    "move_error": "hata {error:.2f}°",
    "cannot_verify": "bu bağlantı türünde doğrulanamıyor",
    "fire_not_rejected": "ateşleme devre dışı bırakılmadan reddedilmedi",
    "inference_slow": "çıkarım süresi yavaş",
    "driver_alarm_pan": "pan sürücü alarmı",
    "driver_alarm_tilt": "eğim sürücü alarmı",
}

# Self-test item names (SelfTestItem.name, a stable English slug used for
# identification) -> Turkish label shown on the self-test screen.
SELF_TEST_ITEM_LABEL_TR: dict[str, str] = {
    "camera": "Kamera",
    "stm32_link": "STM32 Bağlantısı",
    "pan_tilt_move": "Pan/Eğim Hareketi",
    "fire_lock": "Ateşleme Kilidi",
    "inference_time": "Çıkarım Süresi",
    "driver_alarms": "Sürücü Alarmları",
}

# IFF.UNKNOWN here is distinct from UI_LABEL_TR["UNKNOWN_CLASS"]: this one
# is "friend/foe could not be determined", that one is "no TargetClass at
# all" (which, with only L2 built, is every track, always -- see
# vision/l2_color.py's own module docstring).
IFF_LABEL_TR: dict[IFF, str] = {
    IFF.HOSTILE: "DÜŞMAN",
    IFF.FRIENDLY: "DOST",
    IFF.UNKNOWN: "BİLİNMEYEN",
}

TRACK_STATUS_TR: dict[TrackStatus, str] = {
    TrackStatus.TENTATIVE: "BEKLEMEDE",
    TrackStatus.CONFIRMED: "ONAYLANDI",
    TrackStatus.COASTING: "TAHMİNİ",
    TrackStatus.LOST: "KAYBOLDU",
}

# cls is always None today -- L2 never sets it (see vision/l2_color.py) --
# so this is currently unreachable in practice, kept for when an L1/YOLO
# layer lands rather than leaving a raw English enum value on screen the
# day it does.
TARGET_CLASS_TR: dict[TargetClass, str] = {
    TargetClass.F16: "F-16",
    TargetClass.HELICOPTER: "HELİKOPTER",
    TargetClass.MISSILE: "FÜZE",
    TargetClass.UAV: "İHA",
    TargetClass.BALLOON: "BALON",
    TargetClass.UNKNOWN: "BİLİNMEYEN",
}

STAGE_LABEL_TR: dict[Stage, str] = {
    Stage.STAGE_1: "AŞAMA 1",
    Stage.STAGE_2: "AŞAMA 2",
    Stage.STAGE_3: "AŞAMA 3",
}

# L1/L2/L3 by what they actually are, not the internal Layer enum name --
# YOLO does not exist yet (see vision/ package docstrings) and L3 is the
# Stage 1 fully-manual override (see core/cascade.py's own docstring).
LAYER_LABEL_TR: dict[Layer, str] = {
    Layer.L1: "YOLO",
    Layer.L2: "OPENCV",
    Layer.L3: "TAM MANUEL",
}
