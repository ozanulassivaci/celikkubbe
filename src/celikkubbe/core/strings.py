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

from celikkubbe.core.types import ReasonCode

REASON_CODE_TR: dict[ReasonCode, str] = {
    ReasonCode.NOT_OPERATIONAL: "Sistem operasyonel modda degil",
    ReasonCode.NOT_ARMED: "Sistem silahli degil",
    ReasonCode.ESTOP_ACTIVE: "Acil durdurma aktif",
    ReasonCode.POSITION_INVALID: "Konum gecersiz, yeniden sifirlama gerekli",
    ReasonCode.TARGET_FRIENDLY: "Hedef dost olarak isaretlendi",
    ReasonCode.RANGE_UNKNOWN: "Menzil bilgisi yok",
    ReasonCode.RANGE_OUT_OF_BOUNDS: "Hedef menzil disinda",
    ReasonCode.LOW_CONFIDENCE: "Guven skoru dusuk",
    ReasonCode.LIMIT_EXCEEDED: "Yazilim limiti asildi",
    ReasonCode.INFERENCE_SLOW: "Cikarim (inference) yavas",
    ReasonCode.CAMERA_TIMEOUT: "Kameradan veri alinamiyor",
    ReasonCode.LINK_TIMEOUT: "STM32 baglantisi zaman asimina ugradi",
    ReasonCode.DEPTH_UNRELIABLE: "Derinlik verisi guvenilir degil",
    ReasonCode.OPERATOR_OVERRIDE: "Operator mudahalesi",
    ReasonCode.SETPOINT_NOT_ACKED: "Komut edilen aci henuz dogrulanmadi",
    ReasonCode.NO_AIM_SOLUTION: "Hedef icin nisan cozumu yok",
    ReasonCode.MOTION_IN_PROGRESS: "Hareket henuz tamamlanmadi",
    ReasonCode.DRIVER_ALARM: "Surucu alarmi aktif",
    ReasonCode.IFF_UNKNOWN: "Dost/dusman bilgisi belirlenemedi",
    ReasonCode.NOT_HOMED: "Eksenler henuz sifirlanmadi",
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
    "NOT_CALIBRATED": "KALİBRE DEĞİL",
    "TARGET_LOCKED": "HEDEF KİLİTLİ",
    "AXIS_PAN": "PAN",
    "AXIS_TILT": "EĞİM",
    "PASS": "BAŞARILI",
    "FAIL": "BAŞARISIZ",
    "SELF_TEST_TITLE": "KENDİ KENDİNİ TEST",
    "RETRY": "TEKRAR DENE",
    "SAFE_TITLE": "M4 EMNİYET",
    "ACKNOWLEDGE": "ONAYLA",
    "HOMING_WARNING": "KONUM GEÇERSİZ — HOMING GEREKLİ",
    "SELF_TEST_PREFIX": "KENDİ KENDİNİ TEST: {detail}",
    "OPERATOR_ACTIVE": "OPERATÖR AKTİF",
    "TRACKING_AID_ON": "TAKİP YARDIMI ON",
    "L1_UNAVAILABLE": "L1:YOK",
}

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
