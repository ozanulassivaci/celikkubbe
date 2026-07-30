"""Turkish UI strings keyed by ReasonCode.

The only place in this codebase where user-facing language lives. Nothing
in the decision logic imports this module; the outer UI layer maps a
``ReasonCode`` to text with ``REASON_CODE_TR`` when it needs to display one.
"""

from __future__ import annotations

from celikkubbe.core.types import ReasonCode

REASON_CODE_TR: dict[ReasonCode, str] = {
    ReasonCode.NOT_OPERATIONAL: "Sistem operasyonel modda degil",
    ReasonCode.NOT_ARMED: "Sistem silahli degil",
    ReasonCode.ESTOP_ACTIVE: "Acil durdurma aktif",
    ReasonCode.POSITION_INVALID: "Konum gecersiz, yeniden sifirlama gerekli",
    ReasonCode.TARGET_FRIENDLY: "Hedef dost olarak isaretlendi",
    ReasonCode.CLASS_UNKNOWN: "Hedef sinifi belirlenemedi",
    ReasonCode.RANGE_UNKNOWN: "Menzil bilgisi yok",
    ReasonCode.RANGE_OUT_OF_BOUNDS: "Hedef menzil disinda",
    ReasonCode.LOW_CONFIDENCE: "Guven skoru dusuk",
    ReasonCode.ANGLE_NOT_SETTLED: "Acisal konum henuz yerlesmedi",
    ReasonCode.LIMIT_EXCEEDED: "Yazilim limiti asildi",
    ReasonCode.INFERENCE_SLOW: "Cikarim (inference) yavas",
    ReasonCode.CAMERA_TIMEOUT: "Kameradan veri alinamiyor",
    ReasonCode.LINK_TIMEOUT: "STM32 baglantisi zaman asimina ugradi",
    ReasonCode.DEPTH_UNRELIABLE: "Derinlik verisi guvenilir degil",
    ReasonCode.OPERATOR_OVERRIDE: "Operator mudahalesi",
    ReasonCode.SETPOINT_NOT_ACKED: "Komut edilen aci henuz dogrulanmadi",
}


def describe(reason: ReasonCode) -> str:
    return REASON_CODE_TR[reason]
