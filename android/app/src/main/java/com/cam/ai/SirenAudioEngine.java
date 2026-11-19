package com.cam.ai;

import android.content.Context;

/**
 * Tamamen Sessiz Alarm Motoru (v2.3):
 * Telefondan kesinlikle hiçbir ses, bip, siren veya titreşim çıkmaz.
 * Alarm durumu yalnızca ağ üzerinden PC izleyicisine (/state -> alarm:true) iletilir.
 */
public class SirenAudioEngine {

    private volatile boolean isAlarmActive = false;

    public SirenAudioEngine(Context context) {
        // Kesinlikle AudioTrack, ToneGenerator veya Vibrator başlatılmaz.
    }

    public synchronized void startAlarm() {
        isAlarmActive = true;
    }

    public synchronized void stopAlarm() {
        isAlarmActive = false;
    }

    public boolean isAlarmActive() {
        return isAlarmActive;
    }

    public void playLockBeep() {
        // Sessiz mod: Telefondan ses/titreşim çıkmaz.
    }

    public void playErrorBeep() {
        // Sessiz mod: Telefondan ses/titreşim çıkmaz.
    }

    public void release() {
        isAlarmActive = false;
    }
}
