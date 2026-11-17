package com.cam.ai;

import android.content.Context;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;
import android.media.ToneGenerator;
import android.media.AudioManager;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.VibrationEffect;
import android.os.Vibrator;

/**
 * Profesyonel güvenlik alarm sireni (AudioTrack frekans taraması) ve dokunsal titreşim motoru.
 */
public class SirenAudioEngine {

    private static final int SAMPLE_RATE = 22050;
    private final Vibrator vibrator;
    private ToneGenerator beepGenerator;
    private volatile boolean isAlarmPlaying = false;
    private Thread sirenThread;
    private final Handler vibHandler = new Handler(Looper.getMainLooper());

    public SirenAudioEngine(Context context) {
        vibrator = (Vibrator) context.getSystemService(Context.VIBRATOR_SERVICE);
        try {
            beepGenerator = new ToneGenerator(AudioManager.STREAM_NOTIFICATION, 75);
        } catch (Exception ignored) {}
    }

    public synchronized void startAlarm() {
        if (isAlarmPlaying) return;
        isAlarmPlaying = true;

        sirenThread = new Thread(this::runSirenLoop, "SirenAudioThread");
        sirenThread.start();
        startVibrationLoop();
    }

    public synchronized void stopAlarm() {
        isAlarmPlaying = false;
        vibHandler.removeCallbacksAndMessages(null);
        if (vibrator != null) {
            try { vibrator.cancel(); } catch (Exception ignored) {}
        }
        if (sirenThread != null) {
            sirenThread.interrupt();
            sirenThread = null;
        }
    }

    public void playLockBeep() {
        try {
            if (beepGenerator != null) {
                beepGenerator.startTone(ToneGenerator.TONE_PROP_ACK, 70);
            }
            vibrateOnce(25);
        } catch (Exception ignored) {}
    }

    public void playErrorBeep() {
        try {
            if (beepGenerator != null) {
                beepGenerator.startTone(ToneGenerator.TONE_PROP_NACK, 90);
            }
            vibrateOnce(45);
        } catch (Exception ignored) {}
    }

    private void vibrateOnce(long ms) {
        if (vibrator == null || !vibrator.hasVibrator()) return;
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createOneShot(ms, VibrationEffect.DEFAULT_AMPLITUDE));
            } else {
                vibrator.vibrate(ms);
            }
        } catch (Exception ignored) {}
    }

    private void startVibrationLoop() {
        vibHandler.post(new Runnable() {
            @Override
            public void run() {
                if (!isAlarmPlaying) return;
                vibrateOnce(350);
                vibHandler.postDelayed(this, 600);
            }
        });
    }

    private void runSirenLoop() {
        int minBuf = AudioTrack.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT
        );
        int bufSize = Math.max(minBuf, SAMPLE_RATE / 4 * 2);

        AudioTrack track = null;
        try {
            AudioAttributes attrs = new AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ALARM)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build();

            AudioFormat format = new AudioFormat.Builder()
                    .setSampleRate(SAMPLE_RATE)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build();

            track = new AudioTrack.Builder()
                    .setAudioAttributes(attrs)
                    .setAudioFormat(format)
                    .setBufferSizeInBytes(bufSize)
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build();

            track.play();

            short[] buffer = new short[1024];
            double phase = 0.0;
            long sampleIndex = 0;

            while (isAlarmPlaying && !Thread.currentThread().isInterrupted()) {
                for (int i = 0; i < buffer.length; i++) {
                    // Saniyede 2.2 kez 750Hz <-> 1650Hz arası keskin güvenlik sireni taraması
                    double t = (double) sampleIndex / SAMPLE_RATE;
                    double lfo = 0.5 + 0.5 * Math.sin(2.0 * Math.PI * 2.2 * t);
                    double freq = 750.0 + 900.0 * lfo;

                    phase += 2.0 * Math.PI * freq / SAMPLE_RATE;
                    if (phase > 2.0 * Math.PI) phase -= 2.0 * Math.PI;

                    // Hafif keskinleştirilmiş sinüs (harmonik güçlü siren tonu)
                    double wave = Math.sin(phase) + 0.25 * Math.sin(3.0 * phase);
                    wave = Math.max(-1.0, Math.min(1.0, wave));
                    buffer[i] = (short) (wave * 28000);
                    sampleIndex++;
                }
                track.write(buffer, 0, buffer.length);
            }
        } catch (Exception ignored) {
        } finally {
            if (track != null) {
                try {
                    track.stop();
                    track.release();
                } catch (Exception ignored) {}
            }
        }
    }

    public void release() {
        stopAlarm();
        if (beepGenerator != null) {
            try { beepGenerator.release(); } catch (Exception ignored) {}
            beepGenerator = null;
        }
    }
}
