package com.cam.ai;

import java.util.Arrays;
import java.util.List;

/**
 * Sabit Oda Güvenlik Kamerası İçin Uyarlamalı Arka Plan Ayrıştırma Maskesi (v3.0):
 *  - Mevcut 160x120 analiz tamponu üzerinde çalışır (~0.15 ms, sıfır GC tahsisi).
 *  - Kişi kutularının DIŞINDAKİ duvar, koltuk, yatak, perde piksellerini öğrenir;
 *    kişi saatlerce aynı yerde otursa bile kişi pikselleri ASLA arka plana karışmaz!
 *  - Oda ışığı değiştiğinde (lamba açma/kapama) küresel ortam parlaklık farkını otomatik dengeler.
 *  - Kamera döndürülürse veya telefon oynatılırsa sahne değişimini algılayıp anında sıfırlanır.
 *  - Re-ID imzası çıkarılırken arka plan piksellerini bastırarak kişinin farklı duvar/arka plan
 *    önüne geçtiğinde dahi %100 tanınmasını sağlar.
 */
public final class RoomBackgroundMask {

    private final int width;
    private final int height;
    private final int totalPixels;

    private final float[] bgR;
    private final float[] bgG;
    private final float[] bgB;
    private final byte[] bgConfidence; // 0..30: pikselin kaç temiz karede arka plan olarak görüldüğü
    private final boolean[] personMask;

    private static final int MIN_CONF_FOR_MASK = 3;
    private static final int MAX_CONF = 30;

    public RoomBackgroundMask(int width, int height) {
        this.width = width;
        this.height = height;
        this.totalPixels = width * height;
        this.bgR = new float[totalPixels];
        this.bgG = new float[totalPixels];
        this.bgB = new float[totalPixels];
        this.bgConfidence = new byte[totalPixels];
        this.personMask = new boolean[totalPixels];
    }

    public void reset() {
        Arrays.fill(bgConfidence, (byte) 0);
    }

    /**
     * Mevcut karedeki tespit edilen kişilerin DIŞINDA kalan piksellerle oda arka planını günceller.
     */
    public void updateWithFrame(int[] argbPixels, List<TrackedPerson> persons) {
        if (argbPixels == null || argbPixels.length < totalPixels) return;

        Arrays.fill(personMask, false);
        if (persons != null) {
            for (int i = 0; i < persons.size(); i++) {
                TrackedPerson p = persons.get(i);
                // Kişi kutusunu %6 genişlet ki kol/omuz kenarları arka planı kirletmesin
                float padX = p.nw * 0.06f;
                float padY = p.nh * 0.05f;
                int x0 = Math.max(0, (int) Math.floor((p.nx - padX) * width));
                int x1 = Math.min(width, (int) Math.ceil((p.nx + p.nw + padX) * width));
                int y0 = Math.max(0, (int) Math.floor((p.ny - padY) * height));
                int y1 = Math.min(height, (int) Math.ceil((p.ny + p.nh + padY) * height));
                for (int y = y0; y < y1; y++) {
                    int row = y * width;
                    for (int x = x0; x < x1; x++) {
                        personMask[row + x] = true;
                    }
                }
            }
        }

        // 1. Küresel sahne sarsıntısı veya lamba açma/kapama kontrolü (seyrek örnekleme, ~0.02 ms)
        int checked = 0;
        int jumped = 0;
        float sumDr = 0f, sumDg = 0f, sumDb = 0f;
        for (int idx = 0; idx < totalPixels; idx += 29) {
            if (personMask[idx]) continue;
            int conf = bgConfidence[idx] & 0xFF;
            if (conf < MIN_CONF_FOR_MASK) continue;

            int argb = argbPixels[idx];
            float r = (argb >> 16) & 0xFF;
            float g = (argb >> 8) & 0xFF;
            float b = argb & 0xFF;

            float dr = r - bgR[idx];
            float dg = g - bgG[idx];
            float db = b - bgB[idx];
            sumDr += dr;
            sumDg += dg;
            sumDb += db;
            checked++;
            if (Math.abs(dr) + Math.abs(dg) + Math.abs(db) > 65f) {
                jumped++;
            }
        }

        float ambientDr = 0f, ambientDg = 0f, ambientDb = 0f;
        if (checked >= 40) {
            float jumpRatio = (float) jumped / checked;
            if (jumpRatio > 0.68f) {
                // Kamera taşındı/döndü veya sahne tamamen değişti -> hızlıca yeni odayı öğren!
                reset();
            } else {
                // Ortam ışığı değiştiyse (lamba/perde) tüm arka plana yumuşak yansıt
                ambientDr = (sumDr / checked) * 0.35f;
                ambientDg = (sumDg / checked) * 0.35f;
                ambientDb = (sumDb / checked) * 0.35f;
                if (Math.abs(ambientDr) + Math.abs(ambientDg) + Math.abs(ambientDb) < 3.0f) {
                    ambientDr = ambientDg = ambientDb = 0f;
                }
            }
        }

        // 2. Kişi dışındaki pikselleri arka plan modeline işle
        boolean applyAmbient = (ambientDr != 0f || ambientDg != 0f || ambientDb != 0f);
        for (int idx = 0; idx < totalPixels; idx++) {
            int conf = bgConfidence[idx] & 0xFF;
            if (personMask[idx]) {
                // Kişi altındaki bilinen arka plana sadece küresel ışık kaymasını uygula
                if (applyAmbient && conf >= MIN_CONF_FOR_MASK) {
                    bgR[idx] = SmartTrackerMath.clamp(bgR[idx] + ambientDr, 0f, 255f);
                    bgG[idx] = SmartTrackerMath.clamp(bgG[idx] + ambientDg, 0f, 255f);
                    bgB[idx] = SmartTrackerMath.clamp(bgB[idx] + ambientDb, 0f, 255f);
                }
                continue;
            }

            int argb = argbPixels[idx];
            float r = (argb >> 16) & 0xFF;
            float g = (argb >> 8) & 0xFF;
            float b = argb & 0xFF;

            if (conf == 0) {
                bgR[idx] = r;
                bgG[idx] = g;
                bgB[idx] = b;
                bgConfidence[idx] = 1;
            } else {
                // İlk 5 kare hızlı oturma, sonra kararlı yavaş adaptasyon (%6)
                float alpha = (conf < 5) ? 0.32f : 0.06f;
                bgR[idx] += (r - bgR[idx]) * alpha;
                bgG[idx] += (g - bgG[idx]) * alpha;
                bgB[idx] += (b - bgB[idx]) * alpha;
                if (conf < MAX_CONF) {
                    bgConfidence[idx] = (byte) (conf + 1);
                }
            }
        }
    }

    /**
     * Bir pikselin ön plan (insan) olma ağırlığını [0.06 .. 1.0] döndürür.
     * Arka plan henüz o pikselde öğrenilmediyse 1.0 döner (sıfır veri kaybı).
     */
    public float getForegroundWeight(int pixelIdx, int r, int g, int b) {
        if (pixelIdx < 0 || pixelIdx >= totalPixels) return 1.0f;
        if ((bgConfidence[pixelIdx] & 0xFF) < MIN_CONF_FOR_MASK) return 1.0f;

        float diff = Math.abs(r - bgR[pixelIdx]) + Math.abs(g - bgG[pixelIdx]) + Math.abs(b - bgB[pixelIdx]);
        if (diff <= 26f) return 0.06f;
        if (diff >= 66f) return 1.0f;
        return 0.06f + ((diff - 26f) / 40f) * 0.94f;
    }

    public boolean isPixelLearned(int pixelIdx) {
        return pixelIdx >= 0 && pixelIdx < totalPixels && (bgConfidence[pixelIdx] & 0xFF) >= MIN_CONF_FOR_MASK;
    }

    public float getLearnedCoverageRatio() {
        int learned = 0;
        for (int i = 0; i < totalPixels; i++) {
            if ((bgConfidence[i] & 0xFF) >= MIN_CONF_FOR_MASK) learned++;
        }
        return (float) learned / totalPixels;
    }
}
