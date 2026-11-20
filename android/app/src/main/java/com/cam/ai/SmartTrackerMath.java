package com.cam.ai;

import java.util.Arrays;

/**
 * Profesyonel Çoklu Hedef Takip Matematiği (v3.0):
 *  - Sabit hızlı 2D Kalman Kutusu Filtresi (konum + boyut yumuşatma + gizlenme tahmini)
 *  - O(N^3) Tam Global Optimal Eşleştirme (Hungarian / Kuhn-Munkres Algoritması) — iki kişi kesiştiğinde ID karışmasını önler!
 *  - Hızlı Laplacian Keskinlik Ölçümü (bulanık hareketli karelerin hafızayı kirletmesini engeller)
 */
public final class SmartTrackerMath {

    private SmartTrackerMath() {}

    /**
     * Sabit hızlı hafif 2D Kalman filtresi (sıfır matris tahsisi, <0.005 ms).
     */
    public static final class KalmanTrackState {
        public int trackId;
        public float cx, cy, w, h;
        public float vx, vy;
        public float vw, vh;
        public long lastUpdateMs;
        public long lastSeenMs;
        public int missedFrames;
        public float[] lastSig;
        public String name;
        public boolean hasDeepAi;
        public boolean hasFaceAi;

        public KalmanTrackState(int id, TrackedPerson p, long now) {
            this.trackId = id;
            this.cx = p.centerX();
            this.cy = p.centerY();
            this.w = p.nw;
            this.h = p.nh;
            this.vx = 0f;
            this.vy = 0f;
            this.vw = 0f;
            this.vh = 0f;
            this.lastUpdateMs = now;
            this.lastSeenMs = now;
            this.missedFrames = 0;
            this.lastSig = (p.signature != null) ? p.signature.clone() : null;
            this.name = p.name;
            this.hasDeepAi = p.hasDeepAi;
            this.hasFaceAi = p.hasFaceAi;
        }

        public void predict(long now) {
            long dt = Math.max(0L, Math.min(350L, now - lastUpdateMs));
            if (dt <= 0L) return;
            // Hızlı sönümleme: kişi kaybolduğunda kutu ekran dışına uçmasın
            float decay = (missedFrames > 0) ? 0.80f : 0.94f;
            vx *= decay;
            vy *= decay;
            vw *= 0.70f;
            vh *= 0.70f;

            cx = clamp(cx + vx * dt, 0.02f, 0.98f);
            cy = clamp(cy + vy * dt, 0.02f, 0.98f);
            w = clamp(w + vw * dt, 0.02f, 0.98f);
            h = clamp(h + vh * dt, 0.04f, 0.98f);
            lastUpdateMs = now;
        }

        public void update(TrackedPerson meas, long now, boolean crowded) {
            long dt = Math.max(12L, Math.min(400L, now - lastSeenMs));
            float mcx = meas.centerX();
            float mcy = meas.centerY();
            float mw = meas.nw;
            float mh = meas.nh;

            // Kalman kazancı (alpha-beta uyarlamalı): kalabalık çakışmada daha yumuşak güncelle
            float posGain = crowded ? 0.58f : 0.78f;
            float sizeGain = crowded ? 0.32f : 0.55f;
            float velGain = crowded ? 0.18f : 0.38f;

            float measVx = (mcx - cx) / dt;
            float measVy = (mcy - cy) / dt;
            // Aşırı zıplamaları engelle (max ~0.0025 norm/ms = saniyede 2.5 ekran)
            measVx = clamp(measVx, -0.0025f, 0.0025f);
            measVy = clamp(measVy, -0.0025f, 0.0025f);

            vx = vx * (1f - velGain) + measVx * velGain;
            vy = vy * (1f - velGain) + measVy * velGain;
            vw = vw * 0.75f + ((mw - w) / dt) * 0.25f;
            vh = vh * 0.75f + ((mh - h) / dt) * 0.25f;

            cx = cx + (mcx - cx) * posGain;
            cy = cy + (mcy - cy) * posGain;
            w = w + (mw - w) * sizeGain;
            h = h + (mh - h) * sizeGain;

            lastUpdateMs = now;
            lastSeenMs = now;
            missedFrames = 0;
            if (meas.signature != null && !crowded) {
                lastSig = meas.signature.clone();
            }
            if (meas.name != null && !meas.name.trim().isEmpty()) {
                name = meas.name;
            }
            if (meas.hasDeepAi) hasDeepAi = true;
            if (meas.hasFaceAi) hasFaceAi = true;
        }

        public float nx() { return clamp(cx - w * 0.5f, 0f, 0.98f); }
        public float ny() { return clamp(cy - h * 0.5f, 0f, 0.98f); }
    }

    /**
     * Global Maksimum Ağırlıklı Bipartite Eşleştirme (Hungarian / Kuhn-Munkres Algoritması).
     * Verilen rows x cols benzerlik/skor matrisi (yüksek = daha iyi eşleşme) için
     * toplam skoru maksimize eden 1'e 1 atamayı döndürür.
     *
     * @param scores     rows x cols skor matrisi (geçersiz eşleşmeler için <= minAccept)
     * @param minAccept  Bu eşiğin altındaki eşleşmeler kabul edilmez (-1 döner)
     * @return           Her satır i için eşleşen sütun j (veya eşleşme yoksa -1)
     */
    public static int[] solveMaxAssignment(float[][] scores, float minAccept) {
        int rows = scores.length;
        if (rows == 0) return new int[0];
        int cols = scores[0].length;
        int[] result = new int[rows];
        Arrays.fill(result, -1);
        if (cols == 0) return result;

        // Kare matrise genişlet (N x N) — geçersiz hücreler büyük maliyetle doldurulur
        int n = Math.max(rows, cols);
        float maxScore = minAccept;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (scores[i][j] > maxScore) maxScore = scores[i][j];
            }
        }
        float dummyCost = (maxScore - minAccept) + 10.0f;

        float[][] cost = new float[n + 1][n + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (i <= rows && j <= cols && scores[i - 1][j - 1] >= minAccept) {
                    cost[i][j] = maxScore - scores[i - 1][j - 1];
                } else {
                    cost[i][j] = dummyCost;
                }
            }
        }

        // 1-tabanlı Jonker/Hungarian potansiyel uygulaması — O(N^3), N<=16 için <0.02 ms!
        float[] u = new float[n + 1];
        float[] v = new float[n + 1];
        int[] p = new int[n + 1];
        int[] way = new int[n + 1];
        float[] minv = new float[n + 1];
        boolean[] used = new boolean[n + 1];

        for (int i = 1; i <= n; i++) {
            p[0] = i;
            int j0 = 0;
            Arrays.fill(minv, Float.MAX_VALUE);
            Arrays.fill(used, false);
            do {
                used[j0] = true;
                int i0 = p[j0];
                float delta = Float.MAX_VALUE;
                int j1 = 0;
                for (int j = 1; j <= n; j++) {
                    if (!used[j]) {
                        float cur = cost[i0][j] - u[i0] - v[j];
                        if (cur < minv[j]) {
                            minv[j] = cur;
                            way[j] = j0;
                        }
                        if (minv[j] < delta) {
                            delta = minv[j];
                            j1 = j;
                        }
                    }
                }
                for (int j = 0; j <= n; j++) {
                    if (used[j]) {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }
                j0 = j1;
            } while (p[j0] != 0);

            do {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0 != 0);
        }

        for (int j = 1; j <= cols; j++) {
            int rowIdx = p[j] - 1;
            int colIdx = j - 1;
            if (rowIdx >= 0 && rowIdx < rows && scores[rowIdx][colIdx] >= minAccept) {
                result[rowIdx] = colIdx;
            }
        }
        return result;
    }

    /**
     * Bir ARGB piksel tamponundaki belirli dikdörtgen bölgenin Laplacian keskinlik (netlik) puanını hesaplar.
     * Bulanık/hareketli kareler (~15-35) ile net/keskin kareleri (~90-900+) <0.05 ms'de ayırt eder.
     */
    public static float computeLaplacianSharpness(int[] argbPixels, int w, int h, int x0, int y0, int x1, int y1, int stride) {
        if (argbPixels == null || w < 4 || h < 4) return 0f;
        int sx0 = Math.max(1, x0);
        int sy0 = Math.max(1, y0);
        int sx1 = Math.min(w - 2, x1);
        int sy1 = Math.min(h - 2, y1);
        int step = Math.max(1, stride);
        if (sx1 <= sx0 + 2 || sy1 <= sy0 + 2) return 0f;

        double sum = 0.0;
        double sumSq = 0.0;
        int count = 0;
        for (int y = sy0; y < sy1; y += step) {
            int row = y * w;
            int rowUp = (y - 1) * w;
            int rowDn = (y + 1) * w;
            for (int x = sx0; x < sx1; x += step) {
                float c = fastLuma(argbPixels[row + x]);
                float up = fastLuma(argbPixels[rowUp + x]);
                float dn = fastLuma(argbPixels[rowDn + x]);
                float lf = fastLuma(argbPixels[row + x - 1]);
                float rt = fastLuma(argbPixels[row + x + 1]);
                double lap = (up + dn + lf + rt) - 4.0 * c;
                sum += lap;
                sumSq += lap * lap;
                count++;
            }
        }
        if (count < 8) return 0f;
        double mean = sum / count;
        double var = (sumSq / count) - (mean * mean);
        return (float) Math.max(0.0, var);
    }

    private static float fastLuma(int argb) {
        int r = (argb >> 16) & 0xFF;
        int g = (argb >> 8) & 0xFF;
        int b = argb & 0xFF;
        return 0.299f * r + 0.587f * g + 0.114f * b;
    }

    public static float clamp(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
