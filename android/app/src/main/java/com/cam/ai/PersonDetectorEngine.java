package com.cam.ai;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.Log;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.Detection;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector;
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * Ultra-Hızlı Çok-Katmanlı İnsan Tespit ve Kalıcı Hafızalı Re-ID Takip Motoru.
 * Özellikler:
 *  - SSD MobileNet V2 (GPU hızlandırmalı + otomatik CPU geri dönüşlü) tek hafif model.
 *  - 30 FPS Optik NCC Şablon Takibi (fastTrackFrame) + Arka Plan Asenkron AI Düzeltme.
 *  - 3 Bölgeli Merkez-Ağırlıklı Re-ID İmzası (Baş/Omuz, Gövde, Bacak: 16x4 HSV + 12 Gri = 228 boyut).
 *  - Kişi Karışmasını Önleyen Sıkı Gövde Kimlik Kapısı ve Çakışma (Occlusion) Dondurma.
 *  - Çoklu-Açı Hafıza Bankası (8 kareye kadar kilitli hedef hafızası + 90 sn oda kişi hafızası).
 */
public class PersonDetectorEngine {

    private static final String TAG = "PersonDetectorEngine";
    private static final String MODEL_FILE = "ssd_mobilenet_v2.tflite";
    private static final float FULL_MIN_SCORE = 0.29f;
    private static final float CROP_MIN_SCORE = 0.24f;
    private static final int MAX_FULL_SIDE = 448;

    private static final int ANA_W = 160;
    private static final int ANA_H = 120;
    private static final int HUE_BINS = 16;
    private static final int SAT_BINS = 4;
    private static final int GRAY_BINS = 12;
    private static final int ZONE_LEN = HUE_BINS * SAT_BINS + GRAY_BINS; // 76
    private static final int NUM_ZONES = 3; // 0: Baş/Omuz, 1: Gövde, 2: Bacak
    private static final int SIG_LEN = ZONE_LEN * NUM_ZONES; // 228
    private static final int TPL_W = 16;
    private static final int TPL_H = 24;

    private static final int MAX_LOCKED_GALLERY = 8;
    private static final int MAX_ROOM_MEMORY = 24;
    private static final long ROOM_MEMORY_TTL_MS = 90_000L;
    private static final long TARGET_LOST_TIMEOUT_MS = 1100L;

    public enum LockStatus {
        LOCKED,
        EMPTY_AREA
    }

    public static class LockedState {
        public int id;
        public float nx, ny, nw, nh;
        public float score;
        public float vx, vy;
        public float baseArea;
        public float aspectRatio;
        public float[] anchorSig;
        public List<float[]> gallery; // Slot 0 = değişmez anchor, 1..7 = temiz çoklu açı hafızası
        public float[] template;
        public long lastSeen;
        public long lastUpdate;
        public int recoverStreak;
        public boolean isOccluded;
        public boolean lost;
    }

    private static class RememberedPerson {
        int id;
        long lastSeen;
        final List<float[]> gallery = new ArrayList<>(4);
    }

    public static class AnalysisOutput {
        public List<TrackedPerson> persons;
        public LockedState locked;
        public boolean targetJustLost;
        public boolean targetJustRecovered;
        public int inferTimeMs;
    }

    private ObjectDetector detector;
    private volatile boolean initialized = false;
    private volatile boolean closed = false;
    private boolean usingGpu = false;
    private String initError = null;

    private final List<TrackedPerson> prevTracks = new ArrayList<>();
    private final List<RememberedPerson> roomMemory = new ArrayList<>();
    private int nextTrackId = 1;
    private LockedState lockedTarget = null;
    private int frameCounter = 0;

    // AI Worker thread tamponları (yalnızca detectorLock altında kullanılır)
    private final int[] aiPixels = new int[ANA_W * ANA_H];
    private final Bitmap aiSmallBitmap = Bitmap.createBitmap(ANA_W, ANA_H, Bitmap.Config.ARGB_8888);
    private final Canvas aiSmallCanvas = new Canvas(aiSmallBitmap);
    private final Rect aiSrcRect = new Rect();
    private final Rect aiDstRect = new Rect(0, 0, ANA_W, ANA_H);
    private boolean aiPixelsValid = false;

    // 30 FPS Optik Takip tamponları (yalnızca fastLock altında camera thread kullanır)
    private final int[] fastPixels = new int[ANA_W * ANA_H];
    private final Bitmap fastSmallBitmap = Bitmap.createBitmap(ANA_W, ANA_H, Bitmap.Config.ARGB_8888);
    private final Canvas fastSmallCanvas = new Canvas(fastSmallBitmap);
    private final Rect fastSrcRect = new Rect();
    private final Rect fastDstRect = new Rect(0, 0, ANA_W, ANA_H);

    private final Paint fastDrawPaint = new Paint();

    private Bitmap detBitmap = null;
    private Canvas detCanvas = null;
    private final Rect detSrcRect = new Rect();
    private final Rect detDstRect = new Rect();

    private Bitmap lastAiBitmap = null;
    private final Object stateLock = new Object();
    private final Object detectorLock = new Object();
    private final Object fastLock = new Object();

    public PersonDetectorEngine(Context context) {
        // Tüm Android telefonlarda (Mali/Adreno/PowerVR) sıfır sürücü çökmesi için güvenli CPU XNNPACK modu
        if (tryInitDetector(context, Delegate.CPU, false)) {
            usingGpu = false;
            initialized = true;
            Log.i(TAG, "SSD MobileNet V2 (Güvenli Hızlı CPU Modu) başarıyla başlatıldı.");
        } else {
            initialized = false;
            Log.e(TAG, "AI Başlatma Hatası: " + initError);
        }
    }

    private boolean tryInitDetector(Context context, Delegate delegate, boolean runWarmup) {
        ObjectDetector candidate = null;
        try {
            BaseOptions baseOpts = BaseOptions.builder()
                    .setModelAssetPath(MODEL_FILE)
                    .setDelegate(delegate)
                    .build();

            ObjectDetector.ObjectDetectorOptions opts = ObjectDetector.ObjectDetectorOptions.builder()
                    .setBaseOptions(baseOpts)
                    .setRunningMode(RunningMode.IMAGE)
                    .setMaxResults(12)
                    .setScoreThreshold(CROP_MIN_SCORE)
                    .setCategoryAllowlist(Collections.singletonList("person"))
                    .build();

            candidate = ObjectDetector.createFromOptions(context, opts);
            if (runWarmup) {
                Bitmap dummy = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888);
                MPImage mp = new BitmapImageBuilder(dummy).build();
                candidate.detect(mp);
                mp.close();
            }
            detector = candidate;
            return true;
        } catch (Throwable t) {
            initError = t.getMessage();
            Log.w(TAG, delegate.name() + " delegesi kullanılamadı: " + t.getMessage());
            if (candidate != null) {
                try { candidate.close(); } catch (Exception ignored) {}
            }
            return false;
        }
    }

    public boolean isInitialized() {
        return initialized;
    }

    public boolean isUsingGpu() {
        return usingGpu;
    }

    public String getInitError() {
        return initError;
    }

    /**
     * Kamera thread'inden her karede (<3ms) çalışarak kilitli hedefi 25-30 FPS yumuşaklıkta takip eder.
     * Yalnızca mevcut kilit pozisyonunu akıcı tutar; AI yokluğunu uzatmaz (kayıp alarmı AI kararına bağlı kalır).
     */
    public LockedState fastTrackFrame(Bitmap uprightBitmap) {
        if (closed || uprightBitmap == null || uprightBitmap.isRecycled()) {
            return getLockedStateSnapshot();
        }
        int idSnapshot;
        float predX, predY, pw, ph, pvx, pvy;
        float[] tplSnap;
        float[] anchorSnap;
        long now = System.currentTimeMillis();

        synchronized (stateLock) {
            if (lockedTarget == null || lockedTarget.lost || lockedTarget.isOccluded || lockedTarget.template == null) {
                return (lockedTarget != null) ? copyLockedState(lockedTarget) : null;
            }
            idSnapshot = lockedTarget.id;
            long dt = Math.min(80L, Math.max(1L, now - lockedTarget.lastUpdate));
            predX = Math.max(-0.10f, Math.min(0.95f, lockedTarget.nx + lockedTarget.vx * dt));
            predY = Math.max(-0.10f, Math.min(0.95f, lockedTarget.ny + lockedTarget.vy * dt));
            pw = lockedTarget.nw;
            ph = lockedTarget.nh;
            pvx = lockedTarget.vx;
            pvy = lockedTarget.vy;
            tplSnap = lockedTarget.template;
            anchorSnap = lockedTarget.anchorSig;
        }

        float[] hit = null;
        synchronized (fastLock) {
            if (closed || fastSmallBitmap.isRecycled() || uprightBitmap.isRecycled()) {
                return getLockedStateSnapshot();
            }
            fastSrcRect.set(0, 0, uprightBitmap.getWidth(), uprightBitmap.getHeight());
            fastSmallCanvas.drawBitmap(uprightBitmap, fastSrcRect, fastDstRect, fastDrawPaint);
            fastSmallBitmap.getPixels(fastPixels, 0, ANA_W, 0, 0, ANA_W, ANA_H);

            hit = searchTemplate(fastPixels, tplSnap, predX, predY, pw, ph, 0.72f);
            if (hit != null) {
                float[] sig = computeSignature(fastPixels, hit[0], hit[1], pw, ph);
                if (sig != null && anchorSnap != null) {
                    float torsoCheck = zoneSim(anchorSnap, sig, 1);
                    if (torsoCheck < 0.54f) {
                        hit = null;
                    }
                }
            }
        }

        synchronized (stateLock) {
            if (lockedTarget == null || lockedTarget.id != idSnapshot || lockedTarget.lost || lockedTarget.isOccluded) {
                return (lockedTarget != null) ? copyLockedState(lockedTarget) : null;
            }
            if (hit != null) {
                float prevCx = lockedTarget.nx + lockedTarget.nw * 0.5f;
                float prevCy = lockedTarget.ny + lockedTarget.nh * 0.5f;
                float newCx = hit[0] + pw * 0.5f;
                float newCy = hit[1] + ph * 0.5f;
                long dt = Math.max(8L, now - lockedTarget.lastUpdate);

                lockedTarget.vx = pvx * 0.65f + ((newCx - prevCx) / dt) * 0.35f;
                lockedTarget.vy = pvy * 0.65f + ((newCy - prevCy) / dt) * 0.35f;
                lockedTarget.nx = Math.max(0f, Math.min(0.98f, hit[0]));
                lockedTarget.ny = Math.max(0f, Math.min(0.98f, hit[1]));
                lockedTarget.lastUpdate = now;
            }
            return copyLockedState(lockedTarget);
        }
    }

    public List<TrackedPerson> getLatestPersonsSnapshot() {
        synchronized (stateLock) {
            return new ArrayList<>(prevTracks);
        }
    }

    public AnalysisOutput processFrame(Bitmap uprightBitmap) {
        AnalysisOutput out = new AnalysisOutput();
        out.persons = new ArrayList<>();
        if (closed || !initialized || uprightBitmap == null || uprightBitmap.isRecycled()) {
            return out;
        }

        long t0 = System.currentTimeMillis();
        int bw = uprightBitmap.getWidth();
        int bh = uprightBitmap.getHeight();
        frameCounter++;

        // 1. Tam kadraj AI insan tespiti & izole tamponlar
        List<TrackedPerson> candidates;
        LockedState currentLocked;
        synchronized (detectorLock) {
            if (closed || detector == null || uprightBitmap.isRecycled()) {
                return out;
            }
            Bitmap detBmp = uprightBitmap;
            int maxSide = Math.max(bw, bh);
            if (maxSide > MAX_FULL_SIDE) {
                float scale = (float) MAX_FULL_SIDE / maxSide;
                int dw = Math.max(1, Math.round(bw * scale));
                int dh = Math.max(1, Math.round(bh * scale));
                if (detBitmap == null || detBitmap.isRecycled() || detBitmap.getWidth() != dw || detBitmap.getHeight() != dh) {
                    if (detBitmap != null && !detBitmap.isRecycled()) detBitmap.recycle();
                    detBitmap = Bitmap.createBitmap(dw, dh, Bitmap.Config.ARGB_8888);
                    detCanvas = new Canvas(detBitmap);
                }
                detSrcRect.set(0, 0, bw, bh);
                detDstRect.set(0, 0, dw, dh);
                detCanvas.drawBitmap(uprightBitmap, detSrcRect, detDstRect, fastDrawPaint);
                detBmp = detBitmap;
            }
            MPImage mpImage = new BitmapImageBuilder(detBmp).build();
            ObjectDetectorResult res = detector.detect(mpImage);
            candidates = convertDetections(res.detections(), detBmp.getWidth(), detBmp.getHeight(), 0f, 0f, 1f, 1f, FULL_MIN_SCORE);

            lastAiBitmap = uprightBitmap;
            aiPixelsValid = false;

            synchronized (stateLock) {
                currentLocked = (lockedTarget != null) ? copyLockedState(lockedTarget) : null;
            }

            if (currentLocked != null) {
                boolean needZoom = candidates.isEmpty() || (currentLocked.nh < 0.20f);
                if (needZoom) {
                    float cx = currentLocked.nx + currentLocked.nw * 0.5f;
                    float cy = currentLocked.ny + currentLocked.nh * 0.5f;
                    float frac = Math.max(0.24f, Math.min(0.60f, Math.max(currentLocked.nw, currentLocked.nh) * 2.4f));
                    List<TrackedPerson> zoomDets = detectZoomedRegionInternal(uprightBitmap, cx, cy, frac);
                    candidates = mergeDetections(candidates, zoomDets);
                }
            } else if (candidates.isEmpty() && (frameCounter % 8 == 0)) {
                int slot = (frameCounter / 8) % 3;
                float cx = (slot == 0) ? 0.30f : (slot == 1 ? 0.50f : 0.70f);
                List<TrackedPerson> farDets = detectZoomedRegionInternal(uprightBitmap, cx, 0.44f, 0.46f);
                candidates = mergeDetections(candidates, farDets);
            }

            // 2. Re-ID imzalarını AI küçük tamponundan çıkar (detectorLock içinde tamamen thread-safe)
            if (!candidates.isEmpty() || currentLocked != null) {
                ensureAiPixelsLocked(uprightBitmap);
                for (TrackedPerson p : candidates) {
                    p.signature = computeSignature(aiPixels, p.nx, p.ny, p.nw, p.nh);
                }
            }

            synchronized (stateLock) {
                long now = System.currentTimeMillis();

                // 3. Kilitli hedef takibi + Hafıza Bankasına göre kimlik doğrulama
                if (lockedTarget != null) {
                    updateLockedTargetLocked(candidates, now, out);
                    out.locked = copyLockedState(lockedTarget);
                }

                // 4. Tüm kişilere kimlik ve 90 sn oda hafızası ile kararlı ID ata
                assignStableIdsWithMemoryLocked(candidates, now);
                out.persons = candidates;
            }
        }

        out.inferTimeMs = (int) (System.currentTimeMillis() - t0);
        return out;
    }

    private void ensureAiPixelsLocked(Bitmap src) {
        if (closed || aiPixelsValid || src == null || src.isRecycled() || aiSmallBitmap.isRecycled()) return;
        aiSrcRect.set(0, 0, src.getWidth(), src.getHeight());
        aiSmallCanvas.drawBitmap(src, aiSrcRect, aiDstRect, fastDrawPaint);
        aiSmallBitmap.getPixels(aiPixels, 0, ANA_W, 0, 0, ANA_W, ANA_H);
        aiPixelsValid = true;
    }

    public void clearFrameCache() {
        synchronized (detectorLock) {
            lastAiBitmap = null;
            aiPixelsValid = false;
        }
    }

    /**
     * Önce önbellekteki AI kişilerinden anında (<1ms) kilit dener; gerekirse allowZoom=true ile yakınlaştırma yapar.
     */
    public LockStatus tryLockAt(float normX, float normY, LockedState outLocked) {
        return tryLockInternal(normX, normY, outLocked, true);
    }

    public LockStatus tryLockFast(float normX, float normY, LockedState outLocked) {
        return tryLockInternal(normX, normY, outLocked, false);
    }

    private LockStatus tryLockInternal(float normX, float normY, LockedState outLocked, boolean allowZoom) {
        if (closed) return LockStatus.EMPTY_AREA;

        TrackedPerson best;
        synchronized (stateLock) {
            best = findPersonAtLocked(normX, normY, prevTracks);
        }

        if (best == null && allowZoom) {
            synchronized (detectorLock) {
                Bitmap snap = lastAiBitmap;
                if (!closed && snap != null && !snap.isRecycled()) {
                    List<TrackedPerson> zoomed = detectZoomedRegionInternal(snap, normX, normY, 0.36f);
                    best = findPersonAtLocked(normX, normY, zoomed);
                    if (best != null) {
                        ensureAiPixelsLocked(snap);
                        best.signature = computeSignature(aiPixels, best.nx, best.ny, best.nw, best.nh);
                    }
                }
            }
        }

        if (best == null) {
            return LockStatus.EMPTY_AREA;
        }

        synchronized (detectorLock) {
            Bitmap snap = lastAiBitmap;
            if (!closed && snap != null && !snap.isRecycled()) {
                ensureAiPixelsLocked(snap);
            }
            int[] pxSource = aiPixels;
            if (best.signature == null && aiPixelsValid) {
                best.signature = computeSignature(pxSource, best.nx, best.ny, best.nw, best.nh);
            }

            synchronized (stateLock) {
                long now = System.currentTimeMillis();
                LockedState t = new LockedState();
                t.id = (best.id > 0) ? best.id : nextTrackId++;
                t.nx = best.nx;
                t.ny = best.ny;
                t.nw = best.nw;
                t.nh = best.nh;
                t.score = best.score;
                t.vx = 0f;
                t.vy = 0f;
                t.baseArea = best.area();
                t.aspectRatio = best.nw / Math.max(0.01f, best.nh);
                t.anchorSig = (best.signature != null)
                        ? best.signature.clone()
                        : (aiPixelsValid ? computeSignature(pxSource, best.nx, best.ny, best.nw, best.nh) : null);
                t.gallery = new ArrayList<>(MAX_LOCKED_GALLERY);
                if (t.anchorSig != null) {
                    t.gallery.add(t.anchorSig.clone());
                }
                t.template = aiPixelsValid ? extractTemplate(pxSource, best.nx, best.ny, best.nw, best.nh) : null;
                t.lastSeen = now;
                t.lastUpdate = now;
                t.recoverStreak = 0;
                t.isOccluded = false;
                t.lost = false;

                lockedTarget = t;
                if (outLocked != null) {
                    copyInto(t, outLocked);
                }
                return LockStatus.LOCKED;
            }
        }
    }

    public void unlock() {
        synchronized (stateLock) {
            lockedTarget = null;
        }
    }

    public LockedState getLockedStateSnapshot() {
        synchronized (stateLock) {
            return (lockedTarget != null) ? copyLockedState(lockedTarget) : null;
        }
    }

    private TrackedPerson findPersonAtLocked(float nx, float ny, List<TrackedPerson> list) {
        if (list == null || list.isEmpty()) return null;
        TrackedPerson exact = null;
        float bestArea = Float.MAX_VALUE;
        for (TrackedPerson p : list) {
            if (p.contains(nx, ny)) {
                float a = p.area();
                if (a < bestArea) {
                    bestArea = a;
                    exact = p;
                }
            }
        }
        if (exact != null) return exact;

        TrackedPerson near = null;
        float minDist = 0.055f;
        for (TrackedPerson p : list) {
            float d = p.distanceToPoint(nx, ny);
            if (d < minDist) {
                minDist = d;
                near = p;
            }
        }
        return near;
    }

    private List<TrackedPerson> detectZoomedRegionInternal(Bitmap src, float centerX, float centerY, float frac) {
        if (!initialized || src == null || src.isRecycled()) return Collections.emptyList();
        try {
            int sw = src.getWidth();
            int sh = src.getHeight();
            float cw = Math.max(0.22f, Math.min(0.65f, frac));
            float ch = cw;
            float x0 = Math.max(0f, Math.min(1f - cw, centerX - cw * 0.5f));
            float y0 = Math.max(0f, Math.min(1f - ch, centerY - ch * 0.5f));

            int px = Math.max(0, Math.min(sw - 32, Math.round(x0 * sw)));
            int py = Math.max(0, Math.min(sh - 32, Math.round(y0 * sh)));
            int pw = Math.max(32, Math.min(sw - px, Math.round(cw * sw)));
            int ph = Math.max(32, Math.min(sh - py, Math.round(ch * sh)));

            Bitmap cropped = Bitmap.createBitmap(src, px, py, pw, ph);
            MPImage cropMp = new BitmapImageBuilder(cropped).build();
            ObjectDetectorResult res = detector.detect(cropMp);
            cropMp.close();

            return convertDetections(res.detections(), pw, ph, (float) px / sw, (float) py / sh, (float) pw / sw, (float) ph / sh, CROP_MIN_SCORE);
        } catch (Exception e) {
            return Collections.emptyList();
        }
    }

    private List<TrackedPerson> convertDetections(
            List<Detection> detections, int imgW, int imgH,
            float offX, float offY, float scaleW, float scaleH, float minScore) {
        List<TrackedPerson> list = new ArrayList<>();
        if (detections == null) return list;

        for (Detection d : detections) {
            if (d.categories().isEmpty()) continue;
            Category cat = d.categories().get(0);
            if (!"person".equalsIgnoreCase(cat.categoryName())) continue;
            float score = cat.score();
            if (score < minScore) continue;
            RectF bb = d.boundingBox();

            float nx = offX + (bb.left / imgW) * scaleW;
            float ny = offY + (bb.top / imgH) * scaleH;
            float nw = (bb.width() / imgW) * scaleW;
            float nh = (bb.height() / imgH) * scaleH;

            nx = Math.max(0f, Math.min(0.98f, nx));
            ny = Math.max(0f, Math.min(0.98f, ny));
            nw = Math.max(0.01f, Math.min(1f - nx, nw));
            nh = Math.max(0.02f, Math.min(1f - ny, nh));

            if (nw * nh > 0.0003f) {
                list.add(new TrackedPerson(0, nx, ny, nw, nh, score));
            }
        }
        return list;
    }

    private List<TrackedPerson> mergeDetections(List<TrackedPerson> primary, List<TrackedPerson> secondary) {
        List<TrackedPerson> merged = new ArrayList<>(primary);
        for (TrackedPerson s : secondary) {
            boolean dup = false;
            for (TrackedPerson p : merged) {
                if (computeIoU(p.nx, p.ny, p.nw, p.nh, s.nx, s.ny, s.nw, s.nh) > 0.42f) {
                    dup = true;
                    break;
                }
            }
            if (!dup) merged.add(s);
        }
        return merged;
    }

    /**
     * Kişilere hem görsel benzerlik hem 90 sn oda hafızası ile kararlı ID atar.
     * İki kişi yan yana geçtiğinde ID'ler yer değiştirmez; çıkıp tekrar girdiğinde aynı ID korunur.
     */
    private void assignStableIdsWithMemoryLocked(List<TrackedPerson> current, long now) {
        // Eski hafızayı temizle (kilitli ID varsa onu silme)
        Iterator<RememberedPerson> it = roomMemory.iterator();
        while (it.hasNext()) {
            RememberedPerson m = it.next();
            boolean isLocked = (lockedTarget != null && lockedTarget.id == m.id);
            if (!isLocked && (now - m.lastSeen > ROOM_MEMORY_TTL_MS)) {
                it.remove();
            }
        }

        List<TrackedPerson> unmatchedPrev = new ArrayList<>(prevTracks);
        List<TrackedPerson> needMemoryMatch = new ArrayList<>();
        List<Integer> usedIds = new ArrayList<>();

        // Kilitli hedef zaten belirlendiyse o ID'yi rezerve et (aynı karede iki kişiye aynı ID verilmesin)
        if (lockedTarget != null && !lockedTarget.lost) {
            for (TrackedPerson c : current) {
                if (computeIoU(c.nx, c.ny, c.nw, c.nh, lockedTarget.nx, lockedTarget.ny, lockedTarget.nw, lockedTarget.nh) > 0.55f) {
                    c.id = lockedTarget.id;
                    usedIds.add(c.id);
                    break;
                }
            }
        }

        // 1. Aşama: Önceki karedeki aktif kişilerle KONUM + GÖRSEL İMZA eşleştirmesi (karışmayı önler)
        for (TrackedPerson cur : current) {
            if (cur.id > 0) continue;

            TrackedPerson bestPrev = null;
            float bestScore = 0.40f;
            for (TrackedPerson prev : unmatchedPrev) {
                if (usedIds.contains(prev.id)) continue;
                float iou = computeIoU(cur.nx, cur.ny, cur.nw, cur.nh, prev.nx, prev.ny, prev.nw, prev.nh);
                float dist = (float) Math.hypot(cur.centerX() - prev.centerX(), cur.centerY() - prev.centerY());
                float posScore = iou * 1.4f + Math.max(0f, 0.35f - dist) * 2.2f;

                float appScore = 0.65f;
                float torso = 0.65f;
                if (cur.signature != null && prev.signature != null) {
                    appScore = sigSim(cur.signature, prev.signature);
                    torso = zoneSim(cur.signature, prev.signature, 1);
                    // Çoklu kişi ortamında gövde rengi tamamen uyuşmuyorsa yakın diye karıştırma!
                    if ((prevTracks.size() > 1 || current.size() > 1) && (appScore < 0.54f || torso < 0.48f)) {
                        continue;
                    }
                }
                float total = posScore + appScore * 2.4f + torso * 1.0f;
                if (total > bestScore) {
                    bestScore = total;
                    bestPrev = prev;
                }
            }

            if (bestPrev != null) {
                cur.id = bestPrev.id;
                usedIds.add(cur.id);
                unmatchedPrev.remove(bestPrev);
            } else {
                needMemoryMatch.add(cur);
            }
        }

        // 2. Aşama: Kadraja yeniden giren kişileri 90 saniyelik Re-ID hafızasında ara
        for (TrackedPerson cur : needMemoryMatch) {
            RememberedPerson bestMem = null;
            float bestSim = 0f;

            if (cur.signature != null) {
                for (RememberedPerson mem : roomMemory) {
                    if (usedIds.contains(mem.id)) continue;
                    // Kilitli bir hedef kayıpsa onun ID'sini rastgele birine verme (kilit kurtarma ayrıca doğrular)
                    if (lockedTarget != null && lockedTarget.id == mem.id) continue;

                    float maxG = 0f;
                    float bestTorso = 0f;
                    for (float[] gSig : mem.gallery) {
                        float sim = sigSim(gSig, cur.signature);
                        if (sim > maxG) {
                            maxG = sim;
                            bestTorso = zoneSim(gSig, cur.signature, 1);
                        }
                    }
                    if (maxG >= 0.73f && bestTorso >= 0.67f && maxG > bestSim) {
                        bestSim = maxG;
                        bestMem = mem;
                    }
                }
            }

            if (bestMem != null) {
                cur.id = bestMem.id;
                usedIds.add(cur.id);
            } else {
                while (usedIds.contains(nextTrackId) || (lockedTarget != null && lockedTarget.id == nextTrackId)) {
                    nextTrackId = (nextTrackId >= 999) ? 1 : nextTrackId + 1;
                }
                cur.id = nextTrackId++;
                if (nextTrackId > 999) nextTrackId = 1;
                usedIds.add(cur.id);
            }
        }

        // 3. Aşama: Hafızayı temiz çoklu açı örnekleriyle güncelle (kalabalık çakışma anında güncelleme!)
        for (TrackedPerson cur : current) {
            cur.lastSeen = now;
            if (cur.signature == null) continue;
            boolean crowded = false;
            for (TrackedPerson other : current) {
                if (other != cur && computeIoU(cur.nx, cur.ny, cur.nw, cur.nh, other.nx, other.ny, other.nw, other.nh) > 0.14f) {
                    crowded = true;
                    break;
                }
            }
            if (!crowded) {
                updateRoomMemoryLocked(cur.id, cur.signature, now);
            }
        }

        prevTracks.clear();
        prevTracks.addAll(current);
    }

    private void updateRoomMemoryLocked(int id, float[] sig, long now) {
        RememberedPerson target = null;
        for (RememberedPerson m : roomMemory) {
            if (m.id == id) {
                target = m;
                break;
            }
        }
        if (target == null) {
            if (roomMemory.size() >= MAX_ROOM_MEMORY) {
                RememberedPerson oldest = roomMemory.get(0);
                for (RememberedPerson m : roomMemory) {
                    if (m.lastSeen < oldest.lastSeen) oldest = m;
                }
                roomMemory.remove(oldest);
            }
            target = new RememberedPerson();
            target.id = id;
            target.gallery.add(sig.clone());
            roomMemory.add(target);
        } else {
            float bestSim = 0f;
            for (float[] g : target.gallery) {
                bestSim = Math.max(bestSim, sigSim(g, sig));
            }
            if (bestSim < 0.89f) {
                if (target.gallery.size() >= 4) {
                    target.gallery.remove(1); // Slot 0 temel referansı koru
                }
                target.gallery.add(sig.clone());
            }
        }
        target.lastSeen = now;
    }

    private void updateLockedTargetLocked(List<TrackedPerson> detections, long now, AnalysisOutput out) {
        LockedState t = lockedTarget;
        long dt = Math.max(1L, now - t.lastUpdate);

        float predX = t.nx + t.vx * dt;
        float predY = t.ny + t.vy * dt;
        float predW = t.nw;
        float predH = t.nh;
        float diag = (float) Math.hypot(t.nw, t.nh);

        TrackedPerson best = null;
        float bestScore = -Float.MAX_VALUE;
        float bestIoU = 0f;
        float bestGallerySimMatched = 0f;
        float bestAnchorSimMatched = 0f;

        for (TrackedPerson d : detections) {
            if (d.signature == null) continue;

            float iou = computeIoU(predX, predY, predW, predH, d.nx, d.ny, d.nw, d.nh);
            float pcx = predX + predW * 0.5f, pcy = predY + predH * 0.5f;
            float dist = (float) Math.hypot(pcx - d.centerX(), pcy - d.centerY());
            float dArea = d.area();
            float sizeRatio = Math.min(dArea / Math.max(1e-6f, t.baseArea), t.baseArea / Math.max(1e-6f, dArea));
            float dAspect = d.nw / Math.max(0.01f, d.nh);
            float aspectSim = Math.min(dAspect / Math.max(0.01f, t.aspectRatio), t.aspectRatio / Math.max(0.01f, dAspect));

            // Çoklu-Açı Hafıza Bankasındaki (slot 0..7) tüm görünümlerle kıyasla
            float maxGallerySim = 0f;
            float correspondingTorso = 0f;
            if (t.gallery != null && !t.gallery.isEmpty()) {
                for (float[] gSig : t.gallery) {
                    float s = sigSim(gSig, d.signature);
                    if (s > maxGallerySim) {
                        maxGallerySim = s;
                        correspondingTorso = zoneSim(gSig, d.signature, 1);
                    }
                }
            }
            float anchorSim = (t.anchorSig != null) ? sigSim(t.anchorSig, d.signature) : maxGallerySim;
            float anchorTorso = (t.anchorSig != null) ? zoneSim(t.anchorSig, d.signature, 1) : correspondingTorso;

            float score;
            if (!t.lost) {
                // Aktif takipte dahi gövde/görünüm eşiği zorunlu — yanından geçen başka insana KİLİT ATLAMAZ
                if (maxGallerySim < 0.56f || Math.max(correspondingTorso, anchorTorso) < 0.50f) continue;
                if (sizeRatio < 0.20f) continue;
                score = maxGallerySim * 3.8f + correspondingTorso * 1.6f + iou * 1.6f
                        + aspectSim * 0.5f - (dist / Math.max(0.05f, diag)) * 0.45f + sizeRatio * 0.4f;
            } else {
                // Kişi kameradan çıkıp tekrar girdiğinde Re-ID geri yakalama (uzaklık farkı esnek: sizeRatio >= 0.10)
                if (maxGallerySim < 0.72f || correspondingTorso < 0.66f) continue;
                if (sizeRatio < 0.10f || aspectSim < 0.35f) continue;
                score = maxGallerySim * 4.2f + correspondingTorso * 2.0f + anchorSim * 1.2f + sizeRatio * 0.4f;
            }

            if (score > bestScore) {
                bestScore = score;
                best = d;
                bestIoU = iou;
                bestGallerySimMatched = maxGallerySim;
                bestAnchorSimMatched = anchorSim;
            }
        }

        // Çakışma (occlusion) kontrolü: iki kişi üst üste ise şablon ve hafıza güncellemesini dondur
        boolean conflicted = false;
        if (best != null) {
            for (TrackedPerson other : detections) {
                if (other != best && computeIoU(other.nx, other.ny, other.nw, other.nh, best.nx, best.ny, best.nw, best.nh) > 0.12f) {
                    conflicted = true;
                    break;
                }
            }
        }
        t.isOccluded = conflicted;

        // Kayıpken kişi geri göründüyse 2 kare (veya >=0.80 yüksek eşleşme) ile doğrula ve alarmı kapat
        if (t.lost) {
            if (best != null) {
                t.recoverStreak++;
                if (t.recoverStreak >= 2 || bestGallerySimMatched >= 0.80f) {
                    t.lost = false;
                    t.recoverStreak = 0;
                    t.vx = 0f;
                    t.vy = 0f;
                    applyMatchedPosition(t, best, now, conflicted, bestIoU, bestGallerySimMatched, bestAnchorSimMatched);
                    out.targetJustRecovered = true;
                }
            } else {
                t.recoverStreak = 0;
            }
            return;
        }

        // Aktif takipte AI bulduysa güncelle ve çoklu açı hafızasını besle
        if (best != null) {
            applyMatchedPosition(t, best, now, conflicted, bestIoU, bestGallerySimMatched, bestAnchorSimMatched);
            return;
        }

        // AI bu karede kaçırdıysa ama kayıp süresi dolmadıysa NCC piksel şablonu + hız ile tut
        float[] tplHit = (now - t.lastSeen < 650 && t.template != null)
                ? searchTemplate(aiPixels, t.template, predX, predY, predW, predH, 0.70f)
                : null;
        if (tplHit != null) {
            t.nx = Math.max(0f, Math.min(0.98f, tplHit[0]));
            t.ny = Math.max(0f, Math.min(0.98f, tplHit[1]));
            t.lastUpdate = now;
        } else if (now - t.lastSeen < 450) {
            t.nx = Math.max(-0.12f, Math.min(0.95f, predX));
            t.ny = Math.max(-0.12f, Math.min(0.95f, predY));
            t.lastUpdate = now;
        } else if (now - t.lastSeen >= TARGET_LOST_TIMEOUT_MS) {
            // 1.1 saniye boyunca hem AI hem şablon doğrulayamadı -> HEDEF KAYIP, ALARM ÇAL!
            // Hafıza bankası (t.gallery ve anchorSig) korunur; kişi geri geldiğinde otomatik tanır!
            t.lost = true;
            t.recoverStreak = 0;
            out.targetJustLost = true;
        }
    }

    private void applyMatchedPosition(
            LockedState t, TrackedPerson best, long now,
            boolean conflicted, float bestIoU, float maxGallerySim, float anchorSim) {
        float prevCx = t.nx + t.nw * 0.5f;
        float prevCy = t.ny + t.nh * 0.5f;
        float newCx = best.centerX();
        float newCy = best.centerY();
        long dt = Math.max(8L, now - t.lastUpdate);

        float alpha = conflicted ? 0.18f : 0.42f;
        t.vx = t.vx * (1f - alpha) + ((newCx - prevCx) / dt) * alpha;
        t.vy = t.vy * (1f - alpha) + ((newCy - prevCy) / dt) * alpha;

        t.nx = best.nx;
        t.ny = best.ny;
        t.nw = best.nw;
        t.nh = best.nh;
        t.score = best.score;
        t.lastSeen = now;
        t.lastUpdate = now;

        // Sadece çakışma yoksa ve orijinal anchor ile güçlü bağ korunuyorsa hafızayı çeşitlendir
        if (!conflicted && best.signature != null && best.score >= 0.36f && bestIoU > 0.30f && anchorSim >= 0.62f) {
            if (t.gallery == null) t.gallery = new ArrayList<>(MAX_LOCKED_GALLERY);
            if (t.gallery.isEmpty() && t.anchorSig != null) t.gallery.add(t.anchorSig.clone());
            if (maxGallerySim < 0.90f) {
                if (t.gallery.size() >= MAX_LOCKED_GALLERY) {
                    t.gallery.remove(1); // Slot 0 değişmez anchor!
                }
                t.gallery.add(best.signature.clone());
            }
            float[] newTpl = extractTemplate(aiPixels, t.nx, t.ny, t.nw, t.nh);
            if (newTpl != null) t.template = newTpl;
        }
    }

    // =================== 3-BÖLGELİ MERKEZ AĞIRLIKLI RE-ID İMZASI & NCC PİKSEL ŞABLONU ===================
    private float[] computeSignature(int[] pixels, float nx, float ny, float nw, float nh) {
        int x0 = Math.max(0, (int) Math.floor((nx + nw * 0.16f) * ANA_W));
        int x1 = Math.min(ANA_W, (int) Math.ceil((nx + nw * 0.84f) * ANA_W));
        int yHead0 = Math.max(0, (int) Math.floor((ny + nh * 0.04f) * ANA_H));
        int yHead1 = Math.min(ANA_H, Math.max(yHead0 + 1, (int) Math.floor((ny + nh * 0.25f) * ANA_H)));
        int yTorso1 = Math.min(ANA_H, Math.max(yHead1 + 1, (int) Math.floor((ny + nh * 0.56f) * ANA_H)));
        int yLegs1 = Math.min(ANA_H, Math.max(yTorso1 + 1, (int) Math.ceil((ny + nh * 0.92f) * ANA_H)));
        if (x1 - x0 < 2 || yLegs1 - yHead0 < 6) return null;

        float centerX = (x0 + x1) * 0.5f;
        float halfW = Math.max(1f, (x1 - x0) * 0.5f);
        float[] sig = new float[SIG_LEN];
        accumulateZoneHist(pixels, x0, yHead0, x1, yHead1, centerX, halfW, sig, 0);
        accumulateZoneHist(pixels, x0, yHead1, x1, yTorso1, centerX, halfW, sig, ZONE_LEN);
        accumulateZoneHist(pixels, x0, yTorso1, x1, yLegs1, centerX, halfW, sig, ZONE_LEN * 2);
        return sig;
    }

    private void accumulateZoneHist(
            int[] pixels, int x0, int y0, int x1, int y1,
            float cx, float halfW, float[] out, int offset) {
        float wSum = 0f;
        for (int py = y0; py < y1; py++) {
            int row = py * ANA_W;
            for (int px = x0; px < x1; px++) {
                int argb = pixels[row + px];
                int r = (argb >> 16) & 0xFF;
                int g = (argb >> 8) & 0xFF;
                int b = argb & 0xFF;

                int max = Math.max(r, Math.max(g, b));
                int min = Math.min(r, Math.min(g, b));
                float v = max / 255f;
                float s = (max == 0) ? 0f : (float) (max - min) / max;

                // Merkez ağırlıklı pencere: kenarlardaki arka plan duvar/eşyayı bastırır
                float ndx = (px + 0.5f - cx) / halfW;
                float weight = Math.max(0.25f, 1.0f - 0.65f * ndx * ndx);

                int bin;
                if (s < 0.16f || v < 0.12f) {
                    int gb = Math.min(GRAY_BINS - 1, (int) Math.floor(v * GRAY_BINS));
                    bin = HUE_BINS * SAT_BINS + gb;
                } else {
                    float d = max - min;
                    float hue;
                    if (max == r) hue = ((g - b) / d) % 6f;
                    else if (max == g) hue = (b - r) / d + 2f;
                    else hue = (r - g) / d + 4f;
                    hue = (hue * 60f + 360f) % 360f;
                    int hb = ((int) Math.floor(hue / 360f * HUE_BINS)) % HUE_BINS;
                    int sb = Math.min(SAT_BINS - 1, (int) Math.floor((s - 0.16f) / 0.84f * SAT_BINS));
                    bin = hb * SAT_BINS + sb;
                }
                out[offset + bin] += weight;
                wSum += weight;
            }
        }
        if (wSum > 0f) {
            for (int i = 0; i < ZONE_LEN; i++) {
                out[offset + i] /= wSum;
            }
        }
    }

    private float zoneSim(float[] a, float[] b, int zoneIdx) {
        int off = zoneIdx * ZONE_LEN;
        float s = 0f;
        for (int i = 0; i < ZONE_LEN; i++) {
            s += (float) Math.sqrt(a[off + i] * b[off + i]);
        }
        return s;
    }

    private float sigSim(float[] a, float[] b) {
        float head = zoneSim(a, b, 0);
        float torso = zoneSim(a, b, 1);
        float legs = zoneSim(a, b, 2);
        return head * 0.22f + torso * 0.50f + legs * 0.28f;
    }

    private float[] extractTemplate(int[] pixels, float nx, float ny, float nw, float nh) {
        if (nw <= 0.004f || nh <= 0.008f) return null;
        float[] tpl = new float[TPL_W * TPL_H];
        float sum = 0f;
        for (int ty = 0; ty < TPL_H; ty++) {
            int py = Math.min(ANA_H - 1, Math.max(0, (int) Math.floor((ny + (ty + 0.5f) / TPL_H * nh) * ANA_H)));
            int row = py * ANA_W;
            for (int tx = 0; tx < TPL_W; tx++) {
                int px = Math.min(ANA_W - 1, Math.max(0, (int) Math.floor((nx + (tx + 0.5f) / TPL_W * nw) * ANA_W)));
                int argb = pixels[row + px];
                int r = (argb >> 16) & 0xFF;
                int g = (argb >> 8) & 0xFF;
                int b = argb & 0xFF;
                float lum = 0.299f * r + 0.587f * g + 0.114f * b;
                tpl[ty * TPL_W + tx] = lum;
                sum += lum;
            }
        }
        float mean = sum / tpl.length;
        float normSq = 0f;
        for (int i = 0; i < tpl.length; i++) {
            tpl[i] -= mean;
            normSq += tpl[i] * tpl[i];
        }
        float norm = (float) Math.sqrt(normSq);
        if (norm < 1e-3f) norm = 1f;
        for (int i = 0; i < tpl.length; i++) {
            tpl[i] /= norm;
        }
        return tpl;
    }

    private float[] searchTemplate(int[] pixels, float[] tpl, float predX, float predY, float w, float h, float minCorr) {
        float cx = predX + w * 0.5f;
        float cy = predY + h * 0.5f;
        float stepX = Math.max(0.008f, w * 0.14f);
        float stepY = Math.max(0.008f, h * 0.14f);

        float bestScore = -1f;
        float bestNx = 0f, bestNy = 0f;
        boolean found = false;

        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                float candX = cx + dx * stepX - w * 0.5f;
                float candY = cy + dy * stepY - h * 0.5f;
                if (candX < -0.05f || candY < -0.05f || candX + w > 1.05f || candY + h > 1.05f) continue;

                float[] cand = extractTemplate(pixels, candX, candY, w, h);
                if (cand == null) continue;
                float dot = 0f;
                for (int i = 0; i < tpl.length; i++) dot += tpl[i] * cand[i];
                if (dot > bestScore) {
                    bestScore = dot;
                    bestNx = candX;
                    bestNy = candY;
                    found = true;
                }
            }
        }
        return (found && bestScore > minCorr) ? new float[]{bestNx, bestNy} : null;
    }

    private float computeIoU(float x1, float y1, float w1, float h1, float x2, float y2, float w2, float h2) {
        float xA = Math.max(x1, x2);
        float yA = Math.max(y1, y2);
        float xB = Math.min(x1 + w1, x2 + w2);
        float yB = Math.min(y1 + h1, y2 + h2);
        float inter = Math.max(0f, xB - xA) * Math.max(0f, yB - yA);
        float union = Math.max(1e-6f, w1 * h1) + Math.max(1e-6f, w2 * h2) - inter + 1e-6f;
        return inter / union;
    }

    private LockedState copyLockedState(LockedState src) {
        LockedState dst = new LockedState();
        copyInto(src, dst);
        return dst;
    }

    private void copyInto(LockedState src, LockedState dst) {
        dst.id = src.id;
        dst.nx = src.nx;
        dst.ny = src.ny;
        dst.nw = src.nw;
        dst.nh = src.nh;
        dst.score = src.score;
        dst.vx = src.vx;
        dst.vy = src.vy;
        dst.baseArea = src.baseArea;
        dst.aspectRatio = src.aspectRatio;
        dst.lastSeen = src.lastSeen;
        dst.lastUpdate = src.lastUpdate;
        dst.recoverStreak = src.recoverStreak;
        dst.isOccluded = src.isOccluded;
        dst.lost = src.lost;
    }

    public void close() {
        closed = true;
        synchronized (detectorLock) {
            synchronized (fastLock) {
                lastAiBitmap = null;
                aiPixelsValid = false;
                if (detBitmap != null && !detBitmap.isRecycled()) detBitmap.recycle();
                detBitmap = null;
                if (!aiSmallBitmap.isRecycled()) aiSmallBitmap.recycle();
                if (!fastSmallBitmap.isRecycled()) fastSmallBitmap.recycle();
            }
            try {
                if (detector != null) detector.close();
            } catch (Exception ignored) {}
            detector = null;
        }
    }
}
