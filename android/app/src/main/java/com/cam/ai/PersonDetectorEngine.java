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
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * Ultra-Hızlı 3-Katmanlı Yapay Zeka İnsan Tespit, Biyometrik Yüz + Derin Re-ID ve Kalıcı Disk Hafızalı Takip Motoru.
 *
 * Katman 0 (Her Kamera Karesi, <1ms, Sıfır-GC): 30 FPS Optik NCC Şablon Takibi (fastTrackFrame).
 * Katman 1 (Her AI Karesi, ~30ms): SSD MobileNet V2 + 129-D Çok-İpuculu İmza (Histogram + Chroma + Yapısal Desen).
 * Katman 2 (Olay-Odaklı & Akıllı Zamanlamalı Derin AI):
 *   - MediaPipe MobileNetV3-Small (1024-D Derin Vücut/Kıyafet Re-ID)
 *   - MediaPipe BlazeFace + LiteRT MobileFaceNet (192-D Göz-Hizalamalı Biyometrik Yüz Tanıma)
 *   - Kıyafet değişiminde yüz doğrulaması ile otomatik yeni kıyafet öğrenme + yüz uyuşmazlığında sahtekarlık vetosu.
 * Katman 3 (Kalıcı Disk Hafızası): Uygulama kapansa dahi kişileri, yüz/vücut vektörlerini ve verilen isimleri saklar.
 */
public class PersonDetectorEngine {

    private static final String TAG = "PersonDetectorEngine";
    private static final String MODEL_FILE = "ssd_mobilenet_v2.tflite";
    private static final float FULL_MIN_SCORE = 0.29f;
    private static final float CROP_MIN_SCORE = 0.24f;
    private static final int MAX_FULL_SIDE = 352; // SSD 300x300 girişi için optimum ultra-hızlı boyut

    private static final int ANA_W = 160;
    private static final int ANA_H = 120;
    private static final int HUE_BINS = 12;
    private static final int GRAY_BINS = 8;
    private static final int ZONE_LEN = HUE_BINS + GRAY_BINS; // 20
    private static final int NUM_ZONES = 3; // 0: Baş/Omuz, 1: Gövde, 2: Bacak
    private static final int HIST_TOTAL = ZONE_LEN * NUM_ZONES; // 60
    private static final int CHROMA_LEN = NUM_ZONES * 3; // 9 (rg, yb, luma)
    private static final int PATCH_W = 6;
    private static final int PATCH_H = 10;
    private static final int PATCH_LEN = PATCH_W * PATCH_H; // 60
    private static final int SIG_LEN = HIST_TOTAL + CHROMA_LEN + PATCH_LEN; // 129
    private static final int TPL_W = 16;
    private static final int TPL_H = 24;

    private static final int MAX_LOCKED_GALLERY = 12;
    private static final int MAX_LOCKED_DEEP_GALLERY = 8;
    private static final int MAX_LOCKED_FACE_GALLERY = 8;
    private static final long TARGET_LOST_TIMEOUT_MS = 1100L;

    // Akıllı Derin AI Zamanlayıcı (normal karelerde 0ms ek yük, kritik anlarda anında derin doğrulama)
    private static final int MAX_NEURAL_PER_FRAME = 2;
    private static final long ENRICH_FAST_INTERVAL_MS = 900L;
    private static final long ENRICH_STEADY_INTERVAL_MS = 2200L;

    public enum LockStatus {
        LOCKED,
        EMPTY_AREA
    }

    public static class LockedState {
        public int id;
        public String name;
        public float nx, ny, nw, nh;
        public float score;
        public float vx, vy;
        public float baseArea;
        public float aspectRatio;
        public float[] anchorSig;
        public List<float[]> gallery; // 129-D çoklu açı/ışık imzaları (slot 0 = anchor)
        public List<float[]> deepGallery; // 1024-D MobileNetV3 derin vücut vektörleri
        public List<float[]> faceGallery; // 192-D MobileFaceNet biyometrik yüz vektörleri
        public float[] template;
        public long lastSeen;
        public long lastUpdate;
        public int recoverStreak;
        public int memoryCount;
        public int deepCount;
        public int faceCount;
        public boolean hasDeepAi;
        public boolean hasFaceAi;
        public boolean needsImmediateNeural;
        public boolean isOccluded;
        public boolean lost;

        public String displayLabel() {
            return (name != null && !name.trim().isEmpty()) ? name.trim() : ("Kişi #" + id);
        }
    }

    public static class AnalysisOutput {
        public List<TrackedPerson> persons;
        public LockedState locked;
        public boolean targetJustLost;
        public boolean targetJustRecovered;
        public int inferTimeMs;
        public int rememberedCount;
    }

    private ObjectDetector detector;
    private final NeuralIdentityEngine neuralEngine;
    private final PersonMemoryStore memoryStore;
    private volatile boolean initialized = false;
    private volatile boolean closed = false;
    private boolean usingGpu = false;
    private String initError = null;

    private final List<TrackedPerson> prevTracks = new ArrayList<>();
    private final List<PersonMemoryStore.PersonProfile> roomMemory = new ArrayList<>();
    private int nextTrackId = 1;
    private TrackedPerson pendingRecoveryCandidate = null;
    private LockedState lockedTarget = null;
    private int frameCounter = 0;
    private long lastEnrichTs = 0L;
    private boolean memoryDirty = false;

    // AI Worker thread tamponları (yalnızca detectorLock altında kullanılır)
    private final int[] aiPixels = new int[ANA_W * ANA_H];
    private final Bitmap aiSmallBitmap = Bitmap.createBitmap(ANA_W, ANA_H, Bitmap.Config.ARGB_8888);
    private final Canvas aiSmallCanvas = new Canvas(aiSmallBitmap);
    private final Rect aiSrcRect = new Rect();
    private final Rect aiDstRect = new Rect(0, 0, ANA_W, ANA_H);
    private final float[] aiTplScratch = new float[TPL_W * TPL_H];
    private final float[] aiHitScratch = new float[2];
    private boolean aiPixelsValid = false;

    // 30 FPS Optik Takip tamponları (yalnızca fastLock altında, SIFIR per-kare tahsis!)
    private final int[] fastPixels = new int[ANA_W * ANA_H];
    private final Bitmap fastSmallBitmap = Bitmap.createBitmap(ANA_W, ANA_H, Bitmap.Config.ARGB_8888);
    private final Canvas fastSmallCanvas = new Canvas(fastSmallBitmap);
    private final Rect fastSrcRect = new Rect();
    private final Rect fastDstRect = new Rect(0, 0, ANA_W, ANA_H);
    private final float[] fastTplScratch = new float[TPL_W * TPL_H];
    private final float[] fastSigScratch = new float[SIG_LEN];
    private final float[] fastHitScratch = new float[2];

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
        this(context, null);
    }

    public PersonDetectorEngine(Context context, File customMemoryFile) {
        this.memoryStore = (customMemoryFile != null)
                ? new PersonMemoryStore(customMemoryFile)
                : new PersonMemoryStore(context);

        // Kalıcı disk hafızasını yükle
        PersonMemoryStore.LoadResult loaded = memoryStore.load();
        synchronized (stateLock) {
            roomMemory.addAll(loaded.profiles);
            nextTrackId = loaded.nextTrackId;
        }

        // Derin AI (MobileNetV3 + BlazeFace + MobileFaceNet) motorunu başlat
        this.neuralEngine = new NeuralIdentityEngine(context);

        // SSD MobileNet V2 insan tespiti (tüm telefonlarda 0 sürücü çökmesi için güvenli CPU XNNPACK)
        if (tryInitDetector(context, Delegate.CPU, false)) {
            usingGpu = false;
            initialized = true;
            Log.i(TAG, "Hibrit AI Motoru Hazır: " + neuralEngine.getActiveEnginesSummary()
                    + " | Kalıcı Hafıza: " + loaded.profiles.size() + " kişi.");
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

    public boolean isDeepReIdReady() {
        return neuralEngine != null && neuralEngine.isBodyReady();
    }

    public boolean isFaceAiReady() {
        return neuralEngine != null && neuralEngine.isFaceReady();
    }

    public String getAiEngineSummary() {
        return (neuralEngine != null) ? neuralEngine.getActiveEnginesSummary() : "SSD + 129-D";
    }

    public String getInitError() {
        return initError;
    }

    public int getRememberedCount() {
        synchronized (stateLock) {
            return roomMemory.size();
        }
    }

    /**
     * Kamera thread'inden her karede (<1ms, SIFIR heap tahsisi) çalışarak kilitli hedefi 30 FPS yumuşaklıkta takip eder.
     * AI motorlarını ASLA beklemez.
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

        boolean foundHit = false;
        float hitX = 0f, hitY = 0f;
        synchronized (fastLock) {
            if (closed || fastSmallBitmap.isRecycled() || uprightBitmap.isRecycled()) {
                return getLockedStateSnapshot();
            }
            fastSrcRect.set(0, 0, uprightBitmap.getWidth(), uprightBitmap.getHeight());
            fastSmallCanvas.drawBitmap(uprightBitmap, fastSrcRect, fastDstRect, fastDrawPaint);
            fastSmallBitmap.getPixels(fastPixels, 0, ANA_W, 0, 0, ANA_W, ANA_H);

            if (searchTemplateZeroAlloc(fastPixels, tplSnap, predX, predY, pw, ph, 0.72f, fastTplScratch, fastHitScratch)) {
                hitX = fastHitScratch[0];
                hitY = fastHitScratch[1];
                foundHit = true;
                if (anchorSnap != null && computeSignatureInto(fastPixels, hitX, hitY, pw, ph, fastSigScratch)) {
                    float torsoCheck = zoneSim(anchorSnap, fastSigScratch, 1);
                    if (torsoCheck < 0.50f) {
                        foundHit = false;
                    }
                }
            }
        }

        synchronized (stateLock) {
            if (lockedTarget == null || lockedTarget.id != idSnapshot || lockedTarget.lost || lockedTarget.isOccluded) {
                return (lockedTarget != null) ? copyLockedState(lockedTarget) : null;
            }
            if (foundHit) {
                float prevCx = lockedTarget.nx + lockedTarget.nw * 0.5f;
                float prevCy = lockedTarget.ny + lockedTarget.nh * 0.5f;
                float newCx = hitX + pw * 0.5f;
                float newCy = hitY + ph * 0.5f;
                long dt = Math.max(8L, now - lockedTarget.lastUpdate);

                lockedTarget.vx = pvx * 0.65f + ((newCx - prevCx) / dt) * 0.35f;
                lockedTarget.vy = pvy * 0.65f + ((newCy - prevCy) / dt) * 0.35f;
                lockedTarget.nx = Math.max(0f, Math.min(0.98f, hitX));
                lockedTarget.ny = Math.max(0f, Math.min(0.98f, hitY));
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

        List<TrackedPerson> candidates;
        LockedState currentLocked;
        List<TrackedPerson> prevSnapshot;
        List<PersonMemoryStore.PersonProfile> saveSnapshot = null;
        int saveNextId = 1;

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
            // NOT: detBmp yeniden kullanıldığı için mpImage.close() çağrılmaz (close() bitmap'i recycle eder).
            MPImage mpImage = new BitmapImageBuilder(detBmp).build();
            ObjectDetectorResult res = detector.detect(mpImage);
            candidates = convertDetections(res.detections(), detBmp.getWidth(), detBmp.getHeight(), 0f, 0f, 1f, 1f, FULL_MIN_SCORE);

            lastAiBitmap = uprightBitmap;
            aiPixelsValid = false;

            synchronized (stateLock) {
                currentLocked = (lockedTarget != null) ? copyLockedState(lockedTarget) : null;
                prevSnapshot = new ArrayList<>(prevTracks);
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

            // 2. Hızlı 129-D Re-ID imzalarını AI küçük tamponundan çıkar (<0.2ms)
            if (!candidates.isEmpty() || currentLocked != null) {
                ensureAiPixelsLocked(uprightBitmap);
                for (TrackedPerson p : candidates) {
                    p.signature = computeSignature(aiPixels, p.nx, p.ny, p.nw, p.nh);
                }
            }

            // 3. Akıllı Olay-Odaklı Derin AI (MobileNetV3 + BlazeFace + MobileFaceNet) — stateLock TUTMADAN çalışır!
            if (!candidates.isEmpty() && (neuralEngine.isBodyReady() || neuralEngine.isFaceReady())) {
                long nowPre = System.currentTimeMillis();
                runAdaptiveNeuralExtractionLocked(uprightBitmap, candidates, prevSnapshot, currentLocked, nowPre);
            }

            // 4. Kimlik eşleştirme & hafıza güncelleme (kısa stateLock bloğu: <0.3ms)
            synchronized (stateLock) {
                long now = System.currentTimeMillis();
                pendingRecoveryCandidate = null;
                if (lockedTarget != null) {
                    updateLockedTargetLocked(candidates, now, out);
                    out.locked = copyLockedState(lockedTarget);
                }

                assignStableIdsWithMemoryLocked(candidates, now);
                out.persons = candidates;
                out.rememberedCount = roomMemory.size();

                if (memoryDirty) {
                    memoryDirty = false;
                    saveSnapshot = snapshotRoomMemoryLocked();
                    saveNextId = nextTrackId;
                }
            }
        }

        if (saveSnapshot != null) {
            memoryStore.saveDebouncedAsync(saveSnapshot, saveNextId);
        }

        out.inferTimeMs = (int) (System.currentTimeMillis() - t0);
        return out;
    }

    /**
     * Akıllı zamanlayıcı: Normal sabit takipte karelerin ~%90'ında 0ms harcar.
     * Kişi yeni girdiğinde, kilitli hedef kayıpken geri döndüğünde, ilk kilitte,
     * çakışmada veya periyodik zenginleştirmede Derin Vücut + Yüz AI vektörlerini çıkarır.
     */
    private void runAdaptiveNeuralExtractionLocked(
            Bitmap fullBitmap,
            List<TrackedPerson> candidates,
            List<TrackedPerson> prevSnapshot,
            LockedState lockedSnap,
            long now) {
        int budget = MAX_NEURAL_PER_FRAME;
        List<TrackedPerson> selected = new ArrayList<>(MAX_NEURAL_PER_FRAME);

        // Öncelik 1: Kilitli hedef KAYIP ise geri dönüş adaylarına anında Derin Vücut + Yüz AI çalıştır!
        if (lockedSnap != null && lockedSnap.lost) {
            List<TrackedPerson> byArea = new ArrayList<>(candidates);
            Collections.sort(byArea, (a, b) -> Float.compare(b.area(), a.area()));
            for (TrackedPerson c : byArea) {
                if (budget <= 0) break;
                selected.add(c);
                budget--;
            }
        }

        // Öncelik 2: Kilitli hedef yeni kilitlendiyse (tryLockFast) ilk AI karesinde hemen zenginleştir
        if (budget > 0 && lockedSnap != null && !lockedSnap.lost && lockedSnap.needsImmediateNeural) {
            TrackedPerson bestLockCand = findBestOverlapCandidate(candidates, lockedSnap.nx, lockedSnap.ny, lockedSnap.nw, lockedSnap.nh);
            if (bestLockCand != null && !selected.contains(bestLockCand)) {
                selected.add(bestLockCand);
                budget--;
            }
        }

        // Öncelik 3: Kadraja YENİ GİREN (önceki karede örtüşmeyen) kişiler -> kalıcı/oda hafızasından AI ile tanı!
        if (budget > 0) {
            for (TrackedPerson c : candidates) {
                if (budget <= 0) break;
                if (selected.contains(c)) continue;
                boolean matchedPrevPos = false;
                for (TrackedPerson prev : prevSnapshot) {
                    if (computeIoU(c.nx, c.ny, c.nw, c.nh, prev.nx, prev.ny, prev.nw, prev.nh) > 0.18f) {
                        matchedPrevPos = true;
                        break;
                    }
                }
                if (!matchedPrevPos) {
                    selected.add(c);
                    budget--;
                }
            }
        }

        // Öncelik 4: İki kişi birbiriyle çakışıyorsa (kimlik karışmasını önlemek için her 3 karede bir AI kontrolü)
        if (budget > 0 && candidates.size() >= 2 && (frameCounter % 3 == 0)) {
            for (int i = 0; i < candidates.size() && budget > 0; i++) {
                TrackedPerson a = candidates.get(i);
                if (selected.contains(a)) continue;
                for (int j = 0; j < candidates.size(); j++) {
                    if (i == j) continue;
                    TrackedPerson b = candidates.get(j);
                    if (computeIoU(a.nx, a.ny, a.nw, a.nh, b.nx, b.ny, b.nw, b.nh) > 0.12f) {
                        selected.add(a);
                        budget--;
                        break;
                    }
                }
            }
        }

        // Öncelik 5: Arka plan periyodik çoklu-açı zenginleştirme (sakin karelerde 1 temiz kişi)
        if (selected.isEmpty()) {
            boolean needFastEnrich = (lockedSnap != null && !lockedSnap.lost
                    && (lockedSnap.deepCount < 4 || lockedSnap.faceCount < 3));
            long interval = needFastEnrich ? ENRICH_FAST_INTERVAL_MS : ENRICH_STEADY_INTERVAL_MS;
            if (now - lastEnrichTs >= interval) {
                TrackedPerson target = null;
                if (lockedSnap != null && !lockedSnap.lost) {
                    TrackedPerson lc = findBestOverlapCandidate(candidates, lockedSnap.nx, lockedSnap.ny, lockedSnap.nw, lockedSnap.nh);
                    if (lc != null && !isEdgeClipped(lc.nx, lc.ny, lc.nw, lc.nh)) {
                        target = lc;
                    }
                }
                if (target == null) {
                    // Temiz, en büyük kişiyi seç (dönüşümlü indeks)
                    int startIdx = frameCounter % candidates.size();
                    for (int k = 0; k < candidates.size(); k++) {
                        TrackedPerson c = candidates.get((startIdx + k) % candidates.size());
                        if (!isEdgeClipped(c.nx, c.ny, c.nw, c.nh) && c.score >= 0.36f) {
                            target = c;
                            break;
                        }
                    }
                }
                if (target != null) {
                    selected.add(target);
                    lastEnrichTs = now;
                }
            }
        }

        // Seçilen adaylar için Derin Vücut + Biyometrik Yüz vektörlerini çıkar
        for (TrackedPerson c : selected) {
            NeuralIdentityEngine.NeuralFeatures feat =
                    neuralEngine.extractFeatures(fullBitmap, c.nx, c.ny, c.nw, c.nh, true, true);
            if (feat.bodyEmbedding != null) {
                c.deepBodyEmbedding = feat.bodyEmbedding;
                c.hasDeepAi = true;
            }
            if (feat.faceEmbedding != null) {
                c.faceEmbedding = feat.faceEmbedding;
                c.hasFaceAi = true;
            }
        }
    }

    private TrackedPerson findBestOverlapCandidate(List<TrackedPerson> candidates, float nx, float ny, float nw, float nh) {
        TrackedPerson best = null;
        float bestIoU = 0.25f;
        for (TrackedPerson c : candidates) {
            float iou = computeIoU(c.nx, c.ny, c.nw, c.nh, nx, ny, nw, nh);
            if (iou > bestIoU) {
                bestIoU = iou;
                best = c;
            }
        }
        return best;
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

        float[] extractedBody = best.deepBodyEmbedding;
        float[] extractedFace = best.faceEmbedding;
        List<PersonMemoryStore.PersonProfile> saveSnapshot = null;
        int saveNextId = 1;

        synchronized (detectorLock) {
            Bitmap snap = lastAiBitmap;
            if (!closed && snap != null && !snap.isRecycled()) {
                ensureAiPixelsLocked(snap);
                // allowZoom=true (aiExecutor yolu) ise anında Derin Vücut + Yüz AI vektörlerini çıkar!
                if (allowZoom && (extractedBody == null || extractedFace == null)) {
                    NeuralIdentityEngine.NeuralFeatures feat = neuralEngine.extractFeatures(
                            snap, best.nx, best.ny, best.nw, best.nh,
                            extractedBody == null, extractedFace == null);
                    if (extractedBody == null && feat.bodyEmbedding != null) extractedBody = feat.bodyEmbedding;
                    if (extractedFace == null && feat.faceEmbedding != null) extractedFace = feat.faceEmbedding;
                }
            }
            int[] pxSource = aiPixels;
            if (best.signature == null && aiPixelsValid) {
                best.signature = computeSignature(pxSource, best.nx, best.ny, best.nw, best.nh);
            }

            synchronized (stateLock) {
                long now = System.currentTimeMillis();
                LockedState t = new LockedState();
                t.id = (best.id > 0) ? best.id : nextTrackId++;
                t.name = best.name;
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
                if (t.anchorSig != null) t.gallery.add(t.anchorSig.clone());

                t.deepGallery = new ArrayList<>(MAX_LOCKED_DEEP_GALLERY);
                if (extractedBody != null) t.deepGallery.add(extractedBody.clone());

                t.faceGallery = new ArrayList<>(MAX_LOCKED_FACE_GALLERY);
                if (extractedFace != null) t.faceGallery.add(extractedFace.clone());

                t.template = aiPixelsValid ? extractTemplate(pxSource, best.nx, best.ny, best.nw, best.nh) : null;
                t.lastSeen = now;
                t.lastUpdate = now;
                t.recoverStreak = 0;
                t.isOccluded = false;
                t.lost = false;

                // Kalıcı/oda hafızasından bu kişinin mevcut tüm isim, yüz, derin vücut ve imza örneklerini aktar
                PersonMemoryStore.PersonProfile prof = findProfileByIdLocked(t.id);
                if (prof != null) {
                    if (prof.isNamed()) t.name = prof.name;
                    for (float[] g : prof.sigGallery) {
                        if (t.gallery.size() < MAX_LOCKED_GALLERY) t.gallery.add(g.clone());
                    }
                    for (float[] d : prof.deepBodyGallery) {
                        addDiverseEmbedding(t.deepGallery, d, MAX_LOCKED_DEEP_GALLERY, 0.93f);
                    }
                    for (float[] f : prof.faceGallery) {
                        addDiverseEmbedding(t.faceGallery, f, MAX_LOCKED_FACE_GALLERY, 0.93f);
                    }
                }
                // Yeni çıkarılan vektörleri de profille senkronize et
                updateRoomMemoryLocked(t.id, t.anchorSig, extractedBody, extractedFace, now, false);

                t.needsImmediateNeural = (t.deepGallery.isEmpty() || (neuralEngine.isFaceReady() && t.faceGallery.isEmpty()));
                lockedTarget = t;

                if (outLocked != null) {
                    copyInto(t, outLocked);
                }
                if (memoryDirty) {
                    memoryDirty = false;
                    saveSnapshot = snapshotRoomMemoryLocked();
                    saveNextId = nextTrackId;
                }
            }
        }

        if (saveSnapshot != null) {
            memoryStore.saveDebouncedAsync(saveSnapshot, saveNextId);
        }
        return LockStatus.LOCKED;
    }

    public void unlock() {
        List<PersonMemoryStore.PersonProfile> saveSnap = null;
        int saveNextId = 1;
        synchronized (stateLock) {
            if (lockedTarget != null) {
                long now = System.currentTimeMillis();
                if (lockedTarget.gallery != null) {
                    for (float[] g : lockedTarget.gallery) {
                        updateRoomMemoryLocked(lockedTarget.id, g, null, null, now, false);
                    }
                }
                if (lockedTarget.deepGallery != null) {
                    for (float[] d : lockedTarget.deepGallery) {
                        updateRoomMemoryLocked(lockedTarget.id, null, d, null, now, false);
                    }
                }
                if (lockedTarget.faceGallery != null) {
                    for (float[] f : lockedTarget.faceGallery) {
                        updateRoomMemoryLocked(lockedTarget.id, null, null, f, now, false);
                    }
                }
                if (memoryDirty) {
                    memoryDirty = false;
                    saveSnap = snapshotRoomMemoryLocked();
                    saveNextId = nextTrackId;
                }
            }
            lockedTarget = null;
            pendingRecoveryCandidate = null;
        }
        if (saveSnap != null) {
            memoryStore.saveDebouncedAsync(saveSnap, saveNextId);
        }
    }

    public LockedState getLockedStateSnapshot() {
        synchronized (stateLock) {
            return (lockedTarget != null) ? copyLockedState(lockedTarget) : null;
        }
    }

    // =================== KULLANICI HAFIZA YÖNETİMİ (İSİMLENDİRME / LİSTE / TEMİZLEME) ===================
    public List<PersonMemoryStore.ProfileSummary> getMemorySummaries() {
        synchronized (stateLock) {
            List<PersonMemoryStore.ProfileSummary> list = new ArrayList<>(roomMemory.size());
            for (PersonMemoryStore.PersonProfile p : roomMemory) {
                list.add(new PersonMemoryStore.ProfileSummary(
                        p.id, p.name, p.sigGallery.size(), p.deepBodyGallery.size(), p.faceGallery.size(), p.lastSeen));
            }
            Collections.sort(list, (a, b) -> {
                boolean an = (a.name != null && !a.name.trim().isEmpty());
                boolean bn = (b.name != null && !b.name.trim().isEmpty());
                if (an != bn) return an ? -1 : 1;
                return Long.compare(b.lastSeen, a.lastSeen);
            });
            return list;
        }
    }

    public boolean renamePerson(int id, String rawName) {
        String cleanName = (rawName != null && !rawName.trim().isEmpty()) ? rawName.trim() : null;
        List<PersonMemoryStore.PersonProfile> snap;
        int saveNextId;
        boolean found = false;
        synchronized (stateLock) {
            PersonMemoryStore.PersonProfile p = findProfileByIdLocked(id);
            if (p != null) {
                p.name = cleanName;
                p.lastSeen = System.currentTimeMillis();
                found = true;
            }
            if (lockedTarget != null && lockedTarget.id == id) {
                lockedTarget.name = cleanName;
                found = true;
            }
            for (TrackedPerson tp : prevTracks) {
                if (tp.id == id) tp.name = cleanName;
            }
            snap = snapshotRoomMemoryLocked();
            saveNextId = nextTrackId;
            memoryDirty = false;
        }
        if (found) {
            memoryStore.saveSync(snap, saveNextId);
        }
        return found;
    }

    public boolean deletePersonFromMemory(int id) {
        List<PersonMemoryStore.PersonProfile> snap = null;
        int saveNextId = 1;
        boolean removed = false;
        synchronized (stateLock) {
            Iterator<PersonMemoryStore.PersonProfile> it = roomMemory.iterator();
            while (it.hasNext()) {
                if (it.next().id == id) {
                    it.remove();
                    removed = true;
                    break;
                }
            }
            if (lockedTarget != null && lockedTarget.id == id) {
                lockedTarget.name = null;
            }
            for (TrackedPerson tp : prevTracks) {
                if (tp.id == id) tp.name = null;
            }
            if (removed) {
                snap = snapshotRoomMemoryLocked();
                saveNextId = nextTrackId;
                memoryDirty = false;
            }
        }
        if (snap != null) {
            memoryStore.saveSync(snap, saveNextId);
        }
        return removed;
    }

    public void clearAllMemory() {
        synchronized (stateLock) {
            roomMemory.clear();
            prevTracks.clear();
            nextTrackId = 1;
            if (lockedTarget != null) {
                lockedTarget.name = null;
            }
            memoryDirty = false;
        }
        memoryStore.clearFile();
    }

    private List<PersonMemoryStore.PersonProfile> snapshotRoomMemoryLocked() {
        List<PersonMemoryStore.PersonProfile> copy = new ArrayList<>(roomMemory.size());
        for (PersonMemoryStore.PersonProfile p : roomMemory) {
            copy.add(p.deepCopy());
        }
        return copy;
    }

    private PersonMemoryStore.PersonProfile findProfileByIdLocked(int id) {
        for (PersonMemoryStore.PersonProfile p : roomMemory) {
            if (p.id == id) return p;
        }
        return null;
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
            cropMp.close(); // cropMp.close() geçici cropped bitmap'i güvenli şekilde recycle eder.

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
     * Kişilere Konum + 129-D İmza + 1024-D Derin Vücut AI + 192-D Biyometrik Yüz AI + Kalıcı Disk Hafızası ile kararlı ID ve isim atar.
     */
    private void assignStableIdsWithMemoryLocked(List<TrackedPerson> current, long now) {
        Iterator<PersonMemoryStore.PersonProfile> it = roomMemory.iterator();
        while (it.hasNext()) {
            PersonMemoryStore.PersonProfile m = it.next();
            boolean isLocked = (lockedTarget != null && lockedTarget.id == m.id);
            if (!isLocked && PersonMemoryStore.isExpired(m, now)) {
                it.remove();
                memoryDirty = true;
            }
        }

        List<TrackedPerson> unmatchedPrev = new ArrayList<>(prevTracks);
        List<TrackedPerson> needMemoryMatch = new ArrayList<>();
        List<Integer> usedIds = new ArrayList<>();

        if (lockedTarget != null) {
            if (!lockedTarget.lost) {
                for (TrackedPerson c : current) {
                    if (computeIoU(c.nx, c.ny, c.nw, c.nh, lockedTarget.nx, lockedTarget.ny, lockedTarget.nw, lockedTarget.nh) > 0.55f) {
                        c.id = lockedTarget.id;
                        c.name = lockedTarget.name;
                        c.hasDeepAi = c.hasDeepAi || (lockedTarget.deepGallery != null && !lockedTarget.deepGallery.isEmpty());
                        c.hasFaceAi = c.hasFaceAi || (lockedTarget.faceGallery != null && !lockedTarget.faceGallery.isEmpty());
                        usedIds.add(c.id);
                        break;
                    }
                }
            } else if (pendingRecoveryCandidate != null) {
                pendingRecoveryCandidate.id = lockedTarget.id;
                pendingRecoveryCandidate.name = lockedTarget.name;
                usedIds.add(lockedTarget.id);
            }
        }

        // 1. Aşama: Önceki karedeki aktif kişilerle KONUM + HİBRİT AI eşleştirmesi
        for (TrackedPerson cur : current) {
            if (cur.id > 0) continue;

            TrackedPerson bestPrev = null;
            float bestScore = 0.42f;
            boolean legsCropped = (cur.ny + cur.nh > 0.94f);
            for (TrackedPerson prev : unmatchedPrev) {
                if (usedIds.contains(prev.id)) continue;
                float iou = computeIoU(cur.nx, cur.ny, cur.nw, cur.nh, prev.nx, prev.ny, prev.nw, prev.nh);
                float dist = (float) Math.hypot(cur.centerX() - prev.centerX(), cur.centerY() - prev.centerY());
                float posScore = iou * 1.4f + Math.max(0f, 0.35f - dist) * 2.0f;

                PersonMemoryStore.PersonProfile prof = findProfileByIdLocked(prev.id);
                HybridMatch hm = null;
                float appScore = 0.65f;
                float torso = 0.65f;
                if (prof != null) {
                    hm = matchProfileHybrid(prof.sigGallery, prof.deepBodyGallery, prof.faceGallery, cur, legsCropped);
                    if (hm.faceVetoed) continue; // İki farklı yüz birbirinin ID'sini ASLA çalamaz!
                    appScore = hm.combined;
                    torso = hm.torso;
                } else if (cur.signature != null && prev.signature != null) {
                    appScore = reidSim(cur.signature, prev.signature, legsCropped);
                    torso = zoneSim(cur.signature, prev.signature, 1);
                }

                if ((prevTracks.size() > 1 || current.size() > 1) && !(hm != null && hm.faceMatched)
                        && (appScore < 0.56f || torso < 0.48f)) {
                    continue;
                }
                float total = posScore + appScore * 2.7f + torso * 1.0f;
                if (hm != null && hm.faceMatched) total += 1.5f;
                if (total > bestScore) {
                    bestScore = total;
                    bestPrev = prev;
                }
            }

            if (bestPrev != null) {
                cur.id = bestPrev.id;
                cur.name = bestPrev.name;
                usedIds.add(cur.id);
                unmatchedPrev.remove(bestPrev);
            } else {
                needMemoryMatch.add(cur);
            }
        }

        // 2. Aşama: Kadraja yeni/tekrar giren kişileri Kalıcı Disk + Derin Vücut + Biyometrik Yüz Hafızasında ara!
        for (TrackedPerson cur : needMemoryMatch) {
            PersonMemoryStore.PersonProfile bestMem = null;
            HybridMatch bestMatch = null;
            float bestSim = 0f;
            boolean legsCropped = (cur.ny + cur.nh > 0.94f);

            for (PersonMemoryStore.PersonProfile mem : roomMemory) {
                if (usedIds.contains(mem.id)) continue;
                if (lockedTarget != null && lockedTarget.id == mem.id) continue;

                HybridMatch hm = matchProfileHybrid(mem.sigGallery, mem.deepBodyGallery, mem.faceGallery, cur, legsCropped);
                if (hm.faceVetoed) continue;

                boolean accepted;
                if (hm.faceMatched) {
                    accepted = true; // Biyometrik yüz eşleşmesi kıyafet değişiminde bile tanır!
                } else if (hm.deepUsed) {
                    accepted = (hm.combined >= 0.68f && hm.torso >= 0.50f && hm.deepSim >= 0.54f);
                } else {
                    accepted = (hm.combined >= 0.70f && hm.torso >= 0.60f);
                }

                if (accepted && hm.combined > bestSim) {
                    bestSim = hm.combined;
                    bestMem = mem;
                    bestMatch = hm;
                }
            }

            if (bestMem != null) {
                cur.id = bestMem.id;
                cur.name = bestMem.name;
                usedIds.add(cur.id);
                // Yüz tanımayla kıyafet değişimi yakalandıysa yeni kıyafetini profile otomatik ekle!
                if (bestMatch != null && bestMatch.faceMatched) {
                    updateRoomMemoryLocked(cur.id, cur.signature, cur.deepBodyEmbedding, cur.faceEmbedding, now, true);
                }
            } else {
                while (usedIds.contains(nextTrackId)
                        || (lockedTarget != null && lockedTarget.id == nextTrackId)
                        || findProfileByIdLocked(nextTrackId) != null) {
                    nextTrackId = (nextTrackId >= 999) ? 1 : nextTrackId + 1;
                }
                cur.id = nextTrackId++;
                if (nextTrackId > 999) nextTrackId = 1;
                usedIds.add(cur.id);
            }
        }

        // 3. Aşama: Hafızayı temiz, kenardan kesilmemiş çoklu açı + Derin Vücut + Yüz örnekleriyle besle
        for (TrackedPerson cur : current) {
            cur.lastSeen = now;
            PersonMemoryStore.PersonProfile existing = findProfileByIdLocked(cur.id);
            if (existing != null) {
                if (existing.isNamed()) cur.name = existing.name;
                cur.hasDeepAi = cur.hasDeepAi || !existing.deepBodyGallery.isEmpty();
                cur.hasFaceAi = cur.hasFaceAi || !existing.faceGallery.isEmpty();
            }

            if (cur == pendingRecoveryCandidate) continue;
            if (cur.signature == null && cur.deepBodyEmbedding == null && cur.faceEmbedding == null) continue;
            if (isEdgeClipped(cur.nx, cur.ny, cur.nw, cur.nh)) continue;

            boolean crowded = false;
            for (TrackedPerson other : current) {
                if (other != cur && computeIoU(cur.nx, cur.ny, cur.nw, cur.nh, other.nx, other.ny, other.nw, other.nh) > 0.14f) {
                    crowded = true;
                    break;
                }
            }
            if (!crowded) {
                updateRoomMemoryLocked(cur.id, cur.signature, cur.deepBodyEmbedding, cur.faceEmbedding, now, false);
                PersonMemoryStore.PersonProfile updated = findProfileByIdLocked(cur.id);
                if (updated != null) {
                    cur.hasDeepAi = !updated.deepBodyGallery.isEmpty();
                    cur.hasFaceAi = !updated.faceGallery.isEmpty();
                }
            }
        }

        prevTracks.clear();
        prevTracks.addAll(current);
    }

    private void updateRoomMemoryLocked(
            int id, float[] sig, float[] deepBody, float[] face, long now, boolean forceClothingAdapt) {
        PersonMemoryStore.PersonProfile target = findProfileByIdLocked(id);
        if (target == null) {
            if (roomMemory.size() >= PersonMemoryStore.MAX_PROFILES) {
                // İsimlendirilmiş profilleri koru; en eski isimsiz profili çıkar
                PersonMemoryStore.PersonProfile evict = null;
                for (PersonMemoryStore.PersonProfile m : roomMemory) {
                    boolean isLocked = (lockedTarget != null && lockedTarget.id == m.id);
                    if (isLocked || m.isNamed()) continue;
                    if (evict == null || m.lastSeen < evict.lastSeen) evict = m;
                }
                if (evict == null) {
                    for (PersonMemoryStore.PersonProfile m : roomMemory) {
                        boolean isLocked = (lockedTarget != null && lockedTarget.id == m.id);
                        if (isLocked) continue;
                        if (evict == null || m.lastSeen < evict.lastSeen) evict = m;
                    }
                }
                if (evict != null) roomMemory.remove(evict);
            }
            target = new PersonMemoryStore.PersonProfile();
            target.id = id;
            target.firstSeen = now;
            target.lastSeen = now;
            if (sig != null) target.sigGallery.add(sig.clone());
            if (deepBody != null) target.deepBodyGallery.add(deepBody.clone());
            if (face != null) target.faceGallery.add(face.clone());
            roomMemory.add(target);
            memoryDirty = true;
            return;
        }

        boolean changed = false;
        if (sig != null) {
            if (target.sigGallery.isEmpty()) {
                target.sigGallery.add(sig.clone());
                changed = true;
            } else {
                float bestSim = 0f;
                for (float[] g : target.sigGallery) {
                    bestSim = Math.max(bestSim, reidSim(g, sig, false));
                }
                if (forceClothingAdapt || bestSim < 0.89f) {
                    if (target.sigGallery.size() >= PersonMemoryStore.MAX_SIG_PER_PROFILE) {
                        target.sigGallery.remove(forceClothingAdapt ? 0 : 1);
                    }
                    target.sigGallery.add(sig.clone());
                    changed = true;
                }
            }
        }
        if (deepBody != null) {
            if (addDiverseEmbedding(target.deepBodyGallery, deepBody, PersonMemoryStore.MAX_DEEP_PER_PROFILE,
                    forceClothingAdapt ? 0.96f : 0.90f)) {
                changed = true;
            }
        }
        if (face != null) {
            if (addDiverseEmbedding(target.faceGallery, face, PersonMemoryStore.MAX_FACE_PER_PROFILE, 0.91f)) {
                changed = true;
            }
        }
        target.lastSeen = now;
        if (changed) memoryDirty = true;
    }

    private static boolean addDiverseEmbedding(List<float[]> gallery, float[] vec, int maxLen, float dupThreshold) {
        if (gallery == null || vec == null) return false;
        if (gallery.isEmpty()) {
            gallery.add(vec.clone());
            return true;
        }
        float bestSim = -1f;
        for (float[] existing : gallery) {
            if (existing.length == vec.length) {
                bestSim = Math.max(bestSim, NeuralIdentityEngine.cosineSim(existing, vec));
            }
        }
        if (bestSim < dupThreshold) {
            if (gallery.size() >= maxLen) {
                // Slot 0 = ilk temiz anchor; en eski ikincil örneği (slot 1) döndür
                gallery.remove(gallery.size() > 1 ? 1 : 0);
            }
            gallery.add(vec.clone());
            return true;
        }
        return false;
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
        float bestCombinedMatched = 0f;
        float bestAnchorSimMatched = 0f;
        boolean bestFaceMatched = false;

        for (TrackedPerson d : detections) {
            if (d.signature == null && d.deepBodyEmbedding == null && d.faceEmbedding == null) continue;

            float iou = computeIoU(predX, predY, predW, predH, d.nx, d.ny, d.nw, d.nh);
            float pcx = predX + predW * 0.5f, pcy = predY + predH * 0.5f;
            float dist = (float) Math.hypot(pcx - d.centerX(), pcy - d.centerY());
            float dArea = d.area();
            float sizeRatio = Math.min(dArea / Math.max(1e-6f, t.baseArea), t.baseArea / Math.max(1e-6f, dArea));
            float dAspect = d.nw / Math.max(0.01f, d.nh);
            float aspectSim = Math.min(dAspect / Math.max(0.01f, t.aspectRatio), t.aspectRatio / Math.max(0.01f, dAspect));
            boolean legsCropped = (d.ny + d.nh > 0.94f);

            // 3-Katmanlı Hibrit Eşleştirme: Biyometrik Yüz + Derin Vücut AI + 129-D Çoklu-İpucu İmza
            HybridMatch hm = matchProfileHybrid(t.gallery, t.deepGallery, t.faceGallery, d, legsCropped);
            if (hm.faceVetoed) continue; // Farklı yüzlü kişi kilitli hedefi ASLA çalamaz!

            float anchorSim = (t.anchorSig != null && d.signature != null)
                    ? reidSim(t.anchorSig, d.signature, legsCropped)
                    : hm.combined;
            float anchorTorso = (t.anchorSig != null && d.signature != null)
                    ? zoneSim(t.anchorSig, d.signature, 1)
                    : hm.torso;

            float score;
            if (!t.lost) {
                if (!hm.faceMatched) {
                    if (hm.combined < 0.53f || Math.max(hm.torso, anchorTorso) < 0.46f) continue;
                    if (sizeRatio < 0.16f) continue;
                }
                score = hm.combined * 4.2f + hm.torso * 1.4f + iou * 1.5f
                        + aspectSim * 0.45f - (dist / Math.max(0.05f, diag)) * 0.45f + sizeRatio * 0.35f;
                if (hm.faceMatched) score += 2.2f;
            } else {
                // Kayıpken geri yakalama: Biyometrik yüz varsa kıyafet değişiminde bile yakala!
                if (hm.faceMatched) {
                    score = hm.combined * 5.8f + sizeRatio * 0.3f + 2.5f;
                } else if (hm.deepUsed) {
                    if (hm.combined < 0.65f || hm.torso < 0.48f || hm.deepSim < 0.50f) continue;
                    if (sizeRatio < 0.08f || (!legsCropped && aspectSim < 0.28f)) continue;
                    score = hm.combined * 4.8f + hm.deepSim * 1.8f + hm.torso * 1.2f + anchorSim * 0.8f + sizeRatio * 0.3f;
                } else {
                    if (hm.combined < 0.68f || hm.torso < 0.58f) continue;
                    if (sizeRatio < 0.08f || (!legsCropped && aspectSim < 0.30f)) continue;
                    score = hm.combined * 4.5f + hm.torso * 1.8f + anchorSim * 1.0f + sizeRatio * 0.3f;
                }
            }

            if (score > bestScore) {
                bestScore = score;
                best = d;
                bestIoU = iou;
                bestCombinedMatched = hm.combined;
                bestAnchorSimMatched = anchorSim;
                bestFaceMatched = hm.faceMatched;
            }
        }

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

        if (t.lost) {
            if (best != null) {
                pendingRecoveryCandidate = best;
                t.recoverStreak++;
                if (bestFaceMatched || t.recoverStreak >= 2 || bestCombinedMatched >= 0.76f) {
                    t.lost = false;
                    t.recoverStreak = 0;
                    t.vx = 0f;
                    t.vy = 0f;
                    applyMatchedPosition(t, best, now, conflicted, bestIoU, bestCombinedMatched, bestAnchorSimMatched, bestFaceMatched);
                    out.targetJustRecovered = true;
                }
            } else {
                t.recoverStreak = 0;
            }
            return;
        }

        if (best != null) {
            applyMatchedPosition(t, best, now, conflicted, bestIoU, bestCombinedMatched, bestAnchorSimMatched, bestFaceMatched);
            return;
        }

        // AI bu karede kaçırdıysa ama kayıp süresi dolmadıysa sıfır-GC NCC piksel şablonu + hız ile tut
        boolean tplFound = (now - t.lastSeen < 650 && t.template != null)
                && searchTemplateZeroAlloc(aiPixels, t.template, predX, predY, predW, predH, 0.70f, aiTplScratch, aiHitScratch);
        if (tplFound) {
            t.nx = Math.max(0f, Math.min(0.98f, aiHitScratch[0]));
            t.ny = Math.max(0f, Math.min(0.98f, aiHitScratch[1]));
            t.lastUpdate = now;
        } else if (now - t.lastSeen < 450) {
            t.nx = Math.max(-0.12f, Math.min(0.95f, predX));
            t.ny = Math.max(-0.12f, Math.min(0.95f, predY));
            t.lastUpdate = now;
        } else if (now - t.lastSeen >= TARGET_LOST_TIMEOUT_MS) {
            t.lost = true;
            t.recoverStreak = 0;
            out.targetJustLost = true;
        }
    }

    private void applyMatchedPosition(
            LockedState t, TrackedPerson best, long now,
            boolean conflicted, float bestIoU, float maxCombinedSim, float anchorSim, boolean faceMatched) {
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

        // Kilitli hedef temiz görünüyorsa (veya biyometrik yüz doğrulandıysa) tüm hafıza bankalarını zenginleştir
        boolean cleanView = !conflicted && best.score >= 0.34f
                && !isEdgeClipped(best.nx, best.ny, best.nw, best.nh)
                && (bestIoU > 0.25f || t.lost || faceMatched);

        if (cleanView) {
            if (best.signature != null && (faceMatched || anchorSim >= 0.56f)) {
                if (t.gallery == null) t.gallery = new ArrayList<>(MAX_LOCKED_GALLERY);
                if (t.gallery.isEmpty() && t.anchorSig != null) t.gallery.add(t.anchorSig.clone());
                // Yüz eşleşti ama kıyafet değiştiyse yeni kıyafet görünümünü zorla öğren!
                boolean clothingChanged = faceMatched && anchorSim < 0.62f;
                if (clothingChanged || maxCombinedSim < 0.89f) {
                    if (t.gallery.size() >= MAX_LOCKED_GALLERY) {
                        t.gallery.remove(1);
                    }
                    t.gallery.add(best.signature.clone());
                    if (clothingChanged) {
                        t.anchorSig = best.signature.clone();
                    }
                }
            }
            if (best.deepBodyEmbedding != null) {
                if (t.deepGallery == null) t.deepGallery = new ArrayList<>(MAX_LOCKED_DEEP_GALLERY);
                addDiverseEmbedding(t.deepGallery, best.deepBodyEmbedding, MAX_LOCKED_DEEP_GALLERY,
                        faceMatched ? 0.95f : 0.90f);
                t.needsImmediateNeural = false;
            }
            if (best.faceEmbedding != null) {
                if (t.faceGallery == null) t.faceGallery = new ArrayList<>(MAX_LOCKED_FACE_GALLERY);
                addDiverseEmbedding(t.faceGallery, best.faceEmbedding, MAX_LOCKED_FACE_GALLERY, 0.91f);
                t.needsImmediateNeural = false;
            }
            // Kalıcı profili de anlık senkronize et
            updateRoomMemoryLocked(t.id, best.signature, best.deepBodyEmbedding, best.faceEmbedding, now, faceMatched);

            float[] newTpl = extractTemplate(aiPixels, t.nx, t.ny, t.nw, t.nh);
            if (newTpl != null) t.template = newTpl;
        }
    }

    private boolean isEdgeClipped(float nx, float ny, float nw, float nh) {
        return nx < 0.02f || (nx + nw) > 0.98f || ny < 0.02f || (ny + nh) > 0.98f;
    }

    // =================== 3-KATMANLI HİBRİT EŞLEŞTİRME (YÜZ + DERİN VÜCUT + 129-D) ===================
    private static class GalleryMatch {
        float combined;
        float torso;
    }

    private static class HybridMatch {
        float combined;
        float torso;
        float deepSim = -1f;
        float faceSim = -1f;
        boolean deepUsed;
        boolean faceMatched;
        boolean faceVetoed;
    }

    private HybridMatch matchProfileHybrid(
            List<float[]> sigGallery,
            List<float[]> deepGallery,
            List<float[]> faceGallery,
            TrackedPerson cand,
            boolean legsCropped) {
        HybridMatch hm = new HybridMatch();
        GalleryMatch sigGm = matchGallery(sigGallery, cand.signature, legsCropped);
        float deepSim = matchEmbeddingGallery(deepGallery, cand.deepBodyEmbedding);
        float faceSim = matchEmbeddingGallery(faceGallery, cand.faceEmbedding);

        hm.deepSim = deepSim;
        hm.faceSim = faceSim;
        hm.torso = sigGm.torso;

        // 1. Vücut/Kıyafet Skoru (Derin MobileNetV3 + 129-D Çok-İpuculu Füzyon)
        float bodyScore = sigGm.combined;
        if (deepSim >= 0f) {
            hm.deepUsed = true;
            if (sigGm.combined > 0f) {
                bodyScore = deepSim * 0.58f + sigGm.combined * 0.42f;
                if (deepSim >= 0.74f) {
                    // Derin yapay zeka çok eminse sert ışık kaymasının HSV gövde skorunu bozmasını engelle
                    hm.torso = Math.max(hm.torso, deepSim * 0.80f);
                } else if (deepSim < 0.46f) {
                    // Derin yapay zeka açıkça farklı vücut/kıyafet diyorsa düz renk benzerliğini bastır
                    bodyScore = Math.min(bodyScore, deepSim * 0.78f + sigGm.combined * 0.22f);
                }
            } else {
                bodyScore = deepSim;
                hm.torso = deepSim * 0.85f;
            }
        }

        // 2. Biyometrik Yüz Entegrasyonu (Kıyafet-Bağımsız Tanıma + Farklı Yüz Vetosu)
        if (faceSim >= 0f) {
            if (faceSim >= 0.64f) {
                hm.faceMatched = true;
                float faceBoosted = Math.min(0.99f, 0.22f + faceSim * 0.82f);
                hm.combined = Math.max(bodyScore, faceBoosted);
                hm.torso = Math.max(hm.torso, 0.72f);
            } else if (faceSim >= 0.52f) {
                hm.combined = faceSim * 0.55f + bodyScore * 0.45f;
            } else if (faceSim < 0.36f) {
                // İki net yüz birbirinden tamamen farklı -> güçlü veto!
                hm.faceVetoed = true;
                hm.combined = bodyScore * 0.42f;
                hm.torso = hm.torso * 0.45f;
            } else {
                hm.combined = faceSim * 0.30f + bodyScore * 0.70f;
            }
        } else {
            hm.combined = bodyScore;
        }
        return hm;
    }

    private static float matchEmbeddingGallery(List<float[]> gallery, float[] cand) {
        if (gallery == null || gallery.isEmpty() || cand == null) return -1f;
        float best1 = -1f, best2 = -1f;
        int validCount = 0;
        for (float[] g : gallery) {
            if (g == null || g.length != cand.length) continue;
            float s = NeuralIdentityEngine.cosineSim(g, cand);
            validCount++;
            if (s > best1) {
                best2 = best1;
                best1 = s;
            } else if (s > best2) {
                best2 = s;
            }
        }
        if (validCount == 0) return -1f;
        float score = (validCount >= 2 && best2 > 0f)
                ? Math.max(best1 * 0.92f, best1 * 0.68f + best2 * 0.32f)
                : best1;
        return Math.max(0f, Math.min(1f, score));
    }

    private GalleryMatch matchGallery(List<float[]> gallery, float[] candSig, boolean legsCropped) {
        GalleryMatch gm = new GalleryMatch();
        if (gallery == null || gallery.isEmpty() || candSig == null) return gm;

        float best1 = 0f, best2 = 0f;
        float bestTorso = 0f;
        for (float[] g : gallery) {
            if (g == null || g.length != SIG_LEN) continue;
            float s = reidSim(g, candSig, legsCropped);
            if (s > best1) {
                best2 = best1;
                best1 = s;
                bestTorso = zoneSim(g, candSig, 1);
            } else if (s > best2) {
                best2 = s;
            }
        }
        gm.combined = (gallery.size() > 1 && best2 > 0f) ? (best1 * 0.65f + best2 * 0.35f) : best1;
        gm.torso = bestTorso;
        return gm;
    }

    // =================== YUMUŞAK DAĞILIMLI 3-BÖLGELİ 129-D RE-ID + CHROMA + YAPISAL DESEN ===================
    private float[] computeSignature(int[] pixels, float nx, float ny, float nw, float nh) {
        float[] sig = new float[SIG_LEN];
        return computeSignatureInto(pixels, nx, ny, nw, nh, sig) ? sig : null;
    }

    private boolean computeSignatureInto(int[] pixels, float nx, float ny, float nw, float nh, float[] sig) {
        int x0 = Math.max(0, (int) Math.floor((nx + nw * 0.16f) * ANA_W));
        int x1 = Math.min(ANA_W, (int) Math.ceil((nx + nw * 0.84f) * ANA_W));
        int yHead0 = Math.max(0, (int) Math.floor((ny + nh * 0.04f) * ANA_H));
        int yHead1 = Math.min(ANA_H, Math.max(yHead0 + 1, (int) Math.floor((ny + nh * 0.25f) * ANA_H)));
        int yTorso1 = Math.min(ANA_H, Math.max(yHead1 + 1, (int) Math.floor((ny + nh * 0.57f) * ANA_H)));
        int yLegs1 = Math.min(ANA_H, Math.max(yTorso1 + 1, (int) Math.ceil((ny + nh * 0.90f) * ANA_H)));
        if (x1 - x0 < 2 || yLegs1 - yHead0 < 6) return false;

        for (int i = 0; i < SIG_LEN; i++) sig[i] = 0f;

        float centerX = (x0 + x1) * 0.5f;
        float halfW = Math.max(1f, (x1 - x0) * 0.5f);
        accumulateZoneHistAndChroma(pixels, x0, yHead0, x1, yHead1, centerX, halfW, sig, 0, HIST_TOTAL);
        accumulateZoneHistAndChroma(pixels, x0, yHead1, x1, yTorso1, centerX, halfW, sig, ZONE_LEN, HIST_TOTAL + 3);
        accumulateZoneHistAndChroma(pixels, x0, yTorso1, x1, yLegs1, centerX, halfW, sig, ZONE_LEN * 2, HIST_TOTAL + 6);
        extractStructuralPatch(pixels, nx + nw * 0.18f, ny + nh * 0.06f, nw * 0.64f, nh * 0.82f, sig, HIST_TOTAL + CHROMA_LEN);
        return true;
    }

    private void accumulateZoneHistAndChroma(
            int[] pixels, int x0, int y0, int x1, int y1,
            float cx, float halfW, float[] out, int histOffset, int chromaOffset) {
        float wSum = 0f;
        float rgAcc = 0f, ybAcc = 0f, lumAcc = 0f;
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

                float ndx = (px + 0.5f - cx) / halfW;
                float weight = Math.max(0.25f, 1.0f - 0.65f * ndx * ndx);

                float denom = r + g + b + 32f;
                rgAcc += ((r - g) / denom) * weight;
                ybAcc += ((r + g - 2f * b) / (2f * denom)) * weight;
                lumAcc += ((0.299f * r + 0.587f * g + 0.114f * b) / 255f) * weight;

                if (s < 0.16f || v < 0.12f) {
                    int gb = Math.min(GRAY_BINS - 1, (int) Math.floor(v * GRAY_BINS));
                    int gbL = Math.max(0, gb - 1);
                    int gbR = Math.min(GRAY_BINS - 1, gb + 1);
                    out[histOffset + HUE_BINS + gb] += weight * 0.60f;
                    out[histOffset + HUE_BINS + gbL] += weight * 0.20f;
                    out[histOffset + HUE_BINS + gbR] += weight * 0.20f;
                } else {
                    float d = max - min;
                    float hue;
                    if (max == r) hue = ((g - b) / d) % 6f;
                    else if (max == g) hue = (b - r) / d + 2f;
                    else hue = (r - g) / d + 4f;
                    hue = (hue * 60f + 360f) % 360f;
                    int hb = ((int) Math.floor(hue / 360f * HUE_BINS)) % HUE_BINS;
                    int hbL = (hb + HUE_BINS - 1) % HUE_BINS;
                    int hbR = (hb + 1) % HUE_BINS;
                    out[histOffset + hb] += weight * 0.58f;
                    out[histOffset + hbL] += weight * 0.21f;
                    out[histOffset + hbR] += weight * 0.21f;
                }
                wSum += weight;
            }
        }
        if (wSum > 0f) {
            for (int i = 0; i < ZONE_LEN; i++) {
                out[histOffset + i] /= wSum;
            }
            out[chromaOffset] = rgAcc / wSum;
            out[chromaOffset + 1] = ybAcc / wSum;
            out[chromaOffset + 2] = lumAcc / wSum;
        }
    }

    private void extractStructuralPatch(int[] pixels, float nx, float ny, float nw, float nh, float[] out, int offset) {
        if (nw <= 0.005f || nh <= 0.01f) return;
        float sum = 0f;
        for (int py = 0; py < PATCH_H; py++) {
            int yIdx = Math.min(ANA_H - 1, Math.max(0, (int) Math.floor((ny + (py + 0.5f) / PATCH_H * nh) * ANA_H)));
            int row = yIdx * ANA_W;
            for (int px = 0; px < PATCH_W; px++) {
                int xIdx = Math.min(ANA_W - 1, Math.max(0, (int) Math.floor((nx + (px + 0.5f) / PATCH_W * nw) * ANA_W)));
                int argb = pixels[row + xIdx];
                float lum = 0.299f * ((argb >> 16) & 0xFF) + 0.587f * ((argb >> 8) & 0xFF) + 0.114f * (argb & 0xFF);
                out[offset + py * PATCH_W + px] = lum;
                sum += lum;
            }
        }
        float mean = sum / PATCH_LEN;
        float normSq = 0f;
        for (int i = 0; i < PATCH_LEN; i++) {
            out[offset + i] -= mean;
            normSq += out[offset + i] * out[offset + i];
        }
        float norm = (float) Math.sqrt(normSq);
        if (norm < 1e-3f) norm = 1f;
        for (int i = 0; i < PATCH_LEN; i++) {
            out[offset + i] /= norm;
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

    private float reidSim(float[] a, float[] b, boolean legsCropped) {
        float head = zoneSim(a, b, 0);
        float torso = zoneSim(a, b, 1);
        float colorSim;
        if (legsCropped) {
            colorSim = head * 0.30f + torso * 0.70f;
        } else {
            float legs = zoneSim(a, b, 2);
            colorSim = head * 0.20f + torso * 0.52f + legs * 0.28f;
        }

        float rg1A = a[HIST_TOTAL + 3], yb1A = a[HIST_TOTAL + 4];
        float rg1B = b[HIST_TOTAL + 3], yb1B = b[HIST_TOTAL + 4];
        float torsoChromaDist = Math.abs(rg1A - rg1B) + Math.abs(yb1A - yb1B);
        float torsoChromaSim = (float) Math.exp(-3.4f * torsoChromaDist);

        float contrastSim;
        if (legsCropped) {
            float dHeadA = a[HIST_TOTAL + 2] - a[HIST_TOTAL + 5];
            float dHeadB = b[HIST_TOTAL + 2] - b[HIST_TOTAL + 5];
            contrastSim = (float) Math.exp(-2.8f * Math.abs(dHeadA - dHeadB));
        } else {
            float dLegA = a[HIST_TOTAL + 5] - a[HIST_TOTAL + 8];
            float dLegB = b[HIST_TOTAL + 5] - b[HIST_TOTAL + 8];
            contrastSim = (float) Math.exp(-2.8f * Math.abs(dLegA - dLegB));
        }
        float chromaSim = torsoChromaSim * 0.65f + contrastSim * 0.35f;

        int pOff = HIST_TOTAL + CHROMA_LEN;
        float dot = 0f;
        for (int i = 0; i < PATCH_LEN; i++) {
            dot += a[pOff + i] * b[pOff + i];
        }
        float patchSim = Math.max(0f, Math.min(1f, 0.25f + 0.75f * dot));

        return colorSim * 0.52f + chromaSim * 0.26f + patchSim * 0.22f;
    }

    // =================== SIFIR-TAHSİSLİ 30 FPS OPTİK ŞABLON TAKİBİ ===================
    private float[] extractTemplate(int[] pixels, float nx, float ny, float nw, float nh) {
        float[] tpl = new float[TPL_W * TPL_H];
        return extractTemplateInto(pixels, nx, ny, nw, nh, tpl) ? tpl : null;
    }

    private boolean extractTemplateInto(int[] pixels, float nx, float ny, float nw, float nh, float[] tpl) {
        if (nw <= 0.004f || nh <= 0.008f) return false;
        float sum = 0f;
        for (int ty = 0; ty < TPL_H; ty++) {
            int py = Math.min(ANA_H - 1, Math.max(0, (int) Math.floor((ny + (ty + 0.5f) / TPL_H * nh) * ANA_H)));
            int row = py * ANA_W;
            int outRow = ty * TPL_W;
            for (int tx = 0; tx < TPL_W; tx++) {
                int px = Math.min(ANA_W - 1, Math.max(0, (int) Math.floor((nx + (tx + 0.5f) / TPL_W * nw) * ANA_W)));
                int argb = pixels[row + px];
                float lum = 0.299f * ((argb >> 16) & 0xFF) + 0.587f * ((argb >> 8) & 0xFF) + 0.114f * (argb & 0xFF);
                tpl[outRow + tx] = lum;
                sum += lum;
            }
        }
        float mean = sum / tpl.length;
        float normSq = 0f;
        for (int i = 0; i < tpl.length; i++) {
            float v = tpl[i] - mean;
            tpl[i] = v;
            normSq += v * v;
        }
        float norm = (float) Math.sqrt(normSq);
        if (norm < 1e-3f) norm = 1f;
        float invNorm = 1f / norm;
        for (int i = 0; i < tpl.length; i++) {
            tpl[i] *= invNorm;
        }
        return true;
    }

    private boolean searchTemplateZeroAlloc(
            int[] pixels, float[] tpl, float predX, float predY, float w, float h,
            float minCorr, float[] scratch, float[] outXY) {
        float cx = predX + w * 0.5f;
        float cy = predY + h * 0.5f;
        float coarseStepX = Math.max(0.012f, w * 0.22f);
        float coarseStepY = Math.max(0.012f, h * 0.22f);

        float bestScore = -1f;
        float bestNx = predX, bestNy = predY;
        boolean found = false;

        // 1. Kaba 3x3 arama
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                float candX = cx + dx * coarseStepX - w * 0.5f;
                float candY = cy + dy * coarseStepY - h * 0.5f;
                if (candX < -0.05f || candY < -0.05f || candX + w > 1.05f || candY + h > 1.05f) continue;
                if (!extractTemplateInto(pixels, candX, candY, w, h, scratch)) continue;
                float dot = 0f;
                for (int i = 0; i < tpl.length; i++) dot += tpl[i] * scratch[i];
                if (dot > bestScore) {
                    bestScore = dot;
                    bestNx = candX;
                    bestNy = candY;
                    found = true;
                }
            }
        }
        if (!found) return false;

        // 2. İnce 3x3 iyileştirme (toplam maks 18 değerlendirme, 0 bellek tahsisi)
        float fineStepX = coarseStepX * 0.45f;
        float fineStepY = coarseStepY * 0.45f;
        float refX = bestNx;
        float refY = bestNy;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                float candX = refX + dx * fineStepX;
                float candY = refY + dy * fineStepY;
                if (candX < -0.05f || candY < -0.05f || candX + w > 1.05f || candY + h > 1.05f) continue;
                if (!extractTemplateInto(pixels, candX, candY, w, h, scratch)) continue;
                float dot = 0f;
                for (int i = 0; i < tpl.length; i++) dot += tpl[i] * scratch[i];
                if (dot > bestScore) {
                    bestScore = dot;
                    bestNx = candX;
                    bestNy = candY;
                }
            }
        }

        if (bestScore > minCorr) {
            outXY[0] = bestNx;
            outXY[1] = bestNy;
            return true;
        }
        return false;
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
        dst.name = src.name;
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
        dst.memoryCount = (src.gallery != null) ? src.gallery.size() : src.memoryCount;
        dst.deepCount = (src.deepGallery != null) ? src.deepGallery.size() : src.deepCount;
        dst.faceCount = (src.faceGallery != null) ? src.faceGallery.size() : src.faceCount;
        dst.hasDeepAi = dst.deepCount > 0;
        dst.hasFaceAi = dst.faceCount > 0;
        dst.needsImmediateNeural = src.needsImmediateNeural;
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
            if (neuralEngine != null) {
                neuralEngine.close();
            }
        }
        List<PersonMemoryStore.PersonProfile> finalSnap;
        int finalNextId;
        synchronized (stateLock) {
            finalSnap = snapshotRoomMemoryLocked();
            finalNextId = nextTrackId;
            memoryDirty = false;
        }
        memoryStore.close(finalSnap, finalNextId);
    }
}
