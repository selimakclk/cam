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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Pro 3-Katmanlı Yapay Zeka İnsan Tespit, Takip ve Kalıcı Hafıza Motoru (v3.0 — 0 Kasma):
 *  - Ana AI Hattı (Hızlı & Bloklamasız):
 *      * EfficientDet-Lite0 INT8 (320x320 BiFPN, Google MediaPipe XNNPACK) -> daha sıkı kutular, düşük ışık & oturan kişi tespiti
 *      * Sabit Oda Uyarlamalı Arka Plan Maskesi (~0.15 ms) -> duvar/koltuk piksellerini Re-ID'den çıkarır
 *      * Keskinlik Filtreli 129-D Çok-İpuculu Re-ID + 16 Açılı Galeri
 *      * 2D Kalman Filtresi + Tam Global Optimal Eşleştirme (Hungarian) + ~1.8 sn Akıllı Gizlenme Tamponu
 *      * Sıfır-GC 30 FPS Optik NCC Takip (fastTrackFrame)
 *  - Tamamen Asenkron Arka Plan Kimlik Hattı (Düşük öncelikli ayrı thread — Ana FPS'yi ASLA etkilemez):
 *      * OSNet MSMT17 (osnet_x0_25_msmt17.tflite — 512-D gerçek İnsan Re-ID, ~6 ms): sırtı dönük/farklı duvar/gölge tanıma
 *      * BlazeFace + MobileFaceNet (192-D Biyometrik Yüz AI) + Laplacian keskinlik kapısı: kıyafet değişiminde kurtarma
 *      * İsteğe Bağlı Tek Tık "360° Kişiyi Tanıt" Rehberli Kayıt Modu (Ön/Yan/Arka)
 */
public class PersonDetectorEngine {

    private static final String TAG = "PersonDetectorEngine";
    private static final String MODEL_FILE = "efficientdet_lite0_int8.tflite";
    private static final float FULL_MIN_SCORE = 0.28f;
    private static final float CROP_MIN_SCORE = 0.23f;
    private static final int MAX_FULL_SIDE = 320; // EfficientDet-Lite0 doğal 320x320 INT8 girişi

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

    private static final int MAX_LOCKED_GALLERY = 16;
    private static final int MAX_LOCKED_DEEP_GALLERY = 12; // OSNet 512-D
    private static final int MAX_LOCKED_FACE_GALLERY = 10; // MobileFaceNet 192-D
    public static final long DEFAULT_TARGET_LOST_TIMEOUT_MS = 1800L; // ~1.8 sn akıllı kısa gizlenme tamponu
    private static final long KALMAN_GHOST_TTL_MS = 1800L;
    private static final float MIN_SIG_SHARPNESS = 14.0f; // 160x120 tamponda bulanık kare eşiği

    // Asenkron OSNet + Yüz AI tetikleme aralıkları (processFrame'i ASLA bloklamaz!)
    private static final long IDENTITY_BG_INTERVAL_MS = 440L;
    private static final long IDENTITY_LOST_INTERVAL_MS = 220L;
    private static final long IDENTITY_ENROLL_INTERVAL_MS = 160L;

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
        public List<float[]> gallery;     // 16 adede kadar arka plan maskeli 129-D imzası (slot 0 = anchor)
        public List<float[]> deepGallery; // 12 adede kadar OSNet 512-D gerçek İnsan Re-ID vektörü
        public List<float[]> faceGallery; // 10 adede kadar net ön-yüz MobileFaceNet 192-D vektörü
        public float[] template;
        public long lastSeen;
        public long lastUpdate;
        public int recoverStreak;
        public int memoryCount;
        public int deepCount;
        public int faceCount;
        public boolean hasDeepAi;
        public boolean hasFaceAi;
        public boolean isOccluded;
        public boolean lost;

        public String displayLabel() {
            return (name != null && !name.trim().isEmpty()) ? name.trim() : ("Kişi #" + id);
        }
    }

    public static class EnrollmentStatus {
        public boolean active;
        public int targetId;
        public String targetName;
        public int step;        // 1: Ön & Yüz, 2: Yan Açı, 3: Arka / Hareket
        public int totalSteps = 3;
        public int progressPct; // 0..100
        public int sigSamples;
        public int deepSamples;
        public int faceSamples;
        public String instruction;
        public boolean justCompleted;
    }

    public static class AnalysisOutput {
        public List<TrackedPerson> persons;
        public LockedState locked;
        public EnrollmentStatus enrollment;
        public boolean targetJustLost;
        public boolean targetJustRecovered;
        public int inferTimeMs;
        public int rememberedCount;
    }

    private ObjectDetector detector;
    private final NeuralIdentityEngine neuralEngine;
    private final PersonMemoryStore memoryStore;
    private final RoomBackgroundMask roomBgMask = new RoomBackgroundMask(ANA_W, ANA_H);

    private final ExecutorService identityExecutor;
    private final AtomicBoolean identityBusy = new AtomicBoolean(false);
    private volatile long lastIdentityDispatchTs = 0L;
    private volatile boolean pendingAsyncRecovery = false;
    private volatile long targetLostTimeoutMs = DEFAULT_TARGET_LOST_TIMEOUT_MS;

    private volatile boolean initialized = false;
    private volatile boolean closed = false;
    private boolean usingGpu = false;
    private String initError = null;

    private final List<TrackedPerson> prevTracks = new ArrayList<>();
    private final List<SmartTrackerMath.KalmanTrackState> kalmanTracks = new ArrayList<>();
    private final List<PersonMemoryStore.PersonProfile> roomMemory = new ArrayList<>();
    private int nextTrackId = 1;
    private TrackedPerson pendingRecoveryCandidate = null;
    private LockedState lockedTarget = null;
    private int frameCounter = 0;
    private boolean memoryDirty = false;
    private int priorityIdentityCheckId = -1; // Kadraja yeni giren kişinin OSNet/Yüz önceliği

    // 360° Rehberli Kişi Tanıtım Durumu (stateLock)
    private boolean enrollActive = false;
    private int enrollTargetId = -1;
    private String enrollTargetName = null;
    private int enrollStep = 1;
    private long enrollStepStartMs = 0L;
    private int enrollBaseDeepCount = 0;
    private int enrollBaseSigCount = 0;
    private int enrollBaseFaceCount = 0;
    private boolean enrollJustCompleted = false;

    // AI Worker tamponları (detectorLock)
    private final int[] aiPixels = new int[ANA_W * ANA_H];
    private final Bitmap aiSmallBitmap = Bitmap.createBitmap(ANA_W, ANA_H, Bitmap.Config.ARGB_8888);
    private final Canvas aiSmallCanvas = new Canvas(aiSmallBitmap);
    private final Rect aiSrcRect = new Rect();
    private final Rect aiDstRect = new Rect(0, 0, ANA_W, ANA_H);
    private final float[] aiTplScratch = new float[TPL_W * TPL_H];
    private final float[] aiHitScratch = new float[2];
    private final float[] aiPatchWeights = new float[PATCH_LEN];
    private boolean aiPixelsValid = false;

    // Arka plan OSNet + Yüz thread'i için izole kopya tamponu ve ön-plan maske anlık görüntüsü (0 GC)
    private Bitmap identityWorkBitmap = null;
    private Canvas identityWorkCanvas = null;
    private final byte[] identityFgMaskBuf = new byte[ANA_W * ANA_H];
    private final float[] centroidScratch = new float[512];

    // 30 FPS Optik Takip tamponları (fastLock — SIFIR tahsis!)
    private final int[] fastPixels = new int[ANA_W * ANA_H];
    private final Bitmap fastSmallBitmap = Bitmap.createBitmap(ANA_W, ANA_H, Bitmap.Config.ARGB_8888);
    private final Canvas fastSmallCanvas = new Canvas(fastSmallBitmap);
    private final Rect fastSrcRect = new Rect();
    private final Rect fastDstRect = new Rect(0, 0, ANA_W, ANA_H);
    private final float[] fastTplScratch = new float[TPL_W * TPL_H];
    private final float[] fastSigScratch = new float[SIG_LEN];
    private final float[] fastHitScratch = new float[2];

    // Bilinear filtre açık: 512px -> 320px EfficientDet & 160x120 küçültmede sıfır aliasing!
    private final Paint fastDrawPaint = new Paint(Paint.FILTER_BITMAP_FLAG);

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

        PersonMemoryStore.LoadResult loaded = memoryStore.load();
        synchronized (stateLock) {
            roomMemory.addAll(loaded.profiles);
            nextTrackId = loaded.nextTrackId;
        }

        this.identityExecutor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "AsyncProIdentityThread");
            t.setPriority(Thread.MIN_PRIORITY + 1);
            return t;
        });

        this.neuralEngine = new NeuralIdentityEngine(context);

        if (tryInitDetector(context, Delegate.CPU, false)) {
            usingGpu = false;
            initialized = true;
            Log.i(TAG, "Pro v3.0 AI Motoru Hazır: " + neuralEngine.getActiveEnginesSummary()
                    + " | Kalıcı Profil: " + loaded.profiles.size());
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

    public boolean isReidAiReady() {
        return neuralEngine != null && neuralEngine.isReidReady();
    }

    public boolean isFaceAiReady() {
        return neuralEngine != null && neuralEngine.isFaceReady();
    }

    public String getAiEngineSummary() {
        return (neuralEngine != null) ? neuralEngine.getActiveEnginesSummary() : "EfficientDet-INT8 + Maskeli 129-D Re-ID";
    }

    public String getInitError() {
        return initError;
    }

    public int getRememberedCount() {
        synchronized (stateLock) {
            return roomMemory.size();
        }
    }

    public float getBackgroundMaskCoverage() {
        synchronized (detectorLock) {
            return roomBgMask.getLearnedCoverageRatio();
        }
    }

    public void setTargetLostTimeoutMs(long timeoutMs) {
        this.targetLostTimeoutMs = Math.max(400L, Math.min(8000L, timeoutMs));
    }

    public long getTargetLostTimeoutMs() {
        return targetLostTimeoutMs;
    }

    /**
     * Asenkron OSNet + Yüz AI arka plan işinin bitmesini beklemek için yardımcı metod (testler/senkron doğrulama).
     */
    public void waitForPendingIdentityWork(long timeoutMs) {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (identityBusy.get() && System.currentTimeMillis() < deadline) {
            try { Thread.sleep(12); } catch (InterruptedException ignored) { break; }
        }
    }

    public void waitForPendingFaceWork(long timeoutMs) {
        waitForPendingIdentityWork(timeoutMs);
    }

    // =================== 360° REHBERLİ KİŞİ TANITIM MODU ===================
    public boolean startGuidedEnrollment(int personId, String optionalName) {
        String cleanName = (optionalName != null && !optionalName.trim().isEmpty()) ? optionalName.trim() : null;
        synchronized (stateLock) {
            if (closed) return false;
            if (lockedTarget == null || lockedTarget.id != personId) {
                // Eğer o ID görünüyorsa otomatik kilitlemeye çalış
                TrackedPerson found = null;
                for (TrackedPerson tp : prevTracks) {
                    if (tp.id == personId) {
                        found = tp;
                        break;
                    }
                }
                if (found == null && lockedTarget == null) return false;
                if (found != null) {
                    LockedState t = buildLockedStateFromPersonLocked(found, System.currentTimeMillis());
                    lockedTarget = t;
                }
            }
            if (lockedTarget == null) return false;
            int id = lockedTarget.id;
            if (cleanName != null) {
                lockedTarget.name = cleanName;
                PersonMemoryStore.PersonProfile p = findProfileByIdLocked(id);
                if (p != null) p.name = cleanName;
                for (TrackedPerson tp : prevTracks) {
                    if (tp.id == id) tp.name = cleanName;
                }
            }
            enrollActive = true;
            enrollTargetId = id;
            enrollTargetName = lockedTarget.name;
            enrollStep = 1;
            enrollStepStartMs = System.currentTimeMillis();
            enrollBaseSigCount = (lockedTarget.gallery != null) ? lockedTarget.gallery.size() : 0;
            enrollBaseDeepCount = (lockedTarget.deepGallery != null) ? lockedTarget.deepGallery.size() : 0;
            enrollBaseFaceCount = (lockedTarget.faceGallery != null) ? lockedTarget.faceGallery.size() : 0;
            enrollJustCompleted = false;
            return true;
        }
    }

    public boolean finishGuidedEnrollmentNow() {
        List<PersonMemoryStore.PersonProfile> snap = null;
        int saveNextId = 1;
        boolean finished = false;
        synchronized (stateLock) {
            if (enrollActive) {
                completeEnrollmentLocked(System.currentTimeMillis());
                snap = snapshotRoomMemoryLocked();
                saveNextId = nextTrackId;
                memoryDirty = false;
                finished = true;
            }
        }
        if (snap != null) {
            memoryStore.saveSync(snap, saveNextId);
        }
        return finished;
    }

    public void cancelGuidedEnrollment() {
        synchronized (stateLock) {
            enrollActive = false;
            enrollTargetId = -1;
            enrollJustCompleted = false;
        }
    }

    public EnrollmentStatus getEnrollmentStatus() {
        synchronized (stateLock) {
            return buildEnrollmentStatusLocked(System.currentTimeMillis());
        }
    }

    private void updateEnrollmentProgressLocked(long now) {
        if (!enrollActive) return;
        if (lockedTarget == null || lockedTarget.id != enrollTargetId) {
            enrollActive = false;
            return;
        }
        if (lockedTarget.lost) return;

        int sigCnt = (lockedTarget.gallery != null) ? lockedTarget.gallery.size() : 0;
        int deepCnt = (lockedTarget.deepGallery != null) ? lockedTarget.deepGallery.size() : 0;
        int faceCnt = (lockedTarget.faceGallery != null) ? lockedTarget.faceGallery.size() : 0;
        long elapsed = now - enrollStepStartMs;

        if (enrollStep == 1) {
            boolean faceGoalMet = (faceCnt >= Math.min(MAX_LOCKED_FACE_GALLERY, Math.max(2, enrollBaseFaceCount + 1)))
                    && (deepCnt >= Math.min(MAX_LOCKED_DEEP_GALLERY, Math.max(1, enrollBaseDeepCount + 1)));
            if ((faceGoalMet && elapsed >= 1200L) || elapsed >= 3400L) {
                enrollStep = 2;
                enrollStepStartMs = now;
                enrollBaseDeepCount = deepCnt;
                enrollBaseSigCount = sigCnt;
            }
        } else if (enrollStep == 2) {
            boolean angleGoalMet = (deepCnt >= Math.min(MAX_LOCKED_DEEP_GALLERY, enrollBaseDeepCount + 2))
                    || (sigCnt >= Math.min(MAX_LOCKED_GALLERY, enrollBaseSigCount + 3));
            if ((angleGoalMet && elapsed >= 1400L) || elapsed >= 3200L) {
                enrollStep = 3;
                enrollStepStartMs = now;
                enrollBaseDeepCount = deepCnt;
                enrollBaseSigCount = sigCnt;
            }
        } else if (enrollStep == 3) {
            boolean backGoalMet = (deepCnt >= Math.min(MAX_LOCKED_DEEP_GALLERY, enrollBaseDeepCount + 2))
                    || (sigCnt >= Math.min(MAX_LOCKED_GALLERY, enrollBaseSigCount + 3));
            if ((backGoalMet && elapsed >= 1400L) || elapsed >= 3200L) {
                completeEnrollmentLocked(now);
            }
        }
    }

    private void completeEnrollmentLocked(long now) {
        int id = enrollTargetId;
        enrollActive = false;
        enrollJustCompleted = true;
        PersonMemoryStore.PersonProfile prof = findProfileByIdLocked(id);
        if (prof == null && lockedTarget != null && lockedTarget.id == id) {
            updateRoomMemoryLocked(id, lockedTarget.anchorSig, null, null, now, false);
            prof = findProfileByIdLocked(id);
        }
        if (prof != null) {
            prof.enrolled = true;
            if (enrollTargetName != null && !enrollTargetName.trim().isEmpty()) {
                prof.name = enrollTargetName.trim();
            }
            if (lockedTarget != null && lockedTarget.id == id) {
                if (lockedTarget.gallery != null) {
                    for (float[] s : lockedTarget.gallery) updateRoomMemoryLocked(id, s, null, null, now, false);
                }
                if (lockedTarget.deepGallery != null) {
                    for (float[] d : lockedTarget.deepGallery) updateRoomMemoryLocked(id, null, d, null, now, false);
                }
                if (lockedTarget.faceGallery != null) {
                    for (float[] f : lockedTarget.faceGallery) updateRoomMemoryLocked(id, null, null, f, now, false);
                }
            }
            prof.lastSeen = now;
            memoryDirty = true;
        }
    }

    private EnrollmentStatus buildEnrollmentStatusLocked(long now) {
        if (!enrollActive && !enrollJustCompleted) return null;
        EnrollmentStatus st = new EnrollmentStatus();
        st.active = enrollActive;
        st.targetId = enrollTargetId;
        st.targetName = enrollTargetName;
        st.step = enrollStep;
        st.justCompleted = enrollJustCompleted;

        PersonMemoryStore.PersonProfile prof = findProfileByIdLocked(enrollTargetId);
        if (lockedTarget != null && lockedTarget.id == enrollTargetId) {
            st.sigSamples = (lockedTarget.gallery != null) ? lockedTarget.gallery.size() : 0;
            st.deepSamples = (lockedTarget.deepGallery != null) ? lockedTarget.deepGallery.size() : 0;
            st.faceSamples = (lockedTarget.faceGallery != null) ? lockedTarget.faceGallery.size() : 0;
            if (lockedTarget.name != null) st.targetName = lockedTarget.name;
        } else if (prof != null) {
            st.sigSamples = prof.sigGallery.size();
            st.deepSamples = prof.deepBodyGallery.size();
            st.faceSamples = prof.faceGallery.size();
            if (prof.name != null) st.targetName = prof.name;
        }

        if (enrollJustCompleted) {
            st.progressPct = 100;
            st.instruction = "360° Kayıt Tamamlandı! (OSNet:" + st.deepSamples + " Yüz:" + st.faceSamples + ")";
            return st;
        }

        float stepFrac = Math.min(1f, Math.max(0f, (now - enrollStepStartMs) / 3200f));
        st.progressPct = Math.min(99, Math.round(((enrollStep - 1) + stepFrac) / 3.0f * 100f));
        if (enrollStep == 1) {
            st.instruction = "360° Kayıt (1/3) • Kameraya Bakın (Yüz:" + st.faceSamples + " OSNet:" + st.deepSamples + ")";
        } else if (enrollStep == 2) {
            st.instruction = "360° Kayıt (2/3) • Sağa / Sola Dönün (OSNet:" + st.deepSamples + ")";
        } else {
            st.instruction = "360° Kayıt (3/3) • Sırtınızı Dönün / Hareket Edin (OSNet:" + st.deepSamples + ")";
        }
        return st;
    }

    /**
     * Kamera thread'inden her karede (<1ms, SIFIR tahsis) çalışarak kilitli hedefi 30 FPS yumuşaklıkta takip eder.
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
                if (anchorSnap != null && computeSignatureInto(fastPixels, hitX, hitY, pw, ph, fastSigScratch, false)) {
                    float torsoCheck = zoneSim(anchorSnap, fastSigScratch, 1);
                    if (torsoCheck < 0.46f) {
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

    /**
     * Ana AI kare işleme hattı:
     *  - EfficientDet-Lite0 INT8 (320px) + Uyarlamalı Arka Plan Maskesi + 129-D Re-ID + Kalman & Hungarian Eşleştirme.
     *  - OSNet 512-D + BlazeFace + MobileFaceNet 192-D tamamen ayrı düşük öncelikli thread'de asenkron çalışır (0 ms bloklama!).
     */
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
        List<PersonMemoryStore.PersonProfile> saveSnapshot = null;
        int saveNextId = 1;
        TrackedPerson identityCandidate = null;
        boolean lockedNeedsRecovery = false;
        boolean enrollNow = false;

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
            // NOT: detBmp yeniden kullanıldığı için mpImage.close() çağrılmaz!
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
            } else if (candidates.isEmpty() && (frameCounter % 10 == 0)) {
                int slot = (frameCounter / 10) % 3;
                float cx = (slot == 0) ? 0.30f : (slot == 1 ? 0.50f : 0.70f);
                List<TrackedPerson> farDets = detectZoomedRegionInternal(uprightBitmap, cx, 0.44f, 0.46f);
                candidates = mergeDetections(candidates, farDets);
            }

            // 1. Sabit oda arka plan maskesini güncelle (~0.15 ms) -> kişi pikselleri arka plana ASLA karışmaz!
            ensureAiPixelsLocked(uprightBitmap);
            roomBgMask.updateWithFrame(aiPixels, candidates);

            // 2. Arka plan maskeli 129-D Re-ID imzalarını + keskinlik puanlarını çıkar (~0.2 ms)
            for (TrackedPerson p : candidates) {
                p.signature = computeSignature(aiPixels, p.nx, p.ny, p.nw, p.nh, true);
            }

            synchronized (stateLock) {
                long now = System.currentTimeMillis();
                if (pendingAsyncRecovery) {
                    pendingAsyncRecovery = false;
                    out.targetJustRecovered = true;
                }

                pendingRecoveryCandidate = null;
                if (lockedTarget != null) {
                    updateLockedTargetLocked(candidates, now, out);
                    out.locked = copyLockedState(lockedTarget);
                    // Kısa gizlenme (0..1.8 sn) anında da OSNet + Yüz hızlı kurtarma moduna girsin!
                    lockedNeedsRecovery = (lockedTarget.lost || lockedTarget.isOccluded);
                }

                assignStableIdsWithKalmanAndHungarianLocked(candidates, now);
                updateEnrollmentProgressLocked(now);
                out.enrollment = buildEnrollmentStatusLocked(now);
                if (enrollJustCompleted) {
                    enrollJustCompleted = false;
                }
                enrollNow = enrollActive;

                out.persons = candidates;
                out.rememberedCount = roomMemory.size();

                // Asenkron OSNet 512-D + Yüz 192-D için en uygun 1 kişiyi seç (0 ms ana FPS bloklaması!)
                if ((neuralEngine.isReidReady() || neuralEngine.isFaceReady())
                        && !candidates.isEmpty() && !identityBusy.get()) {
                    long interval = enrollNow ? IDENTITY_ENROLL_INTERVAL_MS
                            : (lockedNeedsRecovery ? IDENTITY_LOST_INTERVAL_MS : IDENTITY_BG_INTERVAL_MS);
                    if (now - lastIdentityDispatchTs >= interval || priorityIdentityCheckId > 0) {
                        identityCandidate = pickBestIdentityCandidateLocked(candidates, lockedNeedsRecovery, enrollNow);
                    }
                }

                if (memoryDirty) {
                    memoryDirty = false;
                    saveSnapshot = snapshotRoomMemoryLocked();
                    saveNextId = nextTrackId;
                }
            }

            // Seçilen tek kişi için arka plan OSNet + Yüz AI görevini maskesiyle birlikte kopyalayıp ateşle (~0.3 ms!)
            if (identityCandidate != null) {
                dispatchAsyncIdentityTaskLocked(uprightBitmap, identityCandidate, lockedNeedsRecovery, enrollNow);
            }
        }

        if (saveSnapshot != null) {
            memoryStore.saveDebouncedAsync(saveSnapshot, saveNextId);
        }

        out.inferTimeMs = (int) (System.currentTimeMillis() - t0);
        return out;
    }

    private TrackedPerson pickBestIdentityCandidateLocked(
            List<TrackedPerson> candidates, boolean lockedLost, boolean enrollNow) {
        if (enrollNow && enrollTargetId > 0) {
            for (TrackedPerson c : candidates) {
                if (c.id == enrollTargetId) return c;
            }
        }
        if (lockedLost) {
            // Kilitli hedef kayıpsa sırayla adayları OSNet + Yüz ile tara!
            int idx = frameCounter % candidates.size();
            return candidates.get(idx);
        }
        // Kadraja yeni/tekrar giren kişiye OSNet/Yüz önceliği ver (sırtı dönük girse bile anında tanı!)
        if (priorityIdentityCheckId > 0) {
            for (TrackedPerson c : candidates) {
                if (c.id == priorityIdentityCheckId) {
                    priorityIdentityCheckId = -1;
                    return c;
                }
            }
            priorityIdentityCheckId = -1;
        }
        // Kilitli hedefin OSNet veya Yüz galerisi henüz zenginleşmediyse öncelik ver
        if (lockedTarget != null && !lockedTarget.lost) {
            int deepSz = (lockedTarget.deepGallery != null) ? lockedTarget.deepGallery.size() : 0;
            int faceSz = (lockedTarget.faceGallery != null) ? lockedTarget.faceGallery.size() : 0;
            if (deepSz < 5 || faceSz < 3) {
                for (TrackedPerson c : candidates) {
                    if (c.id == lockedTarget.id && !isHorizontallyEdgeClipped(c.nx, c.ny, c.nw, c.nh)) {
                        return c;
                    }
                }
            }
        }
        int start = frameCounter % candidates.size();
        for (int i = 0; i < candidates.size(); i++) {
            TrackedPerson c = candidates.get((start + i) % candidates.size());
            if (!isHorizontallyEdgeClipped(c.nx, c.ny, c.nw, c.nh) && c.score >= 0.30f) {
                return c;
            }
        }
        return candidates.get(0);
    }

    private void dispatchAsyncIdentityTaskLocked(
            Bitmap srcBitmap, TrackedPerson cand, boolean lockedRecovery, boolean enrollNow) {
        if (closed || !identityBusy.compareAndSet(false, true)) return;
        try {
            int bw = srcBitmap.getWidth();
            int bh = srcBitmap.getHeight();
            if (identityWorkBitmap == null || identityWorkBitmap.isRecycled()
                    || identityWorkBitmap.getWidth() != bw || identityWorkBitmap.getHeight() != bh) {
                if (identityWorkBitmap != null && !identityWorkBitmap.isRecycled()) identityWorkBitmap.recycle();
                identityWorkBitmap = Bitmap.createBitmap(bw, bh, Bitmap.Config.ARGB_8888);
                identityWorkCanvas = new Canvas(identityWorkBitmap);
            }
            identityWorkCanvas.drawBitmap(srcBitmap, 0f, 0f, null);
            // Sabit oda arka plan maskesini OSNet kırpımı için sıfır-GC kopyala (~0.04 ms!)
            final boolean hasFgMask = aiPixelsValid && roomBgMask.snapshotForegroundBytes(aiPixels, identityFgMaskBuf);
            lastIdentityDispatchTs = System.currentTimeMillis();

            final int candId = cand.id;
            final float nx = cand.nx, ny = cand.ny, nw = cand.nw, nh = cand.nh, score = cand.score;
            final float[] candSigCopy = (cand.signature != null) ? cand.signature.clone() : null;
            final boolean runFace = neuralEngine.isFaceReady();
            final boolean runReid = neuralEngine.isReidReady();

            identityExecutor.execute(() -> {
                try {
                    if (closed || identityWorkBitmap == null || identityWorkBitmap.isRecycled()) return;
                    NeuralIdentityEngine.IdentityResult ir = neuralEngine.extractIdentity(
                            identityWorkBitmap, nx, ny, nw, nh, runReid, runFace,
                            hasFgMask ? identityFgMaskBuf : null, ANA_W, ANA_H);
                    if ((ir.body != null && ir.body.embedding != null)
                            || (ir.face != null && ir.face.faceEmbedding != null)) {
                        applyAsyncIdentityResult(candId, nx, ny, nw, nh, score, candSigCopy, ir, lockedRecovery, enrollNow);
                    }
                } catch (Throwable ignored) {
                } finally {
                    identityBusy.set(false);
                }
            });
        } catch (Throwable t) {
            identityBusy.set(false);
        }
    }

    /**
     * Asenkron OSNet 512-D + Yüz 192-D sonucunu güvenli biçimde hafızaya ve aktif takibe uygular.
     *  - Kilitli hedef KAYIP veya KISA GİZLENMEDE (0..1.8 sn) ise:
     *      Yüz (>=0.60) VEYA OSNet Centroid+Top2 (>=0.71 veya 0.64 + 129-D>=0.48) ile ANINDA kurtarır!
     *  - Serbest/yeni giren kişiyi kalıcı profillerle (OSNet + Yüz) eşleştirip ID/İsim kurtarır!
     *  - Sadece keskin (Laplacian filtresini geçen) kareleri prototip birleştirmeyle hafızaya işler.
     */
    private void applyAsyncIdentityResult(
            int candId, float nx, float ny, float nw, float nh, float score,
            float[] candSig, NeuralIdentityEngine.IdentityResult ir, boolean wasLockedRecovery, boolean enrollNow) {
        List<PersonMemoryStore.PersonProfile> saveSnap = null;
        int saveNextId = 1;
        long now = System.currentTimeMillis();

        NeuralIdentityEngine.BodyReidResult br = ir.body;
        NeuralIdentityEngine.FaceResult fr = ir.face;
        float[] osnetVec = (br != null) ? br.embedding : null;
        boolean osnetSharp = (br != null && br.isSharp);
        float[] faceVec = (fr != null) ? fr.faceEmbedding : null;
        boolean faceSharpFrontal = (fr != null && fr.isFrontal && fr.isSharp);

        synchronized (stateLock) {
            if (closed) return;

            boolean matchedLockedNow = false;
            // 1. Kilitli hedef KAYIP veya KISA GİZLENMEDE ise: Yüz AI veya OSNet 512-D ile alarm çalmadan/anında KURTAR!
            if (lockedTarget != null && (lockedTarget.lost || lockedTarget.isOccluded)) {
                boolean wasLostBefore = lockedTarget.lost;
                float faceSim = (faceVec != null && lockedTarget.faceGallery != null && !lockedTarget.faceGallery.isEmpty())
                        ? matchEmbeddingGallery(lockedTarget.faceGallery, faceVec) : -1f;
                float osnetSim = (osnetVec != null && lockedTarget.deepGallery != null && !lockedTarget.deepGallery.isEmpty())
                        ? matchEmbeddingGallery(lockedTarget.deepGallery, osnetVec) : -1f;
                boolean legsCropped = (ny + nh > 0.92f);
                float sigSim = (candSig != null && lockedTarget.gallery != null && !lockedTarget.gallery.isEmpty())
                        ? matchGallery(lockedTarget.gallery, candSig, legsCropped).combined : 0f;

                boolean recoveredByFace = (faceSim >= 0.60f);
                boolean recoveredByOsnet = (osnetSim >= 0.71f) || (osnetSim >= 0.64f && sigSim >= 0.48f);

                if (recoveredByFace || recoveredByOsnet) {
                    matchedLockedNow = true;
                    lockedTarget.lost = false;
                    lockedTarget.isOccluded = false;
                    lockedTarget.recoverStreak = 0;
                    lockedTarget.nx = nx;
                    lockedTarget.ny = ny;
                    lockedTarget.nw = nw;
                    lockedTarget.nh = nh;
                    lockedTarget.score = score;
                    lockedTarget.vx = 0f;
                    lockedTarget.vy = 0f;
                    lockedTarget.lastSeen = now;
                    lockedTarget.lastUpdate = now;

                    // Kıyafet değişimi olduysa yeni görünümü prototip birleştirmeyle hedef galerilerine adapte et!
                    if (candSig != null && (recoveredByFace || osnetSharp)) {
                        if (lockedTarget.gallery == null) lockedTarget.gallery = new ArrayList<>(MAX_LOCKED_GALLERY);
                        addOrMergeSignature(lockedTarget.gallery, candSig, MAX_LOCKED_GALLERY, 0.87f, true, recoveredByFace);
                        if (recoveredByFace) lockedTarget.anchorSig = candSig.clone();
                    }
                    if (osnetVec != null && (osnetSharp || recoveredByFace)) {
                        if (lockedTarget.deepGallery == null) lockedTarget.deepGallery = new ArrayList<>(MAX_LOCKED_DEEP_GALLERY);
                        addDiverseEmbedding(lockedTarget.deepGallery, osnetVec, MAX_LOCKED_DEEP_GALLERY, 0.88f);
                    }
                    if (faceVec != null && faceSharpFrontal) {
                        if (lockedTarget.faceGallery == null) lockedTarget.faceGallery = new ArrayList<>(MAX_LOCKED_FACE_GALLERY);
                        addDiverseEmbedding(lockedTarget.faceGallery, faceVec, MAX_LOCKED_FACE_GALLERY, 0.90f);
                    }
                    updateRoomMemoryLocked(
                            lockedTarget.id,
                            (recoveredByFace || osnetSharp) ? candSig : null,
                            (osnetSharp || recoveredByFace) ? osnetVec : null,
                            faceSharpFrontal ? faceVec : null,
                            now, recoveredByFace);

                    rebindTrackIdLocked(candId, lockedTarget.id, lockedTarget.name,
                            lockedTarget.deepGallery != null && !lockedTarget.deepGallery.isEmpty(),
                            lockedTarget.faceGallery != null && !lockedTarget.faceGallery.isEmpty());

                    if (wasLostBefore) {
                        pendingAsyncRecovery = true;
                    }
                    if (memoryDirty) {
                        memoryDirty = false;
                        saveSnap = snapshotRoomMemoryLocked();
                        saveNextId = nextTrackId;
                    }
                }
            }

            // 2. Aktif / serbest kişi: Yüz AI veya OSNet 512-D ile kalıcı/oda profilini tanı ve galeriyi besle!
            //    (Kilitli hedef kayıpken odaya giren DİĞER kayıtlı kişiler de artık tanınabilir!)
            if (!matchedLockedNow && candId > 0) {
                PersonMemoryStore.PersonProfile matchedProf = null;
                boolean matchedViaFace = false;

                // 2a. Önce Biyometrik Yüz AI kontrolü (en yüksek kesinlik, kıyafet bağımsız!)
                if (faceVec != null) {
                    float bestFace = 0.62f;
                    for (PersonMemoryStore.PersonProfile mem : roomMemory) {
                        if (mem.faceGallery.isEmpty()) continue;
                        if (lockedTarget != null && lockedTarget.id == mem.id) continue;
                        float fs = matchEmbeddingGallery(mem.faceGallery, faceVec);
                        if (fs > bestFace) {
                            bestFace = fs;
                            matchedProf = mem;
                            matchedViaFace = true;
                        }
                    }
                }

                // 2b. Yüz görünmüyorsa (sırtı dönük / yan profil) OSNet 512-D ile profil ara!
                if (matchedProf == null && osnetVec != null) {
                    float bestOsnet = 0.72f;
                    boolean legsCropped = (ny + nh > 0.92f);
                    for (PersonMemoryStore.PersonProfile mem : roomMemory) {
                        if (mem.deepBodyGallery.isEmpty()) continue;
                        if (lockedTarget != null && lockedTarget.id == mem.id) continue;
                        // Başka aktif bir kişiye zaten bağlı profili çalma!
                        if (mem.id != candId && isIdActiveInPrevTracksLocked(mem.id)) continue;
                        float os = matchEmbeddingGallery(mem.deepBodyGallery, osnetVec);
                        float sigBonus = (candSig != null && !mem.sigGallery.isEmpty())
                                ? matchGallery(mem.sigGallery, candSig, legsCropped).combined : 0f;
                        boolean passes = (os >= 0.73f) || (os >= 0.66f && sigBonus >= 0.52f);
                        if (passes && os > bestOsnet) {
                            bestOsnet = os;
                            matchedProf = mem;
                        }
                    }
                }

                int effectiveId = candId;
                if (matchedProf != null && matchedProf.id != candId
                        && (lockedTarget == null || lockedTarget.id != candId)) {
                    effectiveId = matchedProf.id;
                    rebindTrackIdLocked(candId, matchedProf.id, matchedProf.name,
                            !matchedProf.deepBodyGallery.isEmpty() || osnetVec != null,
                            !matchedProf.faceGallery.isEmpty() || faceVec != null);

                    // Geçici isimsiz profil oluştuysa sil ki hafıza temiz kalsın
                    PersonMemoryStore.PersonProfile tempProf = findProfileByIdLocked(candId);
                    if (tempProf != null && !tempProf.isPersistable()) {
                        roomMemory.remove(tempProf);
                    }
                    updateRoomMemoryLocked(
                            effectiveId,
                            candSig,
                            (osnetVec != null && (osnetSharp || matchedViaFace)) ? osnetVec : null,
                            faceSharpFrontal ? faceVec : null,
                            now, matchedViaFace);
                } else if (!wasLockedRecovery || (lockedTarget == null || lockedTarget.id != candId)) {
                    PersonMemoryStore.PersonProfile curProf = findProfileByIdLocked(effectiveId);
                    // Keskinlik kapısı: sadece net kareleri (veya ilk tohum vektörünü) galerilere kaydet!
                    boolean allowStoreOsnet = (osnetVec != null)
                            && (osnetSharp || enrollNow || curProf == null || curProf.deepBodyGallery.isEmpty());
                    boolean allowStoreFace = (faceVec != null)
                            && (faceSharpFrontal || ((curProf == null || curProf.faceGallery.isEmpty()) && fr != null && fr.faceScore >= 0.68f));

                    if (allowStoreOsnet || allowStoreFace) {
                        updateRoomMemoryLocked(
                                effectiveId,
                                null,
                                allowStoreOsnet ? osnetVec : null,
                                allowStoreFace ? faceVec : null,
                                now, false);

                        if (lockedTarget != null && lockedTarget.id == effectiveId) {
                            if (allowStoreOsnet) {
                                if (lockedTarget.deepGallery == null) lockedTarget.deepGallery = new ArrayList<>(MAX_LOCKED_DEEP_GALLERY);
                                addDiverseEmbedding(lockedTarget.deepGallery, osnetVec, MAX_LOCKED_DEEP_GALLERY, 0.88f);
                            }
                            if (allowStoreFace) {
                                if (lockedTarget.faceGallery == null) lockedTarget.faceGallery = new ArrayList<>(MAX_LOCKED_FACE_GALLERY);
                                addDiverseEmbedding(lockedTarget.faceGallery, faceVec, MAX_LOCKED_FACE_GALLERY, 0.90f);
                            }
                        }

                        for (TrackedPerson tp : prevTracks) {
                            if (tp.id == effectiveId) {
                                if (allowStoreOsnet) {
                                    tp.hasDeepAi = true;
                                    tp.deepBodyEmbedding = osnetVec;
                                }
                                if (allowStoreFace) {
                                    tp.hasFaceAi = true;
                                    tp.faceEmbedding = faceVec;
                                }
                            }
                        }
                        for (SmartTrackerMath.KalmanTrackState kt : kalmanTracks) {
                            if (kt.trackId == effectiveId) {
                                if (allowStoreOsnet) kt.hasDeepAi = true;
                                if (allowStoreFace) kt.hasFaceAi = true;
                            }
                        }
                    }
                }

                if (memoryDirty && saveSnap == null) {
                    memoryDirty = false;
                    saveSnap = snapshotRoomMemoryLocked();
                    saveNextId = nextTrackId;
                }
            }
        }

        if (saveSnap != null) {
            memoryStore.saveDebouncedAsync(saveSnap, saveNextId);
        }
    }

    private boolean isIdActiveInPrevTracksLocked(int id) {
        for (TrackedPerson tp : prevTracks) {
            if (tp.id == id) return true;
        }
        return false;
    }

    private void rebindTrackIdLocked(int oldId, int newId, String newName, boolean hasDeep, boolean hasFace) {
        for (TrackedPerson tp : prevTracks) {
            if (tp.id == oldId) {
                tp.id = newId;
                tp.name = newName;
                tp.hasDeepAi = tp.hasDeepAi || hasDeep;
                tp.hasFaceAi = tp.hasFaceAi || hasFace;
            }
        }
        for (SmartTrackerMath.KalmanTrackState kt : kalmanTracks) {
            if (kt.trackId == oldId) {
                kt.trackId = newId;
                kt.name = newName;
                kt.hasDeepAi = kt.hasDeepAi || hasDeep;
                kt.hasFaceAi = kt.hasFaceAi || hasFace;
            }
        }
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
            roomBgMask.reset();
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
                        best.signature = computeSignature(aiPixels, best.nx, best.ny, best.nw, best.nh, true);
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
                best.signature = computeSignature(pxSource, best.nx, best.ny, best.nw, best.nh, true);
            }

            synchronized (stateLock) {
                long now = System.currentTimeMillis();
                LockedState t = buildLockedStateFromPersonLocked(best, now);
                if (aiPixelsValid) {
                    t.template = extractTemplate(pxSource, best.nx, best.ny, best.nw, best.nh);
                }
                lockedTarget = t;
                pendingAsyncRecovery = false;

                if (outLocked != null) {
                    copyInto(t, outLocked);
                }
            }

            // Kilitlenen kişinin OSNet + Yüz verilerini arka planda (0 ms bloklama!) anında öğren
            if ((neuralEngine.isReidReady() || neuralEngine.isFaceReady()) && snap != null && !snap.isRecycled()) {
                dispatchAsyncIdentityTaskLocked(snap, best, false, false);
            }
        }

        return LockStatus.LOCKED;
    }

    private LockedState buildLockedStateFromPersonLocked(TrackedPerson best, long now) {
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
        t.anchorSig = (best.signature != null) ? best.signature.clone() : null;

        t.gallery = new ArrayList<>(MAX_LOCKED_GALLERY);
        if (t.anchorSig != null) t.gallery.add(t.anchorSig.clone());

        t.deepGallery = new ArrayList<>(MAX_LOCKED_DEEP_GALLERY);
        if (best.deepBodyEmbedding != null) t.deepGallery.add(best.deepBodyEmbedding.clone());

        t.faceGallery = new ArrayList<>(MAX_LOCKED_FACE_GALLERY);
        if (best.faceEmbedding != null) t.faceGallery.add(best.faceEmbedding.clone());

        t.lastSeen = now;
        t.lastUpdate = now;
        t.recoverStreak = 0;
        t.isOccluded = false;
        t.lost = false;

        PersonMemoryStore.PersonProfile prof = findProfileByIdLocked(t.id);
        if (prof != null) {
            if (prof.isNamed()) t.name = prof.name;
            for (float[] g : prof.sigGallery) {
                if (t.gallery.size() < MAX_LOCKED_GALLERY) t.gallery.add(g.clone());
            }
            for (float[] d : prof.deepBodyGallery) {
                addDiverseEmbedding(t.deepGallery, d, MAX_LOCKED_DEEP_GALLERY, 0.88f);
            }
            for (float[] f : prof.faceGallery) {
                addDiverseEmbedding(t.faceGallery, f, MAX_LOCKED_FACE_GALLERY, 0.90f);
            }
        }
        updateRoomMemoryLocked(t.id, t.anchorSig, best.deepBodyEmbedding, best.faceEmbedding, now, false);
        return t;
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
            pendingAsyncRecovery = false;
            enrollActive = false;
            enrollTargetId = -1;
            enrollJustCompleted = false;
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

    // =================== KULLANICI HAFIZA YÖNETİMİ ===================
    public List<PersonMemoryStore.ProfileSummary> getMemorySummaries() {
        synchronized (stateLock) {
            List<PersonMemoryStore.ProfileSummary> list = new ArrayList<>(roomMemory.size());
            for (PersonMemoryStore.PersonProfile p : roomMemory) {
                list.add(new PersonMemoryStore.ProfileSummary(
                        p.id, p.name, p.enrolled,
                        p.sigGallery.size(), p.deepBodyGallery.size(), p.faceGallery.size(), p.lastSeen));
            }
            Collections.sort(list, (a, b) -> {
                boolean an = (a.name != null && !a.name.trim().isEmpty()) || a.enrolled;
                boolean bn = (b.name != null && !b.name.trim().isEmpty()) || b.enrolled;
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
            for (SmartTrackerMath.KalmanTrackState kt : kalmanTracks) {
                if (kt.trackId == id) kt.name = cleanName;
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
            for (SmartTrackerMath.KalmanTrackState kt : kalmanTracks) {
                if (kt.trackId == id) kt.name = null;
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
            kalmanTracks.clear();
            nextTrackId = 1;
            if (lockedTarget != null) {
                lockedTarget.name = null;
            }
            enrollActive = false;
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
     * 2D Kalman Tahmini + Tam Global Optimal Eşleştirme (Hungarian / Kuhn-Munkres) + 15 Dk Oda & Kalıcı 3-Katmanlı Hafıza.
     * İki kişi yan yana geçerken veya kısa süreli çakışmada kimlik/ID takasını engeller!
     */
    private void assignStableIdsWithKalmanAndHungarianLocked(List<TrackedPerson> current, long now) {
        // 1. Süresi dolmuş isimsiz oda profillerini temizle (15 dk)
        Iterator<PersonMemoryStore.PersonProfile> mit = roomMemory.iterator();
        while (mit.hasNext()) {
            PersonMemoryStore.PersonProfile m = mit.next();
            boolean isLocked = (lockedTarget != null && lockedTarget.id == m.id);
            if (!isLocked && PersonMemoryStore.isExpired(m, now)) {
                mit.remove();
            }
        }

        // 2. Aktif Kalman takipçilerini tahmin et ve uzun süredir kayıp olan hayaletleri temizle
        Iterator<SmartTrackerMath.KalmanTrackState> kit = kalmanTracks.iterator();
        while (kit.hasNext()) {
            SmartTrackerMath.KalmanTrackState kt = kit.next();
            kt.predict(now);
            if (now - kt.lastSeenMs > KALMAN_GHOST_TTL_MS) {
                kit.remove();
            }
        }

        List<Integer> usedIds = new ArrayList<>();

        // 3. Kilitli hedef zaten eşleştiyse ID'sini sabitle
        if (lockedTarget != null) {
            if (!lockedTarget.lost && !lockedTarget.isOccluded) {
                for (TrackedPerson c : current) {
                    if (computeIoU(c.nx, c.ny, c.nw, c.nh, lockedTarget.nx, lockedTarget.ny, lockedTarget.nw, lockedTarget.nh) > 0.48f) {
                        c.id = lockedTarget.id;
                        c.name = lockedTarget.name;
                        c.hasDeepAi = (lockedTarget.deepGallery != null && !lockedTarget.deepGallery.isEmpty());
                        c.hasFaceAi = (lockedTarget.faceGallery != null && !lockedTarget.faceGallery.isEmpty());
                        usedIds.add(c.id);
                        updateOrInsertKalmanLocked(c, now, false);
                        break;
                    }
                }
            } else if (pendingRecoveryCandidate != null) {
                pendingRecoveryCandidate.id = lockedTarget.id;
                pendingRecoveryCandidate.name = lockedTarget.name;
                usedIds.add(lockedTarget.id);
            }
        }

        // 4. Kalan aktif Kalman takipçileri x kalan tespitler arasında GLOBAL OPTİMAL EŞLEŞTİRME (Hungarian!)
        List<SmartTrackerMath.KalmanTrackState> availTracks = new ArrayList<>();
        for (SmartTrackerMath.KalmanTrackState kt : kalmanTracks) {
            if (!usedIds.contains(kt.trackId)) availTracks.add(kt);
        }
        List<TrackedPerson> availDets = new ArrayList<>();
        for (TrackedPerson c : current) {
            if (c.id <= 0) availDets.add(c);
        }

        boolean multiPersonScene = (availTracks.size() > 1 || availDets.size() > 1 || current.size() > 1);
        List<TrackedPerson> needMemoryMatch = new ArrayList<>();

        if (!availTracks.isEmpty() && !availDets.isEmpty()) {
            int rows = availTracks.size();
            int cols = availDets.size();
            float[][] scoreMatrix = new float[rows][cols];

            for (int i = 0; i < rows; i++) {
                SmartTrackerMath.KalmanTrackState kt = availTracks.get(i);
                PersonMemoryStore.PersonProfile prof = findProfileByIdLocked(kt.trackId);
                float kNx = kt.nx(), kNy = kt.ny(), kNw = kt.w, kNh = kt.h;

                for (int j = 0; j < cols; j++) {
                    TrackedPerson det = availDets.get(j);
                    float iou = computeIoU(det.nx, det.ny, det.nw, det.nh, kNx, kNy, kNw, kNh);
                    float dist = (float) Math.hypot(det.centerX() - kt.cx, det.centerY() - kt.cy);
                    if (dist > 0.52f && iou < 0.01f) {
                        scoreMatrix[i][j] = -1f;
                        continue;
                    }

                    boolean legsCropped = (det.ny + det.nh > 0.92f);
                    float appScore = 0.60f;
                    float torsoScore = 0.60f;
                    if (det.signature != null) {
                        if (prof != null && !prof.sigGallery.isEmpty()) {
                            GalleryMatch gm = matchGallery(prof.sigGallery, det.signature, legsCropped);
                            appScore = gm.combined;
                            torsoScore = gm.torso;
                        } else if (kt.lastSig != null) {
                            appScore = reidSim(kt.lastSig, det.signature, legsCropped);
                            torsoScore = zoneSim(kt.lastSig, det.signature, 1);
                        }
                        // Çoklu kişi kesişmesinde güçlü görünüm koruması -> iki kişi geçerken kimlik takasını önler!
                        if (multiPersonScene && (appScore < 0.50f || torsoScore < 0.43f)) {
                            scoreMatrix[i][j] = -1f;
                            continue;
                        }
                    }

                    float posScore = iou * 1.9f + Math.max(0f, 0.36f - dist) * 2.1f;
                    float total = posScore + appScore * 3.0f + torsoScore * 1.1f;
                    scoreMatrix[i][j] = total;
                }
            }

            int[] assignment = SmartTrackerMath.solveMaxAssignment(scoreMatrix, 1.55f);
            boolean[] detMatched = new boolean[cols];
            for (int i = 0; i < rows; i++) {
                int j = assignment[i];
                SmartTrackerMath.KalmanTrackState kt = availTracks.get(i);
                if (j >= 0 && j < cols) {
                    TrackedPerson det = availDets.get(j);
                    det.id = kt.trackId;
                    det.name = kt.name;
                    det.hasDeepAi = kt.hasDeepAi;
                    det.hasFaceAi = kt.hasFaceAi;
                    usedIds.add(det.id);
                    detMatched[j] = true;
                    boolean crowded = isCrowdedDetection(det, current);
                    kt.update(det, now, crowded);
                } else {
                    kt.missedFrames++;
                }
            }
            for (int j = 0; j < cols; j++) {
                if (!detMatched[j]) needMemoryMatch.add(availDets.get(j));
            }
        } else {
            for (SmartTrackerMath.KalmanTrackState kt : availTracks) {
                kt.missedFrames++;
            }
            needMemoryMatch.addAll(availDets);
        }

        // 5. Eşleşmeyen yeni / tekrar giren tespitleri 15 dk oda + kalıcı 3-katmanlı hafızada ara
        for (TrackedPerson cur : needMemoryMatch) {
            PersonMemoryStore.PersonProfile bestMem = null;
            float bestSim = 0f;
            boolean legsCropped = (cur.ny + cur.nh > 0.92f);

            if (cur.signature != null) {
                for (PersonMemoryStore.PersonProfile mem : roomMemory) {
                    if (usedIds.contains(mem.id)) continue;
                    if (lockedTarget != null && lockedTarget.id == mem.id) continue;

                    GalleryMatch gm = matchGallery(mem.sigGallery, cur.signature, legsCropped);
                    if (gm.combined >= 0.64f && gm.torso >= 0.53f && gm.combined > bestSim) {
                        bestSim = gm.combined;
                        bestMem = mem;
                    }
                }
            }

            if (bestMem != null) {
                cur.id = bestMem.id;
                cur.name = bestMem.name;
                cur.hasDeepAi = !bestMem.deepBodyGallery.isEmpty();
                cur.hasFaceAi = !bestMem.faceGallery.isEmpty();
                usedIds.add(cur.id);
            } else {
                while (usedIds.contains(nextTrackId)
                        || (lockedTarget != null && lockedTarget.id == nextTrackId)
                        || findProfileByIdLocked(nextTrackId) != null) {
                    nextTrackId = (nextTrackId >= 999) ? 1 : nextTrackId + 1;
                }
                cur.id = nextTrackId++;
                if (nextTrackId > 999) nextTrackId = 1;
                usedIds.add(cur.id);
                // Yeni/tanınmayan kişiyi hemen OSNet + Yüz AI kuyruğuna öncelikli al!
                priorityIdentityCheckId = cur.id;
            }
            updateOrInsertKalmanLocked(cur, now, isCrowdedDetection(cur, current));
        }

        // 6. Keskin ve çakışmasız karelerle 129-D hafızasını besle
        for (TrackedPerson cur : current) {
            cur.lastSeen = now;
            PersonMemoryStore.PersonProfile existing = findProfileByIdLocked(cur.id);
            if (existing != null) {
                if (existing.isNamed()) cur.name = existing.name;
                cur.hasDeepAi = !existing.deepBodyGallery.isEmpty();
                cur.hasFaceAi = !existing.faceGallery.isEmpty();
            }

            if (cur.signature == null || cur == pendingRecoveryCandidate) continue;
            if (isHorizontallyEdgeClipped(cur.nx, cur.ny, cur.nw, cur.nh)) continue;
            if (isCrowdedDetection(cur, current)) continue;

            boolean sharpEnough = isAiCropSharpLocked(cur.nx, cur.ny, cur.nw, cur.nh)
                    || existing == null || existing.sigGallery.isEmpty();
            if (sharpEnough) {
                updateRoomMemoryLocked(cur.id, cur.signature, null, null, now, false);
            }
        }

        prevTracks.clear();
        prevTracks.addAll(current);
    }

    private boolean isCrowdedDetection(TrackedPerson cur, List<TrackedPerson> all) {
        for (TrackedPerson other : all) {
            if (other != cur && computeIoU(cur.nx, cur.ny, cur.nw, cur.nh, other.nx, other.ny, other.nw, other.nh) > 0.14f) {
                return true;
            }
        }
        return false;
    }

    private void updateOrInsertKalmanLocked(TrackedPerson p, long now, boolean crowded) {
        for (SmartTrackerMath.KalmanTrackState kt : kalmanTracks) {
            if (kt.trackId == p.id) {
                kt.update(p, now, crowded);
                return;
            }
        }
        kalmanTracks.add(new SmartTrackerMath.KalmanTrackState(p.id, p, now));
    }

    private boolean isAiCropSharpLocked(float nx, float ny, float nw, float nh) {
        if (!aiPixelsValid) return true;
        int x0 = Math.max(1, Math.round((nx + nw * 0.12f) * ANA_W));
        int x1 = Math.min(ANA_W - 2, Math.round((nx + nw * 0.88f) * ANA_W));
        int y0 = Math.max(1, Math.round((ny + nh * 0.08f) * ANA_H));
        int y1 = Math.min(ANA_H - 2, Math.round((ny + nh * 0.90f) * ANA_H));
        float sharp = SmartTrackerMath.computeLaplacianSharpness(aiPixels, ANA_W, ANA_H, x0, y0, x1, y1, 1);
        return sharp >= MIN_SIG_SHARPNESS;
    }

    private void updateRoomMemoryLocked(
            int id, float[] sig, float[] deepOsnet, float[] face, long now, boolean forceClothingAdapt) {
        PersonMemoryStore.PersonProfile target = findProfileByIdLocked(id);
        if (target == null) {
            if (roomMemory.size() >= PersonMemoryStore.MAX_PROFILES) {
                PersonMemoryStore.PersonProfile evict = null;
                for (PersonMemoryStore.PersonProfile m : roomMemory) {
                    boolean isLocked = (lockedTarget != null && lockedTarget.id == m.id);
                    if (isLocked || m.isPersistable()) continue;
                    if (evict == null || m.lastSeen < evict.lastSeen) evict = m;
                }
                if (evict != null) roomMemory.remove(evict);
            }
            target = new PersonMemoryStore.PersonProfile();
            target.id = id;
            target.firstSeen = now;
            target.lastSeen = now;
            if (sig != null) target.sigGallery.add(sig.clone());
            if (deepOsnet != null) target.deepBodyGallery.add(deepOsnet.clone());
            if (face != null) target.faceGallery.add(face.clone());
            roomMemory.add(target);
            return;
        }

        boolean changed = false;
        if (sig != null) {
            if (addOrMergeSignature(target.sigGallery, sig, PersonMemoryStore.MAX_SIG_PER_PROFILE, 0.87f, false, forceClothingAdapt)) {
                changed = true;
            }
        }
        if (deepOsnet != null) {
            if (addDiverseEmbedding(target.deepBodyGallery, deepOsnet, PersonMemoryStore.MAX_DEEP_PER_PROFILE, 0.88f)) {
                changed = true;
            }
        }
        if (face != null) {
            if (addDiverseEmbedding(target.faceGallery, face, PersonMemoryStore.MAX_FACE_PER_PROFILE, 0.90f)) {
                changed = true;
            }
        }
        target.lastSeen = now;
        if (changed && target.isPersistable()) {
            memoryDirty = true;
        }
    }

    private static boolean addDiverseEmbedding(List<float[]> gallery, float[] vec, int maxLen, float dupThreshold) {
        return SmartTrackerMath.addOrMergeEmbedding(gallery, vec, maxLen, dupThreshold);
    }

    /**
     * 129-D çok-açılı görünüm galerisi için çeşitlilik koruyan prototip birleştirme (FIFO unutma yok!).
     * Galeri dolduğunda birbirine en yakın iki açıyı (histogram + chroma + L2 yapısal desen) ortalayıp birleştirir;
     * böylece Ön / Sol / Sağ / Arka açıları saatler geçse de silinmez.
     */
    private boolean addOrMergeSignature(
            List<float[]> gallery, float[] sig, int maxLen, float dupThreshold,
            boolean preserveAnchorSlot0, boolean forceAdapt) {
        if (gallery == null || sig == null || sig.length != SIG_LEN || maxLen <= 0) return false;
        if (gallery.isEmpty()) {
            gallery.add(sig.clone());
            return true;
        }
        if (!forceAdapt) {
            float bestSim = 0f;
            for (float[] g : gallery) {
                if (g != null && g.length == SIG_LEN) {
                    bestSim = Math.max(bestSim, reidSim(g, sig, false));
                }
            }
            if (bestSim >= dupThreshold) {
                return false;
            }
        }
        if (gallery.size() < maxLen) {
            gallery.add(sig.clone());
            return true;
        }

        int startIdx = (preserveAnchorSlot0 && !forceAdapt && gallery.size() > 2) ? 1 : 0;
        int sz = gallery.size();
        int bestI = startIdx;
        int bestJ = Math.min(sz - 1, startIdx + 1);
        float maxPairSim = -1f;
        for (int i = startIdx; i < sz; i++) {
            float[] a = gallery.get(i);
            if (a == null || a.length != SIG_LEN) continue;
            for (int j = i + 1; j < sz; j++) {
                float[] b = gallery.get(j);
                if (b == null || b.length != SIG_LEN) continue;
                float s = reidSim(a, b, false);
                if (s > maxPairSim) {
                    maxPairSim = s;
                    bestI = i;
                    bestJ = j;
                }
            }
        }

        float[] a = gallery.get(bestI);
        float[] b = gallery.get(bestJ);
        if (a != null && b != null && a.length == SIG_LEN && b.length == SIG_LEN) {
            float[] merged = new float[SIG_LEN];
            // 3 bölge histogramı + 9 chroma ortalaması
            int patchStart = HIST_TOTAL + CHROMA_LEN;
            for (int k = 0; k < patchStart; k++) {
                merged[k] = (a[k] + b[k]) * 0.5f;
            }
            // 6x10 yapısal desen ortalaması + L2 normalizasyonu
            double normSq = 0.0;
            for (int k = patchStart; k < SIG_LEN; k++) {
                float m = (a[k] + b[k]) * 0.5f;
                merged[k] = m;
                normSq += (double) m * m;
            }
            if (normSq > 1e-8) {
                float inv = (float) (1.0 / Math.sqrt(normSq));
                for (int k = patchStart; k < SIG_LEN; k++) merged[k] *= inv;
            }
            gallery.set(bestI, merged);
        }
        gallery.set(bestJ, sig.clone());
        return true;
    }

    private void updateLockedTargetLocked(List<TrackedPerson> detections, long now, AnalysisOutput out) {
        LockedState t = lockedTarget;
        long dt = Math.max(1L, Math.min(350L, now - t.lastUpdate));

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
            boolean legsCropped = (d.ny + d.nh > 0.92f);

            GalleryMatch gm = matchGallery(t.gallery, d.signature, legsCropped);
            float anchorSim = (t.anchorSig != null) ? reidSim(t.anchorSig, d.signature, legsCropped) : gm.combined;
            float anchorTorso = (t.anchorSig != null) ? zoneSim(t.anchorSig, d.signature, 1) : gm.torso;

            float score;
            if (!t.lost) {
                // Aktif takipte arka dönüş / gölge / kısa gizlenmede kilit asla kolayca kopmasın
                if (gm.combined < 0.48f || Math.max(gm.torso, anchorTorso) < 0.42f) continue;
                if (sizeRatio < 0.14f) continue;
                score = gm.combined * 4.0f + gm.torso * 1.4f + iou * 1.8f
                        + aspectSim * 0.40f - (dist / Math.max(0.05f, diag)) * 0.45f + sizeRatio * 0.35f;
            } else {
                // Çıkıp giren kişiyi farklı ışık/konum/açı/duvar önünde kararlı geri yakala
                if (gm.combined < 0.61f || gm.torso < 0.49f) continue;
                if (sizeRatio < 0.07f || (!legsCropped && aspectSim < 0.28f)) continue;
                score = gm.combined * 4.6f + gm.torso * 1.7f + anchorSim * 0.9f + sizeRatio * 0.3f;
            }

            if (score > bestScore) {
                bestScore = score;
                best = d;
                bestIoU = iou;
                bestGallerySimMatched = gm.combined;
                bestAnchorSimMatched = anchorSim;
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

        if (t.lost) {
            t.isOccluded = false;
            if (best != null) {
                pendingRecoveryCandidate = best;
                t.recoverStreak++;
                if (t.recoverStreak >= 2 || bestGallerySimMatched >= 0.70f) {
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

        if (best != null) {
            t.isOccluded = conflicted;
            applyMatchedPosition(t, best, now, conflicted, bestIoU, bestGallerySimMatched, bestAnchorSimMatched);
            return;
        }

        // Kişi kısa süreli gizlendi (koltuk / kapı arkası / başka kişinin geçişi) -> akıllı ~1.8 sn tampon!
        long missingMs = now - t.lastSeen;
        boolean tplFound = (missingMs < 700L && t.template != null)
                && searchTemplateZeroAlloc(aiPixels, t.template, predX, predY, predW, predH, 0.70f, aiTplScratch, aiHitScratch);
        if (tplFound) {
            t.nx = Math.max(0f, Math.min(0.98f, aiHitScratch[0]));
            t.ny = Math.max(0f, Math.min(0.98f, aiHitScratch[1]));
            t.isOccluded = true;
            t.lastUpdate = now;
        } else if (missingMs < targetLostTimeoutMs) {
            // Hızı yumuşakça sönümle ki kutu ekran dışına fırlamasın; yanlış alarm çalmadan bekle!
            t.vx *= 0.80f;
            t.vy *= 0.80f;
            if (missingMs < 550L) {
                t.nx = Math.max(-0.08f, Math.min(0.95f, predX));
                t.ny = Math.max(-0.08f, Math.min(0.95f, predY));
            }
            t.isOccluded = true;
            t.lastUpdate = now;
        } else {
            t.lost = true;
            t.isOccluded = false;
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
        t.vx = SmartTrackerMath.clamp(t.vx * (1f - alpha) + ((newCx - prevCx) / dt) * alpha, -0.0025f, 0.0025f);
        t.vy = SmartTrackerMath.clamp(t.vy * (1f - alpha) + ((newCy - prevCy) / dt) * alpha, -0.0025f, 0.0025f);

        t.nx = best.nx;
        t.ny = best.ny;
        t.nw = best.nw;
        t.nh = best.nh;
        t.score = best.score;
        t.lastSeen = now;
        t.lastUpdate = now;

        // Keskin ve çakışmasız karelerde 16-örneklik çok-açılı 129-D hafızasını zenginleştir
        boolean allowAngleLearn = (bestIoU > 0.35f && anchorSim >= 0.45f) || (anchorSim >= 0.54f) || enrollActive;
        boolean sharp = isAiCropSharpLocked(best.nx, best.ny, best.nw, best.nh)
                || t.gallery == null || t.gallery.size() < 2;
        if (!conflicted && sharp && best.signature != null && best.score >= 0.30f
                && !isHorizontallyEdgeClipped(best.nx, best.ny, best.nw, best.nh)
                && allowAngleLearn) {
            if (t.gallery == null) t.gallery = new ArrayList<>(MAX_LOCKED_GALLERY);
            if (t.gallery.isEmpty() && t.anchorSig != null) t.gallery.add(t.anchorSig.clone());
            if (maxGallerySim < 0.87f) {
                addOrMergeSignature(t.gallery, best.signature, MAX_LOCKED_GALLERY, 0.87f, true, false);
            }
            updateRoomMemoryLocked(t.id, best.signature, null, null, now, false);

            float[] newTpl = extractTemplate(aiPixels, t.nx, t.ny, t.nw, t.nh);
            if (newTpl != null) t.template = newTpl;
        }
    }

    private boolean isHorizontallyEdgeClipped(float nx, float ny, float nw, float nh) {
        return nx < 0.015f || (nx + nw) > 0.985f || ny < 0.015f;
    }

    private static class GalleryMatch {
        float combined;
        float torso;
    }

    private float matchEmbeddingGallery(List<float[]> gallery, float[] cand) {
        return SmartTrackerMath.matchEmbeddingGalleryWithCentroid(gallery, cand, centroidScratch);
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
        gm.combined = (gallery.size() > 1 && best2 > 0f) ? (best1 * 0.76f + best2 * 0.24f) : best1;
        gm.torso = bestTorso;
        return gm;
    }

    // =================== ARKA PLAN MASKELİ 3-BÖLGELİ 129-D RE-ID + CHROMA + YAPISAL DESEN ===================
    private float[] computeSignature(int[] pixels, float nx, float ny, float nw, float nh, boolean useBgMask) {
        float[] sig = new float[SIG_LEN];
        return computeSignatureInto(pixels, nx, ny, nw, nh, sig, useBgMask) ? sig : null;
    }

    private boolean computeSignatureInto(
            int[] pixels, float nx, float ny, float nw, float nh, float[] sig, boolean useBgMask) {
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
        accumulateZoneHistAndChroma(pixels, x0, yHead0, x1, yHead1, centerX, halfW, sig, 0, HIST_TOTAL, useBgMask);
        accumulateZoneHistAndChroma(pixels, x0, yHead1, x1, yTorso1, centerX, halfW, sig, ZONE_LEN, HIST_TOTAL + 3, useBgMask);
        accumulateZoneHistAndChroma(pixels, x0, yTorso1, x1, yLegs1, centerX, halfW, sig, ZONE_LEN * 2, HIST_TOTAL + 6, useBgMask);
        extractStructuralPatch(pixels, nx + nw * 0.18f, ny + nh * 0.06f, nw * 0.64f, nh * 0.82f, sig, HIST_TOTAL + CHROMA_LEN, useBgMask);
        return true;
    }

    private void accumulateZoneHistAndChroma(
            int[] pixels, int x0, int y0, int x1, int y1,
            float cx, float halfW, float[] out, int histOffset, int chromaOffset, boolean useBgMask) {
        // Ön plan ağırlık oranını ön-kontrol et: kişi tam duvarla aynı renkse güvenli geri düşüş yap
        boolean applyMask = useBgMask;
        if (applyMask) {
            float fgMass = 0f;
            float baseMass = 0f;
            for (int py = y0; py < y1; py += 2) {
                int row = py * ANA_W;
                for (int px = x0; px < x1; px += 2) {
                    int idx = row + px;
                    int argb = pixels[idx];
                    fgMass += roomBgMask.getForegroundWeight(idx, (argb >> 16) & 0xFF, (argb >> 8) & 0xFF, argb & 0xFF);
                    baseMass += 1f;
                }
            }
            if (baseMass > 0f && (fgMass / baseMass) < 0.14f) {
                applyMask = false;
            }
        }

        float wSum = 0f;
        float rgAcc = 0f, ybAcc = 0f, lumAcc = 0f;
        for (int py = y0; py < y1; py++) {
            int row = py * ANA_W;
            for (int px = x0; px < x1; px++) {
                int idx = row + px;
                int argb = pixels[idx];
                int r = (argb >> 16) & 0xFF;
                int g = (argb >> 8) & 0xFF;
                int b = argb & 0xFF;

                int max = Math.max(r, Math.max(g, b));
                int min = Math.min(r, Math.min(g, b));
                float v = max / 255f;
                float s = (max == 0) ? 0f : (float) (max - min) / max;

                float ndx = (px + 0.5f - cx) / halfW;
                float spatialW = Math.max(0.25f, 1.0f - 0.65f * ndx * ndx);
                float fgW = applyMask ? roomBgMask.getForegroundWeight(idx, r, g, b) : 1.0f;
                float weight = spatialW * fgW;

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

    private void extractStructuralPatch(
            int[] pixels, float nx, float ny, float nw, float nh, float[] out, int offset, boolean useBgMask) {
        if (nw <= 0.005f || nh <= 0.01f) return;
        float weightedSum = 0f;
        float wTotal = 0f;
        float rawSum = 0f;
        float[] weights = useBgMask ? aiPatchWeights : null;
        int pIdx = 0;
        for (int py = 0; py < PATCH_H; py++) {
            int yIdx = Math.min(ANA_H - 1, Math.max(0, (int) Math.floor((ny + (py + 0.5f) / PATCH_H * nh) * ANA_H)));
            int row = yIdx * ANA_W;
            for (int px = 0; px < PATCH_W; px++) {
                int xIdx = Math.min(ANA_W - 1, Math.max(0, (int) Math.floor((nx + (px + 0.5f) / PATCH_W * nw) * ANA_W)));
                int idx = row + xIdx;
                int argb = pixels[idx];
                int r = (argb >> 16) & 0xFF;
                int g = (argb >> 8) & 0xFF;
                int b = argb & 0xFF;
                float lum = 0.299f * r + 0.587f * g + 0.114f * b;
                out[offset + pIdx] = lum;
                if (weights != null) {
                    float fgW = roomBgMask.getForegroundWeight(idx, r, g, b);
                    weights[pIdx] = fgW;
                    weightedSum += lum * fgW;
                    wTotal += fgW;
                }
                rawSum += lum;
                pIdx++;
            }
        }
        // Ön plan kütlesi çok düşükse (kişi tam duvarla aynı renkteyse) güvenli tam-ağırlık geri düşüşü yap
        boolean applyMask = (weights != null) && (wTotal >= PATCH_LEN * 0.16f);
        float mean = applyMask ? (weightedSum / wTotal) : (rawSum / PATCH_LEN);
        float normSq = 0f;
        for (int i = 0; i < PATCH_LEN; i++) {
            float centered = out[offset + i] - mean;
            if (applyMask) centered *= weights[i];
            out[offset + i] = centered;
            normSq += centered * centered;
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
        dst.isOccluded = src.isOccluded;
        dst.lost = src.lost;
    }

    public void close() {
        closed = true;
        identityExecutor.shutdownNow();
        try { identityExecutor.awaitTermination(350, TimeUnit.MILLISECONDS); } catch (InterruptedException ignored) {}

        synchronized (detectorLock) {
            synchronized (fastLock) {
                lastAiBitmap = null;
                aiPixelsValid = false;
                if (detBitmap != null && !detBitmap.isRecycled()) detBitmap.recycle();
                detBitmap = null;
                if (identityWorkBitmap != null && !identityWorkBitmap.isRecycled()) identityWorkBitmap.recycle();
                identityWorkBitmap = null;
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
