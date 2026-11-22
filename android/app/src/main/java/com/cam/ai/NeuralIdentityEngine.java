package com.cam.ai;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.Log;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.Detection;
import com.google.mediapipe.tasks.components.containers.NormalizedKeypoint;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector;
import com.google.mediapipe.tasks.vision.facedetector.FaceDetectorResult;
import org.tensorflow.lite.Interpreter;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;

/**
 * 3-Katmanlı Asenkron Yapay Zeka Kimlik Motoru (v3.0 — Ana FPS'yi ASLA bloklamaz!):
 *  1. OSNet Person Re-ID (osnet_x0_25_msmt17.tflite — MSMT17 çoklu-kamera veri setiyle eğitilmiş gerçek İnsan Re-ID):
 *     - 128x256 vücut kırpımından 512-D L2-normalize kimlik gömmesi çıkarır.
 *     - Arka plan, duvar rengi, gölge ve açı değişimlerinde aynı kişiyi yüksek tutarlılıkla tanır;
 *       farklı insanları kesin ayırır.
 *     - Laplacian keskinlik kapısı ile hareket bulanık karelerin hafızayı kirletmesini engeller.
 *  2. Biyometrik Yüz AI (BlazeFace Short-Range + MobileFaceNet 192-D):
 *     - Göz hizalama + burun-göz ön-yüz simetri kontrolü + Laplacian keskinlik kapısı.
 *     - Kıyafet tamamen değişse veya uygulama yeniden açılsa bile kayıtlı kişiyi yüzünden anında tanır.
 */
public class NeuralIdentityEngine {

    private static final String TAG = "NeuralIdentityEngine";
    private static final String OSNET_REID_MODEL = "osnet_x0_25_msmt17.tflite";
    private static final String FACE_DETECTOR_MODEL = "blaze_face_short_range.tflite";
    private static final String FACE_EMBEDDER_MODEL = "mobilefacenet.tflite";

    // OSNet MSMT17 giriş boyutları ve ImageNet normalizasyon sabitleri
    private static final int REID_W = 128;
    private static final int REID_H = 256;
    private static final float IMAGENET_MEAN_R = 0.485f;
    private static final float IMAGENET_MEAN_G = 0.456f;
    private static final float IMAGENET_MEAN_B = 0.406f;
    private static final float IMAGENET_INV_STD_R = 1.0f / 0.229f;
    private static final float IMAGENET_INV_STD_G = 1.0f / 0.224f;
    private static final float IMAGENET_INV_STD_B = 1.0f / 0.225f;

    // Yüz AI boyutları ve kalite kapıları
    private static final int HEAD_CROP_SIZE = 224;
    private static final int FACE_INPUT_SIZE = 112;
    private static final float MIN_FACE_SCORE = 0.62f;
    private static final float MIN_FACE_PX = 24f;

    // Laplacian keskinlik eşikleri (bulanık karelerin hafızayı zehirlemesini engeller)
    public static final float MIN_BODY_SHARPNESS = 22.0f;
    public static final float MIN_FACE_SHARPNESS = 28.0f;

    public static class BodyReidResult {
        public float[] embedding;   // 512-D L2-normalize OSNet vektörü
        public float sharpness;     // Laplacian keskinlik varyansı
        public boolean isSharp;
    }

    public static class FaceResult {
        public float[] faceEmbedding; // 192-D L2-normalize MobileFaceNet vektörü
        public float faceScore;
        public float sharpness;       // Laplacian keskinlik varyansı
        public boolean isFrontal;
        public boolean isSharp;
    }

    public static class IdentityResult {
        public BodyReidResult body;
        public FaceResult face;
    }

    // OSNet Person Re-ID durumu
    private Interpreter reidInterpreter;
    private int reidEmbedDim = 512;
    private boolean reidInputNchw = false; // Tensor şeklinden otomatik algılanır
    private volatile boolean reidReady = false;

    // Face AI durumu
    private FaceDetector faceDetector;
    private Interpreter faceInterpreter;
    private int faceEmbedDim = 192;
    private volatile boolean faceReady = false;

    private volatile boolean closed = false;
    private final Object reidLock = new Object();
    private final Object faceLock = new Object();

    // OSNet sıfır-GC tamponları
    private final Bitmap reidCropBitmap = Bitmap.createBitmap(REID_W, REID_H, Bitmap.Config.ARGB_8888);
    private final Canvas reidCropCanvas = new Canvas(reidCropBitmap);
    private final int[] reidPixels = new int[REID_W * REID_H];
    private final ByteBuffer reidInputBuffer = ByteBuffer
            .allocateDirect(REID_W * REID_H * 3 * 4)
            .order(ByteOrder.nativeOrder());
    private ByteBuffer reidOutputBuffer = ByteBuffer
            .allocateDirect(512 * 4)
            .order(ByteOrder.nativeOrder());
    private float[] reidOutputFloats = new float[512];
    private final Rect reidSrcRect = new Rect();
    private final Rect reidDstRect = new Rect(0, 0, REID_W, REID_H);

    // Yüz AI sıfır-GC tamponları
    private final Bitmap headCropBitmap = Bitmap.createBitmap(HEAD_CROP_SIZE, HEAD_CROP_SIZE, Bitmap.Config.ARGB_8888);
    private final Canvas headCropCanvas = new Canvas(headCropBitmap);
    private final Bitmap faceInputBitmap = Bitmap.createBitmap(FACE_INPUT_SIZE, FACE_INPUT_SIZE, Bitmap.Config.ARGB_8888);
    private final Canvas faceInputCanvas = new Canvas(faceInputBitmap);
    private final int[] facePixels = new int[FACE_INPUT_SIZE * FACE_INPUT_SIZE];
    private final ByteBuffer faceInputBuffer = ByteBuffer
            .allocateDirect(FACE_INPUT_SIZE * FACE_INPUT_SIZE * 3 * 4)
            .order(ByteOrder.nativeOrder());
    private float[][] faceOutputBuf = new float[1][192];

    private final Paint bilinearPaint = new Paint(Paint.FILTER_BITMAP_FLAG);
    private final Rect faceSrcRect = new Rect();
    private final Rect faceDstRect = new Rect();
    private final Matrix alignMatrix = new Matrix();

    public NeuralIdentityEngine(Context context) {
        initReidPipeline(context);
        initFacePipeline(context);
    }

    private void initReidPipeline(Context context) {
        try {
            ByteBuffer modelBuf = loadAssetModelBuffer(context, OSNET_REID_MODEL);
            Interpreter.Options opts = new Interpreter.Options();
            opts.setNumThreads(2);
            opts.setUseXNNPACK(true);
            reidInterpreter = new Interpreter(modelBuf, opts);

            int[] inShape = reidInterpreter.getInputTensor(0).shape();
            // [1, 3, 256, 128] -> NCHW, [1, 256, 128, 3] -> NHWC
            reidInputNchw = (inShape != null && inShape.length == 4 && inShape[1] == 3);

            int[] outShape = reidInterpreter.getOutputTensor(0).shape();
            int totalElems = 1;
            if (outShape != null && outShape.length > 0) {
                for (int d : outShape) if (d > 0) totalElems *= d;
            }
            if (totalElems > 0) {
                reidEmbedDim = totalElems;
                reidOutputBuffer = ByteBuffer.allocateDirect(reidEmbedDim * 4).order(ByteOrder.nativeOrder());
                reidOutputFloats = new float[reidEmbedDim];
            }
            reidReady = true;
            Log.i(TAG, "Asenkron OSNet MSMT17 Person Re-ID hazır (" + reidEmbedDim + "-D, "
                    + (reidInputNchw ? "NCHW" : "NHWC") + ", DirectByteBuffer).");
        } catch (Throwable t) {
            reidReady = false;
            Log.w(TAG, "OSNet Re-ID başlatılamadı (maskeli 129-D ile devam): " + t.getMessage());
            closeQuietlyReid();
        }
    }

    private void initFacePipeline(Context context) {
        try {
            BaseOptions fdBase = BaseOptions.builder()
                    .setModelAssetPath(FACE_DETECTOR_MODEL)
                    .setDelegate(Delegate.CPU)
                    .build();
            FaceDetector.FaceDetectorOptions fdOpts = FaceDetector.FaceDetectorOptions.builder()
                    .setBaseOptions(fdBase)
                    .setRunningMode(RunningMode.IMAGE)
                    .setMinDetectionConfidence(0.56f)
                    .setMinSuppressionThreshold(0.30f)
                    .build();
            faceDetector = FaceDetector.createFromOptions(context, fdOpts);

            ByteBuffer modelBuf = loadAssetModelBuffer(context, FACE_EMBEDDER_MODEL);
            Interpreter.Options tfliteOpts = new Interpreter.Options();
            tfliteOpts.setNumThreads(2);
            tfliteOpts.setUseXNNPACK(true);
            faceInterpreter = new Interpreter(modelBuf, tfliteOpts);

            int[] outShape = faceInterpreter.getOutputTensor(0).shape();
            if (outShape != null && outShape.length >= 2 && outShape[1] > 0) {
                faceEmbedDim = outShape[1];
                faceOutputBuf = new float[1][faceEmbedDim];
            }
            faceReady = true;
            Log.i(TAG, "Asenkron BlazeFace + MobileFaceNet (" + faceEmbedDim + "-D) yüz motoru hazır.");
        } catch (Throwable t) {
            faceReady = false;
            Log.w(TAG, "Yüz tanıma motoru başlatılamadı: " + t.getMessage());
            closeQuietlyFace();
        }
    }

    private ByteBuffer loadAssetModelBuffer(Context context, String assetName) throws Exception {
        try (AssetFileDescriptor afd = context.getAssets().openFd(assetName);
             FileInputStream fis = new FileInputStream(afd.getFileDescriptor());
             FileChannel channel = fis.getChannel()) {
            MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, afd.getStartOffset(), afd.getDeclaredLength());
            return mapped;
        } catch (Exception ignored) {
            try (InputStream is = context.getAssets().open(assetName);
                 ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
                byte[] chunk = new byte[16384];
                int n;
                while ((n = is.read(chunk)) != -1) baos.write(chunk, 0, n);
                byte[] bytes = baos.toByteArray();
                ByteBuffer direct = ByteBuffer.allocateDirect(bytes.length).order(ByteOrder.nativeOrder());
                direct.put(bytes);
                direct.rewind();
                return direct;
            }
        }
    }

    public boolean isReidReady() {
        return reidReady && !closed;
    }

    public boolean isFaceReady() {
        return faceReady && !closed;
    }

    public int getReidEmbedDim() {
        return reidEmbedDim;
    }

    public int getFaceEmbedDim() {
        return faceEmbedDim;
    }

    public String getActiveEnginesSummary() {
        if (reidReady && faceReady) {
            return "EfficientDet-INT8 + Maskeli 129-D + OSNet 512-D + Yüz AI";
        } else if (reidReady) {
            return "EfficientDet-INT8 + Maskeli 129-D + OSNet 512-D";
        } else if (faceReady) {
            return "EfficientDet-INT8 + Maskeli 129-D + Yüz AI";
        }
        return "EfficientDet-INT8 + Maskeli 129-D Re-ID";
    }

    public IdentityResult extractIdentity(
            Bitmap fullBitmap, float nx, float ny, float nw, float nh,
            boolean runBodyReid, boolean runFace) {
        return extractIdentity(fullBitmap, nx, ny, nw, nh, runBodyReid, runFace, null, 0, 0);
    }

    public IdentityResult extractIdentity(
            Bitmap fullBitmap, float nx, float ny, float nw, float nh,
            boolean runBodyReid, boolean runFace,
            byte[] fgMask255, int maskW, int maskH) {
        IdentityResult res = new IdentityResult();
        if (closed || fullBitmap == null || fullBitmap.isRecycled()) return res;
        if (runBodyReid && reidReady) {
            res.body = extractBodyReid(fullBitmap, nx, ny, nw, nh, fgMask255, maskW, maskH);
        }
        if (runFace && faceReady) {
            res.face = extractFace(fullBitmap, nx, ny, nw, nh);
        }
        return res;
    }

    /**
     * OSNet MSMT17 ile 512-D derin insan Re-ID vektörü ve keskinlik puanı çıkarır.
     */
    public BodyReidResult extractBodyReid(Bitmap fullBitmap, float nx, float ny, float nw, float nh) {
        return extractBodyReid(fullBitmap, nx, ny, nw, nh, null, 0, 0);
    }

    /**
     * Arka plan maskesi destekli OSNet MSMT17 çıkarımı:
     * Statik duvar/koltuk piksellerini yumuşakça ImageNet nötr ortalamasına (R=124, G=116, B=104 -> 0.0)
     * yaklaştırarak kişinin farklı renkli duvar önüne geçtiğinde dahi %100 tanınmasını sağlar.
     */
    public BodyReidResult extractBodyReid(
            Bitmap fullBitmap, float nx, float ny, float nw, float nh,
            byte[] fgMask255, int maskW, int maskH) {
        BodyReidResult out = new BodyReidResult();
        if (closed || !reidReady || fullBitmap == null || fullBitmap.isRecycled()) return out;

        synchronized (reidLock) {
            if (closed || !reidReady || reidInterpreter == null || fullBitmap.isRecycled()) return out;
            try {
                int bw = fullBitmap.getWidth();
                int bh = fullBitmap.getHeight();
                // Kişi kutusunun kenarındaki duvarı biraz dışarıda bırakmak için hafifçe içe sıkıştır (%3 yatay)
                float insetX = nw * 0.03f;
                int x0 = Math.max(0, Math.min(bw - 8, Math.round((nx + insetX) * bw)));
                int x1 = Math.max(x0 + 8, Math.min(bw, Math.round((nx + nw - insetX) * bw)));
                int y0 = Math.max(0, Math.min(bh - 12, Math.round(ny * bh)));
                int y1 = Math.max(y0 + 12, Math.min(bh, Math.round((ny + nh) * bh)));
                if (x1 - x0 < 14 || y1 - y0 < 24) return out;

                reidSrcRect.set(x0, y0, x1, y1);
                reidCropCanvas.drawBitmap(fullBitmap, reidSrcRect, reidDstRect, bilinearPaint);
                reidCropBitmap.getPixels(reidPixels, 0, REID_W, 0, 0, REID_W, REID_H);

                // Laplacian keskinlik kontrolü (<0.04 ms) — maskelemeden önce gerçek görüntü üzerinde!
                out.sharpness = SmartTrackerMath.computeLaplacianSharpness(
                        reidPixels, REID_W, REID_H, 10, 12, REID_W - 10, REID_H - 12, 2);
                out.isSharp = (out.sharpness >= MIN_BODY_SHARPNESS);

                // Uyarlamalı oda arka plan maskesi geçerliyse yalnızca ön plan oranı sağlıklıysa yumuşak uygula
                boolean applyFgMask = false;
                float cropNormX0 = (float) x0 / bw;
                float cropNormW = (float) (x1 - x0) / bw;
                float cropNormY0 = (float) y0 / bh;
                float cropNormH = (float) (y1 - y0) / bh;
                if (fgMask255 != null && maskW > 0 && maskH > 0 && fgMask255.length >= maskW * maskH) {
                    float fgSum = 0f;
                    int sampleCnt = 0;
                    for (int ry = 0; ry < REID_H; ry += 8) {
                        int my = Math.min(maskH - 1, Math.max(0, (int) ((cropNormY0 + (ry + 0.5f) / REID_H * cropNormH) * maskH)));
                        int mRow = my * maskW;
                        for (int rx = 0; rx < REID_W; rx += 8) {
                            int mx = Math.min(maskW - 1, Math.max(0, (int) ((cropNormX0 + (rx + 0.5f) / REID_W * cropNormW) * maskW)));
                            fgSum += (fgMask255[mRow + mx] & 0xFF) * (1f / 255f);
                            sampleCnt++;
                        }
                    }
                    float avgFg = (sampleCnt > 0) ? (fgSum / sampleCnt) : 1f;
                    // En az %16 ön plan ve en az %6 arka plan pikseli varsa duvarı yumuşakça nötrle
                    applyFgMask = (avgFg >= 0.16f && avgFg <= 0.94f);
                }

                reidInputBuffer.rewind();
                if (reidInputNchw) {
                    int planeSize = REID_W * REID_H;
                    for (int ch = 0; ch < 3; ch++) {
                        float mean = (ch == 0) ? IMAGENET_MEAN_R : (ch == 1 ? IMAGENET_MEAN_G : IMAGENET_MEAN_B);
                        float invStd = (ch == 0) ? IMAGENET_INV_STD_R : (ch == 1 ? IMAGENET_INV_STD_G : IMAGENET_INV_STD_B);
                        int shift = (ch == 0) ? 16 : (ch == 1 ? 8 : 0);
                        for (int ry = 0; ry < REID_H; ry++) {
                            int rowOff = ry * REID_W;
                            int mRow = applyFgMask
                                    ? Math.min(maskH - 1, Math.max(0, (int) ((cropNormY0 + (ry + 0.5f) / REID_H * cropNormH) * maskH))) * maskW
                                    : 0;
                            for (int rx = 0; rx < REID_W; rx++) {
                                float v = ((reidPixels[rowOff + rx] >> shift) & 0xFF) * (1f / 255f);
                                float norm = (v - mean) * invStd;
                                if (applyFgMask) {
                                    int mx = Math.min(maskW - 1, Math.max(0, (int) ((cropNormX0 + (rx + 0.5f) / REID_W * cropNormW) * maskW)));
                                    float fg = (fgMask255[mRow + mx] & 0xFF) * (1f / 255f);
                                    // Statik duvarı nötr 0.0 ortalamaya doğru yumuşakça bastır (0.22..1.0)
                                    norm *= (0.22f + 0.78f * fg);
                                }
                                reidInputBuffer.putFloat(norm);
                            }
                        }
                    }
                } else {
                    for (int ry = 0; ry < REID_H; ry++) {
                        int rowOff = ry * REID_W;
                        int mRow = applyFgMask
                                ? Math.min(maskH - 1, Math.max(0, (int) ((cropNormY0 + (ry + 0.5f) / REID_H * cropNormH) * maskH))) * maskW
                                : 0;
                        for (int rx = 0; rx < REID_W; rx++) {
                            int argb = reidPixels[rowOff + rx];
                            float nr = (((argb >> 16) & 0xFF) * (1f / 255f) - IMAGENET_MEAN_R) * IMAGENET_INV_STD_R;
                            float ng = (((argb >> 8) & 0xFF) * (1f / 255f) - IMAGENET_MEAN_G) * IMAGENET_INV_STD_G;
                            float nb = ((argb & 0xFF) * (1f / 255f) - IMAGENET_MEAN_B) * IMAGENET_INV_STD_B;
                            if (applyFgMask) {
                                int mx = Math.min(maskW - 1, Math.max(0, (int) ((cropNormX0 + (rx + 0.5f) / REID_W * cropNormW) * maskW)));
                                float fg = (fgMask255[mRow + mx] & 0xFF) * (1f / 255f);
                                float scale = 0.22f + 0.78f * fg;
                                nr *= scale;
                                ng *= scale;
                                nb *= scale;
                            }
                            reidInputBuffer.putFloat(nr);
                            reidInputBuffer.putFloat(ng);
                            reidInputBuffer.putFloat(nb);
                        }
                    }
                }
                reidInputBuffer.rewind();

                reidOutputBuffer.rewind();
                reidInterpreter.run(reidInputBuffer, reidOutputBuffer);
                reidOutputBuffer.rewind();
                reidOutputBuffer.asFloatBuffer().get(reidOutputFloats);
                out.embedding = l2NormalizeCopy(reidOutputFloats);
            } catch (Throwable t) {
                Log.w(TAG, "OSNet Re-ID hatası: " + t.getMessage());
            }
        }
        return out;
    }

    public FaceResult extractFace(Bitmap fullBitmap, float nx, float ny, float nw, float nh) {
        FaceResult out = new FaceResult();
        if (closed || !faceReady || fullBitmap == null || fullBitmap.isRecycled()) return out;

        synchronized (faceLock) {
            if (closed || !faceReady || faceDetector == null || faceInterpreter == null || fullBitmap.isRecycled()) {
                return out;
            }
            extractFaceLocked(fullBitmap, nx, ny, nw, nh, out);
        }
        return out;
    }

    private void extractFaceLocked(Bitmap full, float nx, float ny, float nw, float nh, FaceResult out) {
        int bw = full.getWidth();
        int bh = full.getHeight();

        float aspect = nw / Math.max(0.01f, nh);
        float headFracH = (aspect > 0.65f) ? 0.78f : 0.54f;

        float hx0f = Math.max(0f, nx - nw * 0.10f);
        float hx1f = Math.min(1f, nx + nw * 1.10f);
        float hy0f = Math.max(0f, ny - nh * 0.06f);
        float hy1f = Math.min(1f, ny + nh * headFracH);

        int hx0 = Math.max(0, Math.min(bw - 8, Math.round(hx0f * bw)));
        int hx1 = Math.max(hx0 + 8, Math.min(bw, Math.round(hx1f * bw)));
        int hy0 = Math.max(0, Math.min(bh - 8, Math.round(hy0f * bh)));
        int hy1 = Math.max(hy0 + 8, Math.min(bh, Math.round(hy1f * bh)));
        int cropW = hx1 - hx0;
        int cropH = hy1 - hy0;
        if (cropW < 24 || cropH < 24) return;

        float scale = Math.min((float) HEAD_CROP_SIZE / cropW, (float) HEAD_CROP_SIZE / cropH);
        int drawW = Math.max(1, Math.round(cropW * scale));
        int drawH = Math.max(1, Math.round(cropH * scale));
        int padX = (HEAD_CROP_SIZE - drawW) / 2;
        int padY = (HEAD_CROP_SIZE - drawH) / 2;

        headCropBitmap.eraseColor(Color.BLACK);
        faceSrcRect.set(hx0, hy0, hx1, hy1);
        faceDstRect.set(padX, padY, padX + drawW, padY + drawH);
        headCropCanvas.drawBitmap(full, faceSrcRect, faceDstRect, bilinearPaint);

        try {
            // NOT: headCropBitmap yeniden kullanıldığı için headMp.close() çağrılmaz (Bitmap.recycle önlenir).
            MPImage headMp = new BitmapImageBuilder(headCropBitmap).build();
            FaceDetectorResult fdRes = faceDetector.detect(headMp);
            if (fdRes == null || fdRes.detections() == null || fdRes.detections().isEmpty()) {
                return;
            }

            float personLeftPx = nx * bw;
            float personRightPx = (nx + nw) * bw;
            float personMarginPx = nw * bw * 0.10f;

            Detection bestDet = null;
            float bestScore = MIN_FACE_SCORE;
            RectF bestFullBox = null;

            for (Detection det : fdRes.detections()) {
                if (det.categories() == null || det.categories().isEmpty()) continue;
                float score = det.categories().get(0).score();
                if (score < bestScore) continue;

                RectF bb = det.boundingBox();
                float fx0 = hx0 + (bb.left - padX) / scale;
                float fy0 = hy0 + (bb.top - padY) / scale;
                float fx1 = hx0 + (bb.right - padX) / scale;
                float fy1 = hy0 + (bb.bottom - padY) / scale;
                float fw = fx1 - fx0;
                float fh = fy1 - fy0;
                if (fw < MIN_FACE_PX || fh < MIN_FACE_PX) continue;

                float fcx = (fx0 + fx1) * 0.5f;
                if (fcx < personLeftPx - personMarginPx || fcx > personRightPx + personMarginPx) {
                    continue;
                }

                bestScore = score;
                bestDet = det;
                bestFullBox = new RectF(fx0, fy0, fx1, fy1);
            }

            if (bestDet == null || bestFullBox == null) return;

            float angleDeg = 0f;
            boolean frontal = false;
            if (bestDet.keypoints().isPresent()) {
                List<NormalizedKeypoint> kps = bestDet.keypoints().get();
                if (kps.size() >= 3) {
                    NormalizedKeypoint e0 = kps.get(0);
                    NormalizedKeypoint e1 = kps.get(1);
                    NormalizedKeypoint nose = kps.get(2);
                    float xA = e0.x() * HEAD_CROP_SIZE;
                    float yA = e0.y() * HEAD_CROP_SIZE;
                    float xB = e1.x() * HEAD_CROP_SIZE;
                    float yB = e1.y() * HEAD_CROP_SIZE;
                    float dx = (xB >= xA) ? (xB - xA) : (xA - xB);
                    float dy = (xB >= xA) ? (yB - yA) : (yA - yB);
                    if (dx > 8f) {
                        float deg = (float) Math.toDegrees(Math.atan2(dy, dx));
                        if (Math.abs(deg) <= 35f) angleDeg = deg;
                        float minEyeX = Math.min(e0.x(), e1.x());
                        float maxEyeX = Math.max(e0.x(), e1.x());
                        float eyeSpan = Math.max(1e-4f, maxEyeX - minEyeX);
                        float noseRatio = (nose.x() - minEyeX) / eyeSpan;
                        frontal = (noseRatio >= 0.18f && noseRatio <= 0.82f);
                    }
                }
            }

            float fcx = bestFullBox.centerX();
            float fcy = bestFullBox.centerY();
            float side = Math.max(bestFullBox.width(), bestFullBox.height()) * 1.24f;
            float faceScale = FACE_INPUT_SIZE / Math.max(1f, side);

            faceInputBitmap.eraseColor(Color.rgb(128, 128, 128));
            alignMatrix.reset();
            alignMatrix.postTranslate(-fcx, -fcy);
            if (Math.abs(angleDeg) > 2.0f) {
                alignMatrix.postRotate(-angleDeg);
            }
            alignMatrix.postScale(faceScale, faceScale);
            alignMatrix.postTranslate(FACE_INPUT_SIZE * 0.5f, FACE_INPUT_SIZE * 0.5f);
            faceInputCanvas.drawBitmap(full, alignMatrix, bilinearPaint);

            faceInputBitmap.getPixels(facePixels, 0, FACE_INPUT_SIZE, 0, 0, FACE_INPUT_SIZE, FACE_INPUT_SIZE);

            // Yüz keskinlik kapısı: hareket bulanık yüzler hafızaya eklenmez!
            out.sharpness = SmartTrackerMath.computeLaplacianSharpness(
                    facePixels, FACE_INPUT_SIZE, FACE_INPUT_SIZE, 14, 14, FACE_INPUT_SIZE - 14, FACE_INPUT_SIZE - 14, 2);
            out.isSharp = (out.sharpness >= MIN_FACE_SHARPNESS);

            faceInputBuffer.rewind();
            for (int i = 0; i < facePixels.length; i++) {
                int argb = facePixels[i];
                faceInputBuffer.putFloat((((argb >> 16) & 0xFF) - 128f) / 128f);
                faceInputBuffer.putFloat((((argb >> 8) & 0xFF) - 128f) / 128f);
                faceInputBuffer.putFloat(((argb & 0xFF) - 128f) / 128f);
            }
            faceInputBuffer.rewind();

            faceInterpreter.run(faceInputBuffer, faceOutputBuf);
            float[] normFace = l2NormalizeCopy(faceOutputBuf[0]);
            if (normFace != null) {
                out.faceEmbedding = normFace;
                out.faceScore = bestScore;
                out.isFrontal = frontal;
            }
        } catch (Throwable t) {
            Log.w(TAG, "Face embedding hatası: " + t.getMessage());
        }
    }

    public static float[] l2NormalizeCopy(float[] vec) {
        if (vec == null || vec.length == 0) return null;
        double sumSq = 0.0;
        for (float v : vec) sumSq += (double) v * (double) v;
        if (sumSq < 1e-10) return null;
        float invNorm = (float) (1.0 / Math.sqrt(sumSq));
        float[] out = new float[vec.length];
        for (int i = 0; i < vec.length; i++) {
            out[i] = vec[i] * invNorm;
        }
        return out;
    }

    public static float cosineSim(float[] a, float[] b) {
        if (a == null || b == null || a.length != b.length || a.length == 0) return 0f;
        float dot = 0f;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
        }
        return Math.max(-1f, Math.min(1f, dot));
    }

    private void closeQuietlyReid() {
        try { if (reidInterpreter != null) reidInterpreter.close(); } catch (Exception ignored) {}
        reidInterpreter = null;
    }

    private void closeQuietlyFace() {
        try { if (faceDetector != null) faceDetector.close(); } catch (Exception ignored) {}
        try { if (faceInterpreter != null) faceInterpreter.close(); } catch (Exception ignored) {}
        faceDetector = null;
        faceInterpreter = null;
    }

    public void close() {
        closed = true;
        synchronized (reidLock) {
            closeQuietlyReid();
            if (!reidCropBitmap.isRecycled()) reidCropBitmap.recycle();
        }
        synchronized (faceLock) {
            closeQuietlyFace();
            if (!headCropBitmap.isRecycled()) headCropBitmap.recycle();
            if (!faceInputBitmap.isRecycled()) faceInputBitmap.recycle();
        }
    }
}
