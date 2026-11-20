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
 * Hafif & Asenkron Biyometrik Yüz Tanıma Motoru (v2.5):
 *  - MediaPipe BlazeFace Short-Range (229 KB) ile yüksek güvenli ön yüz & göz hizalama tespiti.
 *  - Google LiteRT MobileFaceNet (192-D L2-normalize yüz vektörü).
 *
 * Ana SSD insan tespit thread'ini ASLA bloklamaz; ayrı arka plan thread'inde çalışarak
 * kıyafet değişiminde ve isimli kalıcı hafızada ekstra doğruluk sağlar.
 */
public class NeuralIdentityEngine {

    private static final String TAG = "NeuralIdentityEngine";
    private static final String FACE_DETECTOR_MODEL = "blaze_face_short_range.tflite";
    private static final String FACE_EMBEDDER_MODEL = "mobilefacenet.tflite";

    private static final int HEAD_CROP_SIZE = 224;
    private static final int FACE_INPUT_SIZE = 112;
    private static final float MIN_FACE_SCORE = 0.62f; // Yalnızca net yüzler (bulanık yan profiller hafızayı kirletmez)
    private static final float MIN_FACE_PX = 24f;

    public static class FaceResult {
        public float[] faceEmbedding; // 192-D unit-norm MobileFaceNet
        public float faceScore;
        public boolean isFrontal;
    }

    private FaceDetector faceDetector;
    private Interpreter faceInterpreter;
    private int faceEmbedDim = 192;

    private volatile boolean faceReady = false;
    private volatile boolean closed = false;

    private final Object faceLock = new Object();

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
    private final Rect srcRect = new Rect();
    private final Rect dstRect = new Rect();
    private final Matrix alignMatrix = new Matrix();

    public NeuralIdentityEngine(Context context) {
        initFacePipeline(context);
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
            // Arka plan yüz motoru ana SSD'den CPU çalmasın diye 2 iş parçacığı ile sınırlı
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
            Log.w(TAG, "Yüz tanıma motoru başlatılamadı (hızlı Re-ID ile devam): " + t.getMessage());
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

    public boolean isFaceReady() {
        return faceReady && !closed;
    }

    public int getFaceEmbedDim() {
        return faceEmbedDim;
    }

    public String getActiveEnginesSummary() {
        return faceReady
                ? "Hızlı SSD-300 + 129-D Re-ID + Asenkron Yüz AI"
                : "Hızlı SSD-300 + 129-D Çok-İpuculu Re-ID";
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
        srcRect.set(hx0, hy0, hx1, hy1);
        dstRect.set(padX, padY, padX + drawW, padY + drawH);
        headCropCanvas.drawBitmap(full, srcRect, dstRect, bilinearPaint);

        try {
            // NOT: headCropBitmap yeniden kullanıldığı için headMp.close() çağrılmaz.
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
                        // Burun iki göz arasında simetrikse gerçekten ön yüzdür (yan profil değil!)
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

    private void closeQuietlyFace() {
        try { if (faceDetector != null) faceDetector.close(); } catch (Exception ignored) {}
        try { if (faceInterpreter != null) faceInterpreter.close(); } catch (Exception ignored) {}
        faceDetector = null;
        faceInterpreter = null;
    }

    public void close() {
        closed = true;
        synchronized (faceLock) {
            closeQuietlyFace();
            if (!headCropBitmap.isRecycled()) headCropBitmap.recycle();
            if (!faceInputBitmap.isRecycled()) faceInputBitmap.recycle();
        }
    }
}
