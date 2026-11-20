package com.cam.ai;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.PowerManager;
import android.util.Log;
import android.util.Size;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.NotificationCompat;
import androidx.core.app.ServiceCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.Lifecycle;
import androidx.lifecycle.LifecycleOwner;
import androidx.lifecycle.LifecycleRegistry;
import com.google.common.util.concurrent.ListenableFuture;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Ekran kapalı ve telefon kilitliyken dahi CameraX + 4 Modelli Yapay Zeka + Yerel Akış Sunucusunu
 * kesintisiz ve tamamen sessiz çalıştıran Ön Plan Kamera Servisi (Android 14/15/16/17 uyumlu).
 */
public class CameraMonitorService extends Service implements LifecycleOwner {

    private static final String TAG = "CameraMonitorService";
    public static final String ACTION_START = "com.cam.ai.ACTION_START_MONITOR";
    public static final String ACTION_STOP = "com.cam.ai.ACTION_STOP_MONITOR";
    public static final String EXTRA_ROOM_CODE = "room_code";

    private static final String CHANNEL_ID = "cam_ai_silent_monitor";
    private static final int NOTIF_ID = 2401;
    private static final int STREAM_PORT = 8080;
    private static final int MAX_AI_SIDE = 512;

    public interface UiCallback {
        void onSourceInfo(int w, int h, boolean isFront);
        void onFastFrame(List<TrackedPerson> persons, PersonDetectorEngine.LockedState locked, int fps, int inferMs);
        void onAiOutput(PersonDetectorEngine.AnalysisOutput out);
        void onTapLockResult(PersonDetectorEngine.LockedState snap, boolean locked, float sx, float sy);
        void onRemoteLockApplied(PersonDetectorEngine.LockedState snap);
        void onAlarmStateChanged(boolean alarmShowing);
        void onTorchChanged(boolean torchOn);
        void onEngineError(String error);
        void onServiceStopped();
    }

    public interface ServiceReadyListener {
        void onServiceReady(CameraMonitorService service);
    }

    private static volatile CameraMonitorService instance;
    private static volatile ServiceReadyListener readyListener;

    public static CameraMonitorService getInstance() {
        return instance;
    }

    public static boolean isRunning() {
        CameraMonitorService s = instance;
        return s != null && s.monitoringActive;
    }

    public static void setServiceReadyListener(ServiceReadyListener listener) {
        readyListener = listener;
        CameraMonitorService s = instance;
        if (listener != null && s != null && s.monitoringActive) {
            listener.onServiceReady(s);
        }
    }

    public static void startService(Context context, String roomCode) {
        Intent i = new Intent(context, CameraMonitorService.class);
        i.setAction(ACTION_START);
        i.putExtra(EXTRA_ROOM_CODE, roomCode);
        ContextCompat.startForegroundService(context, i);
    }

    public static void stopService(Context context) {
        Intent i = new Intent(context, CameraMonitorService.class);
        i.setAction(ACTION_STOP);
        try {
            context.startService(i);
        } catch (Exception e) {
            context.stopService(new Intent(context, CameraMonitorService.class));
        }
    }

    private final LifecycleRegistry lifecycleRegistry = new LifecycleRegistry(this);
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private PowerManager.WakeLock wakeLock;
    private WifiManager.WifiLock wifiLock;

    private ExecutorService cameraExecutor;
    private ExecutorService aiExecutor;
    private volatile PersonDetectorEngine personEngine;
    private volatile CameraStreamServer streamServer;

    private ProcessCameraProvider cameraProvider;
    private Camera activeCamera;
    private Preview previewUseCase;
    private ImageAnalysis imageAnalysisUseCase;
    private CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
    private volatile boolean isFrontCamera = false;
    private volatile boolean isTorchOn = false;
    private volatile boolean isAlarmShowing = false;
    private volatile boolean monitoringActive = false;
    private volatile String roomCode = "12345";

    private volatile UiCallback uiCallback;
    private volatile Preview.SurfaceProvider pendingSurfaceProvider;

    // Sıfır-GC kare işleme tamponları
    private int frameCounter = 0;
    private long lastFpsCalcTs = System.currentTimeMillis();
    private int framesInSecond = 0;
    private volatile int lastFps = 0;
    private volatile int lastInferTimeMs = 0;
    private volatile int lastFrameW = 480;
    private volatile int lastFrameH = 640;
    private volatile boolean isAnalyzing = false;
    private final AtomicBoolean aiBusy = new AtomicBoolean(false);
    private volatile List<TrackedPerson> latestAiPersons = Collections.emptyList();

    private Bitmap cachedRawBitmap = null;
    private Bitmap cachedUprightBitmap = null;
    private Canvas cachedUprightCanvas = null;
    private Bitmap aiWorkBitmap = null;
    private Canvas aiWorkCanvas = null;
    private final Matrix uprightMatrix = new Matrix();
    private final Paint uprightPaint = new Paint();
    private final ByteArrayOutputStream reusableJpegOut = new ByteArrayOutputStream(24576);

    @NonNull
    @Override
    public Lifecycle getLifecycle() {
        return lifecycleRegistry;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        instance = this;
        lifecycleRegistry.setCurrentState(Lifecycle.State.CREATED);
        cameraExecutor = Executors.newSingleThreadExecutor();
        aiExecutor = Executors.newSingleThreadExecutor();
        createSilentNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && ACTION_STOP.equals(intent.getAction())) {
            stopMonitoringAndSelf();
            return START_NOT_STICKY;
        }

        String requestedCode = (intent != null && intent.hasExtra(EXTRA_ROOM_CODE))
                ? intent.getStringExtra(EXTRA_ROOM_CODE)
                : roomCode;
        if (requestedCode == null || requestedCode.length() != 5) {
            requestedCode = "12345";
        }

        boolean codeChanged = !requestedCode.equals(roomCode);
        roomCode = requestedCode;

        // 1. Android 14/15/16/17 zorunlu ön plan kamera servisi bildirimi (TAMAMEN SESSİZ)
        Notification notif = buildSilentNotification(roomCode);
        try {
            int fgType = (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                    ? ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA
                    : 0;
            ServiceCompat.startForeground(this, NOTIF_ID, notif, fgType);
        } catch (Throwable t) {
            Log.e(TAG, "Ön plan kamera servisi başlatılamadı: " + t.getMessage());
            stopMonitoringAndSelf();
            return START_NOT_STICKY;
        }

        // 2. Ekran kapalıyken CPU ve Wi-Fi uykusunu engelle
        acquirePowerAndWifiLocks();

        if (!monitoringActive) {
            monitoringActive = true;
            lifecycleRegistry.setCurrentState(Lifecycle.State.RESUMED);
            initPersonEngineAsync();
            startStreamServer();
            startCameraX();
        } else if (codeChanged) {
            startStreamServer();
        }

        ServiceReadyListener listener = readyListener;
        if (listener != null) {
            mainHandler.post(() -> listener.onServiceReady(CameraMonitorService.this));
        }

        return START_NOT_STICKY;
    }

    private void createSilentNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel ch = new NotificationChannel(
                    CHANNEL_ID,
                    "Arka Plan Kamera Takibi",
                    NotificationManager.IMPORTANCE_LOW
            );
            ch.setDescription("Ekran kapalı ve telefon kilitliyken kamerayı sessizce çalışır tutar.");
            ch.setSound(null, null);
            ch.enableVibration(false);
            ch.setShowBadge(false);
            NotificationManager nm = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
            if (nm != null) nm.createNotificationChannel(ch);
        }
    }

    private Notification buildSilentNotification(String code) {
        Intent openIntent = new Intent(this, MainActivity.class);
        openIntent.setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        int piFlags = PendingIntent.FLAG_UPDATE_CURRENT
                | (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M ? PendingIntent.FLAG_IMMUTABLE : 0);
        PendingIntent contentPi = PendingIntent.getActivity(this, 0, openIntent, piFlags);

        Intent stopIntent = new Intent(this, CameraMonitorService.class);
        stopIntent.setAction(ACTION_STOP);
        PendingIntent stopPi = PendingIntent.getService(this, 1, stopIntent, piFlags);

        return new NotificationCompat.Builder(this, CHANNEL_ID)
                .setSmallIcon(R.mipmap.ic_launcher)
                .setContentTitle("CAM AI Aktif • Ekran Kapalı Çalışır")
                .setContentText("Oda: " + code + " • Tamamen sessiz takip")
                .setOngoing(true)
                .setSilent(true)
                .setPriority(NotificationCompat.PRIORITY_LOW)
                .setCategory(NotificationCompat.CATEGORY_SERVICE)
                .setContentIntent(contentPi)
                .addAction(0, "Durdur", stopPi)
                .build();
    }

    private void acquirePowerAndWifiLocks() {
        try {
            if (wakeLock == null) {
                PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
                if (pm != null) {
                    wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "CAM_AI:CameraMonitorWakeLock");
                    wakeLock.setReferenceCounted(false);
                }
            }
            if (wakeLock != null && !wakeLock.isHeld()) {
                wakeLock.acquire();
            }
        } catch (Exception e) {
            Log.w(TAG, "WakeLock alınamadı: " + e.getMessage());
        }

        try {
            if (wifiLock == null) {
                WifiManager wm = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
                if (wm != null) {
                    int mode = (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                            ? WifiManager.WIFI_MODE_FULL_LOW_LATENCY
                            : WifiManager.WIFI_MODE_FULL_HIGH_PERF;
                    wifiLock = wm.createWifiLock(mode, "CAM_AI:CameraWifiLock");
                    wifiLock.setReferenceCounted(false);
                }
            }
            if (wifiLock != null && !wifiLock.isHeld()) {
                wifiLock.acquire();
            }
        } catch (Exception e) {
            Log.w(TAG, "WifiLock alınamadı: " + e.getMessage());
        }
    }

    private void releasePowerAndWifiLocks() {
        try {
            if (wakeLock != null && wakeLock.isHeld()) wakeLock.release();
        } catch (Exception ignored) {}
        wakeLock = null;
        try {
            if (wifiLock != null && wifiLock.isHeld()) wifiLock.release();
        } catch (Exception ignored) {}
        wifiLock = null;
    }

    private void initPersonEngineAsync() {
        if (personEngine != null) return;
        aiExecutor.execute(() -> {
            if (!monitoringActive) return;
            PersonDetectorEngine engine = new PersonDetectorEngine(getApplicationContext());
            personEngine = engine;
            if (!engine.isInitialized()) {
                UiCallback cb = uiCallback;
                if (cb != null) {
                    mainHandler.post(() -> cb.onEngineError(engine.getInitError()));
                }
            }
        });
    }

    private void startStreamServer() {
        final String codeSnap = roomCode;
        cameraExecutor.execute(() -> {
            try {
                if (streamServer != null) streamServer.stop();
                CameraStreamServer server = new CameraStreamServer(STREAM_PORT, codeSnap);
                server.setOnRemoteCommandListener(new CameraStreamServer.OnRemoteCommandListener() {
                    @Override
                    public String onRemoteLock(float nx, float ny) {
                        PersonDetectorEngine eng = personEngine;
                        if (eng == null) return "EMPTY";
                        PersonDetectorEngine.LockedState snap = new PersonDetectorEngine.LockedState();
                        PersonDetectorEngine.LockStatus res = eng.tryLockFast(nx, ny, snap);
                        if (res != PersonDetectorEngine.LockStatus.LOCKED && aiExecutor != null) {
                            try {
                                res = aiExecutor.submit(() -> eng.tryLockAt(nx, ny, snap)).get(600, TimeUnit.MILLISECONDS);
                            } catch (Exception ignored) {}
                        }
                        if (res == PersonDetectorEngine.LockStatus.LOCKED) {
                            isAlarmShowing = false;
                            UiCallback cb = uiCallback;
                            if (cb != null) {
                                mainHandler.post(() -> {
                                    cb.onAlarmStateChanged(false);
                                    cb.onRemoteLockApplied(snap);
                                });
                            }
                            return "LOCKED";
                        }
                        return "EMPTY";
                    }

                    @Override
                    public void onRemoteUnlock() {
                        mainHandler.post(() -> unlockTarget());
                    }

                    @Override
                    public void onRemoteDismissAlarm() {
                        mainHandler.post(() -> dismissAlarm());
                    }

                    @Override
                    public void onRemoteFlipCamera() {
                        mainHandler.post(() -> flipCamera());
                    }

                    @Override
                    public void onRemoteToggleTorch() {
                        mainHandler.post(() -> toggleTorch());
                    }
                });
                server.start();
                streamServer = server;
            } catch (Exception e) {
                Log.e(TAG, "Stream sunucusu başlatılamadı: " + e.getMessage());
            }
        });
    }

    private void startCameraX() {
        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            if (!monitoringActive) return;
            try {
                cameraProvider = future.get();
                bindCameraUseCases();
            } catch (Exception e) {
                Log.e(TAG, "CameraX başlatılamadı: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCameraUseCases() {
        if (cameraProvider == null || !monitoringActive) return;
        cameraProvider.unbindAll();
        isTorchOn = false;
        notifyTorchUi();

        previewUseCase = new Preview.Builder().build();
        Preview.SurfaceProvider sp = pendingSurfaceProvider;
        if (sp != null) {
            previewUseCase.setSurfaceProvider(sp);
        }

        imageAnalysisUseCase = new ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setTargetResolution(new Size(480, 640))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();
        imageAnalysisUseCase.setAnalyzer(cameraExecutor, this::analyzeImageProxy);

        try {
            // KRİTİK: Servisin kendi RESUMED yaşam döngüsüne bağla -> ekran kapansa da kamera ASLA kapanmaz!
            activeCamera = cameraProvider.bindToLifecycle(this, cameraSelector, previewUseCase, imageAnalysisUseCase);
        } catch (Exception e) {
            Log.e(TAG, "Kamera yaşam döngüsüne bağlanamadı: " + e.getMessage());
        }
    }

    // =================== UI BAĞLAMA / AYIRMA (EKRAN KAPATIP AÇMADA SIFIR KESİNTİ) ===================
    public void attachUi(Preview.SurfaceProvider surfaceProvider, UiCallback callback) {
        this.uiCallback = callback;
        this.pendingSurfaceProvider = surfaceProvider;
        mainHandler.post(() -> {
            if (!monitoringActive) return;
            if (previewUseCase != null && surfaceProvider != null) {
                previewUseCase.setSurfaceProvider(surfaceProvider);
            }
            UiCallback cb = uiCallback;
            if (cb != null) {
                cb.onSourceInfo(lastFrameW, lastFrameH, isFrontCamera);
                cb.onTorchChanged(isTorchOn);
                cb.onAlarmStateChanged(isAlarmShowing);
                PersonDetectorEngine eng = personEngine;
                PersonDetectorEngine.LockedState locked = (eng != null) ? eng.getLockedStateSnapshot() : null;
                cb.onFastFrame(latestAiPersons, locked, lastFps, lastInferTimeMs);
            }
        });
    }

    public void detachUi() {
        this.uiCallback = null;
        this.pendingSurfaceProvider = null;
        mainHandler.post(() -> {
            // Sadece önizleme yüzeyini serbest bırak; kamera oturumunu ASLA kapatma!
            if (previewUseCase != null) {
                previewUseCase.setSurfaceProvider(null);
            }
        });
    }

    public void tapLockAt(float nx, float ny, float sx, float sy) {
        PersonDetectorEngine eng = personEngine;
        if (eng == null || cameraExecutor == null) return;
        cameraExecutor.execute(() -> {
            PersonDetectorEngine.LockedState fastSnap = new PersonDetectorEngine.LockedState();
            PersonDetectorEngine.LockStatus fastRes = eng.tryLockFast(nx, ny, fastSnap);
            if (fastRes == PersonDetectorEngine.LockStatus.LOCKED) {
                isAlarmShowing = false;
                dispatchTapResult(fastSnap, true, sx, sy);
            } else if (aiExecutor != null) {
                aiExecutor.execute(() -> {
                    PersonDetectorEngine.LockedState zoomSnap = new PersonDetectorEngine.LockedState();
                    PersonDetectorEngine.LockStatus zoomRes = eng.tryLockAt(nx, ny, zoomSnap);
                    boolean ok = (zoomRes == PersonDetectorEngine.LockStatus.LOCKED);
                    if (ok) isAlarmShowing = false;
                    dispatchTapResult(zoomSnap, ok, sx, sy);
                });
            }
        });
    }

    private void dispatchTapResult(PersonDetectorEngine.LockedState snap, boolean locked, float sx, float sy) {
        UiCallback cb = uiCallback;
        if (cb != null) {
            mainHandler.post(() -> {
                if (locked) cb.onAlarmStateChanged(false);
                cb.onTapLockResult(snap, locked, sx, sy);
            });
        }
    }

    public void unlockTarget() {
        PersonDetectorEngine eng = personEngine;
        if (eng != null) eng.unlock();
        isAlarmShowing = false;
        CameraStreamServer srv = streamServer;
        if (srv != null && srv.hasActiveViewer()) {
            srv.updateAiState(lastFrameW, lastFrameH, isFrontCamera, false, latestAiPersons, null);
        }
        UiCallback cb = uiCallback;
        if (cb != null) {
            mainHandler.post(() -> {
                cb.onAlarmStateChanged(false);
                cb.onFastFrame(latestAiPersons, null, lastFps, lastInferTimeMs);
            });
        }
    }

    public void dismissAlarm() {
        isAlarmShowing = false;
        CameraStreamServer srv = streamServer;
        PersonDetectorEngine eng = personEngine;
        if (srv != null && srv.hasActiveViewer()) {
            PersonDetectorEngine.LockedState snap = (eng != null) ? eng.getLockedStateSnapshot() : null;
            srv.updateAiState(lastFrameW, lastFrameH, isFrontCamera, false, latestAiPersons, snap);
        }
        UiCallback cb = uiCallback;
        if (cb != null) {
            mainHandler.post(() -> cb.onAlarmStateChanged(false));
        }
    }

    public void flipCamera() {
        isFrontCamera = !isFrontCamera;
        cameraSelector = isFrontCamera ? CameraSelector.DEFAULT_FRONT_CAMERA : CameraSelector.DEFAULT_BACK_CAMERA;
        unlockTarget();
        bindCameraUseCases();
    }

    public boolean toggleTorch() {
        if (activeCamera == null || isFrontCamera || !activeCamera.getCameraInfo().hasFlashUnit()) {
            return false;
        }
        isTorchOn = !isTorchOn;
        activeCamera.getCameraControl().enableTorch(isTorchOn);
        notifyTorchUi();
        return true;
    }

    private void notifyTorchUi() {
        UiCallback cb = uiCallback;
        if (cb != null) {
            final boolean t = isTorchOn;
            mainHandler.post(() -> cb.onTorchChanged(t));
        }
    }

    public PersonDetectorEngine getPersonEngine() {
        return personEngine;
    }

    public String getRoomCode() {
        return roomCode;
    }

    public boolean isFrontCamera() {
        return isFrontCamera;
    }

    public boolean isTorchOn() {
        return isTorchOn;
    }

    public boolean isAlarmShowing() {
        return isAlarmShowing;
    }

    public boolean isWakeLockHeld() {
        return wakeLock != null && wakeLock.isHeld();
    }

    public boolean isWifiLockHeld() {
        return wifiLock != null && wifiLock.isHeld();
    }

    // =================== 30 FPS KARE + ASENKRON 4-MODELLİ AI İŞLEME ===================
    private void analyzeImageProxy(@NonNull ImageProxy imageProxy) {
        PersonDetectorEngine eng = personEngine;
        if (!monitoringActive || isAnalyzing || eng == null) {
            imageProxy.close();
            return;
        }
        isAnalyzing = true;
        Bitmap tempRaw = null;
        try {
            int rawW = imageProxy.getWidth();
            int rawH = imageProxy.getHeight();
            int rotation = imageProxy.getImageInfo().getRotationDegrees();
            ImageProxy.PlaneProxy plane = imageProxy.getPlanes()[0];
            Bitmap rawSrc;
            if (plane.getPixelStride() == 4 && plane.getRowStride() == rawW * 4) {
                if (cachedRawBitmap == null || cachedRawBitmap.isRecycled()
                        || cachedRawBitmap.getWidth() != rawW || cachedRawBitmap.getHeight() != rawH) {
                    if (cachedRawBitmap != null && !cachedRawBitmap.isRecycled()) cachedRawBitmap.recycle();
                    cachedRawBitmap = Bitmap.createBitmap(rawW, rawH, Bitmap.Config.ARGB_8888);
                }
                ByteBuffer buf = plane.getBuffer();
                buf.rewind();
                cachedRawBitmap.copyPixelsFromBuffer(buf);
                rawSrc = cachedRawBitmap;
            } else {
                tempRaw = imageProxy.toBitmap();
                rawSrc = tempRaw;
            }
            imageProxy.close();

            Bitmap upright = drawIntoReusableUpright(rawSrc, rawW, rawH, rotation);
            if (tempRaw != null && !tempRaw.isRecycled()) {
                tempRaw.recycle();
                tempRaw = null;
            }

            int bw = upright.getWidth();
            int bh = upright.getHeight();
            lastFrameW = bw;
            lastFrameH = bh;

            // 1. Sadece PC veya izleyici aktif video istediğinde JPEG sıkıştır (PC oyun uykusundayken 0 JPEG!)
            frameCounter++;
            CameraStreamServer srv = streamServer;
            if (srv != null && srv.hasActiveVideoStream() && frameCounter % 2 == 0) {
                reusableJpegOut.reset();
                upright.compress(Bitmap.CompressFormat.JPEG, 55, reusableJpegOut);
                srv.updateFrame(reusableJpegOut.toByteArray());
            }

            // 2. 30 FPS Optik Takip (sıfır-GC)
            framesInSecond++;
            long now = System.currentTimeMillis();
            if (now - lastFpsCalcTs >= 1000) {
                lastFps = framesInSecond;
                framesInSecond = 0;
                lastFpsCalcTs = now;
            }

            PersonDetectorEngine.LockedState fastLocked = eng.fastTrackFrame(upright);
            UiCallback cb = uiCallback;
            if (cb != null) {
                final List<TrackedPerson> cachedPersons = latestAiPersons;
                final int shownFps = (lastFps > 0) ? lastFps : framesInSecond;
                final int inferMs = lastInferTimeMs;
                final boolean frontSnap = isFrontCamera;
                mainHandler.post(() -> {
                    UiCallback live = uiCallback;
                    if (live != null) {
                        live.onSourceInfo(bw, bh, frontSnap);
                        live.onFastFrame(cachedPersons, fastLocked, shownFps, inferMs);
                    }
                });
            }

            // 3. Arka plan AI iş parçacığı boşsa yeni kareyi asenkron teslim et (EKRAN KAPALIYKEN DE ÇALIŞIR!)
            if (eng.isInitialized() && aiBusy.compareAndSet(false, true)) {
                try {
                    if (aiWorkBitmap == null || aiWorkBitmap.isRecycled()
                            || aiWorkBitmap.getWidth() != bw || aiWorkBitmap.getHeight() != bh) {
                        if (aiWorkBitmap != null && !aiWorkBitmap.isRecycled()) aiWorkBitmap.recycle();
                        aiWorkBitmap = Bitmap.createBitmap(bw, bh, Bitmap.Config.ARGB_8888);
                        aiWorkCanvas = new Canvas(aiWorkBitmap);
                    }
                    aiWorkCanvas.drawBitmap(upright, 0f, 0f, null);

                    final int aiW = bw;
                    final int aiH = bh;
                    final boolean aiFront = isFrontCamera;
                    aiExecutor.execute(() -> {
                        try {
                            PersonDetectorEngine.AnalysisOutput result = eng.processFrame(aiWorkBitmap);
                            latestAiPersons = (result.persons != null) ? result.persons : Collections.emptyList();
                            lastInferTimeMs = result.inferTimeMs;

                            if (result.targetJustLost) {
                                isAlarmShowing = true;
                            } else if (result.targetJustRecovered) {
                                isAlarmShowing = false;
                            }

                            CameraStreamServer liveSrv = streamServer;
                            if (liveSrv != null && liveSrv.hasActiveViewer()) {
                                liveSrv.updateAiState(aiW, aiH, aiFront, isAlarmShowing, result.persons, result.locked);
                            }

                            UiCallback liveCb = uiCallback;
                            if (liveCb != null) {
                                mainHandler.post(() -> {
                                    UiCallback activeCb = uiCallback;
                                    if (activeCb != null) {
                                        activeCb.onAiOutput(result);
                                    }
                                });
                            }
                        } catch (Exception ignored) {
                        } finally {
                            aiBusy.set(false);
                        }
                    });
                } catch (Exception ex) {
                    aiBusy.set(false);
                }
            }
        } catch (Exception e) {
            try { imageProxy.close(); } catch (Exception ignored) {}
        } finally {
            if (tempRaw != null && !tempRaw.isRecycled()) tempRaw.recycle();
            isAnalyzing = false;
        }
    }

    private Bitmap drawIntoReusableUpright(Bitmap src, int rw, int rh, int rotationDegrees) {
        boolean rotated90or270 = (rotationDegrees == 90 || rotationDegrees == 270);
        int rotW = rotated90or270 ? rh : rw;
        int rotH = rotated90or270 ? rw : rh;
        int maxRotSide = Math.max(rotW, rotH);
        float scale = (maxRotSide > MAX_AI_SIDE) ? ((float) MAX_AI_SIDE / maxRotSide) : 1f;
        int uw = Math.max(1, Math.round(rotW * scale));
        int uh = Math.max(1, Math.round(rotH * scale));

        if (cachedUprightBitmap == null || cachedUprightBitmap.isRecycled()
                || cachedUprightBitmap.getWidth() != uw || cachedUprightBitmap.getHeight() != uh) {
            PersonDetectorEngine eng = personEngine;
            if (eng != null) eng.clearFrameCache();
            if (cachedUprightBitmap != null && !cachedUprightBitmap.isRecycled()) {
                cachedUprightBitmap.recycle();
            }
            cachedUprightBitmap = Bitmap.createBitmap(uw, uh, Bitmap.Config.ARGB_8888);
            cachedUprightCanvas = new Canvas(cachedUprightBitmap);
        }

        uprightMatrix.reset();
        uprightMatrix.postTranslate(-rw * 0.5f, -rh * 0.5f);
        if (rotationDegrees != 0) uprightMatrix.postRotate(rotationDegrees);
        if (scale != 1f) uprightMatrix.postScale(scale, scale);
        uprightMatrix.postTranslate(uw * 0.5f, uh * 0.5f);

        cachedUprightCanvas.drawBitmap(src, uprightMatrix, uprightPaint);
        return cachedUprightBitmap;
    }

    public void stopMonitoringAndSelf() {
        monitoringActive = false;
        isAlarmShowing = false;
        UiCallback cb = uiCallback;
        uiCallback = null;
        pendingSurfaceProvider = null;
        if (cb != null) {
            mainHandler.post(cb::onServiceStopped);
        }
        try {
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE);
        } catch (Exception ignored) {}
        stopSelf();
    }

    @Override
    public void onDestroy() {
        monitoringActive = false;
        lifecycleRegistry.setCurrentState(Lifecycle.State.DESTROYED);
        if (cameraProvider != null) {
            try { cameraProvider.unbindAll(); } catch (Exception ignored) {}
            cameraProvider = null;
        }
        activeCamera = null;
        previewUseCase = null;
        imageAnalysisUseCase = null;

        if (streamServer != null) {
            try { streamServer.stop(); } catch (Exception ignored) {}
            streamServer = null;
        }
        if (cameraExecutor != null) {
            cameraExecutor.shutdownNow();
            try { cameraExecutor.awaitTermination(250, TimeUnit.MILLISECONDS); } catch (InterruptedException ignored) {}
        }
        if (aiExecutor != null) {
            aiExecutor.shutdownNow();
            try { aiExecutor.awaitTermination(350, TimeUnit.MILLISECONDS); } catch (InterruptedException ignored) {}
        }
        if (personEngine != null) {
            personEngine.close();
            personEngine = null;
        }
        releasePowerAndWifiLocks();
        cachedRawBitmap = null;
        cachedUprightBitmap = null;
        aiWorkBitmap = null;
        if (instance == this) instance = null;
        super.onDestroy();
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
