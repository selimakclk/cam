package com.cam.ai;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.InputType;
import android.util.Size;
import android.view.View;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import androidx.activity.OnBackPressedCallback;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.google.common.util.concurrent.ListenableFuture;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

public class MainActivity extends AppCompatActivity {

    private static final int PERMISSION_REQ_CODE = 1001;
    private static final String STEALTH_PIN = "7693";
    private static final int STREAM_PORT = 8080;

    // Ana Konteynerler
    private FrameLayout cameraContainer;
    private FrameLayout viewerContainer;
    private LinearLayout roleContainer;
    private FrameLayout stealthScreen;
    private LinearLayout stealthPinPanel;
    private LinearLayout alarmOverlay;

    // Kamera HUD Elemanları
    private PreviewView previewView;
    private OverlayView overlayView;
    private TextView txtStatus;
    private TextView txtIpBadge;
    private TextView txtTargetStatus;
    private TextView txtWarningBanner;
    private TextView txtCodeDisplay;
    private TextView txtFpsDisplay;
    private Button btnUnlock;
    private Button btnNameTarget;
    private Button btnMemory;
    private Button btnTorch;

    // İzleyici HUD Elemanları
    private ImageView viewerStream;
    private OverlayView viewerOverlay;
    private TextView txtViewerStatus;
    private TextView txtViewerTarget;
    private Button btnViewerUnlock;

    // Giriş & PIN Elemanları
    private EditText editPassword;
    private TextView txtPinDots;

    // Servisler ve Motorlar
    private ProcessCameraProvider cameraProvider;
    private Camera activeCamera;
    private CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
    private boolean isFrontCamera = false;
    private boolean isTorchOn = false;

    private ExecutorService cameraExecutor;
    private ExecutorService aiExecutor;
    private volatile PersonDetectorEngine personEngine;
    private SirenAudioEngine sirenEngine;
    private CameraStreamServer streamServer;
    private ViewerStreamClient viewerClient;
    private GitHubUpdater githubUpdater;
    private Button btnCheckUpdate;

    private final Handler uiHandler = new Handler(Looper.getMainLooper());
    private String roomCode = "12345";
    private boolean pendingStartCameraAfterPerm = false;
    private boolean isAlarmShowing = false;
    private boolean isStealthActive = false;
    private StringBuilder pinInput = new StringBuilder();

    // Performans / Stream sayaçları & çift-thread sıfır-GC tamponları
    private static final int MAX_AI_SIDE = 512;
    private int frameCounter = 0;
    private long lastFpsCalcTs = System.currentTimeMillis();
    private long lastFpsTextTs = 0L;
    private int framesInSecond = 0;
    private int lastFps = 0;
    private volatile int lastInferTimeMs = 0;
    private volatile boolean isAnalyzing = false;
    private final AtomicBoolean aiBusy = new AtomicBoolean(false);
    private volatile List<TrackedPerson> latestAiPersons = Collections.emptyList();
    private final AtomicReference<Bitmap> pendingViewerFrame = new AtomicReference<>();
    private final Runnable applyViewerFrameRunnable = () -> {
        if (isFinishing() || isDestroyed() || viewerContainer.getVisibility() != View.VISIBLE) return;
        Bitmap frame = pendingViewerFrame.getAndSet(null);
        if (frame != null) {
            viewerStream.setImageBitmap(frame);
        }
    };
    private final Runnable refreshIpRunnable = new Runnable() {
        @Override
        public void run() {
            if (isFinishing() || isDestroyed() || cameraContainer.getVisibility() != View.VISIBLE || streamServer == null) {
                return;
            }
            List<String> ips = CameraStreamServer.getAllLocalIpv4Addresses();
            String ipLabel = ips.isEmpty() ? "Wi-Fi/Hotspot Bekleniyor" : String.join(" / ", ips);
            txtCodeDisplay.setText("Oda: " + roomCode + "  •  IP: " + ipLabel);
            uiHandler.postDelayed(this, 2500);
        }
    };

    private Bitmap cachedRawBitmap = null;
    private Bitmap cachedUprightBitmap = null;
    private Canvas cachedUprightCanvas = null;
    private Bitmap aiWorkBitmap = null;
    private Canvas aiWorkCanvas = null;
    private final Matrix uprightMatrix = new Matrix();
    private final Paint uprightPaint = new Paint();
    private final ByteArrayOutputStream reusableJpegOut = new ByteArrayOutputStream(24576);
    private String lastStatusText = "";
    private String lastTargetText = "";
    private int lastUnlockVisibility = View.GONE;
    private String lastViewerTargetText = "";
    private int lastViewerUnlockVisibility = View.GONE;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        bindViews();
        initServices();
        setupBackNavigation();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (githubUpdater != null) {
            githubUpdater.onActivityResumed(this);
        }
    }

    private void bindViews() {
        cameraContainer = findViewById(R.id.cameraContainer);
        viewerContainer = findViewById(R.id.viewerContainer);
        roleContainer = findViewById(R.id.roleContainer);
        stealthScreen = findViewById(R.id.stealthScreen);
        stealthPinPanel = findViewById(R.id.stealthPinPanel);
        alarmOverlay = findViewById(R.id.alarmOverlay);

        previewView = findViewById(R.id.previewView);
        previewView.setImplementationMode(PreviewView.ImplementationMode.PERFORMANCE);
        overlayView = findViewById(R.id.overlayView);
        txtStatus = findViewById(R.id.txtStatus);
        txtIpBadge = findViewById(R.id.txtIpBadge);
        txtTargetStatus = findViewById(R.id.txtTargetStatus);
        txtWarningBanner = findViewById(R.id.txtWarningBanner);
        txtCodeDisplay = findViewById(R.id.txtCodeDisplay);
        txtFpsDisplay = findViewById(R.id.txtFpsDisplay);
        btnUnlock = findViewById(R.id.btnUnlock);
        btnNameTarget = findViewById(R.id.btnNameTarget);
        btnMemory = findViewById(R.id.btnMemory);
        btnTorch = findViewById(R.id.btnTorch);

        viewerStream = findViewById(R.id.viewerStream);
        viewerOverlay = findViewById(R.id.viewerOverlay);
        viewerOverlay.setScaleModeFitCenter(true);
        txtViewerStatus = findViewById(R.id.txtViewerStatus);
        txtViewerTarget = findViewById(R.id.txtViewerTarget);
        btnViewerUnlock = findViewById(R.id.btnViewerUnlock);

        editPassword = findViewById(R.id.editPassword);
        txtPinDots = findViewById(R.id.txtPinDots);

        // Giriş Butonları
        findViewById(R.id.btnRoleCamera).setOnClickListener(v -> onCameraRoleClicked());
        findViewById(R.id.btnRoleViewer).setOnClickListener(v -> onViewerRoleClicked());
        btnCheckUpdate = findViewById(R.id.btnCheckUpdate);
        btnCheckUpdate.setText("Sürüm v" + GitHubUpdater.getCurrentVersionName(this) + " • Güncellemeyi Kontrol Et");
        btnCheckUpdate.setOnClickListener(v -> {
            if (githubUpdater != null) githubUpdater.checkInteractive(MainActivity.this);
        });

        // Kamera İçi Kontroller
        findViewById(R.id.btnStealth).setOnClickListener(v -> enterStealthMode());
        btnMemory.setOnClickListener(v -> showMemoryManagerDialog());
        btnTorch.setOnClickListener(v -> toggleTorch());
        findViewById(R.id.btnFlip).setOnClickListener(v -> flipCamera());
        findViewById(R.id.btnBackMain).setOnClickListener(v -> returnToMain());
        btnUnlock.setOnClickListener(v -> unlockTarget());
        btnNameTarget.setOnClickListener(v -> {
            if (personEngine == null) return;
            PersonDetectorEngine.LockedState locked = personEngine.getLockedStateSnapshot();
            if (locked != null) {
                promptRenamePersonDialog(locked.id, locked.name);
            }
        });

        // Alarm Kapatma
        findViewById(R.id.btnDismissAlarm).setOnClickListener(v -> {
            dismissAlarm();
            if (viewerContainer.getVisibility() == View.VISIBLE && viewerClient != null) {
                viewerClient.sendAction("dismiss_alarm");
            }
        });

        // Gizli Mod Dokunma & PIN Tuş Takımı
        setupStealthKeypad();

        // İzleyici Kontrolleri
        findViewById(R.id.btnViewerDisconnect).setOnClickListener(v -> returnToMain());
        findViewById(R.id.btnViewerManualIp).setOnClickListener(v -> promptManualIpDialog());
        btnViewerUnlock.setOnClickListener(v -> {
            if (viewerClient != null) viewerClient.sendAction("unlock");
        });
        findViewById(R.id.btnViewerTorch).setOnClickListener(v -> {
            if (viewerClient != null) viewerClient.sendAction("torch");
        });
        findViewById(R.id.btnViewerFlip).setOnClickListener(v -> {
            if (viewerClient != null) viewerClient.sendAction("flip");
        });

        // Kamera Ekranında Kişiye Dokunma -> Önce anlık önbellekten (<1ms), gerekirse AI Zoom ile kilitle
        overlayView.setOnTapNormListener((nx, ny, sx, sy) -> {
            if (personEngine == null) return;
            cameraExecutor.execute(() -> {
                PersonDetectorEngine.LockedState lockedSnapshot = new PersonDetectorEngine.LockedState();
                PersonDetectorEngine.LockStatus fastRes = personEngine.tryLockFast(nx, ny, lockedSnapshot);
                if (fastRes == PersonDetectorEngine.LockStatus.LOCKED) {
                    applyTapLockResult(lockedSnapshot, true, sx, sy);
                } else {
                    aiExecutor.execute(() -> {
                        PersonDetectorEngine.LockedState zoomSnap = new PersonDetectorEngine.LockedState();
                        PersonDetectorEngine.LockStatus zoomRes = personEngine.tryLockAt(nx, ny, zoomSnap);
                        applyTapLockResult(zoomSnap, zoomRes == PersonDetectorEngine.LockStatus.LOCKED, sx, sy);
                    });
                }
            });
        });

        // İzleyici Ekranında Kişiye Dokunma -> Kameraya Kilitleme Komutu Gönder
        viewerOverlay.setOnTapNormListener((nx, ny, sx, sy) -> {
            if (viewerClient != null) {
                viewerClient.sendLock(nx, ny);
            }
        });
    }

    private void applyTapLockResult(PersonDetectorEngine.LockedState snap, boolean locked, float sx, float sy) {
        runOnUiThread(() -> {
            if (locked) {
                sirenEngine.playLockBeep();
                dismissAlarm();
                overlayView.triggerFocusRing(sx, sy, false);
                updateTargetHud(snap);
            } else {
                sirenEngine.playErrorBeep();
                overlayView.triggerFocusRing(sx, sy, true);
                showWarningBanner("Kişi seçilmedi — Bir insana dokunun");
            }
        });
    }

    private void initServices() {
        cameraExecutor = Executors.newSingleThreadExecutor();
        aiExecutor = Executors.newSingleThreadExecutor();
        sirenEngine = new SirenAudioEngine(this);
        viewerClient = new ViewerStreamClient(this, createViewerListener());
        githubUpdater = new GitHubUpdater();
        githubUpdater.checkSilently(this, (hasUpdate, latestVer) -> {
            if (!isFinishing() && !isDestroyed() && btnCheckUpdate != null && hasUpdate) {
                btnCheckUpdate.setText("Yeni Sürüm Var (v" + latestVer + ") • Güncelle!");
                btnCheckUpdate.setTextColor(0xFF30D158);
            }
        });
        aiExecutor.execute(() -> {
            PersonDetectorEngine engine = new PersonDetectorEngine(MainActivity.this);
            personEngine = engine;
            if (!engine.isInitialized()) {
                runOnUiThread(() -> Toast.makeText(MainActivity.this,
                        "AI Model Hatası: " + engine.getInitError(), Toast.LENGTH_LONG).show());
            }
        });
    }

    private void setupBackNavigation() {
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) {
            @Override
            public void handleOnBackPressed() {
                if (isStealthActive) {
                    showStealthPinPanel();
                    return;
                }
                if (cameraContainer.getVisibility() == View.VISIBLE || viewerContainer.getVisibility() == View.VISIBLE) {
                    returnToMain();
                    return;
                }
                setEnabled(false);
                getOnBackPressedDispatcher().onBackPressed();
            }
        });
    }

    // =================== ROL BAŞLATMA & İZİNLER ===================
    private void onCameraRoleClicked() {
        String code = editPassword.getText().toString().trim();
        if (code.length() != 5) {
            Toast.makeText(this, "Lütfen 5 haneli bağlantı şifresi girin", Toast.LENGTH_SHORT).show();
            return;
        }
        roomCode = code;

        if (!hasCameraPermission()) {
            pendingStartCameraAfterPerm = true;
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    PERMISSION_REQ_CODE);
            return;
        }
        startCameraMode();
    }

    private boolean hasCameraPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQ_CODE) {
            if (hasCameraPermission() && pendingStartCameraAfterPerm) {
                pendingStartCameraAfterPerm = false;
                startCameraMode();
            } else if (!hasCameraPermission()) {
                pendingStartCameraAfterPerm = false;
                Toast.makeText(this, "Kamera izni olmadan kamera modu başlatılamaz!", Toast.LENGTH_LONG).show();
            }
        }
    }

    private void startCameraMode() {
        roleContainer.setVisibility(View.GONE);
        viewerContainer.setVisibility(View.GONE);
        cameraContainer.setVisibility(View.VISIBLE);

        txtIpBadge.setText("Şifre: " + roomCode);
        txtStatus.setText("Kamera Başlatılıyor...");
        unlockTarget();

        startLocalStreamServer();
        startCameraX();
    }

    private void onViewerRoleClicked() {
        String code = editPassword.getText().toString().trim();
        if (code.length() != 5) {
            Toast.makeText(this, "Lütfen 5 haneli bağlantı şifresi girin", Toast.LENGTH_SHORT).show();
            return;
        }
        roomCode = code;

        roleContainer.setVisibility(View.GONE);
        cameraContainer.setVisibility(View.GONE);
        viewerContainer.setVisibility(View.VISIBLE);

        txtViewerStatus.setText("Aynı ağda kamera aranıyor...");
        txtViewerTarget.setText("Bağlantı bekleniyor");
        btnViewerUnlock.setVisibility(View.GONE);

        viewerClient.startAutoConnect(roomCode);
    }

    // =================== CAMERAX + TFLITE AI İŞLEME ===================
    private void startCameraX() {
        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                cameraProvider = future.get();
                bindCameraUseCases();
            } catch (Exception e) {
                Toast.makeText(this, "Kamera hatası: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCameraUseCases() {
        if (cameraProvider == null) return;
        cameraProvider.unbindAll();
        isTorchOn = false;
        updateTorchUi();

        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setTargetResolution(new Size(480, 640))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(cameraExecutor, this::analyzeImageProxy);

        try {
            activeCamera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
        } catch (Exception e) {
            Toast.makeText(this, "Kamera bağlanamadı: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private void analyzeImageProxy(@NonNull ImageProxy imageProxy) {
        if (isAnalyzing || personEngine == null) {
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
            overlayView.setSourceInfo(bw, bh, isFrontCamera);

            // 1. Yalnızca aktif video akışı isteyen izleyici varken JPEG sıkıştır (PC oyun uykusundayken 0 JPEG yükü!)
            frameCounter++;
            final boolean hasVideoViewer = (streamServer != null && streamServer.hasActiveVideoStream());
            if (hasVideoViewer && frameCounter % 2 == 0) {
                reusableJpegOut.reset();
                upright.compress(Bitmap.CompressFormat.JPEG, 55, reusableJpegOut);
                streamServer.updateFrame(reusableJpegOut.toByteArray());
            }

            // 2. Gerçek canlı kamera & 30 FPS Optik Takip adımı (<3ms, AI'yi ASLA beklemez!)
            framesInSecond++;
            long now = System.currentTimeMillis();
            if (now - lastFpsCalcTs >= 1000) {
                lastFps = framesInSecond;
                framesInSecond = 0;
                lastFpsCalcTs = now;
            }

            PersonDetectorEngine.LockedState fastLocked = personEngine.fastTrackFrame(upright);
            final List<TrackedPerson> cachedPersons = latestAiPersons;
            uiHandler.post(() -> {
                if (isFinishing() || isDestroyed() || cameraContainer.getVisibility() != View.VISIBLE) return;
                overlayView.updateState(cachedPersons, fastLocked);
                long uiNow = System.currentTimeMillis();
                if (uiNow - lastFpsTextTs >= 500) {
                    lastFpsTextTs = uiNow;
                    int shownFps = (lastFps > 0) ? lastFps : framesInSecond;
                    txtFpsDisplay.setText(shownFps + " FPS • " + lastInferTimeMs + "ms");
                }
            });

            // 3. Arka plan AI iş parçacığı boşsa yeni kareyi asenkron teslim et
            if (personEngine.isInitialized() && aiBusy.compareAndSet(false, true)) {
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
                            PersonDetectorEngine.AnalysisOutput result = personEngine.processFrame(aiWorkBitmap);
                            latestAiPersons = (result.persons != null) ? result.persons : Collections.emptyList();
                            lastInferTimeMs = result.inferTimeMs;

                            if (streamServer != null && streamServer.hasActiveViewer()) {
                                streamServer.updateAiState(aiW, aiH, aiFront, isAlarmShowing, result.persons, result.locked);
                            }

                            uiHandler.post(() -> {
                                if (isFinishing() || isDestroyed() || cameraContainer.getVisibility() != View.VISIBLE) return;
                                overlayView.updateState(result.persons, result.locked);
                                updateCameraHud(result);
                            });
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
            if (personEngine != null) personEngine.clearFrameCache();
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

    private void updateCameraHud(PersonDetectorEngine.AnalysisOutput out) {
        int count = (out.persons != null) ? out.persons.size() : 0;

        String newStatus;
        if (personEngine == null || !personEngine.isInitialized()) {
            newStatus = "⚠ AI Model Yüklenemedi";
        } else if (out.locked != null) {
            newStatus = out.locked.lost
                    ? "⚠ HEDEF KAYIP!"
                    : "Kilitli Takip Aktif (" + count + " Kişi)";
        } else if (count > 0) {
            newStatus = "Canlı • " + count + " Kişi Algılandı";
        } else {
            newStatus = "Canlı • İnsan Aranıyor...";
        }

        if (!newStatus.equals(lastStatusText)) {
            lastStatusText = newStatus;
            txtStatus.setText(newStatus);
        }

        if (out.targetJustLost) {
            triggerAlarm();
            if (out.locked != null) updateTargetHud(out.locked);
        } else if (out.targetJustRecovered) {
            dismissAlarm();
            if (out.locked != null) updateTargetHud(out.locked);
        } else if (out.locked != null) {
            updateTargetHud(out.locked);
        }
        if (btnMemory != null) {
            String memLabel = (out.rememberedCount > 0) ? ("Hafıza (" + out.rememberedCount + ")") : "Hafıza";
            if (!memLabel.equals(btnMemory.getText().toString())) {
                btnMemory.setText(memLabel);
            }
        }
    }

    private void updateTargetHud(PersonDetectorEngine.LockedState locked) {
        String newTargetText;
        int newVis;
        if (locked == null) {
            newTargetText = "Kişiye dokunarak kilitleyin";
            newVis = View.GONE;
        } else if (locked.lost) {
            String aiInfo = buildMemoryBadge(locked);
            newTargetText = aiInfo.isEmpty()
                    ? ("KAYIP: " + locked.displayLabel() + "!")
                    : ("KAYIP: " + locked.displayLabel() + "! (" + aiInfo + ")");
            newVis = View.VISIBLE;
        } else {
            int pct = Math.round(locked.score * 100f);
            String badge = locked.hasFaceAi ? " • Yüz+AI" : (locked.hasDeepAi ? " • Derin AI" : "");
            newTargetText = "Kilitli: " + locked.displayLabel() + " (%" + pct + badge + ")";
            newVis = View.VISIBLE;
        }
        if (!newTargetText.equals(lastTargetText)) {
            lastTargetText = newTargetText;
            txtTargetStatus.setText(newTargetText);
        }
        if (newVis != lastUnlockVisibility) {
            lastUnlockVisibility = newVis;
            btnUnlock.setVisibility(newVis);
            if (btnNameTarget != null) btnNameTarget.setVisibility(newVis);
        }
    }

    private String buildMemoryBadge(PersonDetectorEngine.LockedState locked) {
        if (locked == null) return "";
        StringBuilder sb = new StringBuilder();
        if (locked.faceCount > 0) sb.append("Yüz:").append(locked.faceCount);
        if (locked.deepCount > 0) {
            if (sb.length() > 0) sb.append(" ");
            sb.append("AI:").append(locked.deepCount);
        }
        if (sb.length() == 0 && locked.memoryCount > 0) {
            sb.append("Hafıza:").append(locked.memoryCount);
        }
        return sb.toString();
    }

    private void showMemoryManagerDialog() {
        if (personEngine == null) {
            Toast.makeText(this, "Yapay zeka motoru henüz hazırlanıyor", Toast.LENGTH_SHORT).show();
            return;
        }
        List<PersonMemoryStore.ProfileSummary> profiles = personEngine.getMemorySummaries();
        String engineLine = "Motor: " + personEngine.getAiEngineSummary();

        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Yapay Zeka Hafıza Bankası (" + profiles.size() + ")");

        if (profiles.isEmpty()) {
            builder.setMessage(engineLine + "\n\nHenüz kayıtlı kişi yok. Kamerada bir kişi göründüğünde yüz ve derin vücut vektörleri otomatik öğrenilip kalıcı hafızaya kaydedilir.");
        } else {
            CharSequence[] items = new CharSequence[profiles.size()];
            for (int i = 0; i < profiles.size(); i++) {
                PersonMemoryStore.ProfileSummary p = profiles.get(i);
                StringBuilder line = new StringBuilder(p.displayLabel());
                line.append("  •  Yüz: ").append(p.faceCount)
                        .append("  •  Derin AI: ").append(p.deepCount)
                        .append("  •  İmza: ").append(p.sigCount);
                items[i] = line.toString();
            }
            builder.setItems(items, (dialog, which) -> {
                PersonMemoryStore.ProfileSummary chosen = profiles.get(which);
                showProfileActionDialog(chosen);
            });
            builder.setNeutralButton("Tümünü Temizle", (d, w) -> {
                new AlertDialog.Builder(this)
                        .setTitle("Hafızayı Temizle")
                        .setMessage("Kayıtlı tüm kişi isimleri, yüz ve derin yapay zeka vektörleri kalıcı olarak silinsin mi?")
                        .setPositiveButton("Sil", (d2, w2) -> {
                            personEngine.clearAllMemory();
                            btnMemory.setText("Hafıza");
                            updateTargetHud(personEngine.getLockedStateSnapshot());
                            Toast.makeText(this, "Kalıcı yapay zeka hafızası sıfırlandı", Toast.LENGTH_SHORT).show();
                        })
                        .setNegativeButton("İptal", null)
                        .show();
            });
        }
        builder.setPositiveButton("Kapat", null);
        builder.show();
    }

    private void showProfileActionDialog(PersonMemoryStore.ProfileSummary profile) {
        CharSequence[] actions = {"İsim Ver / Değiştir", "Bu Kişiyi Hafızadan Sil"};
        new AlertDialog.Builder(this)
                .setTitle(profile.displayLabel())
                .setItems(actions, (d, which) -> {
                    if (which == 0) {
                        promptRenamePersonDialog(profile.id, profile.name);
                    } else if (which == 1 && personEngine != null) {
                        personEngine.deletePersonFromMemory(profile.id);
                        updateTargetHud(personEngine.getLockedStateSnapshot());
                        Toast.makeText(this, profile.displayLabel() + " hafızadan silindi", Toast.LENGTH_SHORT).show();
                    }
                })
                .setNegativeButton("İptal", null)
                .show();
    }

    private void promptRenamePersonDialog(int personId, String currentName) {
        if (personEngine == null) return;
        AlertDialog.Builder b = new AlertDialog.Builder(this);
        b.setTitle("Kişiye İsim Ver (#" + personId + ")");
        b.setMessage("İsim verilen kişiler kalıcı hafızada saklanır; uygulama kapansa veya kıyafet değişse bile yüz/AI ile tanınır.");
        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_CAP_WORDS);
        input.setHint("Örn: Selim, Anne, Ev Sahibi");
        if (currentName != null && !currentName.trim().isEmpty()) {
            input.setText(currentName.trim());
            input.setSelection(input.getText().length());
        }
        b.setView(input);
        b.setPositiveButton("Kaydet", (d, w) -> {
            String newName = input.getText().toString().trim();
            personEngine.renamePerson(personId, newName);
            PersonDetectorEngine.LockedState snap = personEngine.getLockedStateSnapshot();
            updateTargetHud(snap);
            overlayView.updateState(personEngine.getLatestPersonsSnapshot(), snap);
            Toast.makeText(this, newName.isEmpty() ? "İsim kaldırıldı" : ("Kaydedildi: " + newName), Toast.LENGTH_SHORT).show();
        });
        b.setNegativeButton("İptal", null);
        b.show();
    }

    private void showWarningBanner(String msg) {
        txtWarningBanner.setText(msg);
        txtWarningBanner.setVisibility(View.VISIBLE);
        uiHandler.removeCallbacks(hideWarningRunnable);
        uiHandler.postDelayed(hideWarningRunnable, 1800);
    }

    private final Runnable hideWarningRunnable = () -> txtWarningBanner.setVisibility(View.GONE);

    private void unlockTarget() {
        if (personEngine != null) personEngine.unlock();
        overlayView.updateState(null, null);
        updateTargetHud(null);
        dismissAlarm();
    }

    private void flipCamera() {
        isFrontCamera = !isFrontCamera;
        cameraSelector = isFrontCamera ? CameraSelector.DEFAULT_FRONT_CAMERA : CameraSelector.DEFAULT_BACK_CAMERA;
        unlockTarget();
        bindCameraUseCases();
    }

    private void toggleTorch() {
        if (activeCamera == null || isFrontCamera || !activeCamera.getCameraInfo().hasFlashUnit()) {
            Toast.makeText(this, "Bu kamerada flaş desteklenmiyor", Toast.LENGTH_SHORT).show();
            return;
        }
        isTorchOn = !isTorchOn;
        activeCamera.getCameraControl().enableTorch(isTorchOn);
        updateTorchUi();
    }

    private void updateTorchUi() {
        btnTorch.setText(isTorchOn ? "Flaş: Açık" : "Flaş");
        btnTorch.setTextColor(isTorchOn ? 0xFFFFD60A : 0xFFFFFFFF);
    }

    // =================== ALARM YÖNETİMİ (TELEFON TAMAMEN SESSİZ) ===================
    private void triggerAlarm() {
        if (isAlarmShowing) return;
        isAlarmShowing = true;
        // Gizli OLED karartma açıkken ekran siyah kalır; açık ekranda görsel uyarı görünür; SES KESİNLİKLE ÇIKMAZ.
        if (!isStealthActive) {
            alarmOverlay.setVisibility(View.VISIBLE);
        }
        if (sirenEngine != null) sirenEngine.startAlarm();
    }

    private void dismissAlarm() {
        isAlarmShowing = false;
        alarmOverlay.setVisibility(View.GONE);
        if (sirenEngine != null) sirenEngine.stopAlarm();
    }

    // =================== GİZLİ MOD (OLED KARARTMA & PIN TUŞ TAKIMI) ===================
    private void enterStealthMode() {
        isStealthActive = true;
        pinInput.setLength(0);
        updatePinDots();
        stealthPinPanel.setVisibility(View.GONE);
        stealthScreen.setVisibility(View.VISIBLE);

        // Ekran parlaklığını minimuma çek ve sistem çubuklarını gizle
        WindowManager.LayoutParams lp = getWindow().getAttributes();
        lp.screenBrightness = 0.01f;
        getWindow().setAttributes(lp);
        setImmersiveMode(true);
    }

    private void exitStealthMode() {
        isStealthActive = false;
        uiHandler.removeCallbacks(hidePinPanelRunnable);
        stealthPinPanel.setVisibility(View.GONE);
        stealthScreen.setVisibility(View.GONE);

        WindowManager.LayoutParams lp = getWindow().getAttributes();
        lp.screenBrightness = WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE;
        getWindow().setAttributes(lp);
        setImmersiveMode(false);
    }

    private void showStealthPinPanel() {
        pinInput.setLength(0);
        updatePinDots();
        stealthPinPanel.setVisibility(View.VISIBLE);
        uiHandler.removeCallbacks(hidePinPanelRunnable);
        uiHandler.postDelayed(hidePinPanelRunnable, 7000);
    }

    private final Runnable hidePinPanelRunnable = () -> stealthPinPanel.setVisibility(View.GONE);

    private void setupStealthKeypad() {
        stealthScreen.setOnClickListener(v -> {
            if (stealthPinPanel.getVisibility() != View.VISIBLE) {
                showStealthPinPanel();
            }
        });

        int[] numIds = {R.id.pin0, R.id.pin1, R.id.pin2, R.id.pin3, R.id.pin4,
                R.id.pin5, R.id.pin6, R.id.pin7, R.id.pin8, R.id.pin9};
        for (int i = 0; i < numIds.length; i++) {
            final int digit = i;
            findViewById(numIds[i]).setOnClickListener(v -> onPinDigit(digit));
        }

        findViewById(R.id.pinDel).setOnClickListener(v -> {
            if (pinInput.length() > 0) {
                pinInput.deleteCharAt(pinInput.length() - 1);
                updatePinDots();
            }
            resetPinAutoHide();
        });

        findViewById(R.id.pinCancel).setOnClickListener(v -> {
            stealthPinPanel.setVisibility(View.GONE);
            uiHandler.removeCallbacks(hidePinPanelRunnable);
        });
    }

    private void onPinDigit(int d) {
        if (pinInput.length() >= 4) return;
        pinInput.append(d);
        updatePinDots();
        resetPinAutoHide();

        if (pinInput.length() == 4) {
            if (STEALTH_PIN.equals(pinInput.toString())) {
                sirenEngine.playLockBeep();
                exitStealthMode();
            } else {
                sirenEngine.playErrorBeep();
                Toast.makeText(this, "Hatalı PIN!", Toast.LENGTH_SHORT).show();
                pinInput.setLength(0);
                updatePinDots();
            }
        }
    }

    private void resetPinAutoHide() {
        uiHandler.removeCallbacks(hidePinPanelRunnable);
        uiHandler.postDelayed(hidePinPanelRunnable, 7000);
    }

    private void updatePinDots() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 4; i++) {
            if (i > 0) sb.append(" ");
            sb.append(i < pinInput.length() ? "●" : "○");
        }
        txtPinDots.setText(sb.toString());
    }

    private void setImmersiveMode(boolean enable) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            WindowInsetsController ctrl = getWindow().getInsetsController();
            if (ctrl != null) {
                if (enable) {
                    ctrl.hide(WindowInsets.Type.statusBars() | WindowInsets.Type.navigationBars());
                    ctrl.setSystemBarsBehavior(WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
                } else {
                    ctrl.show(WindowInsets.Type.statusBars() | WindowInsets.Type.navigationBars());
                }
            }
        } else {
            if (enable) {
                getWindow().getDecorView().setSystemUiVisibility(
                        View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                                | View.SYSTEM_UI_FLAG_FULLSCREEN
                                | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION);
            } else {
                getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_VISIBLE);
            }
        }
    }

    // =================== YEREL STREAM SUNUCUSU & İZLEYİCİ İLETİŞİMİ ===================
    private void startLocalStreamServer() {
        cameraExecutor.execute(() -> {
            try {
                if (streamServer != null) streamServer.stop();
                CameraStreamServer server = new CameraStreamServer(STREAM_PORT, roomCode);
                server.setOnRemoteCommandListener(new CameraStreamServer.OnRemoteCommandListener() {
                    @Override
                    public String onRemoteLock(float nx, float ny) {
                        if (personEngine == null) return "EMPTY";
                        PersonDetectorEngine.LockedState snap = new PersonDetectorEngine.LockedState();
                        PersonDetectorEngine.LockStatus res = personEngine.tryLockFast(nx, ny, snap);
                        if (res != PersonDetectorEngine.LockStatus.LOCKED && aiExecutor != null) {
                            try {
                                res = aiExecutor.submit(() -> personEngine.tryLockAt(nx, ny, snap)).get(600, TimeUnit.MILLISECONDS);
                            } catch (Exception ignored) {}
                        }
                        final PersonDetectorEngine.LockStatus finalRes = res;
                        runOnUiThread(() -> {
                            if (finalRes == PersonDetectorEngine.LockStatus.LOCKED) {
                                sirenEngine.playLockBeep();
                                dismissAlarm();
                                updateTargetHud(snap);
                            }
                        });
                        return (finalRes == PersonDetectorEngine.LockStatus.LOCKED) ? "LOCKED" : "EMPTY";
                    }

                    @Override
                    public void onRemoteUnlock() {
                        runOnUiThread(() -> unlockTarget());
                    }

                    @Override
                    public void onRemoteDismissAlarm() {
                        runOnUiThread(() -> dismissAlarm());
                    }

                    @Override
                    public void onRemoteFlipCamera() {
                        runOnUiThread(() -> flipCamera());
                    }

                    @Override
                    public void onRemoteToggleTorch() {
                        runOnUiThread(() -> toggleTorch());
                    }
                });
                server.start();
                streamServer = server;

                uiHandler.post(() -> {
                    uiHandler.removeCallbacks(refreshIpRunnable);
                    uiHandler.post(refreshIpRunnable);
                });
            } catch (Exception e) {
                uiHandler.post(() -> txtCodeDisplay.setText("Oda: " + roomCode + "  •  Yayın Hatası"));
            }
        });
    }

    private ViewerStreamClient.Listener createViewerListener() {
        return new ViewerStreamClient.Listener() {
            @Override
            public void onSearching(String message) {
                runOnUiThread(() -> txtViewerStatus.setText(message));
            }

            @Override
            public void onConnected(String ip, String deviceModel) {
                runOnUiThread(() -> txtViewerStatus.setText("Canlı • " + deviceModel + " (" + ip + ")"));
            }

            @Override
            public void onWrongPassword() {
                runOnUiThread(() -> {
                    Toast.makeText(MainActivity.this, "Bağlantı şifresi hatalı!", Toast.LENGTH_LONG).show();
                    returnToMain();
                });
            }

            @Override
            public void onFrame(Bitmap bitmap) {
                pendingViewerFrame.set(bitmap);
                uiHandler.removeCallbacks(applyViewerFrameRunnable);
                uiHandler.post(applyViewerFrameRunnable);
            }

            @Override
            public void onAiState(int w, int h, boolean isFront, boolean alarm,
                                  List<TrackedPerson> persons, PersonDetectorEngine.LockedState locked) {
                uiHandler.post(() -> {
                    if (viewerContainer.getVisibility() != View.VISIBLE) return;
                    // İzleyiciye gelen MJPEG kare düz (aynasız) olduğu için isFront=false kullanılır
                    viewerOverlay.setSourceInfo(w, h, false);
                    viewerOverlay.updateState(persons, locked);

                    String vt;
                    int vv;
                    if (locked != null) {
                        int pct = Math.round(locked.score * 100f);
                        String badge = locked.hasFaceAi ? " • Yüz+AI" : (locked.hasDeepAi ? " • AI" : "");
                        vt = locked.lost
                                ? "KAYIP: " + locked.displayLabel() + "!"
                                : "Kilitli: " + locked.displayLabel() + " (%" + pct + badge + ")";
                        vv = View.VISIBLE;
                    } else {
                        int c = (persons != null) ? persons.size() : 0;
                        vt = c > 0 ? (c + " Kişi Algılandı — Dokunun") : "İnsan bekleniyor...";
                        vv = View.GONE;
                    }
                    if (!vt.equals(lastViewerTargetText)) {
                        lastViewerTargetText = vt;
                        txtViewerTarget.setText(vt);
                    }
                    if (vv != lastViewerUnlockVisibility) {
                        lastViewerUnlockVisibility = vv;
                        btnViewerUnlock.setVisibility(vv);
                    }

                    if (alarm && !isAlarmShowing) {
                        triggerAlarm();
                    } else if (!alarm && isAlarmShowing) {
                        dismissAlarm();
                    }
                });
            }

            @Override
            public void onDisconnected(String reason) {
                runOnUiThread(() -> {
                    txtViewerStatus.setText("Bağlantı Bulunamadı");
                    Toast.makeText(MainActivity.this, reason, Toast.LENGTH_LONG).show();
                });
            }

            @Override
            public void onRemoteLockResult(boolean locked) {
                runOnUiThread(() -> {
                    if (locked) {
                        sirenEngine.playLockBeep();
                    } else {
                        sirenEngine.playErrorBeep();
                        Toast.makeText(MainActivity.this, "Kişi seçilmedi — Bir insana dokunun", Toast.LENGTH_SHORT).show();
                    }
                });
            }
        };
    }

    private void promptManualIpDialog() {
        AlertDialog.Builder b = new AlertDialog.Builder(this);
        b.setTitle("Kamera IP Adresi");
        b.setMessage("Kamera cihazının altında yazan IP adresini girin (Örn: 192.168.1.45):");
        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        input.setHint("192.168.1.x");
        b.setView(input);
        b.setPositiveButton("Bağlan", (d, w) -> {
            String ip = input.getText().toString().trim();
            if (!ip.isEmpty() && viewerClient != null) {
                viewerClient.connectDirect(ip, roomCode);
            }
        });
        b.setNegativeButton("İptal", (d, w) -> d.cancel());
        b.show();
    }

    private void returnToMain() {
        uiHandler.removeCallbacks(refreshIpRunnable);
        uiHandler.removeCallbacks(applyViewerFrameRunnable);
        pendingViewerFrame.set(null);
        if (isStealthActive) exitStealthMode();
        if (viewerClient != null) viewerClient.stop();
        if (cameraProvider != null) cameraProvider.unbindAll();
        activeCamera = null;
        if (streamServer != null) {
            streamServer.stop();
            streamServer = null;
        }
        if (personEngine != null) {
            personEngine.unlock();
            personEngine.clearFrameCache();
        }
        dismissAlarm();
        lastStatusText = "";
        lastTargetText = "";
        lastUnlockVisibility = View.GONE;
        lastViewerTargetText = "";
        lastViewerUnlockVisibility = View.GONE;
        latestAiPersons = Collections.emptyList();
        lastInferTimeMs = 0;
        lastFps = 0;
        framesInSecond = 0;
        lastFpsTextTs = 0L;

        cameraContainer.setVisibility(View.GONE);
        viewerContainer.setVisibility(View.GONE);
        stealthScreen.setVisibility(View.GONE);
        roleContainer.setVisibility(View.VISIBLE);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        returnToMain();
        if (cameraExecutor != null) {
            cameraExecutor.shutdownNow();
            try { cameraExecutor.awaitTermination(250, TimeUnit.MILLISECONDS); } catch (InterruptedException ignored) {}
        }
        if (aiExecutor != null) {
            aiExecutor.shutdownNow();
            try { aiExecutor.awaitTermination(350, TimeUnit.MILLISECONDS); } catch (InterruptedException ignored) {}
        }
        if (viewerClient != null) viewerClient.release();
        if (personEngine != null) personEngine.close();
        if (sirenEngine != null) sirenEngine.release();
        cachedRawBitmap = null;
        cachedUprightBitmap = null;
        aiWorkBitmap = null;
    }
}
