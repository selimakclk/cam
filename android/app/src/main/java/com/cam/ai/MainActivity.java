package com.cam.ai;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.provider.Settings;
import android.text.InputType;
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
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

public class MainActivity extends AppCompatActivity {

    private static final int PERMISSION_REQ_CODE = 1001;
    private static final String STEALTH_PIN = "7693";

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

    // Servisler
    private ViewerStreamClient viewerClient;
    private GitHubUpdater githubUpdater;
    private Button btnCheckUpdate;

    private final Handler uiHandler = new Handler(Looper.getMainLooper());
    private String roomCode = "12345";
    private boolean pendingStartCameraAfterPerm = false;
    private boolean batteryPromptShownThisSession = false;
    private boolean isAlarmShowing = false;
    private boolean isStealthActive = false;
    private final StringBuilder pinInput = new StringBuilder();

    private long lastFpsTextTs = 0L;
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
            if (isFinishing() || isDestroyed() || cameraContainer.getVisibility() != View.VISIBLE) {
                return;
            }
            List<String> ips = CameraStreamServer.getAllLocalIpv4Addresses();
            String ipLabel = ips.isEmpty() ? "Wi-Fi/Hotspot Bekleniyor" : String.join(" / ", ips);
            txtCodeDisplay.setText("Oda: " + roomCode + "  •  IP: " + ipLabel);
            uiHandler.postDelayed(this, 2500);
        }
    };

    private String lastStatusText = "";
    private String lastTargetText = "";
    private int lastUnlockVisibility = View.GONE;
    private String lastViewerTargetText = "";
    private int lastViewerUnlockVisibility = View.GONE;

    private final CameraMonitorService.UiCallback cameraUiCallback = new CameraMonitorService.UiCallback() {
        @Override
        public void onSourceInfo(int w, int h, boolean isFront) {
            if (isFinishing() || isDestroyed()) return;
            overlayView.setSourceInfo(w, h, isFront);
        }

        @Override
        public void onFastFrame(List<TrackedPerson> persons, PersonDetectorEngine.LockedState locked, int fps, int inferMs) {
            if (isFinishing() || isDestroyed() || cameraContainer.getVisibility() != View.VISIBLE) return;
            overlayView.updateState(persons, locked);
            long uiNow = System.currentTimeMillis();
            if (uiNow - lastFpsTextTs >= 500) {
                lastFpsTextTs = uiNow;
                txtFpsDisplay.setText(fps + " FPS • " + inferMs + "ms");
            }
        }

        @Override
        public void onAiOutput(PersonDetectorEngine.AnalysisOutput result) {
            if (isFinishing() || isDestroyed() || cameraContainer.getVisibility() != View.VISIBLE) return;
            overlayView.updateState(result.persons, result.locked);
            updateCameraHud(result);
        }

        @Override
        public void onTapLockResult(PersonDetectorEngine.LockedState snap, boolean locked, float sx, float sy) {
            if (isFinishing() || isDestroyed()) return;
            if (locked) {
                dismissAlarmVisualOnly();
                overlayView.triggerFocusRing(sx, sy, false);
                updateTargetHud(snap);
            } else {
                overlayView.triggerFocusRing(sx, sy, true);
                showWarningBanner("Kişi seçilmedi — Bir insana dokunun");
            }
        }

        @Override
        public void onRemoteLockApplied(PersonDetectorEngine.LockedState snap) {
            if (isFinishing() || isDestroyed()) return;
            dismissAlarmVisualOnly();
            updateTargetHud(snap);
        }

        @Override
        public void onAlarmStateChanged(boolean alarmShowing) {
            if (isFinishing() || isDestroyed()) return;
            if (alarmShowing) triggerAlarmVisualOnly();
            else dismissAlarmVisualOnly();
        }

        @Override
        public void onTorchChanged(boolean torchOn) {
            if (isFinishing() || isDestroyed()) return;
            btnTorch.setText(torchOn ? "Flaş: Açık" : "Flaş");
            btnTorch.setTextColor(torchOn ? 0xFFFFD60A : 0xFFFFFFFF);
        }

        @Override
        public void onEngineError(String error) {
            if (isFinishing() || isDestroyed()) return;
            Toast.makeText(MainActivity.this, "AI Model Hatası: " + error, Toast.LENGTH_LONG).show();
        }

        @Override
        public void onServiceStopped() {
            if (isFinishing() || isDestroyed()) return;
            if (cameraContainer.getVisibility() == View.VISIBLE) {
                showRoleScreenClean();
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        bindViews();
        initServices();
        setupBackNavigation();
    }

    @Override
    protected void onStart() {
        super.onStart();
        // Ekran kilitlenip tekrar açıldıysa ve servis arka planda çalışıyorsa canlı kamera UI'ını anında geri bağla!
        if (CameraMonitorService.isRunning()) {
            CameraMonitorService svc = CameraMonitorService.getInstance();
            if (svc != null) {
                roomCode = svc.getRoomCode();
                enterCameraUiShell();
                svc.attachUi(previewView.getSurfaceProvider(), cameraUiCallback);
            }
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (githubUpdater != null) {
            githubUpdater.onActivityResumed(this);
        }
    }

    @Override
    protected void onStop() {
        super.onStop();
        // KRİTİK: Ekran kapandığında / güç tuşuna basıldığında servisi ASLA durdurma; yalnızca önizleme yüzeyini ayır!
        CameraMonitorService.setServiceReadyListener(null);
        CameraMonitorService svc = CameraMonitorService.getInstance();
        if (svc != null && CameraMonitorService.isRunning()) {
            svc.detachUi();
        }
        uiHandler.removeCallbacks(refreshIpRunnable);
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
        btnTorch.setOnClickListener(v -> {
            CameraMonitorService svc = CameraMonitorService.getInstance();
            if (svc == null || !svc.toggleTorch()) {
                Toast.makeText(this, "Bu kamerada flaş desteklenmiyor", Toast.LENGTH_SHORT).show();
            }
        });
        findViewById(R.id.btnFlip).setOnClickListener(v -> {
            CameraMonitorService svc = CameraMonitorService.getInstance();
            if (svc != null) svc.flipCamera();
        });
        findViewById(R.id.btnBackMain).setOnClickListener(v -> returnToMain());
        btnUnlock.setOnClickListener(v -> {
            CameraMonitorService svc = CameraMonitorService.getInstance();
            if (svc != null) svc.unlockTarget();
            updateTargetHud(null);
            dismissAlarmVisualOnly();
        });
        btnNameTarget.setOnClickListener(v -> {
            PersonDetectorEngine eng = getActiveEngine();
            if (eng == null) return;
            PersonDetectorEngine.LockedState locked = eng.getLockedStateSnapshot();
            if (locked != null) {
                promptRenamePersonDialog(locked.id, locked.name);
            }
        });

        // Alarm Kapatma
        findViewById(R.id.btnDismissAlarm).setOnClickListener(v -> {
            dismissAlarmVisualOnly();
            CameraMonitorService svc = CameraMonitorService.getInstance();
            if (svc != null && cameraContainer.getVisibility() == View.VISIBLE) {
                svc.dismissAlarm();
            }
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

        // Kamera Ekranında Kişiye Dokunma -> Servise ilet
        overlayView.setOnTapNormListener((nx, ny, sx, sy) -> {
            CameraMonitorService svc = CameraMonitorService.getInstance();
            if (svc != null) {
                svc.tapLockAt(nx, ny, sx, sy);
            }
        });

        // İzleyici Ekranında Kişiye Dokunma -> Kameraya Kilitleme Komutu Gönder
        viewerOverlay.setOnTapNormListener((nx, ny, sx, sy) -> {
            if (viewerClient != null) {
                viewerClient.sendLock(nx, ny);
            }
        });
    }

    private PersonDetectorEngine getActiveEngine() {
        CameraMonitorService svc = CameraMonitorService.getInstance();
        return (svc != null) ? svc.getPersonEngine() : null;
    }

    private void initServices() {
        viewerClient = new ViewerStreamClient(this, createViewerListener());
        githubUpdater = new GitHubUpdater();
        githubUpdater.checkSilently(this, (hasUpdate, latestVer) -> {
            if (!isFinishing() && !isDestroyed() && btnCheckUpdate != null && hasUpdate) {
                btnCheckUpdate.setText("Yeni Sürüm Var (v" + latestVer + ") • Güncelle!");
                btnCheckUpdate.setTextColor(0xFF30D158);
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
            List<String> perms = new ArrayList<>();
            perms.add(Manifest.permission.CAMERA);
            if (Build.VERSION.SDK_INT >= 33
                    && ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                    != PackageManager.PERMISSION_GRANTED) {
                perms.add(Manifest.permission.POST_NOTIFICATIONS);
            }
            ActivityCompat.requestPermissions(this, perms.toArray(new String[0]), PERMISSION_REQ_CODE);
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

    private void enterCameraUiShell() {
        roleContainer.setVisibility(View.GONE);
        viewerContainer.setVisibility(View.GONE);
        cameraContainer.setVisibility(View.VISIBLE);

        txtIpBadge.setText("Şifre: " + roomCode);
        if (lastStatusText.isEmpty()) {
            txtStatus.setText("Kamera Başlatılıyor...");
        }
        uiHandler.removeCallbacks(refreshIpRunnable);
        uiHandler.post(refreshIpRunnable);
    }

    private void startCameraMode() {
        enterCameraUiShell();
        updateTargetHud(null);
        dismissAlarmVisualOnly();

        CameraMonitorService.setServiceReadyListener(svc -> {
            if (isFinishing() || isDestroyed() || cameraContainer.getVisibility() != View.VISIBLE) return;
            svc.attachUi(previewView.getSurfaceProvider(), cameraUiCallback);
        });
        CameraMonitorService.startService(this, roomCode);

        maybeOfferBatteryOptimizationWhitelist();
    }

    private void maybeOfferBatteryOptimizationWhitelist() {
        if (batteryPromptShownThisSession || Build.VERSION.SDK_INT < Build.VERSION_CODES.M) return;
        try {
            PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
            if (pm != null && !pm.isIgnoringBatteryOptimizations(getPackageName())) {
                batteryPromptShownThisSession = true;
                new AlertDialog.Builder(this)
                        .setTitle("Ekran Kapalıyken Kesintisiz Çalışma")
                        .setMessage("Telefon kilitlendiğinde sistemin kamerayı uyutmaması için pil kısıtlamasını kaldırmanız önerilir.")
                        .setPositiveButton("İzin Ver", (d, w) -> {
                            try {
                                Intent i = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS);
                                i.setData(Uri.parse("package:" + getPackageName()));
                                startActivity(i);
                            } catch (Exception ignored) {}
                        })
                        .setNegativeButton("Şimdi Değil", null)
                        .show();
            }
        } catch (Exception ignored) {}
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

    // =================== HUD & HAFIZA YÖNETİMİ ===================
    private void updateCameraHud(PersonDetectorEngine.AnalysisOutput out) {
        int count = (out.persons != null) ? out.persons.size() : 0;
        PersonDetectorEngine eng = getActiveEngine();

        String newStatus;
        if (eng == null || !eng.isInitialized()) {
            newStatus = "⚠ AI Model Yüklenemedi";
        } else if (out.locked != null) {
            newStatus = out.locked.lost
                    ? "⚠ HEDEF KAYIP! (Arka Plan Aktif)"
                    : "Kilitli Takip • Ekran Kapalı Çalışır (" + count + " Kişi)";
        } else if (count > 0) {
            newStatus = "Canlı • Ekran Kapalı Çalışır (" + count + " Kişi)";
        } else {
            newStatus = "Canlı • Ekran Kapalı Çalışır • Aranıyor...";
        }

        if (!newStatus.equals(lastStatusText)) {
            lastStatusText = newStatus;
            txtStatus.setText(newStatus);
        }

        if (out.targetJustLost) {
            triggerAlarmVisualOnly();
            if (out.locked != null) updateTargetHud(out.locked);
        } else if (out.targetJustRecovered) {
            dismissAlarmVisualOnly();
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
        PersonDetectorEngine eng = getActiveEngine();
        if (eng == null) {
            Toast.makeText(this, "Yapay zeka motoru henüz hazırlanıyor", Toast.LENGTH_SHORT).show();
            return;
        }
        List<PersonMemoryStore.ProfileSummary> profiles = eng.getMemorySummaries();
        String engineLine = "Motor: " + eng.getAiEngineSummary();

        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Yapay Zeka Hafıza Bankası (" + profiles.size() + ")");

        if (profiles.isEmpty()) {
            builder.setMessage(engineLine + "\n\nHenüz kayıtlı kişi yok. Kamerada bir kişi göründüğünde yüz ve derin vücut vektörleri otomatik öğrenilip kalıcı hafızaya kaydedilir.");
        } else {
            CharSequence[] items = new CharSequence[profiles.size()];
            for (int i = 0; i < profiles.size(); i++) {
                PersonMemoryStore.ProfileSummary p = profiles.get(i);
                items[i] = p.displayLabel()
                        + "  •  Yüz: " + p.faceCount
                        + "  •  Açı Hafızası: " + p.sigCount;
            }
            builder.setItems(items, (dialog, which) -> showProfileActionDialog(profiles.get(which)));
            builder.setNeutralButton("Tümünü Temizle", (d, w) -> new AlertDialog.Builder(this)
                    .setTitle("Hafızayı Temizle")
                    .setMessage("Kayıtlı tüm kişi isimleri, yüz ve derin yapay zeka vektörleri kalıcı olarak silinsin mi?")
                    .setPositiveButton("Sil", (d2, w2) -> {
                        PersonDetectorEngine e = getActiveEngine();
                        if (e != null) {
                            e.clearAllMemory();
                            btnMemory.setText("Hafıza");
                            updateTargetHud(e.getLockedStateSnapshot());
                        }
                        Toast.makeText(this, "Kalıcı yapay zeka hafızası sıfırlandı", Toast.LENGTH_SHORT).show();
                    })
                    .setNegativeButton("İptal", null)
                    .show());
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
                    } else if (which == 1) {
                        PersonDetectorEngine e = getActiveEngine();
                        if (e != null) {
                            e.deletePersonFromMemory(profile.id);
                            updateTargetHud(e.getLockedStateSnapshot());
                            Toast.makeText(this, profile.displayLabel() + " hafızadan silindi", Toast.LENGTH_SHORT).show();
                        }
                    }
                })
                .setNegativeButton("İptal", null)
                .show();
    }

    private void promptRenamePersonDialog(int personId, String currentName) {
        PersonDetectorEngine eng = getActiveEngine();
        if (eng == null) return;
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
            PersonDetectorEngine e = getActiveEngine();
            if (e == null) return;
            String newName = input.getText().toString().trim();
            e.renamePerson(personId, newName);
            PersonDetectorEngine.LockedState snap = e.getLockedStateSnapshot();
            updateTargetHud(snap);
            overlayView.updateState(e.getLatestPersonsSnapshot(), snap);
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

    // =================== ALARM YÖNETİMİ (TELEFON KESİNLİKLE SESSİZ) ===================
    private void triggerAlarmVisualOnly() {
        if (isAlarmShowing) return;
        isAlarmShowing = true;
        if (!isStealthActive) {
            alarmOverlay.setVisibility(View.VISIBLE);
        }
    }

    private void dismissAlarmVisualOnly() {
        isAlarmShowing = false;
        alarmOverlay.setVisibility(View.GONE);
    }

    // =================== GİZLİ MOD (OLED KARARTMA & PIN TUŞ TAKIMI) ===================
    private void enterStealthMode() {
        isStealthActive = true;
        pinInput.setLength(0);
        updatePinDots();
        stealthPinPanel.setVisibility(View.GONE);
        stealthScreen.setVisibility(View.VISIBLE);

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
                exitStealthMode();
            } else {
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

    // =================== TELEFON İZLEYİCİ MODU ===================
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
                        triggerAlarmVisualOnly();
                    } else if (!alarm && isAlarmShowing) {
                        dismissAlarmVisualOnly();
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
                    if (!locked) {
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

    private void showRoleScreenClean() {
        uiHandler.removeCallbacks(refreshIpRunnable);
        uiHandler.removeCallbacks(applyViewerFrameRunnable);
        pendingViewerFrame.set(null);
        if (isStealthActive) exitStealthMode();
        if (viewerClient != null) viewerClient.stop();
        dismissAlarmVisualOnly();
        overlayView.updateState(null, null);

        lastStatusText = "";
        lastTargetText = "";
        lastUnlockVisibility = View.GONE;
        lastViewerTargetText = "";
        lastViewerUnlockVisibility = View.GONE;
        lastFpsTextTs = 0L;

        cameraContainer.setVisibility(View.GONE);
        viewerContainer.setVisibility(View.GONE);
        stealthScreen.setVisibility(View.GONE);
        roleContainer.setVisibility(View.VISIBLE);
    }

    private void returnToMain() {
        CameraMonitorService.setServiceReadyListener(null);
        if (CameraMonitorService.isRunning()) {
            CameraMonitorService.stopService(this);
        }
        showRoleScreenClean();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        uiHandler.removeCallbacksAndMessages(null);
        CameraMonitorService.setServiceReadyListener(null);
        CameraMonitorService svc = CameraMonitorService.getInstance();
        if (svc != null && CameraMonitorService.isRunning()) {
            // Ekran kapalı/arka planda çalışırken Activity öldürülse bile servisi KAPATMA!
            svc.detachUi();
        }
        if (viewerClient != null) viewerClient.release();
    }
}
