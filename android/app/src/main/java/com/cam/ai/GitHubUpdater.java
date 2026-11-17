package com.cam.ai;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.net.Uri;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.view.Gravity;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AlertDialog;
import androidx.core.content.FileProvider;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

/**
 * GitHub Releases (https://github.com/selimakclk/cam) üzerinden uygulama içi tek dokunuşla
 * sürüm kontrolü, APK indirme ve doğrudan Android paket yükleyici başlatma motoru.
 */
public class GitHubUpdater {

    private static final String RELEASES_API_URL = "https://api.github.com/repos/selimakclk/cam/releases/latest";
    private static final String FALLBACK_APK_URL = "https://github.com/selimakclk/cam/releases/latest/download/CAM.apk";

    public static class ReleaseInfo {
        public final String tagName;
        public final String version;
        public final String notes;
        public final String apkUrl;
        public final long apkSizeBytes;

        public ReleaseInfo(String tagName, String version, String notes, String apkUrl, long apkSizeBytes) {
            this.tagName = tagName;
            this.version = version;
            this.notes = notes;
            this.apkUrl = apkUrl;
            this.apkSizeBytes = apkSizeBytes;
        }
    }

    public interface BadgeCallback {
        void onResult(boolean hasUpdate, String latestVersion);
    }

    private final Handler uiHandler = new Handler(Looper.getMainLooper());
    private File pendingInstallApk = null;
    private volatile boolean downloadCancelled = false;

    public static String getCurrentVersionName(Context context) {
        try {
            PackageInfo pi = context.getPackageManager().getPackageInfo(context.getPackageName(), 0);
            return (pi.versionName != null) ? pi.versionName : "2.0";
        } catch (Exception e) {
            return "2.0";
        }
    }

    /**
     * Sürümleri sayısal parçalarına göre karşılaştırır ("2.1" > "2.0", "2.10" > "2.2", "v2.0" == "2.0").
     */
    public static int compareVersions(String v1, String v2) {
        String a = normalizeVersion(v1);
        String b = normalizeVersion(v2);
        String[] pa = a.isEmpty() ? new String[]{"0"} : a.split("[^0-9]+");
        String[] pb = b.isEmpty() ? new String[]{"0"} : b.split("[^0-9]+");
        int max = Math.max(pa.length, pb.length);
        for (int i = 0; i < max; i++) {
            int na = parseIntSafe(i < pa.length ? pa[i] : "0");
            int nb = parseIntSafe(i < pb.length ? pb[i] : "0");
            if (na != nb) return Integer.compare(na, nb);
        }
        return 0;
    }

    private static String normalizeVersion(String v) {
        if (v == null) return "";
        String s = v.trim();
        if (s.startsWith("v") || s.startsWith("V")) {
            s = s.substring(1);
        }
        return s.trim();
    }

    private static int parseIntSafe(String s) {
        if (s == null || s.isEmpty()) return 0;
        try { return Integer.parseInt(s); } catch (Exception e) { return 0; }
    }

    public void checkSilently(Context context, BadgeCallback callback) {
        final String currentVer = getCurrentVersionName(context);
        new Thread(() -> {
            try {
                ReleaseInfo info = fetchLatestRelease();
                boolean newer = (info != null && compareVersions(info.version, currentVer) > 0);
                String ver = (info != null) ? info.version : currentVer;
                uiHandler.post(() -> callback.onResult(newer, ver));
            } catch (Exception ignored) {}
        }, "GitHubBadgeCheck").start();
    }

    public void checkInteractive(Activity activity) {
        if (activity == null || activity.isFinishing() || activity.isDestroyed()) return;

        AlertDialog checkingDlg = new AlertDialog.Builder(activity)
                .setTitle("Güncelleme Kontrolü")
                .setMessage("GitHub sunucusundan en yeni sürüm kontrol ediliyor...")
                .setCancelable(false)
                .show();

        final String currentVer = getCurrentVersionName(activity);
        new Thread(() -> {
            try {
                ReleaseInfo info = fetchLatestRelease();
                uiHandler.post(() -> {
                    if (checkingDlg.isShowing()) checkingDlg.dismiss();
                    if (activity.isFinishing() || activity.isDestroyed()) return;

                    if (info == null) {
                        new AlertDialog.Builder(activity)
                                .setTitle("Henüz Sürüm Yayınlanmamış")
                                .setMessage("GitHub üzerinde henüz yayınlanmış bir APK sürümü bulunamadı.\nMevcut sürüm: v" + currentVer)
                                .setPositiveButton("Tamam", null)
                                .show();
                        return;
                    }

                    boolean isNewer = compareVersions(info.version, currentVer) > 0;
                    String sizeMb = info.apkSizeBytes > 0
                            ? String.format(Locale.US, "%.1f MB", info.apkSizeBytes / (1024.0 * 1024.0))
                            : "APK";

                    if (isNewer) {
                        String msg = "Yeni Sürüm: " + info.tagName + " (" + sizeMb + ")\n"
                                + "Mevcut Sürüm: v" + currentVer + "\n\n"
                                + (info.notes.isEmpty() ? "Performans ve güvenlik iyileştirmeleri." : info.notes);
                        new AlertDialog.Builder(activity)
                                .setTitle("Yeni Güncelleme Var!")
                                .setMessage(msg)
                                .setPositiveButton("İndir ve Güncelle", (d, w) -> startDownloadAndInstall(activity, info))
                                .setNegativeButton("Daha Sonra", null)
                                .show();
                    } else {
                        new AlertDialog.Builder(activity)
                                .setTitle("En Güncel Sürüm Kurulu")
                                .setMessage("Zaten en yeni CAM AI sürümünü (v" + currentVer + ") kullanıyorsunuz.")
                                .setPositiveButton("Tamam", null)
                                .setNeutralButton("Yeniden İndir (" + info.tagName + ")", (d, w) -> startDownloadAndInstall(activity, info))
                                .show();
                    }
                });
            } catch (Exception e) {
                uiHandler.post(() -> {
                    if (checkingDlg.isShowing()) checkingDlg.dismiss();
                    if (activity.isFinishing() || activity.isDestroyed()) return;
                    new AlertDialog.Builder(activity)
                            .setTitle("Bağlantı Hatası")
                            .setMessage("GitHub'a ulaşılamadı. İnternet bağlantınızı kontrol edin.\n(" + e.getMessage() + ")")
                            .setPositiveButton("Tamam", null)
                            .show();
                });
            }
        }, "GitHubUpdateCheck").start();
    }

    public ReleaseInfo fetchLatestRelease() throws Exception {
        HttpURLConnection conn = null;
        try {
            URL url = new URL(RELEASES_API_URL);
            conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setRequestProperty("Accept", "application/vnd.github+json");
            conn.setRequestProperty("User-Agent", "CAM-AI-Android");
            conn.setConnectTimeout(7000);
            conn.setReadTimeout(7000);

            int code = conn.getResponseCode();
            if (code == 404) {
                return null;
            }
            if (code != 200) {
                throw new Exception("HTTP " + code);
            }

            String bodyStr;
            try (InputStream in = conn.getInputStream(); ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
                byte[] buf = new byte[4096];
                int n;
                while ((n = in.read(buf)) != -1) baos.write(buf, 0, n);
                bodyStr = new String(baos.toByteArray(), StandardCharsets.UTF_8);
            }

            return parseReleaseJson(bodyStr);
        } finally {
            if (conn != null) conn.disconnect();
        }
    }

    public static ReleaseInfo parseReleaseJson(String jsonStr) throws Exception {
        JSONObject root = new JSONObject(jsonStr);
        String tag = root.optString("tag_name", "v2.0");
        String version = normalizeVersion(tag);
        String notes = root.optString("body", "").trim();
        String apkUrl = FALLBACK_APK_URL;
        long apkSize = 0L;

        JSONArray assets = root.optJSONArray("assets");
        if (assets != null) {
            for (int i = 0; i < assets.length(); i++) {
                JSONObject a = assets.getJSONObject(i);
                String name = a.optString("name", "");
                if (name.toLowerCase(Locale.US).endsWith(".apk")) {
                    apkUrl = a.optString("browser_download_url", apkUrl);
                    apkSize = a.optLong("size", 0L);
                    break;
                }
            }
        }
        return new ReleaseInfo(tag, version, notes, apkUrl, apkSize);
    }

    private void startDownloadAndInstall(Activity activity, ReleaseInfo info) {
        if (activity.isFinishing() || activity.isDestroyed()) return;
        downloadCancelled = false;

        int pad = Math.round(20 * activity.getResources().getDisplayMetrics().density);
        LinearLayout box = new LinearLayout(activity);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(pad, pad, pad, pad);

        final TextView txtProg = new TextView(activity);
        txtProg.setText("İndirme başlatılıyor...");
        txtProg.setGravity(Gravity.CENTER_HORIZONTAL);
        txtProg.setPadding(0, 0, 0, pad / 2);

        final ProgressBar bar = new ProgressBar(activity, null, android.R.attr.progressBarStyleHorizontal);
        bar.setMax(100);
        bar.setProgress(0);

        box.addView(txtProg);
        box.addView(bar);

        AlertDialog progDlg = new AlertDialog.Builder(activity)
                .setTitle("Güncelleme İndiriliyor (" + info.tagName + ")")
                .setView(box)
                .setCancelable(false)
                .setNegativeButton("İptal", (d, w) -> downloadCancelled = true)
                .create();
        progDlg.show();

        new Thread(() -> {
            File outFile = null;
            try {
                File dir = new File(activity.getCacheDir(), "updates");
                if (!dir.exists()) dir.mkdirs();
                File[] oldFiles = dir.listFiles();
                if (oldFiles != null) {
                    for (File f : oldFiles) f.delete();
                }

                outFile = new File(dir, "CAM-update.apk");
                HttpURLConnection conn = openFollowingRedirects(info.apkUrl);
                long total = conn.getContentLength();
                if (total <= 0) total = info.apkSizeBytes;

                try (InputStream in = conn.getInputStream(); FileOutputStream fos = new FileOutputStream(outFile)) {
                    byte[] buf = new byte[16384];
                    long downloaded = 0;
                    long lastUiTs = 0;
                    int n;
                    while ((n = in.read(buf)) != -1) {
                        if (downloadCancelled) {
                            outFile.delete();
                            uiHandler.post(() -> {
                                if (progDlg.isShowing()) progDlg.dismiss();
                            });
                            return;
                        }
                        fos.write(buf, 0, n);
                        downloaded += n;

                        long now = System.currentTimeMillis();
                        if (now - lastUiTs >= 120) {
                            lastUiTs = now;
                            final long dlSnap = downloaded;
                            final long totalSnap = total;
                            uiHandler.post(() -> {
                                if (totalSnap > 0) {
                                    int pct = (int) Math.min(100, (dlSnap * 100L) / totalSnap);
                                    bar.setProgress(pct);
                                    txtProg.setText(String.format(Locale.US, "%%%d • %.1f / %.1f MB",
                                            pct, dlSnap / (1024.0 * 1024.0), totalSnap / (1024.0 * 1024.0)));
                                } else {
                                    bar.setIndeterminate(true);
                                    txtProg.setText(String.format(Locale.US, "%.1f MB indirildi...",
                                            dlSnap / (1024.0 * 1024.0)));
                                }
                            });
                        }
                    }
                    fos.flush();
                } finally {
                    conn.disconnect();
                }

                if (!isValidApkFile(outFile)) {
                    throw new Exception("İndirilen APK dosyası geçersiz veya eksik.");
                }

                final File readyApk = outFile;
                uiHandler.post(() -> {
                    if (progDlg.isShowing()) progDlg.dismiss();
                    if (!activity.isFinishing() && !activity.isDestroyed()) {
                        triggerApkInstaller(activity, readyApk);
                    }
                });
            } catch (Exception e) {
                if (outFile != null && outFile.exists()) outFile.delete();
                uiHandler.post(() -> {
                    if (progDlg.isShowing()) progDlg.dismiss();
                    if (!activity.isFinishing() && !activity.isDestroyed()) {
                        Toast.makeText(activity, "İndirme hatası: " + e.getMessage(), Toast.LENGTH_LONG).show();
                    }
                });
            }
        }, "GitHubApkDownloader").start();
    }

    private HttpURLConnection openFollowingRedirects(String urlStr) throws Exception {
        String current = urlStr;
        for (int i = 0; i < 6; i++) {
            HttpURLConnection conn = (HttpURLConnection) new URL(current).openConnection();
            conn.setInstanceFollowRedirects(false);
            conn.setRequestProperty("User-Agent", "CAM-AI-Android");
            conn.setConnectTimeout(10000);
            conn.setReadTimeout(20000);
            int code = conn.getResponseCode();
            if (code == HttpURLConnection.HTTP_MOVED_PERM
                    || code == HttpURLConnection.HTTP_MOVED_TEMP
                    || code == HttpURLConnection.HTTP_SEE_OTHER
                    || code == 307 || code == 308) {
                String loc = conn.getHeaderField("Location");
                conn.disconnect();
                if (loc == null || loc.isEmpty()) throw new Exception("Yönlendirme adresi boş");
                current = new URL(new URL(current), loc).toString();
                continue;
            }
            if (code != 200) {
                conn.disconnect();
                throw new Exception("HTTP " + code);
            }
            return conn;
        }
        throw new Exception("Çok fazla yönlendirme");
    }

    private boolean isValidApkFile(File f) {
        if (f == null || !f.exists() || f.length() < 500_000L) return false;
        try (java.io.FileInputStream fis = new java.io.FileInputStream(f)) {
            int b0 = fis.read();
            int b1 = fis.read();
            return b0 == 0x50 && b1 == 0x4B; // ZIP / APK 'PK' imzası
        } catch (Exception e) {
            return false;
        }
    }

    private void triggerApkInstaller(Activity activity, File apkFile) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && !activity.getPackageManager().canRequestPackageInstalls()) {
            pendingInstallApk = apkFile;
            new AlertDialog.Builder(activity)
                    .setTitle("Kurulum İzni Gerekli")
                    .setMessage("Güncellemeyi kurabilmek için açılan ekrandan 'Bu kaynaktan izin ver' seçeneğini açın; geri döndüğünüzde kurulum otomatik başlayacaktır.")
                    .setPositiveButton("Ayarlara Git", (d, w) -> {
                        Intent permIntent = new Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES,
                                Uri.parse("package:" + activity.getPackageName()));
                        activity.startActivity(permIntent);
                    })
                    .setNegativeButton("İptal", (d, w) -> pendingInstallApk = null)
                    .show();
            return;
        }
        launchInstallIntent(activity, apkFile);
    }

    public void onActivityResumed(Activity activity) {
        if (pendingInstallApk != null && pendingInstallApk.exists()) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O || activity.getPackageManager().canRequestPackageInstalls()) {
                File apk = pendingInstallApk;
                pendingInstallApk = null;
                launchInstallIntent(activity, apk);
            }
        }
    }

    private void launchInstallIntent(Activity activity, File apkFile) {
        try {
            Uri uri = FileProvider.getUriForFile(activity, activity.getPackageName() + ".fileprovider", apkFile);
            Intent intent = new Intent(Intent.ACTION_VIEW);
            intent.setDataAndType(uri, "application/vnd.android.package-archive");
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_ACTIVITY_NEW_TASK);
            activity.startActivity(intent);
        } catch (Exception e) {
            Toast.makeText(activity, "Kurulum başlatılamadı: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }
}
