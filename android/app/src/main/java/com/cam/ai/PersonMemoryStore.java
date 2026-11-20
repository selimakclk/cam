package com.cam.ai;

import android.content.Context;
import android.util.Log;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Kişilerin biyometrik yüz vektörlerini (192-D), derin vücut AI gömmelerini (1024-D),
 * çoklu-ipucu imzalarını (129-D) ve kullanıcı tarafından verilen isimleri
 * telefon hafızasında atomik ikili dosya olarak kalıcı saklar.
 */
public class PersonMemoryStore {

    private static final String TAG = "PersonMemoryStore";
    private static final String FILE_NAME = "cam_ai_memory_v2.bin";
    private static final int MAGIC = 0x43414D32; // "CAM2"
    private static final int FORMAT_VERSION = 2;

    public static final int MAX_PROFILES = 32;
    public static final int MAX_SIG_PER_PROFILE = 12;
    public static final int MAX_DEEP_PER_PROFILE = 0;
    public static final int MAX_FACE_PER_PROFILE = 8;

    public static final long SESSION_PROFILE_TTL_MS = 10L * 60L * 1000L; // 10 dakika aktif oda hafızası (isimsiz)

    public static class PersonProfile {
        public int id;
        public String name;
        public long firstSeen;
        public long lastSeen;
        public final List<float[]> sigGallery = new ArrayList<>();
        public final List<float[]> deepBodyGallery = new ArrayList<>();
        public final List<float[]> faceGallery = new ArrayList<>();

        public boolean isNamed() {
            return name != null && !name.trim().isEmpty();
        }

        public boolean hasNeuralData() {
            return !faceGallery.isEmpty() || !deepBodyGallery.isEmpty();
        }

        public String displayLabel() {
            return isNamed() ? name.trim() : ("Kişi #" + id);
        }

        public PersonProfile deepCopy() {
            PersonProfile c = new PersonProfile();
            c.id = this.id;
            c.name = this.name;
            c.firstSeen = this.firstSeen;
            c.lastSeen = this.lastSeen;
            for (float[] v : this.sigGallery) c.sigGallery.add(v.clone());
            for (float[] v : this.deepBodyGallery) c.deepBodyGallery.add(v.clone());
            for (float[] v : this.faceGallery) c.faceGallery.add(v.clone());
            return c;
        }
    }

    public static class ProfileSummary {
        public final int id;
        public final String name;
        public final int sigCount;
        public final int deepCount;
        public final int faceCount;
        public final long lastSeen;

        public ProfileSummary(int id, String name, int sigCount, int deepCount, int faceCount, long lastSeen) {
            this.id = id;
            this.name = name;
            this.sigCount = sigCount;
            this.deepCount = deepCount;
            this.faceCount = faceCount;
            this.lastSeen = lastSeen;
        }

        public String displayLabel() {
            return (name != null && !name.trim().isEmpty()) ? name.trim() : ("Kişi #" + id);
        }
    }

    public static class LoadResult {
        public final List<PersonProfile> profiles = new ArrayList<>();
        public int nextTrackId = 1;
    }

    private static class SaveTaskData {
        final List<PersonProfile> profiles;
        final int nextTrackId;

        SaveTaskData(List<PersonProfile> profiles, int nextTrackId) {
            this.profiles = profiles;
            this.nextTrackId = nextTrackId;
        }
    }

    private final File storeFile;
    private final File tempFile;
    private final ExecutorService ioExecutor = Executors.newSingleThreadExecutor();
    private final AtomicReference<SaveTaskData> pendingSave = new AtomicReference<>();
    private final Object fileLock = new Object();
    private volatile long lastSaveTs = 0L;
    private volatile boolean closed = false;

    public PersonMemoryStore(Context context) {
        this(new File(context.getFilesDir(), FILE_NAME));
    }

    public PersonMemoryStore(File file) {
        this.storeFile = file;
        this.tempFile = new File(file.getParentFile(), file.getName() + ".tmp");
    }

    public LoadResult load() {
        LoadResult res = new LoadResult();
        synchronized (fileLock) {
            if (!storeFile.exists() || storeFile.length() < 12) {
                return res;
            }
            long now = System.currentTimeMillis();
            int maxId = 0;
            try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(storeFile), 16384))) {
                int magic = in.readInt();
                int ver = in.readInt();
                if (magic != MAGIC || ver != FORMAT_VERSION) {
                    Log.w(TAG, "Eski/uyumsuz hafıza dosyası sürümü, sıfırlanıyor.");
                    return res;
                }
                int savedNextId = in.readInt();
                int count = Math.max(0, Math.min(MAX_PROFILES, in.readInt()));
                for (int i = 0; i < count; i++) {
                    PersonProfile p = new PersonProfile();
                    p.id = in.readInt();
                    boolean hasName = in.readBoolean();
                    p.name = hasName ? in.readUTF() : null;
                    p.firstSeen = in.readLong();
                    p.lastSeen = in.readLong();

                    readVectorList(in, p.sigGallery, MAX_SIG_PER_PROFILE, 512);
                    readVectorList(in, p.deepBodyGallery, MAX_DEEP_PER_PROFILE, 4096);
                    readVectorList(in, p.faceGallery, MAX_FACE_PER_PROFILE, 1024);

                    if (p.id > maxId) maxId = p.id;
                    // Uygulama açılışında yalnızca kullanıcı tarafından İSİMLENDİRİLMİŞ kalıcı profilleri yükle (dünden kalan hayalet ID'leri engeller!)
                    if (p.isNamed() && (!p.sigGallery.isEmpty() || !p.faceGallery.isEmpty())) {
                        res.profiles.add(p);
                    }
                }
                res.nextTrackId = Math.max(1, Math.max(savedNextId, maxId + 1));
                if (res.nextTrackId > 999) res.nextTrackId = 1;
                Log.i(TAG, "Kalıcı AI hafızası yüklendi: " + res.profiles.size() + " kişi profili.");
            } catch (Throwable t) {
                Log.w(TAG, "Hafıza dosyası okunamadı, temiz başlatılıyor: " + t.getMessage());
                res.profiles.clear();
                res.nextTrackId = 1;
            }
        }
        return res;
    }

    public static boolean isExpired(PersonProfile p, long now) {
        if (p == null) return true;
        if (p.isNamed()) return false; // İsimlendirilmiş kişiler asla otomatik silinmez
        long age = Math.max(0L, now - p.lastSeen);
        return age > SESSION_PROFILE_TTL_MS;
    }

    public void saveDebouncedAsync(List<PersonProfile> snapshot, int nextTrackId) {
        if (closed) return;
        long now = System.currentTimeMillis();
        boolean firstInBatch = (pendingSave.getAndSet(new SaveTaskData(snapshot, nextTrackId)) == null);
        if (firstInBatch || (now - lastSaveTs > 2500L)) {
            try {
                ioExecutor.execute(() -> {
                    SaveTaskData data = pendingSave.getAndSet(null);
                    if (data != null) {
                        writeInternal(data.profiles, data.nextTrackId);
                    }
                });
            } catch (Exception ignored) {}
        }
    }

    public void saveSync(List<PersonProfile> snapshot, int nextTrackId) {
        pendingSave.set(null);
        writeInternal(snapshot, nextTrackId);
    }

    public void clearFile() {
        pendingSave.set(null);
        synchronized (fileLock) {
            if (storeFile.exists()) storeFile.delete();
            if (tempFile.exists()) tempFile.delete();
        }
    }

    private void writeInternal(List<PersonProfile> profiles, int nextTrackId) {
        synchronized (fileLock) {
            try {
                File parent = storeFile.getParentFile();
                if (parent != null && !parent.exists()) parent.mkdirs();

                try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(tempFile), 16384))) {
                    out.writeInt(MAGIC);
                    out.writeInt(FORMAT_VERSION);
                    out.writeInt(nextTrackId);

                    // Diske yalnızca İSİMLENDİRİLMİŞ kalıcı profilleri yaz (sıfır gereksiz I/O + sıfır hayalet profil!)
                    List<PersonProfile> persistable = new ArrayList<>();
                    for (PersonProfile p : profiles) {
                        if (p.isNamed() && persistable.size() < MAX_PROFILES) {
                            persistable.add(p);
                        }
                    }
                    int limit = persistable.size();
                    out.writeInt(limit);
                    for (int i = 0; i < limit; i++) {
                        PersonProfile p = persistable.get(i);
                        out.writeInt(p.id);
                        out.writeBoolean(true);
                        out.writeUTF(p.name.trim());
                        out.writeLong(p.firstSeen);
                        out.writeLong(p.lastSeen);

                        writeVectorList(out, p.sigGallery, MAX_SIG_PER_PROFILE);
                        writeVectorList(out, java.util.Collections.emptyList(), 0);
                        writeVectorList(out, p.faceGallery, MAX_FACE_PER_PROFILE);
                    }
                    out.flush();
                }

                if (storeFile.exists() && !storeFile.delete()) {
                    // Android/Windows üzerinde atomik üzerine yazma yedek yolu
                    Log.w(TAG, "Eski hafıza dosyası silinemedi, doğrudan yeniden adlandırılıyor.");
                }
                if (!tempFile.renameTo(storeFile)) {
                    // Fallback kopyalama
                    try (FileInputStream fis = new FileInputStream(tempFile);
                         FileOutputStream fos = new FileOutputStream(storeFile)) {
                        byte[] buf = new byte[8192];
                        int n;
                        while ((n = fis.read(buf)) != -1) fos.write(buf, 0, n);
                    }
                    tempFile.delete();
                }
                lastSaveTs = System.currentTimeMillis();
            } catch (Throwable t) {
                Log.w(TAG, "Kalıcı hafıza yazma hatası: " + t.getMessage());
            }
        }
    }

    private static void writeVectorList(DataOutputStream out, List<float[]> list, int maxCount) throws Exception {
        int n = Math.min(maxCount, list.size());
        out.writeInt(n);
        for (int i = 0; i < n; i++) {
            float[] vec = list.get(i);
            out.writeInt(vec.length);
            for (float v : vec) out.writeFloat(v);
        }
    }

    private static void readVectorList(DataInputStream in, List<float[]> outList, int maxCount, int maxDim) throws Exception {
        int n = in.readInt();
        if (n < 0 || n > maxCount) throw new IllegalStateException("Geçersiz vektör sayısı: " + n);
        for (int i = 0; i < n; i++) {
            int dim = in.readInt();
            if (dim <= 0 || dim > maxDim) throw new IllegalStateException("Geçersiz vektör boyutu: " + dim);
            float[] vec = new float[dim];
            for (int j = 0; j < dim; j++) vec[j] = in.readFloat();
            outList.add(vec);
        }
    }

    public void close(List<PersonProfile> finalSnapshot, int nextTrackId) {
        closed = true;
        if (finalSnapshot != null) {
            saveSync(finalSnapshot, nextTrackId);
        }
        ioExecutor.shutdown();
        try {
            ioExecutor.awaitTermination(400, TimeUnit.MILLISECONDS);
        } catch (InterruptedException ignored) {}
    }
}
