package com.cam.ai;

import static org.junit.Assert.*;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;

@RunWith(AndroidJUnit4.class)
public class PersonDetectorInstrumentedTest {

    @Test
    public void testPackageAndModelLoad() {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        assertEquals("com.cam.ai", ctx.getPackageName());

        PersonDetectorEngine engine = new PersonDetectorEngine(ctx);
        assertTrue("AI Model baslatilamadi: " + engine.getInitError(), engine.isInitialized());
        engine.close();
    }

    @Test
    public void testPersonDetectionLockAndLostAlarm() throws Exception {
        Context targetCtx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Context testCtx = InstrumentationRegistry.getInstrumentation().getContext();

        PersonDetectorEngine engine = new PersonDetectorEngine(targetCtx);
        assertTrue(engine.isInitialized());

        Bitmap roomBmp;
        try (InputStream is = testCtx.getAssets().open("test_person_room.jpg")) {
            roomBmp = BitmapFactory.decodeStream(is);
        }
        assertNotNull("Test gorseli yuklenemedi", roomBmp);

        // 1. Insan tespiti yap
        PersonDetectorEngine.AnalysisOutput out = engine.processFrame(roomBmp);
        assertNotNull(out.persons);
        assertTrue("Insan algilanamadi! Tespit sayisi: " + out.persons.size(), out.persons.size() >= 1);

        TrackedPerson person = out.persons.get(0);

        // 2. Kisi olmayan bos duvara tiklayinca KESINLIKLE kilitlenmemeli
        PersonDetectorEngine.LockedState emptyLock = new PersonDetectorEngine.LockedState();
        // Kisi kutusunun disinda kesin bos bir nokta bul
        float emptyX = person.centerX() < 0.5f ? 0.92f : 0.08f;
        float emptyY = 0.10f;
        if (!person.contains(emptyX, emptyY) && person.distanceToPoint(emptyX, emptyY) > 0.08f) {
            PersonDetectorEngine.LockStatus emptyRes = engine.tryLockAt(emptyX, emptyY, emptyLock);
            assertEquals("Bos alana kilitlenmemeli!", PersonDetectorEngine.LockStatus.EMPTY_AREA, emptyRes);
        }

        // 3. Tespit edilen kisinin uzerine tiklayinca kilitlenmeli
        PersonDetectorEngine.LockedState lockedOut = new PersonDetectorEngine.LockedState();
        PersonDetectorEngine.LockStatus lockRes = engine.tryLockAt(person.centerX(), person.centerY(), lockedOut);
        assertEquals("Kisiye tiklayinca kilitlenmeli!", PersonDetectorEngine.LockStatus.LOCKED, lockRes);

        // 4. Ayni karede takip devam etmeli (alarm vermemeli)
        PersonDetectorEngine.AnalysisOutput out2 = engine.processFrame(roomBmp);
        assertNotNull(out2.locked);
        assertFalse(out2.targetJustLost);
        assertFalse(out2.locked.lost);

        // 5. Kisi kadrajdan cikinca (bos siyah kareler + 1.85s akilli gizlenme tamponu sonrasi) hedef kayip alarmi tetiklenmeli
        Bitmap emptyBmp = Bitmap.createBitmap(480, 640, Bitmap.Config.ARGB_8888);
        emptyBmp.eraseColor(Color.BLACK);

        Thread.sleep(1900);
        PersonDetectorEngine.AnalysisOutput outLost = engine.processFrame(emptyBmp);
        assertNotNull(outLost.locked);
        assertTrue("Kisi ciktiginda targetJustLost tetiklenmeli!", outLost.targetJustLost);
        assertTrue("Kisi ciktiginda locked.lost true olmali!", outLost.locked.lost);

        // 6. Kisi tekrar kadraja girince Re-ID ile otomatik geri kilitlenmeli!
        Bitmap roomBmp2;
        try (InputStream is = testCtx.getAssets().open("test_person_room.jpg")) {
            roomBmp2 = BitmapFactory.decodeStream(is);
        }
        PersonDetectorEngine.AnalysisOutput outRecovered = engine.processFrame(roomBmp2);
        assertNotNull(outRecovered.locked);
        assertTrue("Kisi geri donunce otomatik geri yakalanmali!", outRecovered.targetJustRecovered);
        assertFalse(outRecovered.locked.lost);

        engine.close();
    }

    @Test
    public void testStreamServerPasswordProtection() throws Exception {
        CameraStreamServer server = new CameraStreamServer(18080, "54321");
        server.start();
        try {
            URL badUrl = new URL("http://127.0.0.1:18080/ping?code=99999");
            HttpURLConnection c1 = (HttpURLConnection) badUrl.openConnection();
            assertEquals(403, c1.getResponseCode());
            c1.disconnect();

            URL okUrl = new URL("http://127.0.0.1:18080/ping?code=54321");
            HttpURLConnection c2 = (HttpURLConnection) okUrl.openConnection();
            assertEquals(200, c2.getResponseCode());
            c2.disconnect();
        } finally {
            server.stop();
        }
    }

    @Test
    public void testRemoteCommandAndAiStateSync() throws Exception {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        PersonDetectorEngine engine = new PersonDetectorEngine(ctx);
        CameraStreamServer server = new CameraStreamServer(18081, "54321");

        try (InputStream is = InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("test_person_room.jpg")) {
            Bitmap roomBmp = BitmapFactory.decodeStream(is);
            assertNotNull(roomBmp);

            PersonDetectorEngine.AnalysisOutput out1 = engine.processFrame(roomBmp);
            assertFalse(out1.persons.isEmpty());
            TrackedPerson person = out1.persons.get(0);

            server.setOnRemoteCommandListener(new CameraStreamServer.OnRemoteCommandListener() {
                @Override
                public String onRemoteLock(float nx, float ny) {
                    PersonDetectorEngine.LockedState snap = new PersonDetectorEngine.LockedState();
                    PersonDetectorEngine.LockStatus res = engine.tryLockAt(nx, ny, snap);
                    return (res == PersonDetectorEngine.LockStatus.LOCKED) ? "LOCKED" : "EMPTY";
                }
                @Override public void onRemoteUnlock() { engine.unlock(); }
                @Override public void onRemoteDismissAlarm() {}
                @Override public void onRemoteFlipCamera() {}
                @Override public void onRemoteToggleTorch() {}
            });
            server.start();
            server.updateAiState(roomBmp.getWidth(), roomBmp.getHeight(), false, false, out1.persons, out1.locked);

            JSONObject state1 = new JSONObject(httpGet("http://127.0.0.1:18081/state?code=54321"));
            assertTrue(state1.getJSONArray("persons").length() >= 1);
            assertTrue(state1.isNull("locked"));

            // Boş alana uzaktan kilitleme isteği reddedilmeli (EMPTY)
            JSONObject emptyCmd = new JSONObject(httpGet("http://127.0.0.1:18081/cmd?code=54321&action=lock&nx=0.88&ny=0.35"));
            assertEquals("EMPTY", emptyCmd.getString("status"));

            // Gerçek kişi üzerine uzaktan kilitleme başarılı olmalı (LOCKED) ve /state güncellenmeli
            String lockUrl = String.format(java.util.Locale.US,
                    "http://127.0.0.1:18081/cmd?code=54321&action=lock&nx=%.4f&ny=%.4f",
                    person.centerX(), person.centerY());
            JSONObject lockCmd = new JSONObject(httpGet(lockUrl));
            assertEquals("LOCKED", lockCmd.getString("status"));

            PersonDetectorEngine.AnalysisOutput out2 = engine.processFrame(roomBmp);
            server.updateAiState(roomBmp.getWidth(), roomBmp.getHeight(), false, false, out2.persons, out2.locked);
            JSONObject state2 = new JSONObject(httpGet("http://127.0.0.1:18081/state?code=54321"));
            assertFalse(state2.isNull("locked"));
            assertEquals(person.id, state2.getJSONObject("locked").getInt("id"));
        } finally {
            server.stop();
            engine.close();
        }
    }

    private String httpGet(String urlStr) throws Exception {
        HttpURLConnection conn = (HttpURLConnection) new URL(urlStr).openConnection();
        try {
            assertEquals(200, conn.getResponseCode());
            try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                StringBuilder sb = new StringBuilder();
                String line;
                while ((line = br.readLine()) != null) sb.append(line);
                return sb.toString();
            }
        } finally {
            conn.disconnect();
        }
    }

    @Test
    public void testTargetExitAndReEntryRecoveryMemory() throws Exception {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        PersonDetectorEngine engine = new PersonDetectorEngine(ctx);

        try (InputStream is = InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("test_person_room.jpg")) {
            Bitmap roomBmp = BitmapFactory.decodeStream(is);
            assertNotNull(roomBmp);

            // 1. Kişiyi tespit et ve kilitle
            PersonDetectorEngine.AnalysisOutput out1 = engine.processFrame(roomBmp);
            assertFalse("Odada kişi bulunamadi", out1.persons.isEmpty());
            TrackedPerson person = out1.persons.get(0);

            PersonDetectorEngine.LockedState lockedSnap = new PersonDetectorEngine.LockedState();
            assertEquals(PersonDetectorEngine.LockStatus.LOCKED,
                    engine.tryLockAt(person.centerX(), person.centerY(), lockedSnap));
            int lockedId = lockedSnap.id;

            // 2. Kişi kameradan çıkınca kayıp alarmı tetiklenmeli (hafıza silinmeden)
            Bitmap blank = Bitmap.createBitmap(480, 360, Bitmap.Config.ARGB_8888);
            blank.eraseColor(Color.BLACK);
            engine.processFrame(blank);
            Thread.sleep(1900);
            PersonDetectorEngine.AnalysisOutput lostOut = engine.processFrame(blank);
            assertTrue("Kişi çıkınca kayıp uyarısı vermeli", lostOut.targetJustLost);
            assertTrue(lostOut.locked != null && lostOut.locked.lost);

            // 3. Kişi tekrar kameraya girince aynı ID ile Re-ID hafızasından geri yakalamalı
            PersonDetectorEngine.AnalysisOutput rec1 = engine.processFrame(roomBmp);
            PersonDetectorEngine.AnalysisOutput rec2 = (rec1.targetJustRecovered) ? rec1 : engine.processFrame(roomBmp);
            assertTrue("Geri giren kişi Re-ID hafızasıyla tanınmalı", rec2.targetJustRecovered || !rec2.locked.lost);
            assertFalse(rec2.locked.lost);
            assertEquals(lockedId, rec2.locked.id);
            blank.recycle();
        } finally {
            engine.close();
        }
    }

    @Test
    public void testUnlockedRoomMemoryAndFastTracking() throws Exception {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        PersonDetectorEngine engine = new PersonDetectorEngine(ctx);

        try (InputStream is = InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("test_person_room.jpg")) {
            Bitmap roomBmp = BitmapFactory.decodeStream(is);
            assertNotNull(roomBmp);

            // 1. Serbest kişi #ID hafızaya alınmalı
            PersonDetectorEngine.AnalysisOutput out1 = engine.processFrame(roomBmp);
            assertFalse(out1.persons.isEmpty());
            int originalId = out1.persons.get(0).id;

            // 2. Hızlı 30 FPS optik takip adımını doğrula
            PersonDetectorEngine.LockedState snap = new PersonDetectorEngine.LockedState();
            assertEquals(PersonDetectorEngine.LockStatus.LOCKED,
                    engine.tryLockFast(out1.persons.get(0).centerX(), out1.persons.get(0).centerY(), snap));
            PersonDetectorEngine.LockedState fast = engine.fastTrackFrame(roomBmp);
            assertNotNull(fast);
            assertEquals(originalId, fast.id);
            engine.unlock();

            // 3. Kişi kadrajdan çıkıp aktif liste temizlense bile tekrar girdiğinde aynı ID korunmalı
            Bitmap blank = Bitmap.createBitmap(480, 360, Bitmap.Config.ARGB_8888);
            blank.eraseColor(Color.BLACK);
            PersonDetectorEngine.AnalysisOutput outEmpty = engine.processFrame(blank);
            assertTrue(outEmpty.persons.isEmpty());

            PersonDetectorEngine.AnalysisOutput outBack = engine.processFrame(roomBmp);
            assertFalse(outBack.persons.isEmpty());
            assertEquals("Odadan çıkıp giren kişinin aynı ID korunmalı", originalId, outBack.persons.get(0).id);
            blank.recycle();
        } finally {
            engine.close();
        }
    }

    @Test
    public void testViewerClientConnectsAndReceivesLiveVideoAndState() throws Exception {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        CameraStreamServer server = new CameraStreamServer(8080, "54321");

        // Gerçek JPEG kare oluştur
        Bitmap src = Bitmap.createBitmap(160, 120, Bitmap.Config.ARGB_8888);
        src.eraseColor(Color.rgb(30, 120, 220));
        java.io.ByteArrayOutputStream jpegBaos = new java.io.ByteArrayOutputStream();
        src.compress(Bitmap.CompressFormat.JPEG, 70, jpegBaos);
        byte[] jpegBytes = jpegBaos.toByteArray();
        src.recycle();

        final java.util.concurrent.CountDownLatch connLatch = new java.util.concurrent.CountDownLatch(1);
        final java.util.concurrent.CountDownLatch frameLatch = new java.util.concurrent.CountDownLatch(1);
        final java.util.concurrent.CountDownLatch stateLatch = new java.util.concurrent.CountDownLatch(1);
        final java.util.concurrent.CountDownLatch lockLatch = new java.util.concurrent.CountDownLatch(1);
        final boolean[] lockResultHolder = new boolean[1];

        server.setOnRemoteCommandListener(new CameraStreamServer.OnRemoteCommandListener() {
            @Override public String onRemoteLock(float nx, float ny) { return "LOCKED"; }
            @Override public void onRemoteUnlock() {}
            @Override public void onRemoteDismissAlarm() {}
            @Override public void onRemoteFlipCamera() {}
            @Override public void onRemoteToggleTorch() {}
        });

        ViewerStreamClient client = new ViewerStreamClient(ctx, new ViewerStreamClient.Listener() {
            @Override public void onSearching(String message) {}
            @Override public void onConnected(String ip, String deviceModel) { connLatch.countDown(); }
            @Override public void onWrongPassword() {}
            @Override public void onFrame(Bitmap bitmap) {
                if (bitmap != null && bitmap.getWidth() == 160 && bitmap.getHeight() == 120) {
                    frameLatch.countDown();
                }
            }
            @Override public void onAiState(int w, int h, boolean isFront, boolean alarm,
                                            java.util.List<TrackedPerson> persons, PersonDetectorEngine.LockedState locked) {
                if (persons != null && !persons.isEmpty()) stateLatch.countDown();
            }
            @Override public void onDisconnected(String reason) {}
            @Override public void onRemoteLockResult(boolean locked) {
                lockResultHolder[0] = locked;
                lockLatch.countDown();
            }
        });

        try {
            server.start();
            server.updateFrame(jpegBytes);
            java.util.List<TrackedPerson> mockPersons = new java.util.ArrayList<>();
            mockPersons.add(new TrackedPerson(7, 0.2f, 0.2f, 0.3f, 0.5f, 0.85f));
            server.updateAiState(480, 640, false, false, mockPersons, null);

            // Doğrudan otomatik ağ keşfini başlat
            client.startAutoConnect("54321");

            assertTrue("İzleyici otomatik bağlanamadı", connLatch.await(8, java.util.concurrent.TimeUnit.SECONDS));
            assertTrue("İzleyici MJPEG canlı kare alamadı", frameLatch.await(6, java.util.concurrent.TimeUnit.SECONDS));
            assertTrue("İzleyici AI durumunu alamadı", stateLatch.await(4, java.util.concurrent.TimeUnit.SECONDS));

            client.sendLock(0.35f, 0.45f);
            assertTrue("Uzaktan kilit komutu yanıtı gelmedi", lockLatch.await(4, java.util.concurrent.TimeUnit.SECONDS));
            assertTrue("Uzaktan kilit LOCKED olmalıydı", lockResultHolder[0]);
            assertTrue("Sunucu aktif izleyiciyi görmeliydi", server.hasActiveViewer());
        } finally {
            client.release();
            server.stop();
        }
    }

    @Test
    public void testUpdaterVersionComparisonAndFileProvider() throws Exception {
        assertTrue(GitHubUpdater.compareVersions("2.1", "2.0") > 0);
        assertTrue(GitHubUpdater.compareVersions("v2.10", "2.2") > 0);
        assertEquals(0, GitHubUpdater.compareVersions("v2.0", "2.0"));
        assertTrue(GitHubUpdater.compareVersions("1.9.9", "2.0") < 0);

        String sampleJson = "{\"tag_name\":\"v2.3\",\"body\":\"Hiz artisi\",\"assets\":[{\"name\":\"CAM.apk\",\"browser_download_url\":\"https://github.com/selimakclk/cam/releases/download/v2.3/CAM.apk\",\"size\":25300000}]}";
        GitHubUpdater.ReleaseInfo info = GitHubUpdater.parseReleaseJson(sampleJson);
        assertEquals("2.3", info.version);
        assertEquals("https://github.com/selimakclk/cam/releases/download/v2.3/CAM.apk", info.apkUrl);
        assertEquals(25300000L, info.apkSizeBytes);
    }

    @Test
    public void testLightingAndPositionShiftReIdMemory() throws Exception {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        PersonDetectorEngine engine = new PersonDetectorEngine(ctx);

        try (InputStream is = InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("test_person_room.jpg")) {
            Bitmap roomBmp = BitmapFactory.decodeStream(is);
            assertNotNull(roomBmp);

            // 1. Kişiyi solda tespit et, birkaç kareyle hafıza bankasını doldur ve kilitle
            PersonDetectorEngine.AnalysisOutput out1 = engine.processFrame(roomBmp);
            assertFalse(out1.persons.isEmpty());
            TrackedPerson person = out1.persons.get(0);

            PersonDetectorEngine.LockedState snap = new PersonDetectorEngine.LockedState();
            assertEquals(PersonDetectorEngine.LockStatus.LOCKED,
                    engine.tryLockAt(person.centerX(), person.centerY(), snap));
            int lockedId = snap.id;

            // 2. Kişi kadrajdan tamamen çıksın (alarm tetiklensin)
            Bitmap blank = Bitmap.createBitmap(480, 360, Bitmap.Config.ARGB_8888);
            blank.eraseColor(Color.BLACK);
            engine.processFrame(blank);
            Thread.sleep(1900);
            PersonDetectorEngine.AnalysisOutput lostOut = engine.processFrame(blank);
            assertTrue(lostOut.locked != null && lostOut.locked.lost);

            // 3. Kişiyi %55 karartılmış/sıcak ışık değişimli sahnede sağ tarafa kaydırarak geri sok
            int rw = roomBmp.getWidth(), rh = roomBmp.getHeight();
            Bitmap shiftedDark = Bitmap.createBitmap(rw, rh, Bitmap.Config.ARGB_8888);
            android.graphics.Canvas c = new android.graphics.Canvas(shiftedDark);
            c.drawColor(Color.rgb(28, 30, 34));
            android.graphics.Paint darkPaint = new android.graphics.Paint();
            android.graphics.ColorMatrix cm = new android.graphics.ColorMatrix();
            cm.setScale(0.58f, 0.54f, 0.48f, 1.0f); // %45 karartılmış sıcak oda ışığı
            darkPaint.setColorFilter(new android.graphics.ColorMatrixColorFilter(cm));

            // Soldaki kişiyi alıp sahnenin sağına yerleştir (konum tamamen değişti!)
            int srcLeft = Math.max(0, Math.round((person.nx - 0.04f) * rw));
            int srcRight = Math.min(rw, Math.round((person.nx + person.nw + 0.04f) * rw));
            int srcTop = Math.max(0, Math.round((person.ny - 0.03f) * rh));
            int srcBottom = Math.min(rh, Math.round((person.ny + person.nh + 0.03f) * rh));
            android.graphics.Rect srcRect = new android.graphics.Rect(srcLeft, srcTop, srcRight, srcBottom);
            int dstLeft = Math.max(0, Math.round(rw * 0.55f));
            android.graphics.Rect dstRect = new android.graphics.Rect(
                    dstLeft, srcTop, Math.min(rw, dstLeft + srcRect.width()), srcBottom);
            c.drawBitmap(roomBmp, srcRect, dstRect, darkPaint);

            PersonDetectorEngine.AnalysisOutput rec = engine.processFrame(shiftedDark);
            if (rec.locked != null && rec.locked.lost) {
                rec = engine.processFrame(shiftedDark);
            }
            assertNotNull("Kişi karanlık/sağ konuma geçince de bulunmalı", rec.locked);
            assertFalse("Farklı ışık ve farklı konumdaki aynı kişi Re-ID ile kurtarılmalı", rec.locked.lost);
            assertEquals(lockedId, rec.locked.id);

            shiftedDark.recycle();
            blank.recycle();
        } finally {
            engine.close();
        }
    }

    @Test
    public void testDeepReIdAndFaceAiInitialization() throws Exception {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        java.io.File memFile = new java.io.File(ctx.getCacheDir(), "test_neural_init.bin");
        if (memFile.exists()) memFile.delete();
        PersonDetectorEngine engine = new PersonDetectorEngine(ctx, memFile);

        try (InputStream is = InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("test_person_room.jpg")) {
            assertTrue("EfficientDet-Lite0 INT8 motoru baslatilamadi", engine.isInitialized());
            assertTrue("OSNet MSMT17 512-D Person Re-ID hazir olmali", engine.isReidAiReady());
            assertTrue("BlazeFace + MobileFaceNet asenkron yuz AI hazir olmali", engine.isFaceAiReady());

            Bitmap roomBmp = BitmapFactory.decodeStream(is);
            assertNotNull(roomBmp);

            PersonDetectorEngine.AnalysisOutput out = engine.processFrame(roomBmp);
            assertFalse("Odada kisi algilanmali", out.persons.isEmpty());
            assertNotNull("129-D Re-ID imzasi cikarilmali", out.persons.get(0).signature);

            // Asenkron OSNet + yuz AI'nin arka planda isi bitirmesini bekle ve ikisinin de ogrendigini dogrula
            engine.waitForPendingIdentityWork(3000);
            PersonDetectorEngine.AnalysisOutput out2 = engine.processFrame(roomBmp);
            assertFalse(out2.persons.isEmpty());
            assertTrue("Asenkron OSNet 512-D Re-ID ogrenilmis olmali", out2.persons.get(0).hasDeepAi);
            assertTrue("Asenkron biyometrik yuz AI ogrenilmis olmali", out2.persons.get(0).hasFaceAi);
        } finally {
            engine.close();
            if (memFile.exists()) memFile.delete();
        }
    }

    @Test
    public void testPersistentMemoryAndNamingAcrossAppRestart() throws Exception {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        java.io.File memFile = new java.io.File(ctx.getCacheDir(), "test_persist_mem.bin");
        if (memFile.exists()) memFile.delete();

        int savedId;
        PersonDetectorEngine engine1 = new PersonDetectorEngine(ctx, memFile);
        try (InputStream is = InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("test_person_room.jpg")) {
            Bitmap roomBmp = BitmapFactory.decodeStream(is);
            assertNotNull(roomBmp);

            PersonDetectorEngine.AnalysisOutput out1 = engine1.processFrame(roomBmp);
            assertFalse(out1.persons.isEmpty());
            savedId = out1.persons.get(0).id;
            assertTrue("Isimlendirme basarili olmali", engine1.renamePerson(savedId, "Selim"));
        } finally {
            engine1.close();
        }

        assertTrue("Kalici hafiza dosyasi diske yazilmis olmali", memFile.exists() && memFile.length() > 100);

        // Uygulama yeniden acilmis gibi sifirdan yeni motor baslat
        PersonDetectorEngine engine2 = new PersonDetectorEngine(ctx, memFile);
        try (InputStream is = InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("test_person_room.jpg")) {
            assertEquals("Kalici hafizadan 1 profil yuklenmeli", 1, engine2.getRememberedCount());
            Bitmap roomBmp2 = BitmapFactory.decodeStream(is);
            assertNotNull(roomBmp2);

            PersonDetectorEngine.AnalysisOutput out2 = engine2.processFrame(roomBmp2);
            assertFalse(out2.persons.isEmpty());
            TrackedPerson recognized = out2.persons.get(0);
            assertEquals("Yeniden baslatma sonrasi ayni ID taninmali", savedId, recognized.id);
            assertEquals("Yeniden baslatma sonrasi kayitli isim hatirlanmali", "Selim", recognized.name);
        } finally {
            engine2.close();
            if (memFile.exists()) memFile.delete();
        }
    }

    @Test
    public void testClothingChangeFaceRecognitionRecovery() throws Exception {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        java.io.File memFile = new java.io.File(ctx.getCacheDir(), "test_clothing_change.bin");
        if (memFile.exists()) memFile.delete();
        PersonDetectorEngine engine = new PersonDetectorEngine(ctx, memFile);

        try (InputStream is = InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("test_person_room.jpg")) {
            Bitmap roomBmp = BitmapFactory.decodeStream(is);
            assertNotNull(roomBmp);

            // 1. Kisi tespit edilir, arka planda asenkron yuz ogrenilir, isim verilir ve kilitlenir
            PersonDetectorEngine.AnalysisOutput out1 = engine.processFrame(roomBmp);
            assertFalse(out1.persons.isEmpty());
            engine.waitForPendingFaceWork(2500);
            out1 = engine.processFrame(roomBmp);
            TrackedPerson person = out1.persons.get(0);
            assertTrue("Test gorselinde yuz ogrenilmis olmali", person.hasFaceAi);
            engine.renamePerson(person.id, "Ev Sahibi");

            PersonDetectorEngine.LockedState lockSnap = new PersonDetectorEngine.LockedState();
            assertEquals(PersonDetectorEngine.LockStatus.LOCKED,
                    engine.tryLockAt(person.centerX(), person.centerY(), lockSnap));
            int lockedId = lockSnap.id;
            assertTrue("Kilitli hedefte yuz AI aktif olmali", lockSnap.hasFaceAi);

            // 2. Kisi odadan cikar -> kayip alarmi tetiklenir
            Bitmap blank = Bitmap.createBitmap(480, 360, Bitmap.Config.ARGB_8888);
            blank.eraseColor(Color.BLACK);
            engine.processFrame(blank);
            Thread.sleep(1900);
            PersonDetectorEngine.AnalysisOutput lostOut = engine.processFrame(blank);
            assertTrue(lostOut.locked != null && lostOut.locked.lost);

            // 3. Kisi tamamen farkli renkte kiyafet (parlak kirmizi/sari) giyip geri doner (yuzu ayni!)
            Bitmap newOutfit = roomBmp.copy(Bitmap.Config.ARGB_8888, true);
            android.graphics.Canvas c = new android.graphics.Canvas(newOutfit);
            int rw = newOutfit.getWidth(), rh = newOutfit.getHeight();
            int torsoLeft = Math.round((person.nx + person.nw * 0.14f) * rw);
            int torsoRight = Math.round((person.nx + person.nw * 0.86f) * rw);
            int torsoTop = Math.round((person.ny + person.nh * 0.28f) * rh);
            int torsoMid = Math.round((person.ny + person.nh * 0.60f) * rh);
            int legsBot = Math.round((person.ny + person.nh * 0.92f) * rh);

            android.graphics.Paint redShirt = new android.graphics.Paint();
            redShirt.setColor(Color.argb(215, 225, 35, 35));
            c.drawRect(torsoLeft, torsoTop, torsoRight, torsoMid, redShirt);

            android.graphics.Paint yellowPants = new android.graphics.Paint();
            yellowPants.setColor(Color.argb(215, 235, 200, 30));
            c.drawRect(torsoLeft, torsoMid, torsoRight, legsBot, yellowPants);

            // Asenkron arka plan yuz motorunun kiyafet degisimini yakalamasina izin ver
            Thread.sleep(360);
            PersonDetectorEngine.AnalysisOutput rec = engine.processFrame(newOutfit);
            engine.waitForPendingFaceWork(2500);
            if (rec.locked != null && rec.locked.lost) {
                rec = engine.processFrame(newOutfit);
            }
            assertNotNull(rec.locked);
            assertFalse("Kiyafet degisse bile asenkron biyometrik yuz AI ile geri taninmali!", rec.locked.lost);
            assertEquals("Kiyafet degisimi sonrasi ayni ID korunmali", lockedId, rec.locked.id);
            assertEquals("Kiyafet degisimi sonrasi isim korunmali", "Ev Sahibi", rec.locked.name);

            newOutfit.recycle();
            blank.recycle();
        } finally {
            engine.close();
            if (memFile.exists()) memFile.delete();
        }
    }

    @Test
    public void testZeroPingLongPollWakeupAndSilentPhone() throws Exception {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        // 1. Telefon ses motoru tamamen sessiz (AudioTrack/Vibrator acmadan) calismali
        SirenAudioEngine silent = new SirenAudioEngine(ctx);
        silent.playLockBeep();
        silent.playErrorBeep();
        silent.startAlarm();
        assertTrue(silent.isAlarmActive());
        silent.stopAlarm();
        assertFalse(silent.isAlarmActive());
        silent.release();

        // 2. Uzun beklemeli /state?wait=1&alarm=0 alarm degistigi milisaniyede uyanmali
        CameraStreamServer server = new CameraStreamServer(18082, "54321");
        server.start();
        try {
            server.updateAiState(480, 640, false, false, java.util.Collections.emptyList(), null);
            assertFalse("Sadece state okunurken video akisi acik gorunmemeli", server.hasActiveVideoStream());

            final long[] elapsedMs = new long[1];
            final String[] respHolder = new String[1];
            Thread waiter = new Thread(() -> {
                long t0 = System.currentTimeMillis();
                try {
                    respHolder[0] = httpGet("http://127.0.0.1:18082/state?code=54321&wait=1&alarm=0");
                } catch (Exception ignored) {}
                elapsedMs[0] = System.currentTimeMillis() - t0;
            });
            waiter.start();

            Thread.sleep(220);
            PersonDetectorEngine.LockedState lostLocked = new PersonDetectorEngine.LockedState();
            lostLocked.id = 1;
            lostLocked.lost = true;
            server.updateAiState(480, 640, false, true, java.util.Collections.emptyList(), lostLocked);
            waiter.join(1500);

            assertNotNull("Uzun bekleme yaniti gelmeli", respHolder[0]);
            assertTrue("Alarm degisince aninda uyanmali (2400ms beklenmemeli), sure=" + elapsedMs[0],
                    elapsedMs[0] >= 180 && elapsedMs[0] < 1200);
            assertTrue("Yanitta alarm:true olmali", respHolder[0].contains("\"alarm\":true"));
        } finally {
            server.stop();
        }
    }

    @Test
    public void testCameraMonitorServiceRunsDetachedForScreenOff() throws Exception {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        // Android 14+ camera FGS testi icin CAMERA iznini test surecine ver
        InstrumentationRegistry.getInstrumentation().getUiAutomation()
                .grantRuntimePermission(ctx.getPackageName(), android.Manifest.permission.CAMERA);

        // On plan aktivite baglamindan servisi baslat (Android 14/15/16/17 FGS kurali)
        android.content.Intent actIntent = new android.content.Intent(ctx, MainActivity.class);
        actIntent.addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK);
        android.app.Activity act = InstrumentationRegistry.getInstrumentation().startActivitySync(actIntent);
        try {
            CameraMonitorService.startService(act, "77889");
            long deadline = System.currentTimeMillis() + 5000;
            while (!CameraMonitorService.isRunning() && System.currentTimeMillis() < deadline) {
                Thread.sleep(100);
            }
            CameraMonitorService svc = CameraMonitorService.getInstance();
            assertNotNull("CameraMonitorService baslatilmis olmali", svc);
            assertTrue("Servis aktif olmali", CameraMonitorService.isRunning());
            assertEquals("Servis RESUMED yasam dongusunde olmali",
                    androidx.lifecycle.Lifecycle.State.RESUMED, svc.getLifecycle().getCurrentState());
            assertTrue("Ekran kapaliyken CPU WakeLock aktif olmali", svc.isWakeLockHeld());
            assertTrue("Ekran kapaliyken Wi-Fi Lock aktif olmali", svc.isWifiLockHeld());

            // Ekran kapatma / telefon kilitleme simulasyonu: UI tamamen ayrilir (detachUi)
            svc.detachUi();
            Thread.sleep(600);

            // UI ayrildiktan sonra da servis RESUMED kalmali ve HTTP sunucusu canli yanit vermeli!
            assertTrue("Ekran kapaliyken servis hala calisiyor olmali", CameraMonitorService.isRunning());
            assertEquals("UI ayrilinca servis yasam dongusu dusmemeli",
                    androidx.lifecycle.Lifecycle.State.RESUMED, svc.getLifecycle().getCurrentState());
            String pingResp = httpGet("http://127.0.0.1:8080/ping?code=77889");
            assertTrue("Ekran kapaliyken /ping yanit vermeli", pingResp.contains("\"ok\":true"));
            String stateResp = httpGet("http://127.0.0.1:8080/state?code=77889");
            assertTrue("Ekran kapaliyken /state AI durumu donmeli", stateResp.contains("\"persons\""));
        } finally {
            CameraMonitorService.stopService(ctx);
            long stopDeadline = System.currentTimeMillis() + 4000;
            while (CameraMonitorService.isRunning() && System.currentTimeMillis() < stopDeadline) {
                Thread.sleep(100);
            }
            if (act != null) act.finish();
        }
        assertFalse("Durdurulunca servis temiz kapanmali", CameraMonitorService.isRunning());
    }

    @Test
    public void testSmartOcclusionBufferPreventsFalseAlarm() throws Exception {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        java.io.File memFile = new java.io.File(ctx.getCacheDir(), "test_occlusion.bin");
        if (memFile.exists()) memFile.delete();
        PersonDetectorEngine engine = new PersonDetectorEngine(ctx, memFile);

        try (InputStream is = InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("test_person_room.jpg")) {
            Bitmap roomBmp = BitmapFactory.decodeStream(is);
            PersonDetectorEngine.AnalysisOutput out1 = engine.processFrame(roomBmp);
            assertFalse(out1.persons.isEmpty());
            TrackedPerson p = out1.persons.get(0);

            PersonDetectorEngine.LockedState lockSnap = new PersonDetectorEngine.LockedState();
            assertEquals(PersonDetectorEngine.LockStatus.LOCKED, engine.tryLockAt(p.centerX(), p.centerY(), lockSnap));

            // 1. Kisi kisa sureligine (850 ms) gizlensin -> ~1.8 sn tamponu SAHTE ALARM CALMAMALI!
            Bitmap blank = Bitmap.createBitmap(480, 360, Bitmap.Config.ARGB_8888);
            blank.eraseColor(Color.BLACK);
            engine.processFrame(blank);
            Thread.sleep(850);
            PersonDetectorEngine.AnalysisOutput midOut = engine.processFrame(blank);
            assertNotNull(midOut.locked);
            assertFalse("850ms kisa gizlenmede sahte kayip alarmi tetiklenmemeli!", midOut.targetJustLost);
            assertFalse("850ms kisa gizlenmede lost false kalmali!", midOut.locked.lost);
            assertTrue("Kisa gizlenmede isOccluded true olmali!", midOut.locked.isOccluded);

            // 2. Kisi 1.8 sn'den fazla gorunmeyince gercek alarm tetiklenmeli!
            Thread.sleep(1050);
            PersonDetectorEngine.AnalysisOutput lostOut = engine.processFrame(blank);
            assertTrue("1.9s sonra gercek kayip alarmi tetiklenmeli!", lostOut.targetJustLost);
            assertTrue("lost true olmali!", lostOut.locked.lost);
            blank.recycle();
        } finally {
            engine.close();
            if (memFile.exists()) memFile.delete();
        }
    }

    @Test
    public void testOsnetReidBackgroundAndNoFaceRecovery() throws Exception {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        NeuralIdentityEngine neural = new NeuralIdentityEngine(ctx);
        try (InputStream is = InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("test_person_room.jpg")) {
            assertTrue("OSNet MSMT17 modeli hazir olmali", neural.isReidReady());
            assertEquals("OSNet cikti boyutu 512 olmali", 512, neural.getReidEmbedDim());

            Bitmap roomBmp = BitmapFactory.decodeStream(is);
            // Ayni kisinin orijinal govdesi vs aynalanmis + %40 karartilmis govdesi vs tamamen farkli renkli kisi
            NeuralIdentityEngine.BodyReidResult rOrig = neural.extractBodyReid(roomBmp, 0.24f, 0.12f, 0.48f, 0.84f);
            assertNotNull(rOrig.embedding);
            assertTrue("Net goruntu Laplacian keskinlik kapisini gecmeli, sharpness=" + rOrig.sharpness, rOrig.isSharp);

            // Aynalanmis ve %40 karartilmis kopya (arka plan + isik degisimi + aci tersi)
            Bitmap modBmp = Bitmap.createBitmap(roomBmp.getWidth(), roomBmp.getHeight(), Bitmap.Config.ARGB_8888);
            android.graphics.Canvas c = new android.graphics.Canvas(modBmp);
            android.graphics.Matrix m = new android.graphics.Matrix();
            m.setScale(-1f, 1f, roomBmp.getWidth() * 0.5f, roomBmp.getHeight() * 0.5f);
            android.graphics.Paint p = new android.graphics.Paint();
            android.graphics.ColorMatrix cm = new android.graphics.ColorMatrix();
            cm.setScale(0.60f, 0.58f, 0.54f, 1.0f);
            p.setColorFilter(new android.graphics.ColorMatrixColorFilter(cm));
            c.drawBitmap(roomBmp, m, p);

            NeuralIdentityEngine.BodyReidResult rMirrorDark = neural.extractBodyReid(modBmp, 0.28f, 0.12f, 0.48f, 0.84f);
            assertNotNull(rMirrorDark.embedding);

            // Tamamen farkli sentetik kisi (parlak kirmizi ust + mavi alt)
            Bitmap diffBmp = Bitmap.createBitmap(240, 400, Bitmap.Config.ARGB_8888);
            android.graphics.Canvas dc = new android.graphics.Canvas(diffBmp);
            dc.drawColor(Color.rgb(35, 38, 44));
            android.graphics.Paint dp = new android.graphics.Paint();
            dp.setColor(Color.rgb(215, 175, 145));
            dc.drawRect(80, 25, 160, 100, dp);
            dp.setColor(Color.rgb(220, 35, 30));
            dc.drawRect(50, 100, 190, 240, dp);
            dp.setColor(Color.rgb(30, 130, 225));
            dc.drawRect(60, 240, 180, 380, dp);
            NeuralIdentityEngine.BodyReidResult rDiff = neural.extractBodyReid(diffBmp, 0.10f, 0.04f, 0.80f, 0.92f);
            assertNotNull(rDiff.embedding);

            float sameSim = NeuralIdentityEngine.cosineSim(rOrig.embedding, rMirrorDark.embedding);
            float diffSim = NeuralIdentityEngine.cosineSim(rOrig.embedding, rDiff.embedding);
            assertTrue("OSNet ayni kisiyi karanlik/ayna degisiminde yuksek tanimali (sim=" + sameSim + ")", sameSim >= 0.72f);
            assertTrue("OSNet farkli kisiyi belirgin sekilde ayirmali (same=" + sameSim + ", diff=" + diffSim + ")",
                    sameSim - diffSim >= 0.20f);

            modBmp.recycle();
            diffBmp.recycle();
        } finally {
            neural.close();
        }
    }

    @Test
    public void testRoomBackgroundMaskSuppressesWallPixels() {
        int w = 80, h = 60;
        RoomBackgroundMask mask = new RoomBackgroundMask(w, h);
        int[] emptyRoom = new int[w * h];
        java.util.Arrays.fill(emptyRoom, Color.rgb(210, 205, 195)); // Acik bej duvar

        // 4 bos oda karesiyle duvari ogren
        for (int i = 0; i < 4; i++) {
            mask.updateWithFrame(emptyRoom, java.util.Collections.emptyList());
        }
        assertTrue("Bos oda duvarinin tamami ogrenilmis olmali", mask.getLearnedCoverageRatio() > 0.99f);

        // Koyu lacivert tisortlu kisi sahneye girer: duvar pikseli ~0.06, kisi pikseli 1.0 agirlik almali!
        int wallIdx = 10 * w + 10;
        float wallWeight = mask.getForegroundWeight(wallIdx, 210, 205, 195);
        float personWeight = mask.getForegroundWeight(wallIdx, 25, 40, 110);
        assertTrue("Statik duvar pikseli bastirilmali (w=" + wallWeight + ")", wallWeight < 0.15f);
        assertTrue("On plandaki kisi pikseli tam agirlik almali (w=" + personWeight + ")", personWeight > 0.95f);

        // Kisi kutusu icindeki pikseller 10 kare boyunca arka plana ASLA karismamali!
        int[] withPerson = emptyRoom.clone();
        for (int y = 15; y < 45; y++) {
            for (int x = 25; x < 55; x++) {
                withPerson[y * w + x] = Color.rgb(25, 40, 110);
            }
        }
        TrackedPerson box = new TrackedPerson(1, 25f / w, 15f / h, 30f / w, 30f / h, 0.9f);
        for (int i = 0; i < 10; i++) {
            mask.updateWithFrame(withPerson, java.util.Collections.singletonList(box));
        }
        int personCenterIdx = 30 * w + 40;
        assertTrue("Oturan kisi 10 kare sonra bile hala on plan kalmali!",
                mask.getForegroundWeight(personCenterIdx, 25, 40, 110) > 0.90f);
    }

    @Test
    public void testHungarianAssignmentAndLaplacianSharpnessGate() {
        // 1. Klasik 2-kisi kesisme matrisi:
        //    Satir 0 (Kisi A) -> Sutun 0 (0.85), Sutun 1 (0.90)
        //    Satir 1 (Kisi B) -> Sutun 0 (0.10), Sutun 1 (0.88)
        //    Acgozlu algoritma Satir 0'i Sutun 1'e verip Satir 1'i bozardi; Hungarian (0->0, 1->1 = 1.73) secer!
        float[][] scores = {
                {0.85f, 0.90f},
                {0.10f, 0.88f}
        };
        int[] assign = SmartTrackerMath.solveMaxAssignment(scores, 0.40f);
        assertEquals("Global optimal eslestirme Kisi A'yi 0'a atamali", 0, assign[0]);
        assertEquals("Global optimal eslestirme Kisi B'yi 1'e atamali", 1, assign[1]);

        // 2. Laplacian keskinlik kapisi: net damali desen vs bulanik tek renk/duz gecis
        int w = 64, h = 64;
        int[] sharpPx = new int[w * h];
        int[] blurPx = new int[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int vSharp = (((x / 3) + (y / 3)) % 2 == 0) ? 230 : 25;
                sharpPx[y * w + x] = Color.rgb(vSharp, vSharp, vSharp);
                int vBlur = 120 + (x / 16);
                blurPx[y * w + x] = Color.rgb(vBlur, vBlur, vBlur);
            }
        }
        float sharpScore = SmartTrackerMath.computeLaplacianSharpness(sharpPx, w, h, 4, 4, w - 4, h - 4, 1);
        float blurScore = SmartTrackerMath.computeLaplacianSharpness(blurPx, w, h, 4, 4, w - 4, h - 4, 1);
        assertTrue("Net goruntu yuksek keskinlik vermeli (sharp=" + sharpScore + ")", sharpScore > 500f);
        assertTrue("Bulanik goruntu dusuk keskinlik vermeli (blur=" + blurScore + ")", blurScore < 10f);
    }

    @Test
    public void testGuided360EnrollmentPersistsAllThreeLayers() throws Exception {
        Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
        java.io.File memFile = new java.io.File(ctx.getCacheDir(), "test_enroll_v3.bin");
        if (memFile.exists()) memFile.delete();

        int enrolledId;
        PersonDetectorEngine engine1 = new PersonDetectorEngine(ctx, memFile);
        try (InputStream is = InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("test_person_room.jpg")) {
            Bitmap roomBmp = BitmapFactory.decodeStream(is);
            PersonDetectorEngine.AnalysisOutput out1 = engine1.processFrame(roomBmp);
            assertFalse(out1.persons.isEmpty());
            enrolledId = out1.persons.get(0).id;

            assertTrue("360 derece rehberli tanitim baslamali", engine1.startGuidedEnrollment(enrolledId, "Selim360"));
            Thread.sleep(180);
            PersonDetectorEngine.AnalysisOutput out2 = engine1.processFrame(roomBmp);
            assertNotNull("Tanitim durumu donmeli", out2.enrollment);
            assertTrue("Tanitim aktif olmali", out2.enrollment.active);

            engine1.waitForPendingIdentityWork(3000);
            assertTrue("Tanitim tamamlanmali", engine1.finishGuidedEnrollmentNow());
        } finally {
            engine1.close();
        }

        // Uygulama yeniden acilmis gibi sifirdan baslat -> 3 katmanin tamami (OSNet + Yuz + 129-D) diskten yuklenmeli!
        PersonDetectorEngine engine2 = new PersonDetectorEngine(ctx, memFile);
        try {
            java.util.List<PersonMemoryStore.ProfileSummary> summaries = engine2.getMemorySummaries();
            assertEquals("1 kalici 360 profili yuklenmeli", 1, summaries.size());
            PersonMemoryStore.ProfileSummary prof = summaries.get(0);
            assertEquals(enrolledId, prof.id);
            assertEquals("Selim360", prof.name);
            assertTrue("Enrolled bayragi korunmali", prof.enrolled);
            assertTrue("129-D aci hafizasi diskten yuklenmeli", prof.sigCount >= 1);
            assertTrue("OSNet 512-D derin hafizasi diskten yuklenmeli", prof.deepCount >= 1);
            assertTrue("MobileFaceNet 192-D yuz hafizasi diskten yuklenmeli", prof.faceCount >= 1);
        } finally {
            engine2.close();
            if (memFile.exists()) memFile.delete();
        }
    }
}
