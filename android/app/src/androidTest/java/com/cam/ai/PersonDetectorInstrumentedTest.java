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

        // 5. Kisi kadrajdan cikinca (bos siyah kareler + 1.2s bekleme) hedef kayip alarmi tetiklenmeli
        Bitmap emptyBmp = Bitmap.createBitmap(480, 640, Bitmap.Config.ARGB_8888);
        emptyBmp.eraseColor(Color.BLACK);

        Thread.sleep(1250);
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
            Thread.sleep(1150);
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
}
