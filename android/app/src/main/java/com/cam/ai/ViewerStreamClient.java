package com.cam.ai;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.DhcpInfo;
import android.net.wifi.WifiManager;
import android.util.Log;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.InputStream;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * İzleyici Modu: Yerel Wi-Fi / Mobil Erişim Noktası (Hotspot) üzerindeki kamerayı otomatik keşfeder
 * (Wifi MulticastLock + UDP 8888 + DHCP Ağ Geçidi + Tüm Yerel Alt Ağlar Paralel HTTP Tarama + Otomatik Yeniden Deneme),
 * MJPEG/JPG canlı akışını ve Yapay Zeka durum verilerini senkronize okur, uzaktan kilit komutları yollar.
 */
public class ViewerStreamClient {

    private static final String TAG = "ViewerStreamClient";
    private static final int DEFAULT_CAMERA_PORT = 8080;

    public interface Listener {
        void onSearching(String message);
        void onConnected(String ip, String deviceModel);
        void onWrongPassword();
        void onFrame(Bitmap bitmap);
        void onAiState(int w, int h, boolean isFront, boolean alarm,
                       List<TrackedPerson> persons, PersonDetectorEngine.LockedState locked);
        void onDisconnected(String reason);
        void onRemoteLockResult(boolean locked);
    }

    private final Context appContext;
    private final Listener listener;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private volatile String activeIp = null;
    private volatile int activePort = DEFAULT_CAMERA_PORT;
    private volatile String roomCode = "";

    private Thread discoveryThread;
    private Thread videoThread;
    private Thread stateThread;
    private volatile DatagramSocket activeUdpSocket;
    private final ExecutorService cmdExecutor = Executors.newSingleThreadExecutor();
    private final ByteArrayOutputStream headerLineBuf = new ByteArrayOutputStream(128);
    private byte[] frameBuf = new byte[65536];

    public ViewerStreamClient(Listener listener) {
        this(null, listener);
    }

    public ViewerStreamClient(Context context, Listener listener) {
        this.appContext = (context != null) ? context.getApplicationContext() : null;
        this.listener = listener;
    }

    public void startAutoConnect(String roomCode) {
        stop();
        this.roomCode = roomCode;
        this.running.set(true);

        listener.onSearching("Aynı Wi-Fi / Hotspot ağındaki kamera aranıyor...");

        discoveryThread = new Thread(this::discoverAndConnectLoop, "ViewerDiscoveryThread");
        discoveryThread.start();
    }

    public void connectDirect(String ip, String roomCode) {
        stop();
        this.roomCode = roomCode;
        this.running.set(true);
        String cleanIp = ip.trim();
        int port = DEFAULT_CAMERA_PORT;
        if (cleanIp.contains(":")) {
            String[] sp = cleanIp.split(":");
            cleanIp = sp[0];
            try { port = Integer.parseInt(sp[1]); } catch (Exception ignored) {}
        }
        final String targetIp = cleanIp;
        final int targetPort = port;

        listener.onSearching(targetIp + " adresine bağlanılıyor...");
        new Thread(() -> {
            PingResult ping = pingCamera(targetIp, targetPort, roomCode, 1800);
            if (!running.get()) return;
            if (ping.status == 200) {
                startStreamingThreads(targetIp, targetPort, ping.model != null ? ping.model : "Kamera");
            } else if (ping.status == 403 && ping.isCamServer) {
                listener.onWrongPassword();
            } else {
                listener.onDisconnected("Kameraya ulaşılamadı (" + targetIp + ":" + targetPort + ")");
            }
        }, "DirectConnectThread").start();
    }

    public void stop() {
        running.set(false);
        activeIp = null;
        DatagramSocket sock = activeUdpSocket;
        activeUdpSocket = null;
        if (sock != null) {
            try { sock.close(); } catch (Exception ignored) {}
        }
        if (discoveryThread != null) {
            discoveryThread.interrupt();
            discoveryThread = null;
        }
        if (videoThread != null) {
            videoThread.interrupt();
            videoThread = null;
        }
        if (stateThread != null) {
            stateThread.interrupt();
            stateThread = null;
        }
    }

    public void sendLock(float nx, float ny) {
        final String ip = activeIp;
        if (ip == null) return;
        cmdExecutor.execute(() -> {
            try {
                String res = httpGet("http://" + ip + ":" + activePort + "/cmd?code=" + roomCode
                        + "&action=lock&nx=" + nx + "&ny=" + ny, 2500);
                if (res != null) {
                    JSONObject obj = new JSONObject(res);
                    String status = obj.optString("status", "EMPTY");
                    listener.onRemoteLockResult("LOCKED".equals(status));
                }
            } catch (Exception ignored) {}
        });
    }

    public void sendAction(String action) {
        final String ip = activeIp;
        if (ip == null) return;
        cmdExecutor.execute(() -> httpGet("http://" + ip + ":" + activePort + "/cmd?code=" + roomCode + "&action=" + action, 2000));
    }

    private static class PingResult {
        int status = -1;
        boolean isCamServer = false;
        String model = null;
    }

    private void discoverAndConnectLoop() {
        WifiManager.MulticastLock multicastLock = null;
        if (appContext != null) {
            try {
                WifiManager wm = (WifiManager) appContext.getSystemService(Context.WIFI_SERVICE);
                if (wm != null) {
                    multicastLock = wm.createMulticastLock("CamAiDiscovery");
                    multicastLock.setReferenceCounted(false);
                    multicastLock.acquire();
                }
            } catch (Exception ignored) {}
        }

        final AtomicBoolean found = new AtomicBoolean(false);
        final AtomicBoolean wrongCodeDetected = new AtomicBoolean(false);
        final String[] wrongCodeIp = new String[1];

        // 1. UDP Dinleyici (MulticastLock açıkken router ve hotspot yayınlarını anında alır)
        Thread udpListener = new Thread(() -> {
            DatagramSocket socket = null;
            try {
                socket = new DatagramSocket(null);
                socket.setReuseAddress(true);
                socket.bind(new InetSocketAddress(CameraStreamServer.DISCOVERY_UDP_PORT));
                socket.setSoTimeout(700);
                activeUdpSocket = socket;

                byte[] buf = new byte[512];
                while (running.get() && !found.get()) {
                    try {
                        DatagramPacket packet = new DatagramPacket(buf, buf.length);
                        socket.receive(packet);
                        String msg = new String(packet.getData(), 0, packet.getLength(), StandardCharsets.UTF_8);
                        if (msg.startsWith("CAMPRO|")) {
                            String[] parts = msg.split("\\|");
                            if (parts.length >= 5) {
                                String pktCode = parts[1];
                                String payloadIp = parts[2];
                                int port = Integer.parseInt(parts[3]);
                                String model = parts[4];
                                String senderIp = (packet.getAddress() != null) ? packet.getAddress().getHostAddress() : null;

                                if (roomCode.equals(pktCode)) {
                                    // Önce bildirilen IP veya gönderici IP'nin erişilebilirliğini doğrula
                                    String chosenIp = null;
                                    PingResult p1 = pingCamera(payloadIp, port, roomCode, 600);
                                    if (p1.status == 200) {
                                        chosenIp = payloadIp;
                                    } else if (senderIp != null && !senderIp.equals(payloadIp)) {
                                        PingResult p2 = pingCamera(senderIp, port, roomCode, 600);
                                        if (p2.status == 200) chosenIp = senderIp;
                                    }
                                    if (chosenIp != null && found.compareAndSet(false, true)) {
                                        startStreamingThreads(chosenIp, port, model);
                                        return;
                                    }
                                }
                            }
                        }
                    } catch (Exception ignored) {}
                }
            } catch (Exception e) {
                Log.w(TAG, "UDP dinleme hatası: " + e.getMessage());
            } finally {
                if (activeUdpSocket == socket) activeUdpSocket = null;
                if (socket != null) {
                    try { socket.close(); } catch (Exception ignored) {}
                }
            }
        }, "UdpListenThread");
        udpListener.start();

        // 2. Sürekli Alt Ağ + Ağ Geçidi Tarama Döngüsü (Kamera geç açılsa bile otomatik bağlanır)
        int attempt = 1;
        try {
            while (running.get() && !found.get()) {
                wrongCodeDetected.set(false);
                List<String> candidateIps = buildCandidateIpList();
                List<String> localIps = CameraStreamServer.getAllLocalIpv4Addresses();
                String netHint = localIps.isEmpty() ? "Yerel Ağ yok" : localIps.get(0);

                if (attempt == 1) {
                    listener.onSearching("Kamera aranıyor... (Ağ: " + netHint + ")");
                } else {
                    listener.onSearching("Kamera aranıyor • Deneme #" + attempt + " (" + netHint + ")");
                }

                ExecutorService scanPool = Executors.newFixedThreadPool(48);
                for (String ip : candidateIps) {
                    if (!running.get() || found.get()) break;
                    scanPool.execute(() -> {
                        if (!running.get() || found.get()) return;
                        PingResult ping = pingCamera(ip, DEFAULT_CAMERA_PORT, roomCode, 950);
                        if (ping.status == 200 && found.compareAndSet(false, true)) {
                            startStreamingThreads(ip, DEFAULT_CAMERA_PORT, ping.model != null ? ping.model : "Yerel Kamera");
                        } else if (ping.status == 403 && ping.isCamServer) {
                            wrongCodeIp[0] = ip;
                            wrongCodeDetected.set(true);
                        }
                    });
                }

                long passStart = System.currentTimeMillis();
                while (running.get() && !found.get() && (System.currentTimeMillis() - passStart < 6500)) {
                    try { Thread.sleep(200); } catch (Exception e) { break; }
                }
                scanPool.shutdownNow();

                if (!running.get() || found.get()) break;

                if (wrongCodeDetected.get()) {
                    listener.onSearching("Kamera bulundu (" + wrongCodeIp[0] + ") ama 5 haneli şifre farklı!");
                }
                attempt++;
                try { Thread.sleep(800); } catch (Exception e) { break; }
            }
        } finally {
            DatagramSocket sock = activeUdpSocket;
            activeUdpSocket = null;
            if (sock != null) {
                try { sock.close(); } catch (Exception ignored) {}
            }
            if (multicastLock != null && multicastLock.isHeld()) {
                try { multicastLock.release(); } catch (Exception ignored) {}
            }
        }
    }

    private List<String> buildCandidateIpList() {
        LinkedHashSet<String> ips = new LinkedHashSet<>();

        // 1. Android WifiManager DHCP Ağ Geçidi (Hotspot'a bağlanan telefon doğrudan kamera IP'sini görür!)
        if (appContext != null) {
            try {
                WifiManager wm = (WifiManager) appContext.getSystemService(Context.WIFI_SERVICE);
                if (wm != null) {
                    DhcpInfo dhcp = wm.getDhcpInfo();
                    if (dhcp != null) {
                        addIntIpAndSubnetGateways(dhcp.gateway, ips);
                        addIntIpAndSubnetGateways(dhcp.serverAddress, ips);
                    }
                }
            } catch (Exception ignored) {}
        }

        // 2. Yaygın Hotspot ağ geçitleri & Android Emülatör
        ips.add("192.168.43.1");
        ips.add("192.168.137.1");
        ips.add("172.20.10.1");
        ips.add("192.168.49.1");
        ips.add("10.0.2.2");
        ips.add("127.0.0.1");

        // 3. Cihazın tüm aktif yerel IPv4 arayüzlerinin (Wi-Fi + Hotspot) .1 ve 1..254 aralıkları
        List<String> myIps = CameraStreamServer.getAllLocalIpv4Addresses();
        for (String myIp : myIps) {
            int lastDot = myIp.lastIndexOf('.');
            if (lastDot > 0) {
                String prefix = myIp.substring(0, lastDot + 1);
                ips.add(prefix + "1"); // Önce ağ geçidi (.1)
            }
        }
        for (String myIp : myIps) {
            int lastDot = myIp.lastIndexOf('.');
            if (lastDot > 0) {
                String prefix = myIp.substring(0, lastDot + 1);
                for (int i = 2; i <= 254; i++) {
                    String cand = prefix + i;
                    if (!cand.equals(myIp)) {
                        ips.add(cand);
                    }
                }
            }
        }
        return new ArrayList<>(ips);
    }

    private void addIntIpAndSubnetGateways(int ipInt, LinkedHashSet<String> ips) {
        if (ipInt == 0) return;
        String ip = (ipInt & 0xff) + "." + ((ipInt >> 8) & 0xff) + "." + ((ipInt >> 16) & 0xff) + "." + ((ipInt >> 24) & 0xff);
        if (!"0.0.0.0".equals(ip) && !"127.0.0.1".equals(ip)) {
            ips.add(ip);
            int dot = ip.lastIndexOf('.');
            if (dot > 0) {
                ips.add(ip.substring(0, dot + 1) + "1");
            }
        }
    }

    private PingResult pingCamera(String ip, int port, String code, int timeoutMs) {
        PingResult result = new PingResult();
        HttpURLConnection conn = null;
        try {
            URL url = new URL("http://" + ip + ":" + port + "/ping?code=" + code);
            conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(timeoutMs);
            conn.setReadTimeout(timeoutMs);
            result.status = conn.getResponseCode();
            InputStream stream = (result.status >= 400) ? conn.getErrorStream() : conn.getInputStream();
            if (stream != null) {
                byte[] buf = new byte[512];
                int n = stream.read(buf);
                if (n > 0) {
                    String body = new String(buf, 0, n, StandardCharsets.UTF_8);
                    if (body.contains("WRONG_CODE")) {
                        result.isCamServer = true;
                    } else if (body.contains("\"ok\"")) {
                        result.isCamServer = true;
                        try {
                            JSONObject obj = new JSONObject(body);
                            result.model = obj.optString("model", "Kamera");
                        } catch (Exception ignored) {}
                    }
                }
            }
        } catch (Exception ignored) {
        } finally {
            if (conn != null) conn.disconnect();
        }
        return result;
    }

    private void startStreamingThreads(String ip, int port, String model) {
        this.activeIp = ip;
        this.activePort = port;
        listener.onConnected(ip, model);

        videoThread = new Thread(this::runMjpegVideoLoop, "ViewerVideoLoop");
        videoThread.start();

        stateThread = new Thread(this::runAiStateLoop, "ViewerStateLoop");
        stateThread.start();
    }

    private void runMjpegVideoLoop() {
        int streamFailStreak = 0;
        while (running.get() && activeIp != null) {
            if (streamFailStreak >= 2) {
                // Yedek Tek-Kare Modu (/frame.jpg): Bazı Hotspot/Router ağlarında chunked akış kesilirse akıcı kare çeker
                if (fetchSingleJpegFrame()) {
                    streamFailStreak = 0;
                    try { Thread.sleep(55); } catch (Exception e) { break; }
                    continue;
                }
            }

            HttpURLConnection conn = null;
            try {
                URL url = new URL("http://" + activeIp + ":" + activePort + "/stream?code=" + roomCode);
                conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(3000);
                conn.setReadTimeout(4500);

                int code = conn.getResponseCode();
                if (code == 403) {
                    listener.onWrongPassword();
                    return;
                }

                BufferedInputStream bis = new BufferedInputStream(conn.getInputStream(), 65536);
                DataInputStream dis = new DataInputStream(bis);

                while (running.get()) {
                    int contentLength = -1;
                    String line;
                    while ((line = readAsciiLine(bis)) != null) {
                        if (line.isEmpty()) {
                            if (contentLength > 0) break;
                            else continue;
                        }
                        String lower = line.toLowerCase();
                        if (lower.startsWith("content-length:")) {
                            try {
                                contentLength = Integer.parseInt(line.substring(15).trim());
                            } catch (Exception ignored) {}
                        }
                    }

                    if (contentLength <= 0 || contentLength > 5_000_000) {
                        streamFailStreak++;
                        break;
                    }

                    if (frameBuf.length < contentLength) {
                        frameBuf = new byte[contentLength];
                    }
                    dis.readFully(frameBuf, 0, contentLength);

                    Bitmap bmp = BitmapFactory.decodeByteArray(frameBuf, 0, contentLength);
                    if (bmp != null) {
                        streamFailStreak = 0;
                        listener.onFrame(bmp);
                    }
                }
            } catch (Exception e) {
                streamFailStreak++;
                Log.w(TAG, "MJPEG akış yenileniyor: " + e.getMessage());
            } finally {
                if (conn != null) conn.disconnect();
            }

            if (running.get()) {
                try { Thread.sleep(250); } catch (Exception ignored) { break; }
            }
        }
    }

    private boolean fetchSingleJpegFrame() {
        HttpURLConnection conn = null;
        try {
            URL url = new URL("http://" + activeIp + ":" + activePort + "/frame.jpg?code=" + roomCode);
            conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(1500);
            conn.setReadTimeout(1500);
            int code = conn.getResponseCode();
            if (code == 200) {
                Bitmap bmp = BitmapFactory.decodeStream(conn.getInputStream());
                if (bmp != null) {
                    listener.onFrame(bmp);
                    return true;
                }
            }
        } catch (Exception ignored) {
        } finally {
            if (conn != null) conn.disconnect();
        }
        return false;
    }

    private String readAsciiLine(InputStream in) throws Exception {
        headerLineBuf.reset();
        int b;
        while ((b = in.read()) != -1) {
            if (b == '\n') {
                byte[] arr = headerLineBuf.toByteArray();
                int len = arr.length;
                if (len > 0 && arr[len - 1] == '\r') len--;
                return new String(arr, 0, len, StandardCharsets.US_ASCII);
            }
            headerLineBuf.write(b);
            if (headerLineBuf.size() > 4096) return null;
        }
        return null;
    }

    private void runAiStateLoop() {
        while (running.get() && activeIp != null) {
            try {
                String json = httpGet("http://" + activeIp + ":" + activePort + "/state?code=" + roomCode, 1200);
                if (json != null) {
                    JSONObject root = new JSONObject(json);
                    int w = root.optInt("w", 480);
                    int h = root.optInt("h", 640);
                    boolean isFront = root.optBoolean("front", false);
                    boolean alarm = root.optBoolean("alarm", false);

                    List<TrackedPerson> persons = new ArrayList<>();
                    JSONArray arr = root.optJSONArray("persons");
                    if (arr != null) {
                        for (int i = 0; i < arr.length(); i++) {
                            JSONObject o = arr.getJSONObject(i);
                            persons.add(new TrackedPerson(
                                    o.optInt("id", i + 1),
                                    (float) o.optDouble("x", 0),
                                    (float) o.optDouble("y", 0),
                                    (float) o.optDouble("w", 0),
                                    (float) o.optDouble("h", 0),
                                    (float) o.optDouble("s", 0.5)
                            ));
                        }
                    }

                    PersonDetectorEngine.LockedState locked = null;
                    if (!root.isNull("locked")) {
                        JSONObject lo = root.getJSONObject("locked");
                        locked = new PersonDetectorEngine.LockedState();
                        locked.id = lo.optInt("id", 1);
                        locked.nx = (float) lo.optDouble("x", 0);
                        locked.ny = (float) lo.optDouble("y", 0);
                        locked.nw = (float) lo.optDouble("w", 0);
                        locked.nh = (float) lo.optDouble("h", 0);
                        locked.score = (float) lo.optDouble("s", 0.9);
                        locked.lost = lo.optBoolean("lost", false);
                    }

                    listener.onAiState(w, h, isFront, alarm, persons, locked);
                }
            } catch (Exception ignored) {}

            try { Thread.sleep(75); } catch (Exception e) { break; }
        }
    }

    private String httpGet(String urlStr, int timeoutMs) {
        HttpURLConnection conn = null;
        try {
            URL url = new URL(urlStr);
            conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(timeoutMs);
            conn.setReadTimeout(timeoutMs);
            if (conn.getResponseCode() != 200) return null;
            InputStream in = conn.getInputStream();
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[4096];
            int n;
            while ((n = in.read(buf)) != -1) {
                out.write(buf, 0, n);
            }
            return new String(out.toByteArray(), StandardCharsets.UTF_8);
        } catch (Exception e) {
            return null;
        } finally {
            if (conn != null) conn.disconnect();
        }
    }

    public void release() {
        stop();
        cmdExecutor.shutdownNow();
    }
}
