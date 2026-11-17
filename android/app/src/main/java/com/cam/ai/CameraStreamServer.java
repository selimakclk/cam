package com.cam.ai;

import android.os.Build;
import android.util.Log;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.IOException;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.InterfaceAddress;
import java.net.NetworkInterface;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import fi.iki.elonen.NanoHTTPD;

/**
 * Yerel Wi-Fi / Hotspot üzerinde şifre korumalı MJPEG canlı yayın,
 * JSON yapay zeka durum senkronizasyonu ve UDP otomatik keşif sunucusu.
 */
public class CameraStreamServer extends NanoHTTPD {

    private static final String TAG = "CameraStreamServer";
    public static final int DISCOVERY_UDP_PORT = 8888;

    public interface OnRemoteCommandListener {
        String onRemoteLock(float nx, float ny); // "LOCKED" veya "EMPTY" döner
        void onRemoteUnlock();
        void onRemoteDismissAlarm();
        void onRemoteFlipCamera();
        void onRemoteToggleTorch();
    }

    private final String roomCode;
    private final int port;
    private volatile byte[] latestJpeg = null;
    private final Object frameLock = new Object();

    private volatile String latestStateJson = "{\"w\":480,\"h\":640,\"front\":false,\"alarm\":false,\"persons\":[],\"locked\":null}";
    private OnRemoteCommandListener commandListener;

    private volatile boolean running = false;
    private Thread udpBroadcastThread;
    private final AtomicInteger activeStreamClients = new AtomicInteger(0);
    private volatile long lastViewerActivityTs = 0L;

    public boolean hasActiveViewer() {
        return activeStreamClients.get() > 0 || (System.currentTimeMillis() - lastViewerActivityTs < 4000L);
    }

    public CameraStreamServer(int port, String roomCode) {
        super(port);
        this.port = port;
        this.roomCode = roomCode;
    }

    public void setOnRemoteCommandListener(OnRemoteCommandListener listener) {
        this.commandListener = listener;
    }

    @Override
    public void start() throws IOException {
        try {
            super.start();
        } catch (IOException firstErr) {
            try { Thread.sleep(180); } catch (InterruptedException ignored) {}
            super.start();
        }
        running = true;
        startUdpDiscoveryBroadcaster();
    }

    @Override
    public void stop() {
        running = false;
        if (udpBroadcastThread != null) {
            udpBroadcastThread.interrupt();
            udpBroadcastThread = null;
        }
        synchronized (frameLock) {
            frameLock.notifyAll();
        }
        super.stop();
    }

    public void updateFrame(byte[] jpegBytes) {
        if (jpegBytes == null || jpegBytes.length == 0) return;
        synchronized (frameLock) {
            this.latestJpeg = jpegBytes;
            frameLock.notifyAll();
        }
    }

    public void updateAiState(
            int width, int height, boolean isFront, boolean alarmActive,
            List<TrackedPerson> persons, PersonDetectorEngine.LockedState locked) {
        try {
            JSONObject root = new JSONObject();
            root.put("w", width);
            root.put("h", height);
            root.put("front", isFront);
            root.put("alarm", alarmActive);

            JSONArray arr = new JSONArray();
            if (persons != null) {
                for (TrackedPerson p : persons) {
                    JSONObject o = new JSONObject();
                    o.put("id", p.id);
                    o.put("x", p.nx);
                    o.put("y", p.ny);
                    o.put("w", p.nw);
                    o.put("h", p.nh);
                    o.put("s", p.score);
                    arr.put(o);
                }
            }
            root.put("persons", arr);

            if (locked != null) {
                JSONObject lo = new JSONObject();
                lo.put("id", locked.id);
                lo.put("x", locked.nx);
                lo.put("y", locked.ny);
                lo.put("w", locked.nw);
                lo.put("h", locked.nh);
                lo.put("s", locked.score);
                lo.put("lost", locked.lost);
                root.put("locked", lo);
            } else {
                root.put("locked", JSONObject.NULL);
            }
            latestStateJson = root.toString();
        } catch (Exception ignored) {}
    }

    @Override
    public Response serve(IHTTPSession session) {
        String uri = session.getUri();
        String code = session.getParms().get("code");

        if (roomCode != null && !roomCode.equals(code)) {
            return newFixedLengthResponse(Response.Status.FORBIDDEN, "application/json", "{\"error\":\"WRONG_CODE\"}");
        }

        if ("/ping".equals(uri)) {
            return newFixedLengthResponse(Response.Status.OK, "application/json",
                    "{\"ok\":true,\"model\":\"" + Build.MODEL.replace("\"", "") + "\"}");
        }

        if ("/state".equals(uri)) {
            lastViewerActivityTs = System.currentTimeMillis();
            return newFixedLengthResponse(Response.Status.OK, "application/json", latestStateJson);
        }

        if ("/cmd".equals(uri)) {
            lastViewerActivityTs = System.currentTimeMillis();
            String action = session.getParms().get("action");
            if (commandListener == null) {
                return newFixedLengthResponse(Response.Status.OK, "application/json", "{\"status\":\"ERR\"}");
            }
            if ("unlock".equals(action)) {
                commandListener.onRemoteUnlock();
                return newFixedLengthResponse(Response.Status.OK, "application/json", "{\"status\":\"OK\"}");
            } else if ("dismiss_alarm".equals(action)) {
                commandListener.onRemoteDismissAlarm();
                return newFixedLengthResponse(Response.Status.OK, "application/json", "{\"status\":\"OK\"}");
            } else if ("flip".equals(action)) {
                commandListener.onRemoteFlipCamera();
                return newFixedLengthResponse(Response.Status.OK, "application/json", "{\"status\":\"OK\"}");
            } else if ("torch".equals(action)) {
                commandListener.onRemoteToggleTorch();
                return newFixedLengthResponse(Response.Status.OK, "application/json", "{\"status\":\"OK\"}");
            } else if ("lock".equals(action)) {
                try {
                    float nx = Float.parseFloat(session.getParms().get("nx"));
                    float ny = Float.parseFloat(session.getParms().get("ny"));
                    String res = commandListener.onRemoteLock(nx, ny);
                    return newFixedLengthResponse(Response.Status.OK, "application/json", "{\"status\":\"" + res + "\"}");
                } catch (Exception e) {
                    return newFixedLengthResponse(Response.Status.BAD_REQUEST, "application/json", "{\"status\":\"ERR\"}");
                }
            }
            return newFixedLengthResponse(Response.Status.BAD_REQUEST, "application/json", "{\"status\":\"UNKNOWN\"}");
        }

        if ("/frame.jpg".equals(uri)) {
            lastViewerActivityTs = System.currentTimeMillis();
            byte[] snap = latestJpeg;
            if (snap == null) {
                return newFixedLengthResponse(Response.Status.NO_CONTENT, MIME_PLAINTEXT, "");
            }
            return newFixedLengthResponse(Response.Status.OK, "image/jpeg",
                    new java.io.ByteArrayInputStream(snap), snap.length);
        }

        if ("/stream".equals(uri) || "/".equals(uri)) {
            lastViewerActivityTs = System.currentTimeMillis();
            final PipedOutputStream out = new PipedOutputStream();
            final PipedInputStream in;
            try {
                in = new PipedInputStream(out, 131072);
            } catch (IOException e) {
                return newFixedLengthResponse(Response.Status.INTERNAL_ERROR, MIME_PLAINTEXT, "Stream Error");
            }

            new Thread(() -> {
                activeStreamClients.incrementAndGet();
                byte[] lastSent = null;
                try {
                    while (running && !Thread.currentThread().isInterrupted()) {
                        byte[] frame;
                        synchronized (frameLock) {
                            long waitStart = System.currentTimeMillis();
                            while (running && (latestJpeg == null || latestJpeg == lastSent)) {
                                frameLock.wait(200);
                                if (System.currentTimeMillis() - waitStart > 1000 && latestJpeg != null) break;
                            }
                            frame = latestJpeg;
                        }
                        if (!running || frame == null) continue;
                        lastSent = frame;

                        byte[] header = ("--mjpegframe\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + frame.length + "\r\n\r\n").getBytes(StandardCharsets.US_ASCII);
                        out.write(header);
                        out.write(frame);
                        out.write("\r\n".getBytes(StandardCharsets.US_ASCII));
                        out.flush();
                        Thread.sleep(25);
                    }
                } catch (Exception ignored) {
                } finally {
                    activeStreamClients.decrementAndGet();
                    try { out.close(); } catch (Exception ignored) {}
                    try { in.close(); } catch (Exception ignored) {}
                }
            }, "MJPEGClientWriter").start();

            Response res = newChunkedResponse(Response.Status.OK,
                    "multipart/x-mixed-replace; boundary=--mjpegframe", in);
            res.addHeader("Cache-Control", "no-cache, private");
            return res;
        }

        return newFixedLengthResponse(Response.Status.NOT_FOUND, MIME_PLAINTEXT, "Not Found");
    }

    private void startUdpDiscoveryBroadcaster() {
        udpBroadcastThread = new Thread(() -> {
            DatagramSocket socket = null;
            try {
                socket = new DatagramSocket();
                socket.setBroadcast(true);
                String safeModel = (Build.MODEL != null ? Build.MODEL : "Android").replace("|", " ");
                while (running && !Thread.currentThread().isInterrupted()) {
                    try {
                        List<String> myIps = getAllLocalIpv4Addresses();
                        if (!myIps.isEmpty()) {
                            List<InetAddress> targets = new ArrayList<>();
                            try {
                                targets.add(InetAddress.getByName("255.255.255.255"));
                            } catch (Exception ignored) {}

                            for (NetworkInterface nif : Collections.list(NetworkInterface.getNetworkInterfaces())) {
                                if (nif.isLoopback() || !nif.isUp()) continue;
                                for (InterfaceAddress ia : nif.getInterfaceAddresses()) {
                                    InetAddress bcast = ia.getBroadcast();
                                    if (bcast != null) targets.add(bcast);
                                }
                            }

                            for (String ip : myIps) {
                                int dot = ip.lastIndexOf('.');
                                if (dot > 0) {
                                    try {
                                        InetAddress subBcast = InetAddress.getByName(ip.substring(0, dot + 1) + "255");
                                        if (!targets.contains(subBcast)) targets.add(subBcast);
                                    } catch (Exception ignored) {}
                                }
                            }

                            for (String ip : myIps) {
                                String msg = "CAMPRO|" + roomCode + "|" + ip + "|" + port + "|" + safeModel;
                                byte[] data = msg.getBytes(StandardCharsets.UTF_8);
                                for (InetAddress bcast : targets) {
                                    try {
                                        socket.send(new DatagramPacket(data, data.length, bcast, DISCOVERY_UDP_PORT));
                                    } catch (Exception ignored) {}
                                }
                            }
                        }
                    } catch (Exception ignored) {}
                    Thread.sleep(700);
                }
            } catch (Exception e) {
                Log.w(TAG, "UDP yayın durduruldu: " + e.getMessage());
            } finally {
                if (socket != null) socket.close();
            }
        }, "UdpDiscoveryBroadcaster");
        udpBroadcastThread.start();
    }

    private static boolean isIgnoredInterfaceName(String name) {
        return name.contains("rmnet") || name.contains("ccmni") || name.contains("tun")
                || name.contains("ppp") || name.contains("clat") || name.contains("v4-")
                || name.contains("dummy");
    }

    private static boolean isPrivateIpv4(String ip) {
        if (ip == null) return false;
        if (ip.startsWith("192.168.") || ip.startsWith("10.")) return true;
        if (ip.startsWith("172.")) {
            try {
                int second = Integer.parseInt(ip.split("\\.")[1]);
                return second >= 16 && second <= 31;
            } catch (Exception ignored) {}
        }
        return false;
    }

    public static List<String> getAllLocalIpv4Addresses() {
        LinkedHashSet<String> prioritized = new LinkedHashSet<>();
        LinkedHashSet<String> secondary = new LinkedHashSet<>();
        try {
            Enumeration<NetworkInterface> interfaces = NetworkInterface.getNetworkInterfaces();
            while (interfaces != null && interfaces.hasMoreElements()) {
                NetworkInterface nif = interfaces.nextElement();
                if (nif.isLoopback() || !nif.isUp()) continue;
                String name = nif.getName().toLowerCase();
                if (isIgnoredInterfaceName(name)) continue;

                boolean preferredIf = name.contains("swlan") || name.contains("ap")
                        || name.contains("wlan") || name.contains("wifi")
                        || name.contains("p2p") || name.contains("rndis") || name.contains("eth");

                Enumeration<InetAddress> addrs = nif.getInetAddresses();
                while (addrs.hasMoreElements()) {
                    InetAddress addr = addrs.nextElement();
                    if (addr instanceof Inet4Address && !addr.isLoopbackAddress()) {
                        String ip = addr.getHostAddress();
                        if (!isPrivateIpv4(ip)) continue;
                        if (preferredIf) prioritized.add(ip);
                        else secondary.add(ip);
                    }
                }
            }
        } catch (Exception ignored) {}
        prioritized.addAll(secondary);
        return new ArrayList<>(prioritized);
    }

    public static String getLocalIpAddress() {
        List<String> ips = getAllLocalIpv4Addresses();
        return ips.isEmpty() ? null : ips.get(0);
    }
}
