package com.cam.ai;

/**
 * Normalize [0..1] kamera/görüntü koordinatlarında tespit edilen bir insanı temsil eder.
 */
public class TrackedPerson {
    public int id;
    public float nx;
    public float ny;
    public float nw;
    public float nh;
    public float score;
    public float[] signature;
    public long lastSeen;

    public TrackedPerson(int id, float nx, float ny, float nw, float nh, float score) {
        this.id = id;
        this.nx = nx;
        this.ny = ny;
        this.nw = nw;
        this.nh = nh;
        this.score = score;
        this.lastSeen = System.currentTimeMillis();
    }

    public float centerX() {
        return nx + nw * 0.5f;
    }

    public float centerY() {
        return ny + nh * 0.5f;
    }

    public float area() {
        return Math.max(1e-6f, nw * nh);
    }

    public boolean contains(float px, float py) {
        return px >= nx && px <= (nx + nw) && py >= ny && py <= (ny + nh);
    }

    public float distanceToPoint(float px, float py) {
        float dx = Math.max(nx - px, Math.max(0f, px - (nx + nw)));
        float dy = Math.max(ny - py, Math.max(0f, py - (ny + nh)));
        return (float) Math.hypot(dx, dy);
    }
}
