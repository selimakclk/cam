package com.cam.ai;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;
import androidx.annotation.Nullable;
import java.util.ArrayList;
import java.util.List;

/**
 * 60 FPS Smooth Lerp, Apple Pro Vizör Braketleri, Focus Ring Animasyonu ve
 * Boş Alan Kilit Korumalı Profesyonel Kamera/İzleyici Overlay Katmanı.
 */
public class OverlayView extends View {

    public interface OnTapNormListener {
        void onTapNormalized(float normX, float normY, float rawX, float rawY);
    }

    private final Paint idleBracketPaint = new Paint();
    private final Paint idleDashPaint = new Paint();
    private final Paint lockedBracketPaint = new Paint();
    private final Paint lostBracketPaint = new Paint();
    private final Paint crossPaint = new Paint();
    private final Paint tagBgPaint = new Paint();
    private final Paint tagBorderPaint = new Paint();
    private final Paint tagTextPaint = new Paint();
    private final Paint ringPaint = new Paint();

    private final Object stateLock = new Object();
    private final List<TrackedPerson> persons = new ArrayList<>();
    private PersonDetectorEngine.LockedState lockedState = null;

    // 60 FPS onDraw için ön bellekli nesneler (sıfır GC yükü)
    private final RenderGeom renderGeom = new RenderGeom();
    private final RectF tempRect = new RectF();
    private final RectF tempTagRect = new RectF();
    private final Path tempCornerPath = new Path();
    private final float[] tempNormOut = new float[2];

    // 60 FPS Lerp yumuşatma için görüntülenen kutu
    private float lerpNx, lerpNy, lerpNw, lerpNh;
    private boolean hasLerp = false;
    private boolean hadDrawablesLastUpdate = false;

    // Kaynak video/resim boyutları ve ölçekleme modu
    private int sourceWidth = 720;
    private int sourceHeight = 1280;
    private boolean isFrontCamera = false;
    private boolean useFitCenter = false; // PreviewView=false (FILL_CENTER), Viewer ImageView=true (FIT_CENTER)

    // Dokunma odak halkası animasyonu
    private float ringX, ringY;
    private long ringStartTs = 0;
    private boolean ringIsWarning = false;
    private static final long RING_DURATION_MS = 480;

    private OnTapNormListener tapListener;

    public OverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        initPaints();
    }

    private void initPaints() {
        idleBracketPaint.setColor(Color.parseColor("#CCFFFFFF"));
        idleBracketPaint.setStyle(Paint.Style.STROKE);
        idleBracketPaint.setStrokeWidth(4.5f);
        idleBracketPaint.setStrokeCap(Paint.Cap.ROUND);
        idleBracketPaint.setAntiAlias(true);

        idleDashPaint.setColor(Color.parseColor("#28FFFFFF"));
        idleDashPaint.setStyle(Paint.Style.FILL);
        idleDashPaint.setAntiAlias(true);

        lockedBracketPaint.setColor(Color.parseColor("#30D158")); // Apple Pro Green
        lockedBracketPaint.setStyle(Paint.Style.STROKE);
        lockedBracketPaint.setStrokeWidth(7.5f);
        lockedBracketPaint.setStrokeCap(Paint.Cap.ROUND);
        lockedBracketPaint.setAntiAlias(true);

        lostBracketPaint.setColor(Color.parseColor("#FF453A")); // Apple Alarm Red
        lostBracketPaint.setStyle(Paint.Style.STROKE);
        lostBracketPaint.setStrokeWidth(8f);
        lostBracketPaint.setStrokeCap(Paint.Cap.ROUND);
        lostBracketPaint.setAntiAlias(true);

        crossPaint.setColor(Color.parseColor("#30D158"));
        crossPaint.setStyle(Paint.Style.STROKE);
        crossPaint.setStrokeWidth(3f);
        crossPaint.setStrokeCap(Paint.Cap.ROUND);
        crossPaint.setAntiAlias(true);

        tagBgPaint.setColor(Color.parseColor("#CC141418"));
        tagBgPaint.setStyle(Paint.Style.FILL);
        tagBgPaint.setAntiAlias(true);

        tagBorderPaint.setColor(Color.parseColor("#33FFFFFF"));
        tagBorderPaint.setStyle(Paint.Style.STROKE);
        tagBorderPaint.setStrokeWidth(2f);
        tagBorderPaint.setAntiAlias(true);

        tagTextPaint.setColor(Color.WHITE);
        tagTextPaint.setTextSize(30f);
        tagTextPaint.setFakeBoldText(true);
        tagTextPaint.setAntiAlias(true);

        ringPaint.setStyle(Paint.Style.STROKE);
        ringPaint.setStrokeWidth(5f);
        ringPaint.setAntiAlias(true);
    }

    public void setOnTapNormListener(OnTapNormListener listener) {
        this.tapListener = listener;
    }

    public void setScaleModeFitCenter(boolean fitCenter) {
        this.useFitCenter = fitCenter;
    }

    public void setSourceInfo(int width, int height, boolean frontCamera) {
        synchronized (stateLock) {
            if (this.sourceWidth == width && this.sourceHeight == height && this.isFrontCamera == frontCamera) {
                return;
            }
            if (width > 0 && height > 0) {
                this.sourceWidth = width;
                this.sourceHeight = height;
            }
            this.isFrontCamera = frontCamera;
        }
    }

    public void updateState(List<TrackedPerson> detected, PersonDetectorEngine.LockedState locked) {
        boolean shouldDraw;
        synchronized (stateLock) {
            persons.clear();
            if (detected != null) {
                persons.addAll(detected);
            }
            if (locked != null) {
                if (this.lockedState == null || this.lockedState.id != locked.id) {
                    lerpNx = locked.nx;
                    lerpNy = locked.ny;
                    lerpNw = locked.nw;
                    lerpNh = locked.nh;
                    hasLerp = true;
                }
                this.lockedState = locked;
            } else {
                this.lockedState = null;
                hasLerp = false;
            }
            boolean hasNow = !persons.isEmpty() || locked != null || ringStartTs > 0;
            shouldDraw = hasNow || hadDrawablesLastUpdate;
            hadDrawablesLastUpdate = hasNow;
        }
        if (shouldDraw) {
            postInvalidateOnAnimation();
        }
    }

    public void triggerFocusRing(float screenX, float screenY, boolean isWarning) {
        this.ringX = screenX;
        this.ringY = screenY;
        this.ringIsWarning = isWarning;
        this.ringStartTs = System.currentTimeMillis();
        postInvalidateOnAnimation();
    }

    private static class RenderGeom {
        float rw, rh, ox, oy;
    }

    private void updateRenderGeometry(int srcW, int srcH) {
        float cw = getWidth();
        float ch = getHeight();
        if (cw <= 0 || ch <= 0 || srcW <= 0 || srcH <= 0) {
            renderGeom.rw = cw;
            renderGeom.rh = ch;
            renderGeom.ox = 0;
            renderGeom.oy = 0;
            return;
        }
        float va = (float) srcW / (float) srcH;
        float ca = cw / ch;

        if (useFitCenter) {
            // FIT_CENTER (İzleyici ImageView)
            if (ca > va) {
                renderGeom.rh = ch;
                renderGeom.rw = ch * va;
                renderGeom.oy = 0;
                renderGeom.ox = (renderGeom.rw - cw) * 0.5f;
            } else {
                renderGeom.rw = cw;
                renderGeom.rh = cw / va;
                renderGeom.ox = 0;
                renderGeom.oy = (renderGeom.rh - ch) * 0.5f;
            }
        } else {
            // FILL_CENTER (CameraX PreviewView varsayılanı)
            if (ca > va) {
                renderGeom.rw = cw;
                renderGeom.rh = cw / va;
                renderGeom.ox = 0;
                renderGeom.oy = (renderGeom.rh - ch) * 0.5f;
            } else {
                renderGeom.rh = ch;
                renderGeom.rw = ch * va;
                renderGeom.oy = 0;
                renderGeom.ox = (renderGeom.rw - cw) * 0.5f;
            }
        }
    }

    private void screenToNormalized(float sx, float sy, float[] out) {
        int srcW, srcH;
        boolean front;
        synchronized (stateLock) {
            srcW = sourceWidth;
            srcH = sourceHeight;
            front = isFrontCamera;
        }
        updateRenderGeometry(srcW, srcH);
        if (renderGeom.rw <= 0 || renderGeom.rh <= 0) {
            out[0] = 0.5f;
            out[1] = 0.5f;
            return;
        }
        float nx = (sx + renderGeom.ox) / renderGeom.rw;
        float ny = (sy + renderGeom.oy) / renderGeom.rh;
        if (front) {
            nx = 1f - nx;
        }
        out[0] = Math.max(0f, Math.min(1f, nx));
        out[1] = Math.max(0f, Math.min(1f, ny));
    }

    private void fillScreenRect(float nx, float ny, float nw, float nh, boolean front, RectF out) {
        float x = front ? (1f - (nx + nw)) : nx;
        out.left = x * renderGeom.rw - renderGeom.ox;
        out.top = ny * renderGeom.rh - renderGeom.oy;
        out.right = out.left + nw * renderGeom.rw;
        out.bottom = out.top + nh * renderGeom.rh;
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (event.getAction() == MotionEvent.ACTION_DOWN) {
            float sx = event.getX();
            float sy = event.getY();
            screenToNormalized(sx, sy, tempNormOut);
            triggerFocusRing(sx, sy, false);
            if (tapListener != null) {
                tapListener.onTapNormalized(tempNormOut[0], tempNormOut[1], sx, sy);
            }
            return true;
        }
        return super.onTouchEvent(event);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        long now = System.currentTimeMillis();
        boolean needContinueAnim = false;

        PersonDetectorEngine.LockedState snapLocked;
        int personCount;
        int srcW, srcH;
        boolean front;
        synchronized (stateLock) {
            srcW = sourceWidth;
            srcH = sourceHeight;
            front = isFrontCamera;
            snapLocked = lockedState;
            updateRenderGeometry(srcW, srcH);

            // 1. Kilitli hedef varsa Smooth Lerp ile çiz
            if (snapLocked != null) {
                if (!hasLerp) {
                    lerpNx = snapLocked.nx;
                    lerpNy = snapLocked.ny;
                    lerpNw = snapLocked.nw;
                    lerpNh = snapLocked.nh;
                    hasLerp = true;
                } else {
                    float k = 0.36f;
                    lerpNx += (snapLocked.nx - lerpNx) * k;
                    lerpNy += (snapLocked.ny - lerpNy) * k;
                    lerpNw += (snapLocked.nw - lerpNw) * k;
                    lerpNh += (snapLocked.nh - lerpNh) * k;
                    if (Math.abs(snapLocked.nx - lerpNx) > 0.001f
                            || Math.abs(snapLocked.ny - lerpNy) > 0.001f
                            || Math.abs(snapLocked.nw - lerpNw) > 0.0015f
                            || Math.abs(snapLocked.nh - lerpNh) > 0.0015f) {
                        needContinueAnim = true;
                    }
                }
                fillScreenRect(lerpNx, lerpNy, lerpNw, lerpNh, front, tempRect);
                int pct = Math.round(snapLocked.score * 100f);
                boolean hasCustomName = (snapLocked.name != null && !snapLocked.name.trim().isEmpty());
                String aiBadge;
                if (snapLocked.hasFaceAi && snapLocked.hasDeepAi) {
                    aiBadge = " • Pro AI";
                } else if (snapLocked.hasDeepAi) {
                    aiBadge = " • OSNet";
                } else if (snapLocked.hasFaceAi) {
                    aiBadge = " • Yüz";
                } else {
                    aiBadge = "";
                }
                String label;
                if (snapLocked.lost) {
                    label = hasCustomName ? ("KAYIP: " + snapLocked.name.trim() + "!") : "HEDEF KAYIP!";
                } else if (snapLocked.isOccluded) {
                    label = hasCustomName
                            ? ("BEKLENİYOR • " + snapLocked.name.trim() + aiBadge)
                            : ("BEKLENİYOR #" + snapLocked.id + aiBadge);
                } else {
                    label = hasCustomName
                            ? ("KİLİTLİ • " + snapLocked.name.trim() + " • %" + pct + aiBadge)
                            : ("KİLİTLİ #" + snapLocked.id + " • %" + pct + aiBadge);
                }
                drawLockedViewfinder(canvas, tempRect, snapLocked.lost, label, now);
                if (snapLocked.lost) needContinueAnim = true;
            } else {
                personCount = persons.size();
                for (int i = 0; i < personCount; i++) {
                    TrackedPerson p = persons.get(i);
                    fillScreenRect(p.nx, p.ny, p.nw, p.nh, front, tempRect);
                    int pct = Math.round(p.score * 100f);
                    boolean named = (p.name != null && !p.name.trim().isEmpty());
                    String badge;
                    if (p.hasFaceAi && p.hasDeepAi) badge = " • Pro AI";
                    else if (p.hasDeepAi) badge = " • OSNet";
                    else if (p.hasFaceAi) badge = " • Yüz";
                    else badge = "";
                    String idleLabel = named
                            ? (p.name.trim() + " • %" + pct + badge)
                            : ("İNSAN #" + p.id + " • %" + pct + badge);
                    drawIdlePerson(canvas, tempRect, idleLabel);
                }
            }
        }

        // 3. Odak halkası animasyonu
        if (ringStartTs > 0) {
            long elapsed = now - ringStartTs;
            if (elapsed < RING_DURATION_MS) {
                float p = (float) elapsed / RING_DURATION_MS;
                float radius = 64f - 24f * p;
                int alpha = Math.round((1f - p) * 255f);
                ringPaint.setColor(ringIsWarning ? Color.parseColor("#FFD60A") : Color.WHITE);
                ringPaint.setAlpha(alpha);
                canvas.drawCircle(ringX, ringY, radius, ringPaint);
                needContinueAnim = true;
            } else {
                ringStartTs = 0;
            }
        }

        if (needContinueAnim) {
            postInvalidateOnAnimation();
        }
    }

    private void drawIdlePerson(Canvas canvas, RectF r, String label) {
        canvas.drawRoundRect(r, 12f, 12f, idleDashPaint);

        float corner = Math.max(18f, Math.min(44f, Math.min(r.width(), r.height()) * 0.20f));
        fillCornerPath(r, corner);
        canvas.drawPath(tempCornerPath, idleBracketPaint);

        float textW = tagTextPaint.measureText(label);
        float tagTop = Math.max(8f, r.top - 48f);
        float tagLeft = Math.max(8f, Math.min(getWidth() - textW - 32f, r.left));
        tempTagRect.set(tagLeft, tagTop, tagLeft + textW + 24f, tagTop + 38f);
        canvas.drawRoundRect(tempTagRect, 12f, 12f, tagBgPaint);
        canvas.drawRoundRect(tempTagRect, 12f, 12f, tagBorderPaint);
        canvas.drawText(label, tagLeft + 12f, tagTop + 28f, tagTextPaint);
    }

    private void drawLockedViewfinder(Canvas canvas, RectF r, boolean lost, String label, long now) {
        Paint bp = lost ? lostBracketPaint : lockedBracketPaint;
        if (lost) {
            int pulseAlpha = (int) (140 + 115 * Math.sin(now * 0.015));
            bp.setAlpha(Math.max(80, Math.min(255, pulseAlpha)));
        } else {
            bp.setAlpha(255);
        }

        float corner = Math.max(24f, Math.min(60f, Math.min(r.width(), r.height()) * 0.25f));
        fillCornerPath(r, corner);
        canvas.drawPath(tempCornerPath, bp);

        float cx = r.centerX();
        float cy = r.centerY();
        float cs = 18f;
        crossPaint.setColor(lost ? Color.parseColor("#FF453A") : Color.parseColor("#30D158"));
        canvas.drawLine(cx - cs, cy, cx + cs, cy, crossPaint);
        canvas.drawLine(cx, cy - cs, cx, cy + cs, crossPaint);

        float textW = tagTextPaint.measureText(label);
        float tagW = textW + 32f;
        float tagLeft = Math.max(12f, Math.min(getWidth() - tagW - 12f, cx - tagW * 0.5f));
        float tagTop = Math.max(12f, r.top - 52f);
        tempTagRect.set(tagLeft, tagTop, tagLeft + tagW, tagTop + 42f);

        tagBgPaint.setColor(lost ? Color.parseColor("#DDFF3B30") : Color.parseColor("#CC141418"));
        canvas.drawRoundRect(tempTagRect, 14f, 14f, tagBgPaint);
        tagBgPaint.setColor(Color.parseColor("#CC141418"));
        canvas.drawRoundRect(tempTagRect, 14f, 14f, tagBorderPaint);
        canvas.drawText(label, tagLeft + 16f, tagTop + 31f, tagTextPaint);
    }

    private void fillCornerPath(RectF r, float c) {
        tempCornerPath.rewind();
        tempCornerPath.moveTo(r.left, r.top + c);
        tempCornerPath.lineTo(r.left, r.top);
        tempCornerPath.lineTo(r.left + c, r.top);

        tempCornerPath.moveTo(r.right - c, r.top);
        tempCornerPath.lineTo(r.right, r.top);
        tempCornerPath.lineTo(r.right, r.top + c);

        tempCornerPath.moveTo(r.left, r.bottom - c);
        tempCornerPath.lineTo(r.left, r.bottom);
        tempCornerPath.lineTo(r.left + c, r.bottom);

        tempCornerPath.moveTo(r.right - c, r.bottom);
        tempCornerPath.lineTo(r.right, r.bottom);
        tempCornerPath.lineTo(r.right, r.bottom - c);
    }
}
