#define COBJMACROS
#define WIN32_LEAN_AND_MEAN
#define UNICODE
#define _UNICODE
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <windowsx.h>
#include <commctrl.h>
#include <iphlpapi.h>
#include <mmsystem.h>
#include <objbase.h>
#include <wincodec.h>
#include <dwmapi.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ==================== SABİTLER & KONTROL ID'LERİ ==================== */
#define APP_TITLE_W         L"CAM AI İzleyici (Ultra-Hafif)"
#define WND_CLASS_W         L"CamAiUltraLightViewerWnd"
#define CAM_HTTP_PORT       8080
#define CAM_UDP_PORT        8888
#define MAX_PERSONS         16

#define IDC_EDIT_CODE       101
#define IDC_EDIT_IP         102
#define IDC_BTN_CONNECT     103
#define IDC_BTN_UNLOCK      104
#define IDC_BTN_MUTE        105
#define IDC_SLIDER_VOL      106
#define IDC_BTN_BEEP_TEST   107
#define IDC_LBL_VOL         108
#define IDC_LBL_STATUS      109

#define HOTKEY_MUTE_ID      1
#define TIMER_PULSE_ID      2

#define WM_APP_NEW_FRAME    (WM_APP + 1)
#define WM_APP_STATE_UPD    (WM_APP + 2)
#define WM_APP_CONN_UPD     (WM_APP + 3)
#define WM_APP_LOCK_RES     (WM_APP + 4)

#define TOP_BAR_H           42
#define BOT_BAR_H           40

/* ==================== VERİ YAPILARI ==================== */
typedef struct {
    int id;
    char name[64];
    float x, y, w, h, s;
    int has_deep;
    int has_face;
} PersonBox;

typedef struct {
    int valid;
    int id;
    char name[64];
    float x, y, w, h, s;
    int lost;
    int has_deep;
    int has_face;
    int mem_count;
} LockedBox;

typedef struct {
    int src_w;
    int src_h;
    int is_front;
    int alarm;
    int person_count;
    PersonBox persons[MAX_PERSONS];
    LockedBox locked;
} AiState;

typedef struct {
    uint32_t *pixels; /* Top-down 32bpp BGRX */
    int w;
    int h;
    size_t cap_pixels;
} FrameBuffer;

typedef struct {
    char code[16];
    char manual_ip[64];
    char last_ip[64];
    int volume;
    int win_w;
    int win_h;
} AppConfig;

/* ==================== GLOBAL DURUM (SIFIR HARİCİ KÜTÜPHANE) ==================== */
static HINSTANCE g_hinst = NULL;
static HWND g_hwnd = NULL;
static HWND g_edit_code = NULL;
static HWND g_edit_ip = NULL;
static HWND g_btn_connect = NULL;
static HWND g_btn_unlock = NULL;
static HWND g_btn_mute = NULL;
static HWND g_slider_vol = NULL;
static HWND g_btn_test = NULL;
static HWND g_lbl_vol = NULL;
static HWND g_lbl_status = NULL;

static HFONT g_font_ui = NULL;
static HFONT g_font_bold = NULL;
static HFONT g_font_tag = NULL;
static HBRUSH g_br_bar = NULL;
static HBRUSH g_br_edit = NULL;

static WCHAR g_ini_path[MAX_PATH];
static AppConfig g_cfg;

static CRITICAL_SECTION g_state_cs;
static CRITICAL_SECTION g_frame_cs;
static CRITICAL_SECTION g_conn_cs;

static AiState g_ai_state;
static FrameBuffer g_frame = {0};

static volatile LONG g_running = 1;
static volatile LONG g_window_active = 1; /* 1=Önde (Video Aktif), 0=Arkada/Oyunda (0-Yük Uyku!) */
static volatile LONG g_connected = 0;
static volatile LONG g_reconnect_req = 0;
static volatile LONG g_volume = 80;
static volatile LONG g_alarm_beeping = 0;
static volatile LONG g_muted_current_alarm = 0;
static volatile LONG g_test_beep_req = 0;
static volatile LONG g_hotkey_ctrl_shift = 0;

static char g_active_ip[64] = {0};
static int g_active_port = CAM_HTTP_PORT;
static char g_active_model[64] = {0};
static char g_active_code[16] = "12345";
static char g_conn_status_utf8[160] = "Kamera aranıyor...";
static char g_toast_utf8[128] = {0};
static DWORD g_toast_until = 0;

static volatile SOCKET g_video_sock = INVALID_SOCKET;
static volatile SOCKET g_state_sock = INVALID_SOCKET;
static HANDLE g_video_wake_ev = NULL;
static HANDLE g_audio_wake_ev = NULL;
static HANDLE g_state_wake_ev = NULL;
static HANDLE g_th_state = NULL;
static HANDLE g_th_video = NULL;
static HANDLE g_th_audio = NULL;

/* Flicker-free çift tampon GDI */
static HDC g_mem_dc = NULL;
static HBITMAP g_mem_bmp = NULL;
static HBITMAP g_mem_old = NULL;
static int g_mem_w = 0, g_mem_h = 0;

/* Son çizilen video dikdörtgeni (tıklayıp kilitleme için) */
static RECT g_video_rect = {0, 0, 0, 0};
static int g_ring_x = 0, g_ring_y = 0, g_ring_warn = 0;
static DWORD g_ring_start = 0;

/* ==================== YARDIMCI UTF-8 & AYAR FONKSİYONLARI ==================== */
static void utf8_to_wide(const char *src, WCHAR *dst, int dst_chars) {
    if (!dst || dst_chars <= 0) return;
    if (!src || !src[0]) { dst[0] = 0; return; }
    int n = MultiByteToWideChar(CP_UTF8, 0, src, -1, dst, dst_chars);
    if (n <= 0) dst[0] = 0;
}

static void wide_to_utf8(const WCHAR *src, char *dst, int dst_bytes) {
    if (!dst || dst_bytes <= 0) return;
    if (!src || !src[0]) { dst[0] = 0; return; }
    int n = WideCharToMultiByte(CP_UTF8, 0, src, -1, dst, dst_bytes, NULL, NULL);
    if (n <= 0) dst[0] = 0;
}

static void init_ini_path(void) {
    WCHAR exe[MAX_PATH];
    DWORD len = GetModuleFileNameW(NULL, exe, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) {
        lstrcpynW(g_ini_path, L"cam_viewer.ini", MAX_PATH);
        return;
    }
    WCHAR *slash = wcsrchr(exe, L'\\');
    if (slash) *(slash + 1) = 0;
    _snwprintf(g_ini_path, MAX_PATH - 1, L"%scam_viewer.ini", exe);
    g_ini_path[MAX_PATH - 1] = 0;
}

static void load_config(void) {
    WCHAR buf[128];
    init_ini_path();

    GetPrivateProfileStringW(L"Viewer", L"code", L"12345", buf, 16, g_ini_path);
    wide_to_utf8(buf, g_cfg.code, sizeof(g_cfg.code));
    if (strlen(g_cfg.code) != 5) strcpy(g_cfg.code, "12345");

    GetPrivateProfileStringW(L"Viewer", L"manual_ip", L"", buf, 64, g_ini_path);
    wide_to_utf8(buf, g_cfg.manual_ip, sizeof(g_cfg.manual_ip));

    GetPrivateProfileStringW(L"Viewer", L"last_ip", L"", buf, 64, g_ini_path);
    wide_to_utf8(buf, g_cfg.last_ip, sizeof(g_cfg.last_ip));

    g_cfg.volume = (int)GetPrivateProfileIntW(L"Viewer", L"volume", 80, g_ini_path);
    if (g_cfg.volume < 0) g_cfg.volume = 0;
    if (g_cfg.volume > 100) g_cfg.volume = 100;

    g_cfg.win_w = (int)GetPrivateProfileIntW(L"Viewer", L"width", 880, g_ini_path);
    g_cfg.win_h = (int)GetPrivateProfileIntW(L"Viewer", L"height", 620, g_ini_path);
    if (g_cfg.win_w < 540) g_cfg.win_w = 540;
    if (g_cfg.win_h < 380) g_cfg.win_h = 380;

    InterlockedExchange(&g_volume, g_cfg.volume);
    EnterCriticalSection(&g_conn_cs);
    strncpy(g_active_code, g_cfg.code, sizeof(g_active_code) - 1);
    LeaveCriticalSection(&g_conn_cs);
}

static void save_config(void) {
    WCHAR wbuf[128];
    utf8_to_wide(g_cfg.code, wbuf, 128);
    WritePrivateProfileStringW(L"Viewer", L"code", wbuf, g_ini_path);
    utf8_to_wide(g_cfg.manual_ip, wbuf, 128);
    WritePrivateProfileStringW(L"Viewer", L"manual_ip", wbuf, g_ini_path);
    utf8_to_wide(g_cfg.last_ip, wbuf, 128);
    WritePrivateProfileStringW(L"Viewer", L"last_ip", wbuf, g_ini_path);

    _snwprintf(wbuf, 32, L"%d", (int)InterlockedCompareExchange(&g_volume, 0, 0));
    WritePrivateProfileStringW(L"Viewer", L"volume", wbuf, g_ini_path);
    _snwprintf(wbuf, 32, L"%d", g_cfg.win_w);
    WritePrivateProfileStringW(L"Viewer", L"width", wbuf, g_ini_path);
    _snwprintf(wbuf, 32, L"%d", g_cfg.win_h);
    WritePrivateProfileStringW(L"Viewer", L"height", wbuf, g_ini_path);
}

static void set_conn_status(const char *msg) {
    EnterCriticalSection(&g_conn_cs);
    strncpy(g_conn_status_utf8, msg, sizeof(g_conn_status_utf8) - 1);
    g_conn_status_utf8[sizeof(g_conn_status_utf8) - 1] = 0;
    LeaveCriticalSection(&g_conn_cs);
    if (g_hwnd) PostMessageW(g_hwnd, WM_APP_CONN_UPD, 0, 0);
}

/* ==================== SAF WINSOCK HIZLI HTTP İSTEMCİSİ ==================== */
static SOCKET connect_tcp_timeout(const char *ip, int port, int timeout_ms) {
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return INVALID_SOCKET;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((u_short)port);
    if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1) {
        closesocket(s);
        return INVALID_SOCKET;
    }

    u_long nonblock = 1;
    ioctlsocket(s, FIONBIO, &nonblock);
    int r = connect(s, (struct sockaddr *)&addr, sizeof(addr));
    if (r == SOCKET_ERROR) {
        int err = WSAGetLastError();
        if (err != WSAEWOULDBLOCK && err != WSAEINPROGRESS) {
            closesocket(s);
            return INVALID_SOCKET;
        }
        fd_set wfds, efds;
        FD_ZERO(&wfds);
        FD_ZERO(&efds);
        FD_SET(s, &wfds);
        FD_SET(s, &efds);
        struct timeval tv;
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        int sel = select(0, NULL, &wfds, &efds, &tv);
        if (sel <= 0 || FD_ISSET(s, &efds) || !FD_ISSET(s, &wfds)) {
            closesocket(s);
            return INVALID_SOCKET;
        }
        int so_err = 0;
        int so_len = sizeof(so_err);
        if (getsockopt(s, SOL_SOCKET, SO_ERROR, (char *)&so_err, &so_len) != 0 || so_err != 0) {
            closesocket(s);
            return INVALID_SOCKET;
        }
    }

    nonblock = 0;
    ioctlsocket(s, FIONBIO, &nonblock);
    DWORD rto = (DWORD)timeout_ms;
    DWORD sto = (DWORD)timeout_ms;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&rto, sizeof(rto));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char *)&sto, sizeof(sto));
    int nodelay = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *)&nodelay, sizeof(nodelay));
    return s;
}

static int http_get_ex(const char *ip, int port, const char *path, int timeout_ms,
                       char *out_body, int out_cap, int *out_body_len, volatile SOCKET *cancel_slot) {
    if (out_body && out_cap > 0) out_body[0] = 0;
    if (out_body_len) *out_body_len = 0;

    SOCKET s = connect_tcp_timeout(ip, port, timeout_ms < 1200 ? timeout_ms : 1200);
    if (s == INVALID_SOCKET) return -1;
    DWORD rto = (DWORD)timeout_ms;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&rto, sizeof(rto));
    if (cancel_slot) InterlockedExchangePointer((PVOID *)cancel_slot, (PVOID)s);

    char req[512];
    int req_len = _snprintf(req, sizeof(req) - 1,
        "GET %s HTTP/1.1\r\nHost: %s:%d\r\nConnection: close\r\nUser-Agent: CamViewerC/2.3\r\n\r\n",
        path, ip, port);
    if (req_len <= 0 || send(s, req, req_len, 0) != req_len) {
        if (cancel_slot) InterlockedCompareExchangePointer((PVOID *)cancel_slot, (PVOID)INVALID_SOCKET, (PVOID)s);
        closesocket(s);
        return -1;
    }

    int buf_cap = (out_cap > 0 ? out_cap : 1024) + 1024;
    char *raw = (char *)malloc((size_t)buf_cap);
    if (!raw) {
        if (cancel_slot) InterlockedCompareExchangePointer((PVOID *)cancel_slot, (PVOID)INVALID_SOCKET, (PVOID)s);
        closesocket(s);
        return -1;
    }
    int total = 0;
    while (total < buf_cap - 1) {
        int n = recv(s, raw + total, buf_cap - 1 - total, 0);
        if (n <= 0) break;
        total += n;
    }
    raw[total] = 0;
    if (cancel_slot) InterlockedCompareExchangePointer((PVOID *)cancel_slot, (PVOID)INVALID_SOCKET, (PVOID)s);
    closesocket(s);

    int status = -1;
    if (total >= 12 && strncmp(raw, "HTTP/", 5) == 0) {
        const char *sp = strchr(raw, ' ');
        if (sp) status = atoi(sp + 1);
    }
    char *hdr_end = strstr(raw, "\r\n\r\n");
    if (hdr_end && out_body && out_cap > 0) {
        char *body = hdr_end + 4;
        int blen = total - (int)(body - raw);
        if (blen < 0) blen = 0;
        if (blen >= out_cap) blen = out_cap - 1;
        memcpy(out_body, body, (size_t)blen);
        out_body[blen] = 0;
        if (out_body_len) *out_body_len = blen;
    }
    free(raw);
    return status;
}

static int http_get_simple(const char *ip, int port, const char *path, int timeout_ms,
                           char *out_body, int out_cap, int *out_body_len) {
    return http_get_ex(ip, port, path, timeout_ms, out_body, out_cap, out_body_len, NULL);
}

/* ==================== HAFİF & GÜVENLİ C JSON AYRIŞTIRICI ==================== */
static const char *json_find_key_in_range(const char *start, const char *end, const char *key) {
    char pat[64];
    int plen = _snprintf(pat, sizeof(pat) - 1, "\"%s\"", key);
    if (plen <= 0) return NULL;
    const char *p = start;
    while (p < end) {
        const char *hit = strstr(p, pat);
        if (!hit || hit >= end) return NULL;
        const char *q = hit + plen;
        while (q < end && (*q == ' ' || *q == '\t' || *q == '\r' || *q == '\n')) q++;
        if (q < end && *q == ':') {
            q++;
            while (q < end && (*q == ' ' || *q == '\t' || *q == '\r' || *q == '\n')) q++;
            return q;
        }
        p = hit + 1;
    }
    return NULL;
}

static int json_get_int(const char *s, const char *e, const char *key, int defv) {
    const char *v = json_find_key_in_range(s, e, key);
    if (!v || v >= e) return defv;
    return atoi(v);
}

static float json_get_float(const char *s, const char *e, const char *key, float defv) {
    const char *v = json_find_key_in_range(s, e, key);
    if (!v || v >= e) return defv;
    return (float)atof(v);
}

static int json_get_bool(const char *s, const char *e, const char *key, int defv) {
    const char *v = json_find_key_in_range(s, e, key);
    if (!v || v >= e) return defv;
    if (strncmp(v, "true", 4) == 0) return 1;
    if (strncmp(v, "false", 5) == 0) return 0;
    return defv;
}

static void json_get_str(const char *s, const char *e, const char *key, char *out, int out_cap) {
    out[0] = 0;
    const char *v = json_find_key_in_range(s, e, key);
    if (!v || v >= e || *v != '"') return;
    v++;
    int oi = 0;
    while (v < e && *v && *v != '"' && oi < out_cap - 1) {
        if (*v == '\\' && (v + 1) < e) {
            v++;
            if (*v == 'u' && (v + 4) < e) {
                /* \uXXXX -> UTF-8 */
                unsigned int cp = 0;
                char hex[5] = {v[1], v[2], v[3], v[4], 0};
                cp = (unsigned int)strtoul(hex, NULL, 16);
                v += 5;
                if (cp < 0x80 && oi < out_cap - 1) {
                    out[oi++] = (char)cp;
                } else if (cp < 0x800 && oi < out_cap - 2) {
                    out[oi++] = (char)(0xC0 | (cp >> 6));
                    out[oi++] = (char)(0x80 | (cp & 0x3F));
                } else if (oi < out_cap - 3) {
                    out[oi++] = (char)(0xE0 | (cp >> 12));
                    out[oi++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    out[oi++] = (char)(0x80 | (cp & 0x3F));
                }
                continue;
            }
        }
        out[oi++] = *v++;
    }
    out[oi] = 0;
}

static const char *json_find_matching(const char *open_ch, const char *end, char open_c, char close_c) {
    if (!open_ch || open_ch >= end || *open_ch != open_c) return NULL;
    int depth = 0, in_str = 0;
    for (const char *p = open_ch; p < end; p++) {
        if (in_str) {
            if (*p == '\\' && (p + 1) < end) { p++; continue; }
            if (*p == '"') in_str = 0;
            continue;
        }
        if (*p == '"') { in_str = 1; continue; }
        if (*p == open_c) depth++;
        else if (*p == close_c) {
            depth--;
            if (depth == 0) return p;
        }
    }
    return NULL;
}

static int parse_ai_state_json(const char *json, int len, AiState *out) {
    memset(out, 0, sizeof(*out));
    out->src_w = 480;
    out->src_h = 640;
    if (!json || len < 10) return 0;

    const char *start = json;
    const char *end = json + len;
    const char *persons_val = json_find_key_in_range(start, end, "persons");
    const char *locked_val = json_find_key_in_range(start, end, "locked");

    /* Üst seviye alanlar (persons/locked içindeki alanlarla karışmaması için başlangıçtan tara) */
    const char *top_end = persons_val ? persons_val : (locked_val ? locked_val : end);
    out->src_w = json_get_int(start, top_end, "w", 480);
    out->src_h = json_get_int(start, top_end, "h", 640);
    out->is_front = json_get_bool(start, top_end, "front", 0);
    out->alarm = json_get_bool(start, end, "alarm", 0);

    if (persons_val && *persons_val == '[') {
        const char *arr_end = json_find_matching(persons_val, end, '[', ']');
        if (arr_end) {
            const char *p = persons_val + 1;
            while (p < arr_end && out->person_count < MAX_PERSONS) {
                while (p < arr_end && *p != '{') p++;
                if (p >= arr_end) break;
                const char *obj_end = json_find_matching(p, arr_end, '{', '}');
                if (!obj_end) break;
                PersonBox *pb = &out->persons[out->person_count];
                pb->id = json_get_int(p, obj_end, "id", out->person_count + 1);
                json_get_str(p, obj_end, "name", pb->name, sizeof(pb->name));
                pb->x = json_get_float(p, obj_end, "x", 0.0f);
                pb->y = json_get_float(p, obj_end, "y", 0.0f);
                pb->w = json_get_float(p, obj_end, "w", 0.0f);
                pb->h = json_get_float(p, obj_end, "h", 0.0f);
                pb->s = json_get_float(p, obj_end, "s", 0.5f);
                pb->has_deep = json_get_bool(p, obj_end, "deep", 0);
                pb->has_face = json_get_bool(p, obj_end, "face", 0);
                out->person_count++;
                p = obj_end + 1;
            }
        }
    }

    if (locked_val && *locked_val == '{') {
        const char *obj_end = json_find_matching(locked_val, end, '{', '}');
        if (obj_end) {
            out->locked.valid = 1;
            out->locked.id = json_get_int(locked_val, obj_end, "id", 1);
            json_get_str(locked_val, obj_end, "name", out->locked.name, sizeof(out->locked.name));
            out->locked.x = json_get_float(locked_val, obj_end, "x", 0.0f);
            out->locked.y = json_get_float(locked_val, obj_end, "y", 0.0f);
            out->locked.w = json_get_float(locked_val, obj_end, "w", 0.0f);
            out->locked.h = json_get_float(locked_val, obj_end, "h", 0.0f);
            out->locked.s = json_get_float(locked_val, obj_end, "s", 0.9f);
            out->locked.lost = json_get_bool(locked_val, obj_end, "lost", 0);
            out->locked.has_deep = json_get_bool(locked_val, obj_end, "deep", 0);
            out->locked.has_face = json_get_bool(locked_val, obj_end, "face", 0);
            out->locked.mem_count = json_get_int(locked_val, obj_end, "mem", 0);
        }
    }
    return 1;
}

/* ==================== ASENKRON UZAKTAN KOMUT GÖNDERİCİ ==================== */
typedef struct {
    char path[192];
    int notify_lock;
} RemoteCmdTask;

static DWORD WINAPI remote_cmd_thread(LPVOID param) {
    RemoteCmdTask *task = (RemoteCmdTask *)param;
    char ip[64];
    int port;
    EnterCriticalSection(&g_conn_cs);
    strncpy(ip, g_active_ip, sizeof(ip) - 1);
    ip[sizeof(ip) - 1] = 0;
    port = g_active_port;
    LeaveCriticalSection(&g_conn_cs);

    if (ip[0] && InterlockedCompareExchange(&g_connected, 0, 0)) {
        char body[256];
        int blen = 0;
        int code = http_get_simple(ip, port, task->path, 1500, body, sizeof(body), &blen);
        if (task->notify_lock && g_hwnd) {
            int ok = (code == 200 && strstr(body, "LOCKED") != NULL);
            PostMessageW(g_hwnd, WM_APP_LOCK_RES, (WPARAM)ok, 0);
        }
    }
    free(task);
    return 0;
}

static void send_remote_cmd_async(const char *action_and_args, int notify_lock) {
    if (!InterlockedCompareExchange(&g_connected, 0, 0)) return;
    RemoteCmdTask *t = (RemoteCmdTask *)calloc(1, sizeof(RemoteCmdTask));
    if (!t) return;
    char code[16];
    EnterCriticalSection(&g_conn_cs);
    strncpy(code, g_active_code, sizeof(code) - 1);
    code[sizeof(code) - 1] = 0;
    LeaveCriticalSection(&g_conn_cs);

    _snprintf(t->path, sizeof(t->path) - 1, "/cmd?code=%s&%s", code, action_and_args);
    t->notify_lock = notify_lock;
    HANDLE h = CreateThread(NULL, 0, remote_cmd_thread, t, 0, NULL);
    if (h) CloseHandle(h);
    else free(t);
}

/* ==================== ÇİFT TON BİP-BİP SES SENTEZLEYİCİ (WINMM) ==================== */
#define AUDIO_SR        44100
#define BEEP_TONE_MS    105
#define BEEP_GAP_MS     75
#define BEEP_TOTAL_MS   (BEEP_TONE_MS * 2 + BEEP_GAP_MS + 20)
#define BEEP_SAMPLES    ((AUDIO_SR * BEEP_TOTAL_MS) / 1000)

static void render_dual_beep_pcm(int16_t *pcm, int vol_pct) {
    if (vol_pct < 0) vol_pct = 0;
    if (vol_pct > 100) vol_pct = 100;
    double gain = pow((double)vol_pct / 100.0, 1.55) * 0.85 * 32767.0;
    int s_tone = (AUDIO_SR * BEEP_TONE_MS) / 1000;
    int s_gap  = (AUDIO_SR * BEEP_GAP_MS) / 1000;
    int s_fade = (AUDIO_SR * 8) / 1000;
    int t2_start = s_tone + s_gap;
    int t2_end   = t2_start + s_tone;
    const double two_pi = 6.28318530717958647692;

    for (int i = 0; i < BEEP_SAMPLES; i++) {
        double v = 0.0;
        if (i < s_tone) {
            double env = 1.0;
            if (i < s_fade) env = (double)i / s_fade;
            else if (i > s_tone - s_fade) env = (double)(s_tone - i) / s_fade;
            v = sin(two_pi * 1180.0 * i / AUDIO_SR) * env * gain;
        } else if (i >= t2_start && i < t2_end) {
            int j = i - t2_start;
            double env = 1.0;
            if (j < s_fade) env = (double)j / s_fade;
            else if (j > s_tone - s_fade) env = (double)(s_tone - j) / s_fade;
            v = sin(two_pi * 1520.0 * j / AUDIO_SR) * env * gain;
        }
        if (v > 32767.0) v = 32767.0;
        if (v < -32767.0) v = -32767.0;
        pcm[i] = (int16_t)v;
    }
}

static DWORD WINAPI audio_thread_proc(LPVOID unused) {
    (void)unused;
    int16_t *pcm = (int16_t *)malloc(BEEP_SAMPLES * sizeof(int16_t));
    if (!pcm) return 0;

    WAVEFORMATEX wfx;
    memset(&wfx, 0, sizeof(wfx));
    wfx.wFormatTag = WAVE_FORMAT_PCM;
    wfx.nChannels = 1;
    wfx.nSamplesPerSec = AUDIO_SR;
    wfx.wBitsPerSample = 16;
    wfx.nBlockAlign = 2;
    wfx.nAvgBytesPerSec = AUDIO_SR * 2;

    HANDLE done_ev = CreateEventW(NULL, FALSE, FALSE, NULL);

    while (InterlockedCompareExchange(&g_running, 0, 0)) {
        int test_req = InterlockedExchange(&g_test_beep_req, 0);
        int alarm_on = InterlockedCompareExchange(&g_alarm_beeping, 0, 0);

        if (!test_req && !alarm_on) {
            /* Sessizken 0% CPU ve açık ses aygıtı YOK */
            WaitForSingleObject(g_audio_wake_ev, INFINITE);
            continue;
        }

        int vol = (int)InterlockedCompareExchange(&g_volume, 0, 0);
        if (vol <= 0 && !test_req) {
            WaitForSingleObject(g_audio_wake_ev, 400);
            continue;
        }
        render_dual_beep_pcm(pcm, vol > 0 ? vol : 0);

        HWAVEOUT hwo = NULL;
        if (vol > 0 && waveOutOpen(&hwo, WAVE_MAPPER, &wfx, (DWORD_PTR)done_ev, 0, CALLBACK_EVENT) == MMSYSERR_NOERROR) {
            WAVEHDR hdr;
            memset(&hdr, 0, sizeof(hdr));
            hdr.lpData = (LPSTR)pcm;
            hdr.dwBufferLength = BEEP_SAMPLES * sizeof(int16_t);
            waveOutPrepareHeader(hwo, &hdr, sizeof(hdr));
            ResetEvent(done_ev);
            waveOutWrite(hwo, &hdr, sizeof(hdr));

            /* Çalarken F9 veya çıkış gelirse 20ms içinde anında kes */
            DWORD start_ms = GetTickCount();
            while (GetTickCount() - start_ms < (DWORD)(BEEP_TOTAL_MS + 150)) {
                if (!InterlockedCompareExchange(&g_running, 0, 0)) {
                    waveOutReset(hwo);
                    break;
                }
                if (!test_req && !InterlockedCompareExchange(&g_alarm_beeping, 0, 0)) {
                    waveOutReset(hwo);
                    break;
                }
                if (WaitForSingleObject(done_ev, 25) == WAIT_OBJECT_0) {
                    if (hdr.dwFlags & WHDR_DONE) break;
                }
            }
            waveOutUnprepareHeader(hwo, &hdr, sizeof(hdr));
            waveOutClose(hwo);
        }

        /* Sürekli alarm modunda bip-bip arası bekleme (F9 anında uyandırır) */
        if (InterlockedCompareExchange(&g_alarm_beeping, 0, 0) && InterlockedCompareExchange(&g_running, 0, 0)) {
            WaitForSingleObject(g_audio_wake_ev, 620);
        }
    }

    CloseHandle(done_ev);
    free(pcm);
    return 0;
}

/* ==================== AĞ KEŞFİ (UDP 8888 + PARALEL ALT AĞ TARAMA) ==================== */
static int verify_camera_ping(const char *ip, int port, const char *code, int timeout_ms, char *out_model, int model_cap) {
    char path[96];
    char body[256];
    int blen = 0;
    _snprintf(path, sizeof(path) - 1, "/ping?code=%s", code);
    int st = http_get_simple(ip, port, path, timeout_ms, body, sizeof(body), &blen);
    if (st == 200 && strstr(body, "\"ok\":true")) {
        if (out_model && model_cap > 0) {
            json_get_str(body, body + blen, "model", out_model, model_cap);
            if (!out_model[0]) strncpy(out_model, "Kamera", (size_t)model_cap - 1);
        }
        return 200;
    }
    if (st == 403 && strstr(body, "WRONG_CODE")) return 403;
    return -1;
}

static int listen_udp_discovery(const char *code, int wait_ms, char *out_ip, int *out_port, char *out_model) {
    SOCKET s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == INVALID_SOCKET) return 0;
    int reuse = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *)&reuse, sizeof(reuse));
    DWORD rto = 250;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&rto, sizeof(rto));

    struct sockaddr_in bind_addr;
    memset(&bind_addr, 0, sizeof(bind_addr));
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_addr.s_addr = INADDR_ANY;
    bind_addr.sin_port = htons(CAM_UDP_PORT);
    if (bind(s, (struct sockaddr *)&bind_addr, sizeof(bind_addr)) != 0) {
        closesocket(s);
        return 0;
    }

    DWORD deadline = GetTickCount() + (DWORD)wait_ms;
    char buf[512];
    while (InterlockedCompareExchange(&g_running, 0, 0)
           && !InterlockedCompareExchange(&g_reconnect_req, 0, 0)
           && (int)(deadline - GetTickCount()) > 0) {
        struct sockaddr_in from;
        int from_len = sizeof(from);
        int n = recvfrom(s, buf, sizeof(buf) - 1, 0, (struct sockaddr *)&from, &from_len);
        if (n <= 0) continue;
        buf[n] = 0;
        if (strncmp(buf, "CAMPRO|", 7) != 0) continue;

        /* CAMPRO|code|ip|port|model */
        char *p1 = buf + 7;
        char *p2 = strchr(p1, '|'); if (!p2) continue; *p2++ = 0;
        char *p3 = strchr(p2, '|'); if (!p3) continue; *p3++ = 0;
        char *p4 = strchr(p3, '|'); if (!p4) continue; *p4++ = 0;
        if (strcmp(p1, code) != 0) continue;

        int port = atoi(p3);
        if (port <= 0 || port > 65535) port = CAM_HTTP_PORT;
        char sender_ip[64] = {0};
        inet_ntop(AF_INET, &from.sin_addr, sender_ip, sizeof(sender_ip));

        if (verify_camera_ping(p2, port, code, 450, out_model, 64) == 200) {
            strncpy(out_ip, p2, 63);
            *out_port = port;
            closesocket(s);
            return 1;
        }
        if (sender_ip[0] && strcmp(sender_ip, p2) != 0
            && verify_camera_ping(sender_ip, port, code, 450, out_model, 64) == 200) {
            strncpy(out_ip, sender_ip, 63);
            *out_port = port;
            closesocket(s);
            return 1;
        }
    }
    closesocket(s);
    return 0;
}

static int scan_subnet_fast(const char *code, char *out_ip, int *out_port, char *out_model, int *out_wrong_code) {
    *out_wrong_code = 0;
    ULONG buf_len = 16384;
    PIP_ADAPTER_ADDRESSES addrs = (PIP_ADAPTER_ADDRESSES)malloc(buf_len);
    if (!addrs) return 0;
    if (GetAdaptersAddresses(AF_INET, GAA_FLAG_INCLUDE_GATEWAYS, NULL, addrs, &buf_len) == ERROR_BUFFER_OVERFLOW) {
        free(addrs);
        addrs = (PIP_ADAPTER_ADDRESSES)malloc(buf_len);
        if (!addrs) return 0;
        if (GetAdaptersAddresses(AF_INET, GAA_FLAG_INCLUDE_GATEWAYS, NULL, addrs, &buf_len) != NO_ERROR) {
            free(addrs);
            return 0;
        }
    }

    /* Yerel özel IPv4 alt ağlarını topla (en fazla 3 alt ağ) */
    uint32_t prefixes[4];
    int pcount = 0;
    for (PIP_ADAPTER_ADDRESSES a = addrs; a && pcount < 3; a = a->Next) {
        if (a->OperStatus != IfOperStatusUp || a->IfType == IF_TYPE_SOFTWARE_LOOPBACK) continue;
        for (PIP_ADAPTER_UNICAST_ADDRESS u = a->FirstUnicastAddress; u && pcount < 3; u = u->Next) {
            if (!u->Address.lpSockaddr || u->Address.lpSockaddr->sa_family != AF_INET) continue;
            struct sockaddr_in *sin = (struct sockaddr_in *)u->Address.lpSockaddr;
            uint32_t ip_h = ntohl(sin->sin_addr.s_addr);
            uint8_t o1 = (uint8_t)(ip_h >> 24), o2 = (uint8_t)(ip_h >> 16);
            int is_priv = (o1 == 192 && o2 == 168) || (o1 == 10) || (o1 == 172 && o2 >= 16 && o2 <= 31);
            if (!is_priv) continue;
            uint32_t pfx = ip_h & 0xFFFFFF00u;
            int dup = 0;
            for (int k = 0; k < pcount; k++) if (prefixes[k] == pfx) dup = 1;
            if (!dup) prefixes[pcount++] = pfx;
        }
    }
    free(addrs);

    /* Yumuşak paralel TCP port 8080 taraması (12'li partiler + ara nefes -> router ping'i yükseltmez) */
    for (int pi = 0; pi < pcount; pi++) {
        for (int base = 1; base <= 254; base += 12) {
            if (!InterlockedCompareExchange(&g_running, 0, 0)
                || InterlockedCompareExchange(&g_reconnect_req, 0, 0)
                || !InterlockedCompareExchange(&g_window_active, 0, 0))
                return 0;

            SOCKET socks[12];
            uint32_t ips[12];
            int sc = 0;
            fd_set wfds;
            FD_ZERO(&wfds);

            for (int j = 0; j < 12 && (base + j) <= 254; j++) {
                uint32_t host_ip = prefixes[pi] | (uint32_t)(base + j);
                SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                if (s == INVALID_SOCKET) continue;
                u_long nb = 1;
                ioctlsocket(s, FIONBIO, &nb);
                struct sockaddr_in sa;
                memset(&sa, 0, sizeof(sa));
                sa.sin_family = AF_INET;
                sa.sin_port = htons(CAM_HTTP_PORT);
                sa.sin_addr.s_addr = htonl(host_ip);
                connect(s, (struct sockaddr *)&sa, sizeof(sa));
                socks[sc] = s;
                ips[sc] = host_ip;
                FD_SET(s, &wfds);
                sc++;
            }
            if (sc == 0) continue;

            struct timeval tv;
            tv.tv_sec = 0;
            tv.tv_usec = 160000; /* 160ms */
            int sel = select(0, NULL, &wfds, NULL, &tv);
            for (int j = 0; j < sc; j++) {
                int open_ok = 0;
                if (sel > 0 && FD_ISSET(socks[j], &wfds)) {
                    int err = 0, elen = sizeof(err);
                    if (getsockopt(socks[j], SOL_SOCKET, SO_ERROR, (char *)&err, &elen) == 0 && err == 0)
                        open_ok = 1;
                }
                closesocket(socks[j]);
                if (open_ok) {
                    struct in_addr ia;
                    ia.s_addr = htonl(ips[j]);
                    char cand[64];
                    inet_ntop(AF_INET, &ia, cand, sizeof(cand));
                    int pr = verify_camera_ping(cand, CAM_HTTP_PORT, code, 450, out_model, 64);
                    if (pr == 200) {
                        strncpy(out_ip, cand, 63);
                        *out_port = CAM_HTTP_PORT;
                        for (int k = j + 1; k < sc; k++) closesocket(socks[k]);
                        return 1;
                    } else if (pr == 403) {
                        *out_wrong_code = 1;
                    }
                }
            }
        }
    }
    return 0;
}

/* ==================== DURUM & ALARM İŞ PARÇACIĞI (ARKADA 0.0% GPU!) ==================== */
static void set_window_active_state(HWND hwnd, int is_active);

static void stop_active_video_socket(void) {
    SOCKET s = (SOCKET)InterlockedExchangePointer((PVOID *)&g_video_sock, (PVOID)INVALID_SOCKET);
    if (s != INVALID_SOCKET) {
        shutdown(s, SD_BOTH);
        closesocket(s);
    }
}

static void stop_active_state_socket(void) {
    SOCKET s = (SOCKET)InterlockedExchangePointer((PVOID *)&g_state_sock, (PVOID)INVALID_SOCKET);
    if (s != INVALID_SOCKET) {
        shutdown(s, SD_BOTH);
    }
}

/* Windows EcoQoS (Verimlilik Modu) + IDLE öncelik -> oyundaki FPS'ye %0.0 etki! */
static void apply_gaming_eco_qos(int enable_eco) {
    SetPriorityClass(GetCurrentProcess(), enable_eco ? IDLE_PRIORITY_CLASS : NORMAL_PRIORITY_CLASS);
    if (g_th_state) {
        SetThreadPriority(g_th_state, enable_eco ? THREAD_PRIORITY_IDLE : THREAD_PRIORITY_NORMAL);
    }
    HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
    if (k32) {
        typedef BOOL(WINAPI *SetProcInfoFn)(HANDLE, int, LPVOID, DWORD);
        SetProcInfoFn fn = (SetProcInfoFn)GetProcAddress(k32, "SetProcessInformation");
        if (fn) {
            struct {
                ULONG Version;
                ULONG ControlMask;
                ULONG StateMask;
            } pts;
            pts.Version = 1; /* PROCESS_POWER_THROTTLING_CURRENT_VERSION */
            pts.ControlMask = 0x1; /* PROCESS_POWER_THROTTLING_EXECUTION_SPEED */
            pts.StateMask = enable_eco ? 0x1 : 0x0;
            fn(GetCurrentProcess(), 4 /* ProcessPowerThrottling */, &pts, sizeof(pts));
        }
    }
}

static DWORD WINAPI state_thread_proc(LPVOID unused) {
    (void)unused;
    char state_body[8192];
    int prev_target_lost = 0;

    while (InterlockedCompareExchange(&g_running, 0, 0)) {
        InterlockedExchange(&g_reconnect_req, 0);
        InterlockedExchange(&g_connected, 0);
        InterlockedExchange(&g_alarm_beeping, 0);
        InterlockedExchange(&g_muted_current_alarm, 0);
        prev_target_lost = 0;
        stop_active_video_socket();
        EnterCriticalSection(&g_frame_cs);
        g_frame.w = 0;
        g_frame.h = 0;
        LeaveCriticalSection(&g_frame_cs);
        SetEvent(g_audio_wake_ev);

        char code[16], manual_ip[64], last_ip[64];
        EnterCriticalSection(&g_conn_cs);
        strncpy(code, g_active_code, sizeof(code) - 1); code[sizeof(code) - 1] = 0;
        strncpy(manual_ip, g_cfg.manual_ip, sizeof(manual_ip) - 1); manual_ip[sizeof(manual_ip) - 1] = 0;
        strncpy(last_ip, g_cfg.last_ip, sizeof(last_ip) - 1); last_ip[sizeof(last_ip) - 1] = 0;
        LeaveCriticalSection(&g_conn_cs);

        char found_ip[64] = {0};
        int found_port = CAM_HTTP_PORT;
        char found_model[64] = "Kamera";
        int connected_ok = 0;
        int wrong_code = 0;

        /* 1. Manuel IP varsa doğrudan ona dene (ip:port destekli) */
        if (manual_ip[0]) {
            char mip[64];
            int mport = CAM_HTTP_PORT;
            strncpy(mip, manual_ip, sizeof(mip) - 1); mip[sizeof(mip) - 1] = 0;
            char *colon = strchr(mip, ':');
            if (colon) { *colon = 0; int p = atoi(colon + 1); if (p > 0) mport = p; }

            char msg[128];
            _snprintf(msg, sizeof(msg) - 1, "%s adresine bağlanılıyor...", mip);
            set_conn_status(msg);
            int pr = verify_camera_ping(mip, mport, code, 900, found_model, sizeof(found_model));
            if (pr == 200) {
                strncpy(found_ip, mip, sizeof(found_ip) - 1);
                found_port = mport;
                connected_ok = 1;
            } else if (pr == 403) {
                wrong_code = 1;
            }
        }

        /* 2. Son bilinen IP'yi hızlıca dene (anında yeniden bağlanma) */
        if (!connected_ok && !wrong_code && last_ip[0] && strcmp(last_ip, manual_ip) != 0) {
            set_conn_status("Son kameraya bağlanılıyor...");
            int pr = verify_camera_ping(last_ip, CAM_HTTP_PORT, code, 450, found_model, sizeof(found_model));
            if (pr == 200) {
                strncpy(found_ip, last_ip, sizeof(found_ip) - 1);
                found_port = CAM_HTTP_PORT;
                connected_ok = 1;
            } else if (pr == 403) {
                wrong_code = 1;
            }
        }

        /* 3. UDP 8888 otomatik keşif */
        if (!connected_ok && !wrong_code && !manual_ip[0]) {
            set_conn_status("Aynı ağda kamera aranıyor (UDP)...");
            if (listen_udp_discovery(code, 1200, found_ip, &found_port, found_model)) {
                connected_ok = 1;
            }
        }

        /* 4. Yerel alt ağ taraması: SADECE pencere öndeyken çalışır (oyundayken ASLA alt ağ tarayıp ping yükseltmez!) */
        if (!connected_ok && !wrong_code && !manual_ip[0] && InterlockedCompareExchange(&g_window_active, 0, 0)) {
            set_conn_status("Yerel ağ taranıyor...");
            if (scan_subnet_fast(code, found_ip, &found_port, found_model, &wrong_code)) {
                connected_ok = 1;
            }
        }

        if (!InterlockedCompareExchange(&g_running, 0, 0)) break;
        if (InterlockedCompareExchange(&g_reconnect_req, 0, 0)) continue;

        if (wrong_code) {
            set_conn_status("Hatalı 5 haneli şifre! Şifreyi düzeltip Bağlan'a basın.");
            WaitForSingleObject(g_state_wake_ev, 4000);
            continue;
        }
        if (!connected_ok) {
            set_conn_status("Kamera bulunamadı — Tekrar aranıyor...");
            /* Oyundayken 4 saniye bekle (sıfır ağ yükü), öndeyken 1.2 saniye */
            WaitForSingleObject(g_state_wake_ev, InterlockedCompareExchange(&g_window_active, 0, 0) ? 1200 : 4000);
            continue;
        }

        /* Bağlantı başarılı! */
        EnterCriticalSection(&g_conn_cs);
        strncpy(g_active_ip, found_ip, sizeof(g_active_ip) - 1);
        g_active_port = found_port;
        strncpy(g_active_model, found_model, sizeof(g_active_model) - 1);
        if (strcmp(g_cfg.last_ip, found_ip) != 0) {
            strncpy(g_cfg.last_ip, found_ip, sizeof(g_cfg.last_ip) - 1);
        }
        LeaveCriticalSection(&g_conn_cs);
        save_config();

        InterlockedExchange(&g_connected, 1);
        SetEvent(g_video_wake_ev);

        char ok_msg[140];
        _snprintf(ok_msg, sizeof(ok_msg) - 1, "Bağlı • %s (%s)", found_model, found_ip);
        set_conn_status(ok_msg);

        /* Sürekli sıfır-ping /state kontrol döngüsü (oyundayken uzun-beklemeli itme: ~80 bayt/sn!) */
        char state_path[96];
        int fail_streak = 0;

        while (InterlockedCompareExchange(&g_running, 0, 0)
               && !InterlockedCompareExchange(&g_reconnect_req, 0, 0)) {
            /* Tam ekran oyunlar WM_ACTIVATE göndermese bile ön-plan kontrolüyle 0-yük uykuyu garanti et */
            if (g_hwnd) {
                int fg_now = (!IsIconic(g_hwnd) && IsWindowVisible(g_hwnd) && GetForegroundWindow() == g_hwnd) ? 1 : 0;
                if (fg_now != (int)InterlockedCompareExchange(&g_window_active, 0, 0)) {
                    set_window_active_state(g_hwnd, fg_now);
                }
            }

            int is_active = (int)InterlockedCompareExchange(&g_window_active, 0, 0);
            int timeout_ms;
            if (is_active) {
                _snprintf(state_path, sizeof(state_path) - 1, "/state?code=%s", code);
                timeout_ms = 1000;
            } else {
                /* Oyundayken telefona wait=1&alarm=X gönder: 2.4sn boyunca ağda 0 paket gider, alarm değişince <1ms uyanır! */
                _snprintf(state_path, sizeof(state_path) - 1, "/state?code=%s&wait=1&alarm=%d", code, prev_target_lost ? 1 : 0);
                timeout_ms = 3800;
            }

            DWORD t_req0 = GetTickCount();
            int blen = 0;
            int st = http_get_ex(found_ip, found_port, state_path, timeout_ms, state_body, sizeof(state_body), &blen, &g_state_sock);
            DWORD req_elapsed = GetTickCount() - t_req0;

            /* Alt+Tab ile öne geçerken uzun-bekleme iptal edildiyse hata sayma */
            if (st < 0 && (int)InterlockedCompareExchange(&g_window_active, 0, 0) != is_active) {
                continue;
            }

            if (st == 200 && blen > 2) {
                fail_streak = 0;
                AiState parsed;
                if (parse_ai_state_json(state_body, blen, &parsed)) {
                    EnterCriticalSection(&g_state_cs);
                    g_ai_state = parsed;
                    LeaveCriticalSection(&g_state_cs);

                    /* Yalnızca kilitli hedef kaybolduğunda (alarm=true veya locked.lost=true) bip-bip! */
                    int target_lost_now = (parsed.locked.valid && parsed.locked.lost) || (parsed.locked.valid && parsed.alarm);
                    int alarm_changed = (target_lost_now != prev_target_lost);
                    if (target_lost_now && !prev_target_lost) {
                        InterlockedExchange(&g_muted_current_alarm, 0);
                        InterlockedExchange(&g_alarm_beeping, 1);
                        SetEvent(g_audio_wake_ev);
                    } else if (target_lost_now && prev_target_lost) {
                        if (!InterlockedCompareExchange(&g_muted_current_alarm, 0, 0)) {
                            if (InterlockedExchange(&g_alarm_beeping, 1) == 0)
                                SetEvent(g_audio_wake_ev);
                        }
                    } else if (!target_lost_now && prev_target_lost) {
                        InterlockedExchange(&g_alarm_beeping, 0);
                        InterlockedExchange(&g_muted_current_alarm, 0);
                        SetEvent(g_audio_wake_ev);
                    }
                    prev_target_lost = target_lost_now;

                    /* Oyundayken UI thread'ini ASLA uyandırma (0 DWM / 0 GDI / 0 FPS etki!) */
                    if (g_hwnd && (InterlockedCompareExchange(&g_window_active, 0, 0) || alarm_changed)) {
                        PostMessageW(g_hwnd, WM_APP_STATE_UPD, 0, 0);
                    }
                }
            } else if (st == 403) {
                set_conn_status("Şifre değişti / hatalı!");
                break;
            } else {
                fail_streak++;
                if (fail_streak >= 4) {
                    set_conn_status("Bağlantı koptu, yeniden bağlanılıyor...");
                    break;
                }
            }

            if (InterlockedCompareExchange(&g_window_active, 0, 0)) {
                WaitForSingleObject(g_state_wake_ev, 180);
            } else {
                /* v2.3 telefon uzun-bekleme destekliyorsa (req_elapsed >= 400ms) hemen sessiz beklemeye dön;
                   eski telefon hemen yanıt döndüyse 650ms uyu ki ping'e sıfır etki olsun */
                DWORD sleep_ms = (req_elapsed < 400) ? 650 : 60;
                WaitForSingleObject(g_state_wake_ev, sleep_ms);
            }
        }
    }
    return 0;
}

/* ==================== DONANIM-HIZLI WIC JPEG ÇÖZÜCÜ & VİDEO AKIŞI ==================== */
static int decode_jpeg_wic(IWICImagingFactory *factory, const BYTE *jpeg_data, DWORD jpeg_len) {
    if (!factory || !jpeg_data || jpeg_len < 16) return 0;

    IWICStream *stream = NULL;
    IWICBitmapDecoder *decoder = NULL;
    IWICBitmapFrameDecode *frame = NULL;
    IWICFormatConverter *conv = NULL;
    int ok = 0;

    if (SUCCEEDED(IWICImagingFactory_CreateStream(factory, &stream))) {
        if (SUCCEEDED(IWICStream_InitializeFromMemory(stream, (BYTE *)jpeg_data, jpeg_len))) {
            if (SUCCEEDED(IWICImagingFactory_CreateDecoderFromStream(
                    factory, (IStream *)stream, NULL, WICDecodeMetadataCacheOnDemand, &decoder))) {
                if (SUCCEEDED(IWICBitmapDecoder_GetFrame(decoder, 0, &frame))) {
                    UINT w = 0, h = 0;
                    IWICBitmapFrameDecode_GetSize(frame, &w, &h);
                    if (w > 0 && h > 0 && w <= 4096 && h <= 4096) {
                        if (SUCCEEDED(IWICImagingFactory_CreateFormatConverter(factory, &conv))) {
                            if (SUCCEEDED(IWICFormatConverter_Initialize(
                                    conv, (IWICBitmapSource *)frame, &GUID_WICPixelFormat32bppBGR,
                                    WICBitmapDitherTypeNone, NULL, 0.0, WICBitmapPaletteTypeCustom))) {
                                size_t need = (size_t)w * (size_t)h;
                                EnterCriticalSection(&g_frame_cs);
                                if (need > g_frame.cap_pixels) {
                                    uint32_t *np = (uint32_t *)realloc(g_frame.pixels, need * sizeof(uint32_t));
                                    if (np) {
                                        g_frame.pixels = np;
                                        g_frame.cap_pixels = need;
                                    }
                                }
                                if (g_frame.pixels && g_frame.cap_pixels >= need) {
                                    if (SUCCEEDED(IWICFormatConverter_CopyPixels(
                                            conv, NULL, w * 4, (UINT)(need * 4), (BYTE *)g_frame.pixels))) {
                                        g_frame.w = (int)w;
                                        g_frame.h = (int)h;
                                        ok = 1;
                                    }
                                }
                                LeaveCriticalSection(&g_frame_cs);
                            }
                            IWICFormatConverter_Release(conv);
                        }
                    }
                    IWICBitmapFrameDecode_Release(frame);
                }
                IWICBitmapDecoder_Release(decoder);
            }
        }
        IWICStream_Release(stream);
    }
    return ok;
}

typedef struct {
    SOCKET s;
    BYTE buf[16384];
    int pos;
    int len;
    int chunked;
    int chunk_rem;
} SockReader;

static int sr_read_raw_byte(SockReader *r) {
    if (r->pos >= r->len) {
        r->pos = 0;
        r->len = recv(r->s, (char *)r->buf, sizeof(r->buf), 0);
        if (r->len <= 0) return -1;
    }
    return (int)r->buf[r->pos++];
}

static int sr_read_raw_line(SockReader *r, char *out, int cap) {
    int i = 0;
    while (i < cap - 1) {
        int c = sr_read_raw_byte(r);
        if (c < 0) return -1;
        if (c == '\n') {
            if (i > 0 && out[i - 1] == '\r') i--;
            out[i] = 0;
            return i;
        }
        out[i++] = (char)c;
    }
    out[i] = 0;
    return i;
}

static int sr_read_byte(SockReader *r) {
    if (!r->chunked) return sr_read_raw_byte(r);
    while (r->chunk_rem <= 0) {
        char szline[64];
        int l = sr_read_raw_line(r, szline, sizeof(szline));
        if (l < 0) return -1;
        if (l == 0) continue; /* Önceki chunk sonu CRLF */
        long csz = strtol(szline, NULL, 16);
        if (csz <= 0) return -1;
        r->chunk_rem = (int)csz;
    }
    int b = sr_read_raw_byte(r);
    if (b >= 0) r->chunk_rem--;
    return b;
}

static int sr_read_line(SockReader *r, char *out, int cap) {
    int i = 0;
    while (i < cap - 1) {
        int c = sr_read_byte(r);
        if (c < 0) return -1;
        if (c == '\n') {
            if (i > 0 && out[i - 1] == '\r') i--;
            out[i] = 0;
            return i;
        }
        out[i++] = (char)c;
    }
    out[i] = 0;
    return i;
}

static int sr_read_exact(SockReader *r, BYTE *dst, int count) {
    int got = 0;
    if (!r->chunked) {
        while (got < count) {
            int avail = r->len - r->pos;
            if (avail > 0) {
                int take = (count - got < avail) ? (count - got) : avail;
                memcpy(dst + got, r->buf + r->pos, (size_t)take);
                r->pos += take;
                got += take;
            } else {
                int n = recv(r->s, (char *)(dst + got), count - got, 0);
                if (n <= 0) return 0;
                got += n;
            }
        }
        return 1;
    }
    while (got < count) {
        while (r->chunk_rem <= 0) {
            char szline[64];
            int l = sr_read_raw_line(r, szline, sizeof(szline));
            if (l < 0) return 0;
            if (l == 0) continue;
            long csz = strtol(szline, NULL, 16);
            if (csz <= 0) return 0;
            r->chunk_rem = (int)csz;
        }
        int avail = r->len - r->pos;
        if (avail <= 0) {
            r->pos = 0;
            r->len = recv(r->s, (char *)r->buf, sizeof(r->buf), 0);
            if (r->len <= 0) return 0;
            avail = r->len;
        }
        int take = count - got;
        if (take > r->chunk_rem) take = r->chunk_rem;
        if (take > avail) take = avail;
        memcpy(dst + got, r->buf + r->pos, (size_t)take);
        r->pos += take;
        r->chunk_rem -= take;
        got += take;
    }
    return 1;
}

static int fetch_single_frame_jpg(IWICImagingFactory *wic, const char *ip, int port, const char *code,
                                  BYTE **jpeg_buf, int *jpeg_cap) {
    SOCKET s = connect_tcp_timeout(ip, port, 1000);
    if (s == INVALID_SOCKET) return 0;
    InterlockedExchangePointer((PVOID *)&g_video_sock, (PVOID)s);

    char req[256];
    int rlen = _snprintf(req, sizeof(req) - 1,
        "GET /frame.jpg?code=%s HTTP/1.1\r\nHost: %s:%d\r\nConnection: close\r\n\r\n", code, ip, port);
    if (send(s, req, rlen, 0) != rlen) { stop_active_video_socket(); return 0; }

    SockReader sr;
    memset(&sr, 0, sizeof(sr));
    sr.s = s;
    char line[256];
    int status = 0, clen = -1;
    if (sr_read_raw_line(&sr, line, sizeof(line)) > 0 && strncmp(line, "HTTP/", 5) == 0) {
        char *sp = strchr(line, ' ');
        if (sp) status = atoi(sp + 1);
    }
    while (sr_read_raw_line(&sr, line, sizeof(line)) > 0) {
        if (_strnicmp(line, "Content-Length:", 15) == 0) clen = atoi(line + 15);
    }

    int ok = 0;
    if (status == 200 && clen > 16 && clen < 4 * 1024 * 1024) {
        if (clen > *jpeg_cap) {
            BYTE *nb = (BYTE *)realloc(*jpeg_buf, (size_t)clen);
            if (nb) { *jpeg_buf = nb; *jpeg_cap = clen; }
        }
        if (*jpeg_buf && *jpeg_cap >= clen && sr_read_exact(&sr, *jpeg_buf, clen)) {
            if (decode_jpeg_wic(wic, *jpeg_buf, (DWORD)clen)) {
                ok = 1;
                if (g_hwnd) PostMessageW(g_hwnd, WM_APP_NEW_FRAME, 0, 0);
            }
        }
    }
    stop_active_video_socket();
    return ok;
}

static DWORD WINAPI video_thread_proc(LPVOID unused) {
    (void)unused;
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
    IWICImagingFactory *wic = NULL;
    CoCreateInstance(&CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER,
                     &IID_IWICImagingFactory, (void **)&wic);

    int jpeg_cap = 0;
    BYTE *jpeg_buf = NULL;
    int stream_fails = 0;

    while (InterlockedCompareExchange(&g_running, 0, 0)) {
        /* 0-YÜK UYKU MODU: Bağlı değilsek VEYA pencere arka plandaysa (oyundayken) TAMAMEN UYU ve tamponu boşalt! */
        if (!InterlockedCompareExchange(&g_connected, 0, 0)
            || !InterlockedCompareExchange(&g_window_active, 0, 0)) {
            stop_active_video_socket();
            if (jpeg_buf) {
                free(jpeg_buf);
                jpeg_buf = NULL;
                jpeg_cap = 0;
            }
            WaitForSingleObject(g_video_wake_ev, INFINITE);
            continue;
        }
        if (!jpeg_buf) {
            jpeg_cap = 131072;
            jpeg_buf = (BYTE *)malloc((size_t)jpeg_cap);
            if (!jpeg_buf) { WaitForSingleObject(g_video_wake_ev, 200); continue; }
        }

        char ip[64], code[16];
        int port;
        EnterCriticalSection(&g_conn_cs);
        strncpy(ip, g_active_ip, sizeof(ip) - 1); ip[sizeof(ip) - 1] = 0;
        strncpy(code, g_active_code, sizeof(code) - 1); code[sizeof(code) - 1] = 0;
        port = g_active_port;
        LeaveCriticalSection(&g_conn_cs);
        if (!ip[0]) { WaitForSingleObject(g_video_wake_ev, 200); continue; }

        /* Alt+Tab'da anında ilk kareyi göstermek için hızlı tek kare (veya MJPEG yedeği) */
        if (stream_fails >= 2) {
            if (fetch_single_frame_jpg(wic, ip, port, code, &jpeg_buf, &jpeg_cap)) {
                stream_fails = 0;
                WaitForSingleObject(g_video_wake_ev, 45);
                continue;
            }
        }

        SOCKET s = connect_tcp_timeout(ip, port, 2000);
        if (s == INVALID_SOCKET) {
            stream_fails++;
            WaitForSingleObject(g_video_wake_ev, 250);
            continue;
        }
        DWORD rto = 3000;
        setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&rto, sizeof(rto));
        InterlockedExchangePointer((PVOID *)&g_video_sock, (PVOID)s);

        char req[256];
        int rlen = _snprintf(req, sizeof(req) - 1,
            "GET /stream?code=%s HTTP/1.1\r\nHost: %s:%d\r\nConnection: close\r\n\r\n", code, ip, port);
        if (send(s, req, rlen, 0) != rlen) {
            stop_active_video_socket();
            stream_fails++;
            continue;
        }

        SockReader sr;
        memset(&sr, 0, sizeof(sr));
        sr.s = s;
        char line[256];
        int is_chunked = 0;
        while (sr_read_raw_line(&sr, line, sizeof(line)) > 0) {
            if (_strnicmp(line, "Transfer-Encoding:", 18) == 0 && strstr(line, "chunked"))
                is_chunked = 1;
        }
        sr.chunked = is_chunked;

        /* Tek kalıcı TCP soketi üzerinden sıfır-kopma MJPEG akışı (hem ham hem HTTP chunked destekli!) */
        while (InterlockedCompareExchange(&g_running, 0, 0)
               && InterlockedCompareExchange(&g_connected, 0, 0)
               && InterlockedCompareExchange(&g_window_active, 0, 0)) {
            int clen = -1;
            int got_hdr = 0;
            while (1) {
                int l = sr_read_line(&sr, line, sizeof(line));
                if (l < 0) { got_hdr = -1; break; }
                if (l == 0) {
                    if (clen > 0) { got_hdr = 1; break; }
                    continue;
                }
                if (_strnicmp(line, "Content-Length:", 15) == 0) {
                    clen = atoi(line + 15);
                }
            }
            if (got_hdr <= 0 || clen <= 16 || clen > 4 * 1024 * 1024) {
                stream_fails++;
                break;
            }
            if (clen > jpeg_cap) {
                BYTE *nb = (BYTE *)realloc(jpeg_buf, (size_t)clen);
                if (!nb) break;
                jpeg_buf = nb;
                jpeg_cap = clen;
            }
            if (!sr_read_exact(&sr, jpeg_buf, clen)) {
                stream_fails++;
                break;
            }
            if (decode_jpeg_wic(wic, jpeg_buf, (DWORD)clen)) {
                stream_fails = 0;
                if (g_hwnd && InterlockedCompareExchange(&g_window_active, 0, 0))
                    PostMessageW(g_hwnd, WM_APP_NEW_FRAME, 0, 0);
            }
        }
        stop_active_video_socket();
    }

    free(jpeg_buf);
    if (wic) IWICImagingFactory_Release(wic);
    CoUninitialize();
    return 0;
}

/* ==================== GDI ÇİZİM (KÖŞE BRAKETLERİ + TÜRKÇE ETİKETLER) ==================== */
static void draw_corner_brackets(HDC dc, int l, int t, int r, int b, int c_len, COLORREF color, int thickness) {
    HPEN pen = CreatePen(PS_SOLID, thickness, color);
    HGDIOBJ old = SelectObject(dc, pen);
    /* Sol üst */
    MoveToEx(dc, l, t + c_len, NULL); LineTo(dc, l, t); LineTo(dc, l + c_len, t);
    /* Sağ üst */
    MoveToEx(dc, r - c_len, t, NULL); LineTo(dc, r, t); LineTo(dc, r, t + c_len);
    /* Sağ alt */
    MoveToEx(dc, r, b - c_len, NULL); LineTo(dc, r, b); LineTo(dc, r - c_len, b);
    /* Sol alt */
    MoveToEx(dc, l + c_len, b, NULL); LineTo(dc, l, b); LineTo(dc, l, b - c_len);
    SelectObject(dc, old);
    DeleteObject(pen);
}

static void draw_tag(HDC dc, int x, int y, const WCHAR *text, COLORREF bg, COLORREF fg, int max_right) {
    HGDIOBJ old_f = SelectObject(dc, g_font_tag);
    SIZE sz;
    int tlen = (int)wcslen(text);
    GetTextExtentPoint32W(dc, text, tlen, &sz);
    int pad_x = 8, pad_y = 3;
    int w = sz.cx + pad_x * 2;
    int h = sz.cy + pad_y * 2;
    if (x + w > max_right - 6) x = max_right - w - 6;
    if (x < 6) x = 6;
    if (y < 6) y = 6;

    RECT rc = {x, y, x + w, y + h};
    HBRUSH br = CreateSolidBrush(bg);
    FillRect(dc, &rc, br);
    DeleteObject(br);

    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, fg);
    TextOutW(dc, x + pad_x, y + pad_y, text, tlen);
    SelectObject(dc, old_f);
}

static void paint_video_canvas(HWND hwnd, HDC hdc, RECT canvas_rc) {
    int cw = canvas_rc.right - canvas_rc.left;
    int ch = canvas_rc.bottom - canvas_rc.top;
    if (cw <= 0 || ch <= 0) return;

    if (!g_mem_dc || g_mem_w != cw || g_mem_h != ch) {
        if (g_mem_dc) {
            SelectObject(g_mem_dc, g_mem_old);
            DeleteObject(g_mem_bmp);
            DeleteDC(g_mem_dc);
        }
        g_mem_dc = CreateCompatibleDC(hdc);
        g_mem_bmp = CreateCompatibleBitmap(hdc, cw, ch);
        g_mem_old = (HBITMAP)SelectObject(g_mem_dc, g_mem_bmp);
        g_mem_w = cw;
        g_mem_h = ch;
    }

    RECT local_rc = {0, 0, cw, ch};
    HBRUSH bg = CreateSolidBrush(RGB(10, 10, 14));
    FillRect(g_mem_dc, &local_rc, bg);
    DeleteObject(bg);

    int connected = (int)InterlockedCompareExchange(&g_connected, 0, 0);
    int active = (int)InterlockedCompareExchange(&g_window_active, 0, 0);

    AiState st;
    EnterCriticalSection(&g_state_cs);
    st = g_ai_state;
    LeaveCriticalSection(&g_state_cs);

    int drew_video = 0;
    int vx = 0, vy = 0, vw = cw, vh = ch;

    EnterCriticalSection(&g_frame_cs);
    if (connected && g_frame.pixels && g_frame.w > 0 && g_frame.h > 0) {
        double va = (double)g_frame.w / (double)g_frame.h;
        double ca = (double)cw / (double)ch;
        if (ca > va) {
            vh = ch;
            vw = (int)(ch * va);
        } else {
            vw = cw;
            vh = (int)(cw / va);
        }
        vx = (cw - vw) / 2;
        vy = (ch - vh) / 2;

        BITMAPINFO bmi;
        memset(&bmi, 0, sizeof(bmi));
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = g_frame.w;
        bmi.bmiHeader.biHeight = -g_frame.h; /* top-down */
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;

        SetStretchBltMode(g_mem_dc, COLORONCOLOR);
        StretchDIBits(g_mem_dc, vx, vy, vw, vh, 0, 0, g_frame.w, g_frame.h,
                      g_frame.pixels, &bmi, DIB_RGB_COLORS, SRCCOPY);
        drew_video = 1;
    }
    LeaveCriticalSection(&g_frame_cs);

    g_video_rect.left = canvas_rc.left + vx;
    g_video_rect.top = canvas_rc.top + vy;
    g_video_rect.right = g_video_rect.left + vw;
    g_video_rect.bottom = g_video_rect.top + vh;

    if (!drew_video) {
        HGDIOBJ old_f = SelectObject(g_mem_dc, g_font_bold);
        SetBkMode(g_mem_dc, TRANSPARENT);
        SetTextColor(g_mem_dc, RGB(155, 160, 175));
        WCHAR wmsg[192];
        if (!connected) {
            char cmsg[160];
            EnterCriticalSection(&g_conn_cs);
            strncpy(cmsg, g_conn_status_utf8, sizeof(cmsg) - 1); cmsg[sizeof(cmsg) - 1] = 0;
            LeaveCriticalSection(&g_conn_cs);
            utf8_to_wide(cmsg, wmsg, 192);
        } else if (!active) {
            lstrcpynW(wmsg, L"☾ Uyku Modu Aktif (0% GPU) — Alt+Tab yapınca canlı görüntü başlar", 192);
        } else {
            lstrcpynW(wmsg, L"Canlı görüntü bekleniyor...", 192);
        }
        DrawTextW(g_mem_dc, wmsg, -1, &local_rc, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        SelectObject(g_mem_dc, old_f);
    } else {
        /* AI Kişi / Kilitli Hedef Vizör Çizimi */
        if (st.locked.valid) {
            int l = vx + (int)(st.locked.x * vw);
            int t = vy + (int)(st.locked.y * vh);
            int r = l + (int)(st.locked.w * vw);
            int b = t + (int)(st.locked.h * vh);
            int c = (int)(fmin((double)(r - l), (double)(b - t)) * 0.24);
            if (c < 14) c = 14; if (c > 44) c = 44;
            COLORREF col = st.locked.lost ? RGB(255, 69, 58) : RGB(48, 209, 88);
            draw_corner_brackets(g_mem_dc, l, t, r, b, c, col, 3);

            /* Merkez nişangah */
            int cx = (l + r) / 2, cy = (t + b) / 2;
            HPEN cp = CreatePen(PS_SOLID, 2, col);
            HGDIOBJ op = SelectObject(g_mem_dc, cp);
            MoveToEx(g_mem_dc, cx - 10, cy, NULL); LineTo(g_mem_dc, cx + 10, cy);
            MoveToEx(g_mem_dc, cx, cy - 10, NULL); LineTo(g_mem_dc, cx, cy + 10);
            SelectObject(g_mem_dc, op); DeleteObject(cp);

            WCHAR wname[64], wlbl[128];
            if (st.locked.name[0]) utf8_to_wide(st.locked.name, wname, 64);
            else _snwprintf(wname, 63, L"Kişi #%d", st.locked.id);

            const WCHAR *badge = st.locked.has_face ? L" • Yüz+AI" : (st.locked.has_deep ? L" • AI" : L"");
            if (st.locked.lost) {
                _snwprintf(wlbl, 127, L"⚠ KAYIP: %s!", wname);
                draw_tag(g_mem_dc, l, t - 26, wlbl, RGB(215, 38, 30), RGB(255, 255, 255), cw);
            } else {
                _snwprintf(wlbl, 127, L"KİLİTLİ • %s • %%%d%s", wname, (int)(st.locked.s * 100.0f + 0.5f), badge);
                draw_tag(g_mem_dc, l, t - 26, wlbl, RGB(20, 22, 28), RGB(48, 209, 88), cw);
            }
        } else {
            for (int i = 0; i < st.person_count; i++) {
                const PersonBox *p = &st.persons[i];
                int l = vx + (int)(p->x * vw);
                int t = vy + (int)(p->y * vh);
                int r = l + (int)(p->w * vw);
                int b = t + (int)(p->h * vh);
                int c = (int)(fmin((double)(r - l), (double)(b - t)) * 0.20);
                if (c < 12) c = 12; if (c > 36) c = 36;
                draw_corner_brackets(g_mem_dc, l, t, r, b, c, RGB(225, 230, 240), 2);

                WCHAR wname[64], wlbl[128];
                if (p->name[0]) utf8_to_wide(p->name, wname, 64);
                else _snwprintf(wname, 63, L"İNSAN #%d", p->id);
                const WCHAR *badge = p->has_face ? L" • Yüz" : (p->has_deep ? L" • AI" : L"");
                _snwprintf(wlbl, 127, L"%s • %%%d%s", wname, (int)(p->s * 100.0f + 0.5f), badge);
                draw_tag(g_mem_dc, l, t - 24, wlbl, RGB(22, 24, 30), RGB(255, 255, 255), cw);
            }
        }

        /* Tıklama odak halkası */
        DWORD now = GetTickCount();
        if (g_ring_start && now - g_ring_start < 420) {
            float prog = (float)(now - g_ring_start) / 420.0f;
            int rad = (int)(30.0f - 14.0f * prog);
            int rx = g_ring_x - canvas_rc.left;
            int ry = g_ring_y - canvas_rc.top;
            HPEN rp = CreatePen(PS_SOLID, 2, g_ring_warn ? RGB(255, 214, 10) : RGB(255, 255, 255));
            HGDIOBJ op = SelectObject(g_mem_dc, rp);
            HGDIOBJ ob = SelectObject(g_mem_dc, GetStockObject(NULL_BRUSH));
            Ellipse(g_mem_dc, rx - rad, ry - rad, rx + rad, ry + rad);
            SelectObject(g_mem_dc, ob);
            SelectObject(g_mem_dc, op);
            DeleteObject(rp);
        }
    }

    /* Kırmızı Kayıp Alarm Üst Bandı */
    if (connected && st.locked.valid && (st.locked.lost || st.alarm)) {
        int beeping = (int)InterlockedCompareExchange(&g_alarm_beeping, 0, 0);
        const WCHAR *hk = InterlockedCompareExchange(&g_hotkey_ctrl_shift, 0, 0) ? L"Ctrl+Shift+F9" : L"F9";
        WCHAR banner[160];
        if (beeping)
            _snwprintf(banner, 159, L"⚠ HEDEF KAYIP! Bip-Bip Alarm Aktif — Susturmak için %s", hk);
        else
            _snwprintf(banner, 159, L"⚠ HEDEF KAYIP! (Alarm susturuldu — Hedef bekleniyor)");
        RECT brc = {0, 0, cw, 32};
        HBRUSH rbr = CreateSolidBrush(beeping ? RGB(205, 32, 28) : RGB(110, 42, 20));
        FillRect(g_mem_dc, &brc, rbr);
        DeleteObject(rbr);
        HGDIOBJ of = SelectObject(g_mem_dc, g_font_bold);
        SetBkMode(g_mem_dc, TRANSPARENT);
        SetTextColor(g_mem_dc, RGB(255, 255, 255));
        DrawTextW(g_mem_dc, banner, -1, &brc, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        SelectObject(g_mem_dc, of);
    }

    /* Kısa bilgi uyarısı */
    if (g_toast_utf8[0] && GetTickCount() < g_toast_until) {
        WCHAR wt[128];
        utf8_to_wide(g_toast_utf8, wt, 128);
        draw_tag(g_mem_dc, cw / 2 - 110, ch - 38, wt, RGB(35, 38, 48), RGB(255, 214, 10), cw);
    }

    BitBlt(hdc, canvas_rc.left, canvas_rc.top, cw, ch, g_mem_dc, 0, 0, SRCCOPY);
}

/* ==================== UI DURUM GÜNCELLEME & KISAYOL ==================== */
static void trigger_mute_alarm(void) {
    InterlockedExchange(&g_muted_current_alarm, 1);
    InterlockedExchange(&g_alarm_beeping, 0);
    SetEvent(g_audio_wake_ev);
    send_remote_cmd_async("action=dismiss_alarm", 0);
    strncpy(g_toast_utf8, "Alarm susturuldu", sizeof(g_toast_utf8) - 1);
    g_toast_until = GetTickCount() + 1400;
    if (g_hwnd) {
        PostMessageW(g_hwnd, WM_APP_STATE_UPD, 0, 0);
    }
}

static void update_status_bar_text(void) {
    if (!g_lbl_status) return;
    int conn = (int)InterlockedCompareExchange(&g_connected, 0, 0);
    int active = (int)InterlockedCompareExchange(&g_window_active, 0, 0);
    char cmsg[160];
    EnterCriticalSection(&g_conn_cs);
    strncpy(cmsg, g_conn_status_utf8, sizeof(cmsg) - 1); cmsg[sizeof(cmsg) - 1] = 0;
    LeaveCriticalSection(&g_conn_cs);

    AiState st;
    EnterCriticalSection(&g_state_cs);
    st = g_ai_state;
    LeaveCriticalSection(&g_state_cs);

    WCHAR wconn[160], wout[256];
    utf8_to_wide(cmsg, wconn, 160);
    if (!conn) {
        _snwprintf(wout, 255, L"● %s", wconn);
    } else if (st.locked.valid) {
        WCHAR wname[64];
        if (st.locked.name[0]) utf8_to_wide(st.locked.name, wname, 64);
        else _snwprintf(wname, 63, L"Kişi #%d", st.locked.id);
        _snwprintf(wout, 255, L"%s %s  |  %s: %s",
            active ? L"●" : L"☾", wconn,
            st.locked.lost ? L"KAYIP" : L"Kilitli", wname);
    } else {
        _snwprintf(wout, 255, L"%s %s  |  %d Kişi (Kilitlemek için tıkla)",
            active ? L"●" : L"☾", wconn, st.person_count);
    }
    SetWindowTextW(g_lbl_status, wout);
    EnableWindow(g_btn_unlock, conn && st.locked.valid);
    EnableWindow(g_btn_mute, conn && (InterlockedCompareExchange(&g_alarm_beeping, 0, 0) || (st.locked.valid && st.locked.lost)));
}

static void layout_controls(HWND hwnd) {
    RECT rc;
    GetClientRect(hwnd, &rc);
    int w = rc.right - rc.left;
    int h = rc.bottom - rc.top;
    if (w < 300 || h < 200) return;

    /* Üst bar kontrolleri */
    int y = 8, bh = 26;
    MoveWindow(g_edit_code,    50,  y, 64,  bh, TRUE);
    MoveWindow(g_edit_ip,     144,  y, 130, bh, TRUE);
    MoveWindow(g_btn_connect, 282,  y, 72,  bh, TRUE);
    MoveWindow(g_btn_unlock,  362,  y, 80,  bh, TRUE);
    MoveWindow(g_btn_mute,    450,  y, 118, bh, TRUE);

    /* Alt bar kontrolleri */
    int by = h - BOT_BAR_H + 7;
    int right_w = 248;
    int status_w = w - right_w - 18;
    if (status_w < 100) status_w = 100;
    MoveWindow(g_lbl_status,  10, by + 3, status_w, 22, TRUE);
    int rx = w - right_w;
    MoveWindow(g_slider_vol,  rx + 28,  by,     106, 26, TRUE);
    MoveWindow(g_lbl_vol,     rx + 136, by + 3, 38,  22, TRUE);
    MoveWindow(g_btn_test,    rx + 176, by,     64,  26, TRUE);
}

static void release_sleep_buffers(void) {
    EnterCriticalSection(&g_frame_cs);
    if (g_frame.pixels) {
        free(g_frame.pixels);
        g_frame.pixels = NULL;
        g_frame.cap_pixels = 0;
        g_frame.w = 0;
        g_frame.h = 0;
    }
    LeaveCriticalSection(&g_frame_cs);
}

static void set_window_active_state(HWND hwnd, int is_active) {
    LONG prev = InterlockedExchange(&g_window_active, is_active ? 1 : 0);
    if (prev != (is_active ? 1 : 0)) {
        apply_gaming_eco_qos(!is_active);
        if (is_active) {
            stop_active_state_socket(); /* Uzun-beklemeden hemen çık -> Alt+Tab'da anlık durum yenile! */
            SetEvent(g_video_wake_ev);
            SetEvent(g_state_wake_ev);
            if (GetCurrentThreadId() == GetWindowThreadProcessId(hwnd, NULL)) {
                update_status_bar_text();
                InvalidateRect(hwnd, NULL, FALSE);
            } else if (hwnd) {
                PostMessageW(hwnd, WM_APP_CONN_UPD, 0, 0);
            }
        } else {
            /* Oyuna / arka plana geçildi -> video soketini kapat, zamanlayıcı & tamponları temizle, EcoQoS + RAM buda! */
            stop_active_video_socket();
            release_sleep_buffers();
            if (GetCurrentThreadId() == GetWindowThreadProcessId(hwnd, NULL)) {
                g_ring_start = 0;
                KillTimer(hwnd, TIMER_PULSE_ID);
                if (g_mem_dc) {
                    SelectObject(g_mem_dc, g_mem_old);
                    DeleteObject(g_mem_bmp);
                    DeleteDC(g_mem_dc);
                    g_mem_dc = NULL; g_mem_bmp = NULL; g_mem_old = NULL;
                    g_mem_w = 0; g_mem_h = 0;
                }
                SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
            } else if (hwnd) {
                PostMessageW(hwnd, WM_APP_CONN_UPD, 0, 0);
            }
        }
    }
}

/* ==================== ANA PENCERE İŞLEYİCİSİ ==================== */
static LRESULT CALLBACK MainWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_CREATE: {
        BOOL dark = TRUE;
        DwmSetWindowAttribute(hwnd, 20 /* DWMWA_USE_IMMERSIVE_DARK_MODE */, &dark, sizeof(dark));

        g_font_ui   = CreateFontW(-13, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
        g_font_bold = CreateFontW(-13, 0, 0, 0, FW_SEMIBOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
        g_font_tag  = CreateFontW(-13, 0, 0, 0, FW_BOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
        g_br_bar    = CreateSolidBrush(RGB(20, 21, 27));
        g_br_edit   = CreateSolidBrush(RGB(32, 34, 44));

        WCHAR wcode[16], wip[64];
        utf8_to_wide(g_cfg.code, wcode, 16);
        utf8_to_wide(g_cfg.manual_ip, wip, 64);

        g_edit_code = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", wcode,
            WS_CHILD | WS_VISIBLE | ES_CENTER | ES_NUMBER, 0, 0, 60, 24, hwnd, (HMENU)IDC_EDIT_CODE, g_hinst, NULL);
        SendMessageW(g_edit_code, EM_SETLIMITTEXT, 5, 0);

        g_edit_ip = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", wip,
            WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL, 0, 0, 120, 24, hwnd, (HMENU)IDC_EDIT_IP, g_hinst, NULL);
        SendMessageW(g_edit_ip, EM_SETCUEBANNER, TRUE, (LPARAM)L"Otomatik IP");

        g_btn_connect = CreateWindowExW(0, L"BUTTON", L"Bağlan",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 0, 0, 70, 24, hwnd, (HMENU)IDC_BTN_CONNECT, g_hinst, NULL);
        g_btn_unlock = CreateWindowExW(0, L"BUTTON", L"Kilidi Aç",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON | WS_DISABLED, 0, 0, 76, 24, hwnd, (HMENU)IDC_BTN_UNLOCK, g_hinst, NULL);

        /* Global F9 kısayolu (tam ekran oyundayken bile anında susturur) */
        const WCHAR *mute_lbl = L"Sustur (F9)";
        if (!RegisterHotKey(hwnd, HOTKEY_MUTE_ID, MOD_NOREPEAT, VK_F9)) {
            if (RegisterHotKey(hwnd, HOTKEY_MUTE_ID, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, VK_F9)) {
                InterlockedExchange(&g_hotkey_ctrl_shift, 1);
                mute_lbl = L"Sustur (C+S+F9)";
            }
        }
        g_btn_mute = CreateWindowExW(0, L"BUTTON", mute_lbl,
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON | WS_DISABLED, 0, 0, 110, 24, hwnd, (HMENU)IDC_BTN_MUTE, g_hinst, NULL);

        g_lbl_status = CreateWindowExW(0, L"STATIC", L"● Kamera aranıyor...",
            WS_CHILD | WS_VISIBLE | SS_LEFT | SS_ENDELLIPSIS, 0, 0, 300, 22, hwnd, (HMENU)IDC_LBL_STATUS, g_hinst, NULL);

        g_slider_vol = CreateWindowExW(0, TRACKBAR_CLASSW, L"",
            WS_CHILD | WS_VISIBLE | TBS_HORZ | TBS_NOTICKS, 0, 0, 100, 24, hwnd, (HMENU)IDC_SLIDER_VOL, g_hinst, NULL);
        SendMessageW(g_slider_vol, TBM_SETRANGE, TRUE, MAKELPARAM(0, 100));
        SendMessageW(g_slider_vol, TBM_SETPOS, TRUE, g_cfg.volume);

        WCHAR wvol[16];
        _snwprintf(wvol, 15, L"%%%d", g_cfg.volume);
        g_lbl_vol = CreateWindowExW(0, L"STATIC", wvol,
            WS_CHILD | WS_VISIBLE | SS_LEFT, 0, 0, 36, 22, hwnd, (HMENU)IDC_LBL_VOL, g_hinst, NULL);

        g_btn_test = CreateWindowExW(0, L"BUTTON", L"Bip Test",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 0, 0, 62, 24, hwnd, (HMENU)IDC_BTN_BEEP_TEST, g_hinst, NULL);

        HWND ctrls[] = {g_edit_code, g_edit_ip, g_btn_connect, g_btn_unlock, g_btn_mute, g_lbl_status, g_lbl_vol, g_btn_test};
        for (size_t i = 0; i < sizeof(ctrls) / sizeof(ctrls[0]); i++)
            SendMessageW(ctrls[i], WM_SETFONT, (WPARAM)g_font_ui, TRUE);

        layout_controls(hwnd);
        return 0;
    }

    case WM_SIZE: {
        if (wParam == SIZE_MINIMIZED) {
            set_window_active_state(hwnd, 0);
        } else {
            RECT wr;
            GetWindowRect(hwnd, &wr);
            g_cfg.win_w = wr.right - wr.left;
            g_cfg.win_h = wr.bottom - wr.top;
            layout_controls(hwnd);
            if (GetForegroundWindow() == hwnd) set_window_active_state(hwnd, 1);
            InvalidateRect(hwnd, NULL, FALSE);
        }
        return 0;
    }

    case WM_ACTIVATE: {
        int active = (LOWORD(wParam) != WA_INACTIVE) && !IsIconic(hwnd);
        set_window_active_state(hwnd, active);
        return 0;
    }

    case WM_ACTIVATEAPP: {
        int active = (wParam != FALSE) && !IsIconic(hwnd);
        set_window_active_state(hwnd, active);
        return 0;
    }

    case WM_HOTKEY: {
        if (wParam == HOTKEY_MUTE_ID) {
            trigger_mute_alarm();
        }
        return 0;
    }

    case WM_HSCROLL: {
        if ((HWND)lParam == g_slider_vol) {
            int vol = (int)SendMessageW(g_slider_vol, TBM_GETPOS, 0, 0);
            if (vol < 0) vol = 0; if (vol > 100) vol = 100;
            InterlockedExchange(&g_volume, vol);
            g_cfg.volume = vol;
            WCHAR wvol[16];
            _snwprintf(wvol, 15, L"%%%d", vol);
            SetWindowTextW(g_lbl_vol, wvol);
            if (LOWORD(wParam) == TB_ENDTRACK) save_config();
        }
        return 0;
    }

    case WM_COMMAND: {
        int id = LOWORD(wParam);
        if (id == IDC_BTN_CONNECT) {
            WCHAR wc[16], wi[64];
            GetWindowTextW(g_edit_code, wc, 16);
            GetWindowTextW(g_edit_ip, wi, 64);
            char code[16], ip[64];
            wide_to_utf8(wc, code, sizeof(code));
            wide_to_utf8(wi, ip, sizeof(ip));
            if (strlen(code) != 5) {
                strncpy(g_toast_utf8, "Lütfen 5 haneli şifre girin!", sizeof(g_toast_utf8) - 1);
                g_toast_until = GetTickCount() + 2000;
                InvalidateRect(hwnd, NULL, FALSE);
                return 0;
            }
            EnterCriticalSection(&g_conn_cs);
            strncpy(g_cfg.code, code, sizeof(g_cfg.code) - 1);
            strncpy(g_active_code, code, sizeof(g_active_code) - 1);
            strncpy(g_cfg.manual_ip, ip, sizeof(g_cfg.manual_ip) - 1);
            LeaveCriticalSection(&g_conn_cs);
            save_config();
            InterlockedExchange(&g_reconnect_req, 1);
            SetEvent(g_state_wake_ev);
            stop_active_video_socket();
            stop_active_state_socket();
        } else if (id == IDC_BTN_UNLOCK) {
            send_remote_cmd_async("action=unlock", 0);
            InterlockedExchange(&g_alarm_beeping, 0);
            InterlockedExchange(&g_muted_current_alarm, 0);
            SetEvent(g_audio_wake_ev);
        } else if (id == IDC_BTN_MUTE) {
            trigger_mute_alarm();
        } else if (id == IDC_BTN_BEEP_TEST) {
            InterlockedExchange(&g_test_beep_req, 1);
            SetEvent(g_audio_wake_ev);
        }
        return 0;
    }

    case WM_LBUTTONDOWN: {
        int mx = GET_X_LPARAM(lParam);
        int my = GET_Y_LPARAM(lParam);
        if (InterlockedCompareExchange(&g_connected, 0, 0)
            && mx >= g_video_rect.left && mx <= g_video_rect.right
            && my >= g_video_rect.top && my <= g_video_rect.bottom
            && (g_video_rect.right - g_video_rect.left) > 10
            && (g_video_rect.bottom - g_video_rect.top) > 10) {
            float nx = (float)(mx - g_video_rect.left) / (float)(g_video_rect.right - g_video_rect.left);
            float ny = (float)(my - g_video_rect.top) / (float)(g_video_rect.bottom - g_video_rect.top);
            if (nx < 0.0f) nx = 0.0f; if (nx > 1.0f) nx = 1.0f;
            if (ny < 0.0f) ny = 0.0f; if (ny > 1.0f) ny = 1.0f;

            g_ring_x = mx; g_ring_y = my; g_ring_warn = 0; g_ring_start = GetTickCount();
            SetTimer(hwnd, TIMER_PULSE_ID, 33, NULL);
            char cmd[96];
            _snprintf(cmd, sizeof(cmd) - 1, "action=lock&nx=%.4f&ny=%.4f", nx, ny);
            send_remote_cmd_async(cmd, 1);
            InvalidateRect(hwnd, NULL, FALSE);
        }
        return 0;
    }

    case WM_TIMER: {
        if (wParam == TIMER_PULSE_ID) {
            if (!g_ring_start || GetTickCount() - g_ring_start >= 440) {
                g_ring_start = 0;
                KillTimer(hwnd, TIMER_PULSE_ID);
            }
            if (InterlockedCompareExchange(&g_window_active, 0, 0)) {
                RECT cr; GetClientRect(hwnd, &cr);
                cr.top = TOP_BAR_H; cr.bottom -= BOT_BAR_H;
                InvalidateRect(hwnd, &cr, FALSE);
            }
        }
        return 0;
    }

    case WM_APP_NEW_FRAME: {
        if (InterlockedCompareExchange(&g_window_active, 0, 0)) {
            RECT cr; GetClientRect(hwnd, &cr);
            cr.top = TOP_BAR_H; cr.bottom -= BOT_BAR_H;
            InvalidateRect(hwnd, &cr, FALSE);
        }
        return 0;
    }

    case WM_APP_STATE_UPD:
    case WM_APP_CONN_UPD: {
        if (InterlockedCompareExchange(&g_window_active, 0, 0)) {
            update_status_bar_text();
            RECT cr; GetClientRect(hwnd, &cr);
            cr.top = TOP_BAR_H; cr.bottom -= BOT_BAR_H;
            InvalidateRect(hwnd, &cr, FALSE);
        } else if (g_mem_dc) {
            SelectObject(g_mem_dc, g_mem_old);
            DeleteObject(g_mem_bmp);
            DeleteDC(g_mem_dc);
            g_mem_dc = NULL; g_mem_bmp = NULL; g_mem_old = NULL;
            g_mem_w = 0; g_mem_h = 0;
            SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
        }
        return 0;
    }

    case WM_APP_LOCK_RES: {
        if (wParam) {
            strncpy(g_toast_utf8, "Kişi kilitlendi!", sizeof(g_toast_utf8) - 1);
            g_ring_warn = 0;
        } else {
            strncpy(g_toast_utf8, "Kişi seçilmedi — Bir insana tıklayın", sizeof(g_toast_utf8) - 1);
            g_ring_warn = 1;
        }
        g_toast_until = GetTickCount() + 1800;
        InvalidateRect(hwnd, NULL, FALSE);
        return 0;
    }

    case WM_CTLCOLORSTATIC: {
        HDC dc = (HDC)wParam;
        SetTextColor(dc, RGB(220, 225, 235));
        SetBkColor(dc, RGB(20, 21, 27));
        return (LRESULT)g_br_bar;
    }

    case WM_CTLCOLOREDIT: {
        HDC dc = (HDC)wParam;
        SetTextColor(dc, RGB(255, 255, 255));
        SetBkColor(dc, RGB(32, 34, 44));
        return (LRESULT)g_br_edit;
    }

    case WM_ERASEBKGND:
        return 1;

    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        RECT rc; GetClientRect(hwnd, &rc);

        /* Üst ve Alt Bar koyu arka plan + etiketler */
        RECT top_rc = {0, 0, rc.right, TOP_BAR_H};
        RECT bot_rc = {0, rc.bottom - BOT_BAR_H, rc.right, rc.bottom};
        FillRect(hdc, &top_rc, g_br_bar);
        FillRect(hdc, &bot_rc, g_br_bar);

        HGDIOBJ old_f = SelectObject(hdc, g_font_ui);
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, RGB(175, 180, 195));
        TextOutW(hdc, 10, 12, L"Şifre:", 6);
        TextOutW(hdc, 122, 12, L"IP:", 3);
        int rx = rc.right - 248;
        if (rx > 120) TextOutW(hdc, rx, rc.bottom - BOT_BAR_H + 11, L"Ses:", 4);
        SelectObject(hdc, old_f);

        /* Orta Video Tuvali */
        RECT canvas_rc = {0, TOP_BAR_H, rc.right, rc.bottom - BOT_BAR_H};
        if (canvas_rc.bottom > canvas_rc.top) {
            paint_video_canvas(hwnd, hdc, canvas_rc);
        }
        EndPaint(hwnd, &ps);
        return 0;
    }

    case WM_GETMINMAXINFO: {
        MINMAXINFO *mmi = (MINMAXINFO *)lParam;
        mmi->ptMinTrackSize.x = 600;
        mmi->ptMinTrackSize.y = 400;
        return 0;
    }

    case WM_DESTROY: {
        UnregisterHotKey(hwnd, HOTKEY_MUTE_ID);
        save_config();
        InterlockedExchange(&g_running, 0);
        InterlockedExchange(&g_alarm_beeping, 0);
        stop_active_video_socket();
        stop_active_state_socket();
        if (g_video_wake_ev) SetEvent(g_video_wake_ev);
        if (g_audio_wake_ev) SetEvent(g_audio_wake_ev);
        if (g_state_wake_ev) SetEvent(g_state_wake_ev);
        PostQuitMessage(0);
        return 0;
    }
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

/* ==================== GİRİŞ NOKTASI ==================== */
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrev, LPSTR lpCmdLine, int nCmdShow) {
    (void)hPrev; (void)lpCmdLine;
    g_hinst = hInstance;

    /* Per-Monitor DPI farkındalığı (1080p/2K/4K keskin görüntü) */
    HMODULE u32 = GetModuleHandleW(L"user32.dll");
    if (u32) {
        typedef BOOL(WINAPI *SetDpiCtxFn)(DPI_AWARENESS_CONTEXT);
        SetDpiCtxFn fn = (SetDpiCtxFn)GetProcAddress(u32, "SetProcessDpiAwarenessContext");
        if (fn) fn(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }

    INITCOMMONCONTROLSEX icc = {sizeof(icc), ICC_BAR_CLASSES};
    InitCommonControlsEx(&icc);

    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return 1;

    InitializeCriticalSection(&g_state_cs);
    InitializeCriticalSection(&g_frame_cs);
    InitializeCriticalSection(&g_conn_cs);

    load_config();

    g_video_wake_ev = CreateEventW(NULL, FALSE, FALSE, NULL);
    g_audio_wake_ev = CreateEventW(NULL, FALSE, FALSE, NULL);
    g_state_wake_ev = CreateEventW(NULL, FALSE, FALSE, NULL);

    WNDCLASSEXW wc;
    memset(&wc, 0, sizeof(wc));
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = MainWndProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursorW(NULL, IDC_ARROW);
    wc.hIcon = LoadIconW(NULL, IDI_APPLICATION);
    wc.lpszClassName = WND_CLASS_W;
    RegisterClassExW(&wc);

    g_hwnd = CreateWindowExW(0, WND_CLASS_W, APP_TITLE_W,
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
        CW_USEDEFAULT, CW_USEDEFAULT, g_cfg.win_w, g_cfg.win_h,
        NULL, NULL, hInstance, NULL);
    if (!g_hwnd) return 1;

    ShowWindow(g_hwnd, nCmdShow);
    UpdateWindow(g_hwnd);

    g_th_audio = CreateThread(NULL, 0, audio_thread_proc, NULL, 0, NULL);
    g_th_state = CreateThread(NULL, 0, state_thread_proc, NULL, 0, NULL);
    g_th_video = CreateThread(NULL, 0, video_thread_proc, NULL, 0, NULL);

    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0) > 0) {
        if (msg.message == WM_KEYDOWN && msg.wParam == VK_RETURN
            && (msg.hwnd == g_edit_code || msg.hwnd == g_edit_ip)) {
            SendMessageW(g_hwnd, WM_COMMAND, IDC_BTN_CONNECT, 0);
            continue;
        }
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    /* Temiz kapatma */
    HANDLE ths[3] = {g_th_state, g_th_video, g_th_audio};
    for (int i = 0; i < 3; i++) {
        if (ths[i]) {
            WaitForSingleObject(ths[i], 1200);
            CloseHandle(ths[i]);
        }
    }

    if (g_mem_dc) {
        SelectObject(g_mem_dc, g_mem_old);
        DeleteObject(g_mem_bmp);
        DeleteDC(g_mem_dc);
    }
    if (g_frame.pixels) free(g_frame.pixels);
    if (g_font_ui) DeleteObject(g_font_ui);
    if (g_font_bold) DeleteObject(g_font_bold);
    if (g_font_tag) DeleteObject(g_font_tag);
    if (g_br_bar) DeleteObject(g_br_bar);
    if (g_br_edit) DeleteObject(g_br_edit);
    CloseHandle(g_video_wake_ev);
    CloseHandle(g_audio_wake_ev);
    CloseHandle(g_state_wake_ev);
    DeleteCriticalSection(&g_state_cs);
    DeleteCriticalSection(&g_frame_cs);
    DeleteCriticalSection(&g_conn_cs);
    WSACleanup();
    return 0;
}
