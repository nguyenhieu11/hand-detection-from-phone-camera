#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

// POSIX sockets
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

// ──────────────────────────────────────────────
// YOLOv8 detection parameters
// ──────────────────────────────────────────────
static constexpr float CONF_THRESHOLD = 0.6f;
static constexpr float NMS_THRESHOLD  = 0.45f;
static constexpr int   INPUT_W        = 320;
static constexpr int   INPUT_H        = 320;

struct Detection {
    cv::Rect box;
    float    confidence;
    int      classId;
};

// ──────────────────────────────────────────────
// Letterbox resize (keeps aspect ratio)
// ──────────────────────────────────────────────
cv::Mat letterbox(const cv::Mat& src, int targetW, int targetH,
                  float& scale, int& padLeft, int& padTop)
{
    float scaleX = static_cast<float>(targetW) / src.cols;
    float scaleY = static_cast<float>(targetH) / src.rows;
    scale = std::min(scaleX, scaleY);

    int newW = static_cast<int>(src.cols * scale);
    int newH = static_cast<int>(src.rows * scale);
    padLeft  = (targetW - newW) / 2;
    padTop   = (targetH - newH) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH));

    cv::Mat out(targetH, targetW, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(padLeft, padTop, newW, newH)));
    return out;
}

// ──────────────────────────────────────────────
// Pre-process: BGR → RGB, HWC → CHW, [0,255]→[0,1]  (fast path)
// ──────────────────────────────────────────────
std::vector<float> preprocess(const cv::Mat& img)
{
    // blobFromImage does cvtColor + normalize + HWC→CHW in one optimised pass
    cv::Mat blob4d = cv::dnn::blobFromImage(
        img, 1.0 / 255.0, cv::Size(INPUT_W, INPUT_H),
        cv::Scalar(0, 0, 0), /*swapRB=*/true, /*crop=*/false, CV_32F);
    return std::vector<float>(blob4d.ptr<float>(),
                              blob4d.ptr<float>() + blob4d.total());
}

// ──────────────────────────────────────────────
// Parse YOLOv8 output tensor  [1, num_classes+4, num_anchors]
// YOLOv8 output layout: (cx, cy, w, h, cls0, cls1, …)  — no objectness
// ──────────────────────────────────────────────
std::vector<Detection> parseOutput(const float* data,
                                   int numClasses, int numAnchors,
                                   float scale, int padLeft, int padTop,
                                   int origW, int origH)
{
    std::vector<cv::Rect>  boxes;
    std::vector<float>     scores;
    std::vector<int>       classIds;

    for (int i = 0; i < numAnchors; ++i) {
        // data is stored column-major in YOLOv8: row = attribute, col = anchor
        float cx = data[0 * numAnchors + i];
        float cy = data[1 * numAnchors + i];
        float bw = data[2 * numAnchors + i];
        float bh = data[3 * numAnchors + i];

        float maxConf  = -1.f;
        int   bestCls  = 0;
        for (int c = 0; c < numClasses; ++c) {
            float conf = data[(4 + c) * numAnchors + i];
            if (conf > maxConf) { maxConf = conf; bestCls = c; }
        }

        if (maxConf < CONF_THRESHOLD) continue;

        // Convert from letterboxed space back to original image space
        float x1 = (cx - bw / 2.f - padLeft) / scale;
        float y1 = (cy - bh / 2.f - padTop)  / scale;
        float x2 = (cx + bw / 2.f - padLeft) / scale;
        float y2 = (cy + bh / 2.f - padTop)  / scale;

        x1 = std::clamp(x1, 0.f, static_cast<float>(origW));
        y1 = std::clamp(y1, 0.f, static_cast<float>(origH));
        x2 = std::clamp(x2, 0.f, static_cast<float>(origW));
        y2 = std::clamp(y2, 0.f, static_cast<float>(origH));

        boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1),
                           static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
        scores.push_back(maxConf);
        classIds.push_back(bestCls);
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD, indices);

    std::vector<Detection> detections;
    detections.reserve(indices.size());
    for (int idx : indices)
        detections.push_back({boxes[idx], scores[idx], classIds[idx]});

    return detections;
}

// ──────────────────────────────────────────────
// MJPEG HTTP Server
// ──────────────────────────────────────────────
static std::vector<uint8_t>  g_jpegBuf;
static std::mutex            g_frameMutex;
static std::condition_variable g_frameCv;
static std::atomic<bool>     g_running{true};

// ──────────────────────────────────────────────
// Latest-frame grab buffer (decouples capture from inference)
// ──────────────────────────────────────────────
static cv::Mat       g_latestFrame;
static std::mutex    g_grabMutex;
static std::condition_variable g_grabCv;
static bool          g_frameReady = false;

// Continuously grabs frames, always keeping only the newest one.
// This drains the RTSP/decoder buffer so inference always gets a fresh frame.
void grabThread(cv::VideoCapture& cap)
{
    cv::Mat tmp;
    while (g_running) {
        if (!cap.read(tmp) || tmp.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(g_grabMutex);
            g_latestFrame = tmp;   // overwrite — old frame discarded
            g_frameReady  = true;
        }
        g_grabCv.notify_one();
    }
}

// Push a new encoded JPEG into the shared buffer and notify all clients.
void pushFrame(const cv::Mat& frame)
{
    std::vector<uint8_t> buf;
    cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, 80});
    {
        std::lock_guard<std::mutex> lk(g_frameMutex);
        g_jpegBuf = std::move(buf);
    }
    g_frameCv.notify_all();
}

// Handle one connected HTTP client — streams MJPEG until client disconnects.
void handleClient(int clientFd)
{
    // Send HTTP multipart header
    const char* header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";
    send(clientFd, header, strlen(header), 0);

    std::vector<uint8_t> localBuf;
    while (g_running) {
        // Wait for a new frame
        {
            std::unique_lock<std::mutex> lk(g_frameMutex);
            g_frameCv.wait_for(lk, std::chrono::milliseconds(500));
            if (g_jpegBuf.empty()) continue;
            localBuf = g_jpegBuf;   // copy under lock
        }

        // Build MJPEG part header
        char partHeader[128];
        int phLen = snprintf(partHeader, sizeof(partHeader),
            "--frame\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length: %zu\r\n"
            "\r\n", localBuf.size());

        if (send(clientFd, partHeader, phLen, MSG_NOSIGNAL) < 0) break;
        if (send(clientFd, localBuf.data(), localBuf.size(), MSG_NOSIGNAL) < 0) break;
        const char* crlf = "\r\n";
        if (send(clientFd, crlf, 2, MSG_NOSIGNAL) < 0) break;
    }
    close(clientFd);
}

// TCP accept loop — runs on its own thread.
void mjpegServer(int port)
{
    int serverFd = socket(AF_INET, SOCK_STREAM, 0);
    if (serverFd < 0) { std::cerr << "[MJPEG] socket() failed\n"; return; }

    int opt = 1;
    setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(static_cast<uint16_t>(port));

    if (bind(serverFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "[MJPEG] bind() failed on port " << port << "\n";
        close(serverFd);
        return;
    }
    listen(serverFd, 8);
    std::cout << "[MJPEG] Streaming on http://<this-machine-ip>:" << port << "\n";

    while (g_running) {
        sockaddr_in clientAddr{};
        socklen_t   clientLen = sizeof(clientAddr);
        int clientFd = accept(serverFd,
                              reinterpret_cast<sockaddr*>(&clientAddr),
                              &clientLen);
        if (clientFd < 0) continue;

        char ipStr[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &clientAddr.sin_addr, ipStr, sizeof(ipStr));
        std::cout << "[MJPEG] Client connected: " << ipStr << "\n";

        // Detach a thread per client (fine for small number of viewers)
        std::thread(handleClient, clientFd).detach();
    }
    close(serverFd);
}

// ──────────────────────────────────────────────
int main(int argc, char* argv[])
{
    const std::string modelPath  = (argc > 1) ? argv[1] : "best.onnx";
    const std::string streamSrc   = (argc > 2) ? argv[2] : "0";   // RTSP URL or "0" for webcam
    const int         httpPort    = (argc > 3) ? std::stoi(argv[3]) : 8080;

    // ── Init ONNX Runtime ──────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov8-hand");
    Ort::SessionOptions sessionOpts;
    sessionOpts.SetIntraOpNumThreads(4);
    sessionOpts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, modelPath.c_str(), sessionOpts);

    Ort::AllocatorWithDefaultOptions allocator;

    // Input info
    auto inputNamePtr = session.GetInputNameAllocated(0, allocator);
    std::string inputName(inputNamePtr.get());

    // Output info
    auto outputNamePtr = session.GetOutputNameAllocated(0, allocator);
    std::string outputName(outputNamePtr.get());

    auto outputShape = session.GetOutputTypeInfo(0)
                           .GetTensorTypeAndShapeInfo()
                           .GetShape();
    // outputShape: [1, num_classes+4, num_anchors]
    int numRows    = static_cast<int>(outputShape[1]); // 4 + num_classes
    int numAnchors = static_cast<int>(outputShape[2]);
    int numClasses = numRows - 4;

    std::cout << "Usage: " << argv[0] << " [model.onnx] [rtsp://... or 0] [http_port]\n\n"
              << "Model loaded : " << modelPath  << "\n"
              << "  Input      : " << inputName  << "\n"
              << "  Output     : " << outputName << "\n"
              << "  Classes    : " << numClasses << "\n"
              << "  Anchors    : " << numAnchors << "\n"
              << "Stream source: " << streamSrc  << "\n"
              << "HTTP port    : " << httpPort   << "\n\n";

    // ── Start MJPEG HTTP server ────────────────
    std::thread serverThread(mjpegServer, httpPort);
    serverThread.detach();

    // ── Open video source ─────────────────────
    cv::VideoCapture cap;
    if (streamSrc == "0") {
        cap.open(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    } else {
        // RTSP: use FFMPEG backend, set a small buffer to reduce latency
        cap.open(streamSrc, cv::CAP_FFMPEG);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    }
    if (!cap.isOpened()) {
        std::cerr << "ERROR: cannot open source: " << streamSrc << "\n";
        g_running = false;
        return 1;
    }

    std::array<int64_t, 4> inputShape{1, 3, INPUT_H, INPUT_W};
    const char* inputNames[]  = { inputName.c_str()  };
    const char* outputNames[] = { outputName.c_str() };

    // Start grab thread — drains decoder buffer, inference always gets latest frame
    std::thread grabThr(grabThread, std::ref(cap));

    cv::Mat frame;
    while (g_running) {
        {
            std::unique_lock<std::mutex> lk(g_grabMutex);
            g_grabCv.wait(lk, []{ return g_frameReady || !g_running; });
            if (!g_running) break;
            frame        = g_latestFrame.clone();
            g_frameReady = false;
        }
        if (frame.empty()) continue;

        auto t0 = std::chrono::steady_clock::now();

        // Pre-process
        float scale; int padLeft, padTop;
        cv::Mat lb = letterbox(frame, INPUT_W, INPUT_H, scale, padLeft, padTop);
        auto blob  = preprocess(lb);

        // Run inference
        Ort::MemoryInfo memInfo =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, blob.data(), blob.size(),
            inputShape.data(), inputShape.size());

        auto outputs = session.Run(Ort::RunOptions{nullptr},
                                   inputNames,  &inputTensor, 1,
                                   outputNames, 1);

        const float* outData = outputs[0].GetTensorMutableData<float>();

        auto t1 = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // Post-process
        auto dets = parseOutput(outData, numClasses, numAnchors,
                                scale, padLeft, padTop,
                                frame.cols, frame.rows);

        // Draw
        for (auto& d : dets) {
            cv::rectangle(frame, d.box, cv::Scalar(0, 255, 0), 2);
            std::string label = "hand " + std::to_string(static_cast<int>(d.confidence * 100)) + "%";
            int baseLine = 0;
            cv::Size textSz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
            cv::rectangle(frame,
                          cv::Point(d.box.x, d.box.y - textSz.height - 4),
                          cv::Point(d.box.x + textSz.width, d.box.y),
                          cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(frame, label,
                        cv::Point(d.box.x, d.box.y - 2),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 0, 0), 1);
        }

        std::string fps = "Infer: " + std::to_string(static_cast<int>(ms)) + " ms";
        cv::putText(frame, fps, {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 200, 255), 2);

        // Push annotated frame to MJPEG clients
        pushFrame(frame);

        // Print FPS to console
        std::cout << "\r" << fps << "   " << std::flush;
    }

    g_running = false;
    g_grabCv.notify_all();
    g_frameCv.notify_all();
    if (grabThr.joinable()) grabThr.join();
    cap.release();
    std::cout << "\nStream ended.\n";
    return 0;
}
