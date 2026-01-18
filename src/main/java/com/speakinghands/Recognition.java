package com.speakinghands;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.nio.FloatBuffer;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

public class Recognition {

    // Конфигурация (упрощено, без окон)
    private static final String MODEL_PATH = "mvit_slovo.onnx";
    private static final int WINDOW_SIZE = 16;
    private static final int FRAME_INTERVAL = 2;
    private static final int IMG_SIZE = 224;

    private static final float CONF_THRESHOLD = 0.04f;
    private static final int SMOOTHING = 3;
    private static final double COOLDOWN = 0.2;

    private static final float[] MEAN = {0.45f, 0.45f, 0.45f};
    private static final float[] STD = {0.225f, 0.225f, 0.225f};

    private static OrtSession session;
    private static OrtEnvironment env;
    private static String inputName;

    // Быстрые неблокирующие очереди
    private static final ConcurrentLinkedDeque<float[][][]> frameBuffer = new ConcurrentLinkedDeque<>();
    private static final ConcurrentLinkedDeque<String> predBuffer = new ConcurrentLinkedDeque<>();

    // Для синхронизации
    private static final Object inferenceLock = new Object();

    private static final AtomicBoolean running = new AtomicBoolean(true);
    private static final AtomicReference<String> lastPred = new AtomicReference<>("");
    private static final AtomicReference<Double> lastConf = new AtomicReference<>(0.0);
    private static volatile double lastEmitTime = 0.0;

    // Статистика для мониторинга
    private static volatile long totalFrames = 0;
    private static volatile long totalInferences = 0;
    private static volatile long startTime = 0;

    static {
        // Загружаем OpenCV (без HighGui)
        try {
            // Пробуем разные способы загрузки
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            System.out.println("[INFO] OpenCV loaded from system");
        } catch (UnsatisfiedLinkError e) {
            try {
                System.load("C:\\opencv\\build\\java\\x64\\opencv_java480.dll");
                System.out.println("[INFO] OpenCV loaded from C:\\opencv");
            } catch (UnsatisfiedLinkError e2) {
                try {
                    System.load("opencv_java480.dll");
                    System.out.println("[INFO] OpenCV loaded from current dir");
                } catch (UnsatisfiedLinkError e3) {
                    System.err.println("[ERROR] OpenCV DLL not found!");
                    System.err.println("Download from: https://github.com/opencv/opencv/releases");
                    System.err.println("Place opencv_java480.dll in:");
                    System.err.println("1. C:\\opencv\\build\\java\\x64\\");
                    System.err.println("2. Project folder");
                    System.err.println("3. Or set -Djava.library.path=path_to_dll");
                    System.exit(1);
                }
            }
        }
    }

    // Оптимизированная предобработка (однопроходная)
    private static float[][][] preprocessFast(Mat frame) {
        float[][][] tensor = new float[3][IMG_SIZE][IMG_SIZE];

        // Используем Mat для ресайза (один раз)
        Mat resized = new Mat();
        Imgproc.resize(frame, resized, new Size(IMG_SIZE, IMG_SIZE));

        // Предобработка в один проход
        int rows = resized.rows();
        int cols = resized.cols();
        int channels = resized.channels();

        byte[] data = new byte[rows * cols * channels];
        resized.get(0, 0, data);

        int idx = 0;
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                // BGR формат в OpenCV
                float b = (data[idx++] & 0xFF) / 255.0f;
                float g = (data[idx++] & 0xFF) / 255.0f;
                float r = (data[idx++] & 0xFF) / 255.0f;

                // Нормализация (BGR -> RGB)
                tensor[0][y][x] = (r - MEAN[0]) / STD[0];  // R
                tensor[1][y][x] = (g - MEAN[1]) / STD[1];  // G
                tensor[2][y][x] = (b - MEAN[2]) / STD[2];  // B
            }
        }

        resized.release();
        return tensor;
    }

    // Быстрая очередь инференса
    private static void inferenceLoop() {
        System.out.println("[INFO] Inference thread started");

        while (running.get()) {
            try {
                // Меньше сна для большей производительности
                Thread.sleep(2);

                synchronized (inferenceLock) {
                    if (frameBuffer.size() < WINDOW_SIZE) {
                        continue;
                    }

                    // Собираем окно
                    List<float[][][]> window = new ArrayList<>(WINDOW_SIZE);
                    Iterator<float[][][]> it = frameBuffer.descendingIterator();

                    for (int i = 0; i < WINDOW_SIZE && it.hasNext(); i++) {
                        window.add(it.next());
                    }
                    Collections.reverse(window);

                    // Подготавливаем FloatBuffer напрямую
                    FloatBuffer buffer = FloatBuffer.allocate(3 * WINDOW_SIZE * IMG_SIZE * IMG_SIZE);

                    // Порядок: [C, T, H, W] для модели
                    for (int c = 0; c < 3; c++) {
                        for (int t = 0; t < WINDOW_SIZE; t++) {
                            float[][] channelData = window.get(t)[c];
                            for (int h = 0; h < IMG_SIZE; h++) {
                                for (int w = 0; w < IMG_SIZE; w++) {
                                    buffer.put(channelData[h][w]);
                                }
                            }
                        }
                    }
                    buffer.rewind();

                    // Создаем тензор
                    long[] shape = {1, 3, WINDOW_SIZE, IMG_SIZE, IMG_SIZE};
                    OnnxTensor tensor = OnnxTensor.createTensor(env, buffer, shape);

                    // Запускаем инференс
                    OrtSession.Result results = session.run(
                            Collections.singletonMap(inputName, tensor)
                    );

                    totalInferences++;

                    // Получаем результат
                    float[][] logits = (float[][]) results.get(0).getValue();

                    // Быстрый softmax
                    float[] probs = fastSoftmax(logits[0]);

                    // Находим топ-1
                    int topIdx = 0;
                    float maxProb = probs[0];
                    for (int i = 1; i < probs.length; i++) {
                        if (probs[i] > maxProb) {
                            maxProb = probs[i];
                            topIdx = i;
                        }
                    }

                    // Обновляем буфер предсказаний
                    predBuffer.addLast(Constants.CLASSES[topIdx]);
                    while (predBuffer.size() > SMOOTHING) {
                        predBuffer.removeFirst();
                    }

                    // Сглаживание
                    String smoothed = smoothPrediction();

                    // Проверяем кд и порог
                    double currentTime = System.currentTimeMillis() / 1000.0;
                    if (maxProb >= CONF_THRESHOLD &&
                            (currentTime - lastEmitTime) > COOLDOWN) {
                        lastPred.set(smoothed);
                        lastConf.set((double) maxProb);
                        lastEmitTime = currentTime;

                        // Вывод в консоль
                        System.out.printf("[PRED] %s (%.3f) | FPS: %.1f | Inf: %d%n",
                                smoothed, maxProb,
                                getCurrentFPS(),
                                totalInferences);
                    }

                    tensor.close();
                    results.close();
                }

            } catch (Exception e) {
                if (running.get()) {
                    System.err.println("[ERROR] Inference: " + e.getMessage());
                }
            }
        }
    }

    // Быстрый softmax
    private static float[] fastSoftmax(float[] logits) {
        float[] exps = new float[logits.length];
        float sum = 0.0f;
        float max = logits[0];

        // Находим максимум для стабильности
        for (float val : logits) {
            if (val > max) max = val;
        }

        // Вычисляем exp и сумму
        for (int i = 0; i < logits.length; i++) {
            exps[i] = (float) Math.exp(logits[i] - max);
            sum += exps[i];
        }

        // Нормализуем
        float invSum = 1.0f / sum;
        for (int i = 0; i < exps.length; i++) {
            exps[i] *= invSum;
        }

        return exps;
    }

    // Сглаживание предсказаний
    private static String smoothPrediction() {
        if (predBuffer.isEmpty()) return "";

        // Используем массив для скорости
        String[] bufferArray = predBuffer.toArray(new String[0]);

        // Подсчитываем частоту
        Map<String, Integer> freq = new HashMap<>();
        for (String pred : bufferArray) {
            freq.put(pred, freq.getOrDefault(pred, 0) + 1);
        }

        // Находим наиболее частый
        String best = "";
        int maxCount = 0;
        for (Map.Entry<String, Integer> entry : freq.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                best = entry.getKey();
            }
        }

        return best;
    }

    // Статистика FPS
    private static double getCurrentFPS() {
        if (startTime == 0) return 0;
        double elapsed = (System.currentTimeMillis() - startTime) / 1000.0;
        return elapsed > 0 ? totalFrames / elapsed : 0;
    }

    public static void main(String[] args) {
        System.out.println("========================================");
        System.out.println("Recognition");
        System.out.println("Press Ctrl+C to exit");
        System.out.println("========================================");

        startTime = System.currentTimeMillis();

        try {
            // 1. Инициализация ONNX Runtime с оптимизациями
            env = OrtEnvironment.getEnvironment();

            // Настройки сессии для производительности
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();

            // Включаем оптимизации
            opts.setInterOpNumThreads(1);  // Один поток для параллельных операций
            opts.setIntraOpNumThreads(4);  // 4 потока для вычислений
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

            // Загружаем модель
            session = env.createSession(MODEL_PATH, opts);
            inputName = session.getInputInfo().keySet().iterator().next();

            System.out.println("[INFO] Model loaded: " + MODEL_PATH);
            System.out.println("[INFO] Input name: " + inputName);

            // Информация о модели
            System.out.println("[INFO] Model input shape: [1, 3, " + WINDOW_SIZE + ", " + IMG_SIZE + ", " + IMG_SIZE + "]");
            System.out.println("[INFO] Number of classes: " + Constants.CLASSES.length);

            // 2. Запускаем поток инференса
            Thread inferenceThread = new Thread(() -> {
                inferenceLoop();
            });
            inferenceThread.setDaemon(true);
            inferenceThread.setName("Inference-Thread");
            inferenceThread.start();

            // 3. Захват камеры (быстрый режим)
            System.out.println("[INFO] Opening camera...");
            VideoCapture cap = new VideoCapture(0);

            if (!cap.isOpened()) {
                System.err.println("[ERROR] Cannot open camera!");
                running.set(false);
                return;
            }

            // Настройки камеры для производительности
            cap.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);  // Меньшее разрешение
            cap.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);
            cap.set(Videoio.CAP_PROP_FPS, 30);           // 30 FPS

            System.out.println("[INFO] Camera opened (640x480 @ 30FPS)");
            System.out.println("[INFO] Processing frames...");
            System.out.println("----------------------------------------");

            Mat frame = new Mat();
            int framesSkipped = 0;

            // 4. Главный цикл захвата кадров
            while (running.get()) {
                // Читаем кадр
                if (!cap.read(frame) || frame.empty()) {
                    System.err.println("[WARN] Failed to grab frame");
                    continue;
                }

                totalFrames++;

                // Пропускаем кадры согласно интервалу (для производительности)
                if (totalFrames % FRAME_INTERVAL != 0) {
                    framesSkipped++;
                    continue;
                }

                // Быстрая предобработка
                float[][][] tensor = preprocessFast(frame);

                // Добавляем в буфер (неблокирующе)
                frameBuffer.addLast(tensor);

                // Ограничиваем размер буфера
                while (frameBuffer.size() > WINDOW_SIZE * 2) {
                    frameBuffer.removeFirst();
                }

                // Периодический вывод статистики
                if (totalFrames % 100 == 0) {
                    double fps = getCurrentFPS();
                    System.out.printf("[STAT] FPS: %.1f | Frames: %d | Inf: %d | Buffer: %d%n",
                            fps, totalFrames, totalInferences, frameBuffer.size());
                }

                // Небольшая пауза для CPU
                Thread.sleep(1);
            }

            // 5. Очистка
            cap.release();
            frame.release();

            System.out.println("========================================");
            System.out.println("Session statistics:");
            System.out.printf("Total frames: %d%n", totalFrames);
            System.out.printf("Total inferences: %d%n", totalInferences);
            System.out.printf("Average FPS: %.1f%n", getCurrentFPS());
            System.out.printf("Runtime: %.1f seconds%n",
                    (System.currentTimeMillis() - startTime) / 1000.0);
            System.out.println("========================================");

        } catch (Exception e) {
            System.err.println("[ERROR] " + e.getMessage());
            e.printStackTrace();
        } finally {
            running.set(false);
            try {
                if (session != null) session.close();
                if (env != null) env.close();
            } catch (Exception e) {
                // Игнорируем ошибки при закрытии
            }
            System.out.println("[INFO] Cleanup complete");
        }
    }
}