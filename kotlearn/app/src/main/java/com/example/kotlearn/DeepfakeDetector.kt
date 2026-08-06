package com.example.kotlearn

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class DeepfakeDetector(context: Context) {

    private var interpreter: Interpreter? = null
    private val MODEL_NAME = "lip_flex.tflite"

    // TimeSformer Requirements
    private val NUM_FRAMES = 8
    private val HEIGHT = 224
    private val WIDTH = 224
    private val CHANNELS = 3

    init {
        val options = Interpreter.Options()

        // Use XNNPack for CPU execution (supports the TimeSformer's 3D Attention)
        options.setUseXNNPACK(true)
        val numProcessors = Runtime.getRuntime().availableProcessors()
        options.setNumThreads(if (numProcessors > 2) numProcessors - 1 else 1)

        try {
            val modelBuffer = loadModelFile(context, MODEL_NAME)
            interpreter = Interpreter(modelBuffer, options)
            Log.d("DeepfakeDetector", "🚀 SUCCESS: TimeSformer Engine Online (CPU Mode)!")
        } catch (e: Exception) {
            Log.e("DeepfakeDetector", "❌ FATAL LOAD ERROR: ${e.message}")
            e.printStackTrace()
        }
    }

    private fun loadModelFile(context: Context, modelName: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        return inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    fun analyzeVideo(context: Context, videoUri: Uri): Pair<String, String> {
        val safeInterpreter = interpreter ?: return Pair("Error", "Model Not Loaded")

        return try {
            val frames = extractFrames(context, videoUri)
            if (frames.size < NUM_FRAMES) return Pair("Error", "Clip too short")

            // Allocate buffer for [1, 3, 8, 224, 224] format
            val frameBuffer = ByteBuffer.allocateDirect(1 * CHANNELS * NUM_FRAMES * HEIGHT * WIDTH * 4)
            frameBuffer.order(ByteOrder.nativeOrder())

            fillBuffer(frames, frameBuffer)

            // TimeSformer outputs a single logit [1, 1]
            val outputBuffer = Array(1) { FloatArray(1) }
            safeInterpreter.run(frameBuffer, outputBuffer)

            val rawLogit = outputBuffer[0][0]

            // Sigmoid conversion for PyTorch logits
            val fakeProbability = (1.0 / (1.0 + Math.exp(-rawLogit.toDouble()))).toFloat()
            val verdict = if (fakeProbability > 0.50f) "Fake" else "Authentic"

            val confidenceValue = (if (verdict == "Fake") fakeProbability else 1f - fakeProbability) * 100
            val confidence = String.format("%.2f%%", confidenceValue)

            Pair(verdict, confidence)
        } catch (e: Exception) {
            Log.e("DeepfakeDetector", "Analysis Crash: ${e.message}")
            Pair("Error", "Analysis Failed")
        }
    }

    private fun extractFrames(context: Context, uri: Uri): List<Bitmap> {
        val frames = mutableListOf<Bitmap>()
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, uri)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
            val interval = if (durationMs > 1000) durationMs / NUM_FRAMES else 0

            for (i in 0 until NUM_FRAMES) {
                val timeUs = i * interval * 1000
                val frame = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                frame?.let {
                    frames.add(Bitmap.createScaledBitmap(it, WIDTH, HEIGHT, true))
                }
            }
        } finally {
            try { retriever.release() } catch (e: Exception) { /* Ignore */ }
        }
        return frames
    }

    private fun fillBuffer(frames: List<Bitmap>, frameBuf: ByteBuffer) {
        frameBuf.rewind()

        val pixelArrays = frames.map { bitmap ->
            val pixels = IntArray(WIDTH * HEIGHT)
            bitmap.getPixels(pixels, 0, WIDTH, 0, 0, WIDTH, HEIGHT)
            pixels
        }

        // ImageNet Normalization (Required for TimeSformer)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        // PyTorch layout: [Channels, Frames, Height, Width]
        for (c in 0 until CHANNELS) {
            for (f in 0 until NUM_FRAMES) {
                val pixels = pixelArrays[f]
                for (pixel in pixels) {
                    val color = when (c) {
                        0 -> (pixel shr 16 and 0xFF) / 255.0f // R
                        1 -> (pixel shr 8 and 0xFF) / 255.0f  // G
                        else -> (pixel and 0xFF) / 255.0f     // B
                    }
                    frameBuf.putFloat((color - mean[c]) / std[c])
                }
            }
        }
    }

    fun close() {
        interpreter?.close()
    }
}