package com.example.kotlearn

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class DeepfakeDetector(context: Context) {

    private var interpreter: Interpreter? = null
    // Make sure this filename matches exactly what is in your assets folder
    private val MODEL_NAME = "lipinc_model.tflite"
    private val NUM_FRAMES = 8
    private val HEIGHT = 64
    private val WIDTH = 144
    private val CHANNELS = 3

    init {
        val options = Interpreter.Options()

        // 1. DETECT EMULATOR (Critical to prevent crashes)
        val isEmulator = Build.FINGERPRINT.contains("generic") ||
                Build.FINGERPRINT.contains("unknown") ||
                Build.MODEL.contains("google_sdk") ||
                Build.MODEL.contains("Emulator") ||
                Build.MODEL.contains("Android SDK built for x86") ||
                Build.MANUFACTURER.contains("Genymotion")

        if (isEmulator) {
            Log.d("DeepfakeDetector", "Emulator detected. Forcing CPU mode.")
            options.setNumThreads(4)
        } else {
            // Only try GPU on real phones
            if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                try {
                    options.addDelegate(GpuDelegate())
                    Log.d("DeepfakeDetector", "GPU Delegate Enabled")
                } catch (e: Exception) {
                    Log.e("DeepfakeDetector", "GPU Failed, falling back to CPU", e)
                    options.setNumThreads(4)
                }
            } else {
                options.setNumThreads(4)
            }
        }

        // 2. LOAD MODEL
        try {
            interpreter = Interpreter(loadModelFile(context, MODEL_NAME), options)
            Log.d("DeepfakeDetector", "Model Loaded Successfully")
        } catch (e: Exception) {
            Log.e("DeepfakeDetector", "Error loading model", e)
        }
    }

    fun analyzeVideo(context: Context, videoUri: Uri): Pair<String, String> {
        if (interpreter == null) return Pair("Error", "Model failed to load")

        try {
            val frames = extractFrames(context, videoUri)
            if (frames.size < NUM_FRAMES) {
                return Pair("Error", "Video too short (Need > 1 sec)")
            }

            val frameBuffer = ByteBuffer.allocateDirect(1 * NUM_FRAMES * HEIGHT * WIDTH * CHANNELS * 4)
            frameBuffer.order(ByteOrder.nativeOrder())
            val residueBuffer = ByteBuffer.allocateDirect(1 * (NUM_FRAMES - 1) * HEIGHT * WIDTH * CHANNELS * 4)
            residueBuffer.order(ByteOrder.nativeOrder())

            fillBuffers(frames, frameBuffer, residueBuffer)

            val outputBuffer = Array(1) { FloatArray(2) }
            val inputs = arrayOf(frameBuffer, residueBuffer)
            val outputs = mapOf(0 to outputBuffer)

            interpreter?.runForMultipleInputsOutputs(inputs, outputs)

            // Index 1 is usually the "Fake" probability
            val fakeProbability = outputBuffer[0][1]
            val verdict = if (fakeProbability > 0.50f) "Fake" else "Authentic"
            val confidence = "${(fakeProbability * 100).toInt()}%"

            return Pair(verdict, confidence)

        } catch (e: Exception) {
            e.printStackTrace()
            return Pair("Error", "Analysis Failed: ${e.message}")
        }
    }

    private fun extractFrames(context: Context, uri: Uri): List<Bitmap> {
        val frames = mutableListOf<Bitmap>()
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, uri)
            val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val durationMs = durationStr?.toLong() ?: 0L
            val interval = if (durationMs > 1000) durationMs / NUM_FRAMES else 0

            for (i in 0 until NUM_FRAMES) {
                val timeUs = i * interval * 1000
                val frame = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                frame?.let {
                    frames.add(Bitmap.createScaledBitmap(it, WIDTH, HEIGHT, true))
                }
            }
        } catch (e: Exception) { e.printStackTrace() } finally { retriever.release() }
        return frames
    }

    private fun fillBuffers(frames: List<Bitmap>, frameBuf: ByteBuffer, resBuf: ByteBuffer) {
        frameBuf.rewind(); resBuf.rewind()

        val pixelArrays = frames.map { bitmap ->
            val pixels = IntArray(WIDTH * HEIGHT)
            bitmap.getPixels(pixels, 0, WIDTH, 0, 0, WIDTH, HEIGHT)
            pixels
        }

        // Fill Frames (Normalize)
        for (pixels in pixelArrays) {
            for (pixel in pixels) {
                val r = (pixel shr 16 and 0xFF); val g = (pixel shr 8 and 0xFF); val b = (pixel and 0xFF)
                frameBuf.putFloat(r / 255.0f); frameBuf.putFloat(g / 255.0f); frameBuf.putFloat(b / 255.0f)
            }
        }

        // Fill Residues (Differences)
        for (i in 1 until pixelArrays.size) {
            val curr = pixelArrays[i]; val prev = pixelArrays[i - 1]
            for (j in curr.indices) {
                val c = curr[j]; val p = prev[j]
                val rD = ((c shr 16 and 0xFF) - (p shr 16 and 0xFF)) / 255.0f
                val gD = ((c shr 8 and 0xFF) - (p shr 8 and 0xFF)) / 255.0f
                val bD = ((c and 0xFF) - (p and 0xFF)) / 255.0f
                resBuf.putFloat(rD); resBuf.putFloat(gD); resBuf.putFloat(bD)
            }
        }
    }

    private fun loadModelFile(context: Context, modelName: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun close() {
        interpreter?.close()
    }
}