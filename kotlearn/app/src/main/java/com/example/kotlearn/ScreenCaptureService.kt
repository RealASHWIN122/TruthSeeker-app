package com.example.kotlearn

import android.app.*
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.MediaRecorder
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import java.io.File

class ScreenCaptureService : Service() {

    private var mediaProjection: MediaProjection? = null
    private var mediaRecorder: MediaRecorder? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var videoFile: File? = null

    companion object {
        const val CHANNEL_ID = "ScreenCaptureChannel"
        const val NOTIFICATION_ID = 1
        const val ACTION_STOP = "STOP_RECORDING"
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopRecordingAndAnalyze()
            return START_NOT_STICKY
        }

        val resultCode = intent?.getIntExtra("RESULT_CODE", Activity.RESULT_CANCELED) ?: Activity.RESULT_CANCELED
        
        // Handle getParcelableExtra for different Android versions
        val data = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            intent?.getParcelableExtra("DATA", Intent::class.java)
        } else {
            @Suppress("DEPRECATION")
            intent?.getParcelableExtra<Intent>("DATA")
        }
        
        val scanType = intent?.getStringExtra("SCAN_TYPE") ?: "QUICK"

        // 🔥 FIX 1: Android 14 requires explicit declaration of the FGS type in code
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID, 
                createNotification(), 
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
            )
        } else {
            startForeground(NOTIFICATION_ID, createNotification())
        }

        if (data != null) {
            setupProjection(resultCode, data)
            startRecording()
        }

        return START_STICKY
    }

    private fun setupProjection(resultCode: Int, data: Intent) {
        val mpManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        mediaProjection = mpManager.getMediaProjection(resultCode, data)

        // 🔥 FIX 2: Android 14 strictly requires a callback to be registered BEFORE capturing
        mediaProjection?.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() {
                super.onStop()
                // If the user stops the cast via the Android system tray, stop our app cleanly
                stopRecordingAndAnalyze()
            }
        }, null)
    }

    private fun startRecording() {
        videoFile = File(getExternalFilesDir(null), "capture.mp4")
        
        // Initialize MediaRecorder (handling deprecation for API 31+)
        mediaRecorder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            MediaRecorder(this)
        } else {
            @Suppress("DEPRECATION")
            MediaRecorder()
        }.apply {
            setVideoSource(MediaRecorder.VideoSource.SURFACE)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setVideoEncoder(MediaRecorder.VideoEncoder.H264)
            setVideoEncodingBitRate(2 * 1000 * 1000) // Bumped to 2Mbps for clearer AI analysis
            setVideoFrameRate(30)
            setVideoSize(720, 1280) 
            setOutputFile(videoFile?.absolutePath)
            prepare()
            start()
        }

        virtualDisplay = mediaProjection?.createVirtualDisplay(
            "ScreenCapture", 720, 1280, 160,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            mediaRecorder?.surface, null, null
        )
    }

    private fun stopRecordingAndAnalyze() {
        try {
            mediaRecorder?.stop()
            mediaRecorder?.release()
            mediaRecorder = null
            virtualDisplay?.release()
            virtualDisplay = null
            mediaProjection?.stop()
            mediaProjection = null
        } catch (e: Exception) {
            Log.e("ScreenCapture", "Error stopping: ${e.message}")
        }

        // TODO: Pass the videoFile to DeepfakeDetector here
        Log.d("ScreenCapture", "✅ Recording Stopped successfully. File saved at: ${videoFile?.absolutePath}")
        
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun createNotification(): Notification {
        // 🔥 FIX: NotificationChannel requires API 26 (Android 8.0)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(CHANNEL_ID, "Screen Recording", NotificationManager.IMPORTANCE_LOW)
            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager?.createNotificationChannel(channel)
        }

        val stopIntent = Intent(this, ScreenCaptureService::class.java).apply { action = ACTION_STOP }
        val stopPendingIntent = PendingIntent.getService(
            this, 0, stopIntent, 
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("TruthSeeker is Scanning")
            .setContentText("Tap STOP to analyze screen for deepfakes.")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .addAction(android.R.drawable.ic_media_pause, "STOP", stopPendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW) // Support for older API levels
            .build()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
