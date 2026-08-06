package com.example.kotlearn

import android.app.Activity
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.os.Bundle
import android.widget.Button
import android.widget.Toast
import androidx.core.content.ContextCompat

class WidgetPromptActivity : Activity() {

    private lateinit var projectionManager: MediaProjectionManager
    private val SCREEN_RECORD_REQUEST_CODE = 1001
    private var selectedScanType = "QUICK"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // This makes the activity look like a small floating window
        setContentView(R.layout.activity_widget_prompt)

        projectionManager = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager

        val btnQuickScan = findViewById<Button>(R.id.btnQuickScan)
        val btnDeepScan = findViewById<Button>(R.id.btnDeepScan)

        btnQuickScan.setOnClickListener {
            selectedScanType = "QUICK"
            requestScreenCapture()
        }

        btnDeepScan.setOnClickListener {
            selectedScanType = "DEEP"
            requestScreenCapture()
        }
    }

    private fun requestScreenCapture() {
        // MANDATORY: You must ask the OS for permission to record the screen
        val captureIntent = projectionManager.createScreenCaptureIntent()
        startActivityForResult(captureIntent, SCREEN_RECORD_REQUEST_CODE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == SCREEN_RECORD_REQUEST_CODE) {
            if (resultCode == RESULT_OK && data != null) {
                // Permission granted! Start the recording service
                startRecordingService(resultCode, data)
                finish() // Close the small window, leave them on their home screen
            } else {
                Toast.makeText(this, "Screen record permission denied", Toast.LENGTH_SHORT).show()
                finish() // Close window on denial
            }
        }
    }

    private fun startRecordingService(resultCode: Int, data: Intent) {
        val serviceIntent = Intent(this, ScreenCaptureService::class.java).apply {
            putExtra("RESULT_CODE", resultCode)
            putExtra("DATA", data)
            putExtra("SCAN_TYPE", selectedScanType) // Pass "QUICK" or "DEEP" to the service
        }
        // FIX: startForegroundService requires API 26. ContextCompat handles the version check for us.
        ContextCompat.startForegroundService(this, serviceIntent)
    }
}
