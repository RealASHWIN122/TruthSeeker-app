package com.example.kotlearn

import android.app.*
import android.content.Context
import android.content.Intent
import android.graphics.PixelFormat
import android.net.Uri
import android.os.Build
import android.os.IBinder
import android.view.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.NotificationCompat
import androidx.lifecycle.*
import androidx.savedstate.setViewTreeSavedStateRegistryOwner
import androidx.savedstate.SavedStateRegistry
import androidx.savedstate.SavedStateRegistryOwner
import androidx.savedstate.SavedStateRegistryController
import com.example.kotlearn.ui.theme.KotlearnTheme

class FloatingWidgetService : Service() {

    private lateinit var windowManager: WindowManager
    private var floatingView: View? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == "ACTION_SCREENSHOT" || intent?.action == "ACTION_RECORD") {
            // Simulate capture and send to app
            val dummyUri = Uri.parse("content://dummy/media") 
            sendToApp(dummyUri, isQuickScan = true)
        }
        return START_STICKY
    }

    override fun onCreate() {
        super.onCreate()
        try {
            startForegroundService()
            showFloatingWidget()
        } catch (e: Exception) {
            stopSelf()
        }
    }

    private fun startForegroundService() {
        val channelId = "floating_widget_channel"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId,
                "Floating Widget Service",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }

        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("Truth Seeker")
            .setContentText("Widget is active")
            .setSmallIcon(R.mipmap.ic_launcher)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()

        startForeground(1, notification)
    }

    private fun showFloatingWidget() {
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            else
                @Suppress("DEPRECATION") WindowManager.LayoutParams.TYPE_PHONE,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = 0
            y = 100
        }

        val composeView = ComposeView(this).apply {
            setContent {
                KotlearnTheme {
                    var showMenu by remember { mutableStateOf(false) }
                    
                    Box(contentAlignment = Alignment.Center) {
                        Surface(
                            modifier = Modifier
                                .size(60.dp)
                                .clickable { showMenu = !showMenu },
                            shape = CircleShape,
                            color = MaterialTheme.colorScheme.primary,
                            shadowElevation = 8.dp
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                Text("TS", color = Color.White, fontWeight = androidx.compose.ui.text.font.FontWeight.Bold)
                            }
                        }

                        if (showMenu) {
                            Column(
                                modifier = Modifier
                                    .offset(y = 70.dp)
                                    .background(Color.White, RoundedCornerShape(8.dp))
                                    .padding(8.dp)
                                    .border(1.dp, Color.LightGray, RoundedCornerShape(8.dp)),
                                verticalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                FloatingOption(Icons.Default.CameraAlt, "Screenshot") {
                                    showMenu = false
                                    requestCapture(isScreenshot = true)
                                }
                                FloatingOption(Icons.Default.Videocam, "Record") {
                                    showMenu = false
                                    requestCapture(isScreenshot = false)
                                }
                            }
                        }
                    }
                }
            }
        }

        val lifecycleOwner = object : LifecycleOwner {
            private val lifecycleRegistry = LifecycleRegistry(this).apply {
                handleLifecycleEvent(Lifecycle.Event.ON_CREATE)
                handleLifecycleEvent(Lifecycle.Event.ON_START)
                handleLifecycleEvent(Lifecycle.Event.ON_RESUME)
            }
            override val lifecycle: Lifecycle = lifecycleRegistry
        }
        
        val viewModelStore = ViewModelStore()
        composeView.setViewTreeLifecycleOwner(lifecycleOwner)
        composeView.setViewTreeViewModelStoreOwner(object : ViewModelStoreOwner {
            override val viewModelStore: ViewModelStore = viewModelStore
        })
        
        val ssrOwner = object : SavedStateRegistryOwner {
            private val controller = SavedStateRegistryController.create(this)
            init { controller.performRestore(null) }
            override val lifecycle: Lifecycle = lifecycleOwner.lifecycle
            override val savedStateRegistry: SavedStateRegistry = controller.savedStateRegistry
        }
        composeView.setViewTreeSavedStateRegistryOwner(ssrOwner)

        floatingView = composeView
        windowManager.addView(floatingView, params)

        floatingView?.setOnTouchListener(object : View.OnTouchListener {
            private var initialX = 0
            private var initialY = 0
            private var initialTouchX = 0f
            private var initialTouchY = 0f

            override fun onTouch(v: View, event: MotionEvent): Boolean {
                when (event.action) {
                    MotionEvent.ACTION_DOWN -> {
                        initialX = params.x
                        initialY = params.y
                        initialTouchX = event.rawX
                        initialTouchY = event.rawY
                        return false
                    }
                    MotionEvent.ACTION_MOVE -> {
                        params.x = initialX + (event.rawX - initialTouchX).toInt()
                        params.y = initialY + (event.rawY - initialTouchY).toInt()
                        try {
                            windowManager.updateViewLayout(floatingView, params)
                        } catch (e: Exception) {}
                        return true
                    }
                    MotionEvent.ACTION_UP -> {
                        val diffX = Math.abs(event.rawX - initialTouchX)
                        val diffY = Math.abs(event.rawY - initialTouchY)
                        if (diffX < 10 && diffY < 10) {
                            v.performClick()
                            return false
                        }
                    }
                }
                return false
            }
        })
    }

    @Composable
    fun FloatingOption(icon: androidx.compose.ui.graphics.vector.ImageVector, label: String, onClick: () -> Unit) {
        Row(
            modifier = Modifier
                .clickable(onClick = onClick)
                .padding(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(icon, contentDescription = label, modifier = Modifier.size(24.dp), tint = MaterialTheme.colorScheme.primary)
            Spacer(modifier = Modifier.width(12.dp))
            Text(label, fontSize = 16.sp, color = Color.Black)
        }
    }

    private fun requestCapture(isScreenshot: Boolean) {
        val intent = Intent(this, ProjectionPromptActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            putExtra("isScreenshot", isScreenshot)
        }
        startActivity(intent)
    }

    private fun sendToApp(uri: Uri, isQuickScan: Boolean) {
        val intent = Intent(this, MainActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
            putExtra("capturedMediaUri", uri.toString())
            putExtra("scanType", if (isQuickScan) "quick" else "deep")
        }
        startActivity(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        if (floatingView != null) {
            try {
                windowManager.removeView(floatingView)
            } catch (e: Exception) {}
        }
    }
}
