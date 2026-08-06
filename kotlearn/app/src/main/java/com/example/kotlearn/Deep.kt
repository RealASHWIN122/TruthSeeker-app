package com.example.kotlearn

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.FileOutputStream





import androidx.compose.ui.Alignment
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.material3.RadioButton
import androidx.compose.material3.RadioButtonDefaults
@Composable
fun DeepScanScreen() {
    var selectedUri by remember { mutableStateOf<Uri?>(null) }
    var isAudioFile by remember { mutableStateOf(false) }
    var selectedVideoModel by remember { mutableStateOf("Skyra") } // Default to Skyra
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val scrollState = rememberScrollState()

    var isScanning by remember { mutableStateOf(false) }
    var scanResult by remember { mutableStateOf<DeepScanResponse?>(null) }
    var activeGridIndices by remember { mutableStateOf<List<Int>>(emptyList()) }
    var userQuery by remember { mutableStateOf("") }

    val exoPlayer = remember { ExoPlayer.Builder(context).build().apply { prepare() } }
    DisposableEffect(Unit) { onDispose { exoPlayer.release() } }

    val mediaPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            selectedUri = uri
            val mimeType = context.contentResolver.getType(uri) ?: ""
            isAudioFile = mimeType.startsWith("audio")
            exoPlayer.setMediaItem(MediaItem.fromUri(uri))
            exoPlayer.prepare()
        }
    }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("Skyra Forensic Suite", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold, color = Color.Cyan)

        // 1. Video + Grid
        Box(modifier = Modifier.fillMaxWidth().height(250.dp).padding(vertical = 12.dp)) {
            AndroidView(factory = { ctx -> PlayerView(ctx).apply { player = exoPlayer } }, modifier = Modifier.matchParentSize())
            Canvas(modifier = Modifier.matchParentSize()) {
                val cols = 16
                val cellW = size.width / cols
                val cellH = size.height / cols
                for (i in 0 until 256) {
                    val highlight = activeGridIndices.contains(i)
                    val x = (i % cols) * cellW
                    val y = (i / cols) * cellH
                    if (highlight) {
                        drawRect(color = Color.Red.copy(alpha = 0.4f), topLeft = Offset(x, y), size = Size(cellW, cellH))
                    }
                    drawRect(
                        color = if (highlight) Color.Red else Color.White.copy(alpha = 0.1f),
                        topLeft = Offset(x, y),
                        size = Size(cellW, cellH),
                        style = Stroke(width = 1f)
                    )
                }
            }
        }

        // --- NEW: THE AI TOGGLE SWITCH ---
        if (selectedUri != null && !isAudioFile) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Engine: ", fontWeight = FontWeight.Bold)
                RadioButton(
                    selected = selectedVideoModel == "Skyra",
                    onClick = { selectedVideoModel = "Skyra" },
                    colors = RadioButtonDefaults.colors(selectedColor = MaterialTheme.colorScheme.tertiary)
                )
                Text("Skyra (AMD)")
                Spacer(modifier = Modifier.width(16.dp))
                RadioButton(
                    selected = selectedVideoModel == "TimeSformer",
                    onClick = { selectedVideoModel = "TimeSformer" },
                    colors = RadioButtonDefaults.colors(selectedColor = Color.Cyan)
                )
                Text("TimeSformer (Colab)")
            }
        }

        // 2. Actions
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            Button(
                onClick = { mediaPickerLauncher.launch(arrayOf("video/*", "audio/*")) },
                modifier = Modifier.weight(1f)
            ) {
                Text("Select Media")
            }
            Button(
                onClick = {
                    scope.launch {
                        isScanning = true
                        try {
                            val contentResolver = context.contentResolver
                            val mimeType = contentResolver.getType(selectedUri!!) ?: ""
                            val isAudio = mimeType.startsWith("audio")

                            // Prepare file
                            val tempFile = File(context.cacheDir, if (isAudio) "upload.wav" else "upload.mp4")
                            contentResolver.openInputStream(selectedUri!!)?.use { it.copyTo(FileOutputStream(tempFile)) }

                            val mediaTypeStr = if (isAudio) "audio/*" else "video/mp4"
                            val partName = if (isAudio) "audio" else "video"
                            val body = MultipartBody.Part.createFormData(partName, tempFile.name, tempFile.asRequestBody(mediaTypeStr.toMediaTypeOrNull()))

                            // --- UPDATED: ROUTING LOGIC ---
                            if (isAudio) {
                                // 🎧 ROUTE TO COLAB (AUDIO)
                                val response = RetrofitInstance.audioApi.analyzeAudio(body)
                                scanResult = DeepScanResponse(
                                    verdict = response.verdict ?: "Unknown",
                                    confidence = response.confidence ?: "0%",
                                    details = response.details ?: "No details provided.",
                                    markers = response.markers ?: emptyList()
                                )
                            } else {
                                // 🎬 IT'S A VIDEO - CHECK THE TOGGLE!
                                if (selectedVideoModel == "Skyra") {
                                    // ROUTE TO AMD MI300X
                                    val response = RetrofitInstance.skyraApi.skyraDeepScan(body)
                                    if (response.status == "success") {
                                        scanResult = DeepScanResponse(
                                            verdict = "Analyzed",
                                            confidence = "MI300X Core",
                                            details = response.analysis ?: "No text analysis provided.",
                                            markers = response.markers ?: emptyList()
                                        )
                                    }
                                } else {
                                    // ROUTE TO COLAB (TIMESFORMER)
                                    val response = RetrofitInstance.audioApi.analyzeVideo(body)
                                    scanResult = DeepScanResponse(
                                        verdict = response.verdict ?: "Unknown",
                                        confidence = response.confidence ?: "0%",
                                        details = response.details ?: "No details provided.",
                                        markers = response.markers ?: emptyList()
                                    )
                                }
                            }
                        } catch (e: Exception) {
                            scanResult = DeepScanResponse("Error", "", "Connection failed: ${e.message}")
                        } finally { isScanning = false }
                    }
                },
                enabled = selectedUri != null && !isScanning,
                modifier = Modifier.weight(1f)
            ) {
                if (isScanning) CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                else Text("Deep Scan")
            }
        }

        // 3. Markers
        Spacer(modifier = Modifier.height(16.dp))
        Text("Forensic Markers", style = MaterialTheme.typography.titleSmall)
        LazyRow(modifier = Modifier.fillMaxWidth().height(90.dp)) {
            items(scanResult?.markers ?: emptyList()) { marker ->
                Card(
                    modifier = Modifier
                        .width(180.dp)
                        .padding(4.dp)
                        .clickable {
                            exoPlayer.seekTo(marker.timestampMs)
                            activeGridIndices = marker.gridIndices ?: emptyList()
                        },
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)
                ) {
                    Column(Modifier.padding(8.dp)) {
                        Text(
                            text = marker.summary ?: "Marker",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onSecondaryContainer
                        )
                        Text(
                            text = "Timestamp: ${marker.timestampMs / 1000}s",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.7f)
                        )
                    }
                }
            }
        }

        // 4. Scrollable Reasoning
        Spacer(modifier = Modifier.height(16.dp))
        Text("AI Reasoning", style = MaterialTheme.typography.titleSmall)
        Card(
            modifier = Modifier.fillMaxWidth().weight(1f).padding(vertical = 8.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
        ) {
            Box(modifier = Modifier.fillMaxSize().padding(12.dp).verticalScroll(scrollState)) {
                Text(
                    text = scanResult?.details ?: "Awaiting analysis...",
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }

        OutlinedTextField(
            value = userQuery, onValueChange = { userQuery = it }, modifier = Modifier.fillMaxWidth(),
            placeholder = { Text("Ask the AI...") },
            trailingIcon = { IconButton(onClick = {}) { Icon(Icons.Default.Send, null) } }
        )
    }
}