package com.example.kotlearn

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState // CRITICAL IMPORT
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import coil.compose.AsyncImage
import com.example.kotlearn.ui.theme.KotlearnTheme
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.font.FontFamily

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        if (Settings.canDrawOverlays(this)) {
            startService(Intent(this, FloatingWidgetService::class.java))
        }

        setContent {
            KotlearnTheme {
                val navController = rememberNavController()

                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    NavHost(
                        navController = navController,
                        startDestination = "home",
                        modifier = Modifier.padding(innerPadding)
                    ) {
                        composable("home") {
                            GreetingImage(
                                message = "Truth Seeker",
                                onScanClicked = {
                                    if (!Settings.canDrawOverlays(this@MainActivity)) {
                                        val intent = Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:$packageName"))
                                        startActivity(intent)
                                    } else {
                                        navController.navigate("scan_options")
                                    }
                                },
                                onAboutClicked = { navController.navigate("about") }
                            )
                        }

                        composable("scan_options") {
                            ScanTypeScreen(
                                onQuickClick = { navController.navigate("quick_scan") },
                                onDeepClick = { navController.navigate("deep_upload") }
                            )
                        }

                        composable("quick_scan") {
                            QuickScanScreen(navController)
                        }

                        composable("deep_upload") {
                            DeepScanScreen()
                        }

                        composable("about") {
                            AboutScreen()
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun BouncingButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    containerColor: Color = MaterialTheme.colorScheme.primary,
    content: @Composable RowScope.() -> Unit
) {
    val interactionSource = remember { MutableInteractionSource() }

    // FIXED: Use the extension function syntax
    val isPressed by interactionSource.collectIsPressedAsState()

    val scale by animateFloatAsState(
        targetValue = if (isPressed) 0.92f else 1f,
        label = "buttonScale"
    )

    Button(
        onClick = onClick,
        modifier = modifier.scale(scale),
        enabled = enabled,
        colors = ButtonDefaults.buttonColors(containerColor = containerColor),
        interactionSource = interactionSource,
        content = content
    )
}

@Composable
fun TypewriterText(text: String, modifier: Modifier = Modifier, style: androidx.compose.ui.text.TextStyle) {
    var visibleText by remember { mutableStateOf("") }
    LaunchedEffect(text) {
        visibleText = ""
        text.forEachIndexed { index, _ ->
            delay(60)
            visibleText = text.substring(0, index + 1)
        }
    }
    Text(text = visibleText, modifier = modifier, style = style)
}

@Composable
fun GreetingImage(message: String, onScanClicked: () -> Unit, onAboutClicked: () -> Unit) {
    Box(Modifier.fillMaxSize()) {
        Image(
            painter = painterResource(R.drawable.cyberbg),
            contentDescription = null,
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize(),
            alpha = 0.9f
        )
        Column(
            modifier = Modifier.fillMaxSize().padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(60.dp))
            TypewriterText(
                text = message,
                style = LocalTextStyle.current.copy(
                    fontSize = 50.sp, color = Color.Cyan, textAlign = TextAlign.Center,
                    fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace
                )
            )
            Spacer(modifier = Modifier.weight(1f))
            Image(
                painter = painterResource(id = R.drawable.cyverlogo),
                contentDescription = null,
                modifier = Modifier.size(250.dp).clip(RoundedCornerShape(16.dp)),
                contentScale = ContentScale.Crop
            )
            Spacer(modifier = Modifier.weight(1f))
            BouncingButton(onClick = onScanClicked, modifier = Modifier.width(220.dp)) {
                Text("Scan Media", fontSize = 20.sp)
            }
            Spacer(modifier = Modifier.height(16.dp))
            BouncingButton(onClick = onAboutClicked, modifier = Modifier.width(220.dp), containerColor = MaterialTheme.colorScheme.secondary) {
                Text("Fact Check AI", fontSize = 20.sp)
            }
            Spacer(modifier = Modifier.height(60.dp))
        }
    }
}

@Composable
fun ScanTypeScreen(onQuickClick: () -> Unit, onDeepClick: () -> Unit) {
    Surface(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text("Select Analysis Tier", fontSize = 28.sp, fontWeight = androidx.compose.ui.text.font.FontWeight.Bold)
            Spacer(modifier = Modifier.height(40.dp))
            BouncingButton(onClick = onQuickClick, modifier = Modifier.width(250.dp).height(60.dp)) {
                Text("Quick Local Scan")
            }
            Spacer(modifier = Modifier.height(20.dp))
            BouncingButton(onClick = onDeepClick, modifier = Modifier.width(250.dp).height(60.dp), containerColor = MaterialTheme.colorScheme.tertiary) {
                Text("Cloud Deep Forensic")
            }
        }
    }
}

@Composable


fun QuickScanScreen(navController: NavController) {
    val context = LocalContext.current
    var selectedUri by remember { mutableStateOf<Uri?>(null) }
    var verdict by remember { mutableStateOf("Pending Selection...") }
    var confidence by remember { mutableStateOf("") }
    var isScanning by remember { mutableStateOf(false) }

    // 1. Initialize the Detector
    val detector = remember { DeepfakeDetector(context) }

    // Clean up memory when leaving screen
    DisposableEffect(Unit) {
        onDispose { detector.close() }
    }

    val mediaPicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            selectedUri = uri
        }
    }

    // 2. Listen for URI changes and trigger AI
    LaunchedEffect(selectedUri) {
        selectedUri?.let { uri ->
            isScanning = true
            verdict = "Scanning Pixels..."

            // RUN IN BACKGROUND
            kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                val result = detector.analyzeVideo(context, uri)
                // UPDATE UI ON MAIN THREAD
                kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Main) {
                    verdict = result.first
                    confidence = result.second
                    isScanning = false
                }
            }
        }
    }

    Column(Modifier.fillMaxSize().padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
        Text("Local Edge Scan", style = MaterialTheme.typography.headlineSmall, color = Color.Cyan)
        Spacer(Modifier.height(20.dp))

        Box(Modifier.size(200.dp).background(Color.DarkGray, RoundedCornerShape(12.dp)), contentAlignment = Alignment.Center) {
            if (selectedUri != null) {
                AsyncImage(model = selectedUri, contentDescription = null, contentScale = ContentScale.Crop)
            } else {
                Icon(Icons.Default.Add, "add", tint = Color.White)
            }
            if (isScanning) {
                CircularProgressIndicator(color = Color.Cyan)
            }
        }

        Spacer(Modifier.height(20.dp))
        Button(onClick = { mediaPicker.launch("video/*") }, enabled = !isScanning) {
            Text(if (isScanning) "Analyzing..." else "Select Video")
        }

        if (confidence.isNotEmpty()) {
            Spacer(Modifier.height(30.dp))
            Text("Verdict: $verdict", fontSize = 26.sp, fontWeight = FontWeight.Bold, color = if(verdict == "Fake") Color.Red else Color.Green)
            Text("Confidence: $confidence", fontSize = 16.sp, color = Color.Gray)
        }
    }
}
@Composable
fun AboutScreen() {
    var inputText by remember { mutableStateOf("") }
    val scope = rememberCoroutineScope()
    val messages = remember { mutableStateListOf(ChatMessageData("Skyra Fact-Check Online.", false)) }

    Column(Modifier.fillMaxSize()) {
        Text("Fact Check Assistant", Modifier.padding(16.dp), style = MaterialTheme.typography.titleLarge, color = Color.Cyan)
        LazyColumn(Modifier.weight(1f).padding(8.dp)) {
            items(messages) { msg -> ChatMessage(msg.text, msg.isUser) }
        }
        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                value = inputText,
                onValueChange = { inputText = it },
                modifier = Modifier.weight(1f),
                shape = CircleShape,
                placeholder = { Text("Paste claim here...") }
            )
            IconButton(onClick = {
                if (inputText.isNotBlank()) {
                    val claim = inputText
                    messages.add(ChatMessageData(claim, true))
                    inputText = ""
                    messages.add(ChatMessageData("Consulting Laptop LLM... 🔍", false))
                    scope.launch {
                        try {
                            val response = RetrofitInstance.factCheckApi.verifyClaim(FactCheckRequest(claim))
                            messages.removeAt(messages.size - 1)
                            messages.add(ChatMessageData(response.result, false))
                        } catch (e: Exception) {
                            messages.removeAt(messages.size - 1)
                            messages.add(ChatMessageData("Error: Check Laptop Connection.", false))
                        }
                    }
                }
            }) {
                Icon(Icons.Default.Send, "send", tint = MaterialTheme.colorScheme.primary)
            }
        }
    }
}

@Composable
fun ChatMessage(text: String, isUser: Boolean) {
    Row(Modifier.fillMaxWidth().padding(4.dp), horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start) {
        Surface(
            color = if (isUser) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.secondaryContainer,
            shape = RoundedCornerShape(12.dp)
        ) {
            Text(text, Modifier.padding(12.dp))
        }
    }
}

data class ChatMessageData(val text: String, val isUser: Boolean)