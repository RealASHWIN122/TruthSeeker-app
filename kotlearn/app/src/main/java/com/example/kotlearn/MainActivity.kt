package com.example.kotlearn

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import androidx.navigation.NavController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import coil.compose.AsyncImage
import com.example.kotlearn.ui.theme.KotlearnTheme
import kotlinx.coroutines.delay
import java.net.URLEncoder
import java.nio.charset.StandardCharsets

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        
        handleIntent(intent)

        setContent {
            KotlearnTheme {
                val navController = rememberNavController()

                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    NavHost(
                        navController = navController,
                        startDestination = "home",
                        modifier = Modifier.padding(innerPadding),
                        enterTransition = { slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Left, animationSpec = tween(500)) },
                        exitTransition = { slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Left, animationSpec = tween(500)) },
                        popEnterTransition = { slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Right, animationSpec = tween(500)) },
                        popExitTransition = { slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Right, animationSpec = tween(500)) }
                    ) {
                        composable("home") {
                            GreetingImage(
                                message = "Truth Seeker",
                                onScanClicked = { 
                                    if (!Settings.canDrawOverlays(this@MainActivity)) {
                                        val intent = Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:$packageName"))
                                        startActivity(intent)
                                    } else {
                                        startService(Intent(this@MainActivity, FloatingWidgetService::class.java))
                                    }
                                    navController.navigate("scan_options") 
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
                            DeepUploadScreen(navController)
                        }

                        composable(
                            route = "analysis_result/{imageUri}",
                            arguments = listOf(navArgument("imageUri") { type = NavType.StringType })
                        ) { backStackEntry ->
                            val imageUriString = backStackEntry.arguments?.getString("imageUri")
                            AnalysisResultScreen(imageUriString)
                        }

                        composable("about") {
                            AboutScreen()
                        }
                    }
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: Intent?) {
        val uriStr = intent?.getStringExtra("capturedMediaUri")
        val scanType = intent?.getStringExtra("scanType")
        
        if (uriStr != null && scanType != null) {
            Toast.makeText(this, "Received $scanType scan request for: $uriStr", Toast.LENGTH_LONG).show()
        }
    }
}

@Composable
fun VideoPlayer(uri: Uri, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val exoPlayer = remember {
        ExoPlayer.Builder(context).build().apply {
            setMediaItem(MediaItem.fromUri(uri))
            prepare()
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            exoPlayer.release()
        }
    }

    AndroidView(
        factory = {
            PlayerView(context).apply {
                player = exoPlayer
                useController = true
                setBackgroundColor(android.graphics.Color.BLACK)
            }
        },
        modifier = modifier
    )
}

@Composable
fun HeatmapLayer(modifier: Modifier = Modifier) {
    val rows = 12
    val cols = 12
    
    Canvas(modifier = modifier) {
        val cellWidth = size.width / cols
        val cellHeight = size.height / rows
        
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                val dx = c - 6
                val dy = r - 7
                val distSq = dx * dx + dy * dy
                val intensity = (1.0f - (distSq / 15.0f)).coerceIn(0.0f, 1.0f)
                
                if (intensity > 0.05f) {
                    drawRect(
                        color = Color.Red.copy(alpha = intensity * 0.5f),
                        topLeft = Offset(c * cellWidth, r * cellHeight),
                        size = Size(cellWidth, cellHeight)
                    )
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
    val isPressed by interactionSource.collectIsPressedAsState()

    val scale by animateFloatAsState(
        targetValue = if (isPressed) 0.95f else 1f,
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
fun TypewriterText(
    text: String,
    modifier: Modifier = Modifier,
    style: androidx.compose.ui.text.TextStyle
) {
    var visibleText by remember { mutableStateOf("") }

    LaunchedEffect(text) {
        visibleText = ""
        text.forEachIndexed { index, _ ->
            delay(100)
            visibleText = text.substring(0, index + 1)
        }
    }

    Text(text = visibleText, modifier = modifier, style = style)
}

@Composable
fun GreetingImage(
    message: String,
    modifier: Modifier = Modifier,
    onScanClicked: () -> Unit,
    onAboutClicked: () -> Unit
) {
    val image = painterResource(R.drawable.cyberbg)

    Box(modifier) {
        Image(
            painter = image,
            contentDescription = null,
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize(),
            alpha = 0.9f
        )
        GreetingText(
            message = message,
            modifier = Modifier
                .fillMaxSize()
                .padding(8.dp),
            onScanClicked = onScanClicked,
            onAboutClicked = onAboutClicked
        )
    }
}

@Composable
fun GreetingText(
    message: String,
    modifier: Modifier = Modifier,
    onScanClicked: () -> Unit,
    onAboutClicked: () -> Unit
) {
    var buttonsVisible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        delay(1000)
        buttonsVisible = true
    }

    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(60.dp))

        TypewriterText(
            text = message,
            style = LocalTextStyle.current.copy(
                fontSize = 50.sp,
                color = Color.Cyan,
                lineHeight = 60.sp,
                textAlign = TextAlign.Center,
                fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace
            )
        )

        Spacer(modifier = Modifier.weight(1f))

        Image(
            painter = painterResource(id = R.drawable.cyverlogo),
            contentDescription = null,
            modifier = Modifier
                .size(250.dp)
                .clip(RoundedCornerShape(16.dp)),
            contentScale = ContentScale.Crop
        )

        Spacer(modifier = Modifier.weight(1f))

        AnimatedVisibility(
            visible = buttonsVisible,
            enter = slideInVertically(initialOffsetY = { 100 }) + fadeIn()
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.padding(bottom = 80.dp)
            ) {
                BouncingButton(onClick = onScanClicked, modifier = Modifier.width(200.dp)) {
                    Text(text = "Scan", fontSize = 24.sp)
                }
                Spacer(modifier = Modifier.height(20.dp))
                BouncingButton(onClick = onAboutClicked, modifier = Modifier.width(200.dp)) {
                    Text(text = "Fact Check", fontSize = 24.sp)
                }
            }
        }
    }
}

@Composable
fun ScanTypeScreen(
    onQuickClick: () -> Unit,
    onDeepClick: () -> Unit
) {
    Surface(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text("Select Scan Type", fontSize = 30.sp, fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(40.dp))

            BouncingButton(
                onClick = onQuickClick,
                modifier = Modifier
                    .width(200.dp)
                    .height(60.dp)
            ) {
                Text("Quick Scan", fontSize = 20.sp)
            }

            Spacer(modifier = Modifier.height(20.dp))

            BouncingButton(
                onClick = onDeepClick,
                modifier = Modifier
                    .width(200.dp)
                    .height(60.dp),
                containerColor = MaterialTheme.colorScheme.secondary
            ) {
                Text("Deep Scan", fontSize = 20.sp)
            }
        }
    }
}

@Composable
fun QuickScanScreen(navController: NavController) {
    var screenState by remember { mutableStateOf("upload") }
    var selectedUri by remember { mutableStateOf<Uri?>(null) }

    var verdict by remember { mutableStateOf("Analyzing...") }
    var confidence by remember { mutableStateOf("0%") }
    var mediaType by remember { mutableStateOf("Media") }

    val context = LocalContext.current

    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val pulseScale by infiniteTransition.animateFloat(
        initialValue = 1f, targetValue = 1.2f,
        animationSpec = infiniteRepeatable(tween(800), RepeatMode.Reverse), label = "pulse"
    )

    val mediaPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
        onResult = { uri -> selectedUri = uri }
    )

    LaunchedEffect(screenState) {
        if (screenState == "loading" && selectedUri != null) {
            kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                val detector = DeepfakeDetector(context)
                val result = detector.analyzeVideo(context, selectedUri!!)
                detector.close()

                verdict = result.first
                confidence = result.second
                
                val mime = context.contentResolver.getType(selectedUri!!)
                mediaType = when {
                    mime?.startsWith("video") == true -> "Video"
                    mime?.startsWith("audio") == true -> "Audio"
                    mime?.startsWith("image") == true -> "Image"
                    else -> "Media"
                }

                screenState = "result"
            }
        }
    }

    Surface(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier.fillMaxSize().padding(16.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            AnimatedContent(
                targetState = screenState,
                transitionSpec = { fadeIn(tween(600)) togetherWith fadeOut(tween(600)) },
                label = "scanState"
            ) { targetState ->
                Column(horizontalAlignment = Alignment.CenterHorizontally) {

                    when (targetState) {
                        "upload" -> {
                            Text("Quick Scan Media", fontSize = 24.sp, fontWeight = FontWeight.Bold)
                            Spacer(modifier = Modifier.height(30.dp))

                            Box(
                                modifier = Modifier
                                    .size(200.dp)
                                    .background(Color.LightGray.copy(alpha = 0.3f), RoundedCornerShape(16.dp))
                                    .border(1.dp, Color.Gray, RoundedCornerShape(16.dp)),
                                contentAlignment = Alignment.Center
                            ) {
                                if (selectedUri != null) {
                                    val mimeType = context.contentResolver.getType(selectedUri!!)
                                    when {
                                        mimeType?.startsWith("image") == true -> {
                                            AsyncImage(model = selectedUri, contentDescription = null, modifier = Modifier.clip(RoundedCornerShape(16.dp)), contentScale = ContentScale.Crop)
                                        }
                                        mimeType?.startsWith("audio") == true -> {
                                            Icon(Icons.Default.MusicNote, contentDescription = null, modifier = Modifier.size(64.dp), tint = Color.Black)
                                        }
                                        else -> {
                                            Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(64.dp), tint = Color.Black)
                                        }
                                    }
                                } else {
                                    Icon(Icons.Default.Add, contentDescription = null, tint = Color.Gray, modifier = Modifier.size(40.dp))
                                }
                            }

                            Spacer(modifier = Modifier.height(20.dp))
                            Button(onClick = {
                                mediaPickerLauncher.launch("*/*")
                            }) {
                                Text("Select Media")
                            }
                            Spacer(modifier = Modifier.height(20.dp))
                            Button(
                                onClick = { screenState = "loading" },
                                enabled = selectedUri != null,
                                modifier = Modifier.fillMaxWidth(0.6f)
                            ) {
                                Text("Analyze Now")
                            }
                        }

                        "loading" -> {
                            Box(
                                modifier = Modifier
                                    .scale(pulseScale)
                                    .size(100.dp)
                                    .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.2f), CircleShape)
                                    .border(2.dp, MaterialTheme.colorScheme.primary, CircleShape),
                                contentAlignment = Alignment.Center
                            ) {
                                Text("AI", color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)
                            }
                            Spacer(modifier = Modifier.height(30.dp))
                            Text("Analyzing Media...", fontSize = 18.sp, color = Color.Gray)
                        }

                        "result" -> {
                            val isAuthentic = verdict == "Authentic"
                            val iconColor = if (isAuthentic) Color(0xFF4CAF50) else Color(0xFFFF5252)
                            val iconVector = if (isAuthentic) Icons.Default.CheckCircle else Icons.Default.Warning

                            Icon(
                                imageVector = iconVector,
                                contentDescription = null,
                                tint = iconColor,
                                modifier = Modifier.size(100.dp)
                            )

                            Spacer(modifier = Modifier.height(24.dp))
                            Text("Analysis Complete", fontSize = 22.sp, fontWeight = FontWeight.Bold)

                            Card(
                                elevation = CardDefaults.cardElevation(4.dp),
                                modifier = Modifier.fillMaxWidth().padding(16.dp)
                            ) {
                                Column(modifier = Modifier.padding(20.dp)) {
                                    Text("Verdict: $verdict", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = iconColor)
                                    Spacer(modifier = Modifier.height(8.dp))
                                    Text("Confidence: $confidence", fontSize = 16.sp)
                                    Spacer(modifier = Modifier.height(8.dp))
                                    Text("Media Type: $mediaType", fontSize = 16.sp)
                                }
                            }

                            Spacer(modifier = Modifier.height(40.dp))
                            Button(onClick = {
                                selectedUri = null
                                screenState = "upload"
                            }) {
                                Text("Scan Another")
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun DeepUploadScreen(navController: NavController) {
    var selectedMediaUri by remember { mutableStateOf<Uri?>(null) }
    val context = LocalContext.current

    val mediaPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
        onResult = { uri -> selectedMediaUri = uri }
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("Upload Media for Deep Scan", fontSize = 24.sp, textAlign = TextAlign.Center)

        Spacer(modifier = Modifier.height(30.dp))

        Box(
            modifier = Modifier
                .size(250.dp)
                .background(Color.LightGray.copy(alpha = 0.3f), RoundedCornerShape(16.dp))
                .border(2.dp, Color.Gray, RoundedCornerShape(16.dp)),
            contentAlignment = Alignment.Center
        ) {
            if (selectedMediaUri != null) {
                val mimeType = context.contentResolver.getType(selectedMediaUri!!)
                when {
                    mimeType?.startsWith("image") == true -> {
                        AsyncImage(
                            model = selectedMediaUri,
                            contentDescription = null,
                            modifier = Modifier
                                .fillMaxSize()
                                .clip(RoundedCornerShape(16.dp)),
                            contentScale = ContentScale.Crop
                        )
                    }
                    mimeType?.startsWith("audio") == true -> {
                        Icon(Icons.Default.MusicNote, contentDescription = null, modifier = Modifier.size(100.dp), tint = Color.Gray)
                    }
                    else -> {
                        Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(100.dp), tint = Color.Gray)
                    }
                }
            } else {
                Text("No Media Selected", color = Color.Gray)
            }
        }

        Spacer(modifier = Modifier.height(30.dp))

        BouncingButton(onClick = {
            mediaPickerLauncher.launch("*/*")
        }) {
            Text("Choose Media")
        }

        Spacer(modifier = Modifier.height(16.dp))

        BouncingButton(
            onClick = {
                if (selectedMediaUri != null) {
                    val encodedUri = URLEncoder.encode(selectedMediaUri.toString(), StandardCharsets.UTF_8.toString())
                    navController.navigate("analysis_result/$encodedUri")
                }
            },
            enabled = selectedMediaUri != null,
            containerColor = MaterialTheme.colorScheme.primary
        ) {
            Text("Start Deep Analysis")
        }
    }
}

@Composable
fun AnalysisResultScreen(imageUriString: String?) {
    val context = LocalContext.current
    val uri = imageUriString?.let { Uri.parse(it) }
    val mimeType = uri?.let { context.contentResolver.getType(it) }
    
    var showHeatmap by remember { mutableStateOf(false) }

    Box(modifier = Modifier.fillMaxSize()) {
        Row(modifier = Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .border(width = 1.dp, color = Color.LightGray)
            ) {
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .background(Color.Black),
                    contentAlignment = Alignment.Center
                ) {
                    if (imageUriString != null && uri != null) {
                        when {
                            mimeType?.startsWith("image") == true -> {
                                AsyncImage(
                                    model = imageUriString,
                                    contentDescription = "Analyzed Media",
                                    modifier = Modifier.fillMaxSize(),
                                    contentScale = ContentScale.Fit
                                )
                            }
                            mimeType?.startsWith("audio") == true -> {
                                Icon(Icons.Default.MusicNote, contentDescription = "Audio", tint = Color.White, modifier = Modifier.size(80.dp))
                            }
                            mimeType?.startsWith("video") == true -> {
                                Box(modifier = Modifier.fillMaxSize()) {
                                    VideoPlayer(uri = uri, modifier = Modifier.fillMaxSize())
                                    if (showHeatmap) {
                                        HeatmapLayer(modifier = Modifier.fillMaxSize())
                                    }
                                }
                            }
                            else -> {
                                AsyncImage(
                                    model = imageUriString,
                                    contentDescription = "Analyzed Media",
                                    modifier = Modifier.fillMaxSize(),
                                    contentScale = ContentScale.Fit
                                )
                            }
                        }
                    }
                }

                Column(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .padding(12.dp)
                        .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(8.dp))
                        .padding(12.dp)
                ) {
                    Text("Analysis Report", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    BulletPoint("Signal consistency check: passed.")
                    BulletPoint("Metadata artifacts: none detected.")
                    BulletPoint("AI pattern matching: negative.")
                    BulletPoint("Deepfake Probability: LOW")
                }
            }

            Column(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .padding(8.dp)
            ) {
                Text("AI Assistant", fontSize = 20.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(bottom = 8.dp))

                LazyColumn(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .background(Color.White, RoundedCornerShape(8.dp))
                        .border(1.dp, Color.LightGray, RoundedCornerShape(8.dp)),
                    reverseLayout = true
                ) {
                    item { ChatMessage("Is there anything specific you want to verify?", isUser = false) }
                    item { ChatMessage("I have analyzed the media structure.", isUser = false) }
                }

                Spacer(modifier = Modifier.height(8.dp))

                Row(verticalAlignment = Alignment.CenterVertically) {
                    OutlinedTextField(
                        value = "",
                        onValueChange = {},
                        placeholder = { Text("Ask a question...") },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(24.dp)
                    )
                    IconButton(onClick = { }) {
                        Icon(imageVector = Icons.Default.Send, contentDescription = "Send")
                    }
                }
            }
        }

        if (mimeType?.startsWith("video") == true) {
            Row(
                verticalAlignment = Alignment.CenterVertically, 
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(top = 40.dp, end = 16.dp) // Adjusted padding to move it down
            ) {
                Text("Heatmap", fontSize = 14.sp)
                Switch(checked = showHeatmap, onCheckedChange = { showHeatmap = it })
            }
        }
    }
}

@Composable
fun BulletPoint(text: String) {
    Row(modifier = Modifier.padding(vertical = 4.dp)) {
        Text("•", fontSize = 20.sp, modifier = Modifier.padding(end = 8.dp))
        Text(text, fontSize = 16.sp)
    }
}

@Composable
fun AboutScreen() {
    var inputText by remember { mutableStateOf("") }
    val messages = remember { mutableStateListOf(
        ChatMessageData("Hello! I am your Fact Check AI.", isUser = false),
        ChatMessageData("Paste a news URL or a claim, and I will verify its authenticity.", isUser = false)
    ) }

    Surface(modifier = Modifier.fillMaxSize()) {
        Column(modifier = Modifier.fillMaxSize()) {

            Surface(
                shadowElevation = 4.dp,
                color = MaterialTheme.colorScheme.surface,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = "Fact Checker",
                    modifier = Modifier.padding(16.dp),
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center,
                    color = MaterialTheme.colorScheme.primary
                )
            }

            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp),
                reverseLayout = false
            ) {
                items(messages) { message ->
                    AnimatedVisibility(
                        visible = true,
                        enter = slideInHorizontally(initialOffsetX = { if (message.isUser) 100 else -100 }) + fadeIn()
                    ) {
                        ChatMessage(text = message.text, isUser = message.isUser)
                    }
                }
            }

            Surface(
                shadowElevation = 8.dp,
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    OutlinedTextField(
                        value = inputText,
                        onValueChange = { inputText = it },
                        placeholder = { Text("Enter a claim...") },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(24.dp),
                        maxLines = 3
                    )

                    Spacer(modifier = Modifier.width(8.dp))

                    IconButton(
                        onClick = {
                            if (inputText.isNotBlank()) {
                                messages.add(ChatMessageData(inputText, true))
                                inputText = ""
                            }
                        },
                        modifier = Modifier
                            .size(50.dp)
                            .background(MaterialTheme.colorScheme.primary, CircleShape)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Send,
                            contentDescription = "Send",
                            tint = Color.White
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun ChatMessage(text: String, isUser: Boolean) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        Surface(
            color = if (isUser) MaterialTheme.colorScheme.primary else Color.LightGray,
            shape = RoundedCornerShape(16.dp),
            modifier = Modifier.widthIn(max = 200.dp)
        ) {
            Text(
                text = text,
                modifier = Modifier.padding(12.dp),
                color = if (isUser) Color.White else Color.Black
            )
        }
    }
}

data class ChatMessageData(val text: String, val isUser: Boolean)
