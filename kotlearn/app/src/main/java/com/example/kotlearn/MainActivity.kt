package com.example.kotlearn

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.animation.core.*
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
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
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
        setContent {
            KotlearnTheme {
                val navController = rememberNavController()

                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    // UPDATED: Added slide transitions to the NavHost
                    NavHost(
                        navController = navController,
                        startDestination = "home",
                        modifier = Modifier.padding(innerPadding),
                        enterTransition = { slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Left, animationSpec = tween(500)) },
                        exitTransition = { slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Left, animationSpec = tween(500)) },
                        popEnterTransition = { slideIntoContainer(AnimatedContentTransitionScope.SlideDirection.Right, animationSpec = tween(500)) },
                        popExitTransition = { slideOutOfContainer(AnimatedContentTransitionScope.SlideDirection.Right, animationSpec = tween(500)) }
                    ) {
                        // 1. HOME SCREEN
                        composable("home") {
                            GreetingImage(
                                message = "Truth Seeker",
                                onScanClicked = { navController.navigate("scan_options") },
                                onAboutClicked = { navController.navigate("about") }
                            )
                        }

                        // 2. SCAN OPTIONS
                        composable("scan_options") {
                            ScanTypeScreen(
                                onQuickClick = { navController.navigate("quick_scan") },
                                onDeepClick = { navController.navigate("deep_upload") }
                            )
                        }

                        // 3. QUICK SCAN
                        composable("quick_scan") {
                            QuickScanScreen(navController)
                        }

                        // 4. DEEP SCAN UPLOAD
                        composable("deep_upload") {
                            DeepUploadScreen(navController)
                        }

                        // 5. DEEP ANALYSIS RESULT
                        composable(
                            route = "analysis_result/{imageUri}",
                            arguments = listOf(navArgument("imageUri") { type = NavType.StringType })
                        ) { backStackEntry ->
                            val imageUriString = backStackEntry.arguments?.getString("imageUri")
                            AnalysisResultScreen(imageUriString)
                        }

                        // 6. FACT CHECK
                        composable("about") {
                            AboutScreen()
                        }
                    }
                }
            }
        }
    }
}

// ==========================================
// ANIMATION HELPERS (NEW!)
// ==========================================

// 1. Bouncing Button: Shrinks when pressed
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

    // Animate scale: 1f (normal) -> 0.95f (pressed)
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

// 2. Typewriter Text: Types out string character by character
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
            delay(100) // Speed of typing
            visibleText = text.substring(0, index + 1)
        }
    }

    Text(text = visibleText, modifier = modifier, style = style)
}

// ==========================================
// SCREEN 1: HOME COMPONENTS
// ==========================================

@Composable
fun GreetingImage(
    message: String,
    modifier: Modifier = Modifier,
    onScanClicked: () -> Unit,
    onAboutClicked: () -> Unit
) {
    // Ensure "cyberbg" exists in res/drawable
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
    // Fade in animation for the buttons
    var buttonsVisible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        delay(1000) // Wait for title to type a bit
        buttonsVisible = true
    }

    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(60.dp))

        // ANIMATION: Typewriter effect for title
        TypewriterText(
            text = message,
            style = LocalTextStyle.current.copy(
                fontSize = 50.sp, // Slightly smaller to fit typing
                color = Color.Cyan,
                lineHeight = 60.sp,
                textAlign = TextAlign.Center,
                fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace // Cyber font look
            )
        )

        Spacer(modifier = Modifier.weight(1f))

        // Ensure "cyverlogo" exists in res/drawable
        Image(
            painter = painterResource(id = R.drawable.cyverlogo),
            contentDescription = null,
            modifier = Modifier
                .size(250.dp)
                .clip(RoundedCornerShape(16.dp)),
            contentScale = ContentScale.Crop
        )

        Spacer(modifier = Modifier.weight(1f))

        // ANIMATION: Buttons slide up
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

// ==========================================
// SCREEN 2: SCAN OPTIONS
// ==========================================
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

// ==========================================
// SCREEN 3: QUICK SCAN (Animated)
// ==========================================
@Composable
fun QuickScanScreen(navController: NavController) {
    var screenState by remember { mutableStateOf("upload") }
    var selectedUri by remember { mutableStateOf<Uri?>(null) }

    // Pulse animation for loading
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val pulseScale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 1.2f,
        animationSpec = infiniteRepeatable(
            animation = tween(800),
            repeatMode = RepeatMode.Reverse
        ), label = "pulse"
    )

    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = { uri -> selectedUri = uri }
    )

    LaunchedEffect(screenState) {
        if (screenState == "loading") {
            delay(2000)
            screenState = "result"
        }
    }

    Surface(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // CROSSFADE: Smoothly switch between states
            AnimatedContent(
                targetState = screenState,
                transitionSpec = {
                    fadeIn(animationSpec = tween(600)) togetherWith fadeOut(animationSpec = tween(600))
                }, label = "scanState"
            ) { targetState ->
                Column(horizontalAlignment = Alignment.CenterHorizontally) {

                    if (targetState == "upload") {
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
                                AsyncImage(
                                    model = selectedUri,
                                    contentDescription = null,
                                    modifier = Modifier.fillMaxSize().clip(RoundedCornerShape(16.dp)),
                                    contentScale = ContentScale.Crop
                                )
                            } else {
                                Icon(Icons.Default.Add, contentDescription = null, tint = Color.Gray, modifier = Modifier.size(40.dp))
                            }
                        }

                        Spacer(modifier = Modifier.height(20.dp))
                        BouncingButton(onClick = { photoPickerLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }) {
                            Text("Select Image")
                        }
                        Spacer(modifier = Modifier.height(20.dp))
                        BouncingButton(
                            onClick = { screenState = "loading" },
                            enabled = selectedUri != null,
                            modifier = Modifier.fillMaxWidth(0.6f)
                        ) {
                            Text("Analyze Now")
                        }
                    }

                    else if (targetState == "loading") {
                        // ANIMATION: Pulsing Icon
                        Box(
                            modifier = Modifier
                                .scale(pulseScale) // Applies the pulse
                                .size(100.dp)
                                .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.2f), CircleShape)
                                .border(2.dp, MaterialTheme.colorScheme.primary, CircleShape),
                            contentAlignment = Alignment.Center
                        ) {
                            Text("AI", color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)
                        }
                        Spacer(modifier = Modifier.height(30.dp))
                        Text("Scanning pixels...", fontSize = 18.sp, color = Color.Gray)
                    }

                    else if (targetState == "result") {
                        // ANIMATION: Spring Pop for the checkmark
                        var iconVisible by remember { mutableStateOf(false) }
                        LaunchedEffect(Unit) { iconVisible = true }

                        AnimatedVisibility(
                            visible = iconVisible,
                            enter = scaleIn(animationSpec = spring(dampingRatio = Spring.DampingRatioMediumBouncy, stiffness = Spring.StiffnessLow))
                        ) {
                            Icon(
                                imageVector = Icons.Default.CheckCircle,
                                contentDescription = null,
                                tint = Color(0xFF4CAF50),
                                modifier = Modifier.size(100.dp)
                            )
                        }

                        Spacer(modifier = Modifier.height(24.dp))
                        Text("Analysis Complete", fontSize = 22.sp, fontWeight = FontWeight.Bold)
                        Spacer(modifier = Modifier.height(32.dp))

                        Card(
                            elevation = CardDefaults.cardElevation(4.dp),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp)
                        ) {
                            Column(modifier = Modifier.padding(20.dp)) {
                                VerdictRow(label = "Authenticity", value = "Authentic", valueColor = Color(0xFF4CAF50))
                                HorizontalDivider(modifier = Modifier.padding(vertical = 12.dp))
                                VerdictRow(label = "Confidence", value = "98.4%", valueColor = MaterialTheme.colorScheme.onSurface)
                                HorizontalDivider(modifier = Modifier.padding(vertical = 12.dp))
                                VerdictRow(label = "Media Type", value = "Image (JPEG)", valueColor = MaterialTheme.colorScheme.onSurface)
                            }
                        }

                        Spacer(modifier = Modifier.height(40.dp))
                        BouncingButton(onClick = {
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

@Composable
fun VerdictRow(label: String, value: String, valueColor: Color) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, fontSize = 18.sp, color = Color.Gray)
        Text(value, fontSize = 18.sp, fontWeight = FontWeight.Bold, color = valueColor)
    }
}

// ==========================================
// SCREEN 4: DEEP UPLOAD
// ==========================================
@Composable
fun DeepUploadScreen(navController: NavController) {
    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }

    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = { uri -> selectedImageUri = uri }
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
            if (selectedImageUri != null) {
                AsyncImage(
                    model = selectedImageUri,
                    contentDescription = null,
                    modifier = Modifier
                        .fillMaxSize()
                        .clip(RoundedCornerShape(16.dp)),
                    contentScale = ContentScale.Crop
                )
            } else {
                Text("No Media Selected", color = Color.Gray)
            }
        }

        Spacer(modifier = Modifier.height(30.dp))

        BouncingButton(onClick = {
            photoPickerLauncher.launch(
                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
            )
        }) {
            Text("Choose File")
        }

        Spacer(modifier = Modifier.height(16.dp))

        BouncingButton(
            onClick = {
                if (selectedImageUri != null) {
                    val encodedUri = URLEncoder.encode(selectedImageUri.toString(), StandardCharsets.UTF_8.toString())
                    navController.navigate("analysis_result/$encodedUri")
                }
            },
            enabled = selectedImageUri != null,
            containerColor = MaterialTheme.colorScheme.primary
        ) {
            Text("Start Deep Analysis")
        }
    }
}

// ==========================================
// SCREEN 5: ANALYSIS RESULT
// ==========================================
@Composable
fun AnalysisResultScreen(imageUriString: String?) {
    Row(modifier = Modifier.fillMaxSize()) {
        // --- LEFT SIDE (50%) ---
        Column(
            modifier = Modifier
                .weight(1f)
                .fillMaxHeight()
                .border(width = 1.dp, color = Color.LightGray)
        ) {
            // Left Top: Image
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .background(Color.Black),
                contentAlignment = Alignment.Center
            ) {
                if (imageUriString != null) {
                    AsyncImage(
                        model = imageUriString,
                        contentDescription = "Analyzed Media",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Fit
                    )
                    // Animated Play Button Overlay
                    val infiniteTransition = rememberInfiniteTransition(label = "playPulse")
                    val alpha by infiniteTransition.animateFloat(
                        initialValue = 0.5f, targetValue = 1f,
                        animationSpec = infiniteRepeatable(tween(1000), RepeatMode.Reverse), label = "alpha"
                    )

                    Icon(
                        imageVector = Icons.Default.PlayArrow,
                        contentDescription = "Play",
                        tint = Color.White.copy(alpha = alpha),
                        modifier = Modifier.size(64.dp)
                    )
                }
            }

            // Left Bottom: Bulletin Points
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
                BulletPoint("Facial landmarks show 92% consistency.")
                BulletPoint("Lighting artifacts detected in background.")
                BulletPoint("Audio frequency matches human vocal range.")
                BulletPoint("Deepfake Probability: LOW")
            }
        }

        // --- RIGHT SIDE (50%): Chat ---
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
                item { ChatMessage("I have analyzed the frame by frame breakdown.", isUser = false) }
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
}

@Composable
fun BulletPoint(text: String) {
    Row(modifier = Modifier.padding(vertical = 4.dp)) {
        Text("•", fontSize = 20.sp, modifier = Modifier.padding(end = 8.dp))
        Text(text, fontSize = 16.sp)
    }
}

// ==========================================
// SCREEN 6: FACT CHECK / ABOUT
// ==========================================
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
                    // Animated Entry for Chat Bubbles
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