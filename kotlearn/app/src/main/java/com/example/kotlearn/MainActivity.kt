package com.example.kotlearn

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.* // Import all layout components
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.* // Import all Material3 components
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
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
import java.net.URLEncoder
import java.nio.charset.StandardCharsets

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            KotlearnTheme {
                // Main Navigation Controller
                val navController = rememberNavController()

                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    NavHost(
                        navController = navController,
                        startDestination = "home",
                        modifier = Modifier.padding(innerPadding)
                    ) {
                        // 1. HOME SCREEN
                        composable("home") {
                            GreetingImage(
                                message = "Truth Seeker",
                                onScanClicked = { navController.navigate("scan_options") },
                                onAboutClicked = { navController.navigate("about") }
                            )
                        }

                        // 2. SCAN OPTIONS SCREEN (Quick vs Deep)
                        composable("scan_options") {
                            ScanTypeScreen(navController)
                        }

                        // 3. DEEP SCAN UPLOAD SCREEN
                        composable("deep_upload") {
                            DeepUploadScreen(navController)
                        }

                        // 4. ANALYSIS RESULT SCREEN (The Complex Layout)
                        // We accept a URI argument to show the image selected
                        composable(
                            route = "analysis_result/{imageUri}",
                            arguments = listOf(navArgument("imageUri") { type = NavType.StringType })
                        ) { backStackEntry ->
                            val imageUriString = backStackEntry.arguments?.getString("imageUri")
                            AnalysisResultScreen(imageUriString)
                        }

                        // 5. ABOUT SCREEN
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
// SCREEN 2: SCAN OPTIONS (Quick / Deep)
// ==========================================
@Composable
fun ScanTypeScreen(navController: NavController) {
    Surface(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text("Select Scan Type", fontSize = 30.sp, fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(40.dp))

            Button(
                onClick = { /* TODO: Quick Logic */ },
                modifier = Modifier.width(200.dp).height(60.dp)
            ) {
                Text("Quick Scan", fontSize = 20.sp)
            }

            Spacer(modifier = Modifier.height(20.dp))

            Button(
                onClick = { navController.navigate("deep_upload") },
                modifier = Modifier.width(200.dp).height(60.dp),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary)
            ) {
                Text("Deep Scan", fontSize = 20.sp)
            }
        }
    }
}

// ==========================================
// SCREEN 3: DEEP UPLOAD
// ==========================================
@Composable
fun DeepUploadScreen(navController: NavController) {
    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }

    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = { uri -> selectedImageUri = uri }
    )

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
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
                    modifier = Modifier.fillMaxSize().clip(RoundedCornerShape(16.dp)),
                    contentScale = ContentScale.Crop
                )
            } else {
                Text("No Media Selected", color = Color.Gray)
            }
        }

        Spacer(modifier = Modifier.height(30.dp))

        Button(onClick = {
            photoPickerLauncher.launch(
                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
            )
        }) {
            Text("Choose File")
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = {
                if (selectedImageUri != null) {
                    // Encode URI to pass it safely in navigation
                    val encodedUri = URLEncoder.encode(selectedImageUri.toString(), StandardCharsets.UTF_8.toString())
                    navController.navigate("analysis_result/$encodedUri")
                }
            },
            enabled = selectedImageUri != null,
            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
        ) {
            Text("Start Deep Analysis")
        }
    }
}

// ==========================================
// SCREEN 4: ANALYSIS RESULT (Split Layout)
// ==========================================
@Composable
fun AnalysisResultScreen(imageUriString: String?) {
    Row(modifier = Modifier.fillMaxSize()) {
        // --- LEFT SIDE (50% width) ---
        Column(
            modifier = Modifier
                .weight(1f)
                .fillMaxHeight()
                .border(width = 1.dp, color = Color.LightGray)
        ) {
            // Left Top: Image/Video Display (50% height of Left side)
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
                    // Fake Video Play Button Overlay
                    Icon(
                        imageVector = Icons.Default.PlayArrow,
                        contentDescription = "Play",
                        tint = Color.White.copy(alpha = 0.7f),
                        modifier = Modifier.size(64.dp)
                    )
                }
            }

            // Left Bottom: Bulletin Points (50% height of Left side)
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

        // --- RIGHT SIDE (50% width): Chat Interface ---
        Column(
            modifier = Modifier
                .weight(1f)
                .fillMaxHeight()
                .padding(8.dp)
        ) {
            Text("AI Assistant", fontSize = 20.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(bottom = 8.dp))

            // Chat Messages Area
            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .background(Color.White, RoundedCornerShape(8.dp))
                    .border(1.dp, Color.LightGray, RoundedCornerShape(8.dp)),
                reverseLayout = true // Start from bottom
            ) {
                item { ChatMessage("Is there anything specific you want to verify?", isUser = false) }
                item { ChatMessage("I have analyzed the frame by frame breakdown.", isUser = false) }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Input Area
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

@Composable
fun ChatMessage(text: String, isUser: Boolean) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(8.dp),
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

// ==========================================
// EXISTING COMPONENTS (Unchanged)
// ==========================================

@Composable
fun AboutScreen() {
    Surface(modifier = Modifier.fillMaxSize()) {
        Box(contentAlignment = Alignment.Center) {
            Text(text = "About Page", fontSize = 24.sp)
        }
    }
}

@Composable
fun GreetingImage(
    message: String,
    modifier: Modifier = Modifier,
    onScanClicked: () -> Unit,
    onAboutClicked: () -> Unit
) {
    // Make sure you have R.drawable.cyberbg
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
            modifier = Modifier.fillMaxSize().padding(8.dp),
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
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(60.dp))
        Text(
            text = message,
            fontSize = 66.sp,
            color = Color.Cyan,
            lineHeight = 70.sp,
            textAlign = TextAlign.Center
        )
        Spacer(modifier = Modifier.weight(1f))

        // Make sure you have R.drawable.cyverlogo
        Image(
            painter = painterResource(id = R.drawable.cyverlogo),
            contentDescription = null,
            modifier = Modifier.size(250.dp).clip(RoundedCornerShape(16.dp)),
            contentScale = ContentScale.Crop
        )
        Spacer(modifier = Modifier.weight(1f))

        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(bottom = 80.dp)
        ) {
            Button(onClick = onScanClicked, modifier = Modifier.width(200.dp)) {
                Text(text = "Scan", fontSize = 24.sp)
            }
            Spacer(modifier = Modifier.height(20.dp))
            Button(onClick = onAboutClicked, modifier = Modifier.width(200.dp)) {
                Text(text = "Fact Check", fontSize = 24.sp)
            }
        }
    }
}