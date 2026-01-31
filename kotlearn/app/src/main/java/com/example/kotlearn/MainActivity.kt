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
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import coil.compose.AsyncImage
import com.example.kotlearn.ui.theme.KotlearnTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            KotlearnTheme {
                val navController = rememberNavController()

                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    NavHost(
                        navController = navController,
                        startDestination = "home",
                        modifier = Modifier.padding(innerPadding)
                    ) {
                        // SCREEN 1: Home
                        composable("home") {
                            GreetingImage(
                                message = "Truth Seeker",
                                onScanClicked = { navController.navigate("detection") },
                                onAboutClicked = { navController.navigate("about") }
                            )
                        }

                        // SCREEN 2: Detection
                        composable("detection") {
                            DetectionScreen()
                        }

                        // SCREEN 3: About
                        composable("about") {
                            AboutScreen()
                        }
                    }
                }
            }
        }
    }
}

// --- The New Empty Page ---
@Composable
fun AboutScreen() {
    Surface(modifier = Modifier.fillMaxSize()) {
        Box(contentAlignment = Alignment.Center) {
            Text(text = "About Page (Empty)", fontSize = 24.sp)
        }
    }
}

// --- The Detection Screen ---
@Composable
fun DetectionScreen() {
    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = { uri -> selectedImageUri = uri }
    )

    Surface(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (selectedImageUri != null) {
                AsyncImage(
                    model = selectedImageUri,
                    contentDescription = "Selected Image",
                    modifier = Modifier
                        .size(300.dp)
                        .clip(RoundedCornerShape(16.dp)),
                    contentScale = ContentScale.Crop
                )
            } else {
                Text(
                    text = "No image selected",
                    fontSize = 18.sp,
                    color = Color.Gray // Cleaned up
                )
            }
            Spacer(modifier = Modifier.height(32.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                OutlinedButton(
                    onClick = {
                        photoPickerLauncher.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                        )
                    }
                ) {
                    Text("Select Image")
                }
                Button(
                    onClick = { /* TODO: Analyze Logic */ },
                    enabled = selectedImageUri != null
                ) {
                    Text("Analyze")
                }
            }
        }
    }
}

// --- UPDATED: Home Screen UI ---
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
        // Top Spacer
        Spacer(modifier = Modifier.height(60.dp))

        // 1. The Title Text
        Text(
            text = message,
            fontSize = 66.sp,
            color = Color.Cyan,
            lineHeight = 70.sp,
            textAlign = TextAlign.Center
        )

        // Flexible Spacer to push image to center
        Spacer(modifier = Modifier.weight(1f))

        // 2. The Middle Image
        // Ensure "cyverlogo" exists in res/drawable, otherwise rename to "cyberlogo"
        Image(
            painter = painterResource(id = R.drawable.cyverlogo),
            contentDescription = null,
            modifier = Modifier
                .size(250.dp)
                .clip(RoundedCornerShape(16.dp)),
            contentScale = ContentScale.Crop
        )

        // Flexible Spacer to push buttons down
        Spacer(modifier = Modifier.weight(1f))

        // 3. The Buttons
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(bottom = 80.dp) // Adjusted padding
        ) {
            Button(
                onClick = onScanClicked,
                modifier = Modifier.width(200.dp)
            ) {
                Text(text = "Scan", fontSize = 24.sp)
            }

            Spacer(modifier = Modifier.height(20.dp))

            Button(
                onClick = onAboutClicked,
                modifier = Modifier.width(200.dp)
            ) {
                Text(text = "Fact Check", fontSize = 24.sp)
            }
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
    // Ensure "cyberbg" exists in res/drawable
    val image = painterResource(R.drawable.cyberbg)

    Box(modifier) {
        Image(
            painter = image,
            contentDescription = null,
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize(),
            alpha = 0.9f // FIXED: Must be between 0.0f (transparent) and 1.0f (opaque)
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

@Preview(showBackground = true)
@Composable
fun BirthdayCardPreview() {
    KotlearnTheme {
        GreetingImage(
            message = "Truth Seeker",
            onScanClicked = {},
            onAboutClicked = {}
        )
    }
}