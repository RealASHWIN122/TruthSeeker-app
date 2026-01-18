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
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
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
                // We create the navigation controller here
                val navController = rememberNavController()

                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    // NavHost defines the screens and how to move between them
                    NavHost(
                        navController = navController,
                        startDestination = "home",
                        modifier = Modifier.padding(innerPadding)
                    ) {
                        // SCREEN 1: The Home Screen (Your original UI)
                        composable("home") {
                            GreetingImage(
                                message = "Deepfake Detection",
                                from = "Truth Seeker",
                                onContinueClicked = {
                                    navController.navigate("detection")
                                }
                            )
                        }

                        // SCREEN 2: The New Detection Page
                        composable("detection") {
                            DetectionScreen()
                        }
                    }
                }
            }
        }
    }
}

// --- The New Detection Screen ---
@Composable
fun DetectionScreen() {
    // 1. State to hold the selected image URI
    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }

    // 2. The Launcher that opens the gallery
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

            // 3. Display the Image (or a placeholder if null)
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
                // Placeholder Text
                Text(
                    text = "No image selected",
                    fontSize = 18.sp,
                    color = androidx.compose.ui.graphics.Color.Gray
                )
            }

            Spacer(modifier = Modifier.height(32.dp))

            // 4. Buttons Row
            Row(
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Button 1: Upload
                OutlinedButton(
                    onClick = {
                        // Launch the photo picker (Images Only)
                        photoPickerLauncher.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                        )
                    }
                ) {
                    Text("Select Image")
                }

                // Button 2: Analyze (Enabled only if image is selected)
                Button(
                    onClick = {
                        // TODO: Connect to your Deepfake Detection Model/Backend
                    },
                    enabled = selectedImageUri != null
                ) {
                    Text("Analyze")
                }
            }
        }
    }
}

// --- Your Original Composables ---

@Composable
fun GreetingText(
    from: String,
    message: String,
    modifier: Modifier = Modifier,
    onContinueClicked: () -> Unit
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = message,
            fontSize = 66.sp,
            lineHeight = 70.sp,
            textAlign = TextAlign.Center
        )

        Button(
            onClick = onContinueClicked,
            modifier = Modifier
                .padding(16.dp)
                .align(Alignment.End)
        ) {
            Text(text = from, fontSize = 20.sp)
        }
    }
}

@Composable
fun GreetingImage(
    message: String,
    from: String,
    modifier: Modifier = Modifier,
    onContinueClicked: () -> Unit
) {
    val image = painterResource(R.drawable.androidparty)
    Box(modifier) {
        Image(
            painter = image,
            contentDescription = null,
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize(),
            alpha = 0.5f
        )
        GreetingText(
            message = message,
            from = from,
            modifier = Modifier
                .fillMaxSize()
                .padding(8.dp),
            onContinueClicked = onContinueClicked
        )
    }
}

@Preview(showBackground = true)
@Composable
fun BirthdayCardPreview() {
    KotlearnTheme {
        GreetingImage(
            message = "Deepfake Detection",
            from = "Truth Seeker",
            onContinueClicked = {}
        )
    }
}






