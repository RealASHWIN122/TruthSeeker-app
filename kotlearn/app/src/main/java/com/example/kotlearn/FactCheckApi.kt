package com.example.kotlearn

import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*
import java.util.concurrent.TimeUnit

// --- DATA MODELS ---
data class FactCheckRequest(val claim_text: String)
data class FactCheckResponse(val result: String, val evidence_used: String)

data class ForensicMarker(
    val timestampMs: Long,
    val summary: String, // Maps to the AI's dynamic label
    val gridIndices: List<Int>
)

data class SkyraResponse(
    val status: String,
    val analysis: String?,
    val markers: List<ForensicMarker>?,
    val message: String?
)

data class DeepScanResponse(
    val verdict: String,
    val confidence: String,
    val details: String,
    val markers: List<ForensicMarker> = emptyList()
)

// Data class specifically for your Colab Audio Model response
data class AudioScanResponse(
    val status: String,
    val verdict: String,
    val confidence: String,
    val details: String,
    val markers: List<ForensicMarker>
)

// --- API INTERFACES ---
interface FactCheckService {
    @POST("verify")
    suspend fun verifyClaim(@Body request: FactCheckRequest): FactCheckResponse
}

interface SkyraService {
    @Multipart
    @POST("analyze")
    suspend fun skyraDeepScan(@Part video: MultipartBody.Part): SkyraResponse
}

interface AudioService {
    @Multipart
    @POST("analyze_audio")
    suspend fun analyzeAudio(@Part audio: MultipartBody.Part): AudioScanResponse

    // ADD THIS NEW ROUTE FOR COLAB VIDEO!
    @Multipart
    @POST("analyze_video")
    suspend fun analyzeVideo(@Part video: MultipartBody.Part): AudioScanResponse
}

// --- RETROFIT HUB ---
object RetrofitInstance {
    // Port 8000 for MI300X Signal Stability
    private const val AMD_URL = "http://134.199.203.201:8000/"
    private const val LAPTOP_URL = "http://10.58.55.15:8000/"
    private const val COLAB_AUDIO_URL = "https://late-ears-boil.loca.lt/" // <-- Add the slash right here!

    // Extended Timeouts for 16-frame Qwen2.5-VL Inference
    private val okHttpClient = OkHttpClient.Builder()
        .connectTimeout(120, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS)
        .build()

    val factCheckApi: FactCheckService by lazy {
        Retrofit.Builder()
            .baseUrl(LAPTOP_URL)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(FactCheckService::class.java)
    }

    val skyraApi: SkyraService by lazy {
        Retrofit.Builder()
            .baseUrl(AMD_URL)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(SkyraService::class.java)
    }

    val audioApi: AudioService by lazy {
        Retrofit.Builder()
            .baseUrl(COLAB_AUDIO_URL)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(AudioService::class.java)
    }
}