package flwr.android_client

import android.icu.text.SimpleDateFormat
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import dev.flower.flower_tflite.FlowerClient
import dev.flower.flower_tflite.SampleSpec
import dev.flower.flower_tflite.helpers.loadMappedAssetFile
import dev.flower.flower_tflite.helpers.negativeLogLikelihoodLoss
import dev.flower.flower_tflite.helpers.placeholderAccuracy
import dev.flower.flower_tflite.stockData
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import java.util.*

class MainActivity : AppCompatActivity() {
    private val scope = MainScope()
    lateinit var flowerClient: FlowerClient<Float2DArray, FloatArray>
    private lateinit var evaluateButton: Button
    private lateinit var trainButton: Button
    private lateinit var resultText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        resultText = findViewById(R.id.grpc_response_text)
        resultText.movementMethod = ScrollingMovementMethod()
        evaluateButton = findViewById(R.id.evaluate)
        evaluateButton.isEnabled = false
        trainButton = findViewById(R.id.start_training)
        trainButton.isEnabled = false
        scope.launch {
            prepareFlowerClient()
        }
    }

    private fun prepareFlowerClient() {
        val buffer = loadMappedAssetFile(this, "fed_mcrnn1.tflite")
        val layersSizes =
            intArrayOf(49152, 2359296, 6144, 393216, 65536, 1024, 491520, 3686400, 7680, 13440, 4)
        val sampleSpec = SampleSpec<Float2DArray, FloatArray>(
            { it.toTypedArray() },
            { it.toTypedArray() },
            { Array(it) { FloatArray(1) } },
            ::negativeLogLikelihoodLoss,
            ::placeholderAccuracy,
        )
        flowerClient = FlowerClient(buffer, layersSizes, sampleSpec)
        val data = stockData()

        for ((bottleneck, label) in data) {
            flowerClient.addSample(bottleneck, floatArrayOf(label), true)
            flowerClient.addSample(bottleneck, floatArrayOf(label), false)
        }
        Log.d(TAG, "Samples: ${flowerClient.testSamples}")
        runOnUiThread {
            evaluateButton.isEnabled = true
            trainButton.isEnabled = true
        }
    }

    fun setResultText(text: String) {
        val dateFormat = SimpleDateFormat("HH:mm:ss", Locale.GERMANY)
        val time = dateFormat.format(Date())
        resultText.append("\n$time   $text")
    }

    suspend fun runWithStacktrace(call: suspend () -> Unit) {
        try {
            call()
        } catch (err: Error) {
            Log.e(TAG, Log.getStackTraceString(err))
        }
    }

    suspend fun <T> runWithStacktraceOr(or: T, call: suspend () -> T): T {
        return try {
            call()
        } catch (err: Error) {
            Log.e(TAG, Log.getStackTraceString(err))
            or
        }
    }

    fun evaluate(@Suppress("UNUSED_PARAMETER") view: View) {
        hideKeyboard()
        setResultText("Evaluating...")
        scope.launch {
            evaluateInBackground()
        }
    }

    suspend fun evaluateInBackground() {
        val result = runWithStacktraceOr("Failed to evaluate.") {
            val (loss, _) = flowerClient.evaluate()
            "Evaluation loss is $loss."
        }
        runOnUiThread {
            setResultText(result)
        }
    }

    fun startTraining(@Suppress("UNUSED_PARAMETER") view: View) {
        scope.launch {
            runWithStacktrace {
                trainInBackground()
            }
        }
        hideKeyboard()
        setResultText("Started training.")
    }

    suspend fun trainInBackground() {
        val result = runWithStacktraceOr("Failed to connect to the FL server \n") {
            flowerClient.fit(3) { runOnUiThread { setResultText("Losses: $it.") } }
            "Training successful \n"
        }
        runOnUiThread {
            setResultText(result)
        }
    }

    fun hideKeyboard() {
        val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
        var view = currentFocus
        if (view == null) {
            view = View(this)
        }
        imm.hideSoftInputFromWindow(view.windowToken, 0)
    }
}

private const val TAG = "MainActivity"

typealias Float2DArray = Array<FloatArray>
