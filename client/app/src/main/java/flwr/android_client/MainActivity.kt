package flwr.android_client

import android.icu.text.SimpleDateFormat
import android.os.Bundle
import android.text.TextUtils
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.util.Patterns
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import dev.flower.flower_tflite.FlowerClient
import dev.flower.flower_tflite.FlowerServiceRunnable
import dev.flower.flower_tflite.SampleSpec
import dev.flower.flower_tflite.createFlowerService
import dev.flower.flower_tflite.helpers.loadMappedAssetFile
import dev.flower.flower_tflite.helpers.negativeLogLikelihoodLoss
import dev.flower.flower_tflite.helpers.placeholderAccuracy
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import java.util.*
import kotlin.random.Random

@Suppress("DEPRECATION")
class MainActivity : AppCompatActivity() {
    private val scope = MainScope()
    lateinit var flowerClient: FlowerClient<Float2DArray, FloatArray>
    lateinit var flowerServiceRunnable: FlowerServiceRunnable<Float2DArray, FloatArray>
    private lateinit var ip: EditText
    private lateinit var port: EditText
    private lateinit var evaluateButton: Button
    private lateinit var trainButton: Button
    private lateinit var resultText: TextView
    private lateinit var deviceId: EditText

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        resultText = findViewById(R.id.grpc_response_text)
        resultText.movementMethod = ScrollingMovementMethod()
        deviceId = findViewById(R.id.device_id_edit_text)
        ip = findViewById(R.id.serverIP)
        port = findViewById(R.id.serverPort)
        evaluateButton = findViewById(R.id.evaluate)
        trainButton = findViewById(R.id.trainFederated)
        createFlowerClient()
    }

    private fun createFlowerClient() {
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
        deviceId.isEnabled = false
        evaluateButton.isEnabled = false
        scope.launch {
            evaluateInBackground()
        }
    }

    suspend fun evaluateInBackground() {
        val result = runWithStacktraceOr("Failed to evaluate.") {
            for (_i in 0..100) {
                flowerClient.addSample(
                    Array(7) { FloatArray(8) { Random.nextFloat() } },
                    floatArrayOf(Random.nextFloat()),
                    false
                )
            }
            Log.d(TAG, "test samples: ${flowerClient.testSamples}")
            val (loss, _) = flowerClient.evaluate()
            "Evaluation loss is $loss."
        }
        runOnUiThread {
            setResultText(result)
            trainButton.isEnabled = true
        }
    }

    fun runGrpc(@Suppress("UNUSED_PARAMETER") view: View) {
        val host = ip.text.toString()
        val portStr = port.text.toString()
        if (TextUtils.isEmpty(host) || TextUtils.isEmpty(portStr) || !Patterns.IP_ADDRESS.matcher(
                host
            ).matches()
        ) {
            Toast.makeText(
                this, "Please enter the correct IP and port of the FL server", Toast.LENGTH_LONG
            ).show()
        } else {
            val port = if (TextUtils.isEmpty(portStr)) 0 else portStr.toInt()
            scope.launch {
                runWithStacktrace {
                    runGrpcInBackground(host, port)
                }
            }
            hideKeyboard()
            ip.isEnabled = false
            this.port.isEnabled = false
            trainButton.isEnabled = false
            setResultText("Started training.")
        }
    }

    suspend fun runGrpcInBackground(host: String, port: Int) {
        val address = "dns:///$host:$port"
        val result = runWithStacktraceOr("Failed to connect to the FL server \n") {
            flowerServiceRunnable = createFlowerService(address, false, flowerClient) {
                runOnUiThread {
                    setResultText(it)
                }
            }
            "Connection to the FL server successful \n"
        }
        runOnUiThread {
            setResultText(result)
            trainButton.isEnabled = false
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
