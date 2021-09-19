package com.hophacks.guidedog;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.media.AudioAttributes;
import android.media.Image;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Environment;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Looper;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.os.Bundle;
import android.speech.SpeechRecognizer;
import android.util.Size;
import android.util.Log;
import android.view.View;
import android.view.MotionEvent;
import android.view.Window;
import android.view.WindowManager;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.speech.tts.TextToSpeech;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.common.model.LocalModel;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.objects.DetectedObject;
import com.google.mlkit.vision.objects.ObjectDetection;
import com.google.mlkit.vision.objects.ObjectDetector;
import com.google.mlkit.vision.objects.custom.CustomObjectDetectorOptions;
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions;
import com.hophacks.guidedog.databinding.ActivityMainBinding;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity {

    final Executor executor = Executors.newSingleThreadExecutor();
    private TextView speechText;
    private ImageView micButton;
    private TextView uploadBtn;
    private TextView phonecamBtn;
    private TextView suncamBtn;
    private TextToSpeech textToSpeech;
    private SpeechRecognizer speechRecognizer;
    private PreviewView previewView;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private ImageCapture imageCapture;
    private ConstraintLayout cl_blue_curtain;
    private ObjectDetector objectDetector;
    private MediaPlayer mediaPlayer;
    private Animation rotateLoading;

    private ActivityMainBinding binding;

    final String url = "https://hophack-326223.ue.r.appspot.com/";

//    private static final int CAMERA_PERMISSION_CODE = 100;
//    private static final int MIC_PERMISSION_CODE = 101;
//    private static final int WRITE_CODE = 102;
    private static final int MULTIPLE_PERMISSION_CODE = 123;

    private class ObjectDetectionProcessor implements ImageAnalysis.Analyzer {
        private String[] warnings = { "door", "bannister", "car", "parking meter", "street light", "street sign" };
        private ArrayList<String> seen = new ArrayList<String>();

        @Override
        @androidx.camera.core.ExperimentalGetImage
        public void analyze(ImageProxy imageProxy) {
            Image mediaImage = imageProxy.getImage();
            if (mediaImage != null) {
                InputImage image =
                        InputImage.fromMediaImage(mediaImage, imageProxy.getImageInfo().getRotationDegrees());
                // Pass image to an ML Kit Vision API
                LocalModel localModel = new LocalModel.Builder()
                        .setAssetFilePath("model_mobilenet.tflite")
                        .build();
                CustomObjectDetectorOptions options =
                        new CustomObjectDetectorOptions.Builder(localModel)
                                .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
                                .enableMultipleObjects()
                                .enableClassification()
                                .setClassificationConfidenceThreshold(0.5f)
                                .setMaxPerObjectLabelCount(3)
                                .build();
                objectDetector = ObjectDetection.getClient(options);
                objectDetector.process(image)
                        .addOnSuccessListener(
                                new OnSuccessListener<List<DetectedObject>>() {
                                    @Override
                                    public void onSuccess(List<DetectedObject> detectedObjects) {
                                        // Task completed successfully
                                        for (DetectedObject detectedObject : detectedObjects) {
                                            if (binding.container.getChildCount() > 1) {
                                                binding.container.removeViewAt(1);
                                            }
                                            Rect boundingBox = detectedObject.getBoundingBox();
                                            Integer trackingId = detectedObject.getTrackingId();
                                            String text = "";
                                            if (detectedObject.getLabels().size() != 0) {
                                                text = detectedObject.getLabels().get(0).getText();
                                                if (!seen.contains(text)) {
                                                    checkWarnings(text);
                                                }
                                                seen.add(text);
                                                new android.os.Handler(Looper.getMainLooper()).postDelayed(
                                                        new Runnable() {
                                                            public void run() {
                                                                seen.remove(0);
                                                            }
                                                        },
                                                        5000);

                                            }
//                                            for (DetectedObject.Label label : detectedObject.getLabels()) {
//                                                text = label.getText();
//                                                speechText.setText(text);
//                                                float confidence = label.getConfidence();
//                                            }
                                            Draw graphic = new Draw(MainActivity.this, boundingBox, text);
                                            binding.container.addView(graphic);
                                        }
                                    }
                                })
                        .addOnFailureListener(
                                new OnFailureListener() {
                                    @Override
                                    public void onFailure(@NonNull Exception e) {
                                        // Task failed with an exception
                                        speechText.setText("Fail. Sad!");
                                    }
                                })
                        .addOnCompleteListener(results -> imageProxy.close());
            }
        }

        private void checkWarnings(String text) {
            for (String s: warnings) {
                if (text.contains(s)) {
                    textToSpeech.speak("There is a "+text+" in front of you.",TextToSpeech.QUEUE_ADD,null);
                }
            }
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        if (mediaPlayer == null) {
            mediaPlayer = MediaPlayer.create(this, R.raw.puppy);
            mediaPlayer.setVolume((float)0.2,(float)0.2);

        }
        //change notif bar color ie making it full screen activity
        Window window = this.getWindow();
        window.clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS);
        window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
        window.setStatusBarColor(ContextCompat.getColor(this,R.color.blue_main));
        cl_blue_curtain = findViewById(R.id.cl_blue_transparent);
        phonecamBtn = findViewById(R.id.tv_phonecam);
        uploadBtn = findViewById(R.id.tv_upload);
        suncamBtn = findViewById(R.id.tv_suncam);
        rotateLoading = AnimationUtils.loadAnimation(this, R.anim.rotate);
        //roateImage(binding.ivLoadingIcon);
        phonecamBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                phonecamBtn.setTextColor(getResources().getColor(R.color.yellow));
                suncamBtn.setTextColor(getResources().getColor(R.color.white));
                uploadBtn.setTextColor(getResources().getColor(R.color.white));
                cl_blue_curtain.setVisibility(View.GONE);
                binding.speechText.setText("");
                binding.clUploadScreen.setVisibility(View.GONE);
                binding.constraintLayout.setBackgroundColor(getResources().getColor(R.color.trans_black));
            }
        });

        uploadBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                phonecamBtn.setTextColor(getResources().getColor(R.color.white));
                suncamBtn.setTextColor(getResources().getColor(R.color.white));
                uploadBtn.setTextColor(getResources().getColor(R.color.yellow));
                cl_blue_curtain.setVisibility(View.GONE);
                binding.speechText.setVisibility(View.GONE);
                binding.clUploadScreen.setVisibility(View.VISIBLE);
                binding.constraintLayout.setBackgroundColor(getResources().getColor(R.color.blue_main_transparent));
            }
        });

        suncamBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                phonecamBtn.setTextColor(getResources().getColor(R.color.white));
                suncamBtn.setTextColor(getResources().getColor(R.color.yellow));
                uploadBtn.setTextColor(getResources().getColor(R.color.white));
                binding.clUploadScreen.setVisibility(View.GONE);
                binding.clBlueTransparent.setVisibility(View.GONE);
                binding.constraintLayout.setBackgroundColor(getResources().getColor(R.color.trans_black));

            }
        });

        // Make sure required permissions are granted
        checkPermission(new String[] {
                Manifest.permission.RECORD_AUDIO,
                Manifest.permission.CAMERA,
                Manifest.permission.WRITE_EXTERNAL_STORAGE}, MULTIPLE_PERMISSION_CODE);

        //Initialize text to speech
        textToSpeech = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int i) {
                // if No error is found then only it will run
                if(i!=TextToSpeech.ERROR){
                    // To Choose language of speech
                    textToSpeech.setLanguage(Locale.US);
                }
                textToSpeech.speak("Press the mic button to get started",TextToSpeech.QUEUE_FLUSH,null);
            }
        });
    }

    // Function to check and request permission.
    public void checkPermission(String[] permission, int requestCode) {
        List<String> permissionsNeeded = new ArrayList<>();
        for (String s: permission) {
            if (ContextCompat.checkSelfPermission(MainActivity.this, s) == PackageManager.PERMISSION_DENIED) {
                permissionsNeeded.add(s);
            }
        }
        if (!permissionsNeeded.isEmpty()) {
            ActivityCompat.requestPermissions(MainActivity.this, permissionsNeeded.toArray(new String[permissionsNeeded.size()]), requestCode);
        } else {
            initializeCamera();
            initializeMic();
        }
    }

    // This function is called when the user accepts or decline the permission.
    // Request Code is used to check which permission called this function.
    // This request code is provided when the user is prompt for permission.
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 123 && grantResults.length > 0 ){
            if(grantResults[0] == PackageManager.PERMISSION_GRANTED)
                Toast.makeText(this,"Permission Granted",Toast.LENGTH_SHORT).show();
                initializeCamera();
                initializeMic();
        }
    }

    //Initialize camera view
    private void initializeCamera(){
        previewView = findViewById(R.id.previewView);
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(new Runnable() {
            @Override
            public void run() {
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    imageCapture = bindImageAnalysis(cameraProvider);
                } catch (ExecutionException | InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }, ContextCompat.getMainExecutor(this));
    }

    //Initialize mic
    private void initializeMic(){
        // Initialize speech-to-text module
        speechText = findViewById(R.id.speechText);
        micButton = findViewById(R.id.micButton);

        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);

        final Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault());

        speechRecognizer.setRecognitionListener(new RecognitionListener() {
            int flag = 0;

            @Override
            public void onReadyForSpeech(Bundle bundle) {
                speechText.setText("Start talking");
                new android.os.Handler(Looper.getMainLooper()).postDelayed(
                        new Runnable() {
                            public void run() {
                                if (flag == 0) {
                                    micButton.setImageResource(R.drawable.ic_baseline_mic_24);
                                    speechText.setText("Nothing was said. Please try again.");
                                    textToSpeech.speak("Nothing was said. Please try again.",TextToSpeech.QUEUE_FLUSH,null);
                                }
                            }
                        },
                        5000);
            }

            @Override
            public void onBeginningOfSpeech() {
                flag = 1;
                speechText.setText("");
                speechText.setHint("");
            }

            @Override
            public void onRmsChanged(float v) {

            }

            @Override
            public void onBufferReceived(byte[] bytes) {

            }

            @Override
            public void onEndOfSpeech() {

            }

            @Override
            public void onError(int i) {

            }

            @Override
            public void onResults(Bundle bundle) {
                ArrayList<String> data = bundle.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                speechText.setText(data.get(0));
                micButton.setImageResource(R.drawable.ic_baseline_mic_24);
                if (data.get(0).contains("describe") || data.get(0).contains("going on") || data.get(0).contains("happening")) {
                    captureImage(imageCapture, "caption");
                    cl_blue_curtain.setVisibility(View.VISIBLE);
                    binding.tvAnalyzingAndCaption.setText("Analyzing...");
                    binding.ivLoadingIcon.startAnimation(rotateLoading);
                    binding.ivLoadingIcon.setVisibility(View.VISIBLE);
                    binding.speechText.setVisibility(View.GONE);
                } else if (data.get(0).contains("read")) {
                    captureImage(imageCapture, "ocr/text");
                    binding.tvAnalyzingAndCaption.setText("Analyzing...");
                    cl_blue_curtain.setVisibility(View.VISIBLE);
                    binding.ivLoadingIcon.startAnimation(rotateLoading);
                    binding.ivLoadingIcon.setVisibility(View.VISIBLE);
                    binding.speechText.setVisibility(View.GONE);
                } else {
                    textToSpeech.speak("Sorry, that command does not exist.",TextToSpeech.QUEUE_FLUSH,null);
                }
            }

            @Override
            public void onPartialResults(Bundle bundle) {

            }

            @Override
            public void onEvent(int i, Bundle bundle) {

            }
        });

        // Set an OnTouch Listener for micButton
        micButton.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {
                if (motionEvent.getAction() == MotionEvent.ACTION_UP){
                    speechRecognizer.stopListening();
                    micButton.setImageResource(R.drawable.ic_baseline_mic_24);
                }
                if (motionEvent.getAction() == MotionEvent.ACTION_DOWN){
                    mediaPlayer.start();
                    micButton.setImageResource(R.drawable.ic_mic);
                    speechRecognizer.startListening(intent);
                }
                return false;
            }
        });
    }

    // Capturing the image and then sending it to the API (Flask)
    private void captureImage(ImageCapture imageCapture, String usage) {
        SimpleDateFormat mDateFormat = new SimpleDateFormat("yyyyMMddHHmmss", Locale.US);
        File file = new File(getBatchDirectoryName(), mDateFormat.format(new Date())+ ".jpg");

        ImageCapture.OutputFileOptions outputFileOptions = new ImageCapture.OutputFileOptions.Builder(file).build();
        imageCapture.takePicture(outputFileOptions, executor, new ImageCapture.OnImageSavedCallback () {

            @Override
            public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                MainActivity.this.runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(MainActivity.this, "Image Captured successfully", Toast.LENGTH_SHORT).show();
                        postRequest(url+usage, file, usage);
                    }
                });
            }

            @Override
            public void onError(@NonNull ImageCaptureException error) {
                error.printStackTrace();
            }

        });
    }

    // Bind Image Analysis module
    private ImageCapture bindImageAnalysis(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder()
                .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        preview.setSurfaceProvider(previewView.createSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(1280, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();
        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), new ObjectDetectionProcessor());

        ImageCapture.Builder builder = new ImageCapture.Builder();

        final ImageCapture imageCapture = builder
                .setTargetRotation(this.getWindowManager().getDefaultDisplay().getRotation())
                .build();
        preview.setSurfaceProvider(previewView.createSurfaceProvider());
        Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner)this, cameraSelector, preview, imageAnalysis, imageCapture);

        return imageCapture;
    }

    // Get directory path to save media
    public String getBatchDirectoryName() {
        String app_folder_path = Environment.getExternalStorageDirectory().toString() + "/Android/data/com.hophacks.guidedog/files/Pictures";
        File dir = new File(app_folder_path);
        if (!dir.exists() && !dir.mkdirs()) {
        }
        return app_folder_path;
    }

    // Post Request to send picture to the API
    private void postRequest(String URL, File file, String usage) {
        RequestBody requestBody = buildRequestBody(file);
        OkHttpClient client = new OkHttpClient();
        Request request = new Request.Builder()
                .post(requestBody)
                .url(URL)
                .build();
        textToSpeech.speak("Analyzing",TextToSpeech.QUEUE_FLUSH,null);
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(final Call call, final IOException e) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(MainActivity.this, "Something went wrong:" + " " + e.getMessage(), Toast.LENGTH_SHORT).show();
                        call.cancel();
                    }
                });
            }

            @Override
            public void onResponse(Call call, final Response response) throws IOException {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            String s = response.body().string();
                            if (s.contains("Error")) {
                                textToSpeech.speak("Could not access server. Please try again.",TextToSpeech.QUEUE_FLUSH,null);
                            } else {
                                textToSpeech.speak(s,TextToSpeech.QUEUE_FLUSH,null);
                                if (usage == "caption") {
                                    speechText.setText(s.replace("\"", ""));
                                    binding.speechText.setVisibility(View.VISIBLE);
                                }
                            }
                            cl_blue_curtain.setVisibility(View.GONE);
                            binding.ivLoadingIcon.setVisibility(View.GONE);
                            binding.tvAnalyzingAndCaption.setText(s);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                });
            }
        });
    }

    // Build the media request body for image file
    private RequestBody buildRequestBody(File file) {
        return new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", "androidFlask.jpg", RequestBody.create(MediaType.parse("image/*jpg"), file))
                .build();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        speechRecognizer.destroy();
        if (mediaPlayer != null) mediaPlayer.release();
    }
}