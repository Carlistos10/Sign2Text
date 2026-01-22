lucide.createIcons();

// CONFIGURACIÓN
const IMG_SIZE = 64;
const CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'];

const POTENTIAL_PATHS = [
    'web_model/model.json',
    'model.json',
    'web_model/model.json',
    'Web_Model/model.json'
];

let model = null;
let isWebcamActive = false;
let webcamStream = null;
let animationId = null;

const videoEl = document.getElementById('webcam');
const imgEl = document.getElementById('previewImage');
const placeholderEl = document.getElementById('placeholder');
const resultEl = document.getElementById('predictionResult');
const confidenceEl = document.getElementById('confidenceScore');
const statusEl = document.getElementById('modelStatus');
const predictBtn = document.getElementById('predictBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const manualUploadSection = document.getElementById('manualUploadSection');
const debugSection = document.getElementById('debugSection');
const debugLog = document.getElementById('debugLog');

function logDebug(msg, type = 'info') {
    const color = type === 'error' ? 'text-red-400' : (type === 'success' ? 'text-emerald-400' : 'text-slate-300');
    // Verificamos si debugLog existe antes de escribir (por seguridad)
    if (debugLog) {
        const line = document.createElement('div');
        line.className = `${color} mb-1 border-b border-slate-800 pb-1`;
        line.innerHTML = `> ${msg}`;
        debugLog.appendChild(line);
        debugLog.scrollTop = debugLog.scrollHeight;
    }
}

// --- INICIO AUTOMÁTICO ---
window.addEventListener('DOMContentLoaded', async () => {
    if (window.location.protocol === 'file:') {
        logDebug("ERROR: Protocolo file:// detectado. Carga bloqueada.", 'error');
        showManualUpload();
    } else {
        await findAndLoadModel();
    }
});

// ESTRATEGIA: Descarga Híbrida (Download & Inject)
async function findAndLoadModel() {
    let loaded = false;
    if (debugSection) debugSection.classList.remove('hidden');
    logDebug("Iniciando búsqueda...", 'info');

    const cacheBuster = `?t=${Date.now()}`;

    for (const path of POTENTIAL_PATHS) {
        try {
            const modelUrl = new URL(path, window.location.href).href;
            logDebug(`Probando: .../${path}`, 'info');

            // 1. Descargar JSON
            const response = await fetch(modelUrl + cacheBuster);
            if (!response.ok) {
                logDebug(`❌ JSON 404`, 'error');
                continue;
            }

            const jsonContent = await response.text();
            let modelData;
            try { modelData = JSON.parse(jsonContent); }
            catch (e) { logDebug(`❌ JSON Inválido`, 'error'); continue; }

            logDebug(`✅ JSON descargado.`, 'success');

            // 2. Descargar BIN(s) Manualmente
            const basePath = modelUrl.substring(0, modelUrl.lastIndexOf('/') + 1);
            const binFiles = [];

            if (modelData.weightsManifest) {
                logDebug(`Descargando binarios...`, 'info');
                const manifest = modelData.weightsManifest;

                // Recorremos todos los manifiestos y paths
                for (const group of manifest) {
                    for (const p of group.paths) {
                        // Limpiar path relativo
                        const cleanPath = p.startsWith('./') ? p.slice(2) : p;
                        const binUrl = basePath + cleanPath;

                        logDebug(`Fetching: ${cleanPath}`, 'info');
                        const binResp = await fetch(binUrl + cacheBuster);

                        if (!binResp.ok) throw new Error(`Binario no encontrado: ${cleanPath}`);

                        const blob = await binResp.blob();
                        // Crear objeto File en memoria
                        const file = new File([blob], cleanPath, { type: 'application/octet-stream' });
                        binFiles.push(file);
                    }
                }
            }

            logDebug(`✅ Binarios descargados en memoria.`, 'success');

            // 3. Parchear Arquitectura
            patchModelArchitecture(modelData);
            logDebug(`🔧 Parches aplicados.`, 'info');

            // 4. Crear File para el JSON parcheado
            const jsonFile = new File([JSON.stringify(modelData)], 'model.json', { type: 'application/json' });

            // 5. Cargar usando IO BrowserFiles (Todo en memoria)
            const filesToLoad = [jsonFile, ...binFiles];

            model = await tf.loadLayersModel(tf.io.browserFiles(filesToLoad));

            loaded = true;
            logDebug(`🎉 MODELO CARGADO EXITOSAMENTE`, 'success');

            statusEl.innerHTML = `
            <div class="flex items-center gap-2 text-emerald-400">
                <i data-lucide="check-circle" class="w-4 h-4"></i>
                <span>Modelo listo</span>
            </div>`;

            predictBtn.disabled = false;
            predictBtn.classList.remove('bg-slate-600', 'text-slate-400', 'cursor-not-allowed');
            predictBtn.classList.add('bg-blue-600', 'text-white', 'hover:bg-blue-500');
            predictBtn.innerText = "DETECTAR SIGNO";
            if (isWebcamActive) predictLoop();

            lucide.createIcons();
            break;

        } catch (e) {
            logDebug(`❌ Error en intento: ${e.message}`, 'error');
        }
    }

    if (!loaded) {
        logDebug("FATAL: No se pudo cargar el modelo.", 'error');
        showManualUpload();
    }
}

function showManualUpload() {
    statusEl.innerHTML = `
        <div class="flex items-center gap-2 text-red-400 font-bold">
            <i data-lucide="x-circle" class="w-4 h-4"></i>
            <span>Fallo de sistema</span>
        </div>`;
    manualUploadSection.classList.remove('hidden');
    lucide.createIcons();
    predictBtn.innerText = "Sube el modelo manualmente";
}

// Función de Parcheo
function patchModelArchitecture(modelData) {
    function fixLayers(layers) {
        if (!layers) return;
        layers.forEach(layer => {
            if (layer.class_name === 'InputLayer' && layer.config) {
                if (layer.config.batch_shape && !layer.config.batchInputShape) {
                    layer.config.batchInputShape = layer.config.batch_shape;
                }
            }
        });
    }

    if (modelData.modelTopology?.model_config?.config?.layers) {
        fixLayers(modelData.modelTopology.model_config.config.layers);
    }

    if (modelData.weightsManifest && modelData.weightsManifest.length > 0) {
        const modelName = modelData.modelTopology?.model_config?.config?.name;
        if (modelName) {
            const prefix = modelName + '/';
            modelData.weightsManifest.forEach(manifest => {
                manifest.weights.forEach(weight => {
                    if (weight.name.startsWith(prefix)) {
                        weight.name = weight.name.substring(prefix.length);
                    }
                });
            });
        }
    }
}

// --- Carga Manual (Fallback) ---
const manualInput = document.getElementById('modelUpload');
if (manualInput) {
    manualInput.addEventListener('change', async (event) => {
        const rawFiles = [...event.target.files];
        if (rawFiles.length === 0) return;

        const jsonFile = rawFiles.find(f => f.name.endsWith('.json'));
        const binFiles = rawFiles.filter(f => f.name.endsWith('.bin'));

        if (!jsonFile || binFiles.length === 0) {
            alert("Faltan archivos (json o bin).");
            return;
        }

        try {
            const jsonContent = await new Promise((resolve) => {
                const reader = new FileReader();
                reader.onload = (e) => resolve(e.target.result);
                reader.readAsText(jsonFile);
            });

            const modelData = JSON.parse(jsonContent);
            patchModelArchitecture(modelData);

            const patchedBlob = new Blob([JSON.stringify(modelData)], { type: 'application/json' });
            const patchedFile = new File([patchedBlob], jsonFile.name, { type: 'application/json' });
            const uploadFiles = [patchedFile, ...binFiles];

            model = await tf.loadLayersModel(tf.io.browserFiles(uploadFiles));

            statusEl.innerHTML = `<span class="text-emerald-400 font-bold">¡Modelo manual cargado!</span>`;
            logDebug("Carga manual exitosa", 'success');
            manualUploadSection.classList.add('hidden');

            predictBtn.disabled = false;
            predictBtn.classList.remove('bg-slate-600', 'text-slate-400', 'cursor-not-allowed');
            predictBtn.classList.add('bg-blue-600', 'text-white');
            predictBtn.innerText = "DETECTAR SIGNO";
            if (isWebcamActive) predictLoop();

        } catch (e) {
            console.error(e);
            logDebug(`Error manual: ${e.message}`, 'error');
            alert("Error: " + e.message);
        }
    });
}

// --- UI & Webcam ---
// Estas funciones se asignan a window para que el HTML pueda llamarlas con onclick
window.toggleCamera = async function () {
    if (isWebcamActive) stopWebcam();
    else await startWebcam();
}

window.handleImageUpload = function (event) {
    const file = event.target.files[0];
    if (file) {
        stopWebcam();
        const reader = new FileReader();
        reader.onload = (e) => {
            imgEl.src = e.target.result;
            imgEl.classList.remove('hidden');
            placeholderEl.classList.add('hidden');
            resultEl.innerText = "--";
            confidenceEl.style.opacity = "0";
        };
        reader.readAsDataURL(file);
    }
}

window.predict = async function () {
    const isDemo = document.getElementById('demoMode').checked;
    let imageSource = null;

    if (isWebcamActive && !videoEl.paused) imageSource = videoEl;
    else if (!imgEl.classList.contains('hidden') && imgEl.src) imageSource = imgEl;
    else {
        alert("Enciende la cámara o sube imagen.");
        return;
    }

    if (isDemo) {
        const randomChar = CLASSES[Math.floor(Math.random() * CLASSES.length)];
        updateUI(randomChar, 0.95);
        return;
    }

    if (!model) return;

    tf.tidy(() => {
        let tensor = tf.browser.fromPixels(imageSource);
        tensor = tf.image.resizeBilinear(tensor, [IMG_SIZE, IMG_SIZE]);
        tensor = tensor.div(255.0);
        tensor = tensor.expandDims(0);
        const predictions = model.predict(tensor);
        const data = predictions.dataSync();
        const maxIndex = predictions.argMax(-1).dataSync()[0];
        updateUI(CLASSES[maxIndex] || "?", data[maxIndex]);
    });
}

window.toggleDemoMode = function () {
    const isDemo = document.getElementById('demoMode').checked;
    if (isDemo) {
        predictBtn.disabled = false;
        predictBtn.innerText = "SIMULAR";
        predictBtn.classList.add('bg-purple-600', 'text-white');
    } else if (!model) {
        predictBtn.disabled = true;
        predictBtn.innerText = "Cargando IA...";
        predictBtn.classList.remove('bg-purple-600', 'text-white');
    } else {
        predictBtn.innerText = "DETECTAR SIGNO";
        predictBtn.classList.remove('bg-purple-600');
    }
}

async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
        videoEl.srcObject = webcamStream;
        videoEl.classList.remove('hidden');
        placeholderEl.classList.add('hidden');
        imgEl.classList.add('hidden');
        isWebcamActive = true;
        if (model || document.getElementById('demoMode').checked) predictLoop();
    } catch (err) {
        alert("Error cámara: " + err.message);
    }
}

function stopWebcam() {
    if (webcamStream) webcamStream.getTracks().forEach(t => t.stop());
    videoEl.classList.add('hidden');
    placeholderEl.classList.remove('hidden');
    isWebcamActive = false;
    cancelAnimationFrame(animationId);
}

function updateUI(label, confidence) {
    resultEl.innerText = label;
    confidenceEl.innerText = `Confianza: ${(confidence * 100).toFixed(1)}%`;
    confidenceEl.style.opacity = "1";
    confidenceEl.className = confidence > 0.85 ? "mt-2 font-mono text-lg text-emerald-400" : "mt-2 font-mono text-lg text-yellow-400";
}

function predictLoop() {
    if (isWebcamActive) {
        window.predict(); // Llamamos a la función global
        animationId = requestAnimationFrame(predictLoop);
    }
}