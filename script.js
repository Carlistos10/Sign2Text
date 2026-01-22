// ============================================================================
// CONFIGURACIÓN GLOBAL
// ============================================================================

// El modelo fue entrenado con imágenes de 64x64. Es CRUCIAL que las entradas
// tengan exactamente este tamaño, o el cálculo matricial fallará.
const IMG_SIZE = 64;

// Mapeo de índices a etiquetas.
// Si el modelo predice el índice 0, corresponde a 'A'. El índice 26 es 'del', etc.
const CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'];

// Ruta donde está el modelo.
const MODEL_PATH = 'web_model/model.json';

let model = null; // Aquí guardaremos el objeto del modelo cargado de TensorFlow

// ============================================================================
// REFERENCIAS AL DOM (Interfaz de Usuario)
// ============================================================================
const imgEl = document.getElementById('previewImage');
const placeholderEl = document.getElementById('placeholder');
const resultEl = document.getElementById('predictionResult');
const confidenceEl = document.getElementById('confidenceScore');
const statusEl = document.getElementById('modelStatus');
const predictBtn = document.getElementById('predictBtn');
const manualUploadSection = document.getElementById('manualUploadSection');

// Helper para logs
function logDebug(msg, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${msg}`);
}

// ============================================================================
// INICIO AUTOMÁTICO
// ============================================================================
window.addEventListener('DOMContentLoaded', async () => {
    // Verificación de protocolo file:// (bloqueo CORS)
    if (window.location.protocol === 'file:') {
        logDebug("ERROR: Protocolo file:// detectado. Carga bloqueada.", 'error');
        showManualUpload();
    } else {
        await findAndLoadModel();
    }
});

// ============================================================================
// CARGA DE MODELO
// ============================================================================
async function findAndLoadModel() {
    logDebug("Iniciando carga del modelo...", 'info');

    // Cache buster para evitar que el navegador use una versión vieja
    const cacheBuster = `?t=${Date.now()}`;

    try {
        // Construimos la URL basada en la ruta configurada
        const modelUrl = new URL(MODEL_PATH, window.location.href).href;
        logDebug(`Intentando cargar desde: ${MODEL_PATH}`, 'info');

        // --- PASO 1: Descargar el JSON ---
        const response = await fetch(modelUrl + cacheBuster);

        if (!response.ok) {
            throw new Error(`Archivo JSON no encontrado (404) en ${MODEL_PATH}`);
        }

        const jsonContent = await response.text();
        let modelData;
        try {
            modelData = JSON.parse(jsonContent);
        } catch (e) {
            throw new Error("El archivo model.json está corrupto o no es un JSON válido.");
        }

        logDebug(`✅ JSON descargado.`, 'success');

        // --- PASO 2: Descargar los archivos binarios (.bin) ---
        // Buscamos la ubicación base para descargar los .bin relativos al JSON
        const basePath = modelUrl.substring(0, modelUrl.lastIndexOf('/') + 1);
        const binFiles = [];

        if (modelData.weightsManifest) {
            logDebug(`Descargando pesos binarios...`, 'info');
            const manifest = modelData.weightsManifest;

            for (const group of manifest) {
                for (const p of group.paths) {
                    const cleanPath = p.startsWith('./') ? p.slice(2) : p;
                    const binUrl = basePath + cleanPath;

                    logDebug(`Fetching: ${cleanPath}`, 'info');
                    const binResp = await fetch(binUrl + cacheBuster);

                    if (!binResp.ok) throw new Error(`Binario faltante: ${cleanPath}`);

                    const blob = await binResp.blob();
                    // Creamos un archivo en memoria
                    const file = new File([blob], cleanPath, { type: 'application/octet-stream' });
                    binFiles.push(file);
                }
            }
        }

        logDebug(`✅ Binarios en memoria.`, 'success');

        // --- PASO 3: Corregimos incompatibilidades Keras -> TFJS en memoria
        patchModelArchitecture(modelData);
        logDebug(`🔧 Parches aplicados.`, 'info');

        // --- PASO 4: Cargar en TensorFlow ---
        const jsonFile = new File([JSON.stringify(modelData)], 'model.json', { type: 'application/json' });
        const filesToLoad = [jsonFile, ...binFiles];

        // Carga desde los archivos virtuales en memoria
        model = await tf.loadLayersModel(tf.io.browserFiles(filesToLoad));

        logDebug(`🎉 MODELO CARGADO EXITOSAMENTE`, 'success');

        // Actualizar UI: Éxito
        statusEl.innerHTML = `
        <div class="flex items-center gap-2 text-emerald-400">
            <i data-lucide="check-circle" class="w-4 h-4"></i>
            <span>Modelo listo</span>
        </div>`;

        predictBtn.disabled = false;
        predictBtn.classList.remove('bg-slate-600', 'text-slate-400', 'cursor-not-allowed');
        predictBtn.classList.add('bg-blue-600', 'text-white', 'hover:bg-blue-500');
        predictBtn.innerText = "DETECTAR SIGNO";
        lucide.createIcons();

    } catch (e) {
        // Si falla algo en el único path existente, vamos directo al error/manual
        logDebug(`❌ Error fatal cargando el modelo: ${e.message}`, 'error');
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

// ============================================================================
// FUNCIÓN DE PARCHEO (Corrección de errores JSON)
// ============================================================================
function patchModelArchitecture(modelData) {
    // 1. Fix batch_shape vs batchInputShape
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

    // 2. Fix prefijos en nombres de pesos
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

// ============================================================================
// CARGA MANUAL (Fallback)
// ============================================================================
const manualInput = document.getElementById('modelUpload');
if (manualInput) {
    manualInput.addEventListener('change', async (event) => {
        const rawFiles = [...event.target.files];
        if (rawFiles.length === 0) return;

        const jsonFile = rawFiles.find(f => f.name.endsWith('.json'));
        const binFiles = rawFiles.filter(f => f.name.endsWith('.bin'));

        if (!jsonFile || binFiles.length === 0) {
            alert("Faltan archivos (se necesita 1 .json y al menos 1 .bin).");
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

        } catch (e) {
            console.error(e);
            logDebug(`Error manual: ${e.message}`, 'error');
            alert("Error al cargar archivos: " + e.message);
        }
    });
}

// ============================================================================
// LÓGICA DE UI E INFERENCIA
// ============================================================================
window.handleImageUpload = function (event) {
    const file = event.target.files[0];
    if (file) {
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
    if (imgEl.classList.contains('hidden') || !imgEl.src) {
        alert("Por favor sube una imagen primero.");
        return;
    }

    if (!model) {
        alert("El modelo no está listo.");
        return;
    }

    tf.tidy(() => {
        let tensor = tf.browser.fromPixels(imgEl);
        tensor = tf.image.resizeBilinear(tensor, [IMG_SIZE, IMG_SIZE]);
        tensor = tensor.div(255.0);
        tensor = tensor.expandDims(0);

        const predictions = model.predict(tensor);
        const data = predictions.dataSync();
        const maxIndex = predictions.argMax(-1).dataSync()[0];

        updateUI(CLASSES[maxIndex] || "?", data[maxIndex]);
    });
}

function updateUI(label, confidence) {
    resultEl.innerText = label;
    confidenceEl.innerText = `Confianza: ${(confidence * 100).toFixed(1)}%`;
    confidenceEl.style.opacity = "1";
    confidenceEl.className = confidence > 0.85
        ? "mt-2 font-mono text-lg text-emerald-400"
        : "mt-2 font-mono text-lg text-yellow-400";
}